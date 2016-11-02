#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include<vector>
#include<cv.h>

using namespace std;
using namespace cv;
cv::Mat rotateImage(const cv::Mat &src, cv::Point2f anchor_pt, double angle)
{
    cv::Mat rot_mat = cv::getRotationMatrix2D(anchor_pt, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, src.size());
    
    return dst;
}

int main()
{
        Mat img = imread("/Users/sovietreborn/Desktop/img1.jpg");
        Mat img2=imread("/Users/sovietreborn/Desktop/img2.jpg");
        if(img.empty())
        {
                    fprintf(stderr, "Can not load image \n");
                    return -1;
        }
        if(img2.empty())
         {
                    fprintf(stderr, "Can not load image2 \n");
                    return -1;
         }
        Mat imgline ;
        imshow("image before", img);
        imshow("image2 before",img2);
        SurfFeatureDetector siftdtc(400);
        vector<KeyPoint>kp1,kp2;
        siftdtc.detect(img,kp1);
        siftdtc.detect(img2,kp2);
        SurfDescriptorExtractor extractor;
        Mat descriptor1,descriptor2;
        Mat img_matches;
        extractor.compute(img,kp1,descriptor1);
        extractor.compute(img2,kp2,descriptor2);
    
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptor1, descriptor2, matches );

    
        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < descriptor1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    std::vector< DMatch > good_matches;
    drawMatches( img, kp1, img2, kp2,
                good_matches, imgline, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    for( int i = 0; i < descriptor1.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.6) )
    { good_matches.push_back( matches[i]); }
    }



    drawMatches( img, kp1, img2, kp2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
   // img_matches=img;
    Mat imgpoint=img_matches;
    imwrite("/Users/sovietreborn/Desktop/point.jpg", imgpoint);
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( kp1[ good_matches[i].queryIdx ].pt );
        scene.push_back( kp2[ good_matches[i].trainIdx ].pt );
    }
    
    Mat H = findHomography( obj, scene, CV_RANSAC );
    
    Point2f obj_corners[4] = { cvPoint(0,0), cvPoint( img.cols, 0 ), cvPoint( img.cols, img.rows ), cvPoint( 0, img.rows ) };
    Point scene_corners[4],bp[1];
    for( int i = 0; i < 4; i++ )
    {
        double x = obj_corners[i].x;
        double y = obj_corners[i].y;
        
        double Z = 1./( H.at<double>(2,0)*x + H.at<double>(2,1)*y + H.at<double>(2,2) );
        double X = ( H.at<double>(0,0)*x + H.at<double>(0,1)*y + H.at<double>(0,2) )*Z;
        double Y = ( H.at<double>(1,0)*x + H.at<double>(1,1)*y + H.at<double>(1,2) )*Z;
        scene_corners[i] = cvPoint( cvRound(X) , cvRound(Y) );
    }
    double k=0;
    if (scene_corners[1].x!=scene_corners[2].x) {
        k=(scene_corners[1].y-scene_corners[2].y)/(scene_corners[1].x-scene_corners[2].x);
        k=atan(k);
        k=k*180/3.141592607;
    }
    Mat img_2r=img2;
    cv::Mat dst;
    bp[0].y=0;
    bp[0].x=scene_corners[1].x-(((scene_corners[2].x-scene_corners[1].x)/(scene_corners[2].y-scene_corners[1].y))*scene_corners[1].y);
    dst=rotateImage(img2, bp[0], k-90);
    imwrite("/Users/sovietreborn/Desktop/rotate.jpg", dst);
    int xlarge=scene_corners[1].x;
    IplImage dsti=IplImage(dst),*dstresult;
    cvSetImageROI(&dsti,cvRect(xlarge,0,(dst.cols-xlarge),dst.rows));
    dstresult=cvCreateImage(cvSize((dst.cols-xlarge),dst.rows), IPL_DEPTH_8U, dsti.nChannels);
    cvCopy(&dsti, dstresult);
    cvShowImage("xxx", dstresult);
    cv::Mat img33(dstresult,0);
    imshow("result",dst);
        for( int i = 0; i < 4; i++ )
    {
        printf("%d,%d\n",scene_corners[i].x,scene_corners[i].y);
        //printf("...%f...",k);
    }
    hconcat(img, img33, img_matches);
    imwrite("/Users/sovietreborn/Desktop/cut.jpg", img33);
    scene_corners[1].x+=img.cols;
    scene_corners[2].x+=img.cols;
    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
   //line( img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2 );
    line( imgline, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 2 );
    //line( imgline, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 2 );
   // line( img_matches, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 2 );
       //resultpic[]    //-- Show detected matches
  //  imshow( "Good Matches & Object detection", img_matches );
    //-- Draw only "good" matches
    imwrite("/Users/sovietreborn/Desktop/line.jpg", imgline);

    //-- Show detected matches
  //  imshow( "Good Matches", img_matches );
    imwrite("/Users/sovietreborn/Desktop/good.jpg", img_matches);
    

    return -1;
}
