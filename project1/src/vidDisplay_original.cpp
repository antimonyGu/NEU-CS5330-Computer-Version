#include <iostream> // for standard I/O
#include <string> // for strings
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "filter.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    char lastKey = 's';
    string filePrefix = "./output/";
    // bool isS

    VideoWriter* videoWriter = nullptr;

    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }                
        // cv::imshow("Video", frame);
        // continue;

        cv::Mat dst = frame.clone();

        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        
        if (key == -1) {
            key = lastKey;
        }

        if( key == 'q') {
            break;
        } else if (key == 's') {
            cv::imwrite("test.png", frame); // TBD: error handling
        } else if (key == 'g') {
            cvtColor(frame, dst, COLOR_RGB2GRAY);
            imwrite(filePrefix + "part3_generated_image.png", dst);
            setWindowTitle("Video", "Grayscale Mode by Opencv");
            lastKey = 'g';
        } else if (key == 'h') {
            greyscale(frame, dst);
            imwrite(filePrefix + "part4_generated_image.png", dst);
            setWindowTitle("Video", "Grayscale Mode by Me");
            lastKey = 'h';
        } else if (key == 'b') {
            blur5x5(frame, dst);
            imwrite(filePrefix + "part5_generated_image.png", dst);
            setWindowTitle("Video", "Blurry Mode by Me");
            lastKey = 'b';
        } else if (key == 'x') {
            dst = Mat::zeros(frame.rows, frame.cols, CV_16SC3);
            sobelX3x3(frame, dst);
            convertScaleAbs(dst, dst);
            imwrite(filePrefix + "part6_generated_imageX.png", dst);
            setWindowTitle("Video", "Sobel-X Filter Mode by Me");
            lastKey = 'x';
        } else if (key == 'y') {
            dst = Mat::zeros(frame.rows, frame.cols, CV_16SC3);
            sobelY3x3(frame, dst);
            convertScaleAbs(dst, dst);
            imwrite(filePrefix + "part6_generated_imageY.png", dst);
            setWindowTitle("Video", "Sobel-Y Filter Mode by Me");
            lastKey = 'y';
        } else if (key == 'm') {
            Mat sobel_filtered_Ximage = Mat::zeros(frame.rows, frame.cols, CV_16SC3);
            Mat sobel_filtered_Yimage = Mat::zeros(frame.rows, frame.cols, CV_16SC3);
            
            sobelX3x3(frame, sobel_filtered_Ximage);
            sobelY3x3(frame, sobel_filtered_Yimage);
            
            magnitude(sobel_filtered_Ximage, sobel_filtered_Yimage, dst);
            imwrite(filePrefix + "part7_generated_image.png", dst);
            setWindowTitle("Video", "Magnitude Filter Mode by Me");
            lastKey = 'm';
        } else if (key == 'i') {
            blurQuantize(frame, dst, 15);
            imwrite(filePrefix + "part8_generated_image.png", dst);
            setWindowTitle("Video", "Blur and Quantinized Filter Mode by Me");
            lastKey = 'i';
        } else if (key == 'c') {
            cartoon(frame, dst, 15, 15);
            imwrite(filePrefix + "part9_generated_image.png", dst);
            setWindowTitle("Video", "Cartoonized Filter Mode by Me");
            lastKey = 'c';
        } else if(key == 'p') {
            changeBrightness(frame, dst);
            imwrite(filePrefix + "part10_generated_image.png", dst);
            setWindowTitle("Video", "Extra brightness Mode by Me");
            lastKey = 'p';
        }  else if(key == 'e') {
            trackBar(frame, dst);
            imwrite(filePrefix + "part11_generated_image.png", dst);
            setWindowTitle("Video", "Extension Adding Trackbar to Video");
            lastKey = 'e';
        } else if (key == 'v') {
            videoWriter = new VideoWriter(filePrefix + "outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(refS.width, refS.height));
        }

        if (videoWriter != nullptr) {
            videoWriter -> write(dst);
        }
        cv::imshow("Video", dst);
    }

    if (videoWriter) {
        videoWriter -> release();
    } 
    delete capdev;
    return(0);
}