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
        printf("test\n");

        // cout << "test";

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                
                cv::imshow("Video", frame);
                printf("test\n");

                // see if there is a waiting keystroke
                // char key = cv::waitKey(10);
                // if( key == 'q') {
                //     break;
                // }
        }

        delete capdev;
        return(0);
}