#include <iostream>
#include <vector>
// #include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

int main() {
  cv::VideoCapture *capdev;
  // open the video device
  capdev = new cv::VideoCapture(0);
  if ( !capdev->isOpened() ) {
          printf("Unable to open video device\n");
          return(-1);
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);

  cv::namedWindow("Video", 1); // identifies a window
  cv::Mat frame;
  cv::Size patternsize(8,6); //interior number of corners

  for(;;) {
      *capdev >> frame; // get a new frame from the camera, treat as a stream
      if( frame.empty() ) {
          printf("frame is empty\n");
          break;
      }
      
      // cv::Mat gray = ....; //source image
      std::vector<cv::Point2f> corner_set;
      bool patternfound = cv::findChessboardCorners(frame, patternsize, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);
      if (patternfound) {
        cv::cornerSubPix(frame, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      }

      // cv::drawChessboardCorners(img, patternsize, Mat(corners), patternfound); 
  }


  return 0;
}