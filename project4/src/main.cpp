#include <iostream>
#include <vector>
#include <string>
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
  cv::Mat frameGray;
  cv::Size patternsize(9, 6); //interior number of corners

  std::vector<cv::Vec3f> point_set;
	std::vector<std::vector<cv::Vec3f> > point_list;
	std::vector<std::vector<cv::Point2f> > corner_list;
  std::string delimiter(50, '-');
  int count = 0;

  for(;;) {
      *capdev >> frame; // get a new frame from the camera, treat as a stream
      if( frame.empty() ) {
          printf("frame is empty\n");
          break;
      }

      cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

      std::vector<cv::Point2f> corner_set;
      bool patternfound = cv::findChessboardCorners(frameGray, patternsize, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
            + cv::CALIB_CB_FAST_CHECK);

      std::cout << "patternfound: " << patternfound << std::endl;

      if (patternfound) {
        cv::cornerSubPix(frameGray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      }

      cv::drawChessboardCorners(frame, patternsize, cv::Mat(corner_set), patternfound); 
      cv::imshow("Video", frame);
      char key = cv::waitKey(10);

      if (key == 'q') {
        break;
      }

      if (key == 's') {
        if (!patternfound) {
          std::cout << "No Corners found!" << std::endl;
        }

        // if user types 's' and corners are found, save the frames. Also, save the image points and corner points for further calibration
        else {
          std::cout << delimiter << std::endl;
          // std::cout << "calibration frame " + std::to_string(count) + " saved." << std::endl;

          std::cout << patternsize.height << std::endl;
          std::cout << patternsize.width << std::endl;

          // generate the point set
          for (int i = 0; i < patternsize.height; i++) {
            for (int j = 0; j < patternsize.width; j++) {
              point_set.push_back(cv::Point3f((float) j, (float) (-i), 0.0f));
            }
          }

          corner_list.push_back(std::vector<cv::Point2f>(corner_set));
          point_list.push_back(std::vector<cv::Vec3f>(point_set));

          // print_set("adding corner set" + std::to_string(count) + " to the corner list:", point_set);
          // print_set("", corner_set);

          std::cout << std::string("./img/") + std::string("screenshot_") + std::to_string(count++) + std::string(".jpg") << std::endl;
          cv::imwrite(std::string("./img/") + std::string("screenshot_") + std::to_string(count++) + std::string(".jpg"), frame);
        }
      } else if (key == 'c') {
        
      }
  }
  
  return 0;
} 
