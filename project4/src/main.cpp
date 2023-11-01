#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

int main() {
  cv::Size patternsize(8,6); //interior number of corners
  cv::Mat gray = ....; //source image
  std::vector<cv::Point2f> corner_set;
  bool patternfound = cv::findChessboardCorners(gray, patternsize, corners,
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);
  if (patternfound) {
    cv::cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
      TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  }

  cv::drawChessboardCorners(img, patternsize, Mat(corners), patternfound); 
  return 0;
}