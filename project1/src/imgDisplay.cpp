#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = samples::findFile("test.png");
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    
    for (;;) {
        imshow("Display window", img);
        int k = waitKey(0); // Wait for a keystroke in the window
        if (k == 's') {
            imwrite("starry_night.png", img);
        } else if (k == 'q') {
            break;
        }
    }

    return 0;
}