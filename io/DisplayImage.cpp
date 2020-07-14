#include <iostream>
#include <opencv2/core.hpp>
//#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
int main() {
  //std::string image_path = samples::findFile("starry_night.jpg");
  std::string image_path = "C:\\Users\\User\\Desktop\\rjn.png";
  Mat img = imread(image_path, IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
  imshow("Display window", img);
  int k = waitKey(0);  // Wait for a keystroke in the window
  if (k == 's') {
    imwrite("C:\\Users\\User\\Desktop\\copy.png", img);
  }
  return 0;
}