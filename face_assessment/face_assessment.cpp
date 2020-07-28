#include "face_assessment.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace face_assessment {

const double PI = 3.14158;

// std::istream& CSVRow::readNextRow(std::istream& str) {
//  row_.clear();
//
//  std::string line;
//  std::getline(str, line);
//  std::stringstream lineStream(line);
//  std::string cell;
//  while (std::getline(lineStream, cell, ',')) {
//    row_.push_back(cell);
//  }
//  return str;
//}
//
// ImageProcessor::ImageProcessor(std::ifstream& points_symmetry) {
//  std::string line;
//  std::getline(points_symmetry, line);
//  std::stringstream lineStream(line);
//  int point;
//  while (lineStream >> point) {
//    right_points_.push_back(point);
//  }
//  std::getline(points_symmetry, line);
//  lineStream = std::stringstream(line);
//  for (int i = 0; i < right_points_.size(); ++i) {
//    lineStream >> point;
//    left_points_.push_back(point);
//    if (left_points_[i] == right_points_[i]) {
//      central_points_.push_back(left_points_[i]);
//    }
//  }
//  l_ranges_.resize(left_points_.size(),
//                   std::vector<double>(left_points_.size()));
//  r_ranges_.resize(left_points_.size(),
//                   std::vector<double>(left_points_.size()));
//}
// void ImageProcessor::DisplayImage() {
//  cv::imshow("Display window", img_);
//  int k = cv::waitKey(0);  // Wait for a keystroke in the window
//}
// bool ImageProcessor::Fill(const std::vector<std::string>& row,
//                          std::string img_folder) {
//  name_ = row.front();
//  std::regex re("^.....[a]");
//  if (regex_search(row.front().begin(), row.front().end(), re)) {
//    coordinates_.clear();
//    for (int i = 3; i < row.size(); i += 2) {
//      double x, y;
//      std::stringstream ss(row[i - 1]);
//      ss >> x;
//      ss = std::stringstream(row[i]);
//      ss << row[i];
//      ss >> y;
//      coordinates_.push_back(cv::Point2d(x, y));
//    }
//    CountRanges();
//    img_folder += name_;
//    img_folder += ".jpg";
//    img_ = cv::imread(img_folder, cv::IMREAD_COLOR);
//    if (img_.empty()) {
//      std::cout << "Could not read the image: " << img_folder << std::endl;
//    }
//    return true;
//  }
//  return false;
//}
// double ImageProcessor::Range(cv::Point2d p1, cv::Point2d p2) {
//  return cv::norm(p1 - p2);
//}
//
// void ImageProcessor::CountRanges() {
//  for (int i = 0; i < left_points_.size(); ++i) {
//    cv::Point2d r_p1, l_p1;
//    r_p1 = coordinates_[right_points_[i]];
//    l_p1 = coordinates_[left_points_[i]];
//    for (int j = i + 1; j < left_points_.size(); ++j) {
//      cv::Point2d r_p2, l_p2;
//      r_p2 = coordinates_[right_points_[j]];
//      l_p2 = coordinates_[left_points_[j]];
//      r_ranges_[i][j] = Range(r_p1, r_p2);
//      l_ranges_[i][j] = Range(l_p1, l_p2);
//    }
//  }
//}
// void ImageProcessor::SetDiffToRatio() {
//  range_diff_.clear();
//  for (int i = 0; i < left_points_.size(); ++i) {
//    for (int j = i + 1; j < left_points_.size(); ++j) {
//      range_diff_.push_back(abs(r_ranges_[i][j] / l_ranges_[i][j]));
//    }
//  }
//}
// void ImageProcessor::SetDiffToSubstr() {
//  range_diff_.clear();
//  for (int i = 0; i < left_points_.size(); ++i) {
//    for (int j = i + 1; j < left_points_.size(); ++j) {
//      range_diff_.push_back(r_ranges_[i][j] - l_ranges_[i][j]);
//    }
//  }
//}
// double ImageProcessor::MinMaxMetric() {
//  double max_diff = std::numeric_limits<decltype(max_diff)>::min();
//  double min_diff = std::numeric_limits<decltype(min_diff)>::max();
//  for (int i = 0; i < range_diff_.size(); ++i) {
//    max_diff = std::max(max_diff, range_diff_[i]);
//    min_diff = std::min(min_diff, range_diff_[i]);
//  }
//  return max_diff / min_diff;
//}
// double ImageProcessor::AvgMetric(int sapmling_frame, int repeats,
//                                 double alpha) {
//  std::random_device rd;
//  std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df,
//  11,
//                               0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
//                               1812433253>
//      generator(rd());
//  std::uniform_int_distribution<int> distribution(0, range_diff_.size() - 1);
//  std::vector<double> sums;
//  for (int i = 0; i < repeats; ++i) {
//    double sum = 0;
//    for (int j = 0; j < sapmling_frame; ++j) {
//      sum += range_diff_[distribution(generator)];
//    }
//    sum /= sapmling_frame;
//    sums.push_back(sum);
//  }
//  std::sort(sums.begin(), sums.end());
//  int begin = sums.size() * alpha / 2;
//  int end = sums.size() * (1.0 - alpha / 2);
//  double sum = 0;
//  for (int i = begin; i < end; ++i) {
//    sum += sums[i];
//  }
//  sum /= (sums.size() * (1.0 - alpha));
//  return sum;
//}
// std::vector<double> ImageProcessor::Metrics() {
//  std::vector<double> metrics;
//  SetDiffToRatio();
//  metrics.push_back(MinMaxMetric());
//  metrics.push_back(AvgMetric(7, 100, 0.1));
//  SetDiffToSubstr();
//  metrics.push_back(MinMaxMetric());
//  metrics.push_back(AvgMetric(7, 100, 0.1));
//
//  metrics.push_back(CenterMetric());
//  return metrics;
//}
// std::string ImageProcessor::Name() { return name_; }
//
// std::vector<double> ImageProcessor::CenterLine() {
//  std::vector<double> output(4);  // vx, vy, x0, y0
//  std::vector<cv::Point2d> input;
//  for (int i = 0; i < central_points_.size(); ++i) {
//    input.push_back(coordinates_[central_points_[i]]);
//  }
//  cv::fitLine(input, output, cv::DIST_L2, 0, 0.01, 0.01);
//  return output;
//}
// double ImageProcessor::CenterMetric() {
//  std::vector<double> line(CenterLine());
//  MakeVectorUnit(line);
//  cv::Point2d unit_vec(line[0], line[1]), reference_vec(0, 1);
//  // std::cout<<line[0]<<' '<<line[1]<<'\n';
//  // DrawCenter(line);
//  return Range(unit_vec, reference_vec);
//}
// void ImageProcessor::MakeVectorUnit(std::vector<double>& line) {
//  line[0] = abs(line[0]);
//  line[1] = abs(line[1]);
//  line[0] /= line[1];
//  line[1] = 1;
//}
// void ImageProcessor::DrawCenter(std::vector<double>& line) {
//  int t(1000);
//  cv::Point2d p1(line[2], line[3]), p2;
//  p2.x = p1.x + line[0] * t;
//  p2.y = p1.y + line[1] * t;
//  p1.x -= line[0] * t;
//  p1.y -= line[1] * t;
//  const cv::Scalar blue(255, 0, 0);
//  cv::line(img_, p1, p2, blue);
//}
//
// ImageProcessor::ImageProcessor(std::ifstream& points_symmetry,
//                               std::string classifier)
//    : ImageProcessor(points_symmetry) {
//  if (!face_detector_.load(classifier)) {
//    std::cout << "not loaded classifier\n";
//  }
//}
// void ImageProcessor::FindFace(std::vector<cv::Rect>& output) {
//  double scale(0.25);
//  double face_size(0.2);
//  cv::Mat grayscale;
//  cv::cvtColor(img_, grayscale, cv::COLOR_BGR2GRAY);
//  cv::resize(grayscale, grayscale, cv::Size(), scale, scale);
//  cv::imshow("gray", grayscale);
//  face_detector_.detectMultiScale(
//      grayscale, output, 1.1, 3, 0,
//      cv::Size(grayscale.cols * face_size, grayscale.rows * face_size));
//  for (int i = 0; i < output.size(); ++i) {
//    output[i].x /= scale;
//    output[i].y /= scale;
//    output[i].height /= scale;
//    output[i].width /= scale;
//  }
//}
// void ImageProcessor::OutlineFace() {
//  std::vector<cv::Rect> faces;
//  FindFace(faces);
//  for (auto face : faces) {
//    cv::Scalar red(0, 0, 255);
//    cv::rectangle(img_, face, red);
//  }
//}
// void RotateImage(cv::Mat& img, double angle) {
//  cv::Point center(img.cols / 2, img.rows / 2);
//  cv::Mat rot_mat(cv::getRotationMatrix2D(center, angle, 1));
//  cv::warpAffine(img, img, rot_mat, img.size());
//}
// void ImageProcessor::Rotation() {
//  cv::Mat orig = img_.clone();
//  for (int i = 0; i < 90; ++i) {
//    img_ = orig.clone();
//    cv::imshow("fresh", orig);
//    RotateImage(img_, i);
//    OutlineFace();
//    DisplayImage();
//  }
//}
// why does this not work with neighbors<16
template <typename _Tp>
void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors) {
  neighbors = max(min(neighbors, 31), 1);
  dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
  for (int n = 0; n < neighbors; n++) {
    // sample points
    float x = static_cast<float>(radius) *
              cos(2.0 * PI * n / static_cast<float>(neighbors));
    float y = static_cast<float>(radius) *
              -sin(2.0 * PI * n / static_cast<float>(neighbors));
    // relative indices
    int fx = static_cast<int>(floor(x));
    int fy = static_cast<int>(floor(y));
    int cx = static_cast<int>(ceil(x));
    int cy = static_cast<int>(ceil(y));
    // fractional part
    float ty = y - fy;
    float tx = x - fx;
    // set interpolation weights
    float w1 = (1 - tx) * (1 - ty);
    float w2 = tx * (1 - ty);
    float w3 = (1 - tx) * ty;
    float w4 = tx * ty;
    // iterate through your data
    for (int i = radius; i < src.rows - radius; i++) {
      for (int j = radius; j < src.cols - radius; j++) {
        float t = w1 * src.at<_Tp>(i + fy, j + fx) +
                  w2 * src.at<_Tp>(i + fy, j + cx) +
                  w3 * src.at<_Tp>(i + cy, j + fx) +
                  w4 * src.at<_Tp>(i + cy, j + cx);
        // we are dealing with floating point precision, so add some little
        // tolerance
        dst.at<int>(i - radius, j - radius) +=
            ((t > src.at<_Tp>(i, j)) && (abs(t - src.at<_Tp>(i, j)) >
                                         std::numeric_limits<float>::epsilon()))
            << n;
        if (n == neighbors - 1) {
          // std::cout << dst.at<int>(i - radius, j - radius) << '\n';
        }
      }
    }
  }
}
template <typename _Tp>
void OLBP_(const cv::Mat& src, cv::Mat& dst) {
  dst = cv::Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
  for (int i = 1; i < src.rows - 1; i++) {
    for (int j = 1; j < src.cols - 1; j++) {
      _Tp center = src.at<_Tp>(i, j);
      char code = 0;
      code |= (src.at<_Tp>(i - 1, j - 1) > center) << 7;
      code |= (src.at<_Tp>(i - 1, j) > center) << 6;
      code |= (src.at<_Tp>(i - 1, j + 1) > center) << 5;
      code |= (src.at<_Tp>(i, j + 1) > center) << 4;
      code |= (src.at<_Tp>(i + 1, j + 1) > center) << 3;
      code |= (src.at<_Tp>(i + 1, j) > center) << 2;
      code |= (src.at<_Tp>(i + 1, j - 1) > center) << 1;
      code |= (src.at<_Tp>(i, j - 1) > center) << 0;
      dst.at<uchar>(i - 1, j - 1) = code;
    }
  }
}
bool CutFace(cv::Mat& src, cv::Mat& dest, cv::CascadeClassifier& face_detector,
             const std::vector<double>& center_line) {
  std::vector<cv::Rect> faces;
  double scale(0.25);
  double face_size(0.2);
  cv::resize(src, dest, cv::Size(), scale, scale);
  face_detector.detectMultiScale(
      dest, faces, 1.1, 3, 0,
      cv::Size(dest.cols * face_size, dest.rows * face_size));
  if (faces.size() != 1) {
    return false;
  }
  // return to original scale
  faces[0].x /= scale;
  faces[0].y /= scale;
  faces[0].height /= scale;
  faces[0].width /= scale;
  // change area so that center line is in the center
  if (center_line[2] - faces[0].x <
      faces[0].x + faces[0].width -
          center_line[2]) {  // if left border is closer
    faces[0].width = 2 * (center_line[2] - faces[0].x);
  } else {
    int width = 2 * (faces[0].x + faces[0].width - center_line[2]);
    faces[0].x += (faces[0].width - width);
    faces[0].width = width;
  }
  dest = cv::Mat(src, faces[0]).clone();
  resize(dest, dest, cv::Size(202, 202));
  return true;
}
void FindEyes(cv::Mat& src, std::vector<cv::Rect>& eyes,
              cv::CascadeClassifier eye_detector) {
  double scale(1);
  // double eye_size(0.2);
  cv::Mat dest;
  cv::resize(src, dest, cv::Size(), scale, scale);
  eye_detector.detectMultiScale(
      dest, eyes, 1.1, 3,
      0 /*,cv::Size(dest.cols * eye_size, dest.rows * eye_size)*/);
  for (int i = 0; i < eyes.size(); ++i) {
    eyes[i].x /= scale;
    eyes[i].y /= scale;
    eyes[i].height /= scale;
    eyes[i].width /= scale;
  }
}
void SelectTwoEyes(std::vector<cv::Rect>& eyes) {
  if (eyes.size() < 2) {
    eyes.clear();
    return;
  }
  std::pair<int, int> eye_indexes(0, 1);
  double max = std::numeric_limits<decltype(max)>::min();
  for (int i = 0; i < eyes.size(); ++i) {
    for (int j = i + 1; j < eyes.size(); ++j) {
      if (abs(eyes[j].x - eyes[i].x) < 30) {
        continue;
      }
      double metric = (eyes[j].x - eyes[i].x) * (eyes[j].x - eyes[i].x) -
                      abs(eyes[j].y * eyes[j].y - eyes[i].y * eyes[i].y);
      if (metric > max) {
        eye_indexes = std::make_pair(i, j);
        max = metric;
      }
    }
  }
  std::pair<cv::Rect, cv::Rect> temp(eyes[eye_indexes.first],
                                     eyes[eye_indexes.second]);
  eyes.resize(2);
  eyes[0] = temp.first;
  eyes[1] = temp.second;
}
std::vector<double> SplitFace(cv::Mat& src, const std::vector<cv::Rect>& eyes) {
  cv::Point2f a_center, b_center;
  b_center.x = eyes[0].x + eyes[0].width / 2;
  b_center.y = eyes[0].y + eyes[0].height / 2;
  a_center.x = eyes[1].x + eyes[1].width / 2;
  a_center.y = eyes[1].y + eyes[1].height / 2;

  return CenterLine(b_center, a_center);
}
std::vector<double> CenterLine(cv::Point2d a, cv::Point2d b) {
  std::vector<double> res(4);  // vx, vy, x0, y0
  if (b.x - a.x != 0) {
    res[0] = -(b.y - a.y) / (b.x - a.x);
    res[1] = 1;
  } else {
    res[0] = 1;
    res[1] = -(b.x - a.x) / (b.y - a.y);
  }
  res[2] = a.x + 0.5 * (b.x - a.x);
  res[3] = a.y + 0.5 * (b.y - a.y);
  return res;
}
bool Allign(cv::Mat& img, const std::vector<double>& central_line) {
  double alpha = atan2(central_line[0], central_line[1]) * 180 / PI;
  if (alpha > 30) {
    return false;
  }
  cv::Point center(img.cols / 2, img.rows / 2);
  cv::Mat rot_mat(cv::getRotationMatrix2D(center, -alpha, 1));
  cv::warpAffine(img, img, rot_mat, img.size());
  return true;
}
// img should be square
void RetrieveFeatures(cv::Mat& img, std::vector<double>& features) {
  features.clear();
  int segments(10);  // must img.rows%segments == 0
  cv::Mat mask_l(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
  cv::Mat mask_r(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < segments; ++y) {
    for (int x = 0; x < segments / 2; ++x) {
      cv::Point l_p1(x * (img.cols / segments), y * (img.rows / segments));
      cv::Point l_p2(l_p1.x + (img.cols / segments),
                     l_p1.y + (img.rows / segments));
      cv::Point r_p1(img.cols - l_p1.x, y * (img.rows / segments));
      cv::Point r_p2(img.cols - l_p2.x, l_p1.y + (img.rows / segments));
      rectangle(mask_l, l_p1, l_p2, cv::Scalar(255), cv::LineTypes::FILLED);
      rectangle(mask_r, r_p1, r_p2, cv::Scalar(255), cv::LineTypes::FILLED);

      //imshow("mask_l", mask_l);
      //imshow("mask_r", mask_r);
      //cv::waitKey();

      cv::Mat hist_l, hist_r;
      std::vector<cv::Mat> img_wrapper{img};
      std::vector<int> channels{0};
      std::vector<int> histSize{256};
      std::vector<float> ranges{0, 256};

      cv::calcHist(img_wrapper, channels, mask_l, hist_l, histSize, ranges);
      cv::calcHist(img_wrapper, channels, mask_r, hist_r, histSize, ranges);
      double metric = cv::compareHist(hist_l, hist_r,
                                      cv::HistCompMethods::HISTCMP_CHISQR_ALT);
      features.push_back(metric);

      rectangle(mask_l, l_p1, l_p2, cv::Scalar(0), cv::LineTypes::FILLED);
      rectangle(mask_r, r_p1, r_p2, cv::Scalar(0), cv::LineTypes::FILLED);
    }
  }
}
void RecordFeatures(const std::vector<double>& features,
                    std::ofstream& csv_file, std::string img_name) {
  csv_file << img_name << ';';
  for (int i = 0; i < features.size(); ++i) {
    csv_file << features[i] << ';';
  }
  csv_file << '\n';
}
}  // namespace face_assessment

using namespace face_assessment;
using namespace std;
using namespace cv;
int main() {
  ofstream data("../../resources/data.csv");
  // for (int i = 0; i < 20; ++i) {
  //  data << i << ';' << i * i << '\n';
  //}
  // data.close();

  std::string image_path("../../resources/jpg/");
  CascadeClassifier face_detector;
  face_detector.load("../../resources/haarcascade_frontalface_default.xml");
  CascadeClassifier eye_detector_l;
  eye_detector_l.load("../../resources/haarcascade_lefteye_2splits.xml");
  CascadeClassifier eye_detector_r;
  eye_detector_r.load("../../resources/haarcascade_righteye_2splits.xml");

  for (const auto& entry : std::filesystem::directory_iterator(image_path)) {
    Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);
    imshow("image", image);

    vector<Rect> eyes_r, eyes_l;
    FindEyes(image, eyes_r, eye_detector_r);
    FindEyes(image, eyes_l, eye_detector_l);
    vector<Rect> eyes(eyes_r);
    eyes.insert(eyes.end(), eyes_l.begin(), eyes_l.end());

    SelectTwoEyes(eyes);
    if (eyes.size() == 2) {
      std::vector<double> central_line(SplitFace(image, eyes));
      // if (!central_line.empty()) {
      //  int t(100);
      //  cv::line(image, cv::Point2d(central_line[2], central_line[3]),
      //           cv::Point2d(central_line[2] + central_line[0] * t,
      //                       central_line[3] + central_line[1] * t),
      //           cv::Scalar(255));
      //}
      if (!Allign(image, central_line)) {
        continue;
      }
      // imshow("image_center", image);
      Mat cut_img;
      CutFace(image, cut_img, face_detector, central_line);
      // imshow("cut_img", image);
      Mat LBP_img;
      OLBP_<char>(cut_img, LBP_img);
      imshow("OLBP image", LBP_img);
      std::vector<double> features;
      RetrieveFeatures(LBP_img, features);
      RecordFeatures(features, data, entry.path().filename().string());
      // waitKey();
    }
  }
  // double eps(0.02);

  // std::vector<std::string> names_of_best;
  // std::ifstream best("../../resources/best_img.txt");
  // std::string best_name;
  // while (best >> best_name) {
  //  names_of_best.push_back(best_name);
  //}

  // std::ifstream landmarks("../../resources/muct76-opencv.csv");
  // std::ifstream points("../../resources/points.txt");
  // std::string image_path("../../resources/jpg/");
  // std::string
  // classifier("../../resources/haarcascade_frontalface_default.xml"); CSVRow
  // row; ImageProcessor data(points, classifier); double min(10000);
  // std::string min_name; double max(-10000); std::string max_name; int
  // count(0);
  //// while (row.readNextRow(landmarks)) {
  ////  if (data.Fill(row.row_, image_path)) {
  ////    data.Rotation();
  ////    // std::vector<double> metrics = data.Metrics();
  ////    // if (metrics[4] < eps) {
  ////    //  string name = data.Name();
  ////    //  for (auto x : metrics) {
  ////    //    cout << x << ' ';
  ////    //  }
  ////    //  cout << "name: " << name << ' ';
  ////    //  if (std::find(names_of_best.begin(), names_of_best.end(), name) !=
  ////    //      names_of_best.end()) {
  ////    //    cout << "********";
  ////    //    ++count;
  ////    //  }
  ////    //  cout << '\n';
  ////    //  data.OutlineFace();
  ////    //  data.DisplayImage();
  ////    //}
  ////  }
  ////}
  // cout << count;
  return 0;
}