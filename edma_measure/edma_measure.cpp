#include "edma_measure.h"

#include <fstream>
#include <random>
#include <regex>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace edma_measure {

std::istream& CSVRow::readNextRow(std::istream& str) {
  row_.clear();

  std::string line;
  std::getline(str, line);
  std::stringstream lineStream(line);
  std::string cell;
  while (std::getline(lineStream, cell, ',')) {
    row_.push_back(cell);
  }
  return str;
}

EdmaData::EdmaData(std::ifstream& points_symmetry) {
  std::string line;
  std::getline(points_symmetry, line);
  std::stringstream lineStream(line);
  int point;
  while (lineStream >> point) {
    right_points_.push_back(point);
  }
  std::getline(points_symmetry, line);
  lineStream = std::stringstream(line);
  for (int i = 0; i < right_points_.size(); ++i) {
    lineStream >> point;
    left_points_.push_back(point);
    if (left_points_[i] == right_points_[i]) {
      central_points_.push_back(left_points_[i]);
    }
  }
  l_ranges_.resize(left_points_.size(),
                   std::vector<double>(left_points_.size()));
  r_ranges_.resize(left_points_.size(),
                   std::vector<double>(left_points_.size()));
}
void EdmaData::DisplayImage() {
  cv::imshow("Display window", img_);
  int k = cv::waitKey(0);  // Wait for a keystroke in the window
}
bool EdmaData::Fill(const std::vector<std::string>& row,
                    std::string img_folder) {
  name_ = row.front();
  std::regex re("^.....[a]");
  if (regex_search(row.front().begin(), row.front().end(), re)) {
    coordinates_.clear();
    for (int i = 3; i < row.size(); i += 2) {
      double x, y;
      std::stringstream ss(row[i - 1]);
      ss >> x;
      ss = std::stringstream(row[i]);
      ss << row[i];
      ss >> y;
      coordinates_.push_back(cv::Point2d(x, y));
    }
    CountRanges();
    img_folder += name_;
    img_folder += ".jpg";
    img_ = cv::imread(img_folder, cv::IMREAD_COLOR);
    if (img_.empty()) {
      std::cout << "Could not read the image: " << img_folder << std::endl;
    }
    return true;
  }
  return false;
}
double EdmaData::Range(cv::Point2d p1, cv::Point2d p2) {
  return cv::norm(p1 - p2);
}

void EdmaData::CountRanges() {
  for (int i = 0; i < left_points_.size(); ++i) {
    cv::Point2d r_p1, l_p1;
    r_p1 = coordinates_[right_points_[i]];
    l_p1 = coordinates_[left_points_[i]];
    for (int j = i + 1; j < left_points_.size(); ++j) {
      cv::Point2d r_p2, l_p2;
      r_p2 = coordinates_[right_points_[j]];
      l_p2 = coordinates_[left_points_[j]];
      r_ranges_[i][j] = Range(r_p1, r_p2);
      l_ranges_[i][j] = Range(l_p1, l_p2);
    }
  }
}
void EdmaData::SetDiffToRatio() {
  range_diff_.clear();
  for (int i = 0; i < left_points_.size(); ++i) {
    for (int j = i + 1; j < left_points_.size(); ++j) {
      range_diff_.push_back(abs(r_ranges_[i][j] / l_ranges_[i][j]));
    }
  }
}
void EdmaData::SetDiffToSubstr() {
  range_diff_.clear();
  for (int i = 0; i < left_points_.size(); ++i) {
    for (int j = i + 1; j < left_points_.size(); ++j) {
      range_diff_.push_back(r_ranges_[i][j] - l_ranges_[i][j]);
    }
  }
}
double EdmaData::MinMaxMetric() {
  double max_diff = std::numeric_limits<decltype(max_diff)>::min();
  double min_diff = std::numeric_limits<decltype(min_diff)>::max();
  for (int i = 0; i < range_diff_.size(); ++i) {
    max_diff = std::max(max_diff, range_diff_[i]);
    min_diff = std::min(min_diff, range_diff_[i]);
  }
  return max_diff / min_diff;
}
double EdmaData::AvgMetric(int sapmling_frame, int repeats, double alpha) {
  std::random_device rd;
  std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df, 11,
                               0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
                               1812433253>
      generator(rd());
  std::uniform_int_distribution<int> distribution(0, range_diff_.size() - 1);
  std::vector<double> sums;
  for (int i = 0; i < repeats; ++i) {
    double sum = 0;
    for (int j = 0; j < sapmling_frame; ++j) {
      sum += range_diff_[distribution(generator)];
    }
    sum /= sapmling_frame;
    sums.push_back(sum);
  }
  std::sort(sums.begin(), sums.end());
  int begin = sums.size() * alpha / 2;
  int end = sums.size() * (1.0 - alpha / 2);
  double sum = 0;
  for (int i = begin; i < end; ++i) {
    sum += sums[i];
  }
  sum /= (sums.size() * (1.0 - alpha));
  return sum;
}
std::vector<double> EdmaData::Metrics() {
  std::vector<double> metrics;
  SetDiffToRatio();
  metrics.push_back(MinMaxMetric());
  metrics.push_back(AvgMetric(7, 100, 0.1));
  SetDiffToSubstr();
  metrics.push_back(MinMaxMetric());
  metrics.push_back(AvgMetric(7, 100, 0.1));

  metrics.push_back(CenterMetric());
  return metrics;
}
std::string EdmaData::Name() { return name_; }

std::vector<double> EdmaData::CenterLine() {
  std::vector<double> output(4);  // vx, vy, x0, y0
  std::vector<cv::Point2d> input;
  for (int i = 0; i < central_points_.size(); ++i) {
    input.push_back(coordinates_[central_points_[i]]);
  }
  cv::fitLine(input, output, cv::DIST_L2, 0, 0.01, 0.01);
  return output;
}
double EdmaData::CenterMetric() {
  std::vector<double> line(CenterLine());
  MakeVectorUnit(line);
  cv::Point2d unit_vec(line[0], line[1]), reference_vec(0, 1);
  // std::cout<<line[0]<<' '<<line[1]<<'\n';
  DrawCenter(line);
  return Range(unit_vec, reference_vec);
}
void EdmaData::MakeVectorUnit(std::vector<double>& line) {
  line[0] = abs(line[0]);
  line[1] = abs(line[1]);
  line[0] /= line[1];
  line[1] = 1;
}
void EdmaData::DrawCenter(std::vector<double>& line) {
  int t(1000);
  cv::Point2d p1(line[2], line[3]), p2;
  p2.x = p1.x + line[0] * t;
  p2.y = p1.y + line[1] * t;
  p1.x -= line[0] * t;
  p1.y -= line[1] * t;
  const cv::Scalar blue(255, 0, 0);
  cv::line(img_, p1, p2, blue);
}
}  // namespace edma_measure

using namespace edma_measure;
using namespace std;
using namespace cv;
int main() {
  double eps(0.02);

  std::vector<std::string> names_of_best;
  std::ifstream best("../../src/edma_measure/best_img.txt");
  std::string best_name;
  while (best >> best_name) {
    names_of_best.push_back(best_name);
  }

  std::ifstream landmarks("../../src/edma_measure/muct76-opencv.csv");
  std::ifstream points("../../src/edma_measure/points.txt");
  std::string image_path = "../../src/edma_measure/jpg/";
  CSVRow row;
  EdmaData data(points);
  double min(10000);
  std::string min_name;
  double max(-10000);
  std::string max_name;
  while (row.readNextRow(landmarks)) {
    if (data.Fill(row.row_, image_path)) {
      std::vector<double> metrics = data.Metrics();
      if (metrics[4] < eps) {
        string name = data.Name();
        for (auto x : metrics) {
          cout << x << ' ';
        }
        cout << "name: " << name << ' ';
        if(std::find(names_of_best.begin(),names_of_best.end(),name) != names_of_best.end()){
          cout<<"********";
        }
        cout<<'\n';
        data.DisplayImage();
      }
    }
  }
  return 0;
  //
}