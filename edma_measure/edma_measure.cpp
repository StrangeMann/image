#include "edma_measure.h"

#include <fstream>
#include <random>
#include <regex>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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

EdmaData::EdmaData(std::ifstream& file) {
  std::string line;
  std::getline(file, line);
  std::stringstream lineStream(line);
  int point;
  while (lineStream >> point) {
    right_points.push_back(point);
  }
  std::getline(file, line);
  lineStream = std::stringstream(line);
  while (lineStream >> point) {
    left_points.push_back(point);
  }
  l_ranges.resize(left_points.size(), std::vector<double>(left_points.size()));
  r_ranges.resize(left_points.size(), std::vector<double>(left_points.size()));
}

bool EdmaData::Fill(const std::vector<std::string>& row) {
  name = row.front();

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
      coordinates_.push_back(std::make_pair(x, y));
    }
    CountRanges();
    return true;
  }
  return false;
}
double EdmaData::Range(std::pair<double, double> p1,
                       std::pair<double, double> p2) const {
  return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                   (p1.second - p2.second) * (p1.second - p2.second));
}

void EdmaData::CountRanges() {
  for (int i = 0; i < left_points.size(); ++i) {
    std::pair<double, double> r_p1, l_p1;
    r_p1 = coordinates_[right_points[i]];
    l_p1 = coordinates_[left_points[i]];
    for (int j = i + 1; j < left_points.size(); ++j) {
      std::pair<double, double> r_p2, l_p2;
      r_p2 = coordinates_[right_points[j]];
      l_p2 = coordinates_[left_points[j]];
      r_ranges[i][j] = Range(r_p1, r_p2);
      l_ranges[i][j] = Range(l_p1, l_p2);
    }
  }
}
void EdmaData::SetDiffToRatio() {
  range_diff.clear();
  for (int i = 0; i < left_points.size(); ++i) {
    for (int j = i + 1; j < left_points.size(); ++j) {
      range_diff.push_back(abs(r_ranges[i][j] / l_ranges[i][j]));
    }
  }
}
void EdmaData::SetDiffToSubstr() {
  range_diff.clear();
  for (int i = 0; i < left_points.size(); ++i) {
    for (int j = i + 1; j < left_points.size(); ++j) {
      range_diff.push_back(r_ranges[i][j] - l_ranges[i][j]);
    }
  }
}
double EdmaData::MinMaxMetric() {
  double max_diff = std::numeric_limits<decltype(max_diff)>::min();
  double min_diff = std::numeric_limits<decltype(min_diff)>::max();
  for (int i = 0; i < range_diff.size(); ++i) {
    max_diff = std::max(max_diff, range_diff[i]);
    min_diff = std::min(min_diff, range_diff[i]);
  }
  return max_diff / min_diff;
}
double EdmaData::AvgMetric(int sapmling_frame, int repeats, double alpha) {
  std::random_device rd;
  std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df, 11,
                               0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
                               1812433253>
      generator(rd());
  std::uniform_int_distribution<int> distribution(0, range_diff.size() - 1);
  std::vector<double> sums;
  for (int i = 0; i < repeats; ++i) {
    double sum = 0;
    for (int j = 0; j < sapmling_frame; ++j) {
      sum += range_diff[distribution(generator)];
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
  return metrics;
}
std::string EdmaData::Name() { return name; }
}  // namespace edma_measure

using namespace edma_measure;
using namespace std;
using namespace cv;
int main() {
  std::ifstream landmarks(
      "../../src/edma_measure/muct76-opencv.csv");
  std::ifstream points("../../src/edma_measure/points.txt");
  CSVRow row;
  EdmaData data(points);
  double min(10000);
  std::string min_name;
  double max(-10000);
  std::string max_name;
  while (row.readNextRow(landmarks)) {
    if (data.Fill(row.row_)) {
      std::vector<double> metrics = data.Metrics();
      if (metrics[0] < 3) {
        string name = data.Name();
        cout << metrics[0] << ' ' << metrics[1] << ' ' << metrics[2] << ' '
             << metrics[3] << ' ' << name << '\n';
        std::string image_path = "../../src/edma_measure/jpg/";
        image_path += name;
        image_path += ".jpg";
        Mat img = imread(image_path, IMREAD_COLOR);
        if (img.empty()) {
          std::cout << "Could not read the image: " << image_path << std::endl;
        }
        imshow("Display window", img);
        int k = waitKey(0);  // Wait for a keystroke in the window
        
      }
    }
  }
  return 0;
  //
}