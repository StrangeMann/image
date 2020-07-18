#ifndef EDMA_MEASURE
#define EDMA_MEASURE

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgcodecs.hpp>

namespace edma_measure {
struct CSVRow {
  CSVRow() = default;
  std::istream& readNextRow(std::istream& str);

  std::vector<std::string> row_;
};

class EdmaData {
 public:
  EdmaData(std::ifstream& points_symmetry);

  bool Fill(const std::vector<std::string>& row, std::string img_folder);
  std::vector<double> Metrics();
  std::string Name();
  void DisplayImage();
  void DrawCenter(std::vector<double>& line);

 private:
  void CountRanges();
  static double Range(cv::Point2d p1, cv::Point2d p2);
  void MakeVectorUnit(std::vector<double>& line);

  void SetDiffToRatio();
  void SetDiffToSubstr();

  // Metrics depend on what is contained in range_diff_
  double MinMaxMetric();
  double AvgMetric(int sampling_frame, int repeats, double alpha);

  double CenterMetric();

  std::vector<double> CenterLine();  // output: vx, vy, x0, y0

  std::string name_;
  std::vector<cv::Point2d> coordinates_;
  std::vector<int> right_points_;
  std::vector<int> left_points_;
  std::vector<int> central_points_;
  std::vector<std::vector<double>> r_ranges_;
  std::vector<std::vector<double>> l_ranges_;
  std::vector<double> range_diff_;

  cv::Mat img_;
};

}  // namespace edma_measure

#endif
