#ifndef EDMA_MEASURE
#define EDMA_MEASURE

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgcodecs.hpp>

namespace face_assessment {
//struct CSVRow {
//  CSVRow() = default;
//  std::istream& readNextRow(std::istream& str);
//
//  std::vector<std::string> row_;
//};
//
//class ImageProcessor {
// public:
//  ImageProcessor(std::ifstream& points_symmetry);
//
//  ImageProcessor(std::ifstream& points_symmetry, std::string classifier);
//
//  bool Fill(const std::vector<std::string>& row, std::string img_folder);
//  std::vector<double> Metrics();
//  std::string Name();
//  void DisplayImage();
//  void DrawCenter(std::vector<double>& line);
//  void OutlineFace();
//
//  void Rotation();
//
// private:
//  void CountRanges();
//  static double Range(cv::Point2d p1, cv::Point2d p2);
//  void MakeVectorUnit(std::vector<double>& line);
//
//  void SetDiffToRatio();
//  void SetDiffToSubstr();
//
//  // Metrics depend on what is contained in range_diff_
//  double MinMaxMetric();
//  double AvgMetric(int sampling_frame, int repeats, double alpha);
//
//  double CenterMetric();
//
//  void FindFace(std::vector<cv::Rect>& output);
//
//  std::vector<double> CenterLine();  // output: vx, vy, x0, y0
//
//  std::string name_;
//  std::vector<cv::Point2d> coordinates_;
//  std::vector<int> right_points_;
//  std::vector<int> left_points_;
//  std::vector<int> central_points_;
//  std::vector<std::vector<double>> r_ranges_;
//  std::vector<std::vector<double>> l_ranges_;
//  std::vector<double> range_diff_;
//
//  cv::Mat img_;
//
//  cv::CascadeClassifier face_detector_;
//};

template <typename _Tp>
void OLBP_(const cv::Mat& src, cv::Mat& dst);
template <typename _Tp>
void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors);

bool CutFace(cv::Mat& src, cv::Mat& dest, cv::CascadeClassifier& face_detector,
             const std::vector<double>& center_line);
std::vector<double> SplitFace(cv::Mat& src, const std::vector<cv::Rect>& eyes);
std::vector<double> CenterLine(cv::Point2d a, cv::Point2d b);
void FindEyes(cv::Mat& src, std::vector<cv::Rect>& eyes,
              cv::CascadeClassifier eye_detector);
void SelectTwoEyes(std::vector<cv::Rect>& eyes);
bool Allign(cv::Mat& img, const std::vector<double>& central_line);
void RecordFeatures(cv::Mat& img,std::ofstream& csv_file,std::string img_name);
}  // namespace face_assessment

#endif
