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
// struct CSVRow {
//  CSVRow() = default;
//  std::istream& readNextRow(std::istream& str);
//
//  std::vector<std::string> row_;
//};
//
// class ImageProcessor {
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
struct Sample {
  void write(cv::FileStorage& output) const;
  void read(const cv::FileNode& node);
  std::string name_;
  std::vector<double> features_;
  bool is_right_;
};
static void write(cv::FileStorage& fs, const std::string&, const Sample& x);
static void read(const cv::FileNode& node, Sample& x,
                 const Sample& default_value = Sample());
struct DataRow {
  Sample data_;
  double weight_;
};

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
void RetrieveFeatures(cv::Mat& img, std::vector<double>& features,int n_zones);
// in current implementation 200%n_zones==0 must be true
bool ObtainData(std::string path, std::string right_images_txt,int n_zones);

struct Stump {
  Stump();
  Stump(int feature, double threshold, double weight);

  void SetWeight(double error);
  bool Classify(const Sample& sample);

  void write(cv::FileStorage& output) const;
  void read(const cv::FileNode& node);

  int feature_;
  double threshold_;
  double weight_;
};
static void write(cv::FileStorage& fs, const std::string&, const Stump& x);
static void read(const cv::FileNode& node, Stump& x,
                 const Stump& default_value = Stump());
void TeachAdaBoost(std::string input, std::string output,
                   double use_for_learning_part, int n_stumps);
void ReadData(std::string input, std::vector<DataRow>& output);
void SeparateDataset(std::vector<DataRow>& samples,
                     std::vector<DataRow>& assessment_dataset,
                     double use_for_learning_part);
std::pair<Stump, double> FindThreshold(std::vector<DataRow>& samples,
                                       int column);  // Stump, gini impurity
std::tuple<int, int, int, int> Occurances(
    std::vector<DataRow>& samples,
    Stump& stump);  // yes for true, no for true, yes for false, no for false
double GiniImpurity(std::tuple<int, int, int, int> occurances);
double GiniImpurity(double p1, double p2);
Stump GetBestStump(std::vector<DataRow>& samples);
void UpdateDataset(std::vector<DataRow>& samples, Stump stump);
double IncreasedWeight(double weight, double stump_weight);
double DecreasedWeight(double weight, double stump_weight);

void AssignByAdaBoost(std::vector<DataRow>& data,std::vector<Stump>& stumps);
}  // namespace face_assessment

#endif
