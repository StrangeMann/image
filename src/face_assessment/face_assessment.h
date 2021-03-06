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
void OLBP(const cv::Mat& src, cv::Mat& dst);
template <typename _Tp>
void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors);

bool CutFace(cv::Mat& src, cv::Mat& dest, cv::CascadeClassifier& face_detector,
             const std::vector<double>& center_line);
void AllignRect(cv::Rect& rect);
void EnlargeRect(cv::Rect& rect);
void FitRect(cv::Rect& rect, cv::Mat& mat);
std::vector<double> SplitFace(cv::Mat& src, const std::vector<cv::Rect>& eyes);
std::vector<double> CenterLine(cv::Point2d a, cv::Point2d b);
void FindEyes(cv::Mat& src, std::vector<cv::Rect>& eyes,
              cv::CascadeClassifier eye_detector);
void SelectTwoEyes(std::vector<cv::Rect>& eyes);
bool Allign(cv::Mat& img, const std::vector<double>& central_line);
void RetrieveFeatures(cv::Mat& img, std::vector<double>& features,
                      int min_zone_size);
bool ObtainData(std::string right_path, std::string wrong_path,
                std::string output_path, int min_zone_size);
void AddDataFromUncut(std::string image_path, bool is_right,
                      cv::FileStorage& output,
                      cv::CascadeClassifier& face_detector,
                      cv::CascadeClassifier& eye_detector_l,
                      cv::CascadeClassifier& eye_detector_r, int min_zone_size);
void AddDataFromCut(std::string image_path, bool is_right,
                    cv::FileStorage& output, int min_zone_size);
void CutImages(const std::vector<std::string>& paths);

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
                   std::string assessment_path, double use_for_learning_part,
                   int n_stumps);
void ReadData(std::string input, std::vector<DataRow>& output);
void NormalizeDataset(
    std::vector<DataRow>&
        samples);  // assumes all true comes before any false samples
void SeparateDataset(std::vector<DataRow>& samples,
                     std::vector<Sample>& assessment_dataset,
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
void RemoveNoize(std::vector<DataRow>& samples, double median_eps);
void PickData(std::vector<DataRow>& samples);
double IncreasedWeight(double weight, double stump_weight);
double DecreasedWeight(double weight, double stump_weight);

void ReadSamples(std::string data_path, std::vector<Sample>& output);
void ReadStumps(std::string data_path, std::vector<Stump>& output);
void AssignByAdaBoost(std::vector<Sample>& data, std::vector<Stump>& stumps);

void BGRDistortImages(std::vector<Sample>& image_list, std::string images_path,
                   std::string output_right_path,
                   std::string output_wrong_path,double b_k,double g_k,double r_k);
void BGRDistortImage(cv::Mat& image,double b_k,double g_k,double r_k);
void ClusterColors(std::vector<std::string>& image_paths,std::string output_path,int K);
}  // namespace face_assessment

#endif
