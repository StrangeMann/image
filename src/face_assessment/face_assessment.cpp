#include "face_assessment.h"

#include <algorithm>
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

template <typename _Tp>
void OLBP(const cv::Mat& src, cv::Mat& dst) {
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
  AllignRect(faces[0]);
  EnlargeRect(faces[0]);
  FitRect(faces[0], src);
  dest = cv::Mat(src, faces[0]).clone();
  resize(dest, dest, cv::Size(202, 202));
  return true;
}

void AllignRect(cv::Rect& rect) {
  if (rect.width < 0) {
    rect.x += rect.width;
    rect.width = -rect.width;
  }
  if (rect.height < 0) {
    rect.y += rect.height;
    rect.height = -rect.height;
  }
}
void EnlargeRect(cv::Rect& rect) {
  double up(0.5);
  double down(0.3);
  double sideways(0.4);
  rect.y -= rect.height * up;
  rect.height += rect.height * up + rect.height * down;
  rect.x -= rect.width * sideways;
  rect.width += 2 * rect.width * sideways;
}
void FitRect(cv::Rect& rect, cv::Mat& mat) {
  cv::Point p1(rect.x, rect.y);
  cv::Point p2(rect.x + rect.width, rect.y + rect.height);
  std::vector<cv::Point> points{p1, p2};
  for (int i = 0; i < points.size(); ++i) {
    points[i].x = std::max(0, points[i].x);
    points[i].x = std::min(mat.cols - 1, points[i].x);
    points[i].y = std::max(0, points[i].y);
    points[i].y = std::min(mat.rows - 1, points[i].y);
  }
  rect.x = points[0].x;
  rect.y = points[0].y;
  rect.width = points[1].x - points[0].x;
  rect.height = points[1].y - points[0].y;
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
  // std::cout << alpha << '\n';
  if (abs(alpha) > 20) {
    return false;
  }
  cv::Point center(img.cols / 2, img.rows / 2);
  cv::Mat rot_mat(cv::getRotationMatrix2D(center, -alpha, 1));
  cv::warpAffine(img, img, rot_mat, img.size());
  return true;
}
// img should be square
void RetrieveFeatures(cv::Mat& img, std::vector<double>& features,
                      int min_zone_size) {
  if (img.cols != img.rows) {
    std::cout << "img should be square\n";
    return;
  }
  features.clear();
  cv::Mat mask_l(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
  cv::Mat mask_r(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
  for (int n = min_zone_size; n < img.cols; ++n) {
    if (img.cols % n == 0 && n % 2 == 0) {
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n / 2; ++x) {
          cv::Point l_p1(x * (img.cols / n), y * (img.rows / n));
          cv::Point l_p2(l_p1.x + (img.cols / n), l_p1.y + (img.rows / n));
          cv::Point r_p1(img.cols - l_p1.x, y * (img.rows / n));
          cv::Point r_p2(img.cols - l_p2.x, l_p1.y + (img.rows / n));
          rectangle(mask_l, l_p1, l_p2, cv::Scalar(255), cv::LineTypes::FILLED);
          rectangle(mask_r, r_p1, r_p2, cv::Scalar(255), cv::LineTypes::FILLED);

          // std::cout << l_p1 << ' ' << l_p2 << ' ' << r_p1 << ' ' << l_p2
          //          << '\n';
          // imshow("mask_l", mask_l);
          // imshow("mask_r", mask_r);
          // cv::waitKey();

          cv::Mat hist_l, hist_r;
          std::vector<cv::Mat> img_wrapper{img};
          std::vector<int> channels{0};
          std::vector<int> histSize{256};
          std::vector<float> ranges{0, 256};

          cv::calcHist(img_wrapper, channels, mask_l, hist_l, histSize, ranges);
          cv::calcHist(img_wrapper, channels, mask_r, hist_r, histSize, ranges);
          double metric = cv::compareHist(
              hist_l, hist_r, cv::HistCompMethods::HISTCMP_CHISQR_ALT);
          features.push_back(metric);

          rectangle(mask_l, l_p1, l_p2, cv::Scalar(0), cv::LineTypes::FILLED);
          rectangle(mask_r, r_p1, r_p2, cv::Scalar(0), cv::LineTypes::FILLED);
        }
      }
    }
  }
}
bool ObtainData(std::string right_path, std::string wrong_path,
                std::string output_path, int min_zone_size) {
  using namespace std;
  using namespace cv;
  cv::FileStorage output(output_path, cv::FileStorage::WRITE);
  output << "samples"
         << "[";
  AddDataFromCut(right_path, true, output, min_zone_size);
  AddDataFromCut(wrong_path, false, output, min_zone_size);
  output << "]";
  output.release();
  cvDestroyAllWindows();
  return true;
}
void CutImages(const std::vector<std::string>& paths) {
  cv::CascadeClassifier face_detector;
  face_detector.load("../../resources/haarcascade_frontalface_default.xml");
  cv::CascadeClassifier eye_detector_l;
  eye_detector_l.load("../../resources/haarcascade_lefteye_2splits.xml");
  cv::CascadeClassifier eye_detector_r;
  eye_detector_r.load("../../resources/haarcascade_righteye_2splits.xml");
  for (int i = 0; i < paths.size(); ++i) {
    for (auto entry : std::filesystem::directory_iterator(paths[i])) {
      cv::Mat image = imread(entry.path().string(), cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cout << "image not loaded\n";
        return;
      }
      std::vector<cv::Rect> eyes_r, eyes_l;
      FindEyes(image, eyes_r, eye_detector_r);
      FindEyes(image, eyes_l, eye_detector_l);
      std::vector<cv::Rect> eyes(eyes_r);
      eyes.insert(eyes.end(), eyes_l.begin(), eyes_l.end());
      SelectTwoEyes(eyes);
      if (eyes.size() == 2) {
        std::vector<double> central_line(SplitFace(image, eyes));
        if (!Allign(image, central_line)) {
          continue;
        }
        imshow("image_center", image);
        cv::Mat cut_img;
        if (!CutFace(image, cut_img, face_detector, central_line)) {
          continue;
        }
        imshow("cut_img", cut_img);
        std::string res_name(
            paths[i] + "_cut/" +
            entry.path().filename().replace_extension(".png").string());
        std::cout << res_name << '\n';
        cv::imwrite(res_name, cut_img);
      }
    }
  }
}
void AddDataFromUncut(std::string image_path, bool is_right,
                      cv::FileStorage& output,
                      cv::CascadeClassifier& face_detector,
                      cv::CascadeClassifier& eye_detector_l,
                      cv::CascadeClassifier& eye_detector_r,
                      int min_zone_size) {
  using namespace std;
  using namespace cv;
  auto dir_it(
      std::filesystem::begin(std::filesystem::directory_iterator(image_path)));
  auto dir_end(
      std::filesystem::end(std::filesystem::directory_iterator(image_path)));
  for (; dir_it != dir_end; ++dir_it) {
    Mat image = imread(dir_it->path().string(), IMREAD_GRAYSCALE);
    Sample row;
    row.is_right_ = is_right;
    row.name_ = dir_it->path().filename().string();
    vector<Rect> eyes_r, eyes_l;
    FindEyes(image, eyes_r, eye_detector_r);
    FindEyes(image, eyes_l, eye_detector_l);
    vector<Rect> eyes(eyes_r);
    eyes.insert(eyes.end(), eyes_l.begin(), eyes_l.end());

    SelectTwoEyes(eyes);
    if (eyes.size() == 2) {
      std::vector<double> central_line(SplitFace(image, eyes));
      if (!Allign(image, central_line)) {
        ++dir_it;
        continue;
      }
      imshow("image_center", image);
      Mat cut_img;
      if (!CutFace(image, cut_img, face_detector, central_line)) {
        ++dir_it;
        continue;
      }
      imshow("cut_img", cut_img);

      Mat LBP_img;
      OLBP<char>(cut_img, LBP_img);

      // imshow("OLBP image", LBP_img);
      RetrieveFeatures(LBP_img, row.features_, min_zone_size);
      output << row;
      // cv::waitKey();
    }
    std::cout << dir_it->path().filename().string() << '\n';
  }
}

void AddDataFromCut(std::string image_path, bool is_right,
                    cv::FileStorage& output, int min_zone_size) {
  using namespace std;
  using namespace cv;
  auto dir_it(
      std::filesystem::begin(std::filesystem::directory_iterator(image_path)));
  auto dir_end(
      std::filesystem::end(std::filesystem::directory_iterator(image_path)));
  for (; dir_it != dir_end; ++dir_it) {
    Mat image = imread(dir_it->path().string(), IMREAD_GRAYSCALE);
    // imshow("image", image);
    Sample row;
    row.is_right_ = is_right;
    row.name_ = dir_it->path().filename().string();

    Mat LBP_img;
    OLBP<char>(image, LBP_img);

    // imshow("LBP image", LBP_img);
    RetrieveFeatures(LBP_img, row.features_, min_zone_size);
    output << row;
    // cv::waitKey();
    std::cout << dir_it->path().filename().string() << '\n';
  }
}
void Sample::write(cv::FileStorage& output) const {
  output << "{:"
         << "name" << name_ << "features" << features_ << "is_right"
         << is_right_ << "}";
}
void Sample::read(const cv::FileNode& node) {
  name_ = node["name"];
  cv::FileNode features = node["features"];
  if (features.type() != cv::FileNode::SEQ) {
    std::cerr << "not a sequence! FAIL" << '\n';
    return;
  }
  cv::FileNodeIterator it = features.begin(), it_end = features.end();
  for (; it != it_end; ++it) {
    features_.push_back((double)(*it));
  }
  is_right_ = (int)node["is_right"];
}
static void write(cv::FileStorage& fs, const std::string&, const Sample& x) {
  x.write(fs);
}
static void read(const cv::FileNode& node, Sample& x,
                 const Sample& default_value) {
  if (node.empty())
    x = default_value;
  else
    x.read(node);
}
Stump::Stump() : feature_(-1), threshold_(0), weight_(0) {}
Stump::Stump(int feature, double threshold, double weight)
    : feature_(feature), threshold_(threshold), weight_(weight) {}
void Stump::SetWeight(double error) {
  if (error == 0) {
    error += std::numeric_limits<decltype(error)>::epsilon();
  }
  if (error >= 1) {
    error = 1 - std::numeric_limits<decltype(error)>::epsilon();
  }
  weight_ = 0.5 * log((1.0 - error) / error);
}
bool Stump::Classify(const Sample& sample) {
  if (sample.features_[feature_] < threshold_) {
    return true;
  }
  return false;
}
void Stump::write(cv::FileStorage& output) const {
  output << "{:"
         << "feature" << feature_ << "threshold" << threshold_ << "weight"
         << weight_ << "}";
}
void Stump::read(const cv::FileNode& node) {
  feature_ = (int)node["feature"];
  threshold_ = (double)node["threshold"];
  weight_ = (double)node["weight"];
}
static void write(cv::FileStorage& fs, const std::string&, const Stump& x) {
  x.write(fs);
}
static void read(const cv::FileNode& node, Stump& x,
                 const Stump& default_value) {
  if (node.empty())
    x = default_value;
  else
    x.read(node);
}
void TeachAdaBoost(std::string input, std::string output,
                   std::string assessment_path, double use_for_learning_part,
                   int n_stumps) {
  std::vector<DataRow> samples;
  ReadData(input, samples);
  NormalizeDataset(samples);
  std::vector<Sample> assessment_dataset;
  SeparateDataset(samples, assessment_dataset, use_for_learning_part);
  cv::FileStorage assessment_fs(assessment_path, cv::FileStorage::WRITE);
  assessment_fs << "samples" << assessment_dataset;
  assessment_fs.release();

  std::vector<Stump> stumps(n_stumps);
  for (int i = 0; i < stumps.size(); ++i) {
    cv::FileStorage fs(output, cv::FileStorage::WRITE);
    stumps[i] = GetBestStump(samples);
    UpdateDataset(samples, stumps[i]);
    if (i % 20 == 0) {
      RemoveNoize(samples, 0.01);
    }
    fs << "stumps" << stumps;
    fs.release();
  }
}

void RemoveNoize(std::vector<DataRow>& samples, double median_eps) {
  std::sort(samples.begin(), samples.end(),
            [&](const DataRow& a, const DataRow& b) {
              return a.weight_ < b.weight_;
            });
  double median = samples[samples.size() / 2].weight_;
  std::vector<DataRow> res;
  for (int i = 0; i < samples.size(); ++i) {
    if (samples[i].weight_ - median < median_eps) {
      res.push_back(samples[i]);
    }
  }
  samples = res;
  for (int i = 0; i < samples.size(); ++i) {
    samples[i].weight_ /= samples.size();
  }
}
void NormalizeDataset(std::vector<DataRow>& samples) {
  int count(0);
  for (int i = 0; i < samples.size(); ++i) {
    if (samples[i].data_.is_right_) {
      ++count;
    } else {
      int size(-1);
      if (count > samples.size() / 2) {
        size = samples.size() - count;
        int k = samples.size() - 1;
        for (int j = size; j < size * 2; ++j) {
          samples[j] = samples[k];
          --k;
        }
      } else {
        size = count;
      }
      samples.resize(size * 2);
      break;
    }
  }
  if (count == samples.size()) {
    std::cout << "no false samples\n";
  } else {
    std::random_device device;
    std::mt19937 generator(device());
    std::shuffle(samples.begin(), samples.end(), generator);
    for (int i = 0; i < samples.size(); ++i) {
      samples[i].weight_ = 1.0 / samples.size();
    }
  }
}
void ReadSamples(std::string data_path, std::vector<Sample>& output) {
  output.clear();
  cv::FileStorage fs(data_path, cv::FileStorage::READ);
  auto samples = fs["samples"];
  cv::FileNodeIterator it = samples.begin(), it_end = samples.end();
  for (; it != it_end; ++it) {
    Sample sample;
    (*it) >> sample;
    output.push_back(sample);
  }
  fs.release();
}
void ReadStumps(std::string data_path, std::vector<Stump>& output) {
  output.clear();
  cv::FileStorage fs(data_path, cv::FileStorage::READ);
  auto samples = fs["stumps"];
  cv::FileNodeIterator it = samples.begin(), it_end = samples.end();
  for (; it != it_end; ++it) {
    Stump stump;
    (*it) >> stump;
    output.push_back(stump);
  }
  fs.release();
}
void AssignByAdaBoost(std::vector<Sample>& data, std::vector<Stump>& stumps) {
  int correct_count(0);
  for (int i = 0; i < data.size(); ++i) {
    if (i == data.size() - 1) {
      int a = 3;
    }
    double yes(0);
    double no(0);
    for (int j = 0; j < stumps.size(); ++j) {
      if (stumps[j].Classify(data[i])) {
        yes += stumps[j].weight_;
      } else {
        no += stumps[j].weight_;
      }
    }
    // std::cout<<((yes>no)==data[i].is_right_)<<'\n';
    if ((yes > no) == data[i].is_right_) {
      correct_count += 1;
    }
  }
  std::cout << correct_count << ' ' << data.size() << '\n';
}
Stump GetBestStump(std::vector<DataRow>& samples) {
  std::pair<Stump, double> best_stump(Stump(), 2);
  for (int i = 0; i < samples[0].data_.features_.size(); ++i) {
    auto stump(FindThreshold(samples, i));
    // std::cout << stump.second << ' ' << best_stump.second << ' '
    //          << stump.first.feature_ << '\n';
    if (stump.first.weight_ > best_stump.first.weight_) {
      best_stump = stump;
    }
    // if (stump.second < best_stump.second) {
    //  best_stump = stump;
    //  if (best_stump.second == 0) {
    //    break;
    //  }
    //}
  }
  return best_stump.first;
}
void UpdateDataset(std::vector<DataRow>& samples, Stump stump) {
  double sum(0);
  for (int i = 0; i < samples.size(); ++i) {
    if (stump.Classify(samples[i].data_) == samples[i].data_.is_right_) {
      samples[i].weight_ = DecreasedWeight(samples[i].weight_, stump.weight_);
    } else {
      samples[i].weight_ = IncreasedWeight(samples[i].weight_, stump.weight_);
    }
    sum += samples[i].weight_;
  }
  for (int i = 0; i < samples.size(); ++i) {
    samples[i].weight_ /= sum;
  }
}
void PickData(std::vector<DataRow>& samples) {
  std::vector<double> probabilities;
  for (int i = 0; i < samples.size(); ++i) {
    probabilities.push_back(samples[i].weight_);
  }
  std::discrete_distribution distribution(probabilities.begin(),
                                          probabilities.end());
  std::random_device device;
  std::mt19937 generator(device());
  std::vector<DataRow> new_samples;
  for (int i = 0; i < samples.size(); ++i) {
    new_samples.push_back(samples[distribution(generator)]);
    new_samples.back().weight_ = 1.0 / samples.size();
  }
  samples = new_samples;
}
double IncreasedWeight(double weight, double stump_weight) {
  return weight * exp(stump_weight);
}
double DecreasedWeight(double weight, double stump_weight) {
  return weight * exp(-stump_weight);
}
void ReadData(std::string input, std::vector<DataRow>& output) {
  cv::FileStorage file(input, cv::FileStorage::READ);
  cv::FileNode samples = file["samples"];
  if (samples.type() != cv::FileNode::SEQ) {
    std::cerr << "not a sequence" << '\n';
    file.release();
    return;
  }
  cv::FileNodeIterator s_it = samples.begin(), s_it_end = samples.end();
  double weight = 1.0 / samples.size();
  for (; s_it != s_it_end; ++s_it) {
    output.push_back(DataRow());
    output.back().weight_ = weight;
    output.back().data_ = Sample();
    (*s_it) >> output.back().data_;
  }
  file.release();
}

void SeparateDataset(std::vector<DataRow>& samples,
                     std::vector<Sample>& assessment_dataset,
                     double use_for_learning_part) {
  assessment_dataset.clear();
  for (int i = samples.size() * use_for_learning_part; i < samples.size();
       ++i) {
    assessment_dataset.push_back(samples[i].data_);
  }
  samples.resize(samples.size() * use_for_learning_part);
  for (int i = 0; i < samples.size(); ++i) {
    samples[i].weight_ = 1.0 / samples.size();
  }
}
// stump, gini impurity. samples.size()!=0 is assumed
std::pair<Stump, double> FindThreshold(std::vector<DataRow>& samples,
                                       int column) {
  std::sort(samples.begin(), samples.end(),
            [&](const DataRow& a, const DataRow& b) {
              return a.data_.features_[column] < b.data_.features_[column];
            });
  // std::pair<Stump, double> best(Stump(column, 0, 0),
  //                              std::numeric_limits<double>::max());
  std::pair<Stump, double> best(Stump(column, 0, 0),
                                std::numeric_limits<double>::min());
  if (samples.size() == 1) {
    best.first.threshold_ = samples[0].data_.features_[0];
    best.second = 0;
  } else {
    for (int i = 1; i < samples.size(); ++i) {
      Stump stump(column,
                  (samples[i].data_.features_[column] +
                   samples[i - 1].data_.features_[column]) /
                      2,
                  0);
      Occurances(samples, stump);
      if (stump.weight_ > best.first.weight_) {
        best.first = stump;
      }
      // double impurity(GiniImpurity(Occurances(samples, stump)));
      // if (impurity < best.second) {
      //  best.second = impurity;
      //  best.first = stump;
      //}
    }
  }
  return best;
}
// yes for true, no for true, yes for false, no for false
std::tuple<int, int, int, int> Occurances(std::vector<DataRow>& samples,
                                          Stump& stump) {
  std::tuple<int, int, int, int> res(
      0, 0, 0, 0);  // yes for true, no for true, yes for false, no for false;
                    // <threshold is yes, >=threshold is no
  double error(0);
  for (int i = 0; i < samples.size(); ++i) {
    if (samples[i].data_.features_[stump.feature_] < stump.threshold_) {
      if (samples[i].data_.is_right_) {
        std::get<0>(res) += 1;
      } else {
        std::get<1>(res) += 1;
        error += samples[i].weight_;
      }
    } else {
      if (samples[i].data_.is_right_) {
        std::get<2>(res) += 1;
        error += samples[i].weight_;
      } else {
        std::get<3>(res) += 1;
      }
    }
  }
  stump.SetWeight(error);
  return res;
}
double GiniImpurity(double p1, double p2) { return 1.0 - p1 * p1 - p2 * p2; }
double GiniImpurity(std::tuple<int, int, int, int> occurances) {
  double o1 = static_cast<double>(std::get<0>(occurances));
  double o2 = static_cast<double>(std::get<1>(occurances));
  double o3 = static_cast<double>(std::get<2>(occurances));
  double o4 = static_cast<double>(std::get<3>(occurances));
  double p1, p2, p3, p4, w1, w2;
  if (o1 + o2 != 0) {
    p1 = o1 / (o1 + o2);
    p2 = 1.0 - p1;
  } else {
    p1 = 0;
    p2 = 0;
  }
  if (o3 + o4 != 0) {
    p3 = o3 / (o3 + o4);
    p4 = 1.0 - p3;
  } else {
    p3 = 0;
    p4 = 0;
  }
  if (o1 + o2 + o3 + o4 != 0) {
    w1 = (o1 + o2) / (o1 + o2 + o3 + o4);
    w2 = 1.0 - w1;
  } else {
    w1 = 0;
    w2 = 0;
  }
  return GiniImpurity(p1, p2) * w1 + GiniImpurity(p3, p4) * w2;
}

}  // namespace face_assessment

using namespace face_assessment;
using namespace std;
using namespace cv;
int main() {

  std::string right("../../resources/dataset/cut/right_4_cut");
  std::string wrong("../../resources/dataset/cut/wrong_3_cut");
  std::string data("../../resources/data_cut_2.yaml");
  std::string adaboost_path("../../resources/adaboost_cut_2.yaml");
  std::string unused_path("../../resources/unused_cut_2.yaml");
  std::string all_path("../../resources/dataset/cut/all");
  std::string distorted_right_path(
      "../../resources/dataset/cut/distorted_right");
  std::string distorted_wrong_path(
      "../../resources/dataset/cut/distorted_wrong");
  std::string distorted_data_path("../../resources/data_distorted.yaml");
  int min_zone(2);

   ObtainData(right, wrong, data, min_zone);
   cout << "obtained" << '\n';

   TeachAdaBoost(data, adaboost_path, unused_path, 0.75, 100);
   std::cout << "tought\n";

  std::vector<Sample> unused;
  ReadSamples(unused_path, unused);
  std::vector<Stump> stumps;
  ReadStumps(adaboost_path, stumps);
  std::cout << "read\n";
  AssignByAdaBoost(unused, stumps);


  return 0;
}