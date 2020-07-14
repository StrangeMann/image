#ifndef EDMA_MEASURE
#define EDMA_MEASURE

#include <fstream>
#include <iostream>
#include <vector>

namespace edma_measure {
struct CSVRow {
  CSVRow() = default;
  std::istream& readNextRow(std::istream& str);

  std::vector<std::string> row_;
};

class EdmaData {
 public:
  EdmaData(std::ifstream& file);

  bool Fill(const std::vector<std::string>& row);
  std::vector<double> Metrics();
  std::string Name();
 private:
  void CountRanges();
  double Range(std::pair<double, double> p1,
               std::pair<double, double> p2) const;

  void SetDiffToRatio();
  void SetDiffToSubstr();

  double MinMaxMetric();
  double AvgMetric(int sampling_frame,int repeats,double alpha);

  std::string name;
  std::vector<std::pair<double, double>> coordinates_;
  std::vector<int> right_points;
  std::vector<int> left_points;
  std::vector<std::vector<double>> r_ranges;
  std::vector<std::vector<double>> l_ranges;
  std::vector<double> range_diff;
};

}  // namespace edma_measure

#endif
