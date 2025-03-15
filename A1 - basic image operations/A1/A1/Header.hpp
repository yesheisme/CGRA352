#define filter_hpp
#include <opencv2/core/core.hpp>

cv::Mat convertBGR2HSV(const cv::Mat& m);
cv::Mat convertHSV2BGR(const cv::Mat& m);
std::vector<cv::Mat> getChannels(const cv::Mat& m);
cv::Mat mask(const cv::Mat& m);
void applyLaplacianFilter(const cv::Mat& src, cv::Mat& dst);
void applySobelXFilter(const cv::Mat& src, cv::Mat& dst);
void applySobelYFilter(const cv::Mat& src, cv::Mat& dst);