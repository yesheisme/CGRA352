#include "Header.hpp"
#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat convertBGR2HSV(const Mat& m) {
	Mat clone = m.clone();
	Mat hsv;
	cvtColor(clone, hsv, CV_BGR2HSV);
	return hsv;
}

Mat convertHSV2BGR(const Mat& m) {
	Mat clone = m.clone();
	Mat bgr;
	cvtColor(clone, bgr, CV_HSV2BGR);
	return bgr;
}

vector<Mat> getChannels(const Mat& m) {
	Mat clone = m.clone();
	std::vector<cv::Mat> channels;
	split(clone, channels);
	return channels;
}

Mat mask(const Mat& m) {
	Mat clone = m.clone();
	Mat maskImg;
	unsigned char* point = clone.ptr(80, 80);
	int b1 = point[0];
	int g1 = point[1];
	int r1 = point[2];

	//go through every pixel in image.
	for (int i = 0; i < clone.rows; i++) {
		for (int j = 0; j < clone.cols; j++) {
			Vec3b pt = clone.at<cv::Vec3b>(i, j);
			int b2 = pt[0];
			int g2 = pt[1];
			int r2 = pt[2];
			int thresh = sqrt(((r1 - r2) * (r1 - r2)) + ((g1 - g2) * (g1 - g2)) + ((b1 - b2) * (b1 - b2)));

			if (thresh < 100) {
				//set to white
				clone.at<cv::Vec3b>(i, j)[0] = 255;
				clone.at<cv::Vec3b>(i, j)[1] = 255;
				clone.at<cv::Vec3b>(i, j)[2] = 255;
			}
			else {
				//set to black
				clone.at<cv::Vec3b>(i, j)[0] = 0;
				clone.at<cv::Vec3b>(i, j)[1] = 0;
				clone.at<cv::Vec3b>(i, j)[2] = 0;
			}
		}
	}
	return clone;
}

// ------------------------ Laplacian ------------------------

void applyLaplacianFilter(const Mat& m, Mat& dst) {
	CV_Assert(m.depth() == CV_8U);  // Ensure the input image is of type CV_8U

	const int kSize = 3;  // Kernel size
	dst = Mat::zeros(m.size(), m.type());

	for (int y = 1; y < m.rows - 1; ++y) {
		for (int x = 1; x < m.cols - 1; ++x) {
			float laplacian =
				m.at<uchar>(y - 1, x) +
				m.at<uchar>(y + 1, x) +
				m.at<uchar>(y, x - 1) +
				m.at<uchar>(y, x + 1) -
				4 * m.at<uchar>(y, x);

			// Truncate values to [0, 255] range
			uchar val = static_cast<uchar>(max(0.0f, min(255.0f, laplacian + 128.0f))); // Adding 128 for visualization
			dst.at<uchar>(y, x) = val;
		}
	}
}

// ------------------------ Sobel X ------------------------

void applySobelXFilter(const Mat& m, Mat& dst) {
	CV_Assert(m.depth() == CV_8U);  // Ensure the input image is of type CV_8U

	dst = Mat::zeros(m.size(), m.type());

	for (int y = 1; y < m.rows - 1; ++y) {
		for (int x = 1; x < m.cols - 1; ++x) {
			float gx =
				-m.at<uchar>(y - 1, x - 1) + m.at<uchar>(y - 1, x + 1) +
				-2 * m.at<uchar>(y, x - 1) + 2 * m.at<uchar>(y, x + 1) +
				-m.at<uchar>(y + 1, x - 1) + m.at<uchar>(y + 1, x + 1);

			uchar val = static_cast<uchar>(std::max(0.0f, std::min(255.0f, gx + 128.0f))); // Adding 128 for visualization
			dst.at<uchar>(y, x) = val;
		}
	}
}

// ------------------------ Sobel Y ------------------------

void applySobelYFilter(const Mat& m, Mat& dst) {
	CV_Assert(m.depth() == CV_8U);  // Ensure the input image is of type CV_8U

	dst = Mat::zeros(m.size(), m.type());

	for (int y = 1; y < m.rows - 1; ++y) {
		for (int x = 1; x < m.cols - 1; ++x) {
			float gy =
				-m.at<uchar>(y - 1, x - 1) - 2 * m.at<uchar>(y - 1, x) - m.at<uchar>(y - 1, x + 1) +
				m.at<uchar>(y + 1, x - 1) + 2 * m.at<uchar>(y + 1, x) + m.at<uchar>(y + 1, x + 1);

			uchar val = static_cast<uchar>(std::max(0.0f, std::min(255.0f, gy + 128.0f))); // Adding 128 for visualization
			dst.at<uchar>(y, x) = val;
		}
	}
}

