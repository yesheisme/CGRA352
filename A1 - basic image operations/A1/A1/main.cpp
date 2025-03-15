#include <opencv2/opencv.hpp>
#include "Header.hpp"
using namespace cv; /* can use cv:: instead of the namespace to tell you all functions */
using namespace std; /* standard namespace */

int main(int argc, char* arg[]) {
	// Standard image
	Mat img = imread("Flower.png", 1);

	// Double checking that there is actually a file to read
	if (img.empty()) {
		std::cout << "Could not read the image" << std::endl;
		return -1;
	}

	// Image in hsv
	Mat imgHSV = convertBGR2HSV(img);

	// Split into channels
	Mat B = getChannels(img)[0]; // Blue channel
	Mat G = getChannels(img)[1]; // Green channel
	Mat R = getChannels(img)[2]; // Red channel
	Mat H = getChannels(imgHSV)[0]; // Hugh channel
	Mat S = getChannels(imgHSV)[1]; // Saturation channel
	Mat V = getChannels(imgHSV)[2]; // Brightness channel

	// Creating vector for hue 
	vector<Mat> channel = getChannels(imgHSV);
	Mat hue;

	// Manipulate Hue, merge and convert
	channel[0] = H * 0.0;
	merge(channel, hue);
	Mat H_0 = convertHSV2BGR(hue);

	channel[0] = H * 0.2;
	merge(channel, hue);
	Mat H_1 = convertHSV2BGR(hue);

	channel[0] = H * 0.4;
	merge(channel, hue);
	Mat H_2 = convertHSV2BGR(hue);

	channel[0] = H * 0.6;
	merge(channel, hue);
	Mat H_3 = convertHSV2BGR(hue);

	channel[0] = H * 0.8;
	merge(channel, hue);
	Mat H_4 = convertHSV2BGR(hue);

	// Creating vector for Saturation 
	vector<Mat> channel2 = getChannels(imgHSV);
	Mat Saturation;

	// Manipulate saturation, merge and convert
	channel2[1] = S * 0.0;
	merge(channel2, Saturation);
	Mat S_0 = convertHSV2BGR(Saturation);

	channel2[1] = S * 0.2;
	merge(channel2, Saturation);
	Mat S_1 = convertHSV2BGR(Saturation);

	channel2[1] = S * 0.4;
	merge(channel2, Saturation);
	Mat S_2 = convertHSV2BGR(Saturation);

	channel2[1] = S * 0.6;
	merge(channel2, Saturation);
	Mat S_3 = convertHSV2BGR(Saturation);

	channel2[1] = S * 0.8;
	merge(channel2, Saturation);
	Mat S_4 = convertHSV2BGR(Saturation);

	// Creating vector for brightness 
	vector<Mat> channel3 = getChannels(imgHSV);
	Mat brightness;

	// Manipulate brightness, merge and convert
	channel3[2] = V * 0.0;
	merge(channel3, brightness);
	Mat V_0 = convertHSV2BGR(brightness);

	channel3[2] = V * 0.2;
	merge(channel3, brightness);
	Mat V_1 = convertHSV2BGR(brightness);

	channel3[2] = V * 0.4;
	merge(channel3, brightness);
	Mat V_2 = convertHSV2BGR(brightness);

	channel3[2] = V * 0.6;
	merge(channel3, brightness);
	Mat V_3 = convertHSV2BGR(brightness);

	channel3[2] = V * 0.8;
	merge(channel3, brightness);
	Mat V_4 = convertHSV2BGR(brightness);

	// ------------------------ Core: Part One ------------------------

	// Create canvas for display and show our image inside it
	string coreP1 = "Core: Part One";
	namedWindow(coreP1, WINDOW_AUTOSIZE);

	// Row 1
	hconcat(B, G, B);
	hconcat(B, R, B);

	// Row2
	hconcat(H, S, H);
	hconcat(H, V, H);

	// Combine and display
	vconcat(B, H, B);

	imshow(coreP1, B);
	imwrite("../../A1 output/core1.png", B);

	// ------------------------ Core: Part Two ------------------------

	string coreP2 = "Core: Part Two";
	namedWindow(coreP2, WINDOW_AUTOSIZE);

	// Row 1
	hconcat(H_0, H_1, H_0);
	hconcat(H_0, H_2, H_0);
	hconcat(H_0, H_3, H_0);
	hconcat(H_0, H_4, H_0);
	// Row 2
	hconcat(S_0, S_1, S_0);
	hconcat(S_0, S_2, S_0);
	hconcat(S_0, S_3, S_0);
	hconcat(S_0, S_4, S_0);
	// Combine rows 1 & 2
	vconcat(H_0, S_0, H_0);
	// Row 3
	hconcat(V_0, V_1, V_0);
	hconcat(V_0, V_2, V_0);
	hconcat(V_0, V_3, V_0);
	hconcat(V_0, V_4, V_0);
	// Combine
	vconcat(H_0, V_0, H_0);

	imshow(coreP2, H_0);
	imwrite("../../A1 output/core2.png", H_0);

	// ------------------------ Core: Part Three ------------------------

	string coreP3 = "Core 3 Image Display";
	namedWindow(coreP3, WINDOW_AUTOSIZE);

	Mat maskImg = mask(img);

	imshow(coreP3, maskImg);
	imwrite("../../A1 output/core3.png", maskImg);

	// ------------------------ Completion: Laplacian ------------------------

	Mat img2 = imread("Flower.png", IMREAD_GRAYSCALE);
	if (img2.empty()) {
		std::cout << "Could not read the image" << std::endl;
		return -1;
	}

	cv::Mat laplacian;
	applyLaplacianFilter(img2, laplacian);

	cv::imshow("Laplacian Filtered Image", laplacian);
	imwrite("../../A1 output/laplacian.png", laplacian);

	// ------------------------ Completion: Sobel X ------------------------

	cv::Mat sobelX;
	applySobelXFilter(img2, sobelX);

	cv::imshow("Sobel X Filtered Image", sobelX);
	imwrite("../../A1 output/sobelX.png", sobelX);

	// ------------------------ Completion: Sobel Y ------------------------

	cv::Mat sobelY;
	applySobelYFilter(img2, sobelY);

	cv::imshow("Sobel Y Filtered Image", sobelY);
	imwrite("../../A1 output/sobelY.png", sobelY);

	// ------------------------ END ------------------------

	waitKey(0);
	return 0;
}