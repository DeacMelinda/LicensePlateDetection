#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <string>
#include "opencv2/core.hpp"

Mat RgbToGray(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
	// Varianta ineficienta (lenta)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b p = src.at<Vec3b>(i, j);
			dst.at<uchar>(i, j) = (p[0] + p[1] + p[2]) / 3;
		}
	}

	return dst;
}

std::vector<int> computePeaks(Mat hist)
{
	std::vector<int> peaks;
	int size = hist.cols;
	for (int i = 1; i < size; i++) {
		if (hist.at<float>(i) > hist.at<float>(i - 1) && hist.at<float>(i) > hist.at<float>(i + 1)) {
			peaks.push_back(i);
		}
	}
	return peaks;
}

bool compareContourAreas(std::vector<cv::Point>& contour1, std::vector<cv::Point>& contour2) {
	const double i = fabs(contourArea(cv::Mat(contour1)));
	const double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

// Random generator for cv::Scalar
cv::RNG rng(12345);

void drawLicensePlate(Mat& frame, std::vector<std::vector<Point>>& candidates)
{
	const int width = frame.cols;
	const int height = frame.rows;
	const float ratio_width = width / (float)512;
	const float ratio_height = height / (float)512;

	// Convert to rectangle and also filter out the non-rectangle-shape.
	std::vector<cv::Rect> rectangles;
	for (std::vector<cv::Point> currentCandidate : candidates) {
		Rect temp = boundingRect(currentCandidate);
		float difference = temp.area() - contourArea(currentCandidate);
		if (difference < 2000) {
			rectangles.push_back(temp);
		}
	}

	std::vector<cv::Rect> updatedRectangles;

	for (const auto& temp : rectangles) {
		const float aspect_ratio = (float)temp.width / (float)(temp.height);
		if (aspect_ratio >= 1 && aspect_ratio <= 6) {
			updatedRectangles.push_back(temp);
		}
	}

	rectangles = updatedRectangles;

	for (const Rect& rectangle : rectangles) {
		// Generate random color values for the rectangle
		int red = rng.uniform(0, 256);
		int green = rng.uniform(0, 256);
		int blue = rng.uniform(0, 256);

		// Draw the rectangle on the frame
		cv::rectangle(frame, cv::Point(rectangle.x * ratio_width, rectangle.y * ratio_height),
			cv::Point((rectangle.x + rectangle.width) * ratio_width, (rectangle.y + rectangle.height) * ratio_height),
			cv::Scalar(blue, green, red), 3, cv::LINE_8, 0);
	}

}

int* computeHistogram(Mat src)
{
	int h[256] = { 0 };
	int height = src.rows;
	int width = src.cols;
	// Varianta ineficienta (lenta)
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int val = src.data[i * src.step[0] + j];

			h[val] += 1;
		}
	}
	return h;
}

double computeG2D(int x, int y, float x0, float y0, float sigma)
{
	return (1 / (2 * PI * sigma * sigma)) * (exp(-(((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (2 * sigma * sigma))));
}


Mat gaussianFilter1D(Mat src)
{
	float sigma = 0.8;
	int height = src.rows;
	int width = src.cols;
	Mat inter = Mat(height, width, CV_8UC1);
	Mat dst = Mat(height, width, CV_8UC1);

	int w = (int)(6 * sigma) % 2 == 0 ? (int)(6 * sigma) + 1 : (int)(6 * sigma);

	int x0 = w / 2;
	int y0 = w / 2;

	Mat G2D = Mat(w, w, CV_32F);

	for (int k = 0; k < w; k++)
	{
		for (int l = 0; l < w; l++)
		{
			G2D.at<float>(k, l) = computeG2D(k, l, x0, y0, sigma);
		}
	}

	float sumG2D_row = 0.0;
	float sumG2D_col = 0.0;
	for (int k = 0; k < w; k++)
	{
		sumG2D_row += G2D.at<float>(k, w / 2);
	}

	for (int k = 0; k < w; k++)
	{
		sumG2D_col += G2D.at<float>(w / 2, k);
	}

	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < w; k++)
			{
				sum += G2D.at<float>(k, w / 2) * (float)src.at<uchar>(i - w / 2 + k, j);
			}
			inter.at<uchar>(i, j) = (uchar)sum / sumG2D_row;
		}
	}
	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < w; k++)
			{
				sum += G2D.at<float>(w / 2, k) * (float)inter.at<uchar>(i, j - w / 2 + k);
			}
			dst.at<uchar>(i, j) = (uchar)sum / sumG2D_col;
		}
	}

	return dst;
}

Mat cannyEdgeDetection(Mat src)
{
	float sigma = 0.5;

	//Gaussian Filtering 1D
	int height = src.rows;
	int width = src.cols;
	Mat inter = Mat(height, width, CV_8UC1);
	Mat filtered = inter.clone();

	int w = (int)(6 * sigma) % 2 == 0 ? (int)(6 * sigma) + 1 : (int)(6 * sigma);

	int x0 = w / 2;
	int y0 = w / 2;

	Mat G2D = Mat(w, w, CV_32F);

	for (int k = 0; k < w; k++)
	{
		for (int l = 0; l < w; l++)
		{
			G2D.at<float>(k, l) = computeG2D(k, l, x0, y0, sigma);
		}
	}

	float sumG2D_row = 0.0;
	float sumG2D_col = 0.0;
	for (int k = 0; k < w; k++)
	{
		sumG2D_row += G2D.at<float>(k, w / 2);
	}

	for (int k = 0; k < w; k++)
	{
		sumG2D_col += G2D.at<float>(w / 2, k);
	}

	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0.0;

			for (int k = 0; k < w; k++)
			{
				sum += G2D.at<float>(k, w / 2) * (float)src.at<uchar>(i - w / 2 + k, j);

			}
			inter.at<uchar>(i, j) = (uchar)sum / sumG2D_row;
		}
	}

	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0.0;

			for (int k = 0; k < w; k++)
			{
				sum += G2D.at<float>(w / 2, k) * (float)inter.at<uchar>(i, j - w / 2 + k);

			}
			filtered.at<uchar>(i, j) = (uchar)sum / sumG2D_col;
		}
	}

	//Computation of the gradient magnitude and orientation

	int Sx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};

	int Sy[3][3] = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};

	Mat Gx = Mat(height, width, CV_32FC1);
	Mat Gy = Mat(height, width, CV_32FC1);
	Mat G = Mat(height, width, CV_32FC1);
	Mat Gdisplay = Mat(height, width, CV_8UC1);
	Mat phi = Mat(height, width, CV_32FC1);

	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			int sumX = 0;
			int sumY = 0;

			for (int k = 0; k < w; k++)
			{
				for (int l = 0; l < w; l++)
				{
					sumX += Sx[k][l] * filtered.at<uchar>(i - w / 2 + k, j - w / 2 + l);
					sumY += Sy[k][l] * filtered.at<uchar>(i - w / 2 + k, j - w / 2 + l);
				}
			}

			Gx.at<float>(i, j) = sumX;
			Gy.at<float>(i, j) = sumY;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			G.at<float>(i, j) = (sqrt(Gx.at<float>(i, j) * Gx.at<float>(i, j) + Gy.at<float>(i, j) * Gy.at<float>(i, j))) / (4 * sqrt(2));
			Gdisplay.at<uchar>(i, j) = G.at<float>(i, j);
			phi.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));
		}
	}

	// Non-maxima suppression
	/*int Ml[3][3] = {
		{3, 2, 1},
		{0, 1, 0},
		{1, 2, 3}
	};*/

	Mat GS = Mat(height, width, CV_8UC1);

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			float phi_val = phi.at<float>(i, j);
			if ((phi_val >= -(CV_PI / 8) && phi_val <= (CV_PI / 8)) || (phi_val >= (7 * CV_PI / 8)) || (phi_val <= (-7 * CV_PI / 8))) //ZONA 0
			{
				if (G.at<float>(i, j) >= G.at<float>(i, j - 1) && G.at<float>(i, j) >= G.at<float>(i, j + 1))
				{
					GS.at<uchar>(i, j) = G.at<float>(i, j);
				}
				else
				{
					GS.at<uchar>(i, j) = 0;
				}
			}
			else if ((phi_val >= (CV_PI / 8) && phi_val <= (3 * CV_PI / 8)) || (phi_val >= (-7 * CV_PI / 8) && phi_val <= (-5 * CV_PI / 8))) //ZONA 1
			{
				if (G.at<float>(i, j) >= G.at<float>(i - 1, j + 1) && G.at<float>(i, j) >= G.at<float>(i + 1, j - 1))
				{
					GS.at<uchar>(i, j) = G.at<float>(i, j);
				}
				else
				{
					GS.at<uchar>(i, j) = 0;
				}
			}
			else if ((phi_val >= (3 * CV_PI / 8) && phi_val <= (5 * CV_PI / 8)) || (phi_val >= (-5 * CV_PI / 8) && phi_val <= (-3 * CV_PI / 8))) //ZONA 2
			{
				if (G.at<float>(i, j) >= G.at<float>(i - 1, j) && G.at<float>(i, j) >= G.at<float>(i + 1, j))
				{
					GS.at<uchar>(i, j) = G.at<float>(i, j);
				}
				else
				{
					GS.at<uchar>(i, j) = 0;
				}
			}
			else if ((phi_val >= (5 * CV_PI / 8) && phi_val <= (7 * CV_PI / 8)) || (phi_val >= (-3 * CV_PI / 8) && phi_val <= (-CV_PI / 8))) //ZONA 3
			{
				if (G.at<float>(i, j) >= G.at<float>(i - 1, j - 1) && G.at<float>(i, j) >= G.at<float>(i + 1, j + 1))
				{
					GS.at<uchar>(i, j) = G.at<float>(i, j);
				}
				else
				{
					GS.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	// Adaptive thresholding and edge linking

	// calculul histogramei
	int h[256] = { 0 };

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			h[GS.at<uchar>(i, j)] += 1;
		}
	}
	//showHistogram("Histograma", h, 256, 256);

	float p = 0.1;

	float NoEdgePixels = p * ((height - 2) * (width - 2) - h[0]);

	//printf("NoEdgePixels: %f\n", NoEdgePixels);

	int Thigh = 0;

	// calculul lui Thigh
	int curr = 0;
	for (int i = 255; i >= 0; i--)
	{
		curr += h[i];
		if (curr >= NoEdgePixels)
		{
			Thigh = i;
			break;
		}
	}

	float k = 0.4;
	int Tlow = k * Thigh;

	//printf("Thigh: %d\nTlow: %d\n", Thigh, Tlow);

	// aplicarea binarizarii cu histereza
	//Mat dst = src.clone();
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (GS.at<uchar>(i, j) <= Tlow)
			{
				GS.at<uchar>(i, j) = 0; //no edge
			}
			else if (GS.at<uchar>(i, j) >= Thigh)
			{
				GS.at<uchar>(i, j) = 255; //strong edge
			}
			else
			{
				GS.at<uchar>(i, j) = 127; //weak edge
			}
		}
	}

	//imshow("Weak and Strong Edges", GS);

	// Edge linking
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	int label = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (GS.at<uchar>(i, j) == 255) {
				std::queue<Point> Q;
				Q.push(Point(j, i));
				while (!Q.empty()) {
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++) {
						int row = q.y + di[k];
						int col = q.x + dj[k];
						if (row >= 0 && row < height && col >= 0 && col < width)
							if (GS.at<uchar>(row, col) == 127) {
								GS.at<uchar>(row, col) = 255;
								Q.push(Point(col, row));
							}
					}
				}
			}
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (GS.at<uchar>(i, j) == 127)
			{
				GS.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("gauss filter", filtered);
	imshow("Gradient", Gdisplay);
	imshow("edge", GS);

	return GS;

}

void plateDetect()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);

		imshow("initital image", src);

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		int height = src.rows;
		int width = src.cols;

		Mat gray = RgbToGray(src);

		imshow("grayscale image", gray);

		Mat resized;

		resize(gray, resized, Size(512, 512));

		Mat canny2 = cannyEdgeDetection(resized);

		Mat canny = Mat(canny2.rows - 2, canny2.cols - 2, CV_8UC1);

		for (int i = 1; i < canny2.rows-1; i++)
		{
			for (int j = 1; j < canny2.cols-1; j++)
			{
				canny.at<uchar>(i-1, j-1) = canny2.at<uchar>(i,j);
			}
		}
		imshow("After Canny", canny);

		std::vector<std::vector<cv::Point>> contours;
		findContours(canny, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);

		std::vector<std::vector<cv::Point>> top_contours;

		//Store the smallest 5 contours in descending order
		if (contours.size() >= 5) {
			top_contours.assign(contours.begin() + (contours.size() - 5), contours.end());
		}
		else {
			top_contours = contours; // handle the case where contours has less than 5 elements
		}

		drawLicensePlate(src, top_contours);

		imshow("Result", src);

		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 100 - License Plate Detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 100:
			plateDetect();
			break;
		}
	} while (op != 0);
	return 0;
}