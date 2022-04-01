#include <iostream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat> &hists)
{
    // Min/Max computation
    double hmax[3] = {0, 0, 0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = {"blue", "green", "red"};
    cv::Scalar colors[3] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                            cv::Scalar(0, 0, 255)};

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                1, 8, 0);
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}

int main(int argc, char **argv)
{
    cout << "Lab2 is running" << endl;

    // Loads the image
    Mat inputImage = imread("./data/barbecue.png");

    if (!inputImage.data)
        return 1;

    vector<Mat> histograms;

    // Separate the channels in a multichannels array into multiple single channel arrays
    split(inputImage, histograms);

    int histSize = 256;

    // the upper boundary is not included
    float range[] = {0, 256};
    const float *histRange = {range};

    Mat blueHistogram, greenHistogram, redHistogram;
    calcHist(&histograms[0], 1, 0, Mat(), blueHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&histograms[1], 1, 0, Mat(), greenHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&histograms[2], 1, 0, Mat(), redHistogram, 1, &histSize, &histRange, true, false);

    vector<Mat> outputHistogram{blueHistogram, greenHistogram, redHistogram};
    showHistogram(outputHistogram);
        waitKey(0);

    return 0;
}