/*
    Author: Vittorio Esposito
*/

#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

int kernelSizeGaussian = 1;
int sigmaGaussian = 0;

int sigmaSpaceBilateral = 1;
int sigmaRangeBilateral = 1;

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

void medianFilterCallback(int pos, void *inputImage)
{

    // Correct the kernel size, if needed: it must be odd
    pos = pos % 2 == 0 ? pos + 1 : pos;

    if (pos < 1)
        return;

    Mat outputImage;

    // Blurs an image using the median filter.
    medianBlur(*(Mat *)inputImage, outputImage, pos);
    imshow("Median filtering", outputImage);
}

void gaussianFilterSigmaCallback(int pos, void *inputImage)
{
    sigmaGaussian = pos;
    Mat outputImage;

    // if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros,
    // they are computed from ksize.width and ksize.height, respectively

    GaussianBlur(*(Mat *)inputImage, outputImage, Size(kernelSizeGaussian, kernelSizeGaussian), (double)sigmaGaussian, (double)sigmaGaussian);
    imshow("Gaussian filtering", outputImage);
}

void gaussianFilterKernelCallback(int pos, void *inputImage)
{
    // Correct the kernel size, if needed: it must be odd
    pos = pos % 2 == 0 ? pos + 1 : pos;

    if (pos < 1)
        return;

    kernelSizeGaussian = pos;

    Mat outputImage;

    GaussianBlur(*(Mat *)inputImage, outputImage, Size(kernelSizeGaussian, kernelSizeGaussian), (double)sigmaGaussian, (double)sigmaGaussian);
    imshow("Gaussian filtering", outputImage);
}

void bilateralFilterSigmaRangeCallaback(int pos, void *inputImage)
{
    sigmaRangeBilateral = pos;

    Mat outputImage;

    // Diameter of each pixel neighborhood that is used during filtering.
    int diameter = 6 * sigmaSpaceBilateral;

    bilateralFilter(*(Mat *)inputImage, outputImage, diameter, (double)sigmaRangeBilateral, (double)sigmaSpaceBilateral);
    imshow("Bilateral filtering", outputImage);
}

void bilateralFilterSigmaSpaceCallaback(int pos, void *inputImage)
{
    sigmaSpaceBilateral = pos;

    Mat outputImage;

    // Diameter of each pixel neighborhood that is used during filtering.
    int diameter = 6 * sigmaSpaceBilateral;

    bilateralFilter(*(Mat *)inputImage, outputImage, diameter, (double)sigmaRangeBilateral, (double)sigmaSpaceBilateral);
    imshow("Bilateral filtering", outputImage);
}

Mat equalization()
{
    // ***** PART 1 *****

    // 1. Loads an image (e.g., one of the provided images like “barbecue.jpg” or “countryside.jpg”)
    Mat inputImage = imread("./data/barbecue.png");

    const char *inputWnd = "Input image";
    namedWindow(inputWnd);
    imshow(inputWnd, inputImage);
    cout << "Showing the input image. Close the image(s) to continue..." << endl;
    waitKey(0);

    /*
    2. Prints the histograms of the image. You must compute 3 histograms, one for each channel (i.e., R, G and B)
    with 256 bins and [0, 255] as range. Notice that you need to use the calcHist() function separately on the 3 channels.
    You can use the provided function (in the “show_histogram_function.cpp” file) to visualize the data.
    */

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
    cout << "Showing the histograms of the image. Close the image(s) to continue..." << endl;
    waitKey(0);

    // 3. Equalizes separately the R, G and B channels by using cv::equalizeHist().
    equalizeHist(histograms[0], blueHistogram);
    equalizeHist(histograms[1], greenHistogram);
    equalizeHist(histograms[2], redHistogram);

    // 4. Shows the equalized image and the histogram of its channels.

    vector<Mat> equalizedHistograms{blueHistogram, greenHistogram, redHistogram};
    Mat equalizedImage;

    // Merge equalizedHistograms to make a single multi-channel array equalizedImage
    merge(equalizedHistograms, equalizedImage);

    const char *equalizedWnd = "Equalized Image";
    namedWindow(equalizedWnd);
    imshow(equalizedWnd, equalizedImage);
    cout << "Showing the equalized image. Close the image(s) to continue..." << endl;
    waitKey(0);

    calcHist(&equalizedHistograms[0], 1, 0, Mat(), blueHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&equalizedHistograms[1], 1, 0, Mat(), greenHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&equalizedHistograms[2], 1, 0, Mat(), redHistogram, 1, &histSize, &histRange, true, false);

    outputHistogram = {blueHistogram, greenHistogram, redHistogram};
    showHistogram(outputHistogram);
    cout << "Showing the histograms of the equalized image. Close the image(s) to continue..." << endl;
    waitKey(0);

    /*
    5. Notice the artifacts produced by this approach.
    To obtain a better equalization than the one of point 4, convert the image to
    a different color space, e.g. Lab (use cv::cvtColor() with COLOR_BGR2Lab as color space conversion code),
    and equalize only the luminance (L) channel.
    */

    // Convert the image to Lab color space
    Mat convertedImage;
    cvtColor(inputImage, convertedImage, COLOR_BGR2Lab);

    // Split as before
    vector<Mat> histogramsLab;
    split(convertedImage, histogramsLab);

    // equalize only the luminance (L) channel
    Mat luminanceHistogram;
    equalizeHist(histogramsLab[0], histogramsLab[0]);

    merge(histogramsLab, convertedImage);
    cvtColor(convertedImage, convertedImage, COLOR_Lab2BGR);

    const char *equalizedLumWnd = "Equalized image (luminance only)";
    namedWindow(equalizedLumWnd);
    imshow(equalizedLumWnd, convertedImage);
    cout << "Showing the equalized image (luminance only). Close the image(s) to continue..." << endl;
    waitKey(0);

    split(convertedImage, equalizedHistograms);
    calcHist(&equalizedHistograms[0], 1, 0, Mat(), blueHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&equalizedHistograms[1], 1, 0, Mat(), greenHistogram, 1, &histSize, &histRange, true, false);
    calcHist(&equalizedHistograms[2], 1, 0, Mat(), redHistogram, 1, &histSize, &histRange, true, false);
    outputHistogram = {blueHistogram, greenHistogram, redHistogram};
    showHistogram(outputHistogram);
    cout << "Showing the histograms of the equalized image (luminance only). Close the image(s) to continue..." << endl;
    waitKey(0);

    return convertedImage;
}

void filtering(Mat &convertedImage)
{
    // ***** PART 2 *****

    /*
    Generate a denoised version of the image. You should try different filters and parameter values.
        • Write a program that performs the filtering and shows the result.
        • Table 1 specifies the requested filters to test and the parameters to be set for each filter.
        • You can simply pass the filter parameters from the command line

    */

    int blur = 0;

    const char *medianFilterWnd = "Median filtering";
    namedWindow(medianFilterWnd);
    imshow(medianFilterWnd, convertedImage);
    createTrackbar("Kernel", medianFilterWnd, &blur, 45, medianFilterCallback, (void *)&convertedImage);
    cout << "Showing the median filtering output." << endl;

    waitKey(0);

    const char *gaussianFilterWnd = "Gaussian filtering";
    namedWindow(gaussianFilterWnd);
    imshow(gaussianFilterWnd, convertedImage);
    // This time we need two trackbars: one for the kernel and one for sigma
    createTrackbar("Sigma", gaussianFilterWnd, &blur, 90, gaussianFilterSigmaCallback, (void *)&convertedImage);
    createTrackbar("Kernel", gaussianFilterWnd, &blur, 40, gaussianFilterKernelCallback, (void *)&convertedImage);
    cout << "Showing the gaussian filtering output." << endl;

    waitKey(0);

    // Resize of a factor of 2 to reduce computation times
    cv::resize(convertedImage, convertedImage, cv::Size(convertedImage.cols / 2, convertedImage.rows / 2));

    const char *bilateralFilterWnd = "Bilateral filtering";
    namedWindow(bilateralFilterWnd);
    imshow(bilateralFilterWnd, convertedImage);
    createTrackbar("S space", bilateralFilterWnd, &blur, 70, bilateralFilterSigmaSpaceCallaback, (void *)&convertedImage);
    createTrackbar("S range", bilateralFilterWnd, &blur, 70, bilateralFilterSigmaRangeCallaback, (void *)&convertedImage);

    waitKey(0);
}

int main(int argc, char *argv[])
{
    cout << "Lab2 is running" << endl;

    // Lab 2 - part 1
    Mat equalizedImage = equalization();

    // Lab 2 - part 2
    filtering(equalizedImage);

    return 0;
}