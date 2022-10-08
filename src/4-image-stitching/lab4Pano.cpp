/*
    Author: Vittorio Esposito
    2005795
    Lab 4 Final Project - Image Stitching
*/

#include <iostream>
#include <stdio.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

vector<Mat> images;
vector<vector<DMatch>> matches;
vector<vector<KeyPoint>> kp;
vector<Mat> descr;
vector<Mat> homographies;
vector<vector<Point2f>> srcCoords;
vector<vector<Point2f>> dstCoords;
vector<int> meanDistanceSrc;
vector<int> meanDistanceDst;
Mat stitchedImage;

Mat cylindricalProj(
    const Mat &image,
    const double angle)
{
    cv::Mat tmp, result;
    cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
    result = tmp.clone();

    double alpha(angle / 180 * CV_PI);
    double d((image.cols / 2.0) / tan(alpha));
    double r(d / cos(alpha));
    double d_by_r(d / r);
    int half_height_image(image.rows / 2);
    int half_width_image(image.cols / 2);

    for (int x = -half_width_image + 1,
             x_end = half_width_image;
         x < x_end; ++x)
    {
        for (int y = -half_height_image + 1,
                 y_end = half_height_image;
             y < y_end; ++y)
        {
            double x1(d * tan(x / r));
            double y1(y * d_by_r / cos(x / r));

            if (x1 < half_width_image &&
                x1 > -half_width_image + 1 &&
                y1 < half_height_image &&
                y1 > -half_height_image + 1)
            {
                result.at<uchar>(y + half_height_image, x + half_width_image) = tmp.at<uchar>(round(y1 + half_height_image),
                                                                                              round(x1 + half_width_image));
            }
        }
    }

    return result;
}

// Loads all images located in @basePath
void loadImages(String basePath)
{
    vector<String> imageNames;
    glob(basePath, imageNames);

    images.resize(imageNames.size());

    for (int i = 0; i < imageNames.size(); i++)
    {
        // Load the image, resize it and push it into the output vector
        Mat tmpImage = imread(imageNames[i]);
        images[i] = tmpImage;
    }
}

// Equalizes a single image (only the luminance channel)
Mat equalizeImage(Mat image)
{
    // Convert the images to Lab color space
    cvtColor(image, image, COLOR_BGR2Lab);

    // Split histograms
    vector<Mat> histogramsLab;
    split(image, histogramsLab);

    // equalize only the luminance (L) channel
    Mat luminanceHistogram;
    equalizeHist(histogramsLab[0], histogramsLab[0]);

    merge(histogramsLab, image);

    cvtColor(image, image, COLOR_Lab2BGR);

    return image;
}

// Equalizes images (only the luminance channel)
void equalizeImages()
{
    for (int i = 0; i < images.size(); i++)
    {
        images[i] = equalizeImage(images[i]);
    }
}

// Performs cylindrical projection
void cylindricalProjection(double angle)
{
    for (int i = 0; i < images.size(); i++)
    {
        images[i] = cylindricalProj(images[i], angle);
    }
}

// Extacts SIFT features of the images
void extractFeaturesSIFT()
{
    Ptr<SIFT> siftPtr = SIFT::create(0, 3, 0.035, 10, 1.65);
    kp.resize(images.size());
    descr.resize(images.size());

    for (int i = 0; i < images.size(); i++)
    {
        siftPtr->detectAndCompute(images[i], Mat(), kp[i], descr[i]);
    }
}

// Shows keypoints
void showKeypoints()
{
    for (int i = 0; i < images.size(); i++)
    {
        Mat tmp;
        drawKeypoints(images[i], kp[i], tmp);
        imshow("", tmp);
        waitKey(0);
    }
}

// Finds the matches between the different features
void findMatches(int ratio)
{
    Ptr<BFMatcher> matcherPtr = cv::BFMatcher::create(NORM_L2, true);

    matches.resize(kp.size() - 1);
    double minDistance = 10000;

    // For each couple of adjacent images in the grid
    for (int i = 0; i < kp.size() - 1; i++)
    {
        // Compute the match between the different features
        vector<DMatch> imageMatches;
        matcherPtr->match(descr[i], descr[i + 1], imageMatches, Mat());

        // Find the minimum distance between keypoints
        for (int j = 0; j < imageMatches.size(); j++)
        {
            if (imageMatches[j].distance < minDistance)
                minDistance = imageMatches[j].distance;
        }

        // Select the matches with distance less than ratio*minDistance
        for (int j = 0; j < imageMatches.size(); j++)
        {
            if (imageMatches[j].distance < ratio * minDistance)
                matches[i].push_back(imageMatches[j]);
        }
    }
}

// Show matches
void showMatches()
{

    for (int i = 0; i < matches.size(); i++)
    {
        Mat tmp;
        drawMatches(images[i], kp[i], images[i + 1], kp[i + 1], matches[i], tmp);
        imshow("", tmp);
        waitKey(0);
    }
}

// Finds the transform between matched keypoints
void findHomographies()
{
    int size = images.size() - 1;

    homographies.resize(size);
    srcCoords.resize(size);
    dstCoords.resize(size);

    for (int i = 0; i < size; i++)
    {
        vector<KeyPoint> kp1 = kp[i];
        vector<KeyPoint> kp2 = kp[i + 1];
        vector<DMatch> tmpMatches = matches[i];

        vector<Point2f> obj;
        vector<Point2f> scene;
        vector<uchar> mask;

        for (int j = 0; j < tmpMatches.size(); j++)
        {
            // Get the keypoints from the matches
            obj.push_back(kp1[tmpMatches[j].queryIdx].pt);
            scene.push_back(kp2[tmpMatches[j].trainIdx].pt);
        }

        // if (obj.size() > 0 && scene.size() > 0)
        homographies[i] = findHomography(obj, scene, RANSAC, 3);

        srcCoords[i] = obj;
        dstCoords[i] = scene;
    }
}

// Finds distances
void findDistances()
{
    meanDistanceSrc.resize(0);
    meanDistanceDst.resize(0);

    for (int i = 0; i < images.size() - 1; i++)
    {

        vector<Point2f> tmpSrcCoords = srcCoords[i];
        vector<Point2f> tmpDstCoords = dstCoords[i];

        double tmpDistSrc = 0;
        double tmpDistDst = 0;
        for (int j = 0; j < tmpSrcCoords.size(); j++)
        {
            tmpDistSrc = tmpDistSrc + tmpSrcCoords[j].x;
            tmpDistDst = tmpDistDst + tmpDstCoords[j].x;
        }

        tmpDistSrc = tmpDistSrc / tmpSrcCoords.size();
        tmpDistDst = tmpDistDst / tmpDstCoords.size();
        meanDistanceSrc.push_back(tmpDistSrc);
        meanDistanceDst.push_back(tmpDistDst);
    }
}

// Performs stitching
void performStitching()
{
    vector<int> tmpSum;
    int tmpDimX = 0;

    for (int i = 0; i < meanDistanceSrc.size(); i++)
    {
        tmpSum.push_back(tmpDimX + meanDistanceSrc[i]);
        tmpDimX = tmpDimX + meanDistanceSrc[i] - meanDistanceDst[i];
    }

    tmpDimX += images[images.size() - 1].cols;
    int tmpDimY = images[images.size() - 1].rows;

    Mat tmpStitchedImage(tmpDimY*1.1, tmpDimX*1.1, CV_8UC1, Scalar(0));
    images[0](Rect(0, 0, meanDistanceSrc[0], images[0].rows)).copyTo(tmpStitchedImage(Rect(0, 0, meanDistanceSrc[0], images[0].rows)));

    for (int i = 0; i < images.size() - 2; i++)
    {
        Mat tmpImg = images[i + 1];
        tmpImg(Rect(meanDistanceDst[i], 0, tmpImg.cols - meanDistanceDst[i], tmpImg.rows)).copyTo(tmpStitchedImage(Rect(tmpSum[i], 0, tmpImg.cols - meanDistanceDst[i], tmpImg.rows)));
    }

    int col = images[images.size() - 1].cols - meanDistanceDst[images.size() - 2];
    int row = images[images.size() - 1].rows;
    images[images.size() - 1](Rect(meanDistanceDst[images.size() - 2], 0, col, row)).copyTo(tmpStitchedImage(Rect(tmpSum[images.size() - 2], 0, col, row)));

    // cvtColor(tmpStitchedImage, tmpStitchedImage, COLOR_BGR2GRAY);
    equalizeHist(tmpStitchedImage, tmpStitchedImage);

    stitchedImage = tmpStitchedImage.clone();
}

int main(int argc, char *argv[])
{
    cout << "Lab4 is running - image stitching" << endl;

    // Image acquisition
    cout << "Loading images" << endl;
    loadImages("./data/panoramic_garden/");

    // Equalization
    cout << "Equalizing images" << endl;
    equalizeImages();

    cout << "Performing cylindrical projection" << endl;
    cylindricalProjection(10);

    // Feature detection
    cout << "Extracting features" << endl;
    extractFeaturesSIFT();
    // showKeypoints();

    // Feature matching
    cout << "Finding matches" << endl;
    findMatches(5);
    // showMatches();

    // Image matching - RANSACF
    cout << "Finding homographies" << endl;
    findHomographies();

    cout << "Finding distances" << endl;
    findDistances();

    cout << "Performing stitching" << endl;
    performStitching();

    imwrite("outputPano.jpg", stitchedImage);
    imshow("", stitchedImage);
    waitKey(0);

    cout << "Done" << endl;
    return 0;
}