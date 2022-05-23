/*
    Author: Vittorio Esposito
    2005795
    Lab 4 Final Project - Image Stitching
*/

#include <iostream>
#include <stdio.h>
#include <memory>
#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

vector<vector<Mat>> images(3, vector<Mat>(3));
vector<vector<KeyPoint>> kp(9);
vector<Mat> descr(9);
vector<vector<DMatch>> matches;
vector<Mat> homographies;
vector<vector<Point2f>> srcCoords;
vector<vector<Point2f>> dstCoords;
vector<int> meanDistanceSrc;
vector<int> meanDistanceDst;
vector<Mat> stitchedImages;
Mat outputImage;

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
void loadInputImages(String basePath)
{
    vector<String> imageNames;
    glob(basePath, imageNames);

    int count = 0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            images[i][j] = imread(imageNames[count++]);
        }
}

// Resizes the input images
void resizeInputImages(int dim)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            resize(images[i][j], images[i][j], Size(dim, dim));
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

// Equalizes input images (only the luminance channel)
void equalizeImages()
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            images[i][j] = equalizeImage(images[i][j]);
}

// Performs cylindrical projection
void cylindricalProjection(double angle)
{

    // Perform cylindrical projection on the input images
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            images[i][j] = cylindricalProj(images[i][j], angle);
}

// Extacts SIFT features of the images
void extractFeaturesSIFT(int nfeatures)
{
    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    Ptr<SIFT> siftPtr = SIFT::create(nfeatures);
    int index = 0;

    for (int i = 0; i < 3; i++)
    {
        if (!stitched)
        {
            // If stitched==false, SIFT will be computed on the 3x3 matrix
            for (int j = 0; j < 3; j++)
            {
                siftPtr->detectAndCompute(images[i][j], Mat(), kp[index], descr[index]);
                index++;
            }
        }
        else
        {
            // If stitched==true, SIFT will be computed on the stitched images
            siftPtr->detectAndCompute(stitchedImages[i], Mat(), kp[i], descr[i]);
        }
    }
}

// Extacts ORB features of the images
void extractFeaturesORB(int nfeatures)
{
    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    Ptr<ORB> orbPtr = ORB::create(nfeatures);

    int index = 0;

    for (int i = 0; i < 3; i++)
    {
        if (!stitched)
        {
            // If stitched==false, SIFT will be computed on the 3x3 matrix
            for (int j = 0; j < 3; j++)
            {
                orbPtr->detectAndCompute(images[i][j], Mat(), kp[index], descr[index]);
                index++;
            }
        }
        else
        {
            // If stitched==true, SIFT will be computed on the stitched images
            orbPtr->detectAndCompute(stitchedImages[i], Mat(), kp[i], descr[i]);
        }
    }
}

// Extacts BRISK features of the images
void extractFeaturesBRISK()
{
    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    Ptr<BRISK> orbPtr = BRISK::create();

    int index = 0;

    for (int i = 0; i < 3; i++)
    {
        if (!stitched)
        {
            // If stitched==false, SIFT will be computed on the 3x3 matrix
            for (int j = 0; j < 3; j++)
            {
                orbPtr->detectAndCompute(images[i][j], Mat(), kp[index], descr[index]);
                index++;
            }
        }
        else
        {
            // If stitched==true, SIFT will be computed on the stitched images
            orbPtr->detectAndCompute(stitchedImages[i], Mat(), kp[i], descr[i]);
        }
    }
}

// Shows keypoints
void showKeypoints()
{
    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    Mat tmp;

    for (int i = 0; i < 3; i++)
    {
        if (!stitched)
        {
            // If stitched==false, keypoints of the 3x3 matrix will be shown
            for (int j = 0; j < 3; j++)
            {
                drawKeypoints(images[i][j], kp[i + j], tmp);
                imshow("", tmp);
                waitKey(0);
            }
        }
        else
        {
            // If stitched==true, keypoints of the stitched images will be shown
            drawKeypoints(stitchedImages[i], kp[i], tmp);
            imshow("", tmp);
            waitKey(0);
        }
    }
}

// Gets better matches between the different features (distance < min * ratio)
vector<DMatch> getBetterMatches(vector<DMatch> imageMatches, double ratio)
{
    vector<DMatch> betterMatches(0);
    double minDistance = imageMatches[0].distance;

    // Find the minimum distance between keypoints
    for (int j = 1; j < imageMatches.size(); j++)
    {
        if (imageMatches[j].distance < minDistance)
            minDistance = imageMatches[j].distance;
    }

    // Select the matches with distance less than ratio*minDistance
    for (int j = 0; j < imageMatches.size(); j++)
    {
        if (imageMatches[j].distance < ratio * minDistance)
            betterMatches.push_back(imageMatches[j]);
    }

    return betterMatches;
}

// Gets the best N matches between the different features
vector<DMatch> getBestMatches(vector<DMatch> imageMatches, int num)
{
    vector<DMatch> betterMatches(0);
    double minDistance = imageMatches[0].distance;

    // Return only the best N matches (based on distances)
    bool swap = true;
    while (swap)
    {
        swap = false;
        for (int i = 0; i < imageMatches.size() - 2; i++)
        {
            if (imageMatches[i].distance > imageMatches[i + 1].distance)
            {
                DMatch tmp = imageMatches[i];
                imageMatches[i] = imageMatches[i + 1];
                imageMatches[i + 1] = tmp;
                swap = true;
            }
        }
    }
    imageMatches.resize(num);
    return imageMatches;
}

// Finds the matches between the different features
void findMatches(double ratio)
{
    Ptr<BFMatcher> matcherPtr = cv::BFMatcher::create(NORM_L2, true);

    matches.resize(0);
    vector<DMatch> imageMatches;

    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    // First row
    matcherPtr->match(descr[0], descr[1], imageMatches, Mat());
    matches.push_back(getBetterMatches(imageMatches, ratio));
    matcherPtr->match(descr[1], descr[2], imageMatches, Mat());
    matches.push_back(getBetterMatches(imageMatches, ratio));

    if (!stitched)
    {
        // Second row
        matcherPtr->match(descr[3], descr[4], imageMatches, Mat());
        matches.push_back(getBetterMatches(imageMatches, ratio));
        matcherPtr->match(descr[4], descr[5], imageMatches, Mat());
        matches.push_back(getBetterMatches(imageMatches, ratio));

        // Third row
        matcherPtr->match(descr[6], descr[7], imageMatches, Mat());
        matches.push_back(getBetterMatches(imageMatches, ratio));
        matcherPtr->match(descr[7], descr[8], imageMatches, Mat());
        matches.push_back(getBetterMatches(imageMatches, ratio));
    }
}

// Finds the matches between the different features
void findMatchesV2(int num)
{
    Ptr<BFMatcher> matcherPtr = cv::BFMatcher::create(NORM_L2, true);

    matches.resize(0);
    vector<DMatch> imageMatches;

    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    // First row
    matcherPtr->match(descr[0], descr[1], imageMatches, Mat());
    matches.push_back(getBestMatches(imageMatches, num));
    matcherPtr->match(descr[1], descr[2], imageMatches, Mat());
    matches.push_back(getBestMatches(imageMatches, num));

    if (!stitched)
    {
        // Second row
        matcherPtr->match(descr[3], descr[4], imageMatches, Mat());
        matches.push_back(getBestMatches(imageMatches, num));
        matcherPtr->match(descr[4], descr[5], imageMatches, Mat());
        matches.push_back(getBestMatches(imageMatches, num));

        // Third row
        matcherPtr->match(descr[6], descr[7], imageMatches, Mat());
        matches.push_back(getBestMatches(imageMatches, num));
        matcherPtr->match(descr[7], descr[8], imageMatches, Mat());
        matches.push_back(getBestMatches(imageMatches, num));
    }
}

// Show matches
void showMatches()
{
    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    Mat tmp;

    if (!stitched)
    {
        // First row
        drawMatches(images[0][0], kp[0], images[0][1], kp[1], matches[0], tmp);
        imshow("", tmp);
        waitKey(0);

        drawMatches(images[0][1], kp[1], images[0][2], kp[2], matches[1], tmp);
        imshow("", tmp);
        waitKey(0);

        // Second row
        drawMatches(images[1][0], kp[3], images[1][1], kp[4], matches[2], tmp);
        imshow("", tmp);
        waitKey(0);

        drawMatches(images[1][1], kp[4], images[1][2], kp[5], matches[3], tmp);
        imshow("", tmp);
        waitKey(0);

        // Third row
        drawMatches(images[2][0], kp[6], images[2][1], kp[7], matches[4], tmp);
        imshow("", tmp);
        waitKey(0);

        drawMatches(images[2][1], kp[7], images[2][2], kp[8], matches[5], tmp);
        imshow("", tmp);
        waitKey(0);
    }
    else
    {
        drawMatches(stitchedImages[0], kp[0], stitchedImages[1], kp[1], matches[0], tmp);
        imshow("", tmp);
        waitKey(0);

        drawMatches(stitchedImages[1], kp[1], stitchedImages[2], kp[2], matches[1], tmp);
        imshow("", tmp);
        waitKey(0);
    }
}

// Computes an homography
void computeHomography(vector<KeyPoint> kp1, vector<KeyPoint> kp2, vector<DMatch> tmpMatches)
{

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
    homographies.push_back(findHomography(obj, scene, RANSAC, 3));

    srcCoords.push_back(obj);
    dstCoords.push_back(scene);
}

// Finds the transform between matched keypoints
void findHomographies()
{
    int size = matches.size();

    homographies.resize(0);
    srcCoords.resize(0);
    dstCoords.resize(0);

    // Determine if the operation must be performed on the 3x3 matrix or on the stitched images
    bool stitched = (stitchedImages.size() > 0);

    // First row
    computeHomography(kp[0], kp[1], matches[0]);
    computeHomography(kp[1], kp[2], matches[1]);

    if (!stitched)
    {
        // Second row
        computeHomography(kp[3], kp[4], matches[2]);
        computeHomography(kp[4], kp[5], matches[3]);

        // Third row
        computeHomography(kp[6], kp[7], matches[4]);
        computeHomography(kp[7], kp[8], matches[5]);
    }
}

// Finds distances
void findDistances()
{
    meanDistanceSrc.resize(0);
    meanDistanceDst.resize(0);

    for (int i = 0; i < srcCoords.size(); i++)
    {
        vector<Point2f> tmpSrcCoordsHoriz = srcCoords[i];
        vector<Point2f> tmpDstCoordsHoriz = dstCoords[i];

        double tmpDistSrc = 0;
        double tmpDistDst = 0;
        for (int j = 0; j < tmpSrcCoordsHoriz.size(); j++)
        {
            tmpDistSrc += tmpSrcCoordsHoriz[j].x;
            tmpDistDst += tmpDstCoordsHoriz[j].x;
        }

        tmpDistSrc = tmpDistSrc / tmpSrcCoordsHoriz.size();
        tmpDistDst = tmpDistDst / tmpDstCoordsHoriz.size();
        meanDistanceSrc.push_back(tmpDistSrc);
        meanDistanceDst.push_back(tmpDistDst);
    }
}

// Performs stiching horizontally of images
Mat performRowStitching(int rowIndex, vector<Mat> imgs)
{
    vector<int> tmpSum;
    int tmpDimX = 0;

    for (int i = rowIndex * 2; i <= rowIndex * 2 + 1; i++)
    {
        tmpSum.push_back(tmpDimX + meanDistanceSrc[i]);
        tmpDimX += meanDistanceSrc[i] - meanDistanceDst[i];
    }

    tmpDimX += imgs[2].cols;
    int tmpDimY = max(imgs[0].rows, max(imgs[1].rows, imgs[2].rows));

    Mat tmpStitchedImage(tmpDimY, tmpDimX, CV_8UC1, Scalar(0));
    imgs[0](Rect(0, 0, meanDistanceSrc[rowIndex * 2], imgs[0].rows)).copyTo(tmpStitchedImage(Rect(0, 0, meanDistanceSrc[rowIndex * 2], imgs[0].rows)));
    imgs[1](Rect(meanDistanceDst[rowIndex * 2], 0, imgs[1].cols - meanDistanceDst[rowIndex * 2], imgs[1].rows)).copyTo(tmpStitchedImage(Rect(tmpSum[0], 0, imgs[1].cols - meanDistanceDst[rowIndex * 2], imgs[1].rows)));

    int tmpCol = imgs[2].cols - meanDistanceDst[rowIndex * 2 + 1];
    int tmpRow = imgs[2].rows;
    imgs[2](Rect(meanDistanceDst[rowIndex * 2 + 1], 0, tmpCol, tmpRow)).copyTo(tmpStitchedImage(Rect(tmpSum[1], 0, tmpCol, tmpRow)));

    return tmpStitchedImage;
}

// Performs stitching row by row
void performRowStitching()
{
    for (int i = 0; i < 3; i++)
    {
        stitchedImages.push_back(performRowStitching(i, images[i]));
        cout << "Row n." << i << ": done!" << endl;
    }
}

// Rotates stitched images of 90 degrees clowise or counterclockwise
void rotateStitchedImages(bool isOutputImage)
{
    if (isOutputImage)
    {
        rotate(outputImage, outputImage, ROTATE_90_CLOCKWISE);
    }
    else
    {
        for (int i = 0; i < 3; i++)
            rotate(stitchedImages[i], stitchedImages[i], ROTATE_90_COUNTERCLOCKWISE);
    }
}

// Performs final stitching
void performFinalStitching()
{
    vector<int> tmpSum;
    int tmpDimX = 0;

    for (int i = 0; i < meanDistanceSrc.size(); i++)
    {
        tmpSum.push_back(tmpDimX + meanDistanceSrc[i]);
        tmpDimX += meanDistanceSrc[i] - meanDistanceDst[i];
    }

    tmpDimX += stitchedImages[2].cols;
    int tmpDimY = max(stitchedImages[0].rows, max(stitchedImages[1].rows, stitchedImages[2].rows));

    Mat panoramicMat(tmpDimY, tmpDimX, CV_8UC1, Scalar(0));
    stitchedImages[0](Rect(0, 0, meanDistanceSrc[0], stitchedImages[0].rows)).copyTo(panoramicMat(Rect(0, 0, meanDistanceSrc[0], stitchedImages[0].rows)));
    stitchedImages[1](Rect(meanDistanceDst[0], 0, stitchedImages[1].cols - meanDistanceDst[0], stitchedImages[1].rows)).copyTo(panoramicMat(Rect(tmpSum[0], 0, stitchedImages[1].cols - meanDistanceDst[0], stitchedImages[1].rows)));

    int col = stitchedImages[2].cols - meanDistanceDst[1];
    int row = stitchedImages[2].rows;
    stitchedImages[2](Rect(meanDistanceDst[1], 0, col, row)).copyTo(panoramicMat(Rect(tmpSum[1], 0, col, row)));

    equalizeHist(panoramicMat, outputImage);
}

Mat performBilateralFiltering(Mat inputImage, double sigmaRangeBilateral, double sigmaSpaceBilateral)
{

    Mat tmp;
    int diameter = 6 * sigmaSpaceBilateral;
    bilateralFilter(inputImage, tmp, diameter, sigmaRangeBilateral, sigmaSpaceBilateral, BORDER_DEFAULT);
    return tmp;
}

int main(int argc, char *argv[])
{
    cout << "Lab4 is running - image stitching" << endl;

    cout << "Loading images" << endl;
    loadInputImages("./data/T3/");

    cout << "Resize images" << endl;
    // resizeInputImages(500);

    cout << "Performing blurring on input images" << endl;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ;
    // GaussianBlur(images[i][j], images[i][j], Size(5, 5), 10.0, 10.0); // 5. 10.0
    //  bilateralFilter(images[i][j], images[i][j], 6*6.0, 0.1, 6.0);
    // images[i][j] = performBilateralFiltering(images[i][j], 0.25, 6.0);

    cout << "Performing cylindrical projection on the input images" << endl;
    cylindricalProjection(10);

    // Feature detection
    cout << "Extracting features on the input images" << endl;
    // extractFeaturesSIFT(5000);
    //  extractFeaturesORB(5000);
    extractFeaturesBRISK();
    // showKeypoints();

    // Feature matching
    cout << "Finding matches on the input images" << endl;
    findMatchesV2(6);
    // findMatches(1.8);
    // showMatches();

    // Image matching - RANSACF
    cout << "Finding homographies on the input images" << endl;
    findHomographies();

    cout << "Finding distances on the input images" << endl;
    findDistances();

    cout << "Performing stitching on the input images (row by row)" << endl;
    performRowStitching();

    cout << "Perform rotation of the stitched images" << endl;
    rotateStitchedImages(false);

    cout << "Extracting features of the stitched images" << endl;
    // extractFeaturesSIFT(5000);
    // extractFeaturesORB(5000);
    extractFeaturesBRISK();
    // showKeypoints();

    // Feature matching
    cout << "Finding matches of the stitched images" << endl;
    findMatchesV2(6);
    // findMatches(1.8);
    // showMatches();

    cout << "Finding homographies of the stitched images" << endl;
    findHomographies();

    cout << "Finding distances of the stitched images" << endl;
    findDistances();

    cout << "Performing stitching of the stitched images" << endl;
    outputImage = performRowStitching(0, stitchedImages);
    rotateStitchedImages(true);

    // GaussianBlur(outputImage, outputImage, Size(3, 3), 5.0, 5.0);
    equalizeHist(outputImage, outputImage);

    imwrite("outputImage.jpg", outputImage);
    // imshow("", outputImage);
    // waitKey(0);

    cout << "Done" << endl;
    return 0;
}