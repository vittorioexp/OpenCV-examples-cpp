/*
    Author: Vittorio Esposito
    2005795
    Lab 4 Final Project
*/

#include <iostream>
#include <stdio.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

vector<vector<KeyPoint>> kp;
vector<Mat> descr;
vector<vector<Point2f>> srcCoords;
vector<vector<Point2f>> dstCoords;

// Returns all images located in @basePath
vector<Mat> loadImages(String basePath)
{
    vector<String> imageNames;
    vector<Mat> images;
    glob(basePath, imageNames);

    for (int i = 0; i < imageNames.size(); i++)
    {
        // Load the image, resize it and push it into the output vector
        Mat tmpImage = imread(imageNames[i]);
        resize(tmpImage, tmpImage, Size(), 0.35, 0.35);
        images.push_back(tmpImage);
    }
    return images;
}

// Returns the equalized images (only the luminance channel)
vector<Mat> equalizeImages(vector<Mat> inputImages)
{

    vector<Mat> convertedImages;

    for (int i = 0; i < inputImages.size(); i++)
    {
        Mat inputImage = inputImages[i];
        Mat convertedImage;

        // Convert the images to Lab color space
        cvtColor(inputImage, convertedImage, COLOR_BGR2Lab);

        // Split histograms
        vector<Mat> histogramsLab;
        split(convertedImage, histogramsLab);

        // equalize only the luminance (L) channel
        Mat luminanceHistogram;
        equalizeHist(histogramsLab[0], histogramsLab[0]);

        merge(histogramsLab, convertedImage);

        cvtColor(convertedImage, convertedImage, COLOR_Lab2BGR);

        convertedImages.push_back(convertedImage);
    }

    return convertedImages;
}

// Extacts SIFT features of the images
void extractFeaturesSIFT(vector<Mat> images)
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
void showKeypoints(vector<Mat> images)
{

    for (int i = 0; i < images.size(); i++)
    {
        Mat tmp;
        drawKeypoints(images[i], kp[i], tmp);
        imshow("", tmp);
        waitKey(0);
    }
}

// Returns the matches between the different features
vector<vector<DMatch>> findMatches(int ratio)
{
    Ptr<BFMatcher> matcherPtr = cv::BFMatcher::create(NORM_L2, true);

    vector<vector<DMatch>> refinedMatches(kp.size() - 1);
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
                refinedMatches[i].push_back(imageMatches[j]);
        }
    }

    return refinedMatches;
}

// Show matches
void showMatches(vector<vector<DMatch>> matches, vector<Mat> images)
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
vector<Mat> findHomographies(vector<vector<DMatch>> matches)
{
    int size = kp.size() - 1;

    srcCoords.resize(size);
    dstCoords.resize(size);
    vector<Mat> homographies(size);

    for (int i = 0; i < size; i++)
    {
        vector<KeyPoint> kp1 = kp[i];
        vector<KeyPoint> kp2 = kp[i + 1];
        vector<DMatch> tmpMatches = matches[i];

        vector<Point2f> obj;
        vector<Point2f> scene;
        vector<Point2f> mask;

        for (int j = 0; j < tmpMatches.size(); j++)
        {
            // Get the keypoints from the matches
            obj.push_back(kp1[tmpMatches[j].queryIdx].pt);
            scene.push_back(kp2[tmpMatches[j].trainIdx].pt);
        }

        if (obj.size() > 0 && scene.size() > 0)
            homographies[i] = findHomography(obj, scene, RANSAC);

        srcCoords[i] = obj;
        dstCoords[i] = scene;
    }
    return homographies;
}

int main(int argc, char *argv[])
{
    cout << "Lab4 is running - image stitching" << endl;

    // Image acquisition
    cout << "Loading images" << endl;
    vector<Mat> images = loadImages("./data/T1/");

    // Equalization
    cout << "Equalizing images" << endl;
    images = equalizeImages(images);

    // Feature detection
    cout << "Extracting features" << endl;
    extractFeaturesSIFT(images);
    // showKeypoints(images);

    // Feature matching
    cout << "Finding matches" << endl;
    vector<vector<DMatch>> matches = findMatches(5);
    // showMatches(matches, images);

    // Image matching - RANSACF
    vector<Mat> homographies = findHomographies(matches);
    cout << "done" << endl;

    

    // Global alignment

    // Blending and Composition

    return 0;
}