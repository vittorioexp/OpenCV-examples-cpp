#include <iostream>
#include <cstdlib>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Declare the output variables
    Mat dst, cdst, cdstP;
    const char *default_file = "./data/road1.png";
    const char *filename = argc >= 4 ? argv[1] : default_file;
    int low_th = atoi(argv[2]);
    int high_th = atoi(argv[3]);


    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

    // Check if image is loaded fine
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }

    // Resize of a factor of 2 (for small screens)
    cv::resize(src, src, cv::Size(src.cols / 2, src.rows / 2));

    // blur the image with a filter of kernel size 3
    blur(src, src, Size(3, 3));

    // Edge detection
    Canny(src, dst, low_th, high_th, 3);

    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    vector<Vec2f> lines;                               // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Probabilistic Line Transform
    vector<Vec4i> linesP;                                 // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection

    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        Vec4i l = linesP[i];
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Show results
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

    // Wait and Exit
    waitKey();
    return 0;
}