
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char *window_name = "Edge Map";

// CannyThreshold function
static void CannyThreshold(int, void *)
{
    // blur the image with a filter of kernel size 3
    blur(src_gray, detected_edges, Size(3, 3));

    // apply the OpenCV function cv::Canny
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

    // fill a dst image with zeros (the image is completely black)
    dst = Scalar::all(0);

    // Use the function cv::Mat::copyTo to map only the areas of the image that are identified as edges on a black background.
    // it will only copy the pixels in the locations where they have non-zero values. Since the output of the Canny detector is
    // the edge contours on a black background, the resulting dst will be black in all the area but the detected edges.
    src.copyTo(dst, detected_edges);

    imshow(window_name, dst);
}

int main(int argc, char **argv)
{

    // Loads the source image
    CommandLineParser parser(argc, argv, "{@input | ./data/road1.png | input image}");
    src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n"
                  << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    // Resize of a factor of 2 (for small screens)
    cv::resize(src, src, cv::Size(src.cols / 2, src.rows / 2));

    // Create a matrix of the same type and size of src (to be dst)
    dst.create(src.size(), src.type());

    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    // Create a window to display the results
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // Create a Trackbar for the user to enter the lower threshold for the Canny detector.
    // The variable to be controlled by the Trackbar is lowThreshold with a limit of max_lowThreshold.
    // Each time the Trackbar registers an action, the callback function CannyThreshold will be invoked.
    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

    CannyThreshold(0, 0);
    waitKey(0);
    return 0;
}