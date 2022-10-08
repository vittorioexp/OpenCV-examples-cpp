
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

/*
    WORKS:
     .\HoughCircleTransform.exe ./data/road3.jpg 1 50 500 20 4 30
*/

int main(int argc, char **argv)
{
    const char *filename = argc >= 2 ? argv[1] : "./data/road1.png";
    int dp = atoi(argv[2]);
    int minDist = atoi(argv[3]);
    int p1 = atoi(argv[4]);
    int p2 = atoi(argv[5]);
    int minRad = atoi(argv[6]);
    int maxRad = atoi(argv[7]);

    // Loads an image
    Mat src = imread(samples::findFile(filename), IMREAD_COLOR);

    // Check if image is loaded fine
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", filename);
        return EXIT_FAILURE;
    }

    // Resize of a factor of 2 (for small screens)
    // cv::resize(src, src, cv::Size(src.cols / 2, src.rows / 2));
    // imshow("input image", src);

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    medianBlur(gray, gray, 3);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, dp,
                 minDist,               // change this value to detect circles with different distances to each other
                 p1, p2, minRad, maxRad // change the last two parameters
                                        // (min_radius & max_radius) to detect larger circles
    );
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(src, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(src, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
    }

    imshow("detected circles", src);
    waitKey();
    return EXIT_SUCCESS;
}