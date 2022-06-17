// Computer Vision 2021 (P. Zanuttigh) - LAB 4

#ifndef LAB4__PANORAMIC__UTILS__H
#define LAB4__PANORAMIC__UTILS__H

#include <memory>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

class PanoramicUtils
{
public:
    static cv::Mat cylindricalProj(
        const cv::Mat &image,
        const double angle);
};

#endif // LAB4__PANORAMIC__UTILS__H
