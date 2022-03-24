// Computer Vision 2022 - P. Zanuttigh - LAB1

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

// Window size for averaging color after click
#define NEIGHBORHOOD_Y 9
#define NEIGHBORHOOD_X 9
// Threshold for color similarity (Euclidean distance in CIELAB space)
#define SIMILARITY_THRESHOLD 50

void onMouse( int event, int x, int y, int f, void* userdata){

  // If the left button is pressed
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    // Retrieving the image from the main
    cv::Mat image = *(cv::Mat*) userdata;

    // Preventing segfaults for looking over the image boundaries
    if (y + NEIGHBORHOOD_Y > image.rows
        || x + NEIGHBORHOOD_X > image.cols)
      return;

    // Conversion color space RGB->HSV
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, COLOR_BGR2Lab);

    // Mean on the neighborhood
    cv::Rect rect(x - (NEIGHBORHOOD_X/2), y - (NEIGHBORHOOD_Y/2), NEIGHBORHOOD_X, NEIGHBORHOOD_Y);
    cv::Scalar mean = cv::mean(image_lab(rect));
    std::cout << "Mean selected color: " << mean << std::endl;

    // Color segmentation
    for(int r = 0; r < image_lab.rows; ++r)
    {
      for(int c = 0; c < image_lab.cols; ++c)
      {
        cv::Vec3b& current_color = image_lab.at<cv::Vec3b>(r, c);
        if (std::sqrt((mean[0] - current_color[0])*(mean[0] - current_color[0])+
            (mean[1] - current_color[1])*(mean[1] - current_color[1])+
            (mean[2] - current_color[2])*(mean[2] - current_color[2]) ) < SIMILARITY_THRESHOLD)
        {
            // Change only A and B coordinates
          image_lab.at<cv::Vec3b>(r,c)[1] = 55;
          image_lab.at<cv::Vec3b>(r, c)[2] = -38;
        }
      }
    }

   
    cv::Mat image_out;
    // color conversion HSV->RGB and Visualization
    cv::cvtColor(image_lab, image_out, COLOR_Lab2BGR);
    //  Visualization and save to disk
    cv::imshow("final_result", image_out);
    cv::imwrite("output.jpg", image_out);
    cv::waitKey(0);
  }

}

int main(int argc, char** argv)
{
  // Load the image
  cv::Mat input_img = cv::imread("./data/roma.jpg");
	
  // Resize of a factor of 2 for visualization on small screens
  cv::resize(input_img, input_img, cv::Size(input_img.cols / 2, input_img.rows / 2));
  cv::imshow("img",input_img);
  cv::setMouseCallback( "img", onMouse, (void*)&input_img);

  cv::waitKey(0);
 
  return 0;
}

