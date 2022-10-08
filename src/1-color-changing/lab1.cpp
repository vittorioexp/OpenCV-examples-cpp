#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

Mat srcImage;
Mat outImage;


bool isInsideThreshold(Mat inputMat, int i, int j, Scalar meanColors) {
    int t=20;
    return (inputMat.at<Vec3b>(i,j)[0]>meanColors[0]-t && inputMat.at<Vec3b>(i,j)[0]<meanColors[0]+t) &&
     (inputMat.at<Vec3b>(i,j)[1]>meanColors[1]-t && inputMat.at<Vec3b>(i,j)[1]<meanColors[1]+t) &&
     (inputMat.at<Vec3b>(i,j)[2]>meanColors[2]-t && inputMat.at<Vec3b>(i,j)[2]<meanColors[2]+t);
}

// This function is automatically called when a mouse event happens
// x,y : coordinates of mouse position, event: type of event, flags: get buttons status
void MouseFunc(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        cout << "left click: " << x << " " << y << endl;
        Scalar meanColors = mean(srcImage(Range(y-4, y+4), Range(x-4, x+4)));
        outImage = srcImage.clone();
        for (int i=0; i<srcImage.rows; i++) {
            for (int j=0; j<srcImage.cols; j++) {
                if (isInsideThreshold(srcImage, i, j, meanColors)) {
                    outImage.at<Vec3b>(i,j)[0]=37;
                    outImage.at<Vec3b>(i,j)[1]=201;
                    outImage.at<Vec3b>(i,j)[2]=92;
                }
            } 
        }
        imshow("outImage", outImage);
        waitKey(0);
    }
}


int main( int argc, char** argv )
{
    srcImage = imread("./data/roma.jpg");
    if (!srcImage.data) {
        return 1;
    }
    imshow("srcImage", srcImage);

    // Set the callback function for any mouse event
    // The function MouseFunc will be called when some mouse event happens
    // You can pass data to the function (e.g., the image), use cast to recover the data
    setMouseCallback("srcImage", MouseFunc);

    waitKey(0);


    return 0;
}