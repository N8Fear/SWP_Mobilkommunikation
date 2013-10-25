#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main ( int argc, char **argv)
{
	Mat image;
	image = imread ( argv[1], 1);

	if (argc != 2 || !image.data){
		printf("Failed to read image\n");
		return -1;
	}
	namedWindow("ImgView", CV_WINDOW_AUTOSIZE);
	imshow (" ImgView", image);

	waitKey(0);

	return 0;
}
