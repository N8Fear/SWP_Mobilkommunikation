#include <stdio.h>
#include <opencv2/opencv.hpp>

int main (int argc, char **argv)
{
	cvNamedWindow("VidView", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateFileCapture( argv[1]);
	IplImage* frame;

	while(1){
		frame = cvQueryFrame(capture);
		if (!frame)
			break;
		cvShowImage("VidView",frame);
		char c = cvWaitKey(33);
		if (c == 27)
			break;
		}
	cvReleaseCapture(&capture);
	cvDestroyWindow("VidView");

	return 0;
}
