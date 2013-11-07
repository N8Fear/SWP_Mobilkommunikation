#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main ( int argc, const char * argv[])
{
//	CvCapture* capture = cvCreateFileCapture( argv[1]);
	if (argc != 2){
		cerr << "No Filename specified.";
		return 1;
	}
	int ctr = 0;
	VideoCapture capture (CV_CAP_ANY);
	capture.open( argv[1]);
	Mat img, back, fore, processed, old, temp;
	BackgroundSubtractorMOG2 back_sub;
	vector<vector<Point> > contours;

	namedWindow("BGRemove", CV_WINDOW_AUTOSIZE);

	back_sub.set("nmixtures",3);
//	back_sub.set("bShadowDetection", 0);
	capture >> old;


	while (true) {
		capture >> temp;
		addWeighted(old, 0.5, temp, 0.5, 0.0, img);
		back_sub.operator()(img,fore);
		back_sub.getBackgroundImage(back);
		img = img - back;
		erode(fore, fore, Mat());
		dilate(fore,fore, Mat());
		processed = img - back;
		findContours(fore, contours, CV_RETR_TREE,CV_CHAIN_APPROX_TC89_L1);//,CV_CHAIN_APPROX_TC89_KCOS);
		for(std::vector<vector<Point> >::iterator it = contours.begin(); it != contours.end(); ){
			vector<Point> temp = *it;
			if (temp.size() < 20)
			contours.erase(it);
			else
				it++;
//			cout <<  "test\n";
		}
		drawContours(processed, contours, -1, cv::Scalar(0,0,255),2);

		capture >> old;

		imshow("BGRemove", processed*10);
		if (waitKey(20) >= 0)
			break;
	}
}
