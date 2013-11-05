#include <iostream>
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
	Mat img;

	namedWindow("HOGtest", CV_WINDOW_AUTOSIZE);
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


	while (true) {
		ctr++;
		capture >> img;

		if (ctr == 10) {
			vector<Rect> detected, filtered;
			hog.detectMultiScale(img, detected, 0, Size(8,8), Size(32,32), 1.05, 2);
			size_t i,j;
			for (i=0; i<detected.size(); i++) {
				Rect r = detected[i];
				for(j=0; j<detected.size();j++)
					if(j!=i && (r & detected[j])==r)
						break;
					if (j==detected.size())
						filtered.push_back(r);
			}

			for (i=0; i<filtered.size();i++) {
				Rect r = filtered[i];
				r.x += cvRound(r.width*0.1);
				r.width = cvRound(r.width*0.8);
				r.y += cvRound(r.height*0.06);
				r.height = cvRound(r.height*0.9);
				rectangle(img, r.tl(), r.br(), cv::Scalar(255,0,0), 2);
			}
			ctr=0;
		}
		imshow("HOGtest", img);
		if (waitKey(20) >= 0)
			break;
	}
}
