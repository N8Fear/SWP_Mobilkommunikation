//#include "SPTrack.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;
using namespace gpu;

class SPTrack{
	private:
		long skip_sec;
		int init_player(char *path);
		int parse_cl_param(int argc, char *argv[]);
		VideoCapture cap;

	public:
		int play_stream();
		SPTrack(int argc, char *argv[]);
};

int SPTrack::init_player(char *path)
{
	VideoCapture capture(path);
	cap = capture;
	if (!cap.isOpened()) {
		cerr << "Cannot open video stream!" << endl;
		return -1;
	}
	namedWindow("Player", CV_WINDOW_AUTOSIZE&CV_GUI_NORMAL);
	return 0;
}

int SPTrack::play_stream()
{
	while (1) {
		Mat frame;
		if (skip_sec > 0) {
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip_sec*fps; i++)
				cap.read(frame);
			skip_sec = 0;
		}
		if (!cap.read(frame)) {
			cerr << "Error reading frame from stream" << endl;
			break;
		}
		imshow("Player", frame);
		if(waitKey(30) == 27)
			break;
	}
	return 0;
}

int SPTrack::parse_cl_param(int argc, char *argv[])
{
	switch (argc){
		case 3:
			SPTrack::skip_sec = (long) atoi (argv[2]);
		case 2:
			cout << "Skipping " << skip_sec << " seconds..." << endl;
			break;
		default:
			cout << "Syntax: " << argv[0] << " <filename> [<skipped seconds>]" <<endl;
			exit(-5);
	}
	return 0;
}

SPTrack::SPTrack(int argc, char *argv[])
{
	skip_sec = 0;
	parse_cl_param(argc, argv);
	init_player(argv[1]);
}

int main(int argc, char* argv[]) {
	SPTrack *spt = new SPTrack(argc, argv);
	spt->play_stream();
}
