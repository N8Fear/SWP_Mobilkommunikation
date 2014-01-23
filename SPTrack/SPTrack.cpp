#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "SPTrack.h"
#include "sp_dct.h"
#include "sp_hmm.h"
#include "sp_player.h"

using namespace cv;
using namespace std;

int SPTrack::init_loop(char *path)
{
	VideoCapture capture(path);
	cap = capture;
	if (!cap.isOpened()) {
		cerr << "Cannot open video stream!" << endl;
		return -1;
	}
	/* Initialization of sub processes and stuff: */

	player= new sp_player("Player");

//	sp_hmm *hmm= new sp_hmm();

	run_dct = new sp_dct();
	run_dct->init_dct(&cap);

	memset(histogram, 0, 256);
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
		frame = run_dct->exec_dct(frame);
		gen_histogram(frame);
		player->update_player(frame);
		if(waitKey(30) == 27) {
			for (int i=0; i<256; i++)
				printf("%d;%llu;\n",i,histogram[i]);
			break;
		}
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


void SPTrack::gen_histogram(Mat input){
	// update histrogramm information
	for (int r = 0; r < input.rows; r += BLOCKSIZE)
		for (int c = 0; c < input.cols; c += BLOCKSIZE) {
			SPTrack::histogram[input.at<uint8_t>(r,c)] += 1;
			if (SPTrack::histogram[input.at<uint8_t>(r,c)] == 0)
				cerr << "INTEGER OVERFLOW @HISTROGRAMM" << endl;
		}
}

SPTrack::SPTrack(int argc, char *argv[])
{
	skip_sec = 0;
	parse_cl_param(argc, argv);
	init_loop(argv[1]);
	this->play_stream();
}

int main(int argc, char* argv[])
{
	new SPTrack(argc, argv);
	return 0;
}
