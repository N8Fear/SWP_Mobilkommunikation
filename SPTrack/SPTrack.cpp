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
	this->init_dimensions(cap);

	run_dct = new sp_dct();
	run_dct->dct_init(dim);

	run_hmm= new sp_hmm();
	run_hmm->hmm_init(dim);

	memset(histogram, 0, 256);

	return 0;
}

int SPTrack::init_dimensions(cv::VideoCapture &cap)
{
	int mod;
	dim.width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	dim.height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	((mod=(dim.width % 8))== 0) ? dim.width_offset = 0: dim.width_offset = BLOCKSIZE - mod;
	((mod=(dim.height % 8))== 0) ? dim.height_offset = 0: dim.height_offset = BLOCKSIZE -mod;
	return 0;
}

int SPTrack::play_stream()
{
	frame_container frame_cnt;
	while (1) {
		if (skip_sec > 0) {
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip_sec*fps; i++)
				frame_cnt.process_frame(&cap);
			skip_sec = 0;
		}

		frame_cnt.process_frame(&cap);
		run_dct->dct_exec(&frame_cnt);
		run_hmm->hmm_exec(&frame_cnt);
//		gen_histogram(*frame_cnt.get_current());
		player->update_player(&frame_cnt);
		if(waitKey(30) == 27) {
			for (int i=0; i<256; i++)
//				printf("%d;%llu;\n",i,histogram[i]);
				;
			break;
		}
	}
	// Call destruktor manually to save state of learning
	run_hmm->~sp_hmm();
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
