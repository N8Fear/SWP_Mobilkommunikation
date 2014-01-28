#pragma once

/*TODO: Check necissity of included headers*/

#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

//#include <time.h>

#include "frame_container.h"
#include "sp_player.h"
#include "sp_dct.h"
#include "sp_hmm.h"

#define BLACK 0
#define GRAY 133
#define WHITE 255

#define BLOCKSIZE 8

// TODO: Check necessity
#define DEBUG_R 0
#define DEBUG_C 5

class SPTrack{
	private:
		long skip_sec;
		int width, height, width_offset, height_offset;
		int init_loop(char *path);
		int parse_cl_param(int argc, char *argv[]);
		sp_dct *run_dct;
		sp_player *player;
		long long int histogram[256];
		void gen_histogram(cv::Mat);
		cv::VideoCapture cap;
		int init_dimensions(cv::VideoCapture &cap);

	public:
		int play_stream();
		SPTrack(int argc, char *argv[]);
};

