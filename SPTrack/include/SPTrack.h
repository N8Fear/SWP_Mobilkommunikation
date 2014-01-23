#pragma once

/*TODO: Check necissity of included headers*/

#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <time.h>

#include "sp_player.h"
#include "sp_dct.h"

#define BLACK 0
#define GRAY 133
#define WHITE 255

#define BLOCKSIZE 8


class SPTrack{
	private:
		long skip_sec;
		int init_loop(char *path);
		int parse_cl_param(int argc, char *argv[]);
		sp_dct *run_dct;
		sp_player *player;
		long long int histogram[256];
		void gen_histogram(cv::Mat);
		cv::VideoCapture cap;

	public:
		int play_stream();
		SPTrack(int argc, char *argv[]);
};
