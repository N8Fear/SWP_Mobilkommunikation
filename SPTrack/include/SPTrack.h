#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "config.h"
#include "container.h"
#include "sp_player.h"
#include "sp_dct.h"
#include "sp_hmm.h"

class SPTrack{
	private:
		cv::VideoCapture cap;
		long skip_sec;
		long long int histogram[256];
		Dimensions dim;
		sp_dct *run_dct;
		sp_hmm *run_hmm;
		sp_player *player;

		int init_dimensions(cv::VideoCapture &cap);
		int init_loop(char *path);
		int parse_cl_param(int argc, char *argv[]);
		void gen_histogram(cv::Mat);

	public:
		int play_stream();
		SPTrack(int argc, char *argv[]);
};
