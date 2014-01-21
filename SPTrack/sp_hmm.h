#pragma once
#include "SPTrack.h"
#include "CvHMM/CvHMM.h"

#define OBS1 0
#define OBS2 1
#define OBS3 2

class sp_hmm {
	public:
		int hmm_init();
		int hmm_learn(cv::Mat);
		int hmm_print();
		sp_hmm();
	private:
		int obs_max;
		int seq_max;
		int block_r;
		int block_c;
		int seq_num;
		int obs_num;
		int max_iter;
		CvHMM hmm;
		cv::Mat train_seq;
		cv::Mat TRGUESS;
		cv::Mat EMITGUESS;
		cv::Mat INITGUESS;
};
