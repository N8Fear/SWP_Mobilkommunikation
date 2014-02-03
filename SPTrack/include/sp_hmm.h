#pragma once
#include "SPTrack.h"
#include "container.h"
#include "CvHMM/CvHMM.h"
#include <vector>
#include <deque>

#define OBS1 0		// DC_BLACK | AC_FLAT
#define OBS2 1		// DC_BLACK | AC_EDGE
#define OBS3 2		// DC_GRAY  | AC_FLAT
#define OBS4 3		// DC_GRAY  | AC_EDGE
#define OBS5 4		// DC_WHITE | AC_FLAT
#define OBS6 5		// DC_WHITE | AC_EDGE

#define AC_FLAT_2_EDGE 1	// offset to change AC_FLAT to AC_EDGE in current observation
#define AC_STDDEV 25		// threshold for standard deviation

#define VIT_OBS_MAX 3		// describes how many last observation are used for viterbi

#define TRAIN_OBS_MAX 25	// used by Baum Welch (25 frames = 1 second)
#define TRAIN_SEQ_MAX 15	// used by Baum Welch (15 seconds in total)
#define TRAIN_MAX_ITER 3	// Baum Welch stop criteria (max_int = 2147483647)


class sp_hmm {
	public:
		int hmm_init(Dimensions &dim);
		cv::Mat hmm_exec(cv::Mat);
		sp_hmm();
		~sp_hmm();
	private:
		int num_of_col;
		int num_of_row;
		int train_obs;
		int train_seq;
		int obs_max;
		int seq_max;
		int block_r;
		int block_c;
		int seq_num;
		int obs_num;
		int max_iter;
		CvHMM *hmm;
		//cv::Mat train_seq;
		cv::Mat TRGUESS;
		cv::Mat EMITGUESS;
		cv::Mat INITGUESS;
		//new stuff:

		// create container, which will remember some observatins for a given block to enable viterbi,
		// container with Mat elements describing the HMM for a given Block,
		// and a Matrix containing training sequences for Baum Welch with ALL observations
		std::vector< std::vector < std::deque<int> > > obs_matrix;
		std::vector< std::vector < cv::Mat > > trans_matrix;
		std::vector< std::vector < cv::Mat > > emit_matrix;
		std::vector< std::vector < cv::Mat > > init_matrix;
		std::vector< std::vector < cv::Mat > > train_matrix;
};
