#pragma once
#include "SPTrack.h"
#include "container.h"
#include "CvHMM/CvHMM.h"
#include "config.h"
#include <vector>
#include <deque>

class sp_hmm {
	public:
		int hmm_init(Dimensions &dim);
		cv::Mat hmm_exec(frame_container *cnt);
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
		std::vector< std::vector < std::deque<int> > > state_matrix;
		std::vector< std::vector < cv::Mat > > trans_matrix;
		std::vector< std::vector < cv::Mat > > emit_matrix;
		std::vector< std::vector < cv::Mat > > init_matrix;
		std::vector< std::vector < cv::Mat > > train_matrix;
};
