#include "SPTrack.h"
#include "CvHMM/CvHMM.h"

using namespace std;
using namespace cv;


// CvHMM training sequences variables and GUESS data
sp_hmm::sp_hmm()
{
	double TRGUESSdata[] = { 0.95 , 0.05,	// background
				 0.8 , 0.2 };	// foreground
	double EMITGUESSdata[] = { 0.85 , 0.1 , 0.05 ,
				   0.7 , 0.15 , 0.15 };
	double INITGUESSdata[] = { 0.95 , 0.05 };

	obs_max = 25;			// 25 frames
	seq_max = 20;			// 20 secs
	block_r = 33 * BLOCKSIZE;		// 264
	block_c = 61 * BLOCKSIZE;		// 488
	seq_num = 0;
	obs_num = 0;
	max_iter = 2147483647;

	train_seq = Mat(seq_max, obs_max, CV_32S);

	TRGUESS = cv::Mat(2,2, CV_64F, TRGUESSdata);
	EMITGUESS = cv::Mat(2,3, CV_64F, EMITGUESSdata);
	INITGUESS = cv::Mat(1,2, CV_64F, INITGUESSdata);
}

	/* Init done */
//begin hmm-function
int sp_hmm::hmm_learn(Mat input)
{
	// set current observation value for blocks histrogramm value
	if (input.at<uint8_t>(block_r, block_c) == BLACK)
		train_seq.at<int>(seq_num, obs_num) = OBS1;
	else if (input.at<uint8_t>(block_r, block_c) == GRAY)
		train_seq.at<int>(seq_num, obs_num) = OBS2;
	else if (input.at<uint8_t>(block_r, block_c) == WHITE)
		train_seq.at<int>(seq_num, obs_num) = OBS3;
	else
//		break; original, therefore
		return -1;

	// increment HMM counters ...
	obs_num++;
	if (obs_num == obs_max){

		// start new observation sequence
		obs_num = 0;
		seq_num++;

		// ... and start training if enough information
		if (seq_num == seq_max){

			cout << "Starting Baum-Welch-Training with:"
				<< endl << endl << train_seq << endl;
			hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
			for (int k=0; k<1000; k++){
				hmm.train(train_seq, max_iter,
					TRGUESS, EMITGUESS, INITGUESS);
			}
			cout << endl << "====== Result =======" << endl;
			hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
//			break; original, therefore:
			return 0;
		}
	}

	// mark the block which is used for training
	for (int i=0; i<BLOCKSIZE; ++i){
		for (int j=0; j<BLOCKSIZE; ++j) {
			input.at<uint8_t>(block_r+i, block_c+j) =
				WHITE;
		}
	}
	return 0;
}
