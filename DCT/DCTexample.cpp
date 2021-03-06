#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/background_segm.hpp"

#include "CvHMM.h"

#include <time.h>
#include <math.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <numeric> 

using namespace cv;
using namespace std;
using namespace gpu;

#define BLACK 0
#define GRAY 133
#define WHITE 255

#define OBS1 0		// DC_BLACK | AC_FLAT
#define OBS2 1 		// DC_BLACK | AC_EDGE
#define OBS3 2 		// DC_GRAY  | AC_FLAT
#define OBS4 3 		// DC_GRAY  | AC_EDGE
#define OBS5 4 		// DC_WHITE | AC_FLAT
#define OBS6 5 		// DC_WHITE | AC_EDGE

#define OBS_NUM	6	// NUMBER of observations [!]

#define DC_BLACK_THRESHOLD 190	// every colour value below is marked as DC_BLACK [!]
#define DC_WHITE_THRESHOLD 230	// every colour value aobe ist marked as DC_WHITE [!]

#define AC_FLAT_2_EDGE 1 	// offset to change AC_FLAT to AC_EDGE in current observation
#define AC_STDDEV 25 		// threshold for standard deviation

#define VIT_OBS_MAX 3 		// describes how many last observation are used for viterbi/decode()

#define WHITY_THESHOLD 170	// threshold, every AC pixel above threshold is whity!
#define WHITY_MAX 4			// number of whities, which are required to increment with AC_FLAT_2_EDGE

//#define TIME_FILTER_ON 0	// activates the time filter, which works based on the last states
#define FILTER_STATE_MAX 25	// describes how many last states are used for time based filtering

#define TRAIN_OBS_MAX 25	// used by Baum Welch (25 frames = 1 second)
#define TRAIN_SEQ_MAX 10 	// used by Baum Welch (10 seconds in total)
#define TRAIN_MAX_ITER 100	// Baum Welch stop criteria (max_int = 2147483647)
#define TRAIN_ITERATIVE 1	// 1 learning should happen on the fly, 0 for no live updates, output to files on escape

//#define DEBUG_R 15			// block which is used for debug printing
//#define DEBUG_C 20			// block which is used for debug printing

// #define COMPETING_MODELS 1	// set to 1, if deterministic model and HMM should compete

#define VITERBI_OR_DECODE 3	// set to 1 for viterbi, set to 2 for decode, 3 for both [!]

int main(int argc, char* argv[]) {

	string dir_input = "";
	string dir_output = "";
	long skip = 0;

	// argument checking
	switch (argc){
	case 2:
		//cout << "Play whole file..." << endl;
		break;
	case 3:
		skip = (long)atoi(argv[2]);
		//cout << "Skip " << skip << " seconds..." << endl;
		break;
	case 4:
		dir_input = argv[2]; 
		dir_output = argv[3]; 
		break;
	case 5:
		skip = (long)atoi(argv[2]);
		dir_input = argv[3];
		dir_output = argv[4];
		break;
	default:
		cout << "Syntax: " << argv[0]
			<< " <filename> [<skipped seconds from start of video>]"
			<< endl;
		exit(-5);
	}

	// open the video file for reading
	VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {

		cerr << "Cannot open the video file" << endl;
		return -1;
	}

	// get window size
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int heigth = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	// DCT requires EVEN image dimensions, so calculate offsets
	// however we use blocks of 8x8 pixel (which are even)
	int mod, width_offset, heigth_offset;
	int blocksize = 8;
	((mod = (width % 8)) == 0) ? width_offset = 0 : width_offset = blocksize - mod;
	((mod = (heigth % 8)) == 0) ? heigth_offset = 0 : heigth_offset = blocksize - mod;

	// create a window for playback and DCT
	namedWindow("MyPlayback", CV_WINDOW_AUTOSIZE);
	if (VITERBI_OR_DECODE == 1 || VITERBI_OR_DECODE == 3) namedWindow("Viterbi", CV_WINDOW_AUTOSIZE);
	if (VITERBI_OR_DECODE == 2 || VITERBI_OR_DECODE == 3) namedWindow("Decode", CV_WINDOW_AUTOSIZE);

	int train_seq = 0;					// Baum Welch counter - sequences
	int train_obs = 0;					// Baum Welch counter - observation (duration of sequence)

	CvHMM hmm;							// needed for some static function calls

	// create container, which will remember some observatins for a given block to enable viterbi,
	// container with Mat elements describing the HMM for a given Block,
	// and a Matrix containing training sequences for Baum Welch with ALL observations
	vector < vector < deque<int> > > obs_matrix;
	vector < vector < deque<int> > > state_matrix;
	vector < vector < Mat > > trans_matrix;
	vector < vector < Mat > > emit_matrix;
	vector < vector < Mat > > init_matrix;
	vector < vector < Mat > > train_matrix;

	// now we have an empty 2D-matrix with deques of size (0,0) and empty Matrix. 
	// resize it and init deques with 25 observations, default = obs1
	// Matrix should be initiated with the default array or data read from learn-file
	int num_of_col = (width + width_offset) / blocksize;
	int num_of_row = (heigth + heigth_offset) / blocksize;

	obs_matrix.resize(num_of_col, vector <deque<int> >(num_of_row, deque<int>(VIT_OBS_MAX, OBS1)));
	state_matrix.resize(num_of_col, vector <deque<int> >(num_of_row, deque<int>(FILTER_STATE_MAX, 0)));

	for (int i = 0; i<num_of_col; i++) {

		trans_matrix.push_back(vector<Mat>());
		emit_matrix.push_back(vector<Mat>());
		init_matrix.push_back(vector<Mat>());
		train_matrix.push_back(vector<Mat>());

		for (int j = 0; j<num_of_row; j++){

			trans_matrix[i].push_back(Mat(2, 2, CV_64F, Scalar::all(0.0)));
			emit_matrix[i].push_back(Mat(2, 6, CV_64F, Scalar::all(0.0)));
			init_matrix[i].push_back(Mat(1, 2, CV_64F, Scalar::all(0.0)));
			train_matrix[i].push_back(Mat(TRAIN_SEQ_MAX, TRAIN_OBS_MAX, CV_32S, Scalar::all(0)));
		}
	}

	// [VECTOR INIT DEBUG INFO]
	// hmm.printModel(trans_matrix[DEBUG_C][DEBUG_R], emit_matrix[DEBUG_C][DEBUG_R], init_matrix[DEBUG_C][DEBUG_R]);

	// read 2d hmm vectors from file
	ostringstream fname_emit_i;
	ostringstream fname_init_i;
	ostringstream fname_trans_i;
	fname_emit_i << dir_input << "emit2D.yml";
	fname_init_i << dir_input << "init2D.yml";
	fname_trans_i << dir_input << "trans2D.yml";
	FileStorage file_emit(fname_emit_i.str(), FileStorage::READ);
	FileStorage file_init(fname_init_i.str(), FileStorage::READ);
	FileStorage file_trans(fname_trans_i.str(), FileStorage::READ);
	for (int r = 0; r < num_of_row; r++)
	for (int c = 0; c < num_of_col; c++) {
		ostringstream id_stream;
		id_stream << "row_" << r << "-col_" << c;
		string ID = id_stream.str();
		file_emit[ID] >> emit_matrix[c][r];
		file_init[ID] >> init_matrix[c][r];
		file_trans[ID] >> trans_matrix[c][r];
	}
	file_emit.release();
	file_init.release();
	file_trans.release();

	// [VECTOR INIT DEBUG INFO2]
	// hmm.printModel(trans_matrix[DEBUG_C][DEBUG_R], emit_matrix[DEBUG_C][DEBUG_R], init_matrix[DEBUG_C][DEBUG_R]);

	// always seed your RNG before using it [!]
	srand(time(NULL)); 

	while (1) {

		// skip frames
		Mat frame;
		if (skip > 0){
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip*fps; i++)
				cap.read(frame);
			skip = 0;
		}

		// read new frame from video, stop playback on failure
		if (!cap.read(frame)) {
			cerr << "Cannot read the frame from video file" << endl;
			break;
		}

		// blur image to reduce false detection of edges
		// cv::GaussianBlur(frame, frame, Size(7, 7), 0, 0); [!]

		// create gray snapshot of the current frame (RGB -> GRAY)
		// should not influence img data or quality on IR material
		Mat gray_img;
		cvtColor(frame, gray_img, CV_RGB2GRAY);

		// make sure both image dimensions are multiple of 2 AND blocksize 
		Mat dim_img;
		copyMakeBorder(gray_img, dim_img, 0, heigth_offset, 0, width_offset, IPL_BORDER_REPLICATE);

		// grayscale image is 8bits per pixel, but dct() requires float
		Mat float_img = Mat(dim_img.rows, dim_img.cols, CV_64F);
		dim_img.convertTo(float_img, CV_64F);

		// let's do the DCT now: image => frequencies
		// select eveery 8x8 bock of the image
		Mat dct_img = float_img.clone();
		Mat output_img = float_img.clone();
		float_img.convertTo(output_img, CV_8UC1);		
		Mat output_img2 = float_img.clone();
		float_img.convertTo(output_img2, CV_8UC1);

		for (int r = 0; r < num_of_row; r++)
		for (int c = 0; c < num_of_col; c++) {

			// For each block, split into planes, do dct,
			// and merge back into the block
			Mat block = dct_img(Rect(c*blocksize, r*blocksize, blocksize, blocksize));
			vector<Mat> planes;
			split(block, planes);
			vector<Mat> outplanes(planes.size());

			// note: it seems that only one plane exist, so
			// loop might me redundant
			for (size_t k = 0; k < planes.size(); k++)
				dct(planes[k], outplanes[k]);
			merge(outplanes, block);

			// [HMM DECODING]
			// will contain the current observation ID (OBS is 2D right now)
			int temp_OBS = -1;

			// division by 8 ensures uint_8 range of DC, encode DC in OBS
			double dc = block.at<double>(0, 0) / 8;

			if (dc < DC_BLACK_THRESHOLD)
				temp_OBS = OBS1;

			else if (dc > DC_WHITE_THRESHOLD)
				temp_OBS = OBS5;

			else 
				temp_OBS = OBS3;
				

			// check standard deviation, encode AC-std-dev and whities in OBS
			double whity_counter = 0;
			double standard_deviation = 0;
			for (int i = 0; i < blocksize; ++i)
			for (int j = 0; j < blocksize; ++j) {

				if (!((i == 0) && (j == 0))){ //EXCLUDE DC

					standard_deviation += pow(block.at<double>(i, j), 2);
					
					if (block.at<double>(i, j) / 8 + dc > WHITY_THESHOLD)
						whity_counter++;
				}
			}
			standard_deviation = sqrt(standard_deviation / 63);

			// with whity or standard deviation
			if ((standard_deviation > AC_STDDEV) || (whity_counter > WHITY_MAX))
				temp_OBS += AC_FLAT_2_EDGE;

/*
			// remember last VIT_OBS_MAX many observations, so we can perform viterbi to deduce current state
			obs_matrix[c][r].pop_front();
			obs_matrix[c][r].push_back(temp_OBS);

			// observation recognition done, now use viterbi to get most propable state
			Mat viterbi_seq = Mat(1, VIT_OBS_MAX, CV_32S);
			for (int i = 0; i<obs_matrix[c][r].size(); ++i){
				viterbi_seq.at<int>(0, i) = obs_matrix[c][r][i];
			}
			cv::Mat estates;
			hmm.viterbi(viterbi_seq, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r], estates);
			
			// get current states for HMM and deterministic model
			int current_hmm_state = estates.at<int>(0, estates.cols - 1);
*/

			//****************** OBS GUESS and DETM COMPARISON *************** [!]

			int most_likely_obs = 0;
			int most_likely_obs2 = 0;

			if (VITERBI_OR_DECODE == 2 || VITERBI_OR_DECODE == 3) { // DECODE()

				// prepare seqs to decode
				// arg1: as many rows as possible observations 
	 			// arg2: as many columns as states we remember with VIT_OBS_MAX plus new guessed state
				Mat decode_seqs = Mat(OBS_NUM, VIT_OBS_MAX+1, CV_32S);		
				double max_logpseq = -1000000.0;

				for (int j = 0; j<OBS_NUM; j++){

					// transfer the last seen observations
					for (int i = 0; i<VIT_OBS_MAX; i++)
						decode_seqs.at<int>(j, i) = obs_matrix[c][r][i];
				
					// set now guessed state in last cell
					decode_seqs.at<int>(j, VIT_OBS_MAX) = j; 
				
					// decode current rows probability in logscale
					cv::Mat pstates,forward,backward;
					double logpseq = 0.0;
					hmm.decode(decode_seqs.row(j), trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r], logpseq, pstates, forward, backward);

					// the higher the more probable, save most probable emission row
					if (logpseq > max_logpseq){
						max_logpseq = logpseq;
						most_likely_obs = j;
					}
				}

				if (VITERBI_OR_DECODE >= 2){
					// remember last VIT_OBS_MAX many observations, so we can perform viterbi / decode to deduce current state
					// WATCH OUT: Has to be performed AFTER Decode() but BEFORE Viterbi()
					obs_matrix[c][r].pop_front();
					obs_matrix[c][r].push_back(temp_OBS);
				}

				if (temp_OBS == most_likely_obs){ // decode

					for (int i = 0; i<blocksize; ++i)
						for (int j = 0; j<blocksize; ++j) {
							output_img2.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
						}
				}
			}


			if (VITERBI_OR_DECODE == 1 || VITERBI_OR_DECODE == 3 ) { //VITERBI()


				if (VITERBI_OR_DECODE == 1){
					// remember last VIT_OBS_MAX many observations, so we can perform viterbi / decode to deduce current state
					// WATCH OUT: Has to be performed AFTER Decode() but BEFORE Viterbi()
					obs_matrix[c][r].pop_front();
					obs_matrix[c][r].push_back(temp_OBS);
				}

				// observation recognition done, now use viterbi to get most propable state
				Mat viterbi_seq = Mat(1, VIT_OBS_MAX, CV_32S);
				for (int i = 0; i<obs_matrix[c][r].size(); ++i){
					viterbi_seq.at<int>(0, i) = obs_matrix[c][r][i];
				}
				cv::Mat estates;
				hmm.viterbi(viterbi_seq, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r], estates);
				int current_hmm_state = estates.at<int>(0, estates.cols - 1);

				// now get the emission probabilities for current state
				// prepare rolling the dice with them by creating cummulative sum
				double cum_sum[OBS_NUM+1] = {0.0};
				for (int cs=1; cs<OBS_NUM+1; cs++){
					cum_sum[cs] = cum_sum[cs-1] + emit_matrix[c][r].at<double>(current_hmm_state,cs-1);
				}
				
				// roll the dice
				double prob = (double) rand() / (RAND_MAX);
				for (int dice = 1; dice<OBS_NUM+1; ++dice){

					if ( cum_sum[dice-1] <= prob && prob < cum_sum[dice] ){

						most_likely_obs2 = dice-1;
						break;
					}
				}

				// filter if HMM-guess (background) resembles the current observations
				if (temp_OBS == most_likely_obs2){ // viterbi

					for (int i = 0; i<blocksize; ++i)
						for (int j = 0; j<blocksize; ++j) {
							output_img.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
						}
				}
			}

			//****************** OBS GUESS and DETM COMPARISON *************** [!]

/*
			int det_state = 0;
			if (temp_OBS >= OBS4) det_state = 1;


			// remove background based on filter-rule
			if (COMPETING_MODELS == 1){
				
				// determenistic model VS HMM: mark the block BLACK, if hmm-state and det-state are the same
				// HMM learns the background, so if they differ, sth has appeared in the FG and should be NOT filtered
				// TODO: create best if statement... 
				if (current_hmm_state == det_state || current_hmm_state == 1 && det_state == 0)
				// if (current_hmm_state == det_state)
					for (int i = 0; i<blocksize; ++i)
					for (int j = 0; j<blocksize; ++j) {
						output_img.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
				}
			}
			else {
				// filter based on the HMM: mark the block BLACK, if state 0 (background)
				if (current_hmm_state == 0)
				for (int i = 0; i<blocksize; ++i)
				for (int j = 0; j<blocksize; ++j) {
					output_img.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
				}
			}
*/

			// [HMM LEARNING]
			// save observations in training sequence for current block
			train_matrix[c][r].at<int>(train_seq, train_obs) = temp_OBS;

			// [BAUM WELCH DEBUG INFO]
			// if (r == DEBUG_R && c == DEBUG_C) {
			//  	cout << "OBS"  << temp_OBS << "	" << train_matrix[c][r] << endl;
			// }

/*
			// deterministic time filter based on last states
			if (TIME_FILTER_ON == 1){
				state_matrix[c][r].pop_front();
				state_matrix[c][r].push_back(current_hmm_state);

				for (int k = 0; k<state_matrix[c][r].size(); k++){

					if (state_matrix[c][r][k] == 0) break;

					if (state_matrix[c][r].size() - 1 == k){

						// mark black, as no movement
						for (int i = 0; i<blocksize; ++i)
						for (int j = 0; j<blocksize; ++j) {
							output_img.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
						}
					}
				}
			}
*/
		}

		// increment counters for Baum-Welch Training!
		train_obs++;
		if (train_obs == TRAIN_OBS_MAX){

			// start new observation sequence
			train_obs = 0;
			train_seq++;

			// ... and start training if enough information
			if (train_seq == TRAIN_SEQ_MAX){

				if (TRAIN_ITERATIVE == 1) {

					cout << "Updating HMM..." << endl;
					for (int r = 0; r < num_of_row; r++)
					for (int c = 0; c < num_of_col; c++) {

						// [BAUM WELCH DEBUG INFO 2]
						// if (r == DEBUG_R && c == DEBUG_R) {

						// 	hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						// 	cout << "------------------------------------------" << endl;
						// 	hmm.train(train_matrix[c][r], TRAIN_MAX_ITER, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						// 	hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						// 	cout << "==========================================" << endl;

						// }
						// else
						hmm.train(train_matrix[c][r], TRAIN_MAX_ITER, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
					}
				}

				// begin collecting new trainining sequences
				train_seq = 0;
			}
		}

		// show results
		imshow("MyPlayback", frame);
		if (VITERBI_OR_DECODE == 1 || VITERBI_OR_DECODE == 3) imshow("Viterbi", output_img);
		if (VITERBI_OR_DECODE == 2 || VITERBI_OR_DECODE == 3) imshow("Decode", output_img2);

		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if (waitKey(30) == 27) {

			// write 2d vector into file, all adjusted HMM will be saved!
			ostringstream fname_emit_o;
			ostringstream fname_init_o;
			ostringstream fname_trans_o;
			fname_emit_o << dir_output << "emit2D.yml";
			fname_init_o << dir_output << "init2D.yml";
			fname_trans_o << dir_output << "trans2D.yml";
			FileStorage file_emit(fname_emit_o.str(), FileStorage::WRITE);
			FileStorage file_init(fname_init_o.str(), FileStorage::WRITE);
			FileStorage file_trans(fname_trans_o.str(), FileStorage::WRITE);
			for (int r = 0; r < num_of_row; r++)
			for (int c = 0; c < num_of_col; c++) {

				ostringstream id_stream;
				id_stream << "row_" << r << "-col_" << c;
				string ID = id_stream.str();
				file_emit << ID << emit_matrix[c][r];
				file_init << ID << init_matrix[c][r];
				file_trans << ID << trans_matrix[c][r];
			}
			file_emit.release();
			file_init.release();
			file_trans.release();

			break;
		}
	}
	return 0;
}