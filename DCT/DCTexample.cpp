// Execute with:
// 	./DCTexample /path/to/video/file.avi seconds-offset /path/to/hmm/input/data/ /path/to/hmm/output/data/
//
// 	example: make && ./DCTexample /home/skyo/Desktop/DCT/ir_640x480_8.yuv.avi 15 /home/skyo/Desktop/DCT/data/input/ /home/skyo/Desktop/DCT/data/output/

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

#define AC_FLAT_2_EDGE 1 	// offset to change AC_FLAT to AC_EDGE in current observation
#define AC_STDDEV 25 		// threshold for standard deviation

#define VIT_OBS_MAX 3 		// describes how many last observation are used for viterbi

#define STATE_MAX 25 		// describes how many last states are used for time based filtering

#define TRAIN_OBS_MAX 25	// used by Baum Welch (25 frames = 1 second)
#define TRAIN_SEQ_MAX 15 	// used by Baum Welch (15 seconds in total)
#define TRAIN_MAX_ITER 3	// Baum Welch stop criteria (max_int = 2147483647)

#define DEBUG_R 0
#define DEBUG_C 5


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
		dir_input = argv[2]; // /home/skyo/Desktop/DCT/data/input/"
		dir_output = argv[3]; // /home/skyo/Desktop/DCT/data/output/"
		break;
	case 5:
		skip = (long)atoi(argv[2]); // 15
		dir_input = argv[3]; // /home/skyo/Desktop/DCT/data/input/"
		dir_output = argv[4]; // /home/skyo/Desktop/DCT/data/output/"
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
	namedWindow("DC-AC", CV_WINDOW_AUTOSIZE);

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
	state_matrix.resize(num_of_col, vector <deque<int> >(num_of_row, deque<int>(STATE_MAX, 0)));

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
		cv::GaussianBlur(frame, frame, Size(7, 7), 0, 0);

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
		Mat output_img2 = float_img.clone();
		float_img.convertTo(output_img, CV_8UC1);
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
			for (size_t k = 0; k < planes.size(); k++) {
				dct(planes[k], outplanes[k]);
			}
			merge(outplanes, block);

			// [HMM DECODING]
			// will contain the current observation ID (OBS is 2D right now)
			int temp_OBS = -1;

			// division by 8 ensures uint_8 range of DC, encode DC in OBS
			double dc = block.at<double>(0, 0) / 8;

			if (dc < 50)
				temp_OBS = OBS1;

			else if (dc > 190)
				temp_OBS = OBS5;

			else
				temp_OBS = OBS3;

			// check standard deviation, encode AC-std-dev in OBS


			//whity
			double whity_counter = 0;


			double standard_deviation = 0;
			for (int i = 0; i < blocksize; ++i)
			for (int j = 0; j<blocksize; ++j) {

				if (!((i == 0) && (j == 0))){ //EXCLUDE DC

					standard_deviation +=
						pow(block.at<double>(i, j), 2);
				}
				if (block.at<double>(i, j) / 8 + dc > 190)
					whity_counter++;

			}
			standard_deviation = sqrt(standard_deviation / 63);

			//with whity
			if ((standard_deviation > AC_STDDEV) || (whity_counter > 5))
			//without whity
				//if (standard_deviation > AC_STDDEV)
				temp_OBS += AC_FLAT_2_EDGE;

			// remember last VIT_OBS_MAX many observations, so we can perform viterbi to deduce current state
			obs_matrix[c][r].pop_front();
			obs_matrix[c][r].push_back(temp_OBS);

			// [VITERBI DEBUG INFO]
			// if (r == DEBUG_R && c == DEBUG_C) {
			//  	cout << "OBS"  << temp_OBS << " [" ;
			//  	for (int i=0; i<obs_matrix[c][r].size(); ++i){
			//  		cout << obs_matrix[c][r][i] << ", ";
			//  	}
			//  	cout << "]" << endl;
			// }

			// observation recognition done, now use viterbi to get most propable state
			Mat viterbi_seq = Mat(1, VIT_OBS_MAX, CV_32S);
			for (int i = 0; i<obs_matrix[c][r].size(); ++i){
				viterbi_seq.at<int>(0, i) = obs_matrix[c][r][i];
			}
			cv::Mat estates;
			hmm.viterbi(viterbi_seq, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r], estates);

			// mark the block BLACK, if state 0 (background)
			if (estates.at<int>(0, estates.cols - 1) == 0)
			for (int i = 0; i<blocksize; ++i)
			for (int j = 0; j<blocksize; ++j) {
				output_img.at<uint8_t>((r*blocksize) + i, (c*blocksize) + j) = BLACK;
			}

			// save the state in the HMM, current state gets init-prob. 1 in HMM, other zero
			//init_matrix[c][r].at<double>( 0, estates.at<int>(0,estates.cols-1) ) = 1;
			//init_matrix[c][r].at<double>( 0, (estates.at<int>(0,estates.cols-1)+1)%2 ) = 0;
			// [INIT STATE DEBUG INFO]
			// cout << "Current State: " << estates.at<int>(0,estates.cols-1) << endl;
			// hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);

			// [HMM LEARNING]
			// save observations in training sequence for current block
			train_matrix[c][r].at<int>(train_seq, train_obs) = temp_OBS;

			// [BAUM WELCH DEBUG INFO]
			// if (r == DEBUG_R && c == DEBUG_C) {
			//  	cout << "OBS"  << temp_OBS << "	" << train_matrix[c][r] << endl;
			// }

			state_matrix[c][r].pop_front();
			state_matrix[c][r].push_back(estates.at<int>(0, estates.cols - 1));

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

		// increment counters for Baum-Welch Training!
		train_obs++;
		if (train_obs == TRAIN_OBS_MAX){

			// start new observation sequence
			train_obs = 0;
			train_seq++;

			// ... and start training if enough information
			if (train_seq == TRAIN_SEQ_MAX){

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

		// show results
		imshow("MyPlayback", frame);
		imshow("DC-AC", output_img);

		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if (waitKey(30) == 27) {
			break;
		}
	}
	return 0;
}