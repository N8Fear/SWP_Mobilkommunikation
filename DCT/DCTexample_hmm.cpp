// Execute with one argument:
// 	* Path of the input video (a String)


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

#define OBS_MAX 10
#define FLAT_2_EDGE 1
#define AC_STDDEV 25

int main(int argc, char* argv[]) {

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
	( (mod=(width % 8)) == 0) ? width_offset = 0 : width_offset = blocksize-mod;
	( (mod=(heigth % 8)) == 0) ? heigth_offset = 0 : heigth_offset = blocksize-mod;

	// create a window for playback and DCT
	namedWindow("MyPlayback", CV_WINDOW_AUTOSIZE);
	namedWindow("DC-AC", CV_WINDOW_AUTOSIZE);

	// CvHMM training sequences variables and GUESS data
	int seq_max = 20;				// 20 secs
	int block_r = 33 * blocksize;		// 264
	int block_c = 61 * blocksize;		// 488
	int seq_num = 0;
	int obs_num = 0;
	int max_iter = 2147483647;			//Baum Welche stop criteria
	
	CvHMM hmm;
	cv::Mat train_seq = Mat(seq_max, OBS_MAX, CV_32S);

	double TRGUESSdata[] = { 	9/10.0 , 1/10.0,	// "background" ?
				 				3/10.0 , 7/10.0 };	// "foreground" ?
	cv::Mat TRGUESS = cv::Mat(2,2, CV_64F, TRGUESSdata);
	
	double EMITGUESSdata[] = { 	8/20.0 , 5/20.0 , 3/20.0 , 1/20.0 , 2/20.0 , 1/20.0 ,
				   				1/20.0 , 3/20.0 , 2/20.0 , 5/20.0 , 4/20.0 , 5/20.0 };
	cv::Mat EMITGUESS = cv::Mat(2,6, CV_64F, EMITGUESSdata);
	
	double INITGUESSdata[] = { 9/10.0 , 1/10.0 };
	cv::Mat INITGUESS = cv::Mat(1,2, CV_64F, INITGUESSdata);

	// create container, which will remember all observatins for a given block
	vector < vector < deque<int> > > obs_matrix;
	// now we have an empty 2D-matrix with deques of size (0,0). 
	// resize it and init deques with 25 observations, default = obs1
	int num_of_col = (width + width_offset)/blocksize;
	int num_of_row = (heigth + heigth_offset)/blocksize;
	obs_matrix.resize(num_of_col, vector <deque<int> > (num_of_row, deque<int> (OBS_MAX, OBS1)));

	// counts frames, usually 25/second, abort criterion
	int frame_counter = 0;

	while(1) { 
 		
 		frame_counter++;
		
		Mat frame;
		if (skip > 0){
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip*fps;i++)
				cap.read(frame);
			skip=0;
		}
		// read new frame from video, stop playback on failure
		if (!cap.read(frame)) { 
			cerr << "Cannot read the frame from video file" << endl;
			break;
		}
		//TODO GAUSSIAN BLUT GOOD IDEA!?!?
		cv::GaussianBlur(frame, frame, Size(7, 7), 0, 0);

		// create gray snapshot of the current frame (RGB -> GRAY)
		// should not influence img data or quality on IR material
		Mat gray_img;
		cvtColor(frame, gray_img, CV_RGB2GRAY);
		
		// make sure both image dimensions are multiple of 2 AND blocksize 
		Mat dim_img;
		copyMakeBorder(gray_img, dim_img, 0, heigth_offset, 0, 
				width_offset, IPL_BORDER_REPLICATE);
		 
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
		
		for (int r = 0; r < dct_img.rows; r += blocksize)
		for (int c = 0; c < dct_img.cols; c += blocksize) {
				
			// For each block, split into planes, do dct,
			// and merge back into the block
			Mat block = dct_img(Rect(c, r, blocksize, blocksize));
			vector<Mat> planes;
			split(block, planes);
			vector<Mat> outplanes(planes.size());
		
			// note: it seems that only one plane exist, so
			// loop might me redundant
			for (size_t k = 0; k < planes.size(); k++) {
				dct(planes[k], outplanes[k]);
			}
			merge(outplanes, block);

		//************************ [HMM] ********************************//

		// will contain the current observation ID (OBS is 2D right now)
		int temp_OBS = -1;

		// division by 8 ensures uint_8 range of DC, encode DC in OBS
		double dc = block.at<double>(0,0)/8;

		if (dc < 50)
			temp_OBS = OBS1;
		
		else if (dc > 190)
 			temp_OBS = OBS5;
		
		else 
			temp_OBS = OBS3;

		// check standard deviation, encode AC-std-dev in OBS
		double standard_deviation = 0;
		for (int i=0; i<blocksize; ++i)
	 	for (int j=0; j<blocksize; ++j) {

			if (!((i==0)&&(j==0))){ //EXCLUDE DC
				
				standard_deviation +=
				pow(block.at<double>(i,j), 2);
			}
		}
		standard_deviation = sqrt (standard_deviation/63);

		if (standard_deviation > AC_STDDEV)
			temp_OBS += FLAT_2_EDGE;

		// remember last 25 observations (1 second), so we can perform viterbi to deduce current state
		obs_matrix[c/blocksize][r/blocksize].pop_front();
		obs_matrix[c/blocksize][r/blocksize].push_back(temp_OBS);
		//for (int i=0; i<obs_matrix[c/blocksize][r/blocksize].size(); ++i)
		//	cout << obs_matrix[c/blocksize][r/blocksize][i] << ' ';
		//cout << endl;

		// observation recognition done, now use viterbi to get most propable state
		Mat viterbi_seq = Mat(1, OBS_MAX, CV_32S);
		for (int i=0; i<obs_matrix[c/blocksize][r/blocksize].size(); ++i){
			viterbi_seq.at<int>(0,i) = obs_matrix[c/blocksize][r/blocksize][i];
		}
		cv::Mat estates;
		hmm.viterbi(viterbi_seq, TRGUESS, EMITGUESS, INITGUESS, estates);
		
		// mark the block BLACK, if state 0 (background)
		if (estates.at<int>(0,estates.cols-1) == 0)
			for (int i=0; i<blocksize; ++i)
				for (int j=0; j<blocksize; ++j) {
					output_img.at<uint8_t>(r+i, c+j) = BLACK;
				}
		}

		// [HMM LEARNING] Baum Welch starts after 25 seconds
		/*
		// save observations in training sequence
		// train_seq.at<int>(seq_num, obs_num) = // ADJUST TO CURRENT BLOCK;
		
		obs_num++;
		if (obs_num == OBS_MAX){
	
			// start new observation sequence
			obs_num = 0;
			seq_num++;

			// ... and start training if enough information
			if (seq_num == seq_max){

				cout << endl << "===== Baum-Welch-Training ======" << endl << endl << train_seq << endl;
				hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
				hmm.train(train_seq, max_iter, TRGUESS, EMITGUESS, INITGUESS);
				cout << endl << "====== B.W. - Result =======" << endl;
				hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
				break;
			}
		}
		*/

		// show results
		imshow("MyPlayback", frame);
		imshow("DC-AC", output_img);

		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if(waitKey(30) == 27) {
			break;
		}
	}
	return 0;
}

// TODO EFFIZIENZ
//		* 2dMatrix mit deques als elemente ist sau langsam
// TODO WHITY
//		* also auf OBS12 gehen, +4 addition oder so