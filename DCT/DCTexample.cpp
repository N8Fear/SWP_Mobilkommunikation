// Execute with one or two arguments:
//	* Path of the input video (a String)
//	* Optional: seconds skipped from the start of the video


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

using namespace cv;
using namespace std;
using namespace gpu;

#define BLACK 0
#define GRAY 133
#define WHITE 255

#define OBS1 0
#define OBS2 1
#define OBS3 2

int main(int argc, char* argv[]) {

	long skip = 0;
	// argument checking
	switch (argc){
		case 2:
			cout << "Play whole file..." << endl;
			break;
		case 3:
			skip = (long)atoi(argv[2]);
			cout << "Skip " << skip << "seconds..." << endl;
			break;
		default:
		cout << "Syntax: "<< argv[0] <<" <filename> [<skipped seconds from start of video>]" << endl;
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
	( (mod=(width % 8)) == 0) ? width_offset = 0 : width_offset =
		blocksize-mod;
	( (mod=(heigth % 8)) == 0) ? heigth_offset = 0 : heigth_offset =
		blocksize-mod;

	// create a window for playback and DCT
	namedWindow("MyPlayback", CV_WINDOW_AUTOSIZE);
	namedWindow("MyDCT", CV_WINDOW_AUTOSIZE);

	// histogramm of used gray values
	unsigned long long int histogramm [256] = {0};

	// CvHMM training sequences variables and GUESS data
	int obs_max = 25;			// 25 frames
	int seq_max = 20;			// 20 secs
	int block_r = 33 * blocksize;		// 264
	int block_c = 61 * blocksize;		// 488
	int seq_num, obs_num = 0;
	int max_iter = 2147483647;

	CvHMM hmm;
	cv::Mat train_seq = Mat(seq_max, obs_max, CV_32S);

	double TRGUESSdata[] = { 0.95 , 0.05,	// background
				 0.8 , 0.2 };	// foreground
	cv::Mat TRGUESS = cv::Mat(2,2, CV_64F, TRGUESSdata);

	double EMITGUESSdata[] = { 0.85 , 0.1 , 0.05 ,
				   0.7 , 0.15 , 0.15 };
	cv::Mat EMITGUESS = cv::Mat(2,3, CV_64F, EMITGUESSdata);

	double INITGUESSdata[] = { 0.95 , 0.05 };
	cv::Mat INITGUESS = cv::Mat(1,2, CV_64F, INITGUESSdata);

	while(1) {
		// read new frame from video, stop playback on failure
		Mat frame;
		if (skip > 0){
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip*fps;i++)
				cap.read(frame);
			skip=0;
		}
		if (!cap.read(frame)) {
			cerr << "Cannot read the frame from video file" << endl;
			break;
		}

		// create gray snapshot of the current frame (RGB -> GRAY)
		// should not influence img data or quality on IR material
		Mat gray_img;
		cvtColor(frame, gray_img, CV_RGB2GRAY);

		// make sure both image dimensions are multiple of 2 / blocksize
		Mat dim_img;
		copyMakeBorder(gray_img, dim_img, 0, heigth_offset, 0,
				width_offset, IPL_BORDER_REPLICATE);

		// grayscale image is 8bits per pixel, but dct() requires float
		Mat float_img = Mat(dim_img.rows, dim_img.cols, CV_64F);
		dim_img.convertTo(float_img, CV_64F);

		// let's do the DCT now: image => frequencies
		// select eveery 8x8 bock of the image
		Mat dct_img = float_img.clone();

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

			// division by 8 ensures uint_8 range of DC
			double dc = block.at<double>(0,0)/8;

			// set one value for all pixels in block
			for (int i=0; i<blocksize; ++i){
				for (int j=0; j<blocksize; ++j) {

					if (dc < 90) { //bg
						block.at<double>(i,j) = BLACK;
					}
					else if (dc > 200) { //human
						block.at<double>(i,j) = WHITE;
					} else {
						block.at<double>(i,j) = GRAY;
					}
				}
			 }
		}

		// matrice contains real / complex parts, filter them seperatly
		// see: http://stackoverflow.com/questions/8059989/
		// just convert back to 8 bits per pixel
		dct_img.convertTo(dct_img, CV_8UC1);

		// update histrogramm information
		for (int r = 0; r < dct_img.rows; r += blocksize)
		for (int c = 0; c < dct_img.cols; c += blocksize) {

			histogramm[dct_img.at<uint8_t>(r,c)] += 1;
			if (histogramm[dct_img.at<uint8_t>(r,c)] == 0)
				cerr << "INTEGER OVERFLOW @HISTROGRAMM" << endl;
		}

		// set current observation value for blocks histrogramm value
		if (dct_img.at<uint8_t>(block_r, block_c) == BLACK)
			train_seq.at<int>(seq_num, obs_num) = OBS1;
		else if (dct_img.at<uint8_t>(block_r, block_c) == GRAY)
			train_seq.at<int>(seq_num, obs_num) = OBS2;
		else if (dct_img.at<uint8_t>(block_r, block_c) == WHITE)
			train_seq.at<int>(seq_num, obs_num) = OBS3;
		else
			break;

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
				hmm.train(train_seq, max_iter,
						TRGUESS, EMITGUESS, INITGUESS);
				cout << endl << "====== Result =======" << endl;
				hmm.printModel(TRGUESS, EMITGUESS, INITGUESS);
				break;
			}
		}

		// mark the block which is used for training
		for (int i=0; i<blocksize; ++i){
			for (int j=0; j<blocksize; ++j) {
				dct_img.at<uint8_t>(block_r+i, block_c+j) =
					WHITE;
			}
		}

		// show results
		imshow("MyPlayback", frame);
		imshow("MyDCT", dct_img);

		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if(waitKey(30) == 27) {
			for (int i=0; i<256; i++){
				printf("%d;%llu;\n", i, histogramm[i]);
			}
			break;
		}
	}
	return 0;
}
