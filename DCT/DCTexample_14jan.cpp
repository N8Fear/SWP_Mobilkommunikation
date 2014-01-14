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

using namespace cv;
using namespace std;
using namespace gpu;

#define BLACK 0
#define GRAY 133
#define WHITE 255

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
	( (mod=(width % 8)) == 0) ? width_offset = 0 : width_offset = 
		blocksize-mod;
	( (mod=(heigth % 8)) == 0) ? heigth_offset = 0 : heigth_offset =
		blocksize-mod;

	// create a window for playback and DCT
	namedWindow("MyPlayback", CV_WINDOW_AUTOSIZE);
	namedWindow("DC-AC", CV_WINDOW_AUTOSIZE);
	namedWindow("DC-AC2", CV_WINDOW_AUTOSIZE);
	
	int frame_counter = 0;

	while(1) { 
// 		frame_counter++;
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

			// division by 8 ensures uint_8 range of DC
			double dc = block.at<double>(0,0)/8;

			// update standard deviation and whity offset counter
			double whity_counter = 0;
			double standard_deviation = 0;
			for (int i=0; i<blocksize; ++i)
		 	for (int j=0; j<blocksize; ++j) {

				if (!((i==0)&&(j==0))){ //EXCLUDE DC
					
					standard_deviation +=
					pow(block.at<double>(i,j), 2);
					
					if (block.at<double>(i,j)/8 + dc > 180)
						whity_counter++;
				}
			}
			standard_deviation = sqrt (standard_deviation/63);

			// mark background / not moving objects black
			//  * background = back, gray and not moving
			//  * foreground = white, gray and moving
			// TODO adjust values locally for every block
			int mode = 0; // 1 = ignore dc, 0 = use dc + ac
   			if (dc < 50 && mode == 0) { //bg
	 			for (int i=0; i<blocksize; ++i)
				for (int j=0; j<blocksize; ++j) {

					output_img.at<uint8_t>(r+i, c+j) = BLACK;
					output_img2.at<uint8_t>(r+i, c+j) = BLACK;
				}
			}
			else if (dc > 200 && mode == 0) { 
	 			// human do nothhing
			} 
			else {//grey
	 			//if (whity_counter < 5){
				//	for (int i=0; i<blocksize; ++i)
				//	for (int j=0; j<blocksize; ++j) {

				//		output_img.at<uint8_t>(r+i, c+j) = BLACK;
				//	}
				//}
	 			if (standard_deviation < AC_STDDEV) {

					if (whity_counter < 5){
						for (int i=0; i<blocksize; ++i)
						for (int j=0; j<blocksize; ++j) {

							output_img.at<uint8_t>(r+i, c+j) = BLACK;
						}
					}
					for (int i=0; i<blocksize; ++i)
					for (int j=0; j<blocksize; ++j) {
						output_img2.at<uint8_t>(r+i, c+j) = BLACK;
					}
				}
			}
			// code below does not substract bg as good as above
//			if (standard_deviation < AC_STDDEV){
//				for (int i=0; i<blocksize; ++i)
//				for (int j=0; j<blocksize; ++j) {
//
//					output_img2.at<uint8_t>(r+i, c+j) = BLACK;
//				}
//			}
//			if (dc < 50 && mode == 0) { //bg
//	 			for (int i=0; i<blocksize; ++i)
//				for (int j=0; j<blocksize; ++j) {
//
//					output_img2.at<uint8_t>(r+i, c+j) = BLACK;
//				}
//			}
		}

		// show results
		imshow("MyPlayback", frame);
		imshow("DC-AC", output_img);
		imshow("DC-AC2", output_img2);

		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if(waitKey(30) == 27) {
			break;
		}
	}
	return 0;
}
