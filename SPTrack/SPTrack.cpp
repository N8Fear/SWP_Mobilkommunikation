//#include "SPTrack.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#define BLOCKSIZE 8

using namespace cv;
using namespace std;
using namespace gpu;

class SPTrack{
	private:
		long skip_sec;
		int height_offset, width_offset;
		int init_player(char *path);
		int parse_cl_param(int argc, char *argv[]);
		int init_dct();
		Mat exec_dct(Mat);
		long long int histogram[256];
		void gen_histogram(Mat);
		VideoCapture cap;

	public:
		int play_stream();
		SPTrack(int argc, char *argv[]);
};

int SPTrack::init_player(char *path)
{
	VideoCapture capture(path);
	cap = capture;
	if (!cap.isOpened()) {
		cerr << "Cannot open video stream!" << endl;
		return -1;
	}
	namedWindow("Player", CV_WINDOW_AUTOSIZE&CV_GUI_NORMAL);

	/* Initialization of sub processes and stuff: */

	init_dct();
	memset(histogram, 0, 256);
	return 0;
}

int SPTrack::play_stream()
{
	while (1) {
		Mat frame;

		if (skip_sec > 0) {
			short fps = cap.get(CV_CAP_PROP_FPS);
			for (int i = 0; i< skip_sec*fps; i++)
				cap.read(frame);
			skip_sec = 0;
		}
		if (!cap.read(frame)) {
			cerr << "Error reading frame from stream" << endl;
			break;
		}
		frame =	exec_dct(frame);
		gen_histogram(frame);
		imshow("Player", frame);
		if(waitKey(30) == 27) {
			for (int i=0; i<256; i++)
				printf("%d;%llu;\n",i,histogram[i]);
			break;
		}
	}
	return 0;
}

int SPTrack::parse_cl_param(int argc, char *argv[])
{
	switch (argc){
		case 3:
			SPTrack::skip_sec = (long) atoi (argv[2]);
		case 2:
			cout << "Skipping " << skip_sec << " seconds..." << endl;
			break;
		default:
			cout << "Syntax: " << argv[0] << " <filename> [<skipped seconds>]" <<endl;
			exit(-5);
	}
	return 0;
}

int SPTrack::init_dct()
{

	int mod;
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	((mod=(width % 8))== 0) ? width_offset = 0: width_offset = BLOCKSIZE - mod;
	((mod=(height % 8))== 0) ? height_offset = 0: height_offset = BLOCKSIZE -mod;
	return 0;
}
Mat SPTrack::exec_dct(Mat input)
{
	// create gray snapshot of the current frame (RGB -> GRAY)
	// should not influence img data or quality on IR material
	Mat gray_img;
	cvtColor(input, gray_img, CV_RGB2GRAY);

	// make sure both image dimensions are multiple of 2 / BLOCKSIZE
	Mat dim_img;
	copyMakeBorder(gray_img, dim_img, 0, height_offset, 0,
			width_offset, IPL_BORDER_REPLICATE);

	// grayscale image is 8bits per pixel, but dct() requires float
	Mat float_img = Mat(dim_img.rows, dim_img.cols, CV_64F);
	dim_img.convertTo(float_img, CV_64F);

	// let's do the DCT now: image => frequencies
	// select eveery 8x8 bock of the image
	Mat dct_img = float_img.clone();

	for (int r = 0; r < dct_img.rows; r += BLOCKSIZE)
		for (int c = 0; c < dct_img.cols; c += BLOCKSIZE) {

			// For each block, split into planes, do dct,
			// and merge back into the block
			Mat block = dct_img(Rect(c, r, BLOCKSIZE, BLOCKSIZE));
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
//			double dc = block.at<double>(0,0)/8;

			// set one value for all pixels in block
//			for (int i=0; i<BLOCKSIZE; ++i){
//				for (int j=0; j<BLOCKSIZE; ++j) {
//
//					if (dc < 90) { //bg
//						block.at<double>(i,j) = BLACK;
//					}
//					else if (dc > 200) { //human
//						block.at<double>(i,j) = WHITE;
//					} else {
//						block.at<double>(i,j) = GRAY;
//					}
//				}
//			 }
		}

	// matrice contains real / complex parts, filter them seperatly
	// see: http://stackoverflow.com/questions/8059989/
	// just convert back to 8 bits per pixel
	dct_img.convertTo(dct_img, CV_8UC1);
	return dct_img;
}

void SPTrack::gen_histogram(Mat input){
	// update histrogramm information
	for (int r = 0; r < input.rows; r += BLOCKSIZE)
		for (int c = 0; c < input.cols; c += BLOCKSIZE) {
			SPTrack::histogram[input.at<uint8_t>(r,c)] += 1;
			if (SPTrack::histogram[input.at<uint8_t>(r,c)] == 0)
				cerr << "INTEGER OVERFLOW @HISTROGRAMM" << endl;
		}
}

SPTrack::SPTrack(int argc, char *argv[])
{
	skip_sec = 0;
	parse_cl_param(argc, argv);
	init_player(argv[1]);
}

int main(int argc, char* argv[]) {
	SPTrack *spt = new SPTrack(argc, argv);
	spt->play_stream();
}
