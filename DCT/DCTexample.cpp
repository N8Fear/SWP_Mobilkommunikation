// Execute with one argument:
// 	* Path of the input video (a String)

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

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
using namespace gpu;

int main(int argc, char* argv[]) {

	// argument checking
	if(argc != 2) {
		cout << "You need to supply one argument!" << endl;
		return -1;
	}
	
	// open the video file for reading
	VideoCapture cap(argv[1]); 
	if (!cap.isOpened()) {
	
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	cout << "Initiate playback and DCT of " << argv[1] << endl;

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
	
	while(1) { 
		// read new frame from video, stop playback on failure
		Mat frame;
		if (!cap.read(frame)) { 
			cout << "Cannot read the frame from video file" << endl;
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
				Mat block = dct_img(Rect(c, r, blocksize,
							blocksize));
				vector<Mat> planes;
				split(block, planes);
				vector<Mat> outplanes(planes.size());
			
				// note: it seems that only one plane exist, so
				// loop might me redundant
				for (size_t k = 0; k < planes.size(); k++) {
					dct(planes[k], outplanes[k]);
				}
				merge(outplanes, block);

				// double dc = block.at<double>(0,0);
				// double dc2 = dct_img.at<double>(r,c);
				// cout << dc << "	" << dc2 << endl;

				// for (int i=0; i<8; ++i){
				// 	for (int j=0; j<8; ++j) {

				// 		block.at<double>(i,j) = dc;
				// 	}
				// }

				// cout << "BLOCK " << c << "|" << r << endl 
				//	<< block << endl << endl;
			}
		// break;
	
		// matrice contains real / complex parts, filter them seperatly
		// see: http://stackoverflow.com/questions/8059989/ 
		// just convert back to 8 bits per pixel
		// dct_img.convertTo(dct_img, CV_8UC1);

		// show results
		imshow("MyPlayback", frame);
		imshow("MyDCT", dct_img);
		
		// wait for 'esc' key press for 30 ms -- exit on 'esc' key
		if(waitKey(30) == 27) {
			cout << "Escape was pressed by user" << endl;
			break;
		}
	}
	return 0;
}