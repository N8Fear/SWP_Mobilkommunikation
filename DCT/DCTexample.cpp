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
		cout << "You need to supply one argument!";
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
	int width =cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int heigth =cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		
	// DCT requires EVEN image dimensions, so calculate offsets
	int width_offset, heigth_offset;
	(width % 2 == 0) ? width_offset = 0 : width_offset = 1;
	(heigth % 2 == 0) ? heigth_offset = 0 : heigth_offset = 1;

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
		
		// make sure both image dimensions are a multiple of 2
		Mat dim_img;
		copyMakeBorder(gray_img, dim_img, 0, heigth_offset, 0, 
				width_offset, IPL_BORDER_REPLICATE);
		 
		// grayscale image is 8bits per pixel, but dct() requires float
		Mat float_img = Mat(dim_img.rows, dim_img.cols, CV_64F);
		dim_img.convertTo(float_img, CV_64F);
		 
		// let's do the DCT now: image => frequencies
		Mat dct_img;
		dct(float_img, dct_img);
		
		// matrice contains real / complex parts, filter them seperatly
		// see: http://stackoverflow.com/questions/8059989/ 
		// TODO how to extract using magnitude() / phase() ?
//		Mat dct_real_img;
//		Mat dct_cplx_img;
//		magnitude(dct_img, dct_real_img, dct_img);

		// show the different frames in our windows
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
