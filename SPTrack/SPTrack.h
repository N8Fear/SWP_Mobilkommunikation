#pragma once

/*TODO: Check necissity of included headers*/

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

#include <time.h>


#define BLACK 0
#define GRAY 133
#define WHITE 255

#define BLOCKSIZE 8

/*class sp_hmm {
	public:
		int hmm_init();
		int hmm_learn(cv::Mat);
		int hmm_print();
		sp_hmm();
	private:
		int obs_max;
		int seq_max;
		int block_r;
		int block_c;
		int seq_num;
		int obs_num;
		int max_iter;
		CvHMM hmm;
		cv::Mat train_seq;
		cv::Mat TRGUESS;
		cv::Mat EMITGUESS;
		cv::Mat INITGUESS;
};
*/
