#pragma once

#include <opencv2/opencv.hpp>
#include "SPTrack.h"

enum DCT_Trend {
	Up,
	Down,
	Same,
};

class DCT {
	public:
	//	enum DCT_Trend get_trend();
		DCT();
		cv::Mat exec_dct(cv::Mat input);
		int init_dct(cv::VideoCapture *cap);
	private:
		int height_offset, width_offset;
		cv::Mat DCT_store[5];
		cv::Mat *actual;
};
