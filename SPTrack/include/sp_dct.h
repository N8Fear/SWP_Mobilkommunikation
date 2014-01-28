#pragma once

#include <opencv2/opencv.hpp>
#include "container.h"

enum DCT_Trend {
	Up,
	Down,
	Same,
};

class sp_dct {
	public:
	//	enum DCT_Trend get_trend();
		sp_dct();
		cv::Mat dct_exec(cv::Mat input);
		int dct_init(Dimensions &dim);
	private:
		int height_offset, width_offset;
		cv::Mat DCT_store[5];
		cv::Mat *actual;
};
