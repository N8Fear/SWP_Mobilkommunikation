#pragma once

#include <opencv2/opencv.hpp>

enum DCT_Trend {
	Up,
	Down,
	Same,
};

class sp_dct {
	public:
	//	enum DCT_Trend get_trend();
		sp_dct();
		cv::Mat exec_dct(cv::Mat input);
		int init_dct(int heigth_offset, int width_offset);
	private:
		int height_offset, width_offset;
		cv::Mat DCT_store[5];
		cv::Mat *actual;
};
