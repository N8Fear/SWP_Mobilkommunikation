#pragma once

#include <opencv2/opencv.hpp>
#include "container.h"

class sp_dct {
	public:
		sp_dct();
		cv::Mat dct_exec(frame_container *cnt);
		int dct_init(Dimensions &dim);
	private:
		int height_offset, width_offset;
};
