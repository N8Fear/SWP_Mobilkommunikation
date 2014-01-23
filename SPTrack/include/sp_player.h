#pragma once

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

class sp_player{
	private:

	public:
		int update_player(cv::Mat frame);
		sp_player(const char *name);
};
