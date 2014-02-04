#pragma once

#include <opencv2/opencv.hpp>
#include "config.h"

/*
 * frame_container data type:
 * stores 26 frames of video: 1 sec of pictures in slot 0-24 and the final
 * picture that will be worked on in slot 25
 */
class frame_container{
	private:
		int cur;
		cv::Mat container[CONTAINER_SIZE];
		cv::Mat *current;

	public:
		frame_container();
		cv::Mat *get_current();
		cv::Mat *process_frame(cv::VideoCapture *cap);
		cv::Mat *preprocessed;
		cv::Mat *output;
};

struct Dimensions{
		int width, height, width_offset, height_offset;
};
