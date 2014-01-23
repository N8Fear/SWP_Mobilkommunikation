#pragma once

#include <opencv2/opencv.hpp>

/*
 * frame_container data type:
 * stores 26 frames of video: 1 sec of pictures in slot 0-24 and the final
 * picture that will be worked on in slot 25
 */
class frame_container{
	private:
		cv::Mat container[26];
		cv::Mat *current;

	public:
		frame_container();
		cv::Mat *get_output();
		cv::Mat *update_current();
		cv::Mat *output;
};
