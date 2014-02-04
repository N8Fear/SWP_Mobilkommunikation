#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "container.h"

class sp_player{
	private:

	public:
		int update_player(frame_container *cnt);
		sp_player(const char *name);
};
