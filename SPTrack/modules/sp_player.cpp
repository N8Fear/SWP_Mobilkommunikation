#include "sp_player.h"


using namespace cv;
using namespace std;

sp_player::sp_player(const char *name)
{
//	namedWindow(name, CV_WINDOW_AUTOSIZE&CV_GUI_NORMAL);
	namedWindow(name, CV_WINDOW_AUTOSIZE);
}

int sp_player::update_player(frame_container *cnt)
{
	imshow("Player", *cnt->output);
	return 0;
}

