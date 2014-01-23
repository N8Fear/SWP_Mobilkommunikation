#include "sp_player.h"


using namespace cv;
using namespace std;

sp_player::sp_player(const char *name)
{
	namedWindow(name, CV_WINDOW_AUTOSIZE&CV_GUI_NORMAL);
}

int sp_player::update_player(Mat frame)
{
	imshow("Player", frame);
	return 0;
}

