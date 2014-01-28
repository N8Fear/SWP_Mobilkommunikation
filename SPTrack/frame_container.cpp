#include "container.h"
#include "SPTrack.h"

using namespace cv;
using namespace std;

Mat *frame_container::get_current()
{
	return current;
}

Mat *frame_container::process_frame(VideoCapture *cap)
{
	current = &container[cur];
	output = current;//->clone();
	if (!cap->read(*current)) {
		cerr << "Error reading frame from stream" << endl;
		exit(0);
	}
	cur = (++cur)%25;

	return current;
}

frame_container::frame_container()
{
	cur = 0;
	output = &container[25];
}
