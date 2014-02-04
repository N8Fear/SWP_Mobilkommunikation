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
	output = current;
	if (!cap->read(*current)) {
		cerr << "Error reading frame from stream" << endl;
		exit(0);
	}
	if (++cur > CONTAINER_SIZE-3)
		cur=0;

	return current;
}

frame_container::frame_container()
{
	cur = 0;
	preprocessed = &container[CONTAINER_SIZE-2];
	output = &container[CONTAINER_SIZE-1];
}
