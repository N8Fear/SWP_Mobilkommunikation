#include "frame_container.h"
#include "SPTrack.h"

using namespace cv;

Mat *frame_container::get_output()
{


Mat *frame_container::process_frame()
{

}

frame_container::frame_container()
{
	current = container[0];
	output = container[25];
}
