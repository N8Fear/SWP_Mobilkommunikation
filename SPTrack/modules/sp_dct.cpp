#include <opencv2/opencv.hpp>
#include "sp_dct.h"
#include "SPTrack.h"

using namespace cv;
using namespace std;

sp_dct::sp_dct()
{
	width_offset = 0;
}

int sp_dct::dct_init(Dimensions &dim)
{
	height_offset = dim.height_offset;
	width_offset = dim.width_offset;
}

Mat sp_dct::dct_exec(frame_container *cnt)
{
	Mat input = *cnt->get_current();
	//TODO: possibly move to own module
	//blur image to reduce false detection of edges
	GaussianBlur(input, input, Size(7, 7), 0, 0);

	Mat dct_img;
	// create gray snapshot of the current frame (RGB -> GRAY)
	// should not influence img data or quality on IR material
	cvtColor(input, dct_img, CV_RGB2GRAY);

	// make sure both image dimensions are multiple of 2 / BLOCKSIZE
	copyMakeBorder(dct_img,dct_img,0,height_offset,0,width_offset,IPL_BORDER_REPLICATE);

	// grayscale image is 8bits per pixel, but dct() requires float
	dct_img.convertTo(dct_img, CV_64F);

	// let's do the DCT now: image => frequencies
	// select eveery 8x8 bock of the image

	for (int r = 0; r < dct_img.rows; r += BLOCKSIZE)
		for (int c = 0; c < dct_img.cols; c += BLOCKSIZE) {

			// For each block, split into planes, do dct,
			// and merge back into the block
			Mat block = dct_img(Rect(c, r, BLOCKSIZE, BLOCKSIZE));
/*
			// Support for multiple planes - seemingly not necessary for our
			// case
			vector<Mat> planes;
			split(block, planes);
			vector<Mat> outplanes(planes.size());

			// note: it seems that only one plane exist, so
			// loop might me redundant
			for (size_t k = 0; k < planes.size(); k++) {
				dct(planes[k], outplanes[k]);
			}
			merge(outplanes, block);
*/
			dct(block, block);
			// division by 8 ensures uint_8 range of DC
//			double dc = block.at<double>(0,0)/8;

			// set one value for all pixels in block
		}

	// matrice contains real / complex parts, filter them seperatly
	// see: http://stackoverflow.com/questions/8059989/
	// just convert back to 8 bits per pixel
//	dct_img.convertTo(dct_img, CV_8UC1);
	*cnt->preprocessed = dct_img.clone();
	return dct_img;
}
