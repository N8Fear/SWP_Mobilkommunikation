#pragma once

#define CONTAINER_SIZE 26
#define BLOCKSIZE 8


#define BLACK 0
#define GRAY 133
#define WHITE 255

#define OBS1 0		// DC_BLACK | AC_FLAT
#define OBS2 1		// DC_BLACK | AC_EDGE
#define OBS3 2		// DC_GRAY  | AC_FLAT
#define OBS4 3		// DC_GRAY  | AC_EDGE
#define OBS5 4		// DC_WHITE | AC_FLAT
#define OBS6 5		// DC_WHITE | AC_EDGE

#define DC_BLACK_THRESHOLD 50	// every colour value below is marked as DC_BLACK
#define DC_WHITE_THRESHOLD 190	// every colour value aobe ist marked as DC_WHITE

#define AC_FLAT_2_EDGE 1	// offset to change AC_FLAT to AC_EDGE in current observation
#define AC_STDDEV 25		// threshold for standard deviation

#define VIT_OBS_MAX 3		// describes how many last observation are used for viterbi

#define WHITY_THRESHOLD 170	// threshold, every AC pixel above threshold is whity!
#define WHITY_MAX 4			// number of whities, which are required to set block as foreground

#define TIME_FILTER_ON 0	// activates the time filter, which works based on the last states
#define FILTER_STATE_MAX 25	// describes how many last states are used for time based filtering

#define TRAIN_OBS_MAX 25	// used by Baum Welch (25 frames = 1 second)
#define TRAIN_SEQ_MAX 10	// used by Baum Welch (10 seconds in total)
#define TRAIN_MAX_ITER 100	// Baum Welch stop criteria (max_int = 2147483647)
#define TRAIN_ITERATIVE 1	// 1 learning should happen on the fly, 0 for no live updates, output to files on escape

#define DEBUG_R 15			// block which is used for debug printing
#define DEBUG_C 20			// block which is used for debug printing

#define COMPETING_MODELS 1	// set to 1, if deterministic model and HMM should compete
