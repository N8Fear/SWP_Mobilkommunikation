#include "SPTrack.h"
#include "sp_hmm.h"
#include "container.h"

using namespace std;
using namespace cv;


// CvHMM training sequences variables and GUESS data
sp_hmm::sp_hmm()
{
	double TRGUESSdata[] = { 0.95 , 0.05,	// background
				 0.8 , 0.2 };	// foreground
	double EMITGUESSdata[] = { 0.85 , 0.1 , 0.05 ,
				   0.7 , 0.15 , 0.15 };
	double INITGUESSdata[] = { 0.95 , 0.05 };

	obs_max = 25;			// 25 frames
	seq_max = 20;			// 20 secs
	block_r = 33 * BLOCKSIZE;		// 264
	block_c = 61 * BLOCKSIZE;		// 488
	seq_num = 0;
	obs_num = 0;
	max_iter = 2147483647;

//	train_seq = Mat(seq_max, obs_max, CV_32S);

	TRGUESS = cv::Mat(2,2, CV_64F, TRGUESSdata);
	EMITGUESS = cv::Mat(2,3, CV_64F, EMITGUESSdata);
	INITGUESS = cv::Mat(1,2, CV_64F, INITGUESSdata);
}

	/* Init done */
//begin hmm-function
Mat sp_hmm::hmm_exec(Mat input)
{
	Mat output_img= input.clone();
	//TODO: just temporary clutch
	string dir_output="/tmp/out/";

	for (int r = 0; r < num_of_row; r++)
		for (int c = 0; c < num_of_col; c++) {
			;
			Mat block = input(Rect(c*BLOCKSIZE, r*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE));

			// [HMM DECODING]
			// will contain the current observation ID (OBS is 2D right now)
			int temp_OBS = -1;

			// division by 8 ensures uint_8 range of DC, encode DC in OBS
			double dc = block.at<double>(0,0)/8;

			if (dc < 50)
				temp_OBS = OBS1;

			else if (dc > 190)
				temp_OBS = OBS5;

			else
				temp_OBS = OBS3;

			// check standard deviation, encode AC-std-dev in OBS
			double standard_deviation = 0;
			for (int i=0; i<BLOCKSIZE; ++i)
			for (int j=0; j<BLOCKSIZE; ++j) {

				if (!((i==0)&&(j==0))){ //EXCLUDE DC

					standard_deviation +=
					pow(block.at<double>(i,j), 2);
				}
			}
			standard_deviation = sqrt (standard_deviation/63);

			if (standard_deviation > AC_STDDEV)
				temp_OBS += AC_FLAT_2_EDGE;

			// remember last VIT_OBS_MAX many observations, so we can perform viterbi to deduce current state
			obs_matrix[c][r].pop_front();
			obs_matrix[c][r].push_back(temp_OBS);

			// [VITERBI DEBUG INFO]
			// if (r == DEBUG_R && c == DEBUG_C) {
			//	cout << "OBS"  << temp_OBS << " [" ;
			//	for (int i=0; i<obs_matrix[c][r].size(); ++i){
			//		cout << obs_matrix[c][r][i] << ", ";
			//	}
			//	cout << "]" << endl;
			// }

			// observation recognition done, now use viterbi to get most propable state
			Mat viterbi_seq = Mat(1, VIT_OBS_MAX, CV_32S);
			for (int i=0; i<obs_matrix[c][r].size(); ++i){
				viterbi_seq.at<int>(0,i) = obs_matrix[c][r][i];
			}
			cv::Mat estates;
			hmm.viterbi(viterbi_seq, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r], estates);

			// mark the block BLACK, if state 0 (background)
			if (estates.at<int>(0,estates.cols-1) == 0)
				for (int i=0; i<BLOCKSIZE; ++i)
					for (int j=0; j<BLOCKSIZE; ++j) {
						output_img.at<uint8_t>((r*BLOCKSIZE)+i, (c*BLOCKSIZE)+j) = BLACK;
					}

			// save the state in the HMM, current state gets init-prob. 1 in HMM, other zero
			// init_matrix[c][r].at<double>( 0, estates.at<int>(0,estates.cols-1) ) = 1;
			// init_matrix[c][r].at<double>( 0, (estates.at<int>(0,estates.cols-1)+1)%2 ) = 0;
			// [INIT STATE DEBUG INFO]
			// cout << "Current State: " << estates.at<int>(0,estates.cols-1) << endl;
			// hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);

			// [HMM LEARNING]
			// save observations in training sequence for current block
			train_matrix[c][r].at<int>(train_seq, train_obs) = temp_OBS;

			// [BAUM WELCH DEBUG INFO]
			// if (r == DEBUG_R && c == DEBUG_C) {
			//	cout << "OBS"  << temp_OBS << "	" << train_matrix[c][r] << endl;
			// }
		}
/*
		// increment counters for Baum-Welch Training!
		train_obs++;
		if (train_obs == TRAIN_OBS_MAX){

			// start new observation sequence
			train_obs = 0;
			train_seq++;

			// ... and start training if enough information
			if (train_seq == TRAIN_SEQ_MAX){

				for (int r = 0; r < num_of_row; r++)
					for (int c = 0; c < num_of_col; c++) {

						// [BAUM WELCH DEBUG INFO 2]
						// if (r == DEBUG_R && c == DEBUG_R) {

						//	hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						//	cout << "------------------------------------------" << endl;
						//	hmm.train(train_matrix[c][r], TRAIN_MAX_ITER, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						//	hmm.printModel(trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
						//	cout << "==========================================" << endl;

						// }
						// else
							hmm.train(train_matrix[c][r], TRAIN_MAX_ITER, trans_matrix[c][r], emit_matrix[c][r], init_matrix[c][r]);
				}

				// write 2d vector into file, all adjusted HMM will be saved!
				ostringstream fname_emit_o;
				ostringstream fname_init_o;
				ostringstream fname_trans_o;
				fname_emit_o << dir_output << "emit2D.yml";
				fname_trans_o << dir_output << "trans2D.yml";
				FileStorage file_emit(fname_emit_o.str(), FileStorage::WRITE);
				FileStorage file_init(fname_init_o.str(), FileStorage::WRITE);
				FileStorage file_trans(fname_trans_o.str(), FileStorage::WRITE);
				for (int r = 0; r < num_of_row; r++)
					for (int c = 0; c < num_of_col; c++) {

						ostringstream id_stream;
						id_stream << "row_" << r << "-col_" << c;
						string ID = id_stream.str();
						file_emit << ID << emit_matrix[c][r];
						file_init << ID << init_matrix[c][r];
						file_trans << ID << trans_matrix[c][r];
				}
				file_emit.release();
				file_init.release();
				file_trans.release();
		//		break;
			}
		}
*/
	return output_img;
}


int sp_hmm::hmm_init(Dimensions &dim)
{

	string dir_input="/tmp/";
	train_seq = 0;					// Baum Welch counter - sequences
	train_obs = 0;					// Baum Welch counter - observation (duration of sequence)

	// now we have an empty 2D-matrix with deques of size (0,0) and empty Matrix.
	// resize it and init deques with 25 observations, default = obs1
	// Matrix should be initiated with the default array or data read from learn-file
	num_of_col = (dim.width + dim.width_offset)/BLOCKSIZE;
	num_of_row = (dim.height + dim.height_offset)/BLOCKSIZE;

	this->obs_matrix.resize(num_of_col, vector <deque<int> > (num_of_row, deque<int> (VIT_OBS_MAX, OBS1)));

	for (int i = 0; i<num_of_col; i++) {

		this->trans_matrix.push_back(vector<Mat>());
		this->emit_matrix.push_back(vector<Mat>());
		this->init_matrix.push_back(vector<Mat>());
		this->train_matrix.push_back(vector<Mat>());

		for (int j=0;j<num_of_row;j++){

			this->trans_matrix[i].push_back(Mat(2,2, CV_64F, Scalar::all(0.0)));
			this->emit_matrix[i].push_back(Mat(2,6, CV_64F, Scalar::all(0.0)));
			this->init_matrix[i].push_back(Mat(1,2, CV_64F, Scalar::all(0.0)));
			this->train_matrix[i].push_back(Mat(TRAIN_SEQ_MAX, TRAIN_OBS_MAX, CV_32S, Scalar::all(0)));
		}
	}

	// [VECTOR INIT DEBUG INFO]
	// hmm.printModel(trans_matrix[DEBUG_C][DEBUG_R], emit_matrix[DEBUG_C][DEBUG_R], init_matrix[DEBUG_C][DEBUG_R]);

	// read 2d hmm vectors from file
	ostringstream fname_emit_i;
	ostringstream fname_init_i;
	ostringstream fname_trans_i;
	fname_emit_i << dir_input << "emit2D.yml";
	fname_init_i << dir_input << "init2D.yml";
	fname_trans_i << dir_input << "trans2D.yml";
	FileStorage file_emit(fname_emit_i.str(), FileStorage::READ);
	FileStorage file_init(fname_init_i.str(), FileStorage::READ);
	FileStorage file_trans(fname_trans_i.str(), FileStorage::READ);
	for (int r = 0; r < num_of_row; r++)
		for (int c = 0; c < num_of_col; c++) {
			ostringstream id_stream;
			id_stream << "row_" << r << "-col_" << c;
			string ID = id_stream.str();
			file_emit[ID] >> emit_matrix[c][r];
			file_init[ID] >> init_matrix[c][r];
			file_trans[ID] >> trans_matrix[c][r];
	}
	file_emit.release();
	file_init.release();
	file_trans.release();

	// [VECTOR INIT DEBUG INFO2]
	// hmm.printModel(trans_matrix[DEBUG_C][DEBUG_R], emit_matrix[DEBUG_C][DEBUG_R], init_matrix[DEBUG_C][DEBUG_R]);
	return 0;
}
