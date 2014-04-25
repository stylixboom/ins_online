/*
 * bow_sig_extractor.h
 *
 *  Created on: August 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <unistd.h> // usleep
#include <tr1/unordered_map>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <flann/flann.hpp>
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/imtools.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"

using namespace std;
using namespace tr1;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::imtools;
using namespace alphautils::hdf5io;
using namespace ins;
using namespace cv;

// private variable
//-- Param
ins_param run_param;
int curr_img_width;
int curr_img_height;
//-- Dataset idf
vector<float> idf;
//-- Dataset cluster
Matrix<float> cluster;
size_t cluster_size;
//-- Parallel clustering, clustering params
int PARALLEL_BLOCKS;
int PARALLEL_CPU;
//Mat mask_img;
size_t mask_count = 0;
size_t mask_vertex_count = 0;
vector<float*> mask_vertex_x;
vector<float*> mask_vertex_y;

int num_kp;
int mask_pass;
timespec startTime;

void extractor_init();
void load_idf(const string& in);
void load_cluster(const string& in);
size_t extract_bow_sig(const string& session_id, vector<bow_bin_object>& bow_sig);
void query_quantization(const vector< vector<float> >& desc, const vector< vector<float> >& kp, vector<int>& query_quantized_index, vector<float>& query_quantized_dist);
void Bow(const vector<int>& query_quantized_indices, const vector< vector<float> >& query_keypoints, vector<bow_bin_object>& bow_sig);
bool export_hist(const string& out, const vector<bow_bin_object>& bow_hist);

// Quantizer
Index< ::flann::L2<float> > flann_search_index(KDTreeIndexParams((int)run_param.KDTREE));

//;)
