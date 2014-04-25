/*
 * ins_online.h
 *
 *  Created on: October 7, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <sstream>
#include <bitset>
#include <new>
#include <iomanip>
#include <math.h>
#include <tr1/unordered_map>
#include <unistd.h>
#include <sys/types.h> // for checking exist file and dir
#include <sys/stat.h>
#include <dirent.h>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <flann/flann.hpp>
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/alphautils/imtools.h"
#include "../lib/alphautils/linear_tree.h"
#include "../lib/alphautils/tsp.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"
#include "../lib/ins/dataset_info.h"
#include "../lib/ins/homography.h"

#include "version.h"

using namespace std;
using namespace tr1;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::hdf5io;
using namespace alphautils::imtools;
using namespace ins;
using namespace cv;


// private variable
// Parameter
ins_param run_param;
//-- Dataset list
vector<string> ImgParentPaths;
vector<int> ImgPoolLevels;
vector<int> ImgListsPoolIds;
vector<size_t> ImgParentsIdx;
vector<string> ImgLists;
vector<string> QueryNameLists;
vector<string> QueryImgLists;
vector< vector< vector<Point2f> > > MaskLists;
dataset_info ds_info;
//-- Commercial info
bool is_commercial_film = false;
//-- Bow offset
vector<size_t> bow_offset;
//-- Inverted index
invert_index inverted_hist;
size_t dataset_size = 0;
//-- Parallel clustering, clustering params
int PARALLEL_BLOCKS;
int PARALLEL_CPU;
//-- Search
vector<bow_bin_object> randhist;
int result_id = 0;
vector< pair<size_t, float> > Result; // dataset_id, Val
vector< pair<size_t, int> > ReRanked; // Reindex for Result lookup with new score value
int num_kp;
int mask_pass;
int TotalMatch;
typedef struct _ExportOption{ int max; bool dev; bool ransac; bool showall;} ExportOption;
//-- Multirank combination
vector< vector< pair<size_t, float> > > MultiResult;

string simulated_session;

double searchTime;
double extractTime;
double homoTime;
timespec startTime;

// Map Report
stringstream groundtruth_path;
stringstream evalrank_path;
stringstream rawrank_path;
stringstream map_report;

// function
void LoadTestQuery(const string& DatasetPath);
void LoadDataset(const string& DatasetPath);
void LoadDatasetList(const string& in);
void LoadBowOffset();
void Resetinverted_hist();
void ResetMemory();
void extract_hist(const string& session_name, const vector<string>& queries, vector<bow_bin_object>& bow_hist);
void import_hist(const string& in, vector<bow_bin_object>& load_hist);
string ResizeQuery(const string& query_path);
// Weighting cluster_id by counting total bin
template<typename T, typename U> void calculate_symmat_distance(size_t nvert, const T vertx[], const T verty[], U symmat_dist[]);
// Accumulate Weight
void AUTOW_SD(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight);
void AUTOW_COMPACTNESS(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight);
void ACW(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight);
void QE_AVG(const string& query_path, vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight);
// Weighting cluster_id by Frequent Item Mining (FP-growth)
void SIW(const string& query_path, int minsup, unordered_map<size_t, float>& fi_weight);
void FIW(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fp_weight);
void QB1_Bow(vector<bow_bin_object>& bow);  // bow will be modified internally
void DirectQuery(const string& query_path);
void QueryBootstrapping_v1(const string& query_path); // Boostrapped query to find area, then crop sub area from original size to be sub queries
void QueryBootstrapping_v2(const string& query_path); // Boostrapped query to find rank, then combind new bow_hist from frequent bin of top rank to be searched
void ScanningQuery(const string& query_path);
void LoadSpecificBow(size_t dataset_id, vector<bow_bin_object>& load_hist);
float Compute_map(const string& groundtruth_path);
void Evaluate();
void AddRank(const vector< pair<size_t, float> >& Rank);
void CombineRank();
size_t search_by_id(int q_id);
size_t search_by_bow_sig(const vector<bow_bin_object>& bow_sig);
void RandomHist();
void DisplayRank();
void ExportEvalRank(const string& query_path);
void ExportRawRank(const string& query_path, float map = 0.0f);
void map_push_report(const string& text);
void ExportRank(const string& CurrSessionID, ExportOption Opts);
void FrameCopy();
void RenderQueueGenerator(size_t start, size_t end);
void UpdateStatus(const string& status_path, const string& status_text);
//;)
