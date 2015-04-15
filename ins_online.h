/*
 * ins_online.h
 *
 *  Created on: October 7, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once

using namespace alphautils;
using namespace ins;

// private variable
// Parameter
ins_param run_param;
//-- Dataset list
vector<string> ParentPaths;                 // Creating with ins_offline using deque, but using with vector
vector<int> Img2PoolIdx;                    // To keep track of image_id to pool_id
vector<size_t> Pool2ParentsIdx;
vector< pair<size_t, size_t> > Pool2ImagesIdxRange;    // To map between pool id and image ids [start, end]
vector<size_t> Img2ParentsIdx;
vector<string> ImgLists;
int query_topic_amount;
vector<string> QueryNameLists;
vector< vector<string> > QueryImgLists;
vector< vector<string> > QueryMaskLists;
// Mask variable only for oxford, paris dataset
vector< vector< vector<Point2f> > > oxMaskLists;
// Query Topic for INS
typedef struct _querytopic_object{ string name; int type; string detail; } querytopic_object;
vector<querytopic_object> QueryTopicLists;
const int QTYPE_LOCATION    = 0;
const int QTYPE_PERSON      = 1;
const int QTYPE_OBJ         = 2;
//-- Inverted index
bool rotate_eval;
int rotate_limit;
int rotate_memory_counter;
invert_index inverted_hist;
//-- Bow
bow bow_builder;
//-- Search
vector<bow_bin_object*> rand_bowsig;
int result_id = 0;
int curr_q_id;
vector<int> total_kp;
vector<int> total_mask_pass;
int TotalMatch;
//-- Result skipped list
bool* dataset_skiplist;
unordered_map< string, map<string, bool> > groundtruth_checkup;     // QueryName, <True_positive shot name, true>
//-- Multirank combination
vector< vector< pair<size_t, float> > > MultiResult;
int top_web_export = 2000;
int top_eval_export = 1000;

string simulated_session;

double searchTime;
double extractTime;
double homoTime;
timespec startTime;
double current_search_time;

// Map Report
stringstream groundtruth_path;
stringstream evalrank_path;
stringstream rawrank_path;
stringstream trecrank_path;
report total_report;
report rank_report;

// Ransac information
vector< map<size_t, int> > inlier_count_pack;        // vector< dataset_id -> inlier >
vector< map<size_t, double> > ransac_score_pack;     // vector< dataset_id -> ransac_score >

// Initialize function
void LoadDataset();
void LoadDatasetList();
void LoadQueryPreset();

// Core retrieval
float search_by_id(const int q_id);
float query_handle_basic(vector<string>& queries, const vector<string>& masks = vector<string>(), const int q_id = -1);
float query_handle_basic(const vector<string>& queries, const vector< vector<bow_bin_object*> >& query_bows, vector< vector<result_object> >& results, const int q_id = -1, const char caller_id = CALLER_NONE);   // post_processing e.g. fusion, qb, etc.
size_t search_by_bowsig(const vector<bow_bin_object*>& bow_sig, vector<result_object>& result);
void random_bowsig();   // For testing
void extract_bowsig(const string& session_name, const vector<string>& queries, vector< vector<bow_bin_object*> >& bow_sigs, const vector<string>& masks = vector<string>());
void import_bowsig(const string& in, vector<bow_bin_object*>& bow_sig);

// Tools
void NoisyQuery(const string& query_path);
string ResizeQuery(const string& query_path);
void RestoreQuery(const string& query_scaled_path);
void attache_resultinfo(vector< vector<result_object> >& results, size_t result_idx, const char caller_id);

// QBmining
void FIW(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fp_weight);
void PREFIW(const string& query_path, int minsup, unordered_map<size_t, float>& fi_weight);
void FIX(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fi_weight);
void GLOSD(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fi_weight);

// Late fusion
void late_fusion(const vector< vector<result_object> >& results, vector<result_object>& fused);

// Export result and visualization
void display_rank(const vector<result_object>& result);

// Evaluation
void Evaluate();
void CheckGroundtruth(vector<result_object>& result, const int q_id);
float Compute_map(const string& query_topic = "");
void ExportRawRank(const vector<result_object>& result, const string& query_path, float map = 0.0f);
void ExportEvalRank(const vector<result_object>& result, const string& query_path);
void ExportRank_Trec(const vector<result_object>& result, const string& out_path, const string& query_topic);
void SubmitRank_Trec(const vector<string>& source_trec_path, const vector<double>& timing);

// Misc
void oxMaskExport(const string& query_path, int q_id, const string& query_scaled_path = "");

// Memory management
void release_mem();
//;)
