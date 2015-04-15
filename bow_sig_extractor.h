/*
 * bow_sig_extractor.h
 *
 *  Created on: August 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once

using namespace ins;

// private variable
//-- Param
ins_param run_param;
int total_kp;
int total_mask_pass;
int total_bin;

// SIFT Extractor
vector<bow_bin_object*> empty_result;
// ANN
quantizer ann;
// Inverted index
invert_index inverted_file;
float* idf;

bool visualize_enable;

// Dumper
kp_dumper dumper;

timespec startTime;

void run_listening();
void extractor_init();
bool listening_histrequest(vector<string>& hist_request);
void read_querylist(const string& session_id, vector<string>& queries, vector<unique_ptr<Mask> >& masks, vector<string>& mask_paths);
size_t extract_bowsig(const string& session_id);
bool export_bowsig(const string& out, const vector<bow_bin_object*>& bow_sig, int num_kp = 0, int mask_pass = 0);
void visualize_bow(const string& in_img, const string& out_img, const vector<bow_bin_object*>& bow_sig, const bool checkmask);

// Misc
void oxMaskImport(const string& mask_path, vector< vector<Point2f> >& polygons);

// Memory management

//;)
