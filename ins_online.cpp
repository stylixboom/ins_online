/*
 * ins_online.cpp
 *
 *  Created on: October 7, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <sstream>
#include <bitset>
#include <new>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <unordered_map>
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

// Siriwat's header
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/alphautils/report.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/alphautils/imtools.h"
#include "../lib/alphautils/linear_tree.h"
#include "../lib/alphautils/tsp.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"
#include "../lib/ins/bow.h"
#include "../lib/ins/qb.h"
#include "../lib/ins/qe.h"

// Merlin's header
#include "../lib/ins/utils.hpp"        // Properties

#include "ins_online.h"

#include "version.h"

using namespace std;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::hdf5io;
using namespace alphautils::imtools;
using namespace ins;
using namespace ins::utils;
using namespace cv;

// ==== Main ====
int main(int argc,char *argv[])
{
    // Disable synchronize I/O
    cin.sync_with_stdio(false);

	char menu;
	bool ivReady = false;

	do
	{
	    cout << endl;
		cout << "======== Instant Search - Online (" << ins_online_AutoVersion::ins_online_FULLVERSION_STRING << ") ========" << endl;
        cout << "[l] Load preset dataset" << endl;
		cout << "[i] Load Inverted Histogram" << endl;
		if(ivReady)
		{
		    cout << "[e] Evaluate" << endl;
			cout << "[s] Search by query" << endl;
			cout << "[w] Search by web browser" << endl;
			cout << "[r] Search by random bowsig" << endl;
		}
		cout << "[o] Options" << endl;
		cout << "[q] Quit" << endl;
		cout << "Enter menu:";cin >> menu;

		switch(menu)
		{
        case 'l':
            run_param.LoadPreset();

            LoadDataset();

            LoadQueryPreset();

            // Initial
            inverted_hist.init(run_param);

            if (run_param.stopword_enable)
            {
                cout << "Stop peak " << run_param.stopword_amount << " bins" << endl;
                inverted_hist.set_stopword_peak(run_param.stopword_amount);
            }
            //cout << "Stopword 1%,0% = " << inverted_hist.set_stopword_list(1, 0) << " cluster(s)" << endl;
            bow_builder.init(run_param);

            // Prepare tmp dir
            make_dir_available(run_param.shm_root_dir + "/query", "777");

            break;
		case 'i':
			{
				int TopLoad = 100;
				cout << "Top load = ";
				cin >> TopLoad;
				cout << "Load invert_index...";
				cout.flush();
				inverted_hist.load(TopLoad);
				cout << "OK!     " << endl;

                // Default rotate memory
                rotate_eval = false;    // rotate normal mode by default
                rotate_limit = 1;
                rotate_memory_counter = rotate_limit;

				ivReady = true;
				break;
			}
		case 'e':
			{
			    // Report
			    if (run_param.report_enable)
                {
                    total_report.init(run_param.online_working_path, "Total_report_" + run_param.dataset_prefix);
                    total_report.add_description("run param: " + run_param.dataset_prefix);
                    total_report.add_description("start time: " + currentDateTime());
                    rank_report.init(run_param.online_working_path, "Rank_report_" + run_param.dataset_prefix);
                    rank_report.add_description("run param: " + run_param.dataset_prefix);
                    rank_report.add_description("start time: " + currentDateTime());
                }

				Evaluate();

                // Report
				if (run_param.report_enable)
                {
                    total_report.add_description("end time: " + currentDateTime());
                    rank_report.add_description("end time: " + currentDateTime());
                    total_report.save_report();
                    rank_report.save_report();
                    total_report.reset();
                    rank_report.reset();
                }

				break;
			}
		case 's':
			{
                int q_id;
				cout << "Enter query number :";
				cin >> q_id;

                if (run_param.report_enable)
                {
                    total_report.init(run_param.online_working_path, "Total_report_query_" + toString(q_id) + "_" + run_param.dataset_prefix);
                    total_report.add_description("run param: " + run_param.dataset_prefix);
                    total_report.add_description("start time: " + currentDateTime());
                    rank_report.init(run_param.online_working_path, "Rank_report_query_" + toString(q_id) + "_" + run_param.dataset_prefix);
                    rank_report.add_description("run param: " + run_param.dataset_prefix);
                    rank_report.add_description("start time: " + currentDateTime());
                }

                // Search
				search_by_id(q_id);

				if (run_param.report_enable)
                {
                    total_report.add_description("end time: " + currentDateTime());
                    rank_report.add_description("end time: " + currentDateTime());
                    total_report.save_report();
                    rank_report.save_report();
                    total_report.reset();
                    rank_report.reset();
                }

				break;
			}
		case 'w':
			{
				stringstream state_flag_path;
				stringstream query_stack_path;
				stringstream hist_request_path;
				stringstream status_path;
				stringstream status_text;

				state_flag_path << run_param.online_working_path << "/state_flag.txt";
				query_stack_path << run_param.online_working_path << "/query_stack.txt";
				hist_request_path << run_param.online_working_path << "/hist_request.txt";

				cout << "Opening service..." << endl;

				// Run HistExtractor
				// Run Encoder

				cout << "=============================================" << endl;
				cout << "Listening to queries..." << endl;
				char KillService = '0';

				// Initial state file
				ofstream CreateStateFlagFile (state_flag_path.str().c_str());
				if(CreateStateFlagFile.is_open())
				{
					CreateStateFlagFile << "0";
					CreateStateFlagFile.close();
				}

				do
				{
					// ==== Check exit cmd
					ifstream StateFlagFile (state_flag_path.str().c_str());
					if (StateFlagFile && StateFlagFile.good())
					{
						StateFlagFile.read(&KillService, 1);
						StateFlagFile.close();

						// ==== Read queries
						if(KillService != '1')
						{
							// Import current query
							vector<string> Queries;

							// Waiting for queries
							while(!is_path_exist(query_stack_path.str()))
							{
                                usleep(250000);
                                ls2null(query_stack_path.str());
                            }

							// Load queries
							ifstream QueriesFile (query_stack_path.str().c_str());
							if (QueriesFile)
							{
								cout << "Got queries from client" << endl;
								while (QueriesFile.good())
								{
									string queryline;
									getline(QueriesFile, queryline);
									if (queryline != "")
									{
										vector<string> SubQuery;

										StringExplode(queryline, " ", SubQuery);  // for extracting option

										// Update status: Uploading queries
										status_path.str("");
										status_path << run_param.online_working_path << "/" << SubQuery[0] << "/" << "status.txt";
										status_text.str("");
										status_text << "Uploading queries...";
										text_write(status_path.str(), status_text.str());

										Queries.push_back(SubQuery[0]);
										cout << "SID : " << SubQuery[0] << endl;
									}
								}
								QueriesFile.close();

								// After read!, Send it to HistExtractor by just renaming!
								rename(query_stack_path.str().c_str(), hist_request_path.str().c_str());
							}

							// Extracting histogram and search
							for (size_t query_idx = 0; query_idx < Queries.size(); query_idx++)
							{
								stringstream session_path;
								stringstream query_list_path;
								stringstream hist_load_path;
								stringstream ready_signal_path;

								session_path << run_param.online_working_path << "/" << Queries[query_idx];
								query_list_path << session_path.str() << "/" << run_param.querylist_filename;
								hist_load_path << session_path.str() << "/" << "queryfilename" << run_param.hist_postfix;
								ready_signal_path << session_path.str() << "/" << "ready.sig";

								cout << "==== Histogram Extractor Module ====" << endl;
								cout << "Extracting histogram..";
								cout.flush();
								timespec histTime = CurrentPreciseTime();

								// Update status: Extracting histogram
								status_path.str("");
								status_path << session_path.str() << "/" << "status.txt";
								status_text.str("");
								status_text << "Extracting BoW histogram...";
								text_write(status_path.str(), status_text.str());

								// Waiting for HistXct result
								while(!is_path_exist(hist_load_path.str()))
                                {
                                    //usleep(250000);
                                    ls2null(hist_load_path.str());
                                }

								// Load BOW file
								vector<bow_bin_object*> bow_sig;
								import_bowsig(hist_load_path.str(), bow_sig);

								extractTime = TimeElapse(histTime);
								cout << "completed!";
								cout << " (" <<  extractTime << " s)" << endl;
								cout << "Bin amount: " << bow_sig.size() << " bins" << endl;

								// Update status: Searching
								status_text.str("");
								status_text << "Searching on an inverted index database...";
								text_write(status_path.str(), status_text.str());

								// Search
								cout << "==== Search Module ====" << endl;
								vector<result_object> result;
								startTime = CurrentPreciseTime();
								result_id = search_by_bowsig(bow_sig, result);
								searchTime = TimeElapse(startTime);
								cout << "Match with: " << TotalMatch << " videos" << endl;
								cout << "Search time: " <<  searchTime << " s" << endl;
								cout << "Result (dataset_id) : " << result_id << endl;
								//display_rank();

								// Update status: Exporting result
								status_text.str("");
								status_text << "Returning result...";
								text_write(status_path.str(), status_text.str());

								cout << "-- Export result" << endl;
								timespec ExportTime = CurrentPreciseTime();
								// TO-DO
								cout << "Export time: " <<  TimeElapse(ExportTime) << " s" << endl;
								// ExportRankTrec(-1, -1);

								// Release Memory
								bow::release_bowsig(bow_sig);

								// Send result ready signal
								ofstream CreateReadyFile(ready_signal_path.str().c_str());
								if (CreateReadyFile.is_open())
								{
									CreateReadyFile << ":)";
									CreateReadyFile.close();
								}

								// Update status: Send complete signal
								status_text.str("");
								status_text << "Completed!";
								text_write(status_path.str(), status_text.str());
							}

							cout << "=============================================" << endl << "Listening to queries..." << endl;
						}
					}
					usleep(250000);
				}while(KillService != '1');

				cout << "Closing service..." << endl;

				break;
			}
		case 'r':
			{
				char gen;
				cout << "Generate new bowsig?";
				cin >> gen;
				if (gen == 'y')
                    random_bowsig();
				cout << "Generate completed!" << endl;
				vector<result_object> result;
				startTime = CurrentPreciseTime();
				result_id = search_by_bowsig(rand_bowsig, result);
				searchTime = TimeElapse(startTime);
				cout << "Total time: " <<  searchTime << " s" << endl;
				cout << "Match with: " << TotalMatch << " videos" << endl;
				cout << "Result (dataset_id) : " << result_id << endl;
				display_rank(result);

				// Post Matching
				cout << "==== Result Processor Module ====" << endl;

                //Release memory
                bow::release_bowsig(rand_bowsig);
				break;
			}
		case 'o':
			{
			    cin.clear();
			    cout << "======== Toggle Options ========" << endl;
			    cout << "[1] (" << run_param.reuse_bow_sig << ") Reuse bow sig" << endl;
                cout << "[2] (" << run_param.report_enable << ") Report" << endl;
                cout << "[3] (" << run_param.matching_dump_enable << ") Matching dump" << endl;
                cout << "[4] (" << run_param.submit_enable << ") Submit" << endl;
                cout << "[5] (" << rotate_limit << ") Rotate limit (inverted index search counter, to clear cache)" << endl;
                cout << "[6] (" << top_web_export << ") Top web export" << endl;
                cout << "[7] (" << top_eval_export << ") Top eval export" << endl;
                cout << "[8] (" << run_param.qe_ransac_adint_manual << ") QB-RANSAC manual mode" << endl;
				int opts;
				cout << "Toggle option number: "; cout.flush();
				cin >> opts;
				if (opts == 1)
                    run_param.reuse_bow_sig = !run_param.reuse_bow_sig;
                else if (opts == 2)
                    run_param.report_enable = !run_param.report_enable;
                else if (opts == 3)
                    run_param.matching_dump_enable = !run_param.matching_dump_enable;
                else if (opts == 4)
                    run_param.submit_enable = !run_param.submit_enable;
                else if (opts == 5)
                {
                    cout << "Set search counter limit (round): "; cout.flush();
                    cin >> rotate_limit;
                    rotate_memory_counter = rotate_limit;
                }
                else if (opts == 6)
                {
                    cout << "Set top web export: "; cout.flush();
                    cin >> top_web_export;
                }
                else if (opts == 7)
                {
                    cout << "Set top eval export: "; cout.flush();
                    cin >> top_eval_export;
                }
                else if (opts == 8)
                    run_param.qe_ransac_adint_manual = !run_param.qe_ransac_adint_manual;

				break;
			}
		}
	}
	while (menu != 'q');

	// Release memory
	delete[] dataset_skiplist;

	exit(0);
}

//==== Initialize function
void LoadQueryPreset()
{
    // Release previous query preset
    query_topic_amount = 0;
    vector<string>().swap(QueryNameLists);
    vector< vector<string> >().swap(QueryImgLists);
    size_t dataset_skip_count = 0;

    cout << "Loading query for " << bluec << run_param.dataset_prefix << endc << "...";
    cout.flush();
    startTime = CurrentPreciseTime();

    if (str_contains(run_param.dataset_prefix, "oxbuildings") ||
        str_contains(run_param.dataset_prefix, "paris") ||
        str_contains(run_param.dataset_prefix, "smalltest"))
    {
        stringstream groundtruth_path;
        groundtruth_path << run_param.dataset_path << "/groundtruth";
        //cout << "groundtruth_path: " << groundtruth_path.str() << endl;
        map<string, string> topic_names;
        // Directory traverse
        DIR* dirp = opendir(groundtruth_path.str().c_str());
        dirent* dp;
        while ((dp = readdir(dirp)) != NULL)
        {
            // Topic name
            if (str_contains(string(dp->d_name), "query"))
            {
                // Query name
                string topic_name = str_replace_first(string(dp->d_name), "_query.txt", "");
                topic_names[topic_name] = string(dp->d_name);
            }
        }
        closedir(dirp);

        /// Keep topic_name
        for (auto topic_names_it = topic_names.begin(); topic_names_it != topic_names.end(); topic_names_it++)
            QueryNameLists.push_back(topic_names_it->first);

        /// Keep query filename
        for (size_t topic_id = 0; topic_id < QueryNameLists.size(); topic_id++)
        {
            stringstream query_test_path;
            query_test_path << groundtruth_path.str() << "/" << QueryNameLists[topic_id] << "_query.txt";

            ifstream InFile (query_test_path.str().c_str(), ios::binary);
            if (InFile)
            {
                while (InFile.good())
                {
                    string queryline;
                    getline(InFile, queryline);
                    // oxc1_all_souls_000013 136.5 34.1 648.5 955.7
                    // query                 min_x min_y max_x max_y
                    if (queryline != "")
                    {
                        vector<string> query_split;

                        StringExplode(queryline, " ", query_split);

                        QueryImgLists.push_back(vector<string>());

                        // Query image
                        if (str_contains(run_param.dataset_prefix, "oxbuildings"))                              // oxfordbuildings
                            QueryImgLists.back().push_back(str_replace_first(query_split[0], "oxc1_", "") + ".jpg");
                        else                                                                                    // paris
                            QueryImgLists.back().push_back(query_split[0] + ".jpg");

                        // oxford mask template reader
                        if (run_param.mask_enable && run_param.mask_mode == MASK_ROI)
                        {
                            // Mask
                            Point2f pLeftTop(atoi(query_split[1].c_str()), atoi(query_split[2].c_str()));
                            Point2f pRightTop(atoi(query_split[3].c_str()), atoi(query_split[2].c_str()));
                            Point2f pRightBottom(atoi(query_split[3].c_str()), atoi(query_split[4].c_str()));
                            Point2f pLeftBottom(atoi(query_split[1].c_str()), atoi(query_split[4].c_str()));
                            vector< vector<Point2f> > mask;
                            vector<Point2f> sub_mask;
                            sub_mask.push_back(pLeftTop);
                            sub_mask.push_back(pRightTop);
                            sub_mask.push_back(pRightBottom);
                            sub_mask.push_back(pLeftBottom);
                            sub_mask.push_back(pLeftTop); // Closed polygon
                            mask.push_back(sub_mask);
                            oxMaskLists.push_back(mask);
                        }
                    }
                }
                InFile.close();
            }
        }

        /// Groundtruth
        cout << endl << "Loading groundtruth file " << groundtruth_path.str() << "/*_good.txt *_ok.txt..."; cout.flush();
        for (size_t query_idx = 0; query_idx < QueryNameLists.size(); query_idx++)
        {
            // Good
            vector<string> groundtruth_good = text_readline2vector(groundtruth_path.str() + "/" + QueryNameLists[query_idx] + "_good.txt");
            for (size_t groundtruth_idx = 0; groundtruth_idx < groundtruth_good.size(); groundtruth_idx++)
            {
                /// all_souls_000091
                groundtruth_checkup[QueryNameLists[query_idx]][groundtruth_good[groundtruth_idx]] = true;
            }
            // OK
            vector<string> groundtruth_ok = text_readline2vector(groundtruth_path.str() + "/" + QueryNameLists[query_idx] + "_ok.txt");
            for (size_t groundtruth_idx = 0; groundtruth_idx < groundtruth_ok.size(); groundtruth_idx++)
            {
                /// all_souls_000091
                groundtruth_checkup[QueryNameLists[query_idx]][groundtruth_ok[groundtruth_idx]] = true;
            }
            percentout(query_idx, QueryNameLists.size(), 1);
        }
        cout << "done!" << endl;
    }
    else if (str_contains(run_param.dataset_prefix, "ins201"))
    {
        stringstream queryimg_path;
        queryimg_path << get_directory(run_param.dataset_path) << "/query/frames";

        /// 1. Search topic name
        DIR *topic_dir = opendir(queryimg_path.str().c_str());
        if (topic_dir == NULL) {
            cerr << "query path not found: " << queryimg_path.str() << endl;
            exit(EXIT_FAILURE);
        }

        dirent *topic_ent;
        struct stat s;
        map<string, string> topic_names;
        while ((topic_ent = readdir(topic_dir)) != NULL) {
            stat((queryimg_path.str() + "/" + topic_ent->d_name).c_str(), &s);
            if (topic_ent->d_name[0] != '.' && S_ISDIR(s.st_mode))
            {
                // Keep topic name
                string topic_name = string(topic_ent->d_name);
                topic_names[topic_name] = topic_name;
            }
        }
        closedir(topic_dir);

        // Keep topic_name
        for (auto topic_names_it = topic_names.begin(); topic_names_it != topic_names.end(); topic_names_it++)
            QueryNameLists.push_back(topic_names_it->second);

        string prefer_ext = "jpg";
        if (str_contains(run_param.dataset_prefix, "ins2014-videoquery"))
            prefer_ext = "bmp";
        else if (str_contains(run_param.dataset_prefix, "ins2014"))
            prefer_ext = "png";

        // Keep query file name
        for (size_t topic_id = 0; topic_id < QueryNameLists.size(); topic_id++)
        {
            string topic_name = QueryNameLists[topic_id];

            // Prepare img name
            QueryImgLists.push_back(vector<string>());
            // Prepare mask name
            QueryMaskLists.push_back(vector<string>());

            /// 2. Search query file under topic name
            // Search the image directory, remove duplicate images in different formats.
            string path = queryimg_path.str() + "/" + topic_name;
            DIR *img_dir = opendir(path.c_str());
            if (img_dir == NULL) {
                cerr << "Failed to open directory: " << path << endl;
                exit(EXIT_FAILURE);
            }
            dirent *img_ent;
            struct stat s;
            map<string, string> imagePaths;

            while ((img_ent = readdir(img_dir)) != NULL) {
                stat((path + "/" + img_ent->d_name).c_str(), &s);
                if (S_ISREG(s.st_mode)) { // if it is regular file
                    string name = string(img_ent->d_name);
                    string ext = name.substr(name.rfind('.') + 1);

                    // 9023.1.src.jpg -> 9023.1
                    name = name.substr(0, name.rfind('.', name.rfind('.') - 1));

                    if (imagePaths.count(name) == 0 && ext == prefer_ext) { // check with prefer format
                        imagePaths[name] = img_ent->d_name;
                    }
                }
            }

            // Keep img name
            for (auto imagePaths_it = imagePaths.begin(); imagePaths_it != imagePaths.end(); imagePaths_it++)
                QueryImgLists.back().push_back(imagePaths_it->second);

            closedir(img_dir);

            // Search the corresponding mask directory.
            path = get_directory(run_param.dataset_path) + "/query/masks";
            if (str_contains(run_param.dataset_prefix, "ins2014-videoquery"))
                path += "/" + topic_name;
            img_dir = opendir(path.c_str());
            if (img_dir == NULL) {
                cerr << "Failed to open directory: " << path << endl;
                exit(EXIT_FAILURE);
            }
            map<string, string> maskPaths;

            while ((img_ent = readdir(img_dir)) != NULL) {
                stat((path + "/" + img_ent->d_name).c_str(), &s);
                if (S_ISREG(s.st_mode)) { // if it is regular file
                    string name = string(img_ent->d_name);
                    string ext = name.substr(name.rfind('.') + 1);
                    // 9023.1.mask.jpg -> 9023.1
                    name = name.substr(0, name.rfind('.', name.rfind('.') - 1));

                    if (imagePaths.count(name) != 0
                            && (maskPaths.count(name) == 0 && ext == prefer_ext)) { // check with prefer format
                        maskPaths[name] = img_ent->d_name;
                    }
                }
            }

            closedir(img_dir);

            // Keep mask name
            for (auto maskPaths_it = maskPaths.begin(); maskPaths_it != maskPaths.end(); maskPaths_it++)
                QueryMaskLists.back().push_back(maskPaths_it->second);
        }

        /// Loading dropped list
        if (str_contains(run_param.dataset_prefix, "ins2013") || str_contains(run_param.dataset_prefix, "ins2014"))
        {
            cout << endl << "Initializing dataset skip list.."; cout.flush();
            // Preparing skiplist filter
            size_t pool_size = Pool2ParentsIdx.size();
            dataset_skiplist = new bool[pool_size];
            for (size_t pool_idx = 0; pool_idx < pool_size; pool_idx++)
                dataset_skiplist[pool_idx] = false;
            // Load skiplist file
            string skiplist_filename = "";
            if (str_contains(run_param.dataset_prefix, "ins2013"))
                skiplist_filename = "tv13.ins.dropped.example.image.shots";
            else if (str_contains(run_param.dataset_prefix, "ins2014"))
                skiplist_filename = "tv14.ins.dropped.example.image.shots";

            string dropped_list_filename = run_param.code_root_dir + "/trec_ap/" + skiplist_filename;
            vector<string> raw_skiplist = text_readline2vector(dropped_list_filename);
            set<string> set_skiplist;
            /// Setup skip filter from
            /// 1. Predefined skip
            for (size_t skip_idx = 0; skip_idx < raw_skiplist.size(); skip_idx++)
                set_skiplist.insert(raw_skiplist[skip_idx]);
            /// 2. Developer data shot0*
            for (size_t pool_idx = 0; pool_idx < pool_size; pool_idx++)
            {
                vector<string> split_val;
                // ins2011/img/9661
                StringExplode(ParentPaths[Pool2ParentsIdx[pool_idx]], "/", split_val);
                string shot_name = split_val[2];
                if (str_contains(shot_name, "shot0_") || (set_skiplist.find(shot_name) != set_skiplist.end()))
                {
                    dataset_skiplist[pool_idx] = true;
                    dataset_skip_count++;
                }
                percentout(pool_idx, pool_size, 10);
            }
            cout << "done!" << endl;
        }

        /// Groundtruth path
        groundtruth_path.str("");
        if (str_contains(run_param.dataset_prefix, "ins2011"))
            groundtruth_path << run_param.code_root_dir << "/trec_ap/tv11.ins.truth";
        else if (str_contains(run_param.dataset_prefix, "ins2012"))
            groundtruth_path << run_param.code_root_dir << "/trec_ap/ins.search.qrels.tv12.revised";
        else if (str_contains(run_param.dataset_prefix, "ins2013"))
            groundtruth_path << run_param.code_root_dir << "/trec_ap/ins.search.qrels.tv13";
        else if (str_contains(run_param.dataset_prefix, "ins2014"))
            groundtruth_path << run_param.code_root_dir << "/trec_ap/ins.search.qrels.tv14";

        cout << "Loading groundtruth file " << groundtruth_path.str() << ".."; cout.flush();
        vector<string> groundtruth_raw = text_readline2vector(groundtruth_path.str());
        for (size_t groundtruth_idx = 0; groundtruth_idx < groundtruth_raw.size(); groundtruth_idx++)
        {
            vector<string> queryname_truth;
            /// 9080 0 shot75_1184 0
            StringExplode(groundtruth_raw[groundtruth_idx], " ", queryname_truth);
            groundtruth_checkup[queryname_truth[0]][queryname_truth[2]] = toBool(queryname_truth[3]);
            percentout(groundtruth_idx, groundtruth_raw.size(), 10);
        }
        cout << "done!" << endl;

        /// Loading query topic list
        if (str_contains(run_param.dataset_prefix, "ins2013") || str_contains(run_param.dataset_prefix, "ins2014"))
        {
            string topic_filename;
            if (str_contains(run_param.dataset_prefix, "ins2013"))                                      // INS13
                topic_filename = "ins2013.topics.txt";
            else                                                                                        // INS14
                topic_filename = "ins2014.topics.txt";

            string querytopiclist_filename = run_param.dataset_root_dir + "/" + run_param.dataset_name + "/" + topic_filename;

            cout << "Loading topics file " << greenc << topic_filename << endc << ".."; cout.flush();

            vector<string> raw_type_list = text_readline2vector(querytopiclist_filename);

            for (size_t query_idx = 0; query_idx < raw_type_list.size(); query_idx++)
            {
                // 9099#OBJECT#a checkerboard band on a police cap
                vector<string> split_list;
                StringExplode(raw_type_list[query_idx], "#", split_list);  // for extracting option

                // Add query type
                if (QueryNameLists[query_idx] == split_list[0])
                {
                    if (split_list[1] == "OBJECT")
                        QueryTopicLists.push_back(querytopic_object{split_list[0], QTYPE_OBJ, split_list[2]});
                    else if (split_list[1] == "PERSON")
                        QueryTopicLists.push_back(querytopic_object{split_list[0], QTYPE_PERSON, split_list[2]});
                    else if (split_list[1] == "LOCATION")
                        QueryTopicLists.push_back(querytopic_object{split_list[0], QTYPE_LOCATION, split_list[2]});
                }
                else
                {
                    cout << "Incorrect query type list!" << endl;
                    cout << "Query type file: " << querytopiclist_filename << endl;
                    cout << "idx:" << query_idx << " " << QueryNameLists[query_idx] << " - " << split_list[0] << endl;
                    exit(EXIT_FAILURE);
                }

                percentout(query_idx, raw_type_list.size(), 1);
            }

            cout << raw_type_list.size() << " topic(s) done!" << endl;
        }
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    query_topic_amount = QueryNameLists.size();
    cout << "Total " << query_topic_amount << " query topic(s)" << endl;
    if (str_contains(run_param.dataset_prefix, "ins2013") || str_contains(run_param.dataset_prefix, "ins2014"))
        cout << "Total " << dataset_skip_count << " datasets were skipped" << endl;

    /*
    cout << "QueryImgLists.size() " << QueryImgLists.size() << endl;
    for (size_t qid = 0; qid < QueryImgLists.size(); qid++)
    {
        cout << "  [" << qid << "].size() " << QueryImgLists[qid].size() << endl;
        for (size_t qidd = 0; qidd < QueryImgLists[qid].size(); qidd++)
            cout << "    [" << qid << "] " << QueryImgLists[qid][qidd] << endl;
    }
    cout << "QueryMaskLists.size() " << QueryMaskLists.size() << endl;
    for (size_t qid = 0; qid < QueryMaskLists.size(); qid++)
    {
        cout << "  [" << qid << "].size() " << QueryMaskLists[qid].size() << endl;
        for (size_t qidd = 0; qidd < QueryMaskLists[qid].size(); qidd++)
            cout << "    [" << qid << "] " << QueryMaskLists[qid][qidd] << endl;
    }
    */
}

void LoadDataset()
{
    if (is_path_exist(run_param.dataset_basepath_path))
    {
        cout << "Load dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadDatasetList();
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
    {
        cout << "Dataset list not available!" << endl;
        cout << "path: " << run_param.dataset_basepath_path << endl;
        exit(1);
    }

    // Checking image avalibality
    if (ImgLists.size() > 0)
    {
        cout << "== Dataset information ==" << endl;
        //cout << "Total directory: " << ParentPaths.size() << endl;
        cout << "Total pool: " << Img2PoolIdx[Img2PoolIdx.size() - 1] + 1 << endl;
        cout << "Total image: " << ImgLists.size() << endl;
    }
    else
        cout << "No image available" << endl;
}

void LoadDatasetList()
{
    // Read parent path (dataset based path)
    ifstream InParentFile (run_param.dataset_basepath_path.c_str());
    if (InParentFile)
    {
        string read_line;
        while (!InParentFile.eof())
        {
            getline(InParentFile, read_line);
            if (read_line != "")
            {
                vector<string> split_line;
                // parent_id:parent_path

                StringExplode(read_line, ":", split_line);

                ParentPaths.push_back(split_line[1]);
            }
        }

        // Close file
        InParentFile.close();
    }

    // Read image filename
    ifstream InImgFile (run_param.dataset_filename_path.c_str());
    if (InImgFile)
    {
        string read_line;
        unordered_set<int> pool_set;
        while (!InImgFile.eof())
        {
            getline(InImgFile, read_line);
            if (read_line != "")
            {
                // pool_id:parent_id:image_name

                // Find first and second pos of ":", the rest is an image name
                size_t cpos_start = 0;
                size_t cpos_end = 0;
                size_t line_size = read_line.length();
                bool done_parent_id = false;
                bool done_pool2parent_id = false;
                for (cpos_end = 0; cpos_end < line_size; cpos_end++)
                {
                    if (read_line[cpos_end] == ':')
                    {
                        int idx = atoi(read_line.substr(cpos_start, cpos_end - cpos_start).c_str());
                        cpos_start = cpos_end + 1;
                        if (!done_parent_id)                // Parent id
                        {
                            Img2ParentsIdx.push_back(idx);
                            done_parent_id = true;
                        }
                        else if (!done_pool2parent_id)      // Pool to parent id
                        {
                            if (pool_set.find(idx) == pool_set.end())
                            {
                                pool_set.insert(idx);
                                Pool2ParentsIdx.push_back(idx);
                                Pool2ImagesIdxRange.push_back(pair<size_t, size_t>(ImgLists.size(), ImgLists.size()));  // Add pool2image range
                            }
                            done_pool2parent_id = true;
                            Pool2ImagesIdxRange.back().second = ImgLists.size();                                        // Update pool2image range
                        }
                        else                                // Image to parent id
                        {
                            Img2PoolIdx.push_back(idx);
                            break;                          // Stop search, the rest is image name
                        }
                    }
                }

                // Image name
                ImgLists.push_back(read_line.substr(cpos_start, line_size - cpos_start));
            }
        }

        // Close file
        InImgFile.close();
    }
}

//==== Core retrieval
float search_by_id(const int q_id)
{
    // Save q_id
    curr_q_id = q_id;
	cout << "================ " << cyanc << q_id + 1 << endc << " " << yellowc << QueryNameLists[q_id] << endc << " ================" << endl;

    // Copy image source to query session then create query_list.txt
    vector<string> source_queries;
    vector<string> queries;
    vector<string> queries_scaled;
    vector<string> source_masks;
    vector<string> masks;
    string query_search_path; // temporary directory session

    /// Preparing queries search path
    simulated_session = QueryNameLists[q_id];   // <-- simulated session_id with QueryNameLists[], is used to simulated online session_id through this running session
    query_search_path = run_param.online_working_path + "/" + simulated_session;
    make_dir_available(query_search_path, "777");
    // [/home/stylix/webstylix/code/ins_online/query/oxbuildings5k_sifthesaff-rgb-norm-root_akm_1000000_kd3_16_qscale_r80_mask_roi_qbootstrap2_18_f2_qbmining_5_report]/[balliol_5]/[oxford_001753.jpg]

    /// Preparing queries source path
    if (str_contains(run_param.dataset_prefix, "oxbuildings1m") ||
		str_contains(run_param.dataset_prefix, "oxbuildings105k") ||
        str_contains(run_param.dataset_prefix, "oxbuildings5k") ||
        str_contains(run_param.dataset_prefix, "paris6k"))
    {
        if (str_contains(run_param.dataset_prefix, "oxbuildings1m") ||																/// Oxford1m/105k query system
			str_contains(run_param.dataset_prefix, "oxbuildings105k"))                                                              /// will be directed to 5k system
        {
            source_queries.push_back(run_param.dataset_root_dir + "/" + "oxbuildings/5k" + "/" + QueryImgLists[q_id][0]);           // source query
            // [/home/stylix/webstylix/code/dataset]/[oxbuildings/5k]/[oxford_001753.jpg]
        }
        else if (str_contains(run_param.dataset_prefix, "oxbuildings5k"))                                                           /// Oxford5k query system
        {
            source_queries.push_back(run_param.dataset_path + "/" + QueryImgLists[q_id][0]);                                        // source query
            // [/home/stylix/webstylix/code/dataset]/[oxbuildings/5k]/[oxford_001753.jpg]
        }
        else if (str_contains(run_param.dataset_prefix, "paris6k"))                                                                 /// Paris6k query system
        {
            vector<string> query_image_split;
            StringExplode(QueryImgLists[q_id][0], "_", query_image_split);
            source_queries.push_back(run_param.dataset_path + "/" + query_image_split[1] + "/" + QueryImgLists[q_id][0]);           // source query
            // [/home/stylix/webstylix/code/dataset]/[paris/6k]/[invalides]/[paris_invalides_000360.jpg]
        }
        queries.push_back(query_search_path + "/" + QueryImgLists[q_id][0]);                                                        // destination query

        if (run_param.mask_enable && run_param.mask_mode == MASK_ROI)
            masks.push_back(query_search_path + "/" + QueryImgLists[q_id][0] + ".mask");

        // Groundtruth path
        groundtruth_path.str("");
        groundtruth_path << run_param.dataset_path << "/groundtruth/" << QueryNameLists[q_id];
    }
    else if (str_contains(run_param.dataset_prefix, "ins201"))                                                                      /// INS query system (multiple queries)
    {
        for (size_t img_idx = 0; img_idx < QueryImgLists[q_id].size(); img_idx++)
        {
            source_queries.push_back(get_directory(run_param.dataset_path) + "/query/frames/" + simulated_session + "/" + QueryImgLists[q_id][img_idx]);                // source query
            queries.push_back(query_search_path + "/" + QueryImgLists[q_id][img_idx]);                                                                                  // destination query

            if (run_param.mask_enable && run_param.mask_mode == MASK_IMG)
            {
                if (str_contains(run_param.dataset_prefix, "ins2014-videoquery"))
                    source_masks.push_back(get_directory(run_param.dataset_path) + "/query/masks/" + QueryNameLists[q_id] + "/" + str_replace_first(QueryImgLists[q_id][img_idx], "src", "mask"));   // source mask
                else
                    source_masks.push_back(get_directory(run_param.dataset_path) + "/query/masks/" + str_replace_first(QueryImgLists[q_id][img_idx], "src", "mask"));   // source mask
                masks.push_back(query_search_path + "/" + str_replace_first(QueryImgLists[q_id][img_idx], "src", "mask"));                                              // destination mask
            }
        }
    }

    /// Transfer image and mask to search directory (fastmem)
    cout << "Copying query..."; cout.flush();
    timespec copytime = CurrentPreciseTime();
    stringstream cmd;
    for (size_t img_idx = 0; img_idx < source_queries.size(); img_idx++)
        cmd << "cp " << source_queries[img_idx] << " " << queries[img_idx] << "; ";
    if (run_param.mask_enable && run_param.mask_mode == MASK_IMG)
    {
        cout << "mask..."; cout.flush();
        for (size_t mask_idx = 0; mask_idx < source_masks.size(); mask_idx++)
            cmd << "cp " << source_masks[mask_idx] << " " << masks[mask_idx] << "; ";
    }
    exec(cmd.str());
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(copytime) << " s)" << endl;

    // Add noise
    if (run_param.query_noise_enable)
    {
        for (size_t img_idx = 0; img_idx < queries.size(); img_idx++)
            NoisyQuery(queries[img_idx]);
    }

    // Resize image
	if (run_param.query_scale_enable)
    {
        // Scaling query
        for (size_t img_idx = 0; img_idx < queries.size(); img_idx++)
            queries_scaled.push_back(ResizeQuery(queries[img_idx]));
        // Scaling mask (not restore mode)
        if (!run_param.query_scale_restore_enable)
        {
            if (run_param.mask_enable && run_param.mask_mode == MASK_IMG)
                for (size_t mask_idx = 0; mask_idx < masks.size(); mask_idx++)
                    masks[mask_idx] = ResizeQuery(masks[mask_idx]);
        }
    }

    // Generate mask
    if (run_param.mask_enable)
    {
        /// Preparing mask (oxford mask, one mask per query)
        if (str_contains(run_param.dataset_prefix, "oxbuildings1m") ||
			str_contains(run_param.dataset_prefix, "oxbuildings105k") ||
			str_contains(run_param.dataset_prefix, "oxbuildings5k") ||
			str_contains(run_param.dataset_prefix, "paris6k"))
        {
            if (run_param.mask_enable && run_param.mask_mode == MASK_ROI)
            {
                // Export mask file to scaled query name (not restore mode)
                if (run_param.query_scale_enable && !run_param.query_scale_restore_enable)
                    oxMaskExport(queries[0], q_id, queries_scaled[0]);
                // Export mask file normally
                else
                    oxMaskExport(queries[0], q_id);
            }
        }
    }

    // Maybe error when resize query, please check
    float curr_ap = 0;
    /// Run scale mode if not restore mode
    if (run_param.query_scale_enable && !run_param.query_scale_restore_enable)
        curr_ap = query_handle_basic(queries_scaled, masks, q_id);    // This code not tested
    else
        curr_ap = query_handle_basic(queries, masks, q_id);           // The original working code

    return curr_ap;
}

float query_handle_basic(vector<string>& queries, const vector<string>& masks, const int q_id)
{
    // Restoring for scaled query to its original scale
    if (run_param.query_scale_restore_enable)
    {
        for (size_t img_idx = 0; img_idx < queries.size(); img_idx++)
            RestoreQuery(queries[img_idx]);
    }

    // Report new_data
    string report_data_key;
    if (run_param.report_enable)
    {
        // One image for early fusion
        if (run_param.earlyfusion_enable)
        {
            // Report data key
            report_data_key = QueryNameLists[q_id] + "_" + QueryNameLists[q_id] + ".earlyfused";
            total_report.new_data(report_data_key, QueryNameLists[q_id]);
            rank_report.new_data(report_data_key + "_id", QueryNameLists[q_id]);
            rank_report.new_data(report_data_key + "_score", QueryNameLists[q_id]);
        }
        // All images for normal
        else
        {
            size_t query_idx = 0;
            do
            {
                // Report data key
                report_data_key = QueryNameLists[q_id] + "_" + QueryImgLists[q_id][query_idx];
                total_report.new_data(report_data_key, QueryNameLists[q_id]);
                rank_report.new_data(report_data_key + "_id", QueryNameLists[q_id]);
                rank_report.new_data(report_data_key + "_score", QueryNameLists[q_id]);
            }
            while (!run_param.earlyfusion_enable && ++query_idx < queries.size());
        }
    }

    /*
    bool preset_query = true;
    if (q_id == -1)
        preset_query = false;
    */

    /// Preparing bow_sig and result space
    vector< vector<bow_bin_object*> > bow_sigs;
    vector< vector<result_object> > results;
    do
    {
        // bowsig
        vector<bow_bin_object*> bow_sig;
        bow_sigs.push_back(bow_sig);
        // result
        vector<result_object> result;
        results.push_back(result);
    }
    while (!run_param.earlyfusion_enable && bow_sigs.size() < queries.size());

    /// Request extract hist
    cout << "Extracting BOW histogram.."; cout.flush();
    startTime = CurrentPreciseTime();
    extract_bowsig(simulated_session, queries, bow_sigs, masks);
    extractTime = TimeElapse(startTime);
    cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

    /// Send to bow mode query handling
    float final_ap = 0;
    final_ap = query_handle_basic(queries, bow_sigs, results, q_id);

    // Release memory
    // query bow_sig
    for (size_t bow_idx = 0; bow_idx < bow_sigs.size(); bow_idx++)
        bow::release_bowsig(bow_sigs[bow_idx]);

    // Release memory
    // result ranks
    for (size_t result_idx = 0; result_idx < results.size(); result_idx++)
        vector<result_object>().swap(results[result_idx]);
    vector< vector<result_object> >().swap(results);

    return final_ap;
}

float query_handle_basic(const vector<string>& queries, const vector< vector<bow_bin_object*> >& query_bows, vector< vector<result_object> >& results, const int q_id, const char caller_id)
{
    bool preset_query = true;
    if (q_id == -1)
        preset_query = false;

    /// Searching
    for (size_t bow_idx = 0; bow_idx < query_bows.size(); bow_idx++)
    {
        // Report key
        string report_data_key;
        if (run_param.report_enable)
        {
            if (run_param.earlyfusion_enable)
                report_data_key = QueryNameLists[q_id] + "_" + QueryNameLists[q_id] + ".earlyfused";
            else
            {
                if ((caller_id & CALLER_QB) == CALLER_QB || (caller_id & CALLER_QE) == CALLER_QE)
                    report_data_key = QueryNameLists[q_id] + "_" + get_filename(queries[bow_idx]);
                else
                    report_data_key = QueryNameLists[q_id] + "_" + QueryImgLists[q_id][bow_idx];
            }
        }

        cout << "Query file == " << greenc << get_filename(queries[bow_idx]) << endc << " [" << yellowc << bow_idx + 1 << endc << "/" << bluec << query_bows.size() << endc << "]" << " ==" << endl;

        cout << "Bin amount: " << query_bows[bow_idx].size() << " bins" << endl;
        cout << "Total feature(s): " << total_kp[bow_idx] << " point(s)" << endl;
        cout << "Total mask passed: " << total_mask_pass[bow_idx] << " point(s)" << endl;

        // Report
        if (run_param.report_enable)
        {
            // Report bin_size
            total_report.add_key_at("bin_size", toString(query_bows[bow_idx].size()), report_data_key);
            // Report total_kp
            total_report.add_key_at("num_kp", toString(total_kp[bow_idx]), report_data_key);
            // Report total_mask_pass
            total_report.add_key_at("mask_pass", toString(total_mask_pass[bow_idx]), report_data_key);
        }

        // Skip if no feature extracted
        if (query_bows[bow_idx].size() == 0)
        {
            cout << redc << "**Skip empty bin!**" << endc << endl;
            continue;
        }

        // Search
        cout << "==== Search Timing Info ====" << endl;
        startTime = CurrentPreciseTime();
        result_id = search_by_bowsig(query_bows[bow_idx], results[bow_idx]);
        searchTime = TimeElapse(startTime);
        cout << "Match with: " << TotalMatch << " videos" << endl;
        cout << "Search time: " <<  searchTime << " s" << endl;
        cout << "Result (dataset_id) : " << result_id << endl;

        /// Attache previous info to result
        if ((caller_id & CALLER_QE) == CALLER_QE)
            attache_resultinfo(results, bow_idx, caller_id);

        /// Matching dump
        if (run_param.matching_dump_enable)
        {
            cout << "Matches dumping.."; cout.flush();
            // prepare dump_ids, img_filenames
            vector<size_t> dump_ids;
            vector<string> img_roots;
            vector< vector<string> > img_filenames;
            size_t dump_limit = 200;
            if (dump_limit > results[bow_idx].size())
                dump_limit = results[bow_idx].size();
            for (size_t rank_idx = 0; rank_idx < dump_limit; rank_idx++)
            {
                vector<result_object> result = results[bow_idx];

                // dump_ids
                dump_ids.push_back(result[rank_idx].dataset_id);

                // img_root
                if (run_param.pooling_enable)
                    img_roots.push_back(ParentPaths[Pool2ParentsIdx[result[rank_idx].dataset_id]]); // shot parent path
                else
                    img_roots.push_back(ParentPaths[Img2ParentsIdx[result[rank_idx].dataset_id]]);     // image parent path

                // img_filenames
                img_filenames.push_back(vector<string>());
                vector<string>& curr_img_filename = img_filenames.back();
                if (run_param.pooling_enable) // Result of pool or result of image
                {
                    //dataset_path << run_param.dataset_root_dir << "/" << ParentPaths[Pool2ParentsIdx[result[index].first]] << "/" << ImgLists[Pool2ImagesIdxRange[result[index].first].first];
                    //rank_File << dataset_path.str() << "," << result[index].second << endl;
                    size_t shot_frame_start = Pool2ImagesIdxRange[result[rank_idx].dataset_id].first;
                    size_t shot_frame_end = Pool2ImagesIdxRange[result[rank_idx].dataset_id].second;
                    for (size_t frame_idx = shot_frame_start; frame_idx <= shot_frame_end; frame_idx++)
                        curr_img_filename.push_back(ImgLists[frame_idx]);               // image filename
                }
                else
                {
                    // image_id to pool_id, then pool_id to parent_id // previous work ParentPaths[Img2ParentsIdx[result[index].first]]
                    //dataset_path << run_param.dataset_root_dir << "/" << ParentPaths[Img2ParentsIdx[result[index].first]] << "/" << ImgLists[result[index].first];
                    //rank_File << dataset_path.str() << "," << result[index].second << endl;
                    curr_img_filename.push_back(ImgLists[result[rank_idx].dataset_id]);    // image filename
                }
            }
            inverted_hist.dump(queries[bow_idx] + ".dataset.dump", dump_ids, img_roots, img_filenames);
            cout << "done!" << endl;
        }

        if (run_param.submit_enable && str_contains(run_param.dataset_prefix, "ins201"))
            current_search_time += searchTime;

        // Report
        if (run_param.report_enable)
        {
            // Report bin_size
            total_report.add_key_at("match_dataset", toString(TotalMatch), report_data_key);
            // Report search_time
            total_report.add_key_at("search_time", toString(searchTime), report_data_key);

            // Rank
            for (size_t rank_idx = 0; rank_idx < 100; rank_idx++)
            {
                rank_report.add_key_at(toString(rank_idx), toString(results[bow_idx][rank_idx].dataset_id), report_data_key + "_id");
                rank_report.add_key_at(toString(rank_idx), toString(results[bow_idx][rank_idx].score), report_data_key + "_score");
            }
        }

        if (preset_query)
            cout << "Result (Q:V) : " << q_id << ":" << result_id << endl;

        display_rank(results[bow_idx]);
    }

    /// Calculate map
    float final_ap = 0;
    for (size_t result_idx = 0; result_idx < results.size(); result_idx++)
    {
        string query_filename = queries[result_idx];
        if (run_param.earlyfusion_enable)
            query_filename = get_directory(query_filename) + "/" + get_filename(get_directory(queries[result_idx])) + ".earlyfused";

        // Report key
        string report_data_key;
        if (run_param.report_enable)
        {
            if (run_param.earlyfusion_enable)
                report_data_key = QueryNameLists[q_id] + "_" + QueryNameLists[q_id] + ".earlyfused";
            else
            {
                if ((caller_id & CALLER_QB) == CALLER_QB || (caller_id & CALLER_QE) == CALLER_QE)
                    report_data_key = QueryNameLists[q_id] + "_" + get_filename(queries[result_idx]);
                else
                    report_data_key = QueryNameLists[q_id] + "_" + QueryImgLists[q_id][result_idx];
            }
        }

        // Skip if no feature extracted
        if (results[result_idx].size() == 0)
        {
            cout << redc << "**Skip empty result!** " << endc << greenc << get_filename(query_filename) << endc << endl;
            continue;
        }

        if (preset_query)
        {
            ExportEvalRank(results[result_idx], query_filename);

            // Export trec_eval format ranklist
            if (str_contains(run_param.dataset_prefix, "ins201"))
                ExportRank_Trec(results[result_idx], query_filename, simulated_session);

            // Computing map
            timespec eval_time = CurrentPreciseTime();
            float curr_ap = Compute_map(QueryNameLists[q_id]);

            // Checking with groundtruth file, then mask to result[].info
            CheckGroundtruth(results[result_idx], q_id);

            // Report map
            if (run_param.report_enable)
            {
                // Report eval_time
                total_report.add_key_at("eval_time", toString(TimeElapse(eval_time)), report_data_key);
                // Report map
                total_report.add_key_at("map", toString(curr_ap), report_data_key);
            }

            cout << "Sub query \"" << greenc << get_filename(query_filename) << endc << "\" [" << result_idx + 1 << "/" << results.size() << "] got map = " << setprecision(4) << redc << curr_ap << endc << endl;

            final_ap = curr_ap;
        }

        ExportRawRank(results[result_idx], query_filename, final_ap);
    }

    if ((caller_id & CALLER_NONE) == CALLER_NONE)
    {
        /// QE handling (RANK mode)
        if (run_param.qe_enable)
        {
            // Initializing
            vector<string> qe_query;

            /// Preparing QE Bow, for each queries

            // Release previous memory
            if (inlier_count_pack.size())
            {
                for (size_t result_idx = 0; result_idx < inlier_count_pack.size(); result_idx++)
                {
                    map<size_t, int>().swap(inlier_count_pack[result_idx]);
                    map<size_t, double>().swap(ransac_score_pack[result_idx]);
                }
                vector< map<size_t, int> >().swap(inlier_count_pack);
                vector< map<size_t, double> >().swap(ransac_score_pack);
            }
            inlier_count_pack.resize(results.size());
            ransac_score_pack.resize(results.size());
            vector< vector<bow_bin_object*> > qe_bowsigs(results.size());
            for (size_t result_idx = 0; result_idx < results.size(); result_idx++)
            {
                // Report new_data
                string report_data_key;
                if (run_param.report_enable)
                {
                    report_data_key = QueryNameLists[q_id] + "_" + get_filename(queries[result_idx]) + ".qe";

                    for (size_t query_idx = 0; query_idx < queries.size(); query_idx++)
                    {
                        total_report.new_data(report_data_key, QueryNameLists[q_id]);
                        rank_report.new_data(report_data_key + "_id", QueryNameLists[q_id]);
                        rank_report.new_data(report_data_key + "_score", QueryNameLists[q_id]);
                    }
                }

                qe_query.push_back(get_directory(queries[result_idx]) + "/" + get_filename(queries[result_idx]) + ".qe");

                cout << "Query Expansion..." << endl;
                timespec qeTime = CurrentPreciseTime();
                /// QE initialize
                qe qe_builder(run_param, inverted_hist.get_idf());

                /// Preparing
                qe_builder.add_bow_from_rank(results[result_idx], run_param.qe_topk);

                /// Set foreground bin from query
                qe_builder.set_query_fg(query_bows[result_idx]);

                /// QE Basic
                vector<int> inlier_count;
                vector<double> ransac_score;
                qe_builder.qe_basic(query_bows[result_idx], qe_bowsigs[result_idx], inlier_count, ransac_score);
                for (size_t rank_idx = 0; rank_idx < results[result_idx].size(); rank_idx++)
                {
                    // Stop adding info more than processed selected topk
                    if (rank_idx >= size_t(run_param.qe_topk))
                        break;

                    /// *** Bugs exist around Ransac info
                    inlier_count_pack[result_idx][results[result_idx][rank_idx].dataset_id] = inlier_count[rank_idx];
                    ransac_score_pack[result_idx][results[result_idx][rank_idx].dataset_id] = ransac_score[rank_idx];
                }

                /// Dump QE bow
                if (run_param.matching_dump_enable)
                {
                    // Init kp_dumper
                    kp_dumper dumper;

                    // Collect
                    size_t dataset_id = 0;  // QE, query number always 0
                    for (size_t bin_idx = 0; bin_idx < qe_bowsigs[result_idx].size(); bin_idx++)
                    {
                        bow_bin_object* bin = qe_bowsigs[result_idx][bin_idx];
                        for (size_t feature_idx = 0; feature_idx < bin->features.size(); feature_idx++)
                            dumper.collect_kp(dataset_id, bin->cluster_id, bin->weight, bin->fg, bin->features[feature_idx]->sequence_id, bin->features[feature_idx]->kp);
                            // sequence_id already assigned from QE add bow
                    }

                    // Dump
                    dumper.dump(qe_query.back() + ".query.dump", // Choose query_id 0 as dump name
                                vector<size_t>{dataset_id},
                                vector<string>{get_directory(qe_query.back())},
                                vector< vector<string> >{vector<string>{get_filename(qe_query.back())}});
                }

                // Clear result memory for later process
                vector<result_object>().swap(results[result_idx]);

                cout << "done! (in " << setprecision(2) << fixed << TimeElapse(qeTime) << " s)" << endl;
            }

            /// Send to bow mode query handling
            final_ap = query_handle_basic(qe_query, qe_bowsigs, results, q_id, CALLER_QE);

            // Release memory
            for (size_t bow_idx = 0; bow_idx < qe_bowsigs.size(); bow_idx++)
                bow_builder.release_bowsig(qe_bowsigs[bow_idx]);
            vector< vector<bow_bin_object*> >().swap(qe_bowsigs);
        }

        /// QB handling (RANK mode)
        if (run_param.qb_enable)
        {
            int qb_round = 0;
            string qb_working_path = get_directory(queries[0]);
            while (qb_round < run_param.qb_iteration)
            {
                // Initializing
                vector<string> qb_query;

                /// Preparing QB Bow, for each queries
                vector< vector<bow_bin_object*> > qb_bowsigs(results.size());
                for (size_t result_idx = 0; result_idx < results.size(); result_idx++)
                {
                    // Report new_data
                    string report_data_key;
                    if (run_param.report_enable)
                    {
                        report_data_key = QueryNameLists[q_id] + "_" + get_filename(queries[result_idx]) + ".qb_it" + toString(qb_round);

                        for (size_t query_idx = 0; query_idx < queries.size(); query_idx++)
                        {
                            total_report.new_data(report_data_key, QueryNameLists[q_id]);
                            rank_report.new_data(report_data_key + "_id", QueryNameLists[q_id]);
                            rank_report.new_data(report_data_key + "_score", QueryNameLists[q_id]);
                        }
                    }

                    qb_query.push_back(get_directory(queries[result_idx]) + "/" + get_filename(queries[result_idx]) + ".qb_it" + toString(qb_round));

                    /// QB initialize
                    qb qb_builder(run_param, inverted_hist.get_idf(), "rank2qb_" + get_filename(queries[result_idx]) + "_it" + toString(qb_round), qb_working_path);

                    /// Preparing
                    qb_builder.add_bow_from_rank(results[result_idx], run_param.qb_topk);
                    // Clear result memory for later process, qb next round or late fusion
                    vector<result_object>().swap(results[result_idx]);

                    /// Set foreground bin from query
                    qb_builder.set_query_fg(query_bows[result_idx]);

                    /// Check RANSAC HOMOGRAPHY for top-k
                    if (run_param.qe_ransac_enable)
                        qb_builder.topk_ransac_check(query_bows[result_idx]);

                    /// Mining bow
                    if (run_param.qb_mode == QB_FIM)
                        qb_builder.mining_fim_bow(run_param.qb_minsup, run_param.qb_maxsup, qb_bowsigs[result_idx]);
                    else if (run_param.qb_mode == QB_MAXPAT)
                        qb_builder.mining_maxpat_bow(qb_bowsigs[result_idx]);
                    else if (run_param.qb_mode == QB_MAXBIN)
                        qb_builder.mining_maxbin_bow(qb_bowsigs[result_idx]);

                    /// Dump QB bow
                    if (run_param.matching_dump_enable)
                    {
                        // Init kp_dumper
                        kp_dumper dumper;

                        // Collect
                        size_t dataset_id = 0;  // QB, query number always 0
                        for (size_t bin_idx = 0; bin_idx < qb_bowsigs[result_idx].size(); bin_idx++)
                        {
                            bow_bin_object* bin = qb_bowsigs[result_idx][bin_idx];
                            for (size_t feature_idx = 0; feature_idx < bin->features.size(); feature_idx++)
                                dumper.collect_kp(dataset_id, bin->cluster_id, bin->weight, bin->fg, bin->features[feature_idx]->sequence_id, bin->features[feature_idx]->kp);
                                /// sequence_id already assigned by QE add bow
                        }

                        // Dump
                        dumper.dump(qb_query.back() + ".query.dump", // Choose query_id 0 as dump name
                                    vector<size_t>{dataset_id},
                                    vector<string>{get_directory(qb_query.back())},
                                    vector< vector<string> >{vector<string>{get_filename(qb_query.back())}});
                    }
                }

                /// Send to bow mode query handling
                final_ap = query_handle_basic(qb_query, qb_bowsigs, results, q_id, CALLER_QB);

                // Next round
                qb_round++;

                // Release memory
                for (size_t bow_idx = 0; bow_idx < qb_bowsigs.size(); bow_idx++)
                    bow_builder.release_bowsig(qb_bowsigs[bow_idx]);
                vector< vector<bow_bin_object*> >().swap(qb_bowsigs);
            }
        }

        /// Handling late fusion
        if (run_param.latefusion_enable)
        {
            // Report new_data
            string report_data_key;
            if (run_param.report_enable)
            {
                report_data_key = QueryNameLists[q_id] + "_" + QueryNameLists[q_id] + ".latefused";

                for (size_t query_idx = 0; query_idx < queries.size(); query_idx++)
                {
                    total_report.new_data(report_data_key, QueryNameLists[q_id]);
                    rank_report.new_data(report_data_key + "_id", QueryNameLists[q_id]);
                    rank_report.new_data(report_data_key + "_score", QueryNameLists[q_id]);
                }
            }

            /// Processing late fusion
            vector<result_object> fused;
            late_fusion(results, fused);
            string fused_query_filename = get_directory(queries[0]) + "/" + QueryNameLists[q_id] + ".latefused";

            // Rank
            for (size_t rank_idx = 0; rank_idx < 100; rank_idx++)
            {
                rank_report.add_key_at(toString(rank_idx), toString(fused[rank_idx].dataset_id), report_data_key + "_id");
                rank_report.add_key_at(toString(rank_idx), toString(fused[rank_idx].score), report_data_key + "_score");
            }

            if (preset_query)
            {
                ExportEvalRank(fused, fused_query_filename);

                // Export trec_eval format ranklist
                if (str_contains(run_param.dataset_prefix, "ins201"))
                    ExportRank_Trec(fused, fused_query_filename, simulated_session);

                // Computing map
                float curr_ap = Compute_map(QueryNameLists[q_id]);

                // Checking with groundtruth file, then mask to fused[].info
                CheckGroundtruth(fused, q_id);

                cout << "Fused query got map = " << setprecision(4) << redc << curr_ap << endc << endl;

                // Report map
                if (run_param.report_enable)
                    total_report.add_key_at("map", toString(curr_ap), report_data_key);

                final_ap = curr_ap;
            }

            ExportRawRank(fused, fused_query_filename, final_ap);

            // Release old rank
            for (size_t result_idx = 0; result_idx < results.size(); result_idx++)
                vector<result_object>().swap(results[result_idx]);
            vector< vector<result_object> >().swap(results);
            // Keep fused rank
            results.push_back(fused);
        }
    }

    if (preset_query)
        cout << "Query \"" << greenc << QueryNameLists[q_id] << endc << "\" [" << q_id + 1 << "/" << QueryNameLists.size() << "] got map = " << setprecision(4) << redc << final_ap << endc << endl;

    return final_ap;
}

size_t search_by_bowsig(const vector<bow_bin_object*>& bow_sig, vector<result_object>& result)
{
    // Search method
    if (run_param.SIM_mode == SIM_GVP)
    {
        // search with gvp
        int* sim_param = new int[3];
        sim_param[0] = run_param.GVP_mode;
        sim_param[1] = run_param.GVP_size;
        sim_param[2] = run_param.GVP_length;
        TotalMatch = inverted_hist.search(bow_sig, result, run_param.SIM_mode, sim_param);
        delete[] sim_param;
    }
    else  // search with normal l1
        TotalMatch = inverted_hist.search(bow_sig, result);

    // Release inverted cache when reach counter
    // Only with normal search (not eval)
    if (!rotate_eval && rotate_memory_counter && rotate_memory_counter-- == 1)
    {
        inverted_hist.release_cache();
        rotate_memory_counter = rotate_limit;
    }

    // Filtering
    if (str_contains(run_param.dataset_prefix, "ins2013") || str_contains(run_param.dataset_prefix, "ins2014"))
    {
        vector<result_object> filtered_result;
        for (size_t result_idx = 0; result_idx < result.size(); result_idx++)
        {
            if (!dataset_skiplist[result[result_idx].dataset_id])
                filtered_result.push_back(result[result_idx]);
        }
        cout << yellowc << result.size() - filtered_result.size() << endc << " dataset(s) were skipped" << endl;
        result.swap(filtered_result);
    }

	return result[0].dataset_id;
}

void random_bowsig()
{
	cout << "Random bowsig" << endl;
	// Initial randbin
	vector<bool> randbin;
	for (size_t bin_idx = 0; bin_idx < run_param.CLUSTER_SIZE; bin_idx++)
        randbin.push_back(false);

	int total_bin = rand() % 1500 + 1000;
	//int totalhist = rand() % 15 + 5;
	cout << "Total bin: " << total_bin << endl;
	//cout << "First 5 bin" << endl;
	for(int bin_count = 0; bin_count < total_bin; bin_count++)
	{
	    // Random cluster_id
		int cluster_id;
		do
			cluster_id = (rand() % run_param.CLUSTER_SIZE);
		while(randbin[cluster_id]);
		randbin[cluster_id] = true;
		float val = (rand() % 100000) / 1000000000.0;
		bow_bin_object* bin_obj = new bow_bin_object();
		bin_obj->cluster_id = size_t(cluster_id);
		bin_obj->weight = float(val);

        // Random features
		int total_feature = rand() % 5 + 30;
		for (int feature_count = 0; feature_count < total_feature; feature_count++)
        {
            int ran_x = rand() % 10 + 100;
            int ran_y = rand() % 10 + 100;
            feature_object* ran_fea = new feature_object();
            ran_fea->kp = new float[SIFThesaff::GetSIFTHeadSize()];
            ran_fea->kp[0] = ran_x;
            ran_fea->kp[1] = ran_y;
            bin_obj->features.push_back(ran_fea);
        }

		// skip x y a b c
		rand_bowsig.push_back(bin_obj);
	}
}

void extract_bowsig(const string& session_name, const vector<string>& queries, vector< vector<bow_bin_object*> >& bow_sigs, const vector<string>& masks)
{
    // Release memory if needed
    for (size_t bow_idx = 0; bow_idx < bow_sigs.size(); bow_idx++)
        bow::release_bowsig(bow_sigs[bow_idx]);
    vector<int>().swap(total_kp);
    vector<int>().swap(total_mask_pass);

    stringstream querylist_path;
    querylist_path << run_param.online_working_path << "/" << simulated_session << "/" << run_param.querylist_filename;

    // Create query list file
    Properties<string> queryList;
    for (size_t query_idx = 0; query_idx < queries.size(); query_idx++)
    {
        stringstream sss;
        sss << "query[" << query_idx << "]";
        queryList.put(sss.str() + ".image", queries[query_idx]);
        if (masks.size())   // if provided mask
        {
            queryList.put(sss.str() + ".mask", masks[query_idx]);
            if (run_param.mask_mode == MASK_ROI || run_param.mask_mode == MASK_POLYGON)
                queryList.put(sss.str() + ".mask_type", "POLYGON");
            else    // MASK_IMG
                queryList.put(sss.str() + ".mask_type", "IMAGE");
        }
    }
    queryList.put("query.size", queries.size());
    queryList.save(querylist_path.str());

    // Write path to hist request
    lockfile(run_param.histrequest_path);
    ofstream histrequest_File (run_param.histrequest_path.c_str());
    if (histrequest_File.is_open())
    {
        // See ref send_query
        // Flag mask enable option
        // Dev:0|1 Ransac:0|1 Showall:0|1
        histrequest_File << session_name << " 0 0 0";
        histrequest_File.close();
    }
    unlockfile(run_param.histrequest_path);

    // Read extracted bow_sig
    size_t bow_idx = 0;
    do
    {
        // Report key
        string report_data_key;
        if (run_param.report_enable)
        {
            if (run_param.earlyfusion_enable)
                report_data_key = QueryNameLists[curr_q_id] + "_" + QueryNameLists[curr_q_id] + ".earlyfused";
            else
                report_data_key = QueryNameLists[curr_q_id] + "_" + QueryImgLists[curr_q_id][bow_idx];
        }

        vector<bow_bin_object*>& bow_sig = bow_sigs[bow_idx];

        // Waiting for HistXct result
        stringstream hist_load_path;
        if (run_param.earlyfusion_enable)
            hist_load_path << get_directory(queries[bow_idx]) << "/" << get_filename(get_directory(queries[bow_idx])) << ".earlyfused" << run_param.hist_postfix;
        else
            hist_load_path << queries[bow_idx] << run_param.hist_postfix;

        // Report bowsig_time
        timespec bowsig_time;
        if (run_param.report_enable)
            bowsig_time = CurrentPreciseTime();

        // Read bow bowsig
        if (run_param.reuse_bow_sig && is_path_exist(hist_load_path.str() + ".read"))
        {
            import_bowsig(hist_load_path.str() + ".read", bow_sig);
        }
        else
        {
            while(!is_path_exist(hist_load_path.str() + ".ok")) // Check ok file
            {
                usleep(5000);   // 5 milliseconds
                //ls2null(hist_load_path.str());
            }
            remove((hist_load_path.str() + ".ok").c_str()); // Remove .ok file

            import_bowsig(hist_load_path.str(), bow_sig);

            rename(hist_load_path.str().c_str(), string(hist_load_path.str() + ".read").c_str());
        }

        cout << "loaded: " << hist_load_path.str() << endl;

        // Report bowsig_time
        if (run_param.report_enable)
            total_report.add_key_at("bowsig_time", toString(TimeElapse(bowsig_time)), report_data_key);

        // next bow
        bow_idx++;
    }
    // Since early fusion has bow_sig only one
    // Continue reading the rest bow_sig if this is not early fusion
    while (!run_param.earlyfusion_enable && bow_idx < bow_sigs.size());
}

void import_bowsig(const string& in, vector<bow_bin_object*>& bow_sig)
{
    // Load InFile file
    ifstream InFile (in.c_str(), ios::binary);
    if (InFile)
    {
        /// Prepare buffer
        size_t buffer_size = 0;
        InFile.seekg(0, InFile.end);
        buffer_size = InFile.tellg();
        char* buffer = new char[buffer_size];
        char* buffer_ptr = buffer;
        InFile.seekg(0, InFile.beg);
        InFile.read(buffer, buffer_size);

        // Close file
        InFile.close();

        /// Bow hist
        // Non-zero count
        size_t bin_count = *((size_t*)buffer_ptr);
        buffer_ptr += sizeof(bin_count);

        // ClusterID and FeatureIDs
        int head_size = SIFThesaff::GetSIFTHeadSize();
        for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
        {
            // Create bin_obj
            bow_bin_object* read_bin = new bow_bin_object();

            // Cluster ID
            read_bin->cluster_id = *((size_t*)buffer_ptr);
            buffer_ptr += sizeof(read_bin->cluster_id);

            // Weight
            read_bin->weight = *((float*)buffer_ptr);
            buffer_ptr += sizeof(read_bin->weight);

            // Foreground flag
            read_bin->fg = *((bool*)buffer_ptr);
            buffer_ptr += sizeof(read_bin->fg);

            // Feature count
            size_t feature_count;
            feature_count = *((size_t*)buffer_ptr);
            buffer_ptr += sizeof(feature_count);
            for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
            {
                feature_object* feature = new feature_object();

                // Image ID
                feature->image_id = *((size_t*)buffer_ptr);
                buffer_ptr += sizeof(feature->image_id);
                // Sequence ID
                feature->sequence_id = *((size_t*)buffer_ptr);
                buffer_ptr += sizeof(feature->sequence_id);
                /*
                // Weight (asymmetric weight)
                feature->weight = *((float*)buffer_ptr);
                buffer_ptr += sizeof(feature->weight);
                */
                // x y a b c
                feature->kp = new float[head_size];
                for (int head_idx = 0; head_idx < head_size; head_idx++)
                {
                    feature->kp[head_idx] = *((float*)buffer_ptr);
                    buffer_ptr += sizeof(feature->kp[head_idx]);
                    //cout << feature->kp[head_idx] << " ";
                }
                //cout << endl;

                read_bin->features.push_back(feature);
            }

            // Keep bow
            bow_sig.push_back(read_bin);
        }

        // Read num_kp
        int num_kp = *((int*)buffer_ptr);
        buffer_ptr += sizeof(num_kp);

        // Read mask_pass
        int mask_pass = *((int*)buffer_ptr);
        buffer_ptr += sizeof(mask_pass);

        // Keep for each sig
        total_kp.push_back(num_kp);
        total_mask_pass.push_back(mask_pass);

        // Release buffer
        delete[] buffer;
    }
}

// Tools
void NoisyQuery(const string& query_path)
{
    string cmd;
    cout << "Adding noise size " << run_param.query_noise_amount << "..";
    cmd = "convert " + query_path + " -attenuate " + toString(run_param.query_noise_amount) + " +noise Gaussian " + query_path;
    cout.flush();
    timespec noise_time = CurrentPreciseTime();
    exec(cmd);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(noise_time) << " s)" << endl;
}

string ResizeQuery(const string& query_path)
{
    // Resize query
    string query_scaled_path = query_path + run_param.query_scale_postfix;
    string cmd;
    if (run_param.query_scale_type == SCALE_ABS)
    {
        cout << "Resize query to " << run_param.query_scale << "x" << run_param.query_scale;
        cmd = "convert " + query_path + " -resize " + toString(run_param.query_scale) + "x" + toString(run_param.query_scale) + "\\> " + query_scaled_path;
    }
    else
    {
        cout << "Resize query to " << run_param.query_scale << "%";
        cmd = "convert " + query_path + " -resize " + toString(run_param.query_scale) + "% " + query_scaled_path;
    }
    cout.flush();
    timespec resize_time = CurrentPreciseTime();
    exec(cmd);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(resize_time) << " s)" << endl;

    return query_scaled_path;
}

void RestoreQuery(const string& query_path)
{
    // Resize query
    string query_scaled_path = query_path + run_param.query_scale_postfix;
    string cmd;
    // Only support restoring from ratio mode
    if (run_param.query_scale_type == SCALE_RATIO)
    {
        cout << "Restore query to 100%";
        cmd = "convert " + query_scaled_path + " -resize " + toString(10000.0f / run_param.query_scale) + "% " + query_path;
    }
    else
    {
        cout << "Only support restoring from ratio mode" << endl;
        exit(1);
    }
    cout.flush();
    timespec resize_time = CurrentPreciseTime();
    exec(cmd);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(resize_time) << " s)" << endl;

    //return query_path;
}

void attache_resultinfo(vector< vector<result_object> >& results, size_t result_idx, const char caller_id)
{
    if ((caller_id & CALLER_QE) == CALLER_QE)
    {
        // Update previous ransac info to current result info
        for (size_t rank_idx = 0; rank_idx < results[result_idx].size(); rank_idx++)
        {
            // Stop adding info more than processed selected topk
            if (rank_idx >= size_t(run_param.qe_topk))
                break;

            if (inlier_count_pack[result_idx].find(results[result_idx][rank_idx].dataset_id) != inlier_count_pack[result_idx].end())
            {
                results[result_idx][rank_idx].info += "inlier:" + toString(inlier_count_pack[result_idx][results[result_idx][rank_idx].dataset_id]) + ";" +
                                                      "ransac_score:" + toString(ransac_score_pack[result_idx][results[result_idx][rank_idx].dataset_id]) + ";";
            }
        }
    }
}

// QB Mining
void FIW(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fi_weight)
{
    // Ref http://www.borgelt.net/apriori.html
    // Ref http://www.borgelt.net/fpgrowth.html
    string transaction_path = query_path + "_qb2_transaction.txt";
    string frequent_item_path = query_path + "_qb2_frequent_item.txt";
    string fpg_binary = "timeout -k 0h 2m /home/stylix/webstylix/code/datamining/fpgrowth";

    /// Step 1
    // Convert BOW to transaction
    stringstream transaction_buffer;
    bool with_query = false;
    if (with_query)
    {
        for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
            transaction_buffer << query_bow[bin_id]->cluster_id << " ";
        transaction_buffer << endl;
    }
    for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            transaction_buffer << bow_sigs[bow_id][bin_id]->cluster_id << " ";
        transaction_buffer << endl;
    }
    text_write(transaction_path, transaction_buffer.str(), false);

    /// Step 2
    // Find Frequent Item Set by FP-growth
    // output item set foprmat a,d,b:80
    // -tm = maximal
    // -k, output seperator
    // -v, support value seperator
    // -s lowerbound support value
    string fpg_cmd = fpg_binary + " -tc -k, -v,%S%% -s" + toString(minsup) + " " + transaction_path + " " + frequent_item_path;// + COUT2NULL;
    exec(fpg_cmd);
    /// Step 3
    // Calculate item weight from FIM
    // FIM to FIW
    unordered_map<size_t, size_t> frequent_item_count;
    unordered_map<size_t, float> frequent_item_weight;
    vector<string> frequent_item_sets = text_readline2vector(frequent_item_path);
    float fim_support;
    for (size_t set_id = 0; set_id < frequent_item_sets.size(); set_id++)
    {
        // Accumulate support score for each item
        vector<string> Items;   // Item [0-(n-2)], Support [n-1]
        StringExplode(frequent_item_sets[set_id], ",", Items);

        // Skip last fim if fpgrowth killed detected
        if (!str_contains(Items[Items.size() - 1], "%"))
        {
            cout << "Skip last frequent item set caused by killed!" << endl;
            cout << "FIM output not yet done, but memory not enough!!" << endl;
            break;
        }

        fim_support = atoi(str_replace_last(Items[Items.size() - 1], "%", "").c_str()) * 0.01;  // convert to 0-1 scale
        for (size_t item_idx = 0; item_idx < Items.size() - 1; item_idx++)
        {
            size_t item = strtoull(Items[item_idx].c_str(), NULL, 0);
            if (frequent_item_weight.find(item) == frequent_item_weight.end())
            {
                frequent_item_count[item] = 0;
                frequent_item_weight[item] = 0;
            }
            frequent_item_count[item]++;
            frequent_item_weight[item] += fim_support;
        }
    }
    for (unordered_map<size_t, size_t>::iterator frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
    {
        // Normalize support score by number of occurance
        size_t item = frequent_item_count_it->first;
        //cout << "item: " << item << "   raw_weight: " << frequent_item_weight[item] << "    norm_weight: ";
        frequent_item_weight[item] /= frequent_item_count[item];
        //cout << frequent_item_weight[item] << " count: " << frequent_item_count[item] << endl;
    }
    frequent_item_weight.swap(fi_weight);
}

void PREFIW(const string& query_path, int minsup, unordered_map<size_t, float>& fi_weight)
{
    // Ref http://www.borgelt.net/apriori.html
    // Ref http://www.borgelt.net/fpgrowth.html
    string frequent_item_path = query_path + "_qb2_frequent_item.txt";
    /// Step 3
    // Calculate item weight from FIM
    // FIM to FIW
    unordered_map<size_t, size_t> frequent_item_count;
    unordered_map<size_t, float> frequent_item_weight;
    vector<string> frequent_item_sets = text_readline2vector(frequent_item_path);
    for (size_t set_id = 0; set_id < frequent_item_sets.size(); set_id++)
    {
        // Accumulate support score for each item
        vector<string> Items;   // Item [0-(n-2)], Support [n-1]
        StringExplode(frequent_item_sets[set_id], " ", Items);

        for (size_t item_idx = 0; item_idx < Items.size(); item_idx++)
        {
            size_t item = strtoull(Items[item_idx].c_str(), NULL, 0);
            if (frequent_item_weight.find(item) == frequent_item_weight.end())
            {
                frequent_item_count[item] = 0;
                frequent_item_weight[item] = 0;
            }
            frequent_item_count[item]++;
            frequent_item_weight[item] = frequent_item_count[item];
        }
    }
    // Find max
    size_t item_count_max = 0;
    unordered_map<size_t, size_t>::iterator frequent_item_count_it;
    for (frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
    {
        if (item_count_max < frequent_item_count_it->second)
            item_count_max = frequent_item_count_it->second;
    }
    // Normalize and filtering
    unordered_map<size_t, float>::iterator frequent_item_weight_it;
    for (frequent_item_weight_it = frequent_item_weight.begin(); frequent_item_weight_it != frequent_item_weight.end(); frequent_item_weight_it++)
    {
        frequent_item_weight_it->second /= item_count_max;  // normalize
        if (frequent_item_weight_it->second < minsup / 100.0)
            frequent_item_weight_it->second = 0;            // not pass minsup, set zero
    }
    frequent_item_weight.swap(fi_weight);
}

void FIX(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id]->cluster_id] = true;

    // Calculate item weight from ACW
    unordered_map<size_t, size_t> frequent_item_count;      // cluster_id, count
    unordered_map<size_t, float> frequent_item_weight;      // cluster_id, weight
    for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

            // Skip if not fall within query
            //if (!query_bin_mask[cluster_id])
                //continue;

            if (frequent_item_count.find(cluster_id) == frequent_item_count.end())
            {
                frequent_item_count[cluster_id] = 0;
                frequent_item_weight[cluster_id] = 0;
            }
            frequent_item_count[cluster_id]++;
            frequent_item_weight[cluster_id]++;
        }
    }
    // Find max
    size_t item_count_max = 0;
    unordered_map<size_t, size_t>::iterator frequent_item_count_it;
    for (frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
    {
        if (item_count_max < frequent_item_count_it->second)
            item_count_max = frequent_item_count_it->second;
    }

    // Normalize and filtering
    unordered_map<size_t, float>::iterator frequent_item_weight_it;
    for (frequent_item_weight_it = frequent_item_weight.begin(); frequent_item_weight_it != frequent_item_weight.end(); frequent_item_weight_it++)
    {
        frequent_item_weight_it->second /= item_count_max;  // normalize
        if (frequent_item_weight_it->second < minsup / 100.0f)
            frequent_item_weight_it->second = 0;            // not pass minsup, set zero
    }
    frequent_item_weight.swap(fi_weight);
}

void GLOSD(const string& query_path, const vector<bow_bin_object*>& query_bow, const vector< vector<bow_bin_object*> >& bow_sigs, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id]->cluster_id] = true;

    // Calculate count for each class
    unordered_map<size_t, size_t> frequent_item_count;      // cluster_id, count
    unordered_map<size_t, float> frequent_item_weight;      // cluster_id, weight
    for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

            // Skip if not fall within query
            //if (!query_bin_mask[cluster_id])
                //continue;

            if (frequent_item_count.find(cluster_id) == frequent_item_count.end())
            {
                frequent_item_count[cluster_id] = 0;
                frequent_item_weight[cluster_id] = 0;
            }
            frequent_item_count[cluster_id]++;
            frequent_item_weight[cluster_id]++;
        }
    }
    // Find max
    size_t item_count_max = 0;
    unordered_map<size_t, size_t>::iterator frequent_item_count_it;
    for (frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
    {
        if (item_count_max < frequent_item_count_it->second)
            item_count_max = frequent_item_count_it->second;
    }

/*
    // Normalize and filtering
    unordered_map<size_t, float>::iterator frequent_item_weight_it;
    for (frequent_item_weight_it = frequent_item_weight.begin(); frequent_item_weight_it != frequent_item_weight.end(); frequent_item_weight_it++)
    {
        frequent_item_weight_it->second /= item_count_max;  // normalize
        if (frequent_item_weight_it->second < minsup / 100.0f)
            frequent_item_weight_it->second = 0;            // not pass minsup, set zero
    }
    frequent_item_weight.swap(fi_weight);
*/


    /// Cut test
    stringstream pass_pcut;
    int bin_pass;
    // Write rank_id
    pass_pcut << "rank_id,";
    for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        pass_pcut << bow_id << ",";
    pass_pcut << "all_bin" << endl;
    // Write total bin
    pass_pcut << "total_bin,";
    for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        pass_pcut << bow_sigs[bow_id].size() << ",";
    pass_pcut << frequent_item_count.size() << endl;

    // Write cut for each rank
    // pcut is percentage
    vector< vector<float> > total_sd;
    int pcut_step = 5;
    for (int pcut = pcut_step; pcut <= 100; pcut += pcut_step)
    {
        pass_pcut << pcut << ",";
        /// -- Pass test
        unordered_map<size_t, size_t> bin_pass_count;       // cluster_id, pass_count
        unordered_map<size_t, bool> bin_pass_mask;          // cluster_id, pass_flag
        for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        {
            bin_pass = 0;
            for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                bool pass_cut = float(frequent_item_count[cluster_id]) / item_count_max >= pcut / 100.0f;

                if (pass_cut)
                {
                    if (bin_pass_count.find(cluster_id) == bin_pass_count.end())
                        bin_pass_count[cluster_id] = 0;
                    bin_pass_count[cluster_id]++;
                    bin_pass++; // bin pass without mask
                    bin_pass_mask[cluster_id] = pass_cut;
                }
                else
                    bin_pass_mask[cluster_id] |= pass_cut;
            }
            pass_pcut << bin_pass << ",";
        }
        pass_pcut << bin_pass_count.size() << ",";

        /// -- Compactness test
        // compactness : It is the sum of squared distance from each point to their corresponding centers.
        for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        {
            size_t total_feature_pass_count = 0;

            // Accumulate total feature pass count
            for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                if (bin_pass_mask[cluster_id])
                    total_feature_pass_count += bow_sigs[bow_id][bin_id]->features.size();
            }

            // Continue finding SD if we have vertex pass p_cut more than 3 points
            // 1. Finding centroid
            // 2. Packing distance to data, then get mean
            // 3. Calc SD by sd_with_premean

            float rank_sd = 0.0f;
            if (total_feature_pass_count > 3)
            {
                size_t vert_count = 0;
                float* vertx = new float[total_feature_pass_count];
                float* verty = new float[total_feature_pass_count];
                float rank_centroid[2] = {0.0f, 0.0f};                                      // Centroid for this rank
                float* rank_dist_to_centroid = new float[total_feature_pass_count];
                float rank_dist_mean = 0.0f;

                // 1. Finding centroid
                for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
                {
                    size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                    // Skip if not fall within query
                    //if (!query_bin_mask[cluster_id])
                        //continue;

                    if (bin_pass_mask[cluster_id])
                    {
                        for (size_t feature_id = 0; feature_id < bow_sigs[bow_id][bin_id]->features.size(); feature_id++)
                        {
                            rank_centroid[0] += vertx[vert_count] = bow_sigs[bow_id][bin_id]->features[feature_id]->kp[0];    // Accumulating centroid
                            rank_centroid[1] += verty[vert_count] = bow_sigs[bow_id][bin_id]->features[feature_id]->kp[1];
                            vert_count++;
                        }
                    }
                }
                // Calculate centroid
                rank_centroid[0] /= vert_count;
                rank_centroid[1] /= vert_count;

                // 2. Packing distance to data, then get mean
                for (size_t vert_id = 0; vert_id < vert_count; vert_id++)
                {
                    rank_dist_to_centroid[vert_id] = sqrt((vertx[vert_id] - rank_centroid[0]) * (vertx[vert_id] - rank_centroid[0]) +
                                                          (verty[vert_id] - rank_centroid[1]) * (verty[vert_id] - rank_centroid[1]));
                    //cout << "Dist: " << rank_dist_to_centroid[vert_id] << endl;
                    rank_dist_mean += rank_dist_to_centroid[vert_id];       // Accumulating mean
                }
                // Calculating mean
                rank_dist_mean /= vert_count;


                // 3. Calculate by SD_PREMEAN
                rank_sd = calc_sd_premean(rank_dist_to_centroid, vert_count, rank_dist_mean);

                /*stringstream path_out;
                path_out << "/home/stylix/webstylix/code/ins_online/out_test/group_img_out_" << bow_id << "_" << pcut << "_lnj.png";
                cout << "Write out image.."; cout.flush();
                imwrite(path_out.str(), group_img_out);
                cout << "done" << endl;*/

                Mat test_img_out(1000, 1000, CV_8UC3, Scalar(0));

/*
                // Draw line
                for (size_t point_idx = 0; point_idx < total_feature_pass_count - 1; point_idx++)
                {
                    size_t node_idx = best_path[point_idx];
                    size_t next_node_idx = best_path[point_idx + 1];

                    Point2i pstart(vertx[node_idx] * 1000, verty[node_idx] * 1000);
                    Point2i pend(vertx[next_node_idx] * 1000, verty[next_node_idx] * 1000);

                    // Draw line only pass threshold
                    if (path_solver.get_cost_between(node_idx, next_node_idx) <= threshold)
                        line(test_img_out, pstart, pend, Scalar(0, 0, 255), 1, CV_AA);

                    circle(test_img_out, pstart, 0, Scalar(0, 255, 0), 2, CV_AA);
                    if (point_idx == total_feature_pass_count - 2)
                        circle(test_img_out, pend, 0, Scalar(0, 255, 0), 2, CV_AA);
                }
*/

                /*stringstream path2_out;
                path2_out << "/home/stylix/webstylix/code/ins_online/out_test/group_img_out_" << bow_id << "_" << pcut << "_lnj_p.png";
                cout << "Write out image.."; cout.flush();
                imwrite(path2_out.str(), test_img_out);
                cout << "done" << endl;*/

                // Release mem
                delete[] vertx;
                delete[] verty;
                delete[] rank_dist_to_centroid;
            }

            // Save Compactness for finding best minsup later
            if (bow_id + 1 > total_sd.size())
            {
                vector<float> sub_total_sd;
                sub_total_sd.push_back(rank_sd);
                total_sd.push_back(sub_total_sd);
            }
            else
                total_sd[bow_id].push_back(rank_sd);

            // Print out SD
            pass_pcut << rank_sd << ",";
        }
        // pass one cut, continue next cut
        pass_pcut << endl;
    }

    // Write to file
    text_write(query_path + "_rankcut.csv", pass_pcut.str(), false);

    // Total SD Debug
    cout << setprecision(5) << fixed;
    for (size_t bow_id = 0; bow_id < total_sd.size(); bow_id++)
    {
        for (size_t cut_id = 0; cut_id < total_sd[bow_id].size(); cut_id++)
        {
            cout << total_sd[bow_id][cut_id] << " ";
        }
        cout << endl;
    }


    if (true)
    {
        /// Finding the best p_cut for this top-k_rank
        /*
        Step to find the best pCut
        1. Calculate abs-diff of sd for each top-k
        2. Calculate avg of abs-diff from all top-k rank
        3. Find abs-slope
        4. Find lowest abs-slope is to find the highest stability of function
        */

        // 1-2 // Find avg-abs-diff for each sd_step
        vector<float> avg_abs_diff_sd;
        vector<size_t> avg_abs_diff_sd_label;
        for (size_t pcut_id = 1; pcut_id < total_sd[0].size(); pcut_id++)
        {
            float curr_avg = 0.0f;
            size_t non_zero_count = 0;
            for (size_t bow_id = 0; bow_id < total_sd.size(); bow_id++)
            {
                float curr_diff = total_sd[bow_id][pcut_id - 1] - total_sd[bow_id][pcut_id];
                if (curr_diff != 0)
                {
                    curr_avg += abs(curr_diff);
                    non_zero_count++;
                }
            }
            if (non_zero_count > 0)
            {
                avg_abs_diff_sd.push_back(curr_avg / non_zero_count);
                avg_abs_diff_sd_label.push_back(pcut_id * pcut_step);     // pcut start from pcut_step 1
            }
        }
        cout << "avg_abs_diff_sd_label.size() : " << avg_abs_diff_sd_label.size() << endl;
        for (size_t avg_id = 0; avg_id < avg_abs_diff_sd_label.size(); avg_id++)
            cout << avg_abs_diff_sd_label[avg_id] << "\t"; cout.flush();
        cout << endl;
        for (size_t avg_id = 0; avg_id < avg_abs_diff_sd.size(); avg_id++)
            cout << avg_abs_diff_sd[avg_id] << "\t"; cout.flush();
        cout << endl;

        // 3 calc slope
        vector<float> slope_avg_abs_diff_sd;
        vector<size_t> slope_avg_abs_diff_sd_label;
        if (avg_abs_diff_sd.size() >= 3)
        {
            //for (size_t diff_id = 2; diff_id < avg_abs_diff_sd.size() - 2; diff_id++)
            for (size_t diff_id = 1; diff_id < avg_abs_diff_sd.size() - 1; diff_id++)
            //for (size_t diff_id = 1; diff_id < avg_abs_diff_sd.size(); diff_id++)
            {
                // Packing data to find slope
                /*float* x_data = new float[5];
                float* y_data = new float[5];
                x_data[0] = diff_id - 2;
                x_data[1] = diff_id - 1;
                x_data[2] = diff_id;
                x_data[3] = diff_id + 1;
                x_data[4] = diff_id + 2;
                y_data[0] = avg_abs_diff_sd[diff_id - 2];
                y_data[1] = avg_abs_diff_sd[diff_id - 1];
                y_data[2] = avg_abs_diff_sd[diff_id];
                y_data[3] = avg_abs_diff_sd[diff_id + 1];
                y_data[4] = avg_abs_diff_sd[diff_id + 2];
                */
                float* x_data = new float[3];
                float* y_data = new float[3];
                x_data[0] = diff_id - 1;
                x_data[1] = diff_id;
                x_data[2] = diff_id + 1;
                y_data[0] = avg_abs_diff_sd[diff_id - 1];
                y_data[1] = avg_abs_diff_sd[diff_id];
                y_data[2] = avg_abs_diff_sd[diff_id + 1];
                /*float* x_data = new float[2];
                float* y_data = new float[2];
                x_data[0] = diff_id - 1;
                x_data[1] = diff_id;
                y_data[0] = avg_abs_diff_sd[diff_id - 1];
                y_data[1] = avg_abs_diff_sd[diff_id];*/

                //slope_avg_abs_diff_sd.push_back(abs(calc_slope(x_data, y_data, 5)));
                slope_avg_abs_diff_sd.push_back(abs(calc_slope(x_data, y_data, 3)));
                //slope_avg_abs_diff_sd.push_back(abs(calc_slope(x_data, y_data, 2)));
                slope_avg_abs_diff_sd_label.push_back(avg_abs_diff_sd_label[diff_id]);

                // Release memory
                delete[] x_data;
                delete[] y_data;
            }
        }
        cout << "slope_avg_abs_diff_sd_label.size() : " << slope_avg_abs_diff_sd_label.size() << endl;
        for (size_t slope_id = 0; slope_id < slope_avg_abs_diff_sd_label.size(); slope_id++)
            cout << slope_avg_abs_diff_sd_label[slope_id] << "\t"; cout.flush();
        cout << endl;
        for (size_t slope_id = 0; slope_id < slope_avg_abs_diff_sd.size(); slope_id++)
            cout << slope_avg_abs_diff_sd[slope_id] << "\t"; cout.flush();
        cout << endl;

        // Find the most lowest slope
        float min_slope = 1000000.0f;
        size_t min_slope_id = 0;
        for (size_t slope_id = 0; slope_id < slope_avg_abs_diff_sd.size(); slope_id++)
        {
            if (min_slope > slope_avg_abs_diff_sd[slope_id])
            {
                min_slope = slope_avg_abs_diff_sd[slope_id];
                min_slope_id = slope_id;
            }
        }

        size_t good_pcut = 1;  // default, low default in case less similarity for top-k rank, then minsup should be lower
        if (slope_avg_abs_diff_sd.size() > 0)
            good_pcut = slope_avg_abs_diff_sd_label[min_slope_id];  // start from pcut + 1

        cout << "Debug good_pcut: " << good_pcut << endl;

        // Calculate item weight from ACW
        unordered_map<size_t, size_t>().swap(frequent_item_count);
        unordered_map<size_t, float>().swap(frequent_item_weight);
        for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        {
            for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                if (frequent_item_count.find(cluster_id) == frequent_item_count.end())
                {
                    frequent_item_count[cluster_id] = 0;
                    frequent_item_weight[cluster_id] = 0;
                }
                frequent_item_count[cluster_id]++;
                frequent_item_weight[cluster_id]++;
            }
        }
        // Find max
        item_count_max = 0;
        //unordered_map<size_t, size_t>::iterator frequent_item_count_it;
        for (frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
        {
            if (item_count_max < frequent_item_count_it->second)
                item_count_max = frequent_item_count_it->second;
        }

        // Normalize and filtering
        unordered_map<size_t, float>::iterator frequent_item_weight_it;
        for (frequent_item_weight_it = frequent_item_weight.begin(); frequent_item_weight_it != frequent_item_weight.end(); frequent_item_weight_it++)
        {
            frequent_item_weight_it->second /= item_count_max;          // normalize by max of cluster_frequency
            if (frequent_item_weight_it->second < good_pcut / 100.0f)   // If auto minsup is false -> not pass
                frequent_item_weight_it->second = 0;                    // not pass minsup, set zero
        }
        frequent_item_weight.swap(fi_weight);
    }
    else
    {
        /// Finding the best p_cut for each rank_id
        /*
        Step to find the best pCut for each toprank
        1. Get abs-SD different, for finding the stability of varience(SD)
        2. Find abs-slope
        3. Find lowest abs-slope is to find the highest stability of function
        */
        vector<size_t> good_pcut;
        for (size_t bow_id = 0; bow_id < total_sd.size(); bow_id++)
        {
            // Find different for each sd_step
            vector<float> sd_diff;
            vector<size_t> pcut_sd_diff_label;
            float prev_diff = 0.0f;
            for (size_t pcut_id = 1; pcut_id < total_sd[bow_id].size(); pcut_id++)
            {
                float curr_diff = abs(total_sd[bow_id][pcut_id - 1] - total_sd[bow_id][pcut_id]);

                if (curr_diff != 0 && prev_diff != curr_diff)
                {
                    sd_diff.push_back(curr_diff);
                    pcut_sd_diff_label.push_back(pcut_id * pcut_step);     // pcut start from pcut_step 1

                    prev_diff = curr_diff;
                }
            }

            // Find slope for each sd_step
            vector<float> sd_diff_slope;
            vector<size_t> pcut_sd_diff_slope_label;
            for (size_t sd_id = 1; sd_id < sd_diff.size() - 1; sd_id++)
            {
                // Packing data to find slope
                float* x_data = new float[3];
                float* y_data = new float[3];
                x_data[0] = sd_id - 1;
                x_data[1] = sd_id;
                x_data[2] = sd_id + 1;
                y_data[0] = sd_diff[sd_id - 1];
                y_data[1] = sd_diff[sd_id];
                y_data[2] = sd_diff[sd_id + 1];
                /*float* x_data = new float[2];
                float* y_data = new float[2];
                x_data[0] = sd_id;
                x_data[1] = sd_id + 1;
                y_data[0] = sd_diff[sd_id];
                y_data[1] = sd_diff[sd_id + 1];*/

                sd_diff_slope.push_back(abs(calc_slope(x_data, y_data, 3)));
                //sd_diff_slope.push_back(abs(calc_slope(x_data, y_data, 2)));
                pcut_sd_diff_slope_label.push_back(pcut_sd_diff_label[sd_id]);

                // Release memory
                delete[] x_data;
                delete[] y_data;
            }

            // Find the most lowest slope
            float min_slope = 1000000.0f;
            size_t min_slope_id = 0;
            for (size_t sd_id = 0; sd_id < sd_diff_slope.size(); sd_id++)
            {
                if (min_slope > sd_diff_slope[sd_id])
                {
                    min_slope = sd_diff_slope[sd_id];
                    min_slope_id = sd_id;
                }
            }

            // Save good p_cut for this rank
            if (pcut_sd_diff_slope_label.size() > 0)    // Skip if cannot find any slope, data not enough
                good_pcut.push_back(pcut_sd_diff_slope_label[min_slope_id]);
            else
                good_pcut.push_back(1);                 // Use default

            // Release memory
            vector<float>().swap(sd_diff);
            vector<size_t>().swap(pcut_sd_diff_label);
            vector<float>().swap(sd_diff_slope);
            vector<size_t>().swap(pcut_sd_diff_slope_label);
        }

        // Debut best_pcut
        cout << "Debug good pcut: ";
        for (size_t bow_id = 0; bow_id < good_pcut.size(); bow_id++)
            cout << good_pcut[bow_id] << " ";
        cout << endl;


        // Calculate item weight from ACW
        unordered_map<size_t, size_t>().swap(frequent_item_count);
        unordered_map<size_t, float>().swap(frequent_item_weight);
        for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        {
            for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                if (frequent_item_count.find(cluster_id) == frequent_item_count.end())
                {
                    frequent_item_count[cluster_id] = 0;
                    frequent_item_weight[cluster_id] = 0;
                }
                frequent_item_count[cluster_id]++;
                frequent_item_weight[cluster_id]++;
            }
        }
        // Find max
        item_count_max = 0;
        //unordered_map<size_t, size_t>::iterator frequent_item_count_it;
        for (frequent_item_count_it = frequent_item_count.begin(); frequent_item_count_it != frequent_item_count.end(); frequent_item_count_it++)
        {
            if (item_count_max < frequent_item_count_it->second)
                item_count_max = frequent_item_count_it->second;
        }

        // Cut
        bool* auto_minsup_filter = new bool[run_param.CLUSTER_SIZE];
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            auto_minsup_filter[cluster_id] = false;
        for (size_t bow_id = 0; bow_id < bow_sigs.size(); bow_id++)
        {
            bin_pass = 0;
            for (size_t bin_id = 0; bin_id < bow_sigs[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sigs[bow_id][bin_id]->cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                auto_minsup_filter[cluster_id] |= float(frequent_item_count[cluster_id]) / item_count_max >= good_pcut[bow_id] / 100.0f;
            }
        }

        // Normalize and filtering
        unordered_map<size_t, float>::iterator frequent_item_weight_it;
        for (frequent_item_weight_it = frequent_item_weight.begin(); frequent_item_weight_it != frequent_item_weight.end(); frequent_item_weight_it++)
        {
            frequent_item_weight_it->second /= item_count_max;          // normalize by max of cluster_frequency
            if (!auto_minsup_filter[frequent_item_weight_it->first])    // If auto minsup is false -> not pass
                frequent_item_weight_it->second = 0;                    // not pass minsup, set zero
        }
        frequent_item_weight.swap(fi_weight);
    }

    // Release memory
    delete[] query_bin_mask;
    //delete[] auto_minsup_filter;

}

// Ranklist combination
void late_fusion(const vector< vector<result_object> >& results, vector<result_object>& fused)
{
    size_t results_size = results.size();
    unordered_map<size_t, float> fused_dataset_score;

    // Normalization - Fusion
    for (size_t results_idx = 0; results_idx < results_size; results_idx++)
    {
        const vector<result_object>& result = results[results_idx];
        size_t result_size = result.size();
        vector<float> score(result_size, 0.0f);

        /// Normalization
        // Unit length
        float sum_of_square = 0.0f;
        float unit_length = 0.0f;
        for (size_t dataset_idx = 0; dataset_idx < result_size; dataset_idx++)
            sum_of_square += result[dataset_idx].score * result[dataset_idx].score;
        unit_length = sqrt(sum_of_square);

        // Normalizing, then keep
        for (size_t dataset_idx = 0; dataset_idx < result_size; dataset_idx++)
            score[dataset_idx] = result[dataset_idx].score / unit_length;

        /// Fusion
        if (run_param.latefusion_mode == LATEFUSION_SUM || run_param.latefusion_mode == LATEFUSION_AVG)
        {
            // Sum score
            for (size_t dataset_idx = 0; dataset_idx < result_size; dataset_idx++)
            {
                size_t dataset_id = result[dataset_idx].dataset_id;
                if (fused_dataset_score.find(dataset_id) == fused_dataset_score.end())
                    fused_dataset_score[dataset_id] = 0.0f;
                fused_dataset_score[dataset_id] += score[dataset_idx];
            }
        }
        else if (run_param.latefusion_mode == LATEFUSION_MAX)
        {
            // Max score
            for (size_t dataset_idx = 0; dataset_idx < result_size; dataset_idx++)
            {
                size_t dataset_id = result[dataset_idx].dataset_id;
                // Initial rank
                if (fused_dataset_score.find(dataset_id) == fused_dataset_score.end())
                    fused_dataset_score[dataset_id] = 0;
                // Max
                if (fused_dataset_score[dataset_id] < score[dataset_idx])
                    fused_dataset_score[dataset_id] = score[dataset_idx];
            }
        }

        // Release memory
        vector<float>().swap(score);
    }

    // Average score
    if (run_param.latefusion_mode == LATEFUSION_AVG)
    {
        // Average score
        for (auto score_it = fused_dataset_score.begin(); score_it != fused_dataset_score.end(); score_it++)
            score_it->second /= results_size;
    }

    // Prepare
    vector< pair<result_object, float> > working_fused;
    for (auto score_it = fused_dataset_score.begin(); score_it != fused_dataset_score.end(); score_it++)
        working_fused.push_back(pair<result_object, float>(result_object{score_it->first, score_it->second, ""}, score_it->second));

    // Sorting
    sort(working_fused.begin(), working_fused.end(), compare_pair_second<>());

    // Saving
    for (auto working_fused_it = working_fused.begin(); working_fused_it != working_fused.end(); working_fused_it++)
        fused.push_back(working_fused_it->first);
}

// Export result and visualization
void display_rank(const vector<result_object>& result)
{
	int count = result.size();
	int max = 5;
	if(max > count)
        max = count;
	for(int index = 0; index < max; index++)
		cout << "dataset_id:" << result[index].dataset_id << fixed << " Value:" << result[index].score << endl;
}

// Evaluation
void Evaluate()
{
    // Set rotate to eval mode
    rotate_eval = true;

    /// Oxford building dataset
    if (str_contains(run_param.dataset_prefix, "oxbuildings") ||
		str_contains(run_param.dataset_prefix, "paris") ||
		str_contains(run_param.dataset_prefix, "smalltest"))
    {
        float sum_map = 0.0f;
        float avg_map = 0.0f;

        for (size_t q_id = 0; q_id < QueryNameLists.size(); q_id++)
        {
            // Searching
            float curr_ap = search_by_id(q_id);

            sum_map += curr_ap;

            // Release inverted cache when reach counter
            // Only with eval mode
            if (rotate_eval && rotate_memory_counter && rotate_memory_counter-- == 1)
            {
                inverted_hist.release_cache();
                rotate_memory_counter = rotate_limit;
            }
        }

        avg_map = sum_map / query_topic_amount;
        cout << "Configuration name \"" << run_param.dataset_prefix << "\"" << endl;
        cout << "Total map " << setprecision(4) << fixed << redc << avg_map << endc << endl;
    }
    else if (str_contains(run_param.dataset_prefix, "ins201"))
    {
        float sum_map = 0.0f;
        float avg_map = 0.0f;

        /// Submit info
        vector<string> trecrank_paths;
        vector<double> search_times;

        for (size_t q_id = 0; q_id < QueryNameLists.size(); q_id++)
        {
            if (run_param.submit_enable && str_contains(run_param.dataset_prefix, "ins201"))
                current_search_time = 0.0f;

            // Searching
            float curr_ap = search_by_id(q_id);

            sum_map += curr_ap;

            // Keep submit info (from last search)
            if (run_param.submit_enable && str_contains(run_param.dataset_prefix, "ins201"))
            {
                trecrank_paths.push_back(trecrank_path.str());
                search_times.push_back(current_search_time);
            }

            // Release inverted cache when reach counter
            // Only with eval mode
            if (rotate_eval && rotate_memory_counter && rotate_memory_counter-- == 1)
            {
                inverted_hist.release_cache();
                rotate_memory_counter = rotate_limit;
            }
        }

        avg_map = sum_map / query_topic_amount;
        cout << "Configuration name \"" << run_param.dataset_prefix << "\"" << endl;
        cout << "Total map " << setprecision(4) << fixed << redc << avg_map << endc << endl;

        /// INS Submit
        if (run_param.submit_enable && str_contains(run_param.dataset_prefix, "ins201"))
        {
            SubmitRank_Trec(trecrank_paths, search_times);

            // Release memory
            vector<string>().swap(trecrank_paths);
            vector<double>().swap(search_times);
        }
    }

    // Reset rotate to normal mode
    rotate_eval = false;
}

void CheckGroundtruth(vector<result_object>& result, const int q_id)
{
    size_t result_size = result.size();

    /// OX 5k
    if (str_contains(run_param.dataset_prefix, "oxbuildings1m") ||
		str_contains(run_param.dataset_prefix, "oxbuildings105k") ||
        str_contains(run_param.dataset_prefix, "oxbuildings5k") ||
        str_contains(run_param.dataset_prefix, "paris6k"))
    {
        for (size_t result_idx = 0; result_idx < result_size; result_idx++)
        {
            if (groundtruth_checkup[QueryNameLists[q_id]][str_replace_last(ImgLists[result[result_idx].dataset_id], ".jpg", "")])
                result[result_idx].info += "eval:1;";
            else
                result[result_idx].info += "eval:0;";
        }
    }
    /// INS dataset
    else if (str_contains(run_param.dataset_prefix, "ins"))
    {
        for (size_t result_idx = 0; result_idx < result_size; result_idx++)
        {
            vector<string> shotname;
            // ins2011/img/9661
            StringExplode(ParentPaths[Pool2ParentsIdx[result[result_idx].dataset_id]], "/", shotname);
            if (groundtruth_checkup[QueryNameLists[q_id]][shotname[2]])
                result[result_idx].info += "eval:1;";
            else
                result[result_idx].info += "eval:0;";
        }
    }
}

float Compute_map(const string& query_topic)
{
    stringstream cmd;

    if (str_contains(run_param.dataset_prefix, "oxbuildings") ||
		str_contains(run_param.dataset_prefix, "paris") ||
		str_contains(run_param.dataset_prefix, "smalltest"))
    {
        string ap_binary = run_param.code_root_dir + "/ox_ap/bin/Release/ox_ap";

        cmd << ap_binary << " " << groundtruth_path.str() << " " << evalrank_path.str();
    }
    else if (str_contains(run_param.dataset_prefix, "ins201"))
    {
        string ap_binary = run_param.code_root_dir + "/trec_ap/trec_eval";

        // trec_eval -q -a -c groundtruth_file ranklist_file eval_top_n
        cmd << ap_binary << " -q -a -c " << groundtruth_path.str() << " " << trecrank_path.str() << " 1000 | grep -E 'infAP.*" << query_topic << "' | cut -f3";
    }
    cout << "map_cmd: " << cmd.str() << endl;

    return atof(exec(cmd.str()).c_str());
}

void ExportRawRank(const vector<result_object>& result, const string& query_path, float map)
{
    int rank_size = result.size();
    if (rank_size > top_web_export)
        rank_size = top_web_export;

    rawrank_path.str("");
    rawrank_path << query_path << "_rawrank.txt";

    ofstream rank_File (rawrank_path.str().c_str());
    if(rank_File.is_open())
    {
        rank_File << run_param.raw_param << endl;                                                   // Run name
        rank_File << run_param.detailed_param << endl;                                              // Detail
        string query_reldir = get_directory(str_replace_first(query_path, run_param.query_root_dir + "/", ""));
        rank_File << query_reldir << "|";                                                           // Query relative dir
        string queries_path = get_filename(query_path);                                             // Query filename (only one file)
        if (str_contains(queries_path, "fused"))
        {
            queries_path = "";
            for (size_t sub_query_idx = 0; sub_query_idx < QueryImgLists[curr_q_id].size(); sub_query_idx++)
            {
                if (sub_query_idx != 0)
                    queries_path += ",";    // separator
                queries_path += QueryImgLists[curr_q_id][sub_query_idx];                            // Multiple queries filename
            }
        }
        rank_File << queries_path << endl;                                                          // Query filename(s)

        if (map != 0.0f)
            rank_File << map << endl;
        else
            rank_File << "-" << endl;
        for(int index = 0; index < rank_size; index++)
        {
            if (run_param.pooling_enable) // Result of pool or result of image
            {
                //dataset_path << run_param.dataset_root_dir << "/" << ParentPaths[Pool2ParentsIdx[result[index].first]] << "/" << ImgLists[Pool2ImagesIdxRange[result[index].first].first];
                //rank_File << dataset_path.str() << "," << result[index].second << endl;
                size_t shot_frame_start = Pool2ImagesIdxRange[result[index].dataset_id].first;
                size_t shot_frame_end = Pool2ImagesIdxRange[result[index].dataset_id].second;
                size_t shot_size = 2;
                //size_t shot_size = shot_frame_end - shot_frame_start;
                rank_File << index << "|" <<																					// rank idx
                	result[index].score  << "|" <<																				// score
                	ParentPaths[Pool2ParentsIdx[result[index].dataset_id]] << "|" <<											// path to source dataset image
                	shot_size << "|";	                                                                                        // total frame of this shot
                	rank_File << ImgLists[shot_frame_start] << "," << ImgLists[shot_frame_end];
					/*for (size_t frame_idx = shot_frame_start; frame_idx < shot_frame_end; frame_idx++)
					{
					    if (frame_idx != shot_frame_start)
                            rank_File << ",";
						rank_File << ImgLists[frame_idx];                                                                       // image filename
					}*/
                if (result[index].info != "")
                    rank_File << "|" << result[index].info;                                                                            // extra info
                rank_File << endl;
            }
            else
            {
                // image_id to pool_id, then pool_id to parent_id // previous work ParentPaths[Img2ParentsIdx[result[index].first]]
                //dataset_path << run_param.dataset_root_dir << "/" << ParentPaths[Img2ParentsIdx[result[index].first]] << "/" << ImgLists[result[index].first];
                //rank_File << dataset_path.str() << "," << result[index].second << endl;
                size_t shot_frame_start = result[index].dataset_id;
                size_t shot_size = 1;
                //size_t shot_size = shot_frame_end - shot_frame_start;
                rank_File << index << "|" <<																					// rank idx
                	result[index].score  << "|" <<																				// score
                	ParentPaths[Img2ParentsIdx[result[index].dataset_id]] << "|" <<											    // path to source dataset image
                	shot_size << "|";	                                                                                        // total frame of this shot
                	rank_File << ImgLists[shot_frame_start];
					/*for (size_t frame_idx = shot_frame_start; frame_idx < shot_frame_end; frame_idx++)
					{
					    if (frame_idx != shot_frame_start)
                            rank_File << ",";
						rank_File << ImgLists[frame_idx];                                                                       // image filename
					}*/
                if (result[index].info != "")
                    rank_File << "|" << result[index].info;                                                                     // extra info
                rank_File << endl;
            }
        }
        rank_File.close();
    }
}

void ExportEvalRank(const vector<result_object>& result, const string& query_path)
{
    int rank_size = result.size();
    int export_limit = top_eval_export;
    if (export_limit > rank_size)
        export_limit = rank_size;

    evalrank_path.str("");
    evalrank_path << query_path << "_evalrank.txt";

    ofstream rank_File (evalrank_path.str().c_str());
    if (rank_File.is_open())
    {
        for(int index = 0; index < export_limit; index++)
        {
            string dataset_name = str_replace_first(ImgLists[result[index].dataset_id], ".jpg", "");
            rank_File << dataset_name << endl;
        }
        rank_File.close();
    }
}

void ExportRank_Trec(const vector<result_object>& result, const string& query_path, const string& query_topic)
{
    // SourceCode: http://trec.nist.gov/trec_eval/
    // Ref: http://goanna.cs.rmit.edu.au/~fscholer/trec_eval.php
    // Pattern: topic_number ignored_field document_id rank_position RSV run_label
    /*          topic_number    is query_topic_name
                ignored_field   is 0 (no ignore anything)
                document_id     is shot_name
                rank_position   is a position in rank list, which start from 1
                RSV is score    is similarity score
                run_label       is just a name
    */

    trecrank_path.str("");
    trecrank_path << query_path << "_trecrank.txt";

    int rank_size = result.size();
    int export_limit = top_eval_export;
    if (export_limit > rank_size)
        export_limit = rank_size;

    ofstream out_File (trecrank_path.str().c_str());
    if(out_File.is_open())
    {
        for(int rank_idx = 0; rank_idx < export_limit; rank_idx++)
        {
            vector<string> shotname;
            // ins2011/img/9661
            StringExplode(ParentPaths[Pool2ParentsIdx[result[rank_idx].dataset_id]], "/", shotname);

            out_File << query_topic << " 0 " << shotname[2] << " " << rank_idx + 1 << " " << result[rank_idx].score << " label" << endl;
        }
        out_File.close();
    }
}

void SubmitRank_Trec(const vector<string>& source_trec_paths, const vector<double>& timings)
{
    string submit_path = run_param.trecsubmit_root_dir + "/" + run_param.dataset_prefix;
    make_dir_available(submit_path);

    timespec submit_time = CurrentPreciseTime();
    size_t results_size = source_trec_paths.size();
    cout << "Submitting result..." << endl;
    cout << "to.. " << bluec << submit_path << endc << " "; cout.flush();
    for (size_t result_idx = 0; result_idx < results_size; result_idx++)
    {
        string submit_cmd = "cp " + source_trec_paths[result_idx] + " " + submit_path + "/" + get_filename(source_trec_paths[result_idx]) + COUT2NULL;
        exec(submit_cmd);
        percentout(result_idx, results_size, 1);
    }

    // Creating list file
    stringstream list_txt;
    for (size_t result_idx = 0; result_idx < results_size; result_idx++)
        list_txt << submit_path + "/" + get_filename(source_trec_paths[result_idx]) + " " + toString(timings[result_idx]) << endl;
    text_write(run_param.trecsubmit_root_dir + "/" + run_param.dataset_prefix + "_submit_list.txt", list_txt.str());

    cout << " done! (in " << setprecision(2) << fixed << TimeElapse(submit_time) << " s)" << endl;
}

// Misc
void oxMaskExport(const string& query_path, int q_id, const string& query_scaled_path)
{
    /*Point* verts = new Point[oxMaskLists.size()];
    for (size_t pt_id = 0; pt_id < oxMaskLists.size(); pt_id++)
    {
        verts[pt_id].x = (int)oxMaskLists[q_id][mask_id][pt_id].x;
        verts[pt_id].y = (int)oxMaskLists[q_id][mask_id][pt_id].y;
    }
    // Create Mask file
    draw_mask(query_path + ".mask.png", get_image_size(query_path), verts, oxMaskLists[q_id].size());

    delete[] verts;
    */

    // Scaling
    float qscale = 1.0f;
    if (query_scaled_path != "")    // Force resizing
    {
        if (run_param.query_scale_type == SCALE_ABS)    // Absolute maximum target size
        {
            Size query_original_size = get_image_size(query_path);
            Size query_scaled_size = get_image_size(query_scaled_path);
            qscale = float(query_scaled_size.width) / query_original_size.width;    // x and y scale is equal
        }
        else                                            // Ratio target size
            qscale = run_param.query_scale / 100.0f;
    }

    ofstream MaskFile;
    if (query_scaled_path == "")
        MaskFile.open((query_path + ".mask").c_str());
    else
        MaskFile.open((query_scaled_path + ".mask").c_str());
    if (MaskFile.is_open())
    {
        // Write total mask
        MaskFile << oxMaskLists[q_id].size() << endl;

        // Multiple mask
        for (size_t mask_id = 0; mask_id < oxMaskLists[q_id].size(); mask_id++)
        {
            // Write mask size
            MaskFile << oxMaskLists[q_id][mask_id].size() << endl;

            // Write points
            for (size_t point_id = 0; point_id < oxMaskLists[q_id][mask_id].size(); point_id++)
            {
                float ptx = oxMaskLists[q_id][mask_id][point_id].x * qscale;
                float pty = oxMaskLists[q_id][mask_id][point_id].y * qscale;
                // better not to scale (same as imtools.cpp overlay_mask)
                /*if (run_param.normpoint)
                {
                    Size img_size;
                    if (query_scaled_path == "")
                        img_size = get_image_size(query_path);
                    else
                        img_size = get_image_size(query_scaled_path);
                    ptx /= img_size.width;
                    pty /= img_size.height;
                }*/
                MaskFile << ptx << "," << pty << endl;
            }
        }

        // Close file
        MaskFile.close();
    }
}

// Memory management
void release_mem()
{
	inverted_hist.release_mem();
}
//;)
