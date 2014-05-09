/*
 * ins_online.h
 *
 *  Created on: October 7, 2013
 *      Author: Siriwat Kasamwattanarote
 */

#include "ins_online.h"

using namespace std;
using namespace tr1;
using namespace cv;
using namespace alphautils;
using namespace alphautils::hdf5io;
using namespace alphautils::imtools;
using namespace ins;

// ==== Main ====
int main(int argc,char *argv[])
{
    // Preparing tmpdir through /dev/shm
    make_dir_available("/dev/shm/query", "777");

	char menu;
	bool ivReady = false;

    // Export option initialization
	ExportOption exportOpts;
	exportOpts.max = 200;
	exportOpts.dev = false;
	exportOpts.ransac = false;
	exportOpts.showall = false;

	do
	{
	    cout << endl;
		cout << "======== Instant Search - Online (" << ins_online_AutoVersion::ins_online_FULLVERSION_STRING << ") ========" << endl;
        cout << "[l] Load preset dataset" << endl;
		cout << "[i] Load Inverted Histogram" << endl;
		if(ivReady)
		{
			cout << "[s] Search by query" << endl;
			cout << "[w] Search by web browser" << endl;
			cout << "[r] Search by random hist" << endl;
		}
		cout << "[t] Tools" << endl;
		cout << "[q] Quit" << endl;
		cout << "Enter menu:";cin >> menu;

		switch(menu)
		{
        case 'l':
            run_param.LoadPreset();

            LoadDataset(run_param.path_from_dataset);

            LoadTestQuery(run_param.path_from_dataset);

            if (is_commercial_film)
            {
                dataset_size = ds_info.dataset_cf_lookup_table_init(run_param.database_root_dir + "/" + run_param.dataset_header);//Init dataset_videos Lookup table
                ds_info.commercial_init();// Init and map commercial id
            }
            else
                dataset_size = ds_info.dataset_lookup_table_init(run_param.database_root_dir + "/" + run_param.dataset_header);

            LoadBowOffset();

            inverted_hist.init(run_param.CLUSTER_SIZE, run_param.database_root_dir + "/" + run_param.dataset_header + "/invdata_" + run_param.dataset_header);

            break;
		case 'i':
			{
				int TopLoad = 50;
				cout << "Top load = ";
				cin >> TopLoad;
				cout << "Load invert_index...";
				cout.flush();
				inverted_hist.load_invfile(TopLoad);
				cout << "OK!     " << endl;
				ivReady = true;
				break;
			}
		case 's':
			{
                int q_id;
				cout << "Enter query number :";
				cin >> q_id;
				result_id = search_by_id(q_id);
				cout << "Result (Q:V) : " << q_id << ":" << result_id << endl;

                // Groundtruth path
                groundtruth_path.str("");
                groundtruth_path << run_param.dataset_root_dir << "/" << run_param.path_from_dataset << "/groundtruth/" << QueryNameLists[q_id];

                // Computing map
                float curr_ap = Compute_map(groundtruth_path.str());

                cout << "Query \"" << QueryNameLists[q_id] << "\" [" << q_id + 1 << "/" << QueryNameLists.size() << "] got map = " << setprecision(4) << redc << curr_ap << endc << endl;

				DisplayRank();
				//ExportRankTrec(-1, -1);

				break;
			}
		case 't':
			{
				cout << "==== Tools ====" << endl;
				if(ivReady)
				{
					cout << "[s] Saving an Inverted Histogram" << endl;
					cout << "[r] Reseting Inverted memory and everything" << endl;
				}
				cout << "[1] Middleframe copier" << endl;
				cout << "[2] Render queue generator" << endl;
				cout << "[3] Compute map" << endl;
				cout << "Enter number:";cin >> menu;
				switch(menu)
				{
				case 's':
					{
						inverted_hist.save_invfile();
						cout << "Saved inverted histogram" << endl;
						break;
					}
				case 'r':
					{
						ResetMemory();
						ivReady = false;
						break;
					}
				case '1':
					{
						FrameCopy();
						break;
					}
				case '2':
					{
						size_t start, end;
						cout << "Total clips are " << dataset_size << endl;
						cout << "Please specified.." << endl;
						cout << "start index:";cin >> start;
						cout << "end index:";cin >> end;
						if(start < 0)
							start = 0;
						if (end > dataset_size)
							end = dataset_size - 1;
						RenderQueueGenerator(start, end);
						break;
					}
                case '3':
                    {

                        Evaluate();
                        break;
                    }
                case '4':
                    {
                        //FIX_RUN();
                        break;
                    }
				}
				break;
			}
		case 'w':
			{
				stringstream root_path;
				stringstream state_flag_path;
				stringstream query_stack_path;
				stringstream hist_request_path;
				stringstream status_path;
				stringstream status_text;

				root_path << run_param.query_root_dir << "/" << run_param.dataset_prefix;
				state_flag_path << root_path.str() << "/state_flag.txt";
				query_stack_path << root_path.str() << "/query_stack.txt";
				hist_request_path << root_path.str() << "/hist_request.txt";

				cout << "Opening service..." << endl;

				// Run HistExtractor
				// Run Encoder

				cout << "=============================================" << endl << "Listening to queries..." << endl;
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
										const char* delimsSpace = " ";// space

										vector<string> SubQuery;

										string_splitter(queryline, delimsSpace, SubQuery);

										if (SubQuery.size() > 1)
                                            exportOpts.dev = atoi(SubQuery[1].c_str());
										if (SubQuery.size() > 2)
                                            exportOpts.ransac = atoi(SubQuery[2].c_str());
										if (SubQuery.size() > 3)
                                            exportOpts.showall = atoi(SubQuery[3].c_str());

										// Update status: Uploading queries
										status_path.str("");
										status_path << root_path.str() << "/" << SubQuery[0] << "/" << "status.txt";
										status_text.str("");
										status_text << "Uploading queries...";
										UpdateStatus(status_path.str(), status_text.str());

										Queries.push_back(SubQuery[0]);
										cout << "SID : " << SubQuery[0] << endl;
									}
								}
								QueriesFile.close();

								// After read!, Send it to HistExtractor by just renaming!
								rename(query_stack_path.str().c_str(), hist_request_path.str().c_str());
							}

							// Extracting histogram and search
							vector<string>::iterator itQueries;
							for (itQueries = Queries.begin(); itQueries != Queries.end(); itQueries++)
							{
								stringstream session_path;
								stringstream query_list_path;
								stringstream hist_load_path;
								stringstream ready_signal_path;

								session_path << root_path.str() << "/" << *itQueries;
								query_list_path << session_path.str() << "/" << "list.txt";
								hist_load_path << session_path.str() << "/" << "bow_hist.xct";
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
								UpdateStatus(status_path.str(), status_text.str());

								// Waiting for HistXct result
								while(!is_path_exist(hist_load_path.str()))
                                {
                                    //usleep(250000);
                                    ls2null(hist_load_path.str());
                                }

								// Load BOW file
								vector<bow_bin_object> bow_hist;
								import_hist(hist_load_path.str(), bow_hist);

								extractTime = TimeElapse(histTime);
								cout << "completed!";
								cout << " (" <<  extractTime << " s)" << endl;
								cout << "Bin amount: " << bow_hist.size() << " bins" << endl;

								// Update status: Searching
								status_text.str("");
								status_text << "Searching on an inverted index database...";
								UpdateStatus(status_path.str(), status_text.str());

								// Search
								cout << "==== Search Module ====" << endl;
								startTime = CurrentPreciseTime();
								result_id = search_by_bow_sig(bow_hist);
								searchTime = TimeElapse(startTime);
								cout << "Match with: " << TotalMatch << " videos" << endl;
								cout << "Search time: " <<  searchTime << " s" << endl;
								cout << "Result (dataset_id) : " << result_id << endl;
								//DisplayRank();

								// Post Matching
								if (exportOpts.ransac)
								{
									cout << "==== Result Processor Module ====" << endl;
									// Read query location
									string QueryFilePath;
									string FrameRoot = run_param.database_root_dir + "/frames";
									ifstream QueriesListFile (query_list_path.str().c_str());
									if (QueriesListFile)
									{
										while (QueriesListFile.good())
										{
											string line;
											getline(QueriesListFile, line);
											if (line != "")
                                                QueryFilePath = line; //Matching with one query
										}
										QueriesListFile.close();
									}

									// Update status: Homography re-ranking
									status_text.str("");
									status_text << "LO-RANSAC based reranking...";
									UpdateStatus(status_path.str(), status_text.str());

									// Homo re-ranking
									cout << "-- Homography re-ranking" << endl;

									timespec HomoTimeStart = CurrentPreciseTime();
									homography Homo;
									Homo.Process(QueryFilePath, FrameRoot, ds_info.dataset_frames_dirname(), ds_info.dataset_frames_filename(), Result, exportOpts.max);
									// Get Result
									Homo.GetReRanked(ReRanked);
									homoTime = TimeElapse(HomoTimeStart);
									cout << "Homography time: " << homoTime  << " s" << endl;
								}
								else
								{
									size_t ResultSize = Result.size();
									ReRanked.clear();
									for (size_t index = 0; index < ResultSize; index++)
                                        ReRanked.push_back(pair<size_t, int>(index, (int)(Result[index].second * 100)));
								}

								// Update status: Exporting result
								status_text.str("");
								status_text << "Returning result...";
								UpdateStatus(status_path.str(), status_text.str());

								cout << "-- Export result" << endl;
								timespec ExportTime = CurrentPreciseTime();
								ExportRank(*itQueries, exportOpts);
								cout << "Export time: " <<  TimeElapse(ExportTime) << " s" << endl;
								// ExportRankTrec(-1, -1);

								// Release Memory
								bow_hist.clear();

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
								UpdateStatus(status_path.str(), status_text.str());
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
				cout << "Generate new hist?";
				cin >> gen;
				if (gen == 'y')
                    RandomHist();
				cout << "Generate completed!" << endl;
				startTime = CurrentPreciseTime();
				result_id = search_by_bow_sig(randhist);
				searchTime = TimeElapse(startTime);
				cout << "Total time: " <<  searchTime << " s" << endl;
				cout << "Match with: " << TotalMatch << " videos" << endl;
				cout << "Result (dataset_id) : " << result_id << endl;
				DisplayRank();

				// Post Matching
				cout << "==== Result Processor Module ====" << endl;
				// Read query location
				//stringstream QueryFilePath;
				//string FrameRoot = database_root_dir + "/frames";
				// Homo re-ranking
				//cout << "-- Homography re-ranking" << endl;
				//QueryFilePath << FrameRoot << "/" << ds_info.dataset_frames_dirname()[20] << "/" << ds_info.dataset_frames_filename()[20];
				//homography Homo;
				//cout << "Test query: " << QueryFilePath.str() << endl;
				//Homo.Process(QueryFilePath.str(), FrameRoot, ds_info.dataset_frames_dirname(), ds_info.dataset_frames_filename(), Result, exportOpts.max);
				// Get Result
				//Homo.GetReRanked(ReRanked);

				break;
			}
		}
	}
	while(menu != 'q');

	ds_info.dataset_lookup_table_destroy();//Destroy Lookup table

	return 0;
}

// ==== Function ====
void LoadDataset(const string& DatasetPath)
{
    stringstream dataset_saved_path;
    stringstream dataset_saved_list;
    dataset_saved_path << run_param.database_root_dir << "/" << run_param.dataset_header;
    dataset_saved_list << dataset_saved_path.str() << "/dataset";
    cout << "dataset_saved_list: " << dataset_saved_list.str() << endl;
    if (is_path_exist(dataset_saved_list.str() + "_basepath"))
    {
        cout << "Load dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadDatasetList(dataset_saved_list.str());
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
    {
        cout << "Dataset list not available!" << endl;
        cout << "path: " << dataset_saved_list.str() << endl;
        exit(1);
    }

    // Checking image avalibality
    if (ImgLists.size() > 0)
    {
        cout << "== Dataset information ==" << endl;
        cout << "Total directory: " << ImgParentPaths.size() << endl;
        cout << "Total image: " << ImgLists.size() << endl;
    }
    else
        cout << "No image available" << endl;
}

void LoadDatasetList(const string& in)
{
    // Read parent path (dataset based path)
    ifstream InParentFile ((in + "_basepath").c_str());
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
                const char* delimsColon = ":";

                string_splitter(read_line, delimsColon, split_line);

                ImgParentPaths.push_back(split_line[1]);
            }
        }

        // Close file
        InParentFile.close();
    }

    // Read image filename
    ifstream InImgFile ((in + "_filename").c_str());
    if (InImgFile)
    {
        string read_line;
        while (!InImgFile.eof())
        {
            getline(InImgFile, read_line);
            if (read_line != "")
            {
                vector<string> split_line;
                // parent_id:image_name
                const char* delimsColon = ":";

                string_splitter(read_line, delimsColon, split_line);

                ImgListsPoolIds.push_back(atoi(split_line[0].c_str()));
                ImgParentsIdx.push_back(atoi(split_line[1].c_str()));
                ImgLists.push_back(split_line[2]);
            }
        }

        // Close file
        InImgFile.close();
    }
}

void LoadTestQuery(const string& DatasetPath)
{
    if (str_contains(run_param.dataset_prefix, "oxbuildings") || str_contains(run_param.dataset_prefix, "paris") || str_contains(run_param.dataset_prefix, "smalltest"))
    {
        cout << "Loading query for " << run_param.dataset_prefix << "...";
        cout.flush();
        startTime = CurrentPreciseTime();

        stringstream dataset_path;
        stringstream groundtruth_path;
        dataset_path << run_param.dataset_root_dir << "/" << DatasetPath;
        groundtruth_path << dataset_path.str() << "/groundtruth";
        //cout << "groundtruth_path: " << groundtruth_path.str() << endl;
        // Directory traverse
        DIR* dirp = opendir(groundtruth_path.str().c_str());
        dirent* dp;
        while ((dp = readdir(dirp)) != NULL)
        {
            if (str_contains(string(dp->d_name), "query"))
            {
                //cout << "dp->d_name: " << string(dp->d_name) << endl;
                stringstream query_test_path;
                query_test_path << groundtruth_path.str() << "/" << string(dp->d_name);

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
                            const char* delims = " ";// space

                            vector<string> query_split;

                            string_splitter(queryline, delims, query_split);

                            // Query name
                            QueryNameLists.push_back(str_replace_first(string(dp->d_name), "_query.txt", ""));

                            // Query image
                            if (str_contains(run_param.dataset_prefix, "oxbuildings"))                              // oxfordbuildings
                                QueryImgLists.push_back(str_replace_first(query_split[0], "oxc1_", "") + ".jpg");
                            else                                                                                    // paris
                                QueryImgLists.push_back(query_split[0] + ".jpg");

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
                                MaskLists.push_back(mask);
                            }
                        }
                    }
                    InFile.close();
                }
            }
        }
        closedir(dirp);

        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        cout << "Total " << QueryImgLists.size() << " query topic(s)" << endl;
    }
}

void LoadBowOffset()
{
    string bow_offset_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/bow_offset";

    cout << "Loading BOW offset..."; cout.flush();
    if (!bin_read_vector_SIZET(bow_offset_path, bow_offset))
    {
        cout << "BOW Offset file does not exits, (" << bow_offset_path << ")" << endl;
        exit(-1);
    }
    cout << "done!" << endl;
}

void Resetinverted_hist()
{
	inverted_hist.reset();
}

void ResetMemory()
{
	Resetinverted_hist();
}

void extract_hist(const string& session_name, const vector<string>& queries, vector<bow_bin_object>& bow_hist)
{
    stringstream query_search_path;
    query_search_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << simulated_session;

    stringstream list_path;
    list_path << query_search_path.str() << "/" << "list.txt";

    // Create query list file
    ofstream ListFile (list_path.str().c_str());
    if (ListFile.is_open())
    {
        for (size_t query_idx = 0; query_idx < queries.size(); query_idx++)
            ListFile << queries[query_idx] << endl;
        ListFile.close();
    }

    // Write path to hist request
    stringstream hist_request_path;
	hist_request_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << "hist_request.txt";
    ofstream histrequest_File (hist_request_path.str().c_str());
    if (histrequest_File.is_open())
    {
        // See ref send_query
        // Flag mask enable option
        // Dev:0|1 Ransac:0|1 Showall:0|1
        histrequest_File << session_name << " 0 0 0";
        histrequest_File.close();
    }

    // Waiting for HistXct result
    stringstream hist_load_path;
    hist_load_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << session_name << "/" << "bow_hist.xct";
    while(!is_path_exist(hist_load_path.str()) || islock(hist_load_path.str()))
    {
        //usleep(10000);
        ls2null(hist_load_path.str());
    }

    // Read bow hist
    import_hist(hist_load_path.str(), bow_hist);

    // After read, change file name
    //cout << "From: " << hist_load_path.str() << endl;
    //cout << "To: " << get_directory(hist_load_path.str()) + "/" + queries[0] + "_bow_hist.xct" << endl;
    rename(hist_load_path.str().c_str(), (queries[0] + "_bow_hist.xct").c_str());
}

void import_hist(const string& in, vector<bow_bin_object>& bow_hist)
{
    // Load HistXct file
    ifstream HistXctFile (in.c_str(), ios::binary);
    if (HistXctFile)
    {
        // Read bin count
        size_t bin_count;
        HistXctFile.read((char*)(&bin_count), sizeof(bin_count));

        for(size_t bin_id = 0; bin_id < bin_count; bin_id++)
        {
            bow_bin_object read_bin;

            // Read cluster_id
            HistXctFile.read((char*)(&read_bin.cluster_id), sizeof(read_bin.cluster_id));

            // Read frequency
            HistXctFile.read((char*)(&read_bin.freq), sizeof(read_bin.freq));

            // Read frequency
            size_t feature_count;
            HistXctFile.read((char*)(&feature_count), sizeof(feature_count));

            // Read feature
            for (size_t feature_id = 0; feature_id < feature_count; feature_id++)
            {
                feature_object read_feature;
                HistXctFile.read((char*)(&(read_feature.x)), sizeof(read_feature.x));
                HistXctFile.read((char*)(&(read_feature.y)), sizeof(read_feature.y));
                HistXctFile.read((char*)(&(read_feature.a)), sizeof(read_feature.a));
                HistXctFile.read((char*)(&(read_feature.b)), sizeof(read_feature.b));
                HistXctFile.read((char*)(&(read_feature.c)), sizeof(read_feature.c));
                read_bin.features.push_back(read_feature);
            }
            bow_hist.push_back(read_bin);
        }

        // Read num_kp
        HistXctFile.read((char*)(&num_kp), sizeof(num_kp));

        // Read mask_pass
        HistXctFile.read((char*)(&mask_pass), sizeof(mask_pass));

        // Close file
        HistXctFile.close();
    }
}

string ResizeQuery(const string& query_path)
{
    // Resize query
    string query_scaled_path = query_path + "_scaled_" + toString(run_param.query_scale_type) + "_" + toString(run_param.query_scale) + ".jpg";
    string cmd;
    if (run_param.query_scale_type == SCALE_ABS)
    {
        cout << "Resize query to " << run_param.query_scale << "x" << run_param.query_scale << endl;
        cmd = "convert " + query_path + " -resize " + toString(run_param.query_scale) + "x" + toString(run_param.query_scale) + "\\> " + query_scaled_path;
    }
    else
    {
        cout << "Resize query to " << run_param.query_scale << "%" << endl;
        cmd = "convert " + query_path + " -resize " + toString(run_param.query_scale) + "% " + query_scaled_path;
    }
    exec(cmd);

    // Resize mask
    size_t mask_count = 0;
    size_t mask_vertex_count = 0;
    vector<float*> mask_vertex_x;
    vector<float*> mask_vertex_y;
    ifstream in_mask_File ((query_path + ".mask").c_str());
    if (in_mask_File)  // Check mask exist
    {
        string line;
        getline(in_mask_File, line);

        /// Read original mask
        // Read mask count
        mask_count = atoi(line.c_str());

        for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
        {
            getline(in_mask_File, line);

            // Read vertex count
            mask_vertex_count = atoi(line.c_str());

            // Preparing mask array
            float* curr_mask_vertex_x = new float[mask_vertex_count];
            float* curr_mask_vertex_y = new float[mask_vertex_count];

            // Read mask verticies
            for (size_t mask_idx = 0; mask_idx < mask_vertex_count; mask_idx++)
            {
                getline(in_mask_File, line);

                char const* delims = ",";
                vector<string> vertex;
                string_splitter(line, delims, vertex);

                // Scaling
                float qscale;
                if (run_param.query_scale_type == SCALE_ABS)    // Absolute maximum target size
                {
                    Size query_original_size = get_image_size(query_path);
                    Size query_scaled_size = get_image_size(query_scaled_path);
                    qscale = float(query_scaled_size.width) / query_original_size.width;    // x and y scale is equal
                    //qscale = float(query_scaled_size.height) / query_original_size.height;
                }
                else                                            // Ratio target size
                    qscale = run_param.query_scale / 100.0f;

                curr_mask_vertex_x[mask_idx] = atof(vertex[0].c_str()) * qscale;
                curr_mask_vertex_y[mask_idx] = atof(vertex[1].c_str()) * qscale;
            }

            // Keep multiple masks
            mask_vertex_x.push_back(curr_mask_vertex_x);
            mask_vertex_y.push_back(curr_mask_vertex_y);
        }

        in_mask_File.close();

        /// Write new scaled mask
        ofstream out_mask_File ((query_scaled_path + ".mask").c_str());
        if (out_mask_File.is_open())
        {
            // Write mask_count
            out_mask_File << mask_count << endl;

            // Multiple mask
            for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
            {
                // Write mask size
                out_mask_File << mask_vertex_count << endl;

                // Write points
                for (size_t mask_idx = 0; mask_idx < mask_vertex_count; mask_idx++)
                    out_mask_File << mask_vertex_x[mask_id][mask_idx] << "," << mask_vertex_y[mask_id][mask_idx] << endl;
            }

            // Close file
            out_mask_File.close();
        }
    }

    // Release mem
    for (size_t mask_id = 0; mask_id < mask_vertex_x.size(); mask_id++)
    {
        delete[] mask_vertex_x[mask_id];
        delete[] mask_vertex_y[mask_id];
    }
    mask_vertex_x.clear();
    mask_vertex_y.clear();

    return query_scaled_path;
}

template<typename T, typename U> void calculate_symmat_distance(size_t nvert, const T vertx[], const T verty[], U symmat_dist[])
{
    // Calculate only half matrix
    for (size_t row_id = 1; row_id < nvert; row_id++)
    {
        for (size_t col_id = 0; col_id < row_id; col_id++)
        {
            // Euclidean distance
            symmat_dist[rc2half_idx(row_id, col_id)] = U(sqrt( (vertx[row_id] - vertx[col_id]) * (vertx[row_id] - vertx[col_id]) +
                                                        (verty[row_id] - verty[col_id]) * (verty[row_id] - verty[col_id])));
        }
    }
}


void QB1_Bow(vector<bow_bin_object>& bow)
{
	cout << "==== QB1 Bow ====" << endl;
    // Step 2, Search BOW pass 1
    // Search
    cout << "Searching.."; cout.flush();
    startTime = CurrentPreciseTime();
    search_by_bow_sig(bow);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Step 3, Matching bow with top rank, Filtering the same cluster_id between query and top results
    cout << "Filtering.."; cout.flush();
    int* active_bow = new int[run_param.CLUSTER_SIZE];
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)      // Empty active_bow
        active_bow[cluster_id] = 0;

    // Matching cluster_id
    size_t max_rank_check = run_param.query_bootstrap_rankcheck;
    if (max_rank_check > Result.size())
        max_rank_check = Result.size();
    for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
    {
        vector<bow_bin_object> bow_result;
        size_t dataset_id = Result[rank_id].first;
        LoadSpecificBow(dataset_id, bow_result);

        for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
        {
            size_t cluster_id = bow_result[bin_id].cluster_id;
            if (run_param.query_bootstrap_minbow_type == MIN_BIN)
                active_bow[cluster_id]++;                                       // accumulate by bin
            else
                active_bow[cluster_id] += bow_result[bin_id].features.size();   // accumulate by feature size
        }
    }

    // Filtering by min_bow_thre
    vector<bow_bin_object> ret_bow;
    int min_bow_thre = run_param.query_bootstrap_minbow;
    for (size_t bin_id = 0; bin_id < bow.size(); bin_id++)
    {
        size_t cluster_id = bow[bin_id].cluster_id;
        if (active_bow[cluster_id] >= min_bow_thre)
            ret_bow.push_back(bow[bin_id]);
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    cout << "Bow in " << bow.size() << " bin(s)." << endl;
    cout << "Bow out " << ret_bow.size() << " bin(s)." << endl;

    // Swap
    ret_bow.swap(bow);

    delete[] active_bow;
}

void QueryBootstrapping_v1(const string& query_path)
{
    int totalBin = 0;
    int totalFeature = 0;
    int totalPass = 0;
    int maxMatch = 0;
    double extractTime;
    double totalExtractTime = 0;
    double totalSearchTime = 0;

	// Resize image
	string query_scaled_path = query_path;
    if (run_param.query_scale_enable)
        query_scaled_path = ResizeQuery(query_path);

    cout << "==== Query bootstrapping v1 ====" << endl;
    /// Pass 1
    cout << "## Pass 1 ##" << endl;
    // Step 1, Extract BOW
    // Pack query list (with bootstrap)
    vector<string> queries;
    queries.push_back(query_scaled_path);

    // Request extract hist
    vector<bow_bin_object> bootstrap_bow_hist;
    cout << "Extracting BOW histogram.."; cout.flush();
    startTime = CurrentPreciseTime();
    extract_hist(simulated_session, queries, bootstrap_bow_hist);
    extractTime = TimeElapse(startTime);
    totalExtractTime = extractTime;
    cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

    if (int(bootstrap_bow_hist.size()) < run_param.query_bootstrap_rankcheck)
    {
        cout << "Too small query or no feature can be extracted!" << endl;
        cout << "Switched to original query..." << endl;

        // Pack query list (with bootstrap)
        queries.clear();
        queries.push_back(query_path);

        // Request extract hist
        cout << "Extracting BOW histogram.."; cout.flush();
        startTime = CurrentPreciseTime();
        extract_hist(simulated_session, queries, bootstrap_bow_hist);
        extractTime = TimeElapse(startTime);
        totalExtractTime = extractTime;
        cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;
    }

    // MAP report bin size
    //map_push_report(toString(bootstrap_bow_hist.size()) + ",");
    cout << "Bin amount: " << bootstrap_bow_hist.size() << " bins" << endl;
    totalBin += bootstrap_bow_hist.size();

    // MAP report num_kp
    //map_push_report(toString(num_kp) + ",");
    cout << "Total feature(s): " << num_kp << " point(s)" << endl;
    totalFeature += num_kp;

    // MAP report mask_pass
    //map_push_report(toString(mask_pass) + ",");
    cout << "Mask passed: " << mask_pass << " point(s)" << endl;
    totalPass += mask_pass;

    // Step 2, Search BOW pass 1
    // Search
    cout << "==== Search Pass 1 Timing Info ====" << endl;
    cout << "Searching.."; cout.flush();
    startTime = CurrentPreciseTime();
    result_id = search_by_bow_sig(bootstrap_bow_hist);    // Full ranking was in Result< pair<size_t dataset_id, float score> >
    searchTime = TimeElapse(startTime);
    cout << "Match with: " << TotalMatch << " videos" << endl;
    cout << "Search time: " <<  searchTime << " s" << endl;
    cout << "Result (dataset_id) : " << result_id << endl;

    // MAP report number of matched dataset
    //map_push_report(toString(TotalMatch) + ",");
    if (maxMatch < TotalMatch)
        maxMatch = TotalMatch;

    // MAP report search time usage
    //map_push_report(toString(searchTime) + ",");
    totalSearchTime += searchTime;

    ExportEvalRank(query_scaled_path);

    // Computing map
    float curr_ap = Compute_map(groundtruth_path.str());
    // MAP report map value, and close report for topic
    //map_push_report(toString(curr_ap) + "\n");

    ExportRawRank(query_scaled_path, curr_ap);

    // Step 3, Matching bow with top rank, Filtering the same cluster_id between query and top results
    bool* bootstrap_bow_mask = new bool[run_param.CLUSTER_SIZE];
    int* active_bow = new int[run_param.CLUSTER_SIZE];
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)      // Empty bootstrap_mask and active_bow
    {
        bootstrap_bow_mask[cluster_id] = false;
        active_bow[cluster_id] = 0;
    }
    for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)                // Initialize bootstrap_mask from bootstrap_bow_hist
        bootstrap_bow_mask[bootstrap_bow_hist[bin_id].cluster_id] = true;

    // Matching cluster_id
    size_t max_rank_check = run_param.query_bootstrap_rankcheck;
    if (max_rank_check > Result.size())
        max_rank_check = Result.size();
    // Matching dump (for dumping bootstrap)
    bool is_dump = false;
    unordered_map<size_t, size_t> bootstrap_bow_lut; // cluster_id -> vector_idx
    if (is_dump)
    {
        // Mapping bootstrap_bow_hist lut
        for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)
            bootstrap_bow_lut[bootstrap_bow_hist[bin_id].cluster_id] = bin_id;
        // Making dump list
        vector<size_t> dump_list;
        for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
            dump_list.push_back(Result[rank_id].first);
        // Start dump
        inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, dump_list, query_scaled_path);
        // Release mem
        dump_list.clear();
    }
    for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
    {
        vector<bow_bin_object> bow_result;
        size_t dataset_id = Result[rank_id].first;
        LoadSpecificBow(dataset_id, bow_result);

        for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
        {
            size_t cluster_id = bow_result[bin_id].cluster_id;
            if (run_param.query_bootstrap_minbow_type == MIN_BIN)
                active_bow[cluster_id]++;                                       // accumulate by bin
            else
                active_bow[cluster_id] += bow_result[bin_id].features.size();   // accumulate by feature size
            // For dumping
            if (is_dump && bootstrap_bow_mask[cluster_id])
            {
                vector<feature_object>::const_iterator query_feature_it;
                for (query_feature_it = bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.begin(); query_feature_it != bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.end(); query_feature_it++)
                {
                    vector<feature_object>::iterator dataset_feature_it; // feature from inverted_index
                    for (dataset_feature_it = bow_result[bin_id].features.begin(); dataset_feature_it != bow_result[bin_id].features.end(); dataset_feature_it++)
                        inverted_hist.feature_matching_dump(dataset_id, cluster_id, bow_result[bin_id].freq, dataset_feature_it->x, dataset_feature_it->y, dataset_feature_it->a, dataset_feature_it->b, dataset_feature_it->c, query_feature_it->x, query_feature_it->y, query_feature_it->a, query_feature_it->b, query_feature_it->c);
                }
            }
        }
    }
    if (is_dump)
        inverted_hist.stop_matching_dump();
    // Reset bootstrap_bow_mask for union only result
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        bootstrap_bow_mask[cluster_id] = false;
    // Filtering mask by min_bow_thre
    int feature_pass = 0;
    int min_bow_thre = run_param.query_bootstrap_minbow;
    for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)
    {
        size_t cluster_id = bootstrap_bow_hist[bin_id].cluster_id;
        if (active_bow[cluster_id] >= min_bow_thre)
        {
            bootstrap_bow_mask[cluster_id] = true;
            feature_pass += bootstrap_bow_hist[bin_id].features.size();  // Accumulate feature pass
        }
    }


    // Step 4, Detect approx object amount in bootstrap size
    Size bootstrap_img_size = get_image_size(query_scaled_path);
    int approx_object = 0;
    unordered_map<int, vector<Point> > active_object;
    int patch_size = run_param.query_bootstrap_patch;
    if (patch_size > 0) // patch normal operation
    {
        int total_patch = patch_size * patch_size;
        int* active_patch = new int[total_patch];
        bool* active_patch_binary = new bool[total_patch];
        bool* active_patch_check = new bool[total_patch];
        // Initial active patch to zero
        for (int patch_id = 0; patch_id < total_patch; patch_id++)
        {
            active_patch[patch_id] = 0;
            active_patch_binary[patch_id] = false;
            active_patch_check[patch_id] = false;
        }

        // Accumulating active patch
        for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)
        {
            if (bootstrap_bow_mask[bootstrap_bow_hist[bin_id].cluster_id])
            {
                for (size_t feature_id = 0; feature_id < bootstrap_bow_hist[bin_id].features.size(); feature_id++)
                {
                    float vertx = bootstrap_bow_hist[bin_id].features[feature_id].x;
                    float verty = bootstrap_bow_hist[bin_id].features[feature_id].y;
                    if (run_param.normpoint) // scale back to original
                    {
                        vertx *= bootstrap_img_size.width;
                        verty *= bootstrap_img_size.height;
                    }

                    // Calculate corresponding patch location
                    int patch_col = ceil((float)patch_size * vertx / bootstrap_img_size.width) - 1;
                    int patch_row = ceil((float)patch_size * verty / bootstrap_img_size.height) - 1;
                    active_patch[patch_row * patch_size + patch_col]++;
                }
            }
        }

        // Thresholding
        // Find max
        int active_max = 0;
        for (int patch_id = 0; patch_id < total_patch; patch_id++)
        {
            if (active_max < active_patch[patch_id])
                active_max = active_patch[patch_id];
        }
        // Cut > 50%
        for (int patch_id = 0; patch_id < total_patch; patch_id++)
        {
            if (active_patch[patch_id] / (float)active_max >= 0.5f)
                active_patch_binary[patch_id] = true;
            else
                active_patch_binary[patch_id] = false;
        }

        // Display active_patch_binary
        for (int row_id = 0; row_id < patch_size; row_id++)
        {
            for (int col_id = 0; col_id < patch_size; col_id++)
                cout << boolalpha << active_patch_binary[row_id * patch_size + col_id] << " ";
            cout << endl;
        }

        // Counting possible object from continueous patch
        for (int row_id = 0; row_id < patch_size; row_id++)
        {
            for (int col_id = 0; col_id < patch_size; col_id++)
            {
                int patch_id;
                vector<Point> curr_neighbor_list;
                // Current
                patch_id = row_id * patch_size + col_id;
                if (active_patch_binary[patch_id] && !active_patch_check[patch_id])
                {
                    active_patch_check[patch_id] = true;
                    curr_neighbor_list.push_back(Point(col_id, row_id));
                }

                for (size_t neighbor_list_id = 0; neighbor_list_id < curr_neighbor_list.size(); neighbor_list_id++)
                {
                    int curr_row_id = curr_neighbor_list[neighbor_list_id].y;
                    int curr_col_id = curr_neighbor_list[neighbor_list_id].x;
                    // Top
                    if (curr_row_id > 0)
                    {
                        patch_id = (curr_row_id - 1) * patch_size + curr_col_id;
                        if (active_patch_binary[patch_id] && !active_patch_check[patch_id])
                        {
                            active_patch_check[patch_id] = true;
                            curr_neighbor_list.push_back(Point(curr_col_id, curr_row_id - 1));
                        }
                    }
                    // Down
                    if (curr_row_id < patch_size - 1)
                    {
                        patch_id = (curr_row_id + 1) * patch_size + curr_col_id;
                        if (active_patch_binary[patch_id] && !active_patch_check[patch_id])
                        {
                            active_patch_check[patch_id] = true;
                            curr_neighbor_list.push_back(Point(curr_col_id, curr_row_id + 1));
                        }
                    }
                    // Left
                    if (curr_col_id > 0)
                    {
                        patch_id = curr_row_id * patch_size + (curr_col_id - 1);
                        if (active_patch_binary[patch_id] && !active_patch_check[patch_id])
                        {
                            active_patch_check[patch_id] = true;
                            curr_neighbor_list.push_back(Point(curr_col_id - 1, curr_row_id));
                        }
                    }
                    // Right
                    if (curr_col_id < patch_size - 1)
                    {
                        patch_id = curr_row_id * patch_size + (curr_col_id + 1);
                        if (active_patch_binary[patch_id] && !active_patch_check[patch_id])
                        {
                            active_patch_check[patch_id] = true;
                            curr_neighbor_list.push_back(Point(curr_col_id + 1, curr_row_id));
                        }
                    }
                }

                if (curr_neighbor_list.size() > 0)
                    active_object[approx_object++].swap(curr_neighbor_list);
            }
        }
        cout << "Total object: " << approx_object << endl;

        // Release mem
        delete[] active_patch;
        delete[] active_patch_binary;
        delete[] active_patch_check;
    }
    else // manually assign number of approx. object
        approx_object = abs(patch_size); // explicitly use negative number from patch_size to be number of cluster


    // Step 5, Find nearest neighbor to active patch center
    Index< ::flann::L2<float> > search_index(KDTreeIndexParams((int)run_param.KDTREE));

    // Make cluster center
    int xy_dimension = 2;
    int curr_centroid_idx = 0;
    float* centroid_data = new float[approx_object * xy_dimension];
    unordered_map<int, vector<Point> >::iterator active_object_it;
    for (active_object_it = active_object.begin(); active_object_it != active_object.end(); active_object_it++)
    {
        size_t curr_feature_size = active_object_it->second.size();
        float curr_centroid_x = 0;
        float curr_centroid_y = 0;
        // Calculate current centroid for patch
        for(size_t feature_id = 0; feature_id < curr_feature_size; feature_id++)
        {
            curr_centroid_x += active_object_it->second[feature_id].x;
            curr_centroid_y += active_object_it->second[feature_id].y;
        }
        // Ref http://www.quickmath.com/webMathematica3/quickmath/algebra/simplify/basic.jsp#c=simplify_stepssimplify&v1=r+*+w+%2F+s+-+(w+%2F+s)+%2F+2
        if (run_param.normpoint)
        {
            centroid_data[curr_centroid_idx * xy_dimension] = ((curr_centroid_x / curr_feature_size) + 1) / patch_size - (1.0f / patch_size / 2);
            centroid_data[curr_centroid_idx * xy_dimension + 1] = ((curr_centroid_y / curr_feature_size) + 1) / patch_size - (1.0f / patch_size / 2);
        }
        else
        {
            centroid_data[curr_centroid_idx * xy_dimension] = ((curr_centroid_x / curr_feature_size) + 1) * bootstrap_img_size.width / patch_size - (float(bootstrap_img_size.width / patch_size) / 2);
            centroid_data[curr_centroid_idx * xy_dimension + 1] = ((curr_centroid_y / curr_feature_size) + 1) * bootstrap_img_size.height / patch_size - (float(bootstrap_img_size.height / patch_size) / 2);
        }
        cout << curr_centroid_idx << " " << centroid_data[curr_centroid_idx * xy_dimension] << ", " << centroid_data[curr_centroid_idx * xy_dimension + 1] << endl;
        curr_centroid_idx++;
    }
    Matrix<float> centroids(centroid_data, approx_object, xy_dimension);

    // Building search index
    cout << "Build FLANN search index..";
    cout.flush();
    startTime = CurrentPreciseTime();
    search_index.buildIndex(centroids);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // KNN search
    SearchParams sparams = SearchParams();
    sparams.checks = 512;
    //sparams.cores = run_param.MAXCPU;
    sparams.cores = 4;
    size_t knn = 1;

    cout << "KNN searching..";
    cout.flush();
    startTime = CurrentPreciseTime();

    // Feature preparation
    float* matched_point_data = new float[feature_pass * xy_dimension];
    int acc_point_count = 0;
    for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)
    {
        if (bootstrap_bow_mask[bootstrap_bow_hist[bin_id].cluster_id])
        {
            for (size_t feature_id = 0; feature_id < bootstrap_bow_hist[bin_id].features.size(); feature_id++)
            {
                matched_point_data[acc_point_count * xy_dimension] = bootstrap_bow_hist[bin_id].features[feature_id].x;
                matched_point_data[acc_point_count * xy_dimension + 1] = bootstrap_bow_hist[bin_id].features[feature_id].y;
                acc_point_count++;
            }
        }
    }
    Matrix<float> matched_point(matched_point_data, feature_pass, xy_dimension);
    Matrix<int> result_index(new int[feature_pass * knn], feature_pass, knn); // size = feature_amount x knn
    Matrix<float> result_dist(new float[feature_pass * knn], feature_pass, knn);
    search_index.knnSearch(matched_point, result_index, result_dist, knn, sparams);

    /*
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 2; col++)
            cout << matched_point_data[row * 2 + col] << ", ";
        cout << endl;
    }*/

    // Keep result
    const int* result_index_idx = result_index.ptr();
    //const float* result_dist_idx = result_dist.ptr();

    // Step 6, Crop image
    Mat original_img = imread(query_path);
    vector<Mat> cropped_obj;
    vector<string> cropped_obj_path;
    Size query_img_size = get_image_size(query_path);
    for (int centroid_id = 0; centroid_id < approx_object; centroid_id++)
    {
        // Build current centroid
        Point2f centroid(centroid_data[centroid_id * xy_dimension], centroid_data[centroid_id * xy_dimension + 1]);

        // Packing point group
        vector<Point2f> obj_points;
        //for(int point_id = 0; point_id < mask_pass; point_id++) // pass from mask
        for (int point_id = 0; point_id < feature_pass; point_id++) // pass from bow_bin_threshold
        {
            int label = result_index_idx[point_id];
            if (label == centroid_id)
            {
                Point2f pnt(matched_point_data[point_id * xy_dimension], matched_point_data[point_id * xy_dimension + 1]);
                obj_points.push_back(pnt);
            }
        }

        // Visualize Polygon
        bool visualize_detected_obj = true;
        if (visualize_detected_obj)
        {
            // Visualize polygon with centroid
            overlay_point2centroid(query_path, query_path + "_ptcls_" + ZeroPadNumber(approx_object, 2) + ".png", centroid, obj_points);
        }

        // Normalize if normed point
        if (run_param.normpoint)
        {
            for (size_t point_id = 0; point_id < obj_points.size(); point_id++)
            {
                obj_points[point_id].x *= query_img_size.width;   // Re-scale obj_points
                obj_points[point_id].y *= query_img_size.height;
            }
        }

        // Detect Bounding rectangle
        Rect obj_box = boundingRect(obj_points);

        // Extend box
        int margin = 10;
        if (obj_box.width < 64)
            margin = (64 - obj_box.width) >> 1;
        if (obj_box.height < 64)
            margin = (64 - obj_box.height) >> 1;
        obj_box.x -= margin;
        if (obj_box.x < 0)
            obj_box.x = 0;
        obj_box.y -= margin;
        if (obj_box.y < 0)
            obj_box.y = 0;
        obj_box.width += (margin << 1);
        if (obj_box.x + obj_box.width >= original_img.cols)
            obj_box.width = original_img.cols - obj_box.x - 1;
        obj_box.height += (margin << 1);
        if (obj_box.y + obj_box.height >= original_img.rows)
            obj_box.height = original_img.rows - obj_box.y - 1;

        // Visualize object bounding box
        if (visualize_detected_obj)
            overlay_rect(query_path, query_path + "_ptcls_" + ZeroPadNumber(approx_object, 2) + ".png", obj_box);

        // Do Crop
        cropped_obj.push_back(original_img(obj_box));
        string curr_cropped_obj_path_str = query_path + "_" + toString(obj_box.x) + "_" + toString(obj_box.y) + "_" + toString(obj_box.width) + "_" + toString(obj_box.height) + ".jpg";
        cropped_obj_path.push_back(curr_cropped_obj_path_str);

        // Write image object
        vector<int> imwrite_param;
        imwrite_param.push_back(CV_IMWRITE_JPEG_QUALITY);
        imwrite_param.push_back(85);
        imwrite(cropped_obj_path[cropped_obj.size() - 1], cropped_obj[cropped_obj.size() - 1], imwrite_param);

        // Write focused mask
        if (run_param.query_bootstrap_mask_enable && obj_points.size() > 2) // polygon combined with minimum 3 vertices
        {
            vector<Point2f> mask_point;
            // Find convex hull
            convexHull(obj_points, mask_point);
            ofstream out_mask_File ((curr_cropped_obj_path_str + ".mask").c_str());
            if (out_mask_File.is_open())
            {
                // Write mask_count (it always be only 1 mask for each object)
                out_mask_File << 1 << endl;

                // Multiple mask
                //for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
                //{
                    // Write mask size
                    out_mask_File << mask_point.size() << endl;

                    // Write points
                    for (size_t mask_idx = 0; mask_idx < mask_point.size(); mask_idx++)
                    {
                        // Extend box mask
                        // Ref http://stackoverflow.com/questions/7740507/extend-a-line-segment-a-specific-distance
                        float curr_x = mask_point[mask_idx].x;
                        float curr_y = mask_point[mask_idx].y;
                        float centroid_diff_x = curr_x - centroid.x;
                        float centroid_diff_y = curr_y - centroid.y;
                        float unit_length = sqrt(centroid_diff_x * centroid_diff_x + centroid_diff_y * centroid_diff_y);
                        float extended_x = curr_x + centroid_diff_x / unit_length * 2;
                        float extended_y = curr_y + centroid_diff_y / unit_length * 2;

                        out_mask_File << extended_x - obj_box.x << "," << extended_y - obj_box.y << endl;
                    }
                //}

                // Close file
                out_mask_File.close();
            }
        }

        // Release mem
        obj_points.clear();
    }

    /// Pass 2
    cout << "## Pass 2 ##" << endl;
    // Step 7, Search pass 2, by sub obj
    for (size_t obj_id = 0; obj_id < cropped_obj.size(); obj_id++)
    {
        cout << "## Object no." << obj_id << " ##" << endl;

        // MAP report query id
        //map_push_report("n/a,");

        // MAP report query name
        //map_push_report("n/a,");

        // MAP report query source image
        //map_push_report(get_filename(cropped_obj_path[obj_id]) + ",");

        // MAP report mask
        //map_push_report("bootstrap-cropped,");

        // Pack query list (with bootstrap)
        vector<string> cropped_obj_query;
        cropped_obj_query.push_back(cropped_obj_path[obj_id]);

        // Request extract hist
        vector<bow_bin_object> cropped_obj_bow_hist;
        cout << "Extracting BOW histogram.."; cout.flush();
        startTime = CurrentPreciseTime();
        extract_hist(simulated_session, cropped_obj_query, cropped_obj_bow_hist);
        extractTime = TimeElapse(startTime);
        totalExtractTime += extractTime;
        cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

        if (cropped_obj_bow_hist.empty())
        {
            cout << "Skipped query: " << cropped_obj_path[obj_id] << " - no keypoint detected" << endl;
            continue;
        }

        // MAP report bin size
        //map_push_report(toString(cropped_obj_bow_hist.size()) + ",");
        cout << "Bin amount: " << cropped_obj_bow_hist.size() << " bins" << endl;
        totalBin += cropped_obj_bow_hist.size();

        // MAP report num_kp
        //map_push_report(toString(num_kp) + ",");
        cout << "Total feature(s): " << num_kp << " point(s)" << endl;
        totalFeature += num_kp;

        // MAP report mask_pass
        //map_push_report(toString(mask_pass) + ",");
        cout << "Mask passed: " << mask_pass << " point(s)" << endl;
        totalPass += mask_pass;

        // Search
        cout << "==== Search Pass 2 Timing Info ====" << endl;
        cout << "Searching.."; cout.flush();
        startTime = CurrentPreciseTime();
        result_id = search_by_bow_sig(cropped_obj_bow_hist);    // Full ranking was in Result< pair<size_t dataset_id, float score> >
        searchTime = TimeElapse(startTime);
        totalSearchTime += searchTime;
        cout << "Match with: " << TotalMatch << " videos" << endl;
        cout << "Search time: " <<  searchTime << " s" << endl;
        cout << "Result (dataset_id) : " << result_id << endl;
        cout << "Rawrank path: " << cropped_obj_path[obj_id] << "_rawrank.txt" << endl;

        // Matching dump
        if (is_dump)
        {
            bool* cropped_bow_mask = new bool[run_param.CLUSTER_SIZE];
            for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)      // Empty cropped_bow_mask
                cropped_bow_mask[cluster_id] = false;
            for (size_t bin_id = 0; bin_id < cropped_obj_bow_hist.size(); bin_id++)                // Initialize cropped_bow_mask from cropped_obj_bow_hist
                cropped_bow_mask[cropped_obj_bow_hist[bin_id].cluster_id] = true;
            unordered_map<size_t, size_t> cropped_obj_bow_lut; // cluster_id -> vector_idx

            // Mapping cropped_obj_bow_hist lut
            for (size_t bin_id = 0; bin_id < cropped_obj_bow_hist.size(); bin_id++)
                cropped_obj_bow_lut[cropped_obj_bow_hist[bin_id].cluster_id] = bin_id;
            // Making dump list
            vector<size_t> dump_list;
            for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
                dump_list.push_back(Result[rank_id].first);
            // Start dump
            inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, dump_list, cropped_obj_path[obj_id]);
            // Release mem
            dump_list.clear();

            for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
            {
                vector<bow_bin_object> bow_result;
                size_t dataset_id = Result[rank_id].first;
                LoadSpecificBow(dataset_id, bow_result);

                for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
                {
                    size_t cluster_id = bow_result[bin_id].cluster_id;
                    if (cropped_bow_mask[cluster_id])
                    {
                        vector<feature_object>::const_iterator query_feature_it;
                        for (query_feature_it = cropped_obj_bow_hist[cropped_obj_bow_lut[cluster_id]].features.begin(); query_feature_it != cropped_obj_bow_hist[cropped_obj_bow_lut[cluster_id]].features.end(); query_feature_it++)
                        {
                            vector<feature_object>::iterator dataset_feature_it; // feature from inverted_index
                            for (dataset_feature_it = bow_result[bin_id].features.begin(); dataset_feature_it != bow_result[bin_id].features.end(); dataset_feature_it++)
                                inverted_hist.feature_matching_dump(dataset_id, cluster_id, bow_result[bin_id].freq, dataset_feature_it->x, dataset_feature_it->y, dataset_feature_it->a, dataset_feature_it->b, dataset_feature_it->c, query_feature_it->x, query_feature_it->y, query_feature_it->a, query_feature_it->b, query_feature_it->c);
                        }
                    }
                }

                // Release mem
                for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
                    bow_result[bin_id].features.clear();
                bow_result.clear();

            }

            inverted_hist.stop_matching_dump();

            delete[] cropped_bow_mask;
        }

        // Push Result to multirank
        AddRank(Result);

        // MAP report number of matched dataset
        //map_push_report(toString(TotalMatch) + ",");
        if (maxMatch < TotalMatch)
            maxMatch = TotalMatch;

        // MAP report search time usage
        //map_push_report(toString(searchTime) + ",");

        ExportEvalRank(cropped_obj_path[obj_id]);

        // Computing map
        float curr_ap = Compute_map(groundtruth_path.str());
        // MAP report map value, and close report for topic
        //map_push_report(toString(curr_ap) + "\n");

        ExportRawRank(cropped_obj_path[obj_id], curr_ap);
    }

    // MAP report bin size
    map_push_report(toString(totalBin) + ",");

    // MAP report num_kp
    map_push_report(toString(totalFeature) + ",");

    // MAP report mask_pass
    map_push_report(toString(totalPass) + ",");

    // Combine Multiple rank
    CombineRank();

    // MAP report total matches
    map_push_report(toString(maxMatch) + ",");

    // MAP report extract time usage
    map_push_report(toString(totalExtractTime) + ",");

    // MAP report search time usage
    map_push_report(toString(totalSearchTime) + ",");

    ExportEvalRank(query_path);

    ExportRawRank(query_path);

    // Release mem
    delete[] bootstrap_bow_mask;
    delete[] active_bow;
}

void QueryBootstrapping_v2(const string& query_path)
{
    double totalSearchTime = 0;

	// Resize image
	string query_scaled_path = query_path;
    if (run_param.query_scale_enable)
        query_scaled_path = ResizeQuery(query_path);

    cout << "==== Query bootstrapping v2 ====" << endl;
    /// Pass 1
    cout << "## Pass 1 ##" << endl;
    // Step 1, Extract BOW
    // Pack query list (with bootstrap)
    vector<string> queries;
    queries.push_back(query_scaled_path);

    // Request extract hist
    vector<bow_bin_object> bootstrap_bow_hist;
    cout << "Extracting BOW histogram.."; cout.flush();
    startTime = CurrentPreciseTime();
    extract_hist(simulated_session, queries, bootstrap_bow_hist);
    extractTime = TimeElapse(startTime);
    cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

    if (int(bootstrap_bow_hist.size()) < run_param.query_bootstrap_rankcheck)
    {
        cout << "Too small query or no feature can be extracted!" << endl;
        cout << "Switched to original query..." << endl;

        // Pack query list (with bootstrap)
        queries.clear();
        queries.push_back(query_path);

        // Request extract hist
        cout << "Extracting BOW histogram.."; cout.flush();
        startTime = CurrentPreciseTime();
        extract_hist(simulated_session, queries, bootstrap_bow_hist);
        extractTime = TimeElapse(startTime);
        cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;
    }


    // MAP report bin size
    //map_push_report(toString(bootstrap_bow_hist.size()) + ",");
    cout << "Bin amount: " << bootstrap_bow_hist.size() << " bins" << endl;

    // MAP report num_kp
    //map_push_report(toString(num_kp) + ",");
    cout << "Total feature(s): " << num_kp << " point(s)" << endl;

    // MAP report mask_pass
    //map_push_report(toString(mask_pass) + ",");
    cout << "Mask passed: " << mask_pass << " point(s)" << endl;

    // Step 2, Search BOW pass 1
    // Search
    cout << "==== Search Pass 1 Timing Info ====" << endl;
    cout << "Searching.."; cout.flush();
    startTime = CurrentPreciseTime();
    result_id = search_by_bow_sig(bootstrap_bow_hist);    // Full ranking was in Result< pair<size_t dataset_id, float score> >
    searchTime = TimeElapse(startTime);
    totalSearchTime += searchTime;
    cout << "Match with: " << TotalMatch << " videos" << endl;
    cout << "Search time: " <<  searchTime << " s" << endl;
    cout << "Result (dataset_id) : " << result_id << endl;

    // MAP report number of matched dataset
    //map_push_report(toString(TotalMatch) + ",");

    // MAP report search time usage
    //map_push_report(toString(searchTime) + ",");

    ExportEvalRank(query_scaled_path);

    // Computing map
    float curr_ap = Compute_map(groundtruth_path.str());
    // MAP report map value, and close report for topic
    //map_push_report(toString(curr_ap) + "\n");

    ExportRawRank(query_scaled_path, curr_ap);


    // Step 3, Matching bow with top rank, Filtering the same cluster_id between query and top results
    bool* bootstrap_bow_mask = new bool[run_param.CLUSTER_SIZE];
    int* active_bow = new int[run_param.CLUSTER_SIZE];
    vector< vector<bow_bin_object> > top_bow_sig;
    vector< pair<size_t, float> > active_bow_idf;
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)      // Empty bootstrap_mask and active_bow
    {
        bootstrap_bow_mask[cluster_id] = false;
        active_bow[cluster_id] = 0;
    }
    for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)                // Initialize bootstrap_mask from bootstrap_bow_hist
        bootstrap_bow_mask[bootstrap_bow_hist[bin_id].cluster_id] = true;

    // Matching cluster_id
    size_t max_rank_check = run_param.qbmining_topk;
    if (max_rank_check > Result.size())
        max_rank_check = Result.size();

    // Matching dump (for dumping bootstrap)
    bool is_dump = true;
    unordered_map<size_t, size_t> bootstrap_bow_lut; // cluster_id -> vector_idx
    if (is_dump)
    {
        // Mapping bootstrap_bow_hist lut
        for (size_t bin_id = 0; bin_id < bootstrap_bow_hist.size(); bin_id++)
            bootstrap_bow_lut[bootstrap_bow_hist[bin_id].cluster_id] = bin_id;
        // Making dump list
        vector<size_t> dump_list;
        for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
            dump_list.push_back(Result[rank_id].first);
        // Start dump
        inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, dump_list, query_scaled_path);
        // Release mem
        dump_list.clear();
    }
    for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
    {
        vector<bow_bin_object> bow_result;
        size_t dataset_id = Result[rank_id].first;
        LoadSpecificBow(dataset_id, bow_result);

        // Keep read bow to be reused
        top_bow_sig.push_back(bow_result);

        // QB1
        //QB1_Bow(bow_result);

        // Accumulate active bow
        for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
        {
            size_t cluster_id = bow_result[bin_id].cluster_id;
            if (run_param.query_bootstrap_minbow_type == MIN_BIN)
                active_bow[cluster_id]++;                                       // accumulate by bin
            else // MIN_FEATURE, MIN_IDF, QB_Mining
                active_bow[cluster_id] += bow_result[bin_id].features.size();   // accumulate by feature size

            // For dumping
            if (is_dump && bootstrap_bow_mask[cluster_id])
            {
                vector<feature_object>::const_iterator query_feature_it;
                for (query_feature_it = bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.begin(); query_feature_it != bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.end(); query_feature_it++)
                {
                    vector<feature_object>::iterator dataset_feature_it; // feature from inverted_index
                    for (dataset_feature_it = bow_result[bin_id].features.begin(); dataset_feature_it != bow_result[bin_id].features.end(); dataset_feature_it++)
                        inverted_hist.feature_matching_dump(dataset_id, cluster_id, bow_result[bin_id].freq, dataset_feature_it->x, dataset_feature_it->y, dataset_feature_it->a, dataset_feature_it->b, dataset_feature_it->c, query_feature_it->x, query_feature_it->y, query_feature_it->a, query_feature_it->b, query_feature_it->c);
                }
            }
        }
    }
    if (is_dump)
        inverted_hist.stop_matching_dump();

    // QB Mining
    unordered_map<size_t, float> fi_weight;
    if (run_param.qbmining_enable)
    {
        if (run_param.qbmining_mode == QB_FIW)
            FIW(query_path, bootstrap_bow_hist, top_bow_sig, run_param.qbmining_minsup, fi_weight);
        else if (run_param.qbmining_mode == QB_PREFIW)
            PREFIW(query_path, run_param.qbmining_minsup, fi_weight);
        else if (run_param.qbmining_mode == QB_GLOSD)
            GLOSD(query_path, bootstrap_bow_hist, top_bow_sig, run_param.qbmining_minsup, fi_weight);
        else if (run_param.qbmining_mode == QB_LOCSD)
            LOCSD(query_path, bootstrap_bow_hist, top_bow_sig, run_param.qbmining_minsup, fi_weight);
        else if (run_param.qbmining_mode == QB_FIX)
            FIX(query_path, bootstrap_bow_hist, top_bow_sig, run_param.qbmining_minsup, fi_weight);
        else if (run_param.qbmining_mode == QB_QEAVG)
            QE_AVG(query_path, bootstrap_bow_hist, top_bow_sig, run_param.qbmining_minsup, fi_weight);
    }

    // Matching Dump for QB_Mining
    if (run_param.qbmining_enable && is_dump)
    {
        ExportRawRank(query_scaled_path + "_qbmining", curr_ap);

        // Making dump list
        vector<size_t> dump_list;
        for (size_t rank_id = 0; rank_id < max_rank_check; rank_id++)
            dump_list.push_back(Result[rank_id].first);
        // Start dump
        inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, dump_list, query_scaled_path + "_qbmining");
        // Release mem
        dump_list.clear();

        for (size_t rank_id = 0; rank_id < top_bow_sig.size(); rank_id++)
        {
            size_t dataset_id = Result[rank_id].first;

            // QB1
            //QB1_Bow(top_bow_sig[rank_id]);

            // Accumulate active bow
            for (size_t bin_id = 0; bin_id < top_bow_sig[rank_id].size(); bin_id++)
            {
                size_t cluster_id = top_bow_sig[rank_id][bin_id].cluster_id;

                // For dumping only found in fi_weight
                if (bootstrap_bow_mask[cluster_id] && fi_weight.find(cluster_id) != fi_weight.end() && fi_weight[cluster_id] > 0)
                {
                    vector<feature_object>::const_iterator query_feature_it;
                    for (query_feature_it = bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.begin(); query_feature_it != bootstrap_bow_hist[bootstrap_bow_lut[cluster_id]].features.end(); query_feature_it++)
                    {
                        vector<feature_object>::iterator dataset_feature_it; // feature from inverted_index
                        for (dataset_feature_it = top_bow_sig[rank_id][bin_id].features.begin(); dataset_feature_it != top_bow_sig[rank_id][bin_id].features.end(); dataset_feature_it++)
                            inverted_hist.feature_matching_dump(dataset_id, cluster_id, top_bow_sig[rank_id][bin_id].freq, dataset_feature_it->x, dataset_feature_it->y, dataset_feature_it->a, dataset_feature_it->b, dataset_feature_it->c, query_feature_it->x, query_feature_it->y, query_feature_it->a, query_feature_it->b, query_feature_it->c);
                    }
                }
            }
        }
        if (is_dump)
            inverted_hist.stop_matching_dump();
    }

    // for MIN_IDF
    size_t total_pass_feature = 0;
    bool* idf_pass_bin = new bool[run_param.CLUSTER_SIZE];
    if (run_param.qbmining_enable || run_param.query_bootstrap_minbow_type == MIN_IDF)
    {
        /// Recalculate idf score
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        {
            if (active_bow[cluster_id] > 0)
            {
                if (run_param.qbmining_enable)
                {
                    // tf-fi-idf (fi = frequent item, relevant to top list)
                    if (fi_weight.find(cluster_id) != fi_weight.end() && fi_weight[cluster_id] > 0)  // Skip if no frequent item, or has zero weight
                    {
                        //cout << fi_weight[cluster_id] << " ";
                        active_bow_idf.push_back(pair<size_t, float>(cluster_id, (1 + log10(active_bow[cluster_id])) * fi_weight[cluster_id] * inverted_hist.get_idf(cluster_id)));
                        total_pass_feature += active_bow[cluster_id];
                    }
                }
                else
                    // tf-idf
                    active_bow_idf.push_back(pair<size_t, float>(cluster_id, (1 + log10(active_bow[cluster_id])) * inverted_hist.get_idf(cluster_id)));
            }
        }
        //cout << endl;
        // Sort
        sort(active_bow_idf.begin(), active_bow_idf.end(), compare_pair_second<>());

        /// Mask top bin in idf sorted list
        //int top_idf_bin = active_bow_idf.size() * run_param.query_bootstrap_minbow / 100;
        //int bin_count = 0;
        size_t top_idf_feature = total_pass_feature * run_param.query_bootstrap_minbow / 100;
        size_t feature_count = 0;
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            idf_pass_bin[cluster_id] = false;
        for (vector< pair<size_t, float> >::iterator active_bow_idf_it = active_bow_idf.begin(); active_bow_idf_it != active_bow_idf.end(); active_bow_idf_it++)
        {
            size_t cluster_id = active_bow_idf_it->first;
            if (active_bow[cluster_id] > 0)    // Best value to filter bin with feature count > 1
            {
                if (feature_count < top_idf_feature)
                {
                    idf_pass_bin[cluster_id] = true;
                    feature_count += active_bow[cluster_id];
                    //cout << "feature_count:" << feature_count << " cluster_id:" << cluster_id << " idf:" << active_bow_idf_it->second << endl;
                }
                else
                    break;
            }
        }
    }

    // Step 4, Combine new BOW and make new mask
    // Preparing new big bow
    // Filtering mask by min_bow_thre
    int min_bow_thre = run_param.query_bootstrap_minbow;
    unordered_map<size_t, vector<feature_object> > big_bow_prepare_space;   // cluster_id -> features
    for (size_t rank_id = 0; rank_id < top_bow_sig.size(); rank_id++)
    {
        // QB1
        //QB1_Bow(top_bow_sig[rank_id]);

        // Combine bow
        int skip_bin = 0;
        int pass_bin = 0;
        int skip_feature = 0;
        int pass_feature = 0;
        for (size_t bin_id = 0; bin_id < top_bow_sig[rank_id].size(); bin_id++)
        {
            size_t cluster_id = top_bow_sig[rank_id][bin_id].cluster_id;
            if (run_param.qbmining_enable || run_param.query_bootstrap_minbow_type == MIN_IDF)
            {
                // QB_Mining, MIN_IDF
                if (idf_pass_bin[cluster_id])
                {
                    vector<feature_object>::iterator dataset_feature_it; // features read from dataset bow
                    for (dataset_feature_it = top_bow_sig[rank_id][bin_id].features.begin(); dataset_feature_it != top_bow_sig[rank_id][bin_id].features.end(); dataset_feature_it++)
                        big_bow_prepare_space[cluster_id].push_back(*dataset_feature_it);
                    pass_bin++;
                    pass_feature += top_bow_sig[rank_id][bin_id].features.size();
                }
                else
                {
                    skip_bin++;
                    skip_feature += top_bow_sig[rank_id][bin_id].features.size();
                }
            }
            else    // MIN_FEATURE, MIN_BIN
            {
                if (active_bow[cluster_id] >= min_bow_thre)
                {
                    vector<feature_object>::iterator dataset_feature_it; // features read from dataset bow
                    for (dataset_feature_it = top_bow_sig[rank_id][bin_id].features.begin(); dataset_feature_it != top_bow_sig[rank_id][bin_id].features.end(); dataset_feature_it++)
                        big_bow_prepare_space[cluster_id].push_back(*dataset_feature_it);
                    pass_bin++;
                    pass_feature += top_bow_sig[rank_id][bin_id].features.size();
                }
                else
                {
                    skip_bin++;
                    skip_feature += top_bow_sig[rank_id][bin_id].features.size();
                }
            }
        }
        cout << "Skip bin: " << skip_bin << " Pass bin: " << pass_bin << " Skip feature: " << skip_feature << " Pass feature: " << pass_feature << endl;
    }

    // Release Mem
    delete[] idf_pass_bin;

    // Convert to big_bow with idf
    vector<bow_bin_object> big_bow_hist;    // cluster_id -> bow_bin_object
    bool* big_bow_mask = new bool[run_param.CLUSTER_SIZE];
    int feature_pass = 0;
    unordered_map<size_t, vector<feature_object> >::iterator big_bow_prepare_space_it;
    for (big_bow_prepare_space_it = big_bow_prepare_space.begin(); big_bow_prepare_space_it != big_bow_prepare_space.end(); big_bow_prepare_space_it++)
    {
        // Preparing new bin with idf frequency
        bow_bin_object big_bin;
        big_bin.cluster_id = big_bow_prepare_space_it->first;
        big_bin.features.swap(big_bow_prepare_space_it->second);
        big_bin.freq = (1 + log10(big_bin.features.size())) * inverted_hist.get_idf(big_bin.cluster_id);

        // Keep new cin to big_bow_hist
        big_bow_hist.push_back(big_bin);

        // Make big_bow_mask
        big_bow_mask[big_bin.cluster_id] = true;
        feature_pass += big_bin.features.size();
    }

    /*
    /// Normalization
    // Unit length
    float sum_of_square = 0.0f;
    float unit_length = 0.0f;
    for (size_t bin_idx = 0; bin_idx < big_bow_hist.size(); bin_idx++)
        sum_of_square += big_bow_hist[bin_idx].freq * big_bow_hist[bin_idx].freq;
    unit_length = sqrt(sum_of_square);

    // Normalizing
    for (size_t bin_idx = 0; bin_idx < big_bow_hist.size(); bin_idx++)
        big_bow_hist[bin_idx].freq = big_bow_hist[bin_idx].freq / unit_length;
    */

    /// Pass 2
    cout << "## Pass 2 ##" << endl;
    // Step 5, Search pass 2, by new big_bow_hist

    // MAP report bin size
    map_push_report(toString(big_bow_hist.size()) + ",");
    cout << "Bin amount: " << big_bow_hist.size() << " bins" << endl;

    // MAP report num_kp
    map_push_report(toString(feature_pass) + ",");
    cout << "Total feature(s): " << feature_pass << " point(s)" << endl;

    // MAP report mask_pass
    map_push_report(toString(feature_pass) + ",");
    cout << "Mask passed: " << feature_pass << " point(s)" << endl;

    // Search
    cout << "==== Search Pass 2 Timing Info ====" << endl;
    cout << "Searching.."; cout.flush();
    startTime = CurrentPreciseTime();
    result_id = search_by_bow_sig(big_bow_hist);    // Full ranking was in Result< pair<size_t dataset_id, float score> >
    searchTime = TimeElapse(startTime);
    totalSearchTime += searchTime;
    cout << "Match with: " << TotalMatch << " videos" << endl;
    cout << "Search time: " <<  searchTime << " s" << endl;
    cout << "Result (dataset_id) : " << result_id << endl;
    cout << "Rawrank path: " << query_path << "_rawrank.txt" << endl;


    // MAP report number of matched dataset
    map_push_report(toString(TotalMatch) + ",");

    // MAP report extract time usage
    map_push_report(toString(extractTime) + ",");

    // MAP report search time usage
    map_push_report(toString(totalSearchTime) + ",");

    ExportEvalRank(query_path);

    // Computing map
    curr_ap = Compute_map(groundtruth_path.str());

    ExportRawRank(query_path, curr_ap);

    // Release mem
    delete[] bootstrap_bow_mask;
    delete[] active_bow;
}

void FIW(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight)
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
            transaction_buffer << query_bow[bin_id].cluster_id << " ";
        transaction_buffer << endl;
    }
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            transaction_buffer << bow_sig_pack[bow_id][bin_id].cluster_id << " ";
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
        const char* delimsItem = ",";
        vector<string> Items;   // Item [0-(n-2)], Support [n-1]
        string_splitter(frequent_item_sets[set_id], delimsItem, Items);

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
        const char* delimsItem = " ";
        vector<string> Items;   // Item [0-(n-2)], Support [n-1]
        string_splitter(frequent_item_sets[set_id], delimsItem, Items);

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

void FIX(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id].cluster_id] = true;

    // Calculate item weight from ACW
    unordered_map<size_t, size_t> frequent_item_count;      // cluster_id, count
    unordered_map<size_t, float> frequent_item_weight;      // cluster_id, weight
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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

void GLOSD(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id].cluster_id] = true;

    // Calculate count for each class
    unordered_map<size_t, size_t> frequent_item_count;      // cluster_id, count
    unordered_map<size_t, float> frequent_item_weight;      // cluster_id, weight
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        pass_pcut << bow_id << ",";
    pass_pcut << "all_bin" << endl;
    // Write total bin
    pass_pcut << "total_bin,";
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        pass_pcut << bow_sig_pack[bow_id].size() << ",";
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
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            bin_pass = 0;
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            size_t total_feature_pass_count = 0;

            // Accumulate total feature pass count
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

                // Skip if not fall within query
                //if (!query_bin_mask[cluster_id])
                    //continue;

                if (bin_pass_mask[cluster_id])
                    total_feature_pass_count += bow_sig_pack[bow_id][bin_id].features.size();
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
                for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
                {
                    size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

                    // Skip if not fall within query
                    //if (!query_bin_mask[cluster_id])
                        //continue;

                    if (bin_pass_mask[cluster_id])
                    {
                        for (size_t feature_id = 0; feature_id < bow_sig_pack[bow_id][bin_id].features.size(); feature_id++)
                        {
                            rank_centroid[0] += vertx[vert_count] = bow_sig_pack[bow_id][bin_id].features[feature_id].x;    // Accumulating centroid
                            rank_centroid[1] += verty[vert_count] = bow_sig_pack[bow_id][bin_id].features[feature_id].y;
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
        frequent_item_count.clear();
        frequent_item_weight.clear();
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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
            sd_diff.clear();
            pcut_sd_diff_label.clear();
            sd_diff_slope.clear();
            pcut_sd_diff_slope_label.clear();
        }

        // Debut best_pcut
        cout << "Debug good pcut: ";
        for (size_t bow_id = 0; bow_id < good_pcut.size(); bow_id++)
            cout << good_pcut[bow_id] << " ";
        cout << endl;


        // Calculate item weight from ACW
        frequent_item_count.clear();
        frequent_item_weight.clear();
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            bin_pass = 0;
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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

void LOCSD(const string& query_path, const vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id].cluster_id] = true;

    // Calculate item weight from ACW
    unordered_map<size_t, size_t> frequent_item_count;      // cluster_id, count
    unordered_map<size_t, float> frequent_item_weight;      // cluster_id, weight
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

            // Skip if not fall within query
            if (!query_bin_mask[cluster_id])
                continue;

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
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        pass_pcut << bow_id << ",";
    pass_pcut << "all_bin" << endl;
    // Write total bin
    pass_pcut << "total_bin,";
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        pass_pcut << bow_sig_pack[bow_id].size() << ",";
    pass_pcut << frequent_item_count.size() << endl;

    // Write cut for each rank
    // pcut is percentage
    vector< vector<float> > compactness;
    int pcut_step = 5;
    for (int pcut = pcut_step; pcut <= 100; pcut += pcut_step)
    {
        pass_pcut << pcut << ",";
        /// -- Pass test
        unordered_map<size_t, size_t> bin_pass_count;       // cluster_id, pass_count
        unordered_map<size_t, bool> bin_pass_mask;          // cluster_id, pass_flag
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            bin_pass = 0;
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

                // Skip if not fall within query
                if (!query_bin_mask[cluster_id])
                    continue;

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
        for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
        {
            size_t total_feature_pass_count = 0;

            // Accumulate total feature pass count
            for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
            {
                size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

                // Skip if not fall within query
                if (!query_bin_mask[cluster_id])
                    continue;

                if (bin_pass_mask[cluster_id])
                    total_feature_pass_count += bow_sig_pack[bow_id][bin_id].features.size();
            }

            // Continue finding TSP if we have vertex pass p_cut more than 3 points
            // 1. Finding TSP
            // 2. Cut far distance out by otsu thresholding
            // 3. Calculate compactness for each small group
            // 4. Average compactness
            //
            // Compactness is mean of square distance to centriod
            // Distance = sqrt((x1-xm)^2+(y1-ym)^2)
            // So, compactness = sum(distance^2)/n
            //                 = sum(sqrt((x1-xm)^2+(y1-ym)^2)^2)/n
            //                 = sum((x1-xm)^2+(y1-ym)^2)/n
            float total_average_compactness = 0.0f;
            if (total_feature_pass_count > 3)
            {
                // 1.1 Packing feature
                size_t vert_count = 0;
                float* vertx = new float[total_feature_pass_count];
                float* verty = new float[total_feature_pass_count];
                for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
                {
                    size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

                    // Skip if not fall within query
                    if (!query_bin_mask[cluster_id])
                        continue;

                    if (bin_pass_mask[cluster_id])
                    {
                        for (size_t feature_id = 0; feature_id < bow_sig_pack[bow_id][bin_id].features.size(); feature_id++)
                        {
                            vertx[vert_count] = bow_sig_pack[bow_id][bin_id].features[feature_id].x;
                            verty[vert_count] = bow_sig_pack[bow_id][bin_id].features[feature_id].y;
                            vert_count++;
                        }
                    }
                }

                // 1.2 Create symmat distance matrix
                float* symmat_dist;
                create_symmat(symmat_dist, total_feature_pass_count);
                calculate_symmat_distance(total_feature_pass_count, vertx, verty, symmat_dist);

                // 1.3 Finding TSP
                tsp path_solver(total_feature_pass_count, symmat_dist);

                vector<size_t> best_path = path_solver.search_path(tsp::TSP_LINKNJOY);
                cout << "get path size: " << best_path.size() << " done!" << endl;

                // 2.1 Finding threshold
                float threshold = path_solver.get_otsu_inter_distance() * 0.30;
                cout << "Threshold: " << threshold << endl;

                // 3.1 Cut group
                size_t best_path_size = best_path.size();
                size_t group_feature_first_idx = 0;
                size_t group_feature_last_idx = 0;
                float curr_group_average_x = 0.0f;
                float curr_group_average_y = 0.0f;
                vector< pair<size_t, size_t> > feature_groups;
                Mat group_img_out(1000, 1000, CV_8UC3, Scalar(0));  // visualize group
                for (size_t node_idx = 0; node_idx < best_path_size - 1; node_idx++)
                {
                    size_t node_label = best_path[node_idx];
                    size_t next_node_label = best_path[node_idx + 1];

                    // Accumulating sum of vertex
                    curr_group_average_x += vertx[node_label];
                    curr_group_average_y += verty[node_label];

                    // Found seperated group
                    // If the group was found, and the size of group is large enough, then running
                    if (path_solver.get_cost_between(node_label, next_node_label) > threshold || node_idx == best_path_size - 2)
                    {
                        group_feature_last_idx = node_idx;
                        size_t group_size = group_feature_last_idx - group_feature_first_idx + 1;
                        // Filtering total sub group with at least 3 points per group
                        if (group_size >= 3)
                        {
                            // Save group range
                            feature_groups.push_back(pair<size_t, size_t>(group_feature_first_idx, group_feature_last_idx));

                            // Calculating Centroid
                            curr_group_average_x /= group_size;
                            curr_group_average_y /= group_size;

                            // Calculate Compactness
                            float current_group_compactness = 0.0f;
                            for (size_t group_node_idx = group_feature_first_idx; group_node_idx < group_feature_last_idx; group_node_idx++)
                            {
                                // Calculating square_distance to centroid
                                size_t group_node_label = best_path[group_node_idx];
                                float square_dist = (vertx[group_node_label] - curr_group_average_x) * (vertx[group_node_label] - curr_group_average_x) +
                                                    (verty[group_node_label] - curr_group_average_y) * (verty[group_node_label] - curr_group_average_y);

                                // Sum of square distance
                                current_group_compactness += square_dist;
                            }
                            // Compactness = Sum(d^2) / n
                            current_group_compactness /= group_size;

                            // Accumulating total compactness
                            total_average_compactness += current_group_compactness;

                            // Visualize group and centriod
                            Point2f pcentroid(curr_group_average_x * 1000, curr_group_average_y * 1000);
                            for (size_t group_node_idx = group_feature_first_idx; group_node_idx <= group_feature_last_idx; group_node_idx++)
                            {
                                size_t node_idx = best_path[group_node_idx];
                                Point2f pstart(vertx[node_idx] * 1000, verty[node_idx] * 1000);

                                // Point-and-centroid
                                line(group_img_out, pstart, pcentroid, Scalar(0, 0, 255), 1, CV_AA);
                                circle(group_img_out, pstart, 0, Scalar(0, 255, 0), 2, CV_AA);
                            }
                            circle(group_img_out, pcentroid, 0, Scalar(255, 0, 0), 2, CV_AA);
                        }
                        // Rollnext
                        group_feature_first_idx = group_feature_last_idx + 1;

                        // Reset temperature variable
                        curr_group_average_x = 0;
                        curr_group_average_y = 0;
                    }
                }
                // Calculating average compactness
                if (feature_groups.size() > 0)
                    total_average_compactness /= feature_groups.size();
                else
                    total_average_compactness = 0.0f;

                /*stringstream path_out;
                path_out << "/home/stylix/webstylix/code/ins_online/out_test/group_img_out_" << bow_id << "_" << pcut << "_lnj.png";
                cout << "Write out image.."; cout.flush();
                imwrite(path_out.str(), group_img_out);
                cout << "done" << endl;*/

                Mat test_img_out(1000, 1000, CV_8UC3, Scalar(0));

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

                /*stringstream path2_out;
                path2_out << "/home/stylix/webstylix/code/ins_online/out_test/group_img_out_" << bow_id << "_" << pcut << "_lnj_p.png";
                cout << "Write out image.."; cout.flush();
                imwrite(path2_out.str(), test_img_out);
                cout << "done" << endl;*/

                // Release mem
                delete[] vertx;
                delete[] verty;
                delete[] symmat_dist;

                // Save Compactness for finding best minsup later
                if (bow_id + 1 > compactness.size())
                {
                    vector<float> sub_compactness;
                    sub_compactness.push_back(total_average_compactness);
                    compactness.push_back(sub_compactness);
                }
                else
                    compactness[bow_id].push_back(total_average_compactness);
            }
            // Print out compactness
            pass_pcut << total_average_compactness << ",";
        }
        // pass one cut, continue next cut
        pass_pcut << endl;
    }


    // Compactness Debug
    cout << setprecision(5) << fixed;
    for (size_t bow_id = 0; bow_id < compactness.size(); bow_id++)
    {
        for (size_t cut_id = 0; cut_id < compactness[bow_id].size(); cut_id++)
        {
            cout << compactness[bow_id][cut_id] << " ";
        }
        cout << endl;
    }

    /// Finding best p_cut for each rank_id
    vector<size_t> good_pcut;
    for (size_t bow_id = 0; bow_id < compactness.size(); bow_id++)
    {
        // Find average compactness for all pcut
        float average_compactness = 0.0f;
        size_t pcut_count = 0;
        for (size_t pcut_id = 0; pcut_id < compactness[bow_id].size(); pcut_id++)
        {
            if (compactness[bow_id][pcut_id] > 0.0f)
            {
                average_compactness += compactness[bow_id][pcut_id];
                pcut_count++;
            }
        }
        average_compactness /= pcut_count;

        // Find first pcut below average_compactness
        for (size_t pcut_id = 0; pcut_id < compactness[bow_id].size(); pcut_id++)
        {
            if (compactness[bow_id][pcut_id] < average_compactness)
            {
                good_pcut.push_back(pcut_id);
                break;
            }
        }
    }

    // Debut best_pcut
    cout << "Debug good pcut: ";
    for (size_t pcut_id = 0; pcut_id < good_pcut.size(); pcut_id++)
        cout << good_pcut[pcut_id] << " ";
    cout << endl;


    // Calculate item weight from ACW
    frequent_item_count.clear();
    frequent_item_weight.clear();
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

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
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        bin_pass = 0;
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

            // Skip if not fall within query
            //if (!query_bin_mask[cluster_id])
                //continue;

            auto_minsup_filter[cluster_id] |= float(frequent_item_count[cluster_id]) / item_count_max >= ((good_pcut[bow_id] + 6) * pcut_step) / 100.0f;
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

    // Write to file
    text_write(query_path + "_rankcut.csv", pass_pcut.str(), false);

    // Release memory
    delete[] query_bin_mask;
    delete[] auto_minsup_filter;
}

void QE_AVG(const string& query_path, vector<bow_bin_object>& query_bow, const vector< vector<bow_bin_object> >& bow_sig_pack, int minsup, unordered_map<size_t, float>& fi_weight)
{
    bool* query_bin_mask = new bool[run_param.CLUSTER_SIZE];        // cluster_id, flag
    for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        query_bin_mask[cluster_id] = false;
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bin_mask[query_bow[bin_id].cluster_id] = true;


    // Skip if not fall within query
    //if (!query_bin_mask[cluster_id])
        //continue;


    // Clear old bow
    for (size_t bin_id = 0; bin_id < query_bow.size(); bin_id++)
        query_bow[bin_id].features.clear();
    query_bow.clear();

    // Initialize blank sparse bow
    unordered_map<size_t, float> frequent_item_weight;
    unordered_map<size_t, vector<feature_object> > curr_sparse_bow; // cluster_id, features
    mask_pass = 0;
    // Set bow
    // Add feature to curr_sparse_bow at cluster_id
    // Frequency of bow is curr_sparse_bow[].size()
    for (size_t bow_id = 0; bow_id < bow_sig_pack.size(); bow_id++)
    {
        for (size_t bin_id = 0; bin_id < bow_sig_pack[bow_id].size(); bin_id++)
        {
            // Get cluster from quantizad index of feature
            size_t cluster_id = bow_sig_pack[bow_id][bin_id].cluster_id;

            for (size_t feature_id = 0; feature_id < bow_sig_pack[bow_id][bin_id].features.size(); feature_id++)
                // Keep new feature into its corresponding bin (cluster_id)
                curr_sparse_bow[cluster_id].push_back(bow_sig_pack[bow_id][bin_id].features[feature_id]);

            if (frequent_item_weight.find(cluster_id) == frequent_item_weight.end())
                frequent_item_weight[cluster_id] = 0;
            frequent_item_weight[cluster_id] += bow_sig_pack[bow_id][bin_id].features.size();
        }
    }

    /// Make compact bow, with tf-idf frequency
    for (unordered_map<size_t, vector<feature_object> >::iterator sparse_bow_it = curr_sparse_bow.begin(); sparse_bow_it != curr_sparse_bow.end(); sparse_bow_it++)
    {
        // Looking for non-zero bin of cluster,
        // then put that bin together with specified cluster_id
        // Create new bin with cluster_id, frequency, and its features
        bow_bin_object bow_bin;
        bow_bin.cluster_id = sparse_bow_it->first;

        // Feature weight acculumating
        float feature_weight = 0.0f;
        for (size_t feature_id = 0; feature_id < sparse_bow_it->second.size(); feature_id++)
            feature_weight += sparse_bow_it->second[feature_id].weight;

        // tf-idf
        if ((int)feature_weight > 0)
            feature_weight = (1 + log10(feature_weight)) * inverted_hist.get_idf(bow_bin.cluster_id); // tf*idf = log10(feature_weight) * idf[cluster_id]
        else
            continue;   // Skip adding to bow

        // Average QE
        frequent_item_weight[bow_bin.cluster_id] = 1;
        bow_bin.freq = feature_weight / bow_sig_pack.size();
        bow_bin.features.swap(sparse_bow_it->second);

        // Keep new bin into compact_bow
        query_bow.push_back(bow_bin);
    }

    /// Normalization
    // Unit length
    float sum_of_square = 0.0f;
    float unit_length = 0.0f;
    for (size_t bin_idx = 0; bin_idx < query_bow.size(); bin_idx++)
        sum_of_square += query_bow[bin_idx].freq * query_bow[bin_idx].freq;
    unit_length = sqrt(sum_of_square);

    // Normalizing
    for (size_t bin_idx = 0; bin_idx < query_bow.size(); bin_idx++)
        query_bow[bin_idx].freq = query_bow[bin_idx].freq / unit_length;

    // Keep weight
    fi_weight.swap(frequent_item_weight);
}

void ScanningQuery(const string& query_path)
{
    int totalBin = 0;
    int totalFeature = 0;
    int totalPass = 0;
    int maxMatch = 0;
    double totalExtractTime = 0;
    double totalSearchTime = 0;

    Mat original_img = imread(query_path);
    vector<string> scan_subwin_path;

    cout << "==== Scanning window ====" << endl;
    // Scanning holizontal, vertical count
    int scan_x_count = (original_img.cols - run_param.scanning_window_width) / run_param.scanning_window_shift_x;
    int scan_y_count = (original_img.rows - run_param.scanning_window_height) / run_param.scanning_window_shift_y;
    cout << "Total window count = " << scan_x_count * scan_y_count << " [" << scan_x_count << "x" << scan_y_count << "]" << endl;
    cout << "Cropping windows.."; cout.flush();
    startTime = CurrentPreciseTime();
    for (int x_count = 0; x_count <= scan_x_count; x_count++)
    {
        for (int y_count = 0; y_count <= scan_y_count; y_count++)
        {
            // Calculate current scanning window
            Rect curr_window(   x_count * run_param.scanning_window_shift_x,
                                y_count * run_param.scanning_window_shift_y,
                                run_param.scanning_window_width,
                                run_param.scanning_window_height     );

            // Keep cropped sub-window path
            scan_subwin_path.push_back(query_path + "_" + toString(curr_window.x) + "_" + toString(curr_window.y) + "_" + toString(curr_window.width) + "_" + toString(curr_window.height) + ".jpg");

            // Crop
            // Write image object
            vector<int> imwrite_param;
            imwrite_param.push_back(CV_IMWRITE_JPEG_QUALITY);
            imwrite_param.push_back(85);
            imwrite(scan_subwin_path[scan_subwin_path.size() - 1], original_img(curr_window), imwrite_param);
        }
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Searching
    for (size_t window_id = 0; window_id < scan_subwin_path.size(); window_id++)
    {
        cout << "Searching window #" << window_id << endl;

        // MAP report query id
        //map_push_report("n/a,");

        // MAP report query name
        //map_push_report("n/a,");

        // MAP report query source image
        //map_push_report(get_filename(scan_subwin_path[obj_id]) + ",");

        // MAP report mask
        //map_push_report("bootstrap-cropped,");

        // Pack query list (with bootstrap)
        vector<string> subwindow_query;
        subwindow_query.push_back(scan_subwin_path[window_id]);

        // Request extract hist
        vector<bow_bin_object> subwindow_bow_hist;
        cout << "Extracting BOW histogram.."; cout.flush();
        startTime = CurrentPreciseTime();
        extract_hist(simulated_session, subwindow_query, subwindow_bow_hist);
        extractTime = TimeElapse(startTime);
        totalExtractTime += extractTime;
        cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

        if (subwindow_bow_hist.empty())
        {
            cout << "Skipped query: " << scan_subwin_path[window_id] << " - no keypoint detected" << endl;
            continue;
        }

        // MAP report bin size
        //map_push_report(toString(subwindow_bow_hist.size()) + ",");
        cout << "Bin amount: " << subwindow_bow_hist.size() << " bins" << endl;
        totalBin += subwindow_bow_hist.size();

        // MAP report num_kp
        //map_push_report(toString(num_kp) + ",");
        cout << "Total feature(s): " << num_kp << " point(s)" << endl;
        totalFeature += num_kp;

        // MAP report mask_pass
        //map_push_report(toString(mask_pass) + ",");
        cout << "Mask passed: " << mask_pass << " point(s)" << endl;
        totalPass += mask_pass;

        // Search
        cout << "==== Search Pass 2 Timing Info ====" << endl;
        cout << "Searching.."; cout.flush();
        startTime = CurrentPreciseTime();
        result_id = search_by_bow_sig(subwindow_bow_hist);    // Full ranking was in Result< pair<size_t dataset_id, float score> >
        searchTime = TimeElapse(startTime);
        totalSearchTime += searchTime;
        cout << "Match with: " << TotalMatch << " videos" << endl;
        cout << "Search time: " <<  searchTime << " s" << endl;
        cout << "Result (dataset_id) : " << result_id << endl;
        cout << "Rawrank path: " << scan_subwin_path[window_id] << "_rawrank.txt" << endl;

        // Matching dump
        bool is_dump = false;
        int max_matches_show = 20;
        if (is_dump)
        {
            bool* subwin_bow_mask = new bool[run_param.CLUSTER_SIZE];
            for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)      // Empty subwin_bow_mask
                subwin_bow_mask[cluster_id] = false;
            for (size_t bin_id = 0; bin_id < subwindow_bow_hist.size(); bin_id++)                // Initialize subwin_bow_mask from subwindow_bow_hist
                subwin_bow_mask[subwindow_bow_hist[bin_id].cluster_id] = true;
            unordered_map<size_t, size_t> subwin_bow_lut; // cluster_id -> vector_idx

            // Mapping subwindow_bow_hist lut
            for (size_t bin_id = 0; bin_id < subwindow_bow_hist.size(); bin_id++)
                subwin_bow_lut[subwindow_bow_hist[bin_id].cluster_id] = bin_id;
            // Making dump list
            vector<size_t> dump_list;
            for (int rank_id = 0; rank_id < max_matches_show; rank_id++)
                dump_list.push_back(Result[rank_id].first);
            // Start dump
            inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, dump_list, scan_subwin_path[window_id]);
            // Release mem
            dump_list.clear();

            for (int rank_id = 0; rank_id < max_matches_show; rank_id++)
            {
                vector<bow_bin_object> bow_result;
                size_t dataset_id = Result[rank_id].first;
                LoadSpecificBow(dataset_id, bow_result);

                for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
                {
                    size_t cluster_id = bow_result[bin_id].cluster_id;
                    if (subwin_bow_mask[cluster_id])
                    {
                        vector<feature_object>::const_iterator query_feature_it;
                        for (query_feature_it = subwindow_bow_hist[subwin_bow_lut[cluster_id]].features.begin(); query_feature_it != subwindow_bow_hist[subwin_bow_lut[cluster_id]].features.end(); query_feature_it++)
                        {
                            vector<feature_object>::iterator dataset_feature_it; // feature from inverted_index
                            for (dataset_feature_it = bow_result[bin_id].features.begin(); dataset_feature_it != bow_result[bin_id].features.end(); dataset_feature_it++)
                                inverted_hist.feature_matching_dump(dataset_id, cluster_id, bow_result[bin_id].freq, dataset_feature_it->x, dataset_feature_it->y, dataset_feature_it->a, dataset_feature_it->b, dataset_feature_it->c, query_feature_it->x, query_feature_it->y, query_feature_it->a, query_feature_it->b, query_feature_it->c);
                        }
                    }
                }

                // Release mem
                for (size_t bin_id = 0; bin_id < bow_result.size(); bin_id++)
                    bow_result[bin_id].features.clear();
                bow_result.clear();

            }

            inverted_hist.stop_matching_dump();

            delete[] subwin_bow_mask;
        }

        // Push Result to multirank
        AddRank(Result);

        // MAP report number of matched dataset
        //map_push_report(toString(TotalMatch) + ",");
        if (maxMatch < TotalMatch)
            maxMatch = TotalMatch;

        // MAP report search time usage
        //map_push_report(toString(searchTime) + ",");

        ExportEvalRank(scan_subwin_path[window_id]);

        // Computing map
        float curr_ap = Compute_map(groundtruth_path.str());
        // MAP report map value, and close report for topic
        //map_push_report(toString(curr_ap) + "\n");

        ExportRawRank(scan_subwin_path[window_id], curr_ap);
    }

    // MAP report bin size
    map_push_report(toString(totalBin) + ",");

    // MAP report num_kp
    map_push_report(toString(totalFeature) + ",");

    // MAP report mask_pass
    map_push_report(toString(totalPass) + ",");

    // Combine Multiple rank
    CombineRank();

    // MAP report total matches
    map_push_report(toString(maxMatch) + ",");

    // MAP report extract time usage
    map_push_report(toString(totalExtractTime) + ",");

    // MAP report search time usage
    map_push_report(toString(totalSearchTime) + ",");

    ExportEvalRank(query_path);

    ExportRawRank(query_path);

    // Release mem

}

void DirectQuery(const string& query_path)
{
	// Resize image
	string query_scaled_path = query_path;
    if (run_param.query_scale_enable)
        query_scaled_path = ResizeQuery(query_path);

    // Pack query list
    vector<string> queries;
    queries.push_back(query_scaled_path);

    // Request extract hist
    vector<bow_bin_object> bow_hist;
    cout << "Extracting BOW histogram.."; cout.flush();
    startTime = CurrentPreciseTime();
    extract_hist(simulated_session, queries, bow_hist);
    extractTime = TimeElapse(startTime);
    cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;

	if (int(bow_hist.size()) < 4)
    {
        cout << "Too small query or no feature can be extracted!" << endl;
        cout << "Switched to original query..." << endl;

        // Pack query list
        queries.clear();
        queries.push_back(query_path);

        // Request extract hist
        cout << "Extracting BOW histogram.."; cout.flush();
        startTime = CurrentPreciseTime();
        extract_hist(simulated_session, queries, bow_hist);
        extractTime = TimeElapse(startTime);
        cout << "done! (in " << setprecision(2) << fixed << extractTime << " s)" << endl;
    }

    // MAP report bin size
    map_push_report(toString(bow_hist.size()) + ",");
    cout << "Bin amount: " << bow_hist.size() << " bins" << endl;

    // MAP report num_kp
    map_push_report(toString(num_kp) + ",");
    cout << "Total feature(s): " << num_kp << " point(s)" << endl;

    // MAP report mask_pass
    map_push_report(toString(mask_pass) + ",");
    cout << "Mask passed: " << mask_pass << " point(s)" << endl;

    // Search matching visualization
    bool is_dump = false;
    if (is_dump)
        inverted_hist.start_matching_dump(run_param.dataset_root_dir, ImgParentPaths, ImgParentsIdx, ImgLists, vector<size_t>(), query_scaled_path);

    // Search
    cout << "==== Search Timing Info ====" << endl;
    startTime = CurrentPreciseTime();
    result_id = search_by_bow_sig(bow_hist);
    searchTime = TimeElapse(startTime);
    cout << "Match with: " << TotalMatch << " videos" << endl;
    cout << "Search time: " <<  searchTime << " s" << endl;
    cout << "Result (dataset_id) : " << result_id << endl;

    // Stop matching dump
    if (is_dump)
        inverted_hist.stop_matching_dump();

    // MAP report number of matched dataset
    map_push_report(toString(TotalMatch) + ",");

    // MAP report extractTime
    map_push_report(toString(extractTime) + ",");

    // MAP report search time usage
    map_push_report(toString(searchTime) + ",");

    ExportEvalRank(query_scaled_path);

    ExportRawRank(query_path);

    // Release memory
    bow_hist.clear();
}

void LoadSpecificBow(size_t dataset_id, vector<bow_bin_object>& load_hist)
{
    string in = run_param.database_root_dir + "/" + run_param.dataset_header + "/bow";

    ifstream InFile (in.c_str(), ios::binary);
    if (InFile)
    {
        /// Skip to bow of specific dataset_id
        size_t curr_offset = bow_offset[dataset_id];
        InFile.seekg(curr_offset, InFile.beg);

        /// Bow hist
        // Dataset ID (read, but not use)
        size_t dataset_id_read;
        InFile.read((char*)(&dataset_id_read), sizeof(dataset_id_read));

        // Dataset bow
        vector<bow_bin_object> read_bow;

        // Non-zero count
        size_t bin_count;
        InFile.read((char*)(&bin_count), sizeof(bin_count));

        // ClusterID and FeatureIDs
        for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
        {
            bow_bin_object read_bin;

            // Cluster ID
            InFile.read((char*)(&(read_bin.cluster_id)), sizeof(read_bin.cluster_id));

            // Frequency
            InFile.read((char*)(&(read_bin.freq)), sizeof(read_bin.freq));

            // Feature count
            size_t feature_count;
            InFile.read((char*)(&feature_count), sizeof(feature_count));
            for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
            {
                feature_object feature;

                // Feature ID
                InFile.read((char*)(&(feature.feature_id)), sizeof(feature.feature_id));
                // x
                InFile.read((char*)(&(feature.x)), sizeof(feature.x));
                // y
                InFile.read((char*)(&(feature.y)), sizeof(feature.y));
                // a
                InFile.read((char*)(&(feature.a)), sizeof(feature.a));
                // b
                InFile.read((char*)(&(feature.b)), sizeof(feature.b));
                // c
                InFile.read((char*)(&(feature.c)), sizeof(feature.c));

                read_bin.features.push_back(feature);
            }

            // Keep bow
            read_bow.push_back(read_bin);
        }

        // Keep hist
        load_hist.swap(read_bow);

        // Close file
        InFile.close();
    }
}

float Compute_map(const string& groundtruth_path)
{
    string ap_binary = "/home/stylix/webstylix/code/ox_ap/bin/Release/ox_ap";

    stringstream cmd;
    cmd << ap_binary << " " << groundtruth_path << " " << evalrank_path.str();

    cout << "map_cmd: " << cmd.str() << endl;

    return atof(exec(cmd.str()).c_str());
}

void Evaluate()
{
    /// Oxford building dataset
    if (str_contains(run_param.dataset_prefix, "oxbuildings") || str_contains(run_param.dataset_prefix, "paris") || str_contains(run_param.dataset_prefix, "smalltest"))
    {
        float sum_map = 0.0f;
        float avg_map = 0.0f;

        // Clear report
        map_report.str("");

        // MAP report configuration name
        map_push_report(run_param.dataset_prefix + "\r\n");

        // MAP report header
        map_push_report("query_id,query_name,query_source_image,mask,bin_size,num_kp,mask_pass,matched_dataset_size,extract_time,search_time,map\r\n");

        for (size_t q_id = 0; q_id < QueryNameLists.size(); q_id++)
        {
            // MAP report query id
            map_push_report(toString(q_id) + ",");

            // MAP report query name
            map_push_report(QueryNameLists[q_id] + ",");

            // Groundtruth path
            groundtruth_path.str("");
            groundtruth_path << run_param.dataset_root_dir << "/" << run_param.path_from_dataset << "/groundtruth/" << QueryNameLists[q_id];

            // Searching
            search_by_id(q_id);

            // Computing map
            float curr_ap = Compute_map(groundtruth_path.str());

            // MAP report map value, and close report for topic
            map_push_report(toString(curr_ap) + "\n");

            cout << "Query \"" << QueryNameLists[q_id] << "\" [" << q_id + 1 << "/" << QueryNameLists.size() << "] got map = " << setprecision(4) << redc << curr_ap << endc << endl;

            sum_map += curr_ap;
        }

        avg_map = sum_map / QueryImgLists.size();
        cout << "Configuration name \"" << run_param.dataset_prefix << "\"" << endl;
        cout << "Total map " << avg_map << endl;
    }
}

void FIX_RUN()
{
    int pcut_step = 5;
    for (int pcut = pcut_step; pcut < 100; pcut++)
    {

    }
}

void AddRank(const vector< pair<size_t, float> >& Rank)
{
    // Clone
    MultiResult.push_back(vector< pair<size_t, float> >(Rank));
}

void CombineRank()
{
    vector< pair<size_t, float> > NewResult;
    unordered_map<size_t, pair <size_t, float> > CombinedResult; // dataset_id, <count, score>

    // Initial
    vector< pair<float, float> > MultiResult_MinMax;
    vector<float> MultiResult_Length;
    if (run_param.multirank_mode == NORM_ADD_COMBINATION || run_param.multirank_mode == NORM_MEAN_COMBINATION)
    {
        float rank_min = 1000;
        float rank_max = -1000;
        for (size_t result_id = 0; result_id < MultiResult.size(); result_id++)
        {
            for (size_t rank_id = 0; rank_id < MultiResult[result_id].size(); rank_id++)
            {
                float curr_score = MultiResult[result_id][rank_id].second;
                if (rank_min > curr_score)
                    rank_min = curr_score;
                if (rank_max < curr_score)
                    rank_max = curr_score;
            }
            MultiResult_MinMax.push_back(pair<float, float>(rank_min, rank_max));
            MultiResult_Length.push_back(rank_max - rank_min);
        }
    }

    for (size_t result_id = 0; result_id < MultiResult.size(); result_id++)
    {
        for (size_t rank_id = 0; rank_id < MultiResult[result_id].size(); rank_id++)
        {
            size_t dataset_id = MultiResult[result_id][rank_id].first;
            // Initial
            if (CombinedResult.find(dataset_id) == CombinedResult.end())
            {
                CombinedResult[dataset_id].first = 0;     // dataset count
                CombinedResult[dataset_id].second = 0;    // dataset score
            }
            // Accumulate
            if (run_param.multirank_mode == ADD_COMBINATION || run_param.multirank_mode == MEAN_COMBINATION)
            {
                CombinedResult[dataset_id].first++;
                CombinedResult[dataset_id].second += MultiResult[result_id][rank_id].second;
            }
            else // NORM_ADD_COMBINATION, NORM_MEAN_COMBINATION
            {
                CombinedResult[dataset_id].first++;
                CombinedResult[dataset_id].second += ((MultiResult[result_id][rank_id].second - MultiResult_MinMax[result_id].first) / MultiResult_Length[result_id]);   // (xi - min) / (max - min)
            }
        }
    }

    // Combine and add to new result
    unordered_map<size_t, pair <size_t, float> >::iterator CombinedResult_it;
    for (CombinedResult_it = CombinedResult.begin(); CombinedResult_it != CombinedResult.end(); CombinedResult_it++)
    {
        if (run_param.multirank_mode == MEAN_COMBINATION || run_param.multirank_mode == NORM_MEAN_COMBINATION)
            CombinedResult_it->second.second /= CombinedResult_it->second.first;
        NewResult.push_back(pair<size_t, float>(CombinedResult_it->first, CombinedResult_it->second.second));
    }

    // Replace Result
    sort(NewResult.begin(), NewResult.end(), compare_pair_second<>());
    Result.swap(NewResult);
    NewResult.clear();

    // Clear Multi Result Space
    for (size_t result_id = 0; result_id < MultiResult.size(); result_id++)
        MultiResult[result_id].clear();
    MultiResult.clear();
}

// C++ Index
size_t search_by_id(int q_id)
{
	Result.clear();

	cout << "================ " << cyanc << q_id + 1 << endc << " " << yellowc << QueryNameLists[q_id] << endc << " ================" << endl;

    // Copy image to query session then create list.txt
    // Preparing path
    stringstream query_original_path; // copy from prefix
    stringstream query_search_path; // copy to session

    simulated_session = QueryNameLists[q_id];
    if (str_contains(run_param.dataset_prefix, "oxbuildings105k"))                                                              /// Oxford105k query system
        query_original_path << run_param.dataset_root_dir << "/" << "oxbuildings/5k" << "/" << QueryImgLists[q_id];             // [/home/stylix/webstylix/code/dataset]/[oxbuildings/5k]/[oxford_001753.jpg]
    else if (str_contains(run_param.dataset_prefix, "oxbuildings5k"))                                                           /// Oxford5k query system
        query_original_path << run_param.dataset_root_dir << "/" << run_param.path_from_dataset << "/" << QueryImgLists[q_id];  // [/home/stylix/webstylix/code/dataset]/[oxbuildings/5k]/[oxford_001753.jpg]
    else                                                                                                                        /// Paris6k query system
    {
        const char* delim = "_";// slash
        vector<string> query_image_split;
        string_splitter(QueryImgLists[q_id], delim, query_image_split);
        query_original_path << run_param.dataset_root_dir << "/" << run_param.path_from_dataset << "/" << query_image_split[1] << "/" << QueryImgLists[q_id];  // [/home/stylix/webstylix/code/dataset]/[paris/6k]/[invalides]/[paris_invalides_000360.jpg]
    }
    query_search_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << simulated_session;               // [/home/stylix/webstylix/code/ins_online/query]/[oxbuildings5k_sifthesaff-rgb-norm-root_akm_1000000_kd3_16_qscale_r80_mask_roi_qbootstrap2_18_f2_qbmining_5_report]/[balliol_5]/[oxford_001753.jpg]

    stringstream query_path;
    query_path << query_search_path.str() << "/" << QueryImgLists[q_id];

    // MAP report query source image
    map_push_report(QueryImgLists[q_id] + ",");

    // Copy image
    stringstream cmd;
    cmd << "cp " << query_original_path.str() << " " << query_path.str() << COUT2NULL;
    //cmd << "cp " << query_original_path.str() << " " << query_path.str();
    //cout << cmd.str() << endl;
    make_dir_available(query_search_path.str(), "777");
    exec(cmd.str());

    // Prepare mask
    if (run_param.mask_enable && MaskLists[q_id].size() > 0)
    {
        /*Point* verts = new Point[MaskLists.size()];
        for (size_t pt_id = 0; pt_id < MaskLists.size(); pt_id++)
        {
            verts[pt_id].x = (int)MaskLists[q_id][mask_id][pt_id].x;
            verts[pt_id].y = (int)MaskLists[q_id][mask_id][pt_id].y;
        }
        // Create Mask file
        draw_mask(query_path.str() + ".mask.png", get_image_size(query_path.str()), verts, MaskLists[q_id].size());

        delete[] verts;
        */

        ofstream MaskFile ((query_path.str() + ".mask").c_str());
        if (MaskFile.is_open())
        {
            // Write total mask
            MaskFile << MaskLists[q_id].size() << endl;

            // Multiple mask
            for (size_t mask_id = 0; mask_id < MaskLists[q_id].size(); mask_id++)
            {
                // Write mask size
                MaskFile << MaskLists[q_id][mask_id].size() << endl;

                // Write points
                for (size_t point_id = 0; point_id < MaskLists[q_id][mask_id].size(); point_id++)
                    MaskFile << MaskLists[q_id][mask_id][point_id].x << "," << MaskLists[q_id][mask_id][point_id].y << endl;
            }

            // Close file
            MaskFile.close();
        }

        // MAP report mask enable
        map_push_report("mask,");
    }
    else
    {
        // MAP report mask enable
        map_push_report("no_mask,");
    }

    // Query Bootstrapping v1
    if (run_param.query_bootstrap1_enable)
        QueryBootstrapping_v1(query_path.str());
    // Query Bootstrapping v2 (query expansion)
    else if (run_param.query_bootstrap2_enable)
        QueryBootstrapping_v2(query_path.str());
    // Scanning window
    else if (run_param.scanning_window_enable)
        ScanningQuery(query_path.str());
    // Normal search
    else
        DirectQuery(query_path.str());

    // Return first rank dataset_id
	return Result[0].first;
}

size_t search_by_bow_sig(const vector<bow_bin_object>& bow_sig)
{
	Result.clear();

    // Search method
    if (run_param.SIM_mode == SIM_GVP)
    {
        // search with gvp
        int* sim_param = new int[3];
        sim_param[0] = run_param.GVP_mode;
        sim_param[1] = run_param.GVP_size;
        sim_param[2] = run_param.GVP_length;
        TotalMatch = inverted_hist.search(bow_sig, Result, run_param.SIM_mode, sim_param);
        delete[] sim_param;
    }
    else  // search with normal l1
        TotalMatch = inverted_hist.search(bow_sig, Result);

	return Result[0].first;
}

void RandomHist()
{
	cout << "Random hist" << endl;
	randhist.clear();

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
		int bin_id;
		do
		{
			bin_id = (rand() % run_param.CLUSTER_SIZE);
		}
		while(randbin[bin_id]);
		randbin[bin_id] = true;
		float val = (rand() % 100000) / 1000000000.0;
		bow_bin_object bin_obj;
		bin_obj.cluster_id = (size_t)bin_id;
		bin_obj.freq = (float)val;

		int total_feature = rand() % 5 + 30;
		for (int feature_count = 0; feature_count < total_feature; feature_count++)
        {
            int ran_x = rand() % 10 + 100;
            int ran_y = rand() % 10 + 100;
            feature_object ran_fea;
            ran_fea.x = ran_x;
            ran_fea.y = ran_y;
            bin_obj.features.push_back(ran_fea);
        }

		// skip x y a b c
		randhist.push_back(bin_obj);
	}
}

void DisplayRank()
{
	int count = Result.size();
	int max = 5;
	if(max > count)
        max = count;
	for(int index = 0; index < max; index++)
		cout << "dataset_id:" << Result[index].first << fixed << " Value:" << Result[index].second << endl;
}

void ExportEvalRank(const string& query_path)
{
    int count = Result.size();

    evalrank_path.str("");
    evalrank_path << query_path << "_evalrank.txt";

    ofstream rank_File (evalrank_path.str().c_str());
    if(rank_File.is_open())
    {
        for(int index = 0; index < count; index++)
        {
            string dataset_name = str_replace_first(ImgLists[Result[index].first], ".jpg", "");
            rank_File << dataset_name << endl;
        }
        rank_File.close();
    }
}

void ExportRawRank(const string& query_path, float map)
{
    int count = Result.size();

    rawrank_path.str("");
    rawrank_path << query_path << "_rawrank.txt";

    ofstream rank_File (rawrank_path.str().c_str());
    if(rank_File.is_open())
    {
        rank_File << run_param.raw_param << endl;
        rank_File << run_param.detailed_param << endl;
        rank_File << query_path << endl;
        if (map != 0.0f)
            rank_File << map << endl;
        else
            rank_File << "-" << endl;
        for(int index = 0; index < count; index++)
        {
            stringstream dataset_path;
            dataset_path << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[Result[index].first]] << "/" << ImgLists[Result[index].first];
            rank_File << dataset_path.str() << "," << Result[index].second << endl;
        }
        rank_File.close();
    }

    text_write(get_directory(query_path) + "/rawrank_list.txt", rawrank_path.str() + "\n", true);
}

void map_push_report(const string& text)
{
    if (run_param.report_enable)
    {
        string map_report_path = run_param.query_root_dir + "/" + run_param.dataset_prefix + "/" + run_param.dataset_prefix + "_map_report.csv";

        map_report << text;
        map_report.flush();

        text_write(map_report_path, map_report.str(), false);
    }
}

void ExportRank(const string& CurrSessionID, ExportOption Opts)
{
	// dataset_ideo path ./videos/
	// Frame path ./frames/

	stringstream ResultPath;
	ResultPath << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << CurrSessionID << "/result.txt";
	ofstream exportResult (ResultPath.str().c_str());

	if (exportResult.is_open())
	{
		// Query header
		exportResult << "[Q]" << endl; // Session name
		exportResult << CurrSessionID << endl;
		exportResult << "[X]" << endl; // Metadata extension
		exportResult << TotalMatch << "|" << extractTime << "|" << searchTime << "|" << Opts.dev << "|" << Opts.ransac << "|" << Opts.showall << endl;
		exportResult << "[R]" << endl; // dataset_ideo Info
		size_t resultCount = ReRanked.size();
		if (resultCount > (size_t)Opts.max)
		resultCount = (size_t)Opts.max;
		for(size_t top = 0; top < resultCount; top++)//Retrieve top
		{
			exportResult << "[" << top << "]" << endl;
			// Information
			size_t ReRankIdx = ReRanked[top].first;
			// 0 dataset_id
			size_t dataset_id = Result[ReRankIdx].first;
			exportResult << dataset_id << "|";
			// 1 RankScore
			exportResult << Result[ReRankIdx].second << "|";
			// 2 ReRankedScore
			exportResult << ReRanked[top].second << "|";
			// 3 Index shifting
			int intReRankIdx = (int)ReRankIdx;
			int inttop = (int)top;
			exportResult << intReRankIdx - inttop << "|";
			if (is_commercial_film)
			{
                // 4 5 6 7 Time
                exportResult << ds_info.dataset_description()[dataset_id].year << "|" <<
                ds_info.dataset_description()[dataset_id].month << "|" <<
                ds_info.dataset_description()[dataset_id].day << "|" <<
                ds_info.dataset_description()[dataset_id].hour << "|";
                // 8 9 start-end clip
                exportResult << ds_info.dataset_description()[dataset_id].cStart << "|" << ds_info.dataset_description()[dataset_id].cEnd << "|";
                // 10 11 start-end shot
                exportResult << ds_info.dataset_description()[dataset_id].sStart << "|" << ds_info.dataset_description()[dataset_id].sEnd << "|";
                // 12 Channel name
                exportResult << ds_info.dataset_description()[dataset_id].channel << "|";
                // 13 dataset_video file
                stringstream dataset_source_path;
                dataset_source_path << "./videos/cap24-lv/" <<
                ds_info.dataset_description()[dataset_id].channel << "/" <<
                ds_info.dataset_description()[dataset_id].year << "/" <<
                ds_info.dataset_description()[dataset_id].year << "_" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].month, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].day, 2) << "_04_56/" <<
                ds_info.dataset_description()[dataset_id].year << "_" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].month, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].day, 2) << "_04_56-" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].hour, 2) << ".mpg";
                exportResult << dataset_source_path.str() << "|";
                // exportResult << "./videos/cap24-lv/tbs/2012/2012_05_15_04_56/2012_05_15_04_56-00.mpg" << "|";
                // FILE_START_TIME: 2012/01/01_09:56:08.001
                // 14 Encoded dataset_ideo file
                stringstream dataset_converted_path;
                // 2009_08_17_00_002532_0000tbs short frame start
                dataset_converted_path <<
                ds_info.dataset_description()[dataset_id].year << "_" <<
                ZeroPadNumber(ds_info.dataset_description()[dataset_id].month, 2) << "_" <<	ZeroPadNumber(ds_info.dataset_description()[dataset_id].day, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[dataset_id].hour, 2) << "_" <<
                ZeroPadNumber(ds_info.dataset_description()[dataset_id].cStart, 6) << "_" <<
                ZeroPadString(ds_info.dataset_description()[dataset_id].channel, 7);
                exportResult << "./encoded/" << dataset_converted_path.str() << ".mp4|"; // not include dataset_converted_path because wanna reuse it with commercial static page name
                // 15 Commercial info
                if(ds_info.dataset_description()[dataset_id].year <= 2010)// Select database
                    exportResult << "./infos/cf/statistics/" << ds_info.dataset_description()[dataset_id].cfid << ".html|";
                    //exportResult << "../../../users/wxmeng/cf/statistics/" << ds_info.dataset_description()[dataset_id].cfid << ".html|";
                else
                    exportResult << "./infos/cf.24/statistics/" << dataset_converted_path.str() << ".html|";
                    //exportResult << "../../../users/wxmeng/cf.24/statistics/" << dataset_converted_path.str() << ".html|";

                // 16 Freq
                exportResult << ds_info.dataset_description()[dataset_id].freq << "|";

                // 17 tag
                exportResult << "tag" << "|";

                string dirName = ds_info.dataset_description()[dataset_id].path;
                stringstream dataset_frames_path;
                dataset_frames_path << "./dataset/" << run_param.dataset_prefix << "/frames";
                // 18 Relative root + dir name
                exportResult << dataset_frames_path.str() << "/" << dirName << "|";

                // 19 Middle Images
                exportResult << ds_info.dataset_frames_filename()[dataset_id];
            }
            else
            {
                stringstream dataset_frames_path;
                dataset_frames_path << "./dataset/" << run_param.dataset_prefix << "/frames";
                // 4 Relative root + dir name
                exportResult << dataset_frames_path.str() << "|";
                // 5 Middle Images
                exportResult << ds_info.dataset_frames_filename()[dataset_id];
			}
            // Ending
            exportResult << endl;
		}
		exportResult.close();
	}
}

// For copy only the middle frames from original directory to target directory
void FrameCopy()
{
	string OriginalFramePath = run_param.database_root_dir + "/frames";
	string TargetFramePath = run_param.database_root_dir + "/frames_tmp";
	string Postfix32KLink = "/more";
	stringstream OriginalMiddleFilePath;
	stringstream TargetMiddleFilePath;
	stringstream TargetMiddleDirPath;
	stringstream cmd;
	int maxDir = 31980;

	if (dataset_size != ds_info.dataset_frames_dirname().size() || dataset_size != ds_info.dataset_frames_filename().size())
	{
		cout << "Critical!! dataset_size=" << dataset_size << " FrameDirSize=" << ds_info.dataset_frames_dirname().size() << " FrameFileSize=" << ds_info.dataset_frames_filename().size() << endl;
		exit(1);
	}

	for (size_t idx = 0; idx < dataset_size; idx++)
	{
		// Original file
		OriginalMiddleFilePath.str("");
		OriginalMiddleFilePath << OriginalFramePath << "/" << ds_info.dataset_frames_dirname()[idx] << "/" << ds_info.dataset_frames_filename()[idx];

		// Target file
		TargetMiddleDirPath.str("");
		TargetMiddleDirPath << TargetFramePath << "/" << ds_info.dataset_frames_dirname()[idx];
		TargetMiddleFilePath.str("");
		TargetMiddleFilePath << TargetFramePath << "/" << ds_info.dataset_frames_dirname()[idx] << "/" << ds_info.dataset_frames_filename()[idx];

		// Target file existance checking
		if(!is_path_exist(TargetMiddleFilePath.str())) // Target file does not exist
		{
			// Max directory limitation to 32K for Ext3 preparation
			cmd.str("");
			cmd << "ls -f " << TargetFramePath << " | wc -l";
			int maxDep = atoi(exec(cmd.str()).c_str());
			if(maxDep > maxDir)
			{
				// Root dir extracting
				const char* delimsSlash = "/";// slash
				vector<string> SlashSub;
				string_splitter(ds_info.dataset_frames_dirname()[idx], delimsSlash, SlashSub);
				string LinkName = SlashSub[0]; // Root dir

				// Making symbolink
				stringstream TargetDir;
				stringstream LinkDir;
				TargetDir << "." << Postfix32KLink << "/" << LinkName;
				LinkDir << TargetFramePath << "/" << LinkName;
				if(!is_path_exist(LinkDir.str())) // Link does not exist
				{
					cmd.str("");
					// ln -s target_dir link_name (relative target)
					cmd << "ln -s " << TargetDir.str() << " " << LinkDir.str();
					exec(cmd.str());
				}

				// Change Target file
				TargetMiddleDirPath.str("");
				TargetMiddleDirPath << TargetFramePath << Postfix32KLink << "/" << ds_info.dataset_frames_dirname()[idx];
				TargetMiddleFilePath.str("");
				TargetMiddleFilePath << TargetFramePath << Postfix32KLink << "/" << ds_info.dataset_frames_dirname()[idx] << "/" << ds_info.dataset_frames_filename()[idx];
			}

			// Directory preparation
			make_dir_available(TargetMiddleDirPath.str());

			// Original existance checking
			if(!is_path_exist(OriginalMiddleFilePath.str())) // Original file does not exist
			{
				// Copy from prev frame
				OriginalMiddleFilePath.str("");
				OriginalMiddleFilePath << OriginalFramePath << "/" << ds_info.dataset_frames_dirname()[idx - 1] << "/" << ds_info.dataset_frames_filename()[idx - 1];
			}

			// Frame copy
			cmd.str("");
			cmd << "cp " << OriginalMiddleFilePath.str() << " " << TargetMiddleFilePath.str();
			//cout << cmd.str() << "..";
			exec(cmd.str());
		}

		if(idx%500 == 0)
		{
			cout << "Copied.. \"" << ds_info.dataset_frames_filename()[idx] << "\" .. ";
			cout << 100.0 * idx / dataset_size << "%" << endl;
		}
	}
}

void RenderQueueGenerator(size_t start, size_t end)
{
	string QueuesPath = run_param.database_root_dir + "/encoder/Queues.txt";

	size_t total = end - start;
	cout << "Generating render queue from #" << start << " to #" << end << " [total = " << total << "]" << endl;
	ofstream QueuesFile(QueuesPath.c_str());
	if (QueuesFile.is_open())
	{
		for(size_t currIdx = start; currIdx != end; currIdx++)
		{
			int cStart = ds_info.dataset_description()[currIdx].cStart;
			int cEnd = ds_info.dataset_description()[currIdx].cEnd;
			stringstream vOrg;
			vOrg << "./videos/cap24-lv/" <<
				ds_info.dataset_description()[currIdx].channel << "/" <<
				ds_info.dataset_description()[currIdx].year << "/" <<
				ds_info.dataset_description()[currIdx].year << "_" << ZeroPadNumber(ds_info.dataset_description()[currIdx].month, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[currIdx].day, 2) << "_04_56/" <<
				ds_info.dataset_description()[currIdx].year << "_" << ZeroPadNumber(ds_info.dataset_description()[currIdx].month, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[currIdx].day, 2) << "_04_56-" << ZeroPadNumber(ds_info.dataset_description()[currIdx].hour, 2) << ".mpg";
			stringstream vEnc;
			vEnc << "./encoded/" <<
				ds_info.dataset_description()[currIdx].year << "_" <<
				ZeroPadNumber(ds_info.dataset_description()[currIdx].month, 2) << "_" <<	ZeroPadNumber(ds_info.dataset_description()[currIdx].day, 2) << "_" << ZeroPadNumber(ds_info.dataset_description()[currIdx].hour, 2) << "_" <<
				ZeroPadNumber(ds_info.dataset_description()[currIdx].cStart, 6) << "_" <<
				ZeroPadString(ds_info.dataset_description()[currIdx].channel, 7) << ".mp4";

			float fps = 29.97;

			//Start timecode
			int fStart = cStart;
			int sHr = (int)(fStart / (fps * 60 * 60));
			fStart %= (int)(fps * 60 * 60);
			int sMi = (int)(fStart / (fps * 60));
			fStart %= (int)(fps * 60);
			int sSe = (int)(fStart / (fps));
			fStart %= (int)fps;
			int sFr = fStart;

			stringstream ss;
			ss << sHr << ":" << sMi << ":" << sSe << "." << sFr;

			//Length timecode
			int fLength = cEnd - cStart;
			int tHr = (int)(fLength / (fps * 60 * 60));
			fLength %= (int)(fps * 60 * 60);
			int tMi = (int)(fLength / (fps * 60));
			fLength %= (int)(fps * 60);
			int tSe = (int)(fLength / (fps));
			fLength %= (int)fps;
			int tFr = fLength;

			stringstream t;
			t << tHr << ":" << tMi << ":" << tSe << "." << tFr;

			// pipe (|) is for embedding file name to command for encoder to know file name
			stringstream ConvertCmd;
			ConvertCmd << "ffmpeg -loglevel panic -ss " << ss.str() << " -t " << t.str() << " -i ." << vOrg.str() << " -n -vcodec libx264 -vpre ipod320 -b:v 512k -bt 50k -acodec libfaac -ab 96k -ac 2 ." << vEnc.str() << "|" << vEnc.str();

			QueuesFile << ConvertCmd.str() << endl;

			if((currIdx - start)%500 == 0)
			{
				cout << "Generated.. \"" << vEnc.str() << "\" .. ";
				cout << 100.0 * (currIdx - start) / total << "%" << endl;
			}
		}
		cout << "Done!.. please wait for RenderTaskManager process to finish..." << endl;

		QueuesFile.close();
	}

}

void UpdateStatus(const string& status_path, const string& status_text)
{
	ofstream StatusFile(status_path.c_str());
	if (StatusFile.is_open())
	{
		StatusFile << status_text;
		StatusFile.close();
	}
}
//;)
