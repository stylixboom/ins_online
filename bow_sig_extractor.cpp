/*
 * bow_sig_extractor.cpp
 *
 *  Created on: August 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */

#include "bow_sig_extractor.h"

using namespace std;
using namespace tr1;
using namespace cv;
using namespace ::flann;
using namespace alphautils;
using namespace ins;

int main(int argc,char *argv[])
{
    char menu;
	cout << "==== BOW Histogram online extractor ====" << endl;
    cout << "[l] Load preset dataset" << endl;
    cout << "[q] Quit" << endl;
    cout << "Enter menu:";cin >> menu;
	switch(menu)
    {
    case 'l':
        run_param.LoadPreset();
        break;
    case 'q':
        exit(0);
        break;
    }

	extractor_init();//Init extract_bow_sig

	cout << "Listening to query request.." << endl;


	stringstream hist_request_path;

	hist_request_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << "hist_request.txt";

	while (true)
	{
		// Waiting for hists
		while(!is_path_exist(hist_request_path.str()))
        {
            usleep(100);
            ls2null(hist_request_path.str());
        }

		//Load hist_request list
		vector<string> hist_request;
		ifstream hist_request_File (hist_request_path.str().c_str());
		if (hist_request_File)
		{
			while (hist_request_File.good())
			{
				string line;
				getline(hist_request_File, line);
				if (line != "")
				{
					char const* delimsSpace = " ";// space

					vector<string> SubQuery;

					string_splitter(line, delimsSpace, SubQuery);

                    // See ref push_query for query parameter
					hist_request.push_back(SubQuery[0]);
				}
			}
			hist_request_File.close();

			// After read!, Delete it!
			remove(hist_request_path.str().c_str());
		}

		// Extracting histogram and search
		vector<string>::iterator hist_request_it;
		for (hist_request_it = hist_request.begin(); hist_request_it != hist_request.end(); hist_request_it++)
		{
			stringstream session_path;
			stringstream hist_return_path;
			stringstream selected_query_id_return_path;
			session_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << *hist_request_it;
			hist_return_path << session_path.str() << "/" << "bow_hist.xct";
			selected_query_id_return_path << session_path.str() << "/" << "query.irep";

			vector<bow_bin_object> bow_hist;

			// Extract histogram
			int selected_query_id = 0;
			cout << "Extracting " << session_path.str() << "...";cout.flush();
			selected_query_id = extract_bow_sig(*hist_request_it, bow_hist); // session id
			cout << bow_hist.size() << " bin(s)"; cout.flush();
			cout << "OK!" << endl;

            if (bow_hist.empty())
                cout << "Too small image!!, cannot extract histogram" << endl;

			// Selected query id
			ofstream selected_query_id_return_File (selected_query_id_return_path.str().c_str());
			lockfile(selected_query_id_return_path.str());
			if (selected_query_id_return_File.is_open())
			{
				selected_query_id_return_File << selected_query_id;

				selected_query_id_return_File.close();
			}
			unlockfile(selected_query_id_return_path.str());

			// Write Histogram result
			export_hist(hist_return_path.str(), bow_hist);
		}
	}

	cout << "Closing service..." << endl;
	//extract_bow_sigDest();//Destroy extract_bow_sigor
}

void extractor_init()
{
    string invertedhist_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/invdata_" + run_param.dataset_header + "/invertedhist.def";
    string cluster_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/cluster";
    string search_index_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/searchindex";

    cout << "Extractor initializing.." << endl;

    // Load existing inverted header with idf
    cout << "Loading idf..";
    cout.flush();
    startTime = CurrentPreciseTime();
    load_idf(invertedhist_path);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Load existing cluster
    cout << "Loading cluster..";
    cout.flush();
    startTime = CurrentPreciseTime();
    //cout << "cluster_path: " << cluster_path << endl;
    load_cluster(cluster_path);
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Load existing FLANN search index
    cout << "Load FLANN search index..";
    cout.flush();
    startTime = CurrentPreciseTime();
    //cout << "search_index_path: " << search_index_path << endl;
    Index< ::flann::L2<float> > search_index(cluster, SavedIndexParams(search_index_path)); // load index with provided dataset
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Keep search index
    flann_search_index = search_index;
}

void load_idf(const string& in)
{
    if (is_path_exist(in))
	{
        // Load existing invert_index Def
        ifstream iv_header_File (in.c_str(), ios::binary);
        if (iv_header_File)
        {
            // Load dataset_size (but not use)
            size_t dataset_size;
            iv_header_File.read((char*)(&dataset_size), sizeof(dataset_size));

            // Load cluster_amount
            for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            {
                // Read cluster_amount (but not use)
                size_t read_cluster_amount;
                iv_header_File.read((char*)(&read_cluster_amount), sizeof(read_cluster_amount));
            }

            // Load idf
            for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            {
                // Read idf
                float read_idf;
                iv_header_File.read((char*)(&read_idf), sizeof(read_idf));
                idf.push_back(read_idf);
            }

            // Close file
            iv_header_File.close();
        }
	}
}

void load_cluster(const string& in)
{
    // Release memory
    delete[] cluster.ptr();

    size_t cluster_dimension;   // Feature dimension

    // Get HDF5 header
    HDF_get_2Ddimension(in, "clusters", cluster_size, cluster_dimension);

    // Wrap data to maxrix for flann knn searching
    float* empty_cluster = new float[cluster_size * cluster_dimension];

    // Read from HDF5
    HDF_read_2DFLOAT(in, "clusters", empty_cluster, cluster_size, cluster_dimension);

    Matrix<float> new_cluster(empty_cluster, cluster_size, cluster_dimension);

    // Keep cluster
    cluster = new_cluster;
}

size_t extract_bow_sig(const string& session_id, vector<bow_bin_object>& bow_sig)
{
    int repID = 0; // Selected query index
    int repGoodID = 0; // Selected index of good query

	stringstream query_path;
	stringstream query_list_path;

	query_path << run_param.query_root_dir << "/" << run_param.dataset_prefix << "/" << session_id;
	query_list_path << query_path.str() << "/" << "list.txt";

	// Read query list
	vector<string> masks;
	vector<string> queries;
	ifstream query_list_File (query_list_path.str().c_str());
	if (query_list_File)
	{
		while (query_list_File.good())
		{
			string line;
			getline(query_list_File, line);
			if (line != "")
			{
                // Add query
				queries.push_back(line);
				// Add mask (to be check)
				masks.push_back(line + ".mask");
            }
		}
		query_list_File.close();
	}
	cout << endl << "Got " << queries.size() << " queries." << endl;

	/// Multiple queries
	if(false && queries.size() > 1)
	{
        int thre = 30;

		vector< vector<int> > prev_assign;
		vector< vector<float> > prev_distance;
		vector< vector< vector<float> > > prev_sift_header;
		vector<int> good_query_index;
		vector< unordered_map<int, bool> > good_match;
		vector< vector<int> > good_assign;
		vector< vector<float> > good_distance;
		vector< vector< vector<float> > > good_sift_header;

        // Representative feature selection
		bool quit_early = false;
		for(size_t query_index = 0; query_index != queries.size() && !quit_early; query_index++)
		{
			cout << "Checking query " << query_index << ":" << get_filename(queries[query_index]) << endl;

			cout << "SIFT Hessian Affine extracting..";
            cout.flush();
            startTime = CurrentPreciseTime();
			// Sift Extraction
            SIFThesaff sifthesaff(run_param.colorspace, run_param.normpoint, run_param.rootsift);
            sifthesaff.extractPerdochSIFT(queries[query_index]);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Keep image width, height
            // Please pay attention on multiple width height
            curr_img_width = sifthesaff.width;
            curr_img_height = sifthesaff.height;

			cout << "Quantizing..";
			cout.flush();
            startTime = CurrentPreciseTime();
			// Quantization
            vector<int> currAssign;
            vector<float> currDistance;
            vector< vector<float> > currSiftHead;
            query_quantization(sifthesaff.desc, sifthesaff.kp, currAssign, currDistance);
            currSiftHead.swap(sifthesaff.kp);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

			cout << "Selecting.." << endl;
			// Find matching
			unordered_map<int, bool> bitMatch;
			vector<int> matchAssign;
			vector<float> matchDistance;
			vector< vector<float> > matchSiftHead;
			for(size_t prevImgIdx = 0; prevImgIdx < prev_assign.size(); prevImgIdx++)  // for all prev assign
			{
				for(size_t currAssIdx = 0; currAssIdx < currAssign.size(); currAssIdx++) // for all assign in curr image
				{
					for(size_t prevAssIdx = 0; prevAssIdx < prev_assign[prevImgIdx].size(); prevAssIdx++) // check with all assign of prev image
					{
                        int currAssignNum = currAssign[currAssIdx];
						if(currAssignNum == prev_assign[prevImgIdx][prevAssIdx] && !good_match[prevImgIdx][currAssign[currAssIdx]])
						{
							// Add match to current
							bitMatch[currAssignNum] = true; // set match to curr
							matchAssign.push_back(currAssignNum);
							matchDistance.push_back(currDistance[currAssIdx]);
							matchSiftHead.push_back(currSiftHead[currAssIdx]);

							// Update match to prev good
							good_match[prevImgIdx][currAssignNum] = true; // update match to prev
							good_assign[prevImgIdx].push_back(prev_assign[prevImgIdx][prevAssIdx]); // update assign to prev
							good_distance[prevImgIdx].push_back(prev_distance[prevImgIdx][prevAssIdx]); // update assign to prev
							good_sift_header[prevImgIdx].push_back(prev_sift_header[prevImgIdx][prevAssIdx]); // update assign to prev
							break;
						}
					}
				}
				cout << "\tcurrQ" << prev_assign.size() << " -> prevQ" << prevImgIdx << " found " << matchAssign.size() << "matches" << endl;
			}

			// Good enough of assign
			//if((int)matchAssign.size() > 0.6*min(currAssign.size(),prev_assign[prevImgIdx].size()))
			//	continue;
			if((int)matchAssign.size() > thre)
				quit_early = true;

			good_query_index.push_back(query_index); // Current query index
			good_match.push_back(bitMatch);
			good_assign.push_back(matchAssign);
			good_distance.push_back(matchDistance);
			good_sift_header.push_back(matchSiftHead);
			// Keep current bit match, assign, distance, sift head
			prev_assign.push_back(currAssign);
			prev_distance.push_back(currDistance);
			prev_sift_header.push_back(currSiftHead);

			// Freemem
			bitMatch.clear();
			matchAssign.clear();
			matchDistance.clear();
			matchSiftHead.clear();
			currAssign.clear();
			currDistance.clear();
			currSiftHead.clear();
		}

		size_t repFeatSize = 0;
		for(size_t idx = 0; idx < good_assign.size(); idx++)
		{
			sort(good_assign[idx].begin(), good_assign[idx].end());
			good_assign[idx].resize(distance(good_assign[idx].begin(), unique(good_assign[idx].begin(), good_assign[idx].end())));
			cout << "#" << idx << " has " << good_assign[idx].size() << " features" << endl;
			if(good_assign[idx].size() > repFeatSize) // Find max matches
			{
				repGoodID = (int)idx;
				repFeatSize = good_assign[repGoodID].size();
				repID = good_query_index[repGoodID];
			}
		}
		cout << endl << "Selected query id : " << repID << endl;
		cout << "Query image: " << queries[repID] << endl;
		cout << "Matched " << good_assign[repGoodID].size() << " from " << prev_assign[repID].size() << " features" << endl;
		cout.flush();

        cout << "Bow..";
        cout.flush();
        startTime = CurrentPreciseTime();
        // Bow
		Bow(good_assign[repGoodID], good_sift_header[repGoodID], bow_sig);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

		// Export affine region
		stringstream good_sift_headerRetPath;
		good_sift_headerRetPath << queries[repID] << ".frep";
		ofstream good_sift_headerRetFile (good_sift_headerRetPath.str().c_str());
		if (good_sift_headerRetFile.is_open())
		{
			// Write dimension = 1 (no descriptor)
			good_sift_headerRetFile << "1.0" << endl;

			// Write sift amount
			good_sift_headerRetFile << repFeatSize << endl;

			// Write sift head
			for(size_t idx = 0; idx < repFeatSize; idx += 5)
			{
				good_sift_headerRetFile << good_sift_header[repGoodID][idx][0] << " ";	// u
				good_sift_headerRetFile << good_sift_header[repGoodID][idx][1] << " ";	// v
				good_sift_headerRetFile << good_sift_header[repGoodID][idx][2] << " ";	// a
				good_sift_headerRetFile << good_sift_header[repGoodID][idx][3] << " ";	// b
				good_sift_headerRetFile << good_sift_header[repGoodID][idx][4] << endl;	// c
			}

			good_sift_headerRetFile.close();
		}

		// Draw feature
		stringstream goodSiftOverlayPath;
		goodSiftOverlayPath << queries[repID] << ".frep.jpg";
		Mat goodSiftOverlayImg = imread(queries[repID], CV_LOAD_IMAGE_COLOR);
		// Write sift point
		for(size_t idx = 0; idx < repFeatSize * 5; idx += 5)
			circle(goodSiftOverlayImg, Point2f((float)good_sift_header[repGoodID][idx][0], (float)good_sift_header[repGoodID][idx][1]), 4, Scalar(0, 255, 0), 2, CV_AA, 0);
        vector<int> imwrite_param;
        imwrite_param.push_back(CV_IMWRITE_JPEG_QUALITY);
        imwrite_param.push_back(85);
		imwrite(goodSiftOverlayPath.str().c_str(), goodSiftOverlayImg);

		// Freemem
		good_query_index.clear();
		for(size_t idx =0; idx < prev_assign.size(); idx++)
		{
			prev_assign[idx].clear();
			prev_distance[idx].clear();
			prev_sift_header[idx].clear();
		}
		prev_assign.clear();
		prev_distance.clear();
		prev_sift_header.clear();
		for(size_t idx =0; idx < good_assign.size(); idx++)
		{
			good_assign[idx].clear();
			good_distance[idx].clear();
			good_sift_header[idx].clear();
		}
		good_assign.clear();
		good_distance.clear();
		good_sift_header.clear();
	}
	else /// One query
	{
        cout << "Query: " << get_filename(queries[0]) << endl;
        /// Read Mask as a polygon
        // Load mask
        ifstream mask_File (masks[0].c_str());
        if (mask_File)
        {
            cout << "Reading mask..";
            cout.flush();
            startTime = CurrentPreciseTime();

            string line;
            getline(mask_File, line);

            // Read mask count
            mask_count = strtoull(line.c_str(), NULL, 0);

            for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
            {
                getline(mask_File, line);

                // Read vertex count
                mask_vertex_count = strtoull(line.c_str(), NULL, 0);

                // Preparing mask array
                float* curr_mask_vertex_x = new float[mask_vertex_count];
                float* curr_mask_vertex_y = new float[mask_vertex_count];

                // Read mask vertecies
                for (size_t mask_idx = 0; mask_idx < mask_vertex_count; mask_idx++)
                {
                    getline(mask_File, line);

                    char const* delims = ",";
                    vector<string> vertex;
                    string_splitter(line, delims, vertex);

                    curr_mask_vertex_x[mask_idx] = atof(vertex[0].c_str());
                    curr_mask_vertex_y[mask_idx] = atof(vertex[1].c_str());
                }

                // Keep multiple masks
                mask_vertex_x.push_back(curr_mask_vertex_x);
                mask_vertex_y.push_back(curr_mask_vertex_y);
            }

            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            mask_File.close();
        }

        cout << "SIFT Hessian Affine extracting..";
        cout.flush();
        startTime = CurrentPreciseTime();
        // Sift Extraction
        num_kp = 0;
        SIFThesaff sifthesaff(run_param.colorspace, run_param.normpoint, run_param.rootsift);
        sifthesaff.extractPerdochSIFT(queries[0]);
        num_kp = sifthesaff.num_kp;
        cout << num_kp << " keypoint(s).."; cout.flush();
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Skip if no keypoint detected
        if (num_kp == 0)
            return -1;

        // Keep image width, height
        curr_img_width = sifthesaff.width;
        curr_img_height = sifthesaff.height;

        cout << "Quantizing..";
        cout.flush();
        startTime = CurrentPreciseTime();
        // Quantization
        vector<int> currAssign;
        vector<float> currDistance;
        vector< vector<float> > currSiftHead;
        query_quantization(sifthesaff.desc, sifthesaff.kp, currAssign, currDistance);
        currSiftHead.swap(sifthesaff.kp);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        cout << "Building BOW..";
        cout.flush();
        startTime = CurrentPreciseTime();
        // Bow
		Bow(currAssign, currSiftHead, bow_sig);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Reset mask memory
        if (mask_count > 0)
        {
            mask_count = 0;
            mask_vertex_count = 0;
            for (size_t mask_id = 0; mask_id < mask_vertex_x.size(); mask_id++)
            {
                delete[] mask_vertex_x[mask_id];
                delete[] mask_vertex_y[mask_id];
            }
            mask_vertex_x.clear();
            mask_vertex_y.clear();
        }
    }

	// const char* query_dirname = SEARCHROOT "/queries/2012-22-10-18-42-09-93-7686db3bfea96259df596fcc28d8ae3c";
    // const char* temp_dirname = "/tmpfs/tmp";
    // const char* sift_extractor = "/home/caizhizhu/caizhizhu/cf/code/compute_descriptors_64bit.ln";
    // float* bow_sig = get_query_signature(query_dirname, temp_dirname, pQinfo, sift_extractor, true);

	return repID;
}

void query_quantization(const vector< vector<float> >& desc, const vector< vector<float> >& kp, vector<int>& query_quantized_index, vector<float>& query_quantized_dist)
{
    string search_index_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/searchindex";

    //size_t num_kp = desc.size();
    size_t dimension = desc[0].size();

    // KNN search
    SearchParams sparams = SearchParams();
    sparams.checks = 512;
    sparams.cores = run_param.MAXCPU;
    size_t knn = 1;

    // Prepare feature vector to be quantized
    float* current_feature = new float[num_kp * dimension];
    for (int row = 0; row < num_kp; row++)
    {
        for (size_t col = 0; col < dimension; col++)
        {
            current_feature[row * dimension + col] = desc[row][col];
        }
    }

    // KNN Search
    Matrix<float> feature_data(current_feature, num_kp, dimension);
    Matrix<int> result_index(new int[num_kp * knn], num_kp, knn); // size = feature_amount x knn
    Matrix<float> result_dist(new float[num_kp * knn], num_kp, knn);

    flann_search_index.knnSearch(feature_data, result_index, result_dist, knn, sparams);

    // Keep result
    int* result_index_idx = result_index.ptr();
    float* result_dist_idx = result_dist.ptr();
    // row base result
    // col is nn number
    for(size_t col = 0; col < knn; col++)
    {
        for(size_t row = 0; row < result_index.rows; row++)
        {
            query_quantized_index.push_back(result_index_idx[row * knn + col]);
            query_quantized_dist.push_back(result_dist_idx[row * knn + col]);
        }
    }

    // Release memory
    delete[] feature_data.ptr();
    delete[] result_index.ptr();
    delete[] result_dist.ptr();
}

void Bow(const vector<int>& query_quantized_indices, const vector< vector<float> >& query_keypoints, vector<bow_bin_object>& bow_sig)
{
    // Checking dataset keypoint avalibality
    if (query_keypoints.size() == 0)
    {
        cout << "No dataset keypoint available" << endl;
        return;
    }
    // Checking quantized dataset avalibality
    if (query_quantized_indices.size() == 0)
    {
        cout << "No quantized dataset available" << endl;
        return;
    }

    /// Bow gen

    // Initialize blank sparse bow
    unordered_map<size_t, vector<feature_object> > curr_sparse_bow; // cluster_id, features
    mask_pass = 0;
    // Set bow
    // Add feature to curr_sparse_bow at cluster_id
    // Frequency of bow is curr_sparse_bow[].size()
    for (size_t feature_id = 0; feature_id < query_quantized_indices.size(); feature_id++)
    {
        // Get cluster from quantizad index of feature
        size_t cluster_id = query_quantized_indices[feature_id];

        // Create new feature object with feature_id and geo location, x,y,a,b,c
        feature_object feature;
        feature.feature_id = feature_id;
        feature.x = query_keypoints[feature_id][0];
        feature.y = query_keypoints[feature_id][1];
        feature.a = query_keypoints[feature_id][2];
        feature.b = query_keypoints[feature_id][3];
        feature.c = query_keypoints[feature_id][4];

        // Feature weighting
        if (mask_count > 0)
        {
            // Test if feature locates inside the mask
            //if ((int)mask_img.at<uchar>(Point ((int)(feature.x * curr_img_width), (int)(feature.y * curr_img_height))) > 0)

            // Hit test with multiple masks
            bool point_hit = false;
            for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
            {
                if (run_param.normpoint)
                    point_hit |= pnpoly(mask_vertex_count, mask_vertex_x[mask_id], mask_vertex_y[mask_id], feature.x * curr_img_width, feature.y * curr_img_height);
                else
                    point_hit |= pnpoly(mask_vertex_count, mask_vertex_x[mask_id], mask_vertex_y[mask_id], feature.x, feature.y);
                if (point_hit)
                    break;
            }

            if (point_hit)
            {
                feature.weight = 1.0f;
                mask_pass++;
                //cout << feature.x * curr_img_width << ", " << feature.y * curr_img_height << " passed!" << endl;
            }
            else
            {
                feature.weight = 0.0f;
                // if we don't want to use multiple weight for mask, please skip this feature
                continue;
                //cout << feature.x * curr_img_width << ", " << feature.y * curr_img_height << " failed!" << endl;
            }
        }
        else // no mask
        {
            feature.weight = 1.0f;
            mask_pass++;
        }

        // Keep new feature into its corresponding bin (cluster_id)
        curr_sparse_bow[cluster_id].push_back(feature);
    }
    cout << "Total points: " << query_quantized_indices.size() << " Passed: " << mask_pass << ".."; cout.flush();

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
            feature_weight = (1 + log10(feature_weight)) * idf[bow_bin.cluster_id]; // tf*idf = log10(feature_weight) * idf[cluster_id]
        else
            continue;   // Skip adding to bow

        bow_bin.freq = feature_weight;
        bow_bin.features.swap(sparse_bow_it->second);

        // Keep new bin into compact_bow
        bow_sig.push_back(bow_bin);
    }

    /// Normalization
    // Unit length
    float sum_of_square = 0.0f;
    float unit_length = 0.0f;
    for (size_t bin_idx = 0; bin_idx < bow_sig.size(); bin_idx++)
        sum_of_square += bow_sig[bin_idx].freq * bow_sig[bin_idx].freq;
    unit_length = sqrt(sum_of_square);

    // Normalizing
    for (size_t bin_idx = 0; bin_idx < bow_sig.size(); bin_idx++)
        bow_sig[bin_idx].freq = bow_sig[bin_idx].freq / unit_length;
}

bool export_hist(const string& out, const vector<bow_bin_object>& bow_hist)
{
    bool ret = true;
    ofstream bow_hist_File (out.c_str(), ios::binary);
    ret &= lockfile(out);
    if (ret &= bow_hist_File.is_open())
    {
        // Write non-zero count
        size_t bin_count = bow_hist.size();
        bow_hist_File.write(reinterpret_cast<char*>(&bin_count), sizeof(bin_count));
        //cout << bin_count << endl;

        for (size_t bin_id = 0; bin_id < bin_count; bin_id++)
        {
            // Write cluster_id
            size_t cluster_id = bow_hist[bin_id].cluster_id;
            bow_hist_File.write(reinterpret_cast<char*>(&cluster_id), sizeof(cluster_id));
            //cout << cluster_id << endl;

            // Write freq
            float freq = bow_hist[bin_id].freq;
            bow_hist_File.write(reinterpret_cast<char*>(&freq), sizeof(freq));
            //cout << freq << endl;

            // Write feature_count
            size_t feature_count = bow_hist[bin_id].features.size();
            bow_hist_File.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
            //cout << feature_count << endl;

            for (size_t feature_id = 0; feature_id < feature_count; feature_id++)
            {
                // write feature (x,y,a,b,c)
                feature_object curr_feature = bow_hist[bin_id].features[feature_id];
                /*float x = curr_feature.x;
                float y = curr_feature.y;
                float a = curr_feature.a;
                float b = curr_feature.b;
                float c = curr_feature.c;
                cout << x << " " << y << " " << a << " " << b << " " << c << endl;*/
                bow_hist_File.write(reinterpret_cast<char*>(&(curr_feature.x)), sizeof(curr_feature.x));
                bow_hist_File.write(reinterpret_cast<char*>(&(curr_feature.y)), sizeof(curr_feature.y));
                bow_hist_File.write(reinterpret_cast<char*>(&(curr_feature.a)), sizeof(curr_feature.a));
                bow_hist_File.write(reinterpret_cast<char*>(&(curr_feature.b)), sizeof(curr_feature.b));
                bow_hist_File.write(reinterpret_cast<char*>(&(curr_feature.c)), sizeof(curr_feature.c));
            }

        }

        // Write num_kp
        bow_hist_File.write(reinterpret_cast<char*>(&num_kp), sizeof(num_kp));

        // Write mask_pass
        bow_hist_File.write(reinterpret_cast<char*>(&mask_pass), sizeof(mask_pass));

        // Close file
        bow_hist_File.close();
    }

    ret &= unlockfile(out);

    return ret;
}
//;)
