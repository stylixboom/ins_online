/*
 * bow_sig_extractor.cpp
 *
 *  Created on: August 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <unistd.h>         // usleep
#include <unordered_map>
#include <memory>           // unique_ptr
#include <omp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <flann/flann.hpp>

// Siriwat's header
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/imtools.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"
#include "../lib/ins/bow.h"
#include "../lib/ins/quantizer.h"

// Merlin's header
#include "../lib/ins/core.hpp"      // Mask
#include "../lib/ins/utils.hpp"
#include "../lib/ins/compat.hpp"

#include "bow_sig_extractor.h"

using namespace std;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::imtools;
using namespace alphautils::hdf5io;
using namespace ins;
using namespace cv;


using namespace std;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::imtools;
using namespace alphautils::hdf5io;
using namespace ins;

using namespace ins::utils;
using namespace ins::compat;

using namespace cv;

int main(int argc,char *argv[])
{
    visualize_enable = false;

    char menu;
    do
    {
        cout << "==== BOW Histogram online extractor ====" << endl;
        cout << "[l] Load preset dataset" << endl;
        cout << "[d] Turn on draw feature and mask" << endl;
        cout << "[q] Quit" << endl;
        cout << "Enter menu:";cin >> menu;
        switch(menu)
        {
        case 'l':
            run_param.LoadPreset();
            run_listening();
            break;
        case 'd':
            visualize_enable = true;
            break;
        case 'q':
            exit(0);
            break;
        }
    }
    while (menu != 'q');
}

void run_listening()
{
    // Initial environment
    extractor_init();

    cout << "Listening to query request.." << endl;
    //cout << run_param.histrequest_path << endl;

    // Query mode description
    string multi_query_string;
    if (run_param.earlyfusion_enable)
    {
        multi_query_string += "Early fusion, ";
        if (run_param.earlyfusion_mode == EARLYFUSION_SUM)
            multi_query_string += "sum";
        else if (run_param.earlyfusion_mode == EARLYFUSION_MAX)
            multi_query_string += "max";
        else if (run_param.earlyfusion_mode == EARLYFUSION_AVG)
            multi_query_string += "avg";
        else if (run_param.earlyfusion_mode == EARLYFUSION_FIM)
            multi_query_string += "fim";
    }
    else if (run_param.latefusion_enable)
    {
        multi_query_string += "Late fusion, ";
        if (run_param.latefusion_mode == LATEFUSION_SUM)
            multi_query_string += "sum";
        else if (run_param.latefusion_mode == LATEFUSION_MAX)
            multi_query_string += "max";
        else if (run_param.latefusion_mode == LATEFUSION_AVG)
            multi_query_string += "avg";
    }

    // Listening request
    vector<string> hist_request;
    while (listening_histrequest(hist_request))
    {
        // Extracting histogram and search
        size_t hist_request_size = hist_request.size();
        for (size_t hist_request_idx = 0; hist_request_idx < hist_request_size; hist_request_idx++)
        {
            cout << bluec << "================ " << endc << redc << "Feature Extraction" << endc << bluec << " ================" << endc << endl;
            cout << "Multiple query mode: " << redc << multi_query_string << endc << endl;

            stringstream session_path;
            stringstream selected_query_id_return_path;
            session_path << run_param.online_working_path << "/" << hist_request[hist_request_idx];
            selected_query_id_return_path << session_path.str() << "/" << "query.irep";

            // Extract histogram
            int selected_query_id = 0;
            cout << "Extracting " << "[" << hist_request_idx + 1 << "/" << hist_request_size << "] " << session_path.str() << "...";cout.flush();
            size_t bin_count = extract_bowsig(hist_request[hist_request_idx]); // session id
            cout << "OK!" << endl;

            if (!bin_count)
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
        }

        cout << "Listening to query request.." << endl;
    }

    cout << "Closing service..." << endl;
}

void extractor_init()
{
    cout << "Extractor initializing.." << endl;
    ann.init(run_param);
    inverted_file.init(run_param, true);    // <-- resume 'true' is to load header including idf
    idf = inverted_file.get_idf();
    cout << "done!" << endl;
}

bool listening_histrequest(vector<string>& hist_request)
{
    // Waiting for a request
    while (!is_path_exist(run_param.histrequest_path) || islock(run_param.histrequest_path))
        usleep(5000);   // 5 milliseconds

    // Reading request
    vector<string> read_hist_request;
    ifstream hist_request_File (run_param.histrequest_path.c_str());
    if (hist_request_File)
    {
        while (hist_request_File.good())
        {
            string line;
            getline(hist_request_File, line);
            if (line != "")
            {
                // Check quit signal
                if (line == SIGQUIT)
                {
                    // Remove after read SIGQUIT command
                    remove(run_param.histrequest_path.c_str());
                    return false;
                }

                vector<string> SubQuery;

                StringExplode(line, " ", SubQuery);

                // See ref push_query for query parameter
                read_hist_request.push_back(SubQuery[0]);
            }
        }
        hist_request_File.close();

        // After read!, Delete it!, and take care all of this query stack
        remove(run_param.histrequest_path.c_str());
    }
    else
    {
        cout << "Error occur in hist_request file" << endl;
        exit(-1);
    }

    hist_request.swap(read_hist_request);

    // Continue running if true
    return true;
}

void read_querylist(const string& session_id, vector<string>& queries, vector<unique_ptr<Mask> >& masks, vector<string>& mask_paths)
{
	ostringstream oss;

	oss << run_param.online_working_path << "/" << session_id << "/" << run_param.querylist_filename;

    Properties<string> queryList(Properties<string>::load(oss.str()));
    size_t numQueries = strtoull(queryList.get("query.size").c_str(), NULL, 0);
	cout << endl << "Got " << numQueries << " queries." << endl;

    for (size_t i = 0; i < numQueries; i++)
    {
        oss.str("");
        oss << "query[" << i << "]";

        string imagePath = queryList.get(oss.str() + ".image");
        queries.push_back(imagePath);

        cout << "Query: " << get_filename(imagePath) << endl;

        // Read mask
        cout << "Reading mask.." << endl;
        startTime = CurrentPreciseTime();
        if (queryList.contains(oss.str() + ".mask")) {
            string maskType = queryList.get(oss.str() + ".mask_type");
            string maskPath = queryList.get(oss.str() + ".mask");
            mask_paths.push_back(maskPath);
            if (maskType == "IMAGE")
                masks.push_back(std::unique_ptr<Mask>(
                        new ImageMask(ImageMask::load(maskPath))));
            else if (maskType == "POLYGON")
                masks.push_back(std::unique_ptr<Mask>(
                        new PolygonMask(PolygonMask::load(maskPath))));
            else
                throw runtime_error("Unknown mask type: " + maskType);
        } else {
            masks.push_back(std::unique_ptr<Mask>(new NoMask()));
        }
    }
}

size_t extract_bowsig(const string& session_id)
{
    // Reset environment
    total_kp = 0;
    total_mask_pass = 0;
    total_bin = 0;

    //int repID = 0; // Selected query index
    //int repGoodID = 0; // Selected index of good query

	// Read query list
	vector<string> queries;
	vector< unique_ptr<Mask> > masks;
	vector<string> mask_paths;
	read_querylist(session_id, queries, masks, mask_paths);

	/// Representative feature selection
	if(false)
	{
	    /*
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
		*/
	}
	/// Normal feature extraction
	else
	{
	    // SIFT memory
	    size_t query_size = queries.size();
	    vector< vector<float*> > extracted_kp(query_size);
	    vector< vector<float*> > extracted_desc(query_size);
	    vector<Size> extracted_imgsize(query_size);
	    cout << "SIFT Hessian Affine extracting.." << endl;
	    timespec totalextractTime = CurrentPreciseTime();
	    /// Parallel extracting
        #pragma omp parallel shared(total_kp,queries,run_param,query_size,extracted_kp,extracted_desc,extracted_imgsize)
        {
            /// Feature extraction
            #pragma omp for schedule(dynamic,1) reduction(+ : total_kp)
            for (size_t img_idx = 0; img_idx < query_size; img_idx++)
            {
                stringstream out_txt;
                out_txt << img_idx << "/" << query_size << " " << greenc << get_filename(queries[img_idx]) << endc << " ";
                SIFThesaff sift_extractor;
                sift_extractor.init(run_param.colorspace, run_param.normpoint, run_param.rootsift);
                timespec extractTime = CurrentPreciseTime();
                // Sift Extraction
                int num_kp = sift_extractor.extractPerdochSIFT(queries[img_idx]);
                total_kp += num_kp;
                out_txt << num_kp << " keypoint(s)..";
                out_txt << "done! (in " << setprecision(2) << fixed << TimeElapse(extractTime) << " s)" << endl;

                // Print Info
                cout << out_txt.str();

                // Keep features
                extracted_kp[img_idx].swap(sift_extractor.kp);
                extracted_desc[img_idx].swap(sift_extractor.desc);
                extracted_imgsize[img_idx] = Size(sift_extractor.width, sift_extractor.height);

                sift_extractor.unlink_kp(); // This will tell sift extractor not to delete kp internally
                sift_extractor.unlink_desc(); // This will tell sift extractor not to delete desc internally

                // Release memory
                sift_extractor.reset();
            }
        }
        cout << "SIFT extracting done! (in " << setprecision(2) << fixed << TimeElapse(totalextractTime) << " s)" << endl;

	    /// Normal extraction and early fusion
	    if (!run_param.latefusion_enable)
        {
            /// Vector quantization
            bow bow_builder;
            bow_builder.init(run_param);
            for (size_t img_idx = 0; img_idx < query_size; img_idx++)
            {
                size_t num_kp = extracted_kp[img_idx].size();
                // Skip if no keypoint detected
                if (!num_kp)
                    continue;   // Skip to next query

                cout << "Quantizing.."; cout.flush();
                startTime = CurrentPreciseTime();
                // Packing desc data
                int sift_len = SIFThesaff::GetSIFTD();
                float* kp_dat = new float[num_kp * sift_len];
                for (size_t kp_idx = 0; kp_idx < num_kp; kp_idx++)
                    for (int desc_idx = 0; desc_idx < sift_len; desc_idx++)
                        kp_dat[kp_idx * sift_len + desc_idx] = extracted_desc[img_idx][kp_idx][desc_idx];
                Matrix<float> data(kp_dat, num_kp, sift_len);  // size = num_kp x sift_len
                // Prepare result space
                Matrix<int> result_index;                           // size = num_kp x knn
                Matrix<float> result_dist;                          // size = num_kp x knn
                // Quantization
                ann.quantize(data, num_kp, result_index, result_dist);
                cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

                /// BOW building
                cout << "Building BOW.."; cout.flush();
                startTime = CurrentPreciseTime();
                bow_builder.build_bow(result_index.ptr(), extracted_kp[img_idx], 0);    // All frame will be in the same image_id 0, but different sequence_id (multi_bow.size())
                cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

                /// Visualize full keypoints
                if (visualize_enable)
                    visualize_bow(queries[img_idx], queries[img_idx] + ".display.png", bow_builder.get_last_bow(), false);

                /// Masking
                if (run_param.mask_enable)
                {
                    int mask_pass = 0;
                    cout << "Masking.."; cout.flush();
                    startTime = CurrentPreciseTime();
                    if (run_param.normpoint)
                        mask_pass = bow_builder.masking_lastbow(masks[img_idx], extracted_imgsize[img_idx].width, extracted_imgsize[img_idx].height);
                    else
                        mask_pass = bow_builder.masking_lastbow(masks[img_idx]);
                    cout << "[" << yellowc << mask_pass << endc << "/" << bluec << num_kp << endc << "] ";
                    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
                    total_mask_pass += mask_pass;

                    /// Visualize masked keypoints
                    if (visualize_enable)
                    {
                        // Polygon mask overlay
                        if (check_extension(mask_paths[img_idx], "mask"))
                        {
                            vector< vector<Point2f> > polygons;
                            oxMaskImport(mask_paths[img_idx], polygons);
                            overlay_mask(queries[img_idx], queries[img_idx] + ".mask.png", polygons, run_param.normpoint);
                        }
                        // Image mask overlay
                        else
                            overlay_mask(queries[img_idx], queries[img_idx] + ".mask.png", mask_paths[img_idx]);

                        visualize_bow(queries[img_idx] + ".mask.png", queries[img_idx] + ".mask.display.png", bow_builder.get_last_bow(), true);
                    }
                }

                /// Root bow_sig
                if (run_param.powerlaw_enable)
                    bow_builder.rooting_lastbow();

                /// Release memory
                delete[] kp_dat;
                delete[] result_index.ptr();
                delete[] result_dist.ptr();
            }
            /// Finalizing bow_sig, Internally TF-Normalized
            vector<bow_bin_object*>& bow_sig = bow_builder.finalize_bow();   // Have new memory of bow_bin_object and reuse memory of feature_object, and kp

            /// TF-IDF. Normalize
            bow_builder.logtf_idf_unitnormalize(bow_sig, idf);

            /// Write Histogram result
            // dirname/dirname.bowsig
            if (run_param.earlyfusion_enable)
                export_bowsig(get_directory(queries[0]) + "/" + get_filename(get_directory(queries[0])) + ".earlyfused" + run_param.hist_postfix, bow_sig, total_kp, total_mask_pass);
            // dirname/dirname.earlyfused.bowsig
            else
                export_bowsig(queries[0] + run_param.hist_postfix, bow_sig, total_kp, total_mask_pass);

            cout << "Total:" << endl;
            cout << redc << bow_sig.size() << endc << " bin(s)" << endl;
            cout << yellowc << total_mask_pass << endc << "/" << bluec << total_kp << endc  << " keypoint(s)" << endl;

            total_bin = bow_sig.size();

            /// Dump
            if (run_param.matching_dump_enable)
            {
                // Collect
                size_t dataset_id = 0;  // Early fusion, query number always 0
                for (size_t bin_idx = 0; bin_idx < bow_sig.size(); bin_idx++)
                {
                    bow_bin_object* bin = bow_sig[bin_idx];
                    for (size_t feature_idx = 0; feature_idx < bin->features.size(); feature_idx++)
                        dumper.collect_kp(dataset_id, bin->cluster_id, bin->weight, bin->fg, bin->features[feature_idx]->sequence_id, bin->features[feature_idx]->kp);
                        // sequence_id already assigned by bow_pooling
                }

                // Dump
                dumper.dump(queries[0] + ".query.dump", // Choose query_id 0 as dump name
                            vector<size_t>{dataset_id},
                            vector<string>{get_directory(queries[0])},
                            vector< vector<string> >{vector<string>{get_filename(queries[0])}});
            }

            /// Release memory
            bow_builder.reset_bow();        // Delete previously allocated memory (bow_bin_object, feature_object, and kp)
            if (run_param.earlyfusion_enable)       // delete pool if pool was build
                bow_builder.reset_bow_pool();
        }
        /// Late fusion
        else
        {
            //int current_num_thread = omp_get_num_threads();
            //omp_set_dynamic(0);
            //omp_set_num_threads(4);
            //#pragma omp parallel shared(total_bin,total_mask_pass,queries,run_param,query_size,extracted_kp,extracted_desc,extracted_imgsize)
            {
                /// Feature extraction
                //#pragma omp for reduction(+ : total_bin,total_mask_pass)
                for (size_t img_idx = 0; img_idx < query_size; img_idx++)
                {
                    size_t num_kp = extracted_kp[img_idx].size();
                    // Skip if no keypoint detected
                    if (!num_kp)
                        continue;   // Skip to next query

                    stringstream out_txt;

                    // BOW
                    bow bow_builder;
                    bow_builder.init(run_param);

                    /// Vector quantization
                    out_txt << "Quantizing.." << img_idx << "/" << query_size << " " << greenc << get_filename(queries[img_idx]) << endc << " ";
                    timespec quantizeTime = CurrentPreciseTime();
                    // Packing desc data
                    int sift_len = SIFThesaff::GetSIFTD();
                    float* kp_dat = new float[num_kp * sift_len];
                    for (size_t kp_idx = 0; kp_idx < num_kp; kp_idx++)
                        for (int desc_idx = 0; desc_idx < sift_len; desc_idx++)
                            kp_dat[kp_idx * sift_len + desc_idx] = extracted_desc[img_idx][kp_idx][desc_idx];
                    Matrix<float> data(kp_dat, num_kp, sift_len);  // size = num_kp x sift_len
                    // Prepare result space
                    Matrix<int> result_index;                           // size = num_kp x knn
                    Matrix<float> result_dist;                          // size = num_kp x knn
                    // Quantization
                    //omp_set_num_threads(current_num_thread);
                    ann.quantize(data, num_kp, result_index, result_dist);
                    //omp_set_num_threads(4);
                    out_txt << "done! (in " << setprecision(2) << fixed << TimeElapse(quantizeTime) << " s)" << endl;

                    /// BOW building
                    out_txt << "Building BOW..";
                    timespec bowTime = CurrentPreciseTime();
                    bow_builder.build_bow(result_index.ptr(), extracted_kp[img_idx], img_idx);   // All frame all will be assigned to sequence_id 0 (multi_bow only one) but with different image_id
                    out_txt << "done! (in " << setprecision(2) << fixed << TimeElapse(bowTime) << " s)" << endl;

                    /// Visualize full keypoints
                    if (visualize_enable)
                        visualize_bow(queries[img_idx], queries[img_idx] + ".display.png", bow_builder.get_last_bow(), false);

                    /// Masking
                    int mask_pass = 0;
                    if (run_param.mask_enable)
                    {
                        out_txt << "Masking..";
                        timespec maskTime = CurrentPreciseTime();
                        if (run_param.normpoint)
                            mask_pass = bow_builder.masking_lastbow(masks[img_idx], extracted_imgsize[img_idx].width, extracted_imgsize[img_idx].height);
                        else
                            mask_pass = bow_builder.masking_lastbow(masks[img_idx]);
                        out_txt << "[" << yellowc << mask_pass << endc << "/" << bluec << num_kp << endc << "] ";
                        out_txt << "done! (in " << setprecision(2) << fixed << TimeElapse(maskTime) << " s)" << endl;
                        total_mask_pass += mask_pass;

                        /// Visualize masked keypoints
                        if (visualize_enable)
                        {
                            // Polygon mask overlay
                            if (check_extension(mask_paths[img_idx], "mask"))
                            {
                                vector< vector<Point2f> > polygons;
                                oxMaskImport(mask_paths[img_idx], polygons);
                                overlay_mask(queries[img_idx], queries[img_idx] + ".mask.png", polygons, run_param.normpoint);
                            }
                            // Image mask overlay
                            else
                                overlay_mask(queries[img_idx], queries[img_idx] + ".mask.png", mask_paths[img_idx]);
                            visualize_bow(queries[img_idx] + ".mask.png", queries[img_idx] + ".mask.display.png", bow_builder.get_last_bow(), true);
                        }
                    }

                    /// Root bow_sig
                    if (run_param.powerlaw_enable)
                        bow_builder.rooting_lastbow();

                    /// Release memory
                    delete[] kp_dat;
                    delete[] result_index.ptr();
                    delete[] result_dist.ptr();

                    /// Finalizing bow_sig, Internally TF-Normalized
                    vector<bow_bin_object*>& bow_sig = bow_builder.finalize_bow();   // Have new memory of bow_bin_object and reuse memory of feature_object, and kp

                    /// TF-IDF. Normalize
                    bow_builder.logtf_idf_unitnormalize(bow_sig, idf);

                    /// Write Histogram result
                    export_bowsig(queries[img_idx] + run_param.hist_postfix, bow_sig, int(num_kp), mask_pass);

                    out_txt << redc << bow_sig.size() << endc << " bin(s)" << endl;

                    total_bin += bow_sig.size();

                    // Print Info
                    cout << out_txt.str();

                    /// Dump
                    if (run_param.matching_dump_enable)
                    {
                        // Collect
                        size_t dataset_id = img_idx;  // Late fusion, query numbers are different
                        for (size_t bin_idx = 0; bin_idx < bow_sig.size(); bin_idx++)
                        {
                            bow_bin_object* bin = bow_sig[bin_idx];
                            for (size_t feature_idx = 0; feature_idx < bin->features.size(); feature_idx++)
                                dumper.collect_kp(dataset_id, bin->cluster_id, bin->weight, bin->fg, bin->features[feature_idx]->sequence_id, bin->features[feature_idx]->kp);
                                // sequence_id of late fusion is always 0, sequence_id from bow_pool is already 0
                        }

                        // Dump
                        dumper.dump(queries[img_idx] + ".query.dump",
                                    vector<size_t>{img_idx},
                                    vector<string>{get_directory(queries[img_idx])},
                                    vector< vector<string> >{vector<string>{get_filename(queries[img_idx])}});
                    }

                    /// Release memory
                    bow_builder.reset_bow();        // Delete previously allocated memory (bow_bin_object, feature_object, and kp)
                }
            }
            //omp_set_dynamic(1);

            cout << "Total:" << endl;
            cout << redc << total_bin << endc << " bin(s)" << endl;
            cout << yellowc << total_mask_pass << endc << "/" << bluec << total_kp << endc  << " keypoint(s)" << endl;
        }

        // Release SIFT memory
        for (size_t img_idx = 0; img_idx < extracted_desc.size(); img_idx++)
        {
            for(size_t desc_idx = 0; desc_idx < extracted_desc[img_idx].size(); desc_idx++)
                delete[] extracted_desc[img_idx][desc_idx];   // delete only descriptor, since keypoint were used in build bow
            vector<float*>().swap(extracted_kp[img_idx]);
            vector<float*>().swap(extracted_desc[img_idx]);
        }
		vector< vector<float*> >().swap(extracted_kp);
		vector< vector<float*> >().swap(extracted_desc);
		vector<Size>().swap(extracted_imgsize);

        return total_bin;
    }
}

bool export_bowsig(const string& out, const vector<bow_bin_object*>& bow_sig, int num_kp, int mask_pass)
{
    bool ret = true;
    ofstream OutFile(out.c_str(), ios::binary);
    if (ret &= OutFile.is_open())
    {
        // Non-zero count
        size_t bin_count = bow_sig.size();
        OutFile.write(reinterpret_cast<char*>(&bin_count), sizeof(bin_count));

        // Bin
        for (size_t bin_id = 0; bin_id < bin_count; bin_id++)
        {
            // Cluster ID
            size_t cluster_id = bow_sig[bin_id]->cluster_id;
            OutFile.write(reinterpret_cast<char*>(&cluster_id), sizeof(cluster_id));

            // Weight
            float weight = bow_sig[bin_id]->weight;
            OutFile.write(reinterpret_cast<char*>(&weight), sizeof(weight));

            // Foreground flag
            bool fg = bow_sig[bin_id]->fg;
            OutFile.write(reinterpret_cast<char*>(&fg), sizeof(fg));

            // Feature Count
            size_t feature_count = bow_sig[bin_id]->features.size();
            OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));

            int head_size = SIFThesaff::GetSIFTHeadSize();
            for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
            {
                // Write all features from bin
                feature_object* feature = bow_sig[bin_id]->features[bow_feature_id];
                // Image ID
                OutFile.write(reinterpret_cast<char*>(&(feature->image_id)), sizeof(feature->image_id));
                // Sequence ID
                OutFile.write(reinterpret_cast<char*>(&(feature->sequence_id)), sizeof(feature->sequence_id));
                /*
                // Weight (asymmetric weight)
                OutFile.write(reinterpret_cast<char*>(&(feature->weight)), sizeof(feature->weight));
                */
                // x y a b c
                OutFile.write(reinterpret_cast<char*>(feature->kp), head_size * sizeof(*(feature->kp)));
            }
        }

        // Write num_kp
        OutFile.write(reinterpret_cast<char*>(&num_kp), sizeof(num_kp));

        // Write mask_pass
        OutFile.write(reinterpret_cast<char*>(&mask_pass), sizeof(mask_pass));

        // Close file
        OutFile.close();
    }

    ofstream OkFile((out + ".ok").c_str(), ios::binary);
    OkFile.close();

    return ret;
}

void visualize_bow(const string& in_img, const string& out_img, const vector<bow_bin_object*>& bow_sig, const bool checkmask)
{
    vector<INS_KP> keypoints;
    for (size_t bow_idx = 0; bow_idx < bow_sig.size(); bow_idx++)
    {
        // Draw only foreground
        if (!checkmask || bow_sig[bow_idx]->fg)
        {
            for (size_t feature_idx = 0; feature_idx < bow_sig[bow_idx]->features.size(); feature_idx++)
            {
                float* kp = bow_sig[bow_idx]->features[feature_idx]->kp;
                keypoints.push_back(INS_KP{kp[0], kp[1], kp[2], kp[3], kp[4]});
            }
        }
    }
	if (run_param.feature_type == FEAT_SIFTHESAFF)
	{
		SIFThesaff sift_obj;
		sift_obj.draw_sifts(in_img, out_img, keypoints, DRAW_POINT, run_param.colorspace, run_param.normpoint, run_param.rootsift);
	}
}

// Misc
void oxMaskImport(const string& mask_path, vector< vector<Point2f> >& polygons)
{
    /// Read Mask as a polygon
    // Load mask
    ifstream mask_File (mask_path.c_str());
    if (mask_File)
    {
        cout << "Reading mask..";
        cout.flush();
        startTime = CurrentPreciseTime();

        string line;
        getline(mask_File, line);

        // Read mask count
        size_t mask_count = strtoull(line.c_str(), NULL, 0);

        for (size_t mask_id = 0; mask_id < mask_count; mask_id++)
        {
            getline(mask_File, line);

            // Read vertex count
            size_t mask_vertex_count = strtoull(line.c_str(), NULL, 0);

            // Preparing mask array
            vector<Point2f> curr_mask_polygon;

            // Read mask vertecies
            for (size_t mask_idx = 0; mask_idx < mask_vertex_count; mask_idx++)
            {
                getline(mask_File, line);

                vector<string> vertex;
                StringExplode(line, ",", vertex);

                curr_mask_polygon.push_back(Point2f(atof(vertex[0].c_str()), atof(vertex[1].c_str())));
            }

            // Keep multiple masks
            polygons.push_back(curr_mask_polygon);
        }

        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        mask_File.close();
    }
}

// Memory management
//;)
