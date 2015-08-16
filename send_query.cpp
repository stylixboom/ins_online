/*
* send_query.cpp
*
*  Created on: June 15, 2012
*      Author: Siriwat Kasamwattanarote
*/
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <unistd.h> // usleep
#include <unordered_map>

#include "../lib/alphautils/alphautils.h"

using namespace std;
using namespace alphautils;

int main(int argc, char* argv[])
{
	stringstream debug;
	//stringstream OutputPath;
	stringstream session_path;
	stringstream list_path;
	stringstream query_stack_path;
	stringstream ready_signal_path;

	if(argc < 6)
	{
		cout << "Wrong parameter." << endl;
		cout << "Usage: ./send_query sessionname Mask:0|1 DevOpt:0|1 RansacOpt:0|1 ShowAllOpt:0|1 Img1 [Img2 ...]" << endl;
		exit(-1);
	}
	string query_root_dir = "/home/stylix/webstylix/code/ins_online/query";

	session_path << query_root_dir << "/" << argv[1];
	list_path << session_path.str() << "/list.txt";
	ready_signal_path << session_path.str() << "/ready.sig";
	query_stack_path << query_root_dir << "/query_stack.txt";

	//Options
	int Mask = 0;
	int DevOpt = 0;
	int RansacOpt = 0;
	int ShowAllOpt = 0;
	Mask = atoi(argv[2]);
	DevOpt = atoi(argv[3]);
	RansacOpt = atoi(argv[4]);
	ShowAllOpt = atoi(argv[5]);

	//GetFIles
	/*string dir = query_stack_path.str();
	vector<string> files = vector<string>();
	getdir(dir,files);*/

	//Session Dir
	debug << "Chk exist dir: " << session_path.str() << " ";
	make_dir_available(session_path.str());

	//ListFile
	debug << "Openning " << list_path.str() << " ";
	ofstream CreateListFile (list_path.str().c_str());
	if(CreateListFile.is_open())
	term_out
		for(int i = 5; i < argc; i++)
		CreateListFile << session_path.str() << "/" << argv[i] << endl; // self-machine
		//CreateListFile << OutputPath.str() << session_path.str() << "/" << argv[i] << endl; // per900a to per900b
		//CreateListFile.flush();
		CreateListFile.close();
		debug << list_path.str() << " OK!" << " ";
	}

	debug << "Openning " << query_stack_path.str() << " ";
	//Add session id
	ofstream CreateQueriesFile (query_stack_path.str().c_str(), ios::app);
	if(CreateQueriesFile.is_open())
	{
		// sessionname Mask:0|1 Dev:0|1 Ransac:0|1 Showall:0|1
		CreateQueriesFile << argv[1] << " " << Mask << " " << DevOpt << " " << RansacOpt << " " << ShowAllOpt << endl;
		//CreateQueriesFile.flush();
		CreateQueriesFile.close();
		debug << query_stack_path.str() << " OK!" << " ";
	}

	cout << debug.str();

	return 0;
}
//:)
