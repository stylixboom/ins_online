#ifndef ins_online_VERSION_H
#define ins_online_VERSION_H

namespace ins_online_AutoVersion{
	
	//Date Version Types
	static const char ins_online_DATE[] = "24";
	static const char ins_online_MONTH[] = "04";
	static const char ins_online_YEAR[] = "2014";
	static const char ins_online_UBUNTU_VERSION_STYLE[] =  "14.04";
	
	//Software Status
	static const char ins_online_STATUS[] =  "Alpha";
	static const char ins_online_STATUS_SHORT[] =  "a";
	
	//Standard Version Type
	static const long ins_online_MAJOR  = 2;
	static const long ins_online_MINOR  = 0;
	static const long ins_online_BUILD  = 6;
	static const long ins_online_REVISION  = 9;
	
	//Miscellaneous Version Types
	static const long ins_online_BUILDS_COUNT  = 2;
	#define ins_online_RC_FILEVERSION 2,0,6,9
	#define ins_online_RC_FILEVERSION_STRING "2, 0, 6, 9\0"
	static const char ins_online_FULLVERSION_STRING [] = "2.0.6.9";
	
	//These values are to keep track of your versioning state, don't modify them.
	static const long ins_online_BUILD_HISTORY  = 2;
	

}
#endif //ins_online_VERSION_H
