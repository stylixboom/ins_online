#ifndef ins_online_VERSION_H
#define ins_online_VERSION_H

namespace ins_online_AutoVersion{
	
	//Date Version Types
	static const char ins_online_DATE[] = "09";
	static const char ins_online_MONTH[] = "05";
	static const char ins_online_YEAR[] = "2014";
	static const char ins_online_UBUNTU_VERSION_STYLE[] =  "14.05";
	
	//Software Status
	static const char ins_online_STATUS[] =  "Alpha";
	static const char ins_online_STATUS_SHORT[] =  "a";
	
	//Standard Version Type
	static const long ins_online_MAJOR  = 2;
	static const long ins_online_MINOR  = 0;
	static const long ins_online_BUILD  = 9;
	static const long ins_online_REVISION  = 20;
	
	//Miscellaneous Version Types
	static const long ins_online_BUILDS_COUNT  = 10;
	#define ins_online_RC_FILEVERSION 2,0,9,20
	#define ins_online_RC_FILEVERSION_STRING "2, 0, 9, 20\0"
	static const char ins_online_FULLVERSION_STRING [] = "2.0.9.20";
	
	//These values are to keep track of your versioning state, don't modify them.
	static const long ins_online_BUILD_HISTORY  = 5;
	

}
#endif //ins_online_VERSION_H
