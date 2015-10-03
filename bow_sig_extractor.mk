##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Release
ProjectName            :=bow_sig_extractor
ConfigurationName      :=Release
WorkspacePath          := "${HOME}/webstylix/code"
ProjectPath            := "${HOME}/webstylix/code/ins_online"
IntermediateDirectory  :=./Release
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Siriwat Kasamwattanarote
Date                   :=03/10/15
CodeLitePath           :="${HOME}/webstylix/configurations/.codelite"
LinkerName             :=g++
SharedObjectLinkerName :=g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=$(PreprocessorSwitch)NDEBUG 
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="bow_sig_extractor.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  `pkg-config opencv --libs` `pkg-config --libs lapacke`
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch)${HOME}/local/include 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)ins $(LibrarySwitch)orb $(LibrarySwitch)sifthesaff $(LibrarySwitch)alphautils $(LibrarySwitch)opencv_core $(LibrarySwitch)opencv_features2d $(LibrarySwitch)opencv_highgui $(LibrarySwitch)x264 $(LibrarySwitch)faac $(LibrarySwitch)ransac $(LibrarySwitch)lapacke $(LibrarySwitch)lapack $(LibrarySwitch)blas $(LibrarySwitch)tmglib $(LibrarySwitch)hdf5 $(LibrarySwitch)hdf5_hl_cpp $(LibrarySwitch)hdf5_cpp $(LibrarySwitch)hdf5_hl $(LibrarySwitch)mpi_cxx $(LibrarySwitch)mpi $(LibrarySwitch)rt $(LibrarySwitch)gomp $(LibrarySwitch)pthread $(LibrarySwitch)dl 
ArLibs                 :=  "libins.a" "liborb.a" "libsifthesaff.a" "libalphautils.a" "opencv_core" "opencv_features2d" "opencv_highgui" "x264" "faac" "libransac.a" "lapacke" "lapack" "blas" "tmglib" "hdf5" "hdf5_hl_cpp" "hdf5_cpp" "hdf5_hl" "mpi_cxx" "mpi" "rt" "gomp" "pthread" "dl" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)${HOME}/local/lib $(LibraryPathSwitch)../lib/ins/$(ConfigurationName) $(LibraryPathSwitch)../lib/orb/$(ConfigurationName) $(LibraryPathSwitch)../lib/sifthesaff/$(ConfigurationName) $(LibraryPathSwitch)../lib/alphautils/$(ConfigurationName) $(LibraryPathSwitch)../lib/ransac/$(ConfigurationName) 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := ar rcu
CXX      := g++
CC       := gcc
CXXFLAGS :=  -O3 -fopenmp -std=c++11 -Wall `pkg-config --cflags opencv` $(Preprocessors)
CFLAGS   :=  -O2 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/bow_sig_extractor.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d ./Release || $(MakeDirCommand) ./Release


$(IntermediateDirectory)/.d:
	@test -d ./Release || $(MakeDirCommand) ./Release

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/bow_sig_extractor.cpp$(ObjectSuffix): bow_sig_extractor.cpp $(IntermediateDirectory)/bow_sig_extractor.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "${HOME}/webstylix/code/ins_online/bow_sig_extractor.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/bow_sig_extractor.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/bow_sig_extractor.cpp$(DependSuffix): bow_sig_extractor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/bow_sig_extractor.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/bow_sig_extractor.cpp$(DependSuffix) -MM "bow_sig_extractor.cpp"

$(IntermediateDirectory)/bow_sig_extractor.cpp$(PreprocessSuffix): bow_sig_extractor.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/bow_sig_extractor.cpp$(PreprocessSuffix) "bow_sig_extractor.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Release/


