# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/harrisonward/Desktop/CS/Git/virtual_yellow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/harrisonward/Desktop/CS/Git/virtual_yellow/build

# Include any dependencies generated for this target.
include CMakeFiles/background_fill.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/background_fill.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/background_fill.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/background_fill.dir/flags.make

CMakeFiles/background_fill.dir/src/background_fill.cpp.o: CMakeFiles/background_fill.dir/flags.make
CMakeFiles/background_fill.dir/src/background_fill.cpp.o: /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill.cpp
CMakeFiles/background_fill.dir/src/background_fill.cpp.o: CMakeFiles/background_fill.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/harrisonward/Desktop/CS/Git/virtual_yellow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/background_fill.dir/src/background_fill.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/background_fill.dir/src/background_fill.cpp.o -MF CMakeFiles/background_fill.dir/src/background_fill.cpp.o.d -o CMakeFiles/background_fill.dir/src/background_fill.cpp.o -c /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill.cpp

CMakeFiles/background_fill.dir/src/background_fill.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/background_fill.dir/src/background_fill.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill.cpp > CMakeFiles/background_fill.dir/src/background_fill.cpp.i

CMakeFiles/background_fill.dir/src/background_fill.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/background_fill.dir/src/background_fill.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill.cpp -o CMakeFiles/background_fill.dir/src/background_fill.cpp.s

CMakeFiles/background_fill.dir/src/fill_functions.cpp.o: CMakeFiles/background_fill.dir/flags.make
CMakeFiles/background_fill.dir/src/fill_functions.cpp.o: /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/fill_functions.cpp
CMakeFiles/background_fill.dir/src/fill_functions.cpp.o: CMakeFiles/background_fill.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/harrisonward/Desktop/CS/Git/virtual_yellow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/background_fill.dir/src/fill_functions.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/background_fill.dir/src/fill_functions.cpp.o -MF CMakeFiles/background_fill.dir/src/fill_functions.cpp.o.d -o CMakeFiles/background_fill.dir/src/fill_functions.cpp.o -c /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/fill_functions.cpp

CMakeFiles/background_fill.dir/src/fill_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/background_fill.dir/src/fill_functions.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/fill_functions.cpp > CMakeFiles/background_fill.dir/src/fill_functions.cpp.i

CMakeFiles/background_fill.dir/src/fill_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/background_fill.dir/src/fill_functions.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/fill_functions.cpp -o CMakeFiles/background_fill.dir/src/fill_functions.cpp.s

# Object files for target background_fill
background_fill_OBJECTS = \
"CMakeFiles/background_fill.dir/src/background_fill.cpp.o" \
"CMakeFiles/background_fill.dir/src/fill_functions.cpp.o"

# External object files for target background_fill
background_fill_EXTERNAL_OBJECTS =

/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: CMakeFiles/background_fill.dir/src/background_fill.cpp.o
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: CMakeFiles/background_fill.dir/src/fill_functions.cpp.o
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: CMakeFiles/background_fill.dir/build.make
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libboost_filesystem-mt.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_gapi.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_stitching.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_alphamat.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_aruco.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_bgsegm.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_bioinspired.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_ccalib.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_dnn_objdetect.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_dnn_superres.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_dpm.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_face.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_freetype.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_fuzzy.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_hfs.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_img_hash.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_intensity_transform.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_line_descriptor.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_mcc.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_quality.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_rapid.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_reg.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_rgbd.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_saliency.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_sfm.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_stereo.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_structured_light.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_superres.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_surface_matching.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_tracking.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_videostab.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_viz.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_wechat_qrcode.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_xfeatures2d.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_xobjdetect.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_xphoto.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libboost_atomic-mt.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_shape.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_highgui.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_datasets.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_plot.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_text.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_ml.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_phase_unwrapping.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_optflow.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_ximgproc.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_video.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_videoio.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_imgcodecs.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_objdetect.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_calib3d.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_dnn.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_features2d.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_flann.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_photo.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_imgproc.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: /usr/local/lib/libopencv_core.4.8.1.dylib
/Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill: CMakeFiles/background_fill.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/harrisonward/Desktop/CS/Git/virtual_yellow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/background_fill.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/background_fill.dir/build: /Users/harrisonward/Desktop/CS/Git/virtual_yellow/src/background_fill
.PHONY : CMakeFiles/background_fill.dir/build

CMakeFiles/background_fill.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/background_fill.dir/cmake_clean.cmake
.PHONY : CMakeFiles/background_fill.dir/clean

CMakeFiles/background_fill.dir/depend:
	cd /Users/harrisonward/Desktop/CS/Git/virtual_yellow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/harrisonward/Desktop/CS/Git/virtual_yellow /Users/harrisonward/Desktop/CS/Git/virtual_yellow /Users/harrisonward/Desktop/CS/Git/virtual_yellow/build /Users/harrisonward/Desktop/CS/Git/virtual_yellow/build /Users/harrisonward/Desktop/CS/Git/virtual_yellow/build/CMakeFiles/background_fill.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/background_fill.dir/depend

