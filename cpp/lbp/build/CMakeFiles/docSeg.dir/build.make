# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jobin/docSegmentation/cpp/lbp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jobin/docSegmentation/cpp/lbp/build

# Include any dependencies generated for this target.
include CMakeFiles/docSeg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/docSeg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/docSeg.dir/flags.make

CMakeFiles/docSeg.dir/main.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/main.cpp.o -c /home/jobin/docSegmentation/cpp/lbp/main.cpp

CMakeFiles/docSeg.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jobin/docSegmentation/cpp/lbp/main.cpp > CMakeFiles/docSeg.dir/main.cpp.i

CMakeFiles/docSeg.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jobin/docSegmentation/cpp/lbp/main.cpp -o CMakeFiles/docSeg.dir/main.cpp.s

CMakeFiles/docSeg.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/main.cpp.o.requires

CMakeFiles/docSeg.dir/main.cpp.o.provides: CMakeFiles/docSeg.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/main.cpp.o.provides

CMakeFiles/docSeg.dir/main.cpp.o.provides.build: CMakeFiles/docSeg.dir/main.cpp.o

CMakeFiles/docSeg.dir/lbp.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/lbp.cpp.o: ../lbp.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/lbp.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/lbp.cpp.o -c /home/jobin/docSegmentation/cpp/lbp/lbp.cpp

CMakeFiles/docSeg.dir/lbp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/lbp.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jobin/docSegmentation/cpp/lbp/lbp.cpp > CMakeFiles/docSeg.dir/lbp.cpp.i

CMakeFiles/docSeg.dir/lbp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/lbp.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jobin/docSegmentation/cpp/lbp/lbp.cpp -o CMakeFiles/docSeg.dir/lbp.cpp.s

CMakeFiles/docSeg.dir/lbp.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/lbp.cpp.o.requires

CMakeFiles/docSeg.dir/lbp.cpp.o.provides: CMakeFiles/docSeg.dir/lbp.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/lbp.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/lbp.cpp.o.provides

CMakeFiles/docSeg.dir/lbp.cpp.o.provides.build: CMakeFiles/docSeg.dir/lbp.cpp.o

CMakeFiles/docSeg.dir/histogram.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/histogram.cpp.o: ../histogram.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/histogram.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/histogram.cpp.o -c /home/jobin/docSegmentation/cpp/lbp/histogram.cpp

CMakeFiles/docSeg.dir/histogram.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/histogram.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jobin/docSegmentation/cpp/lbp/histogram.cpp > CMakeFiles/docSeg.dir/histogram.cpp.i

CMakeFiles/docSeg.dir/histogram.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/histogram.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jobin/docSegmentation/cpp/lbp/histogram.cpp -o CMakeFiles/docSeg.dir/histogram.cpp.s

CMakeFiles/docSeg.dir/histogram.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/histogram.cpp.o.requires

CMakeFiles/docSeg.dir/histogram.cpp.o.provides: CMakeFiles/docSeg.dir/histogram.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/histogram.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/histogram.cpp.o.provides

CMakeFiles/docSeg.dir/histogram.cpp.o.provides.build: CMakeFiles/docSeg.dir/histogram.cpp.o

CMakeFiles/docSeg.dir/GCoptimization.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/GCoptimization.cpp.o: ../GCoptimization.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/GCoptimization.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/GCoptimization.cpp.o -c /home/jobin/docSegmentation/cpp/lbp/GCoptimization.cpp

CMakeFiles/docSeg.dir/GCoptimization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/GCoptimization.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jobin/docSegmentation/cpp/lbp/GCoptimization.cpp > CMakeFiles/docSeg.dir/GCoptimization.cpp.i

CMakeFiles/docSeg.dir/GCoptimization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/GCoptimization.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jobin/docSegmentation/cpp/lbp/GCoptimization.cpp -o CMakeFiles/docSeg.dir/GCoptimization.cpp.s

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides: CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides

CMakeFiles/docSeg.dir/GCoptimization.cpp.o.provides.build: CMakeFiles/docSeg.dir/GCoptimization.cpp.o

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o: CMakeFiles/docSeg.dir/flags.make
CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o: ../LinkedBlockList.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o -c /home/jobin/docSegmentation/cpp/lbp/LinkedBlockList.cpp

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jobin/docSegmentation/cpp/lbp/LinkedBlockList.cpp > CMakeFiles/docSeg.dir/LinkedBlockList.cpp.i

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jobin/docSegmentation/cpp/lbp/LinkedBlockList.cpp -o CMakeFiles/docSeg.dir/LinkedBlockList.cpp.s

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires:
.PHONY : CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires
	$(MAKE) -f CMakeFiles/docSeg.dir/build.make CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides.build
.PHONY : CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides

CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.provides.build: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o

# Object files for target docSeg
docSeg_OBJECTS = \
"CMakeFiles/docSeg.dir/main.cpp.o" \
"CMakeFiles/docSeg.dir/lbp.cpp.o" \
"CMakeFiles/docSeg.dir/histogram.cpp.o" \
"CMakeFiles/docSeg.dir/GCoptimization.cpp.o" \
"CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o"

# External object files for target docSeg
docSeg_EXTERNAL_OBJECTS =

docSeg: CMakeFiles/docSeg.dir/main.cpp.o
docSeg: CMakeFiles/docSeg.dir/lbp.cpp.o
docSeg: CMakeFiles/docSeg.dir/histogram.cpp.o
docSeg: CMakeFiles/docSeg.dir/GCoptimization.cpp.o
docSeg: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o
docSeg: CMakeFiles/docSeg.dir/build.make
docSeg: /usr/local/lib/libopencv_viz.so.2.4.9
docSeg: /usr/local/lib/libopencv_videostab.so.2.4.9
docSeg: /usr/local/lib/libopencv_video.so.2.4.9
docSeg: /usr/local/lib/libopencv_ts.a
docSeg: /usr/local/lib/libopencv_superres.so.2.4.9
docSeg: /usr/local/lib/libopencv_stitching.so.2.4.9
docSeg: /usr/local/lib/libopencv_photo.so.2.4.9
docSeg: /usr/local/lib/libopencv_ocl.so.2.4.9
docSeg: /usr/local/lib/libopencv_objdetect.so.2.4.9
docSeg: /usr/local/lib/libopencv_nonfree.so.2.4.9
docSeg: /usr/local/lib/libopencv_ml.so.2.4.9
docSeg: /usr/local/lib/libopencv_legacy.so.2.4.9
docSeg: /usr/local/lib/libopencv_imgproc.so.2.4.9
docSeg: /usr/local/lib/libopencv_highgui.so.2.4.9
docSeg: /usr/local/lib/libopencv_gpu.so.2.4.9
docSeg: /usr/local/lib/libopencv_flann.so.2.4.9
docSeg: /usr/local/lib/libopencv_features2d.so.2.4.9
docSeg: /usr/local/lib/libopencv_core.so.2.4.9
docSeg: /usr/local/lib/libopencv_contrib.so.2.4.9
docSeg: /usr/local/lib/libopencv_calib3d.so.2.4.9
docSeg: /usr/lib/x86_64-linux-gnu/libGLU.so
docSeg: /usr/lib/x86_64-linux-gnu/libGL.so
docSeg: /usr/lib/x86_64-linux-gnu/libSM.so
docSeg: /usr/lib/x86_64-linux-gnu/libICE.so
docSeg: /usr/lib/x86_64-linux-gnu/libX11.so
docSeg: /usr/lib/x86_64-linux-gnu/libXext.so
docSeg: /usr/local/lib/libopencv_nonfree.so.2.4.9
docSeg: /usr/local/lib/libopencv_ocl.so.2.4.9
docSeg: /usr/local/lib/libopencv_gpu.so.2.4.9
docSeg: /usr/local/lib/libopencv_photo.so.2.4.9
docSeg: /usr/local/lib/libopencv_objdetect.so.2.4.9
docSeg: /usr/local/lib/libopencv_legacy.so.2.4.9
docSeg: /usr/local/lib/libopencv_video.so.2.4.9
docSeg: /usr/local/lib/libopencv_ml.so.2.4.9
docSeg: /usr/local/lib/libopencv_calib3d.so.2.4.9
docSeg: /usr/local/lib/libopencv_features2d.so.2.4.9
docSeg: /usr/local/lib/libopencv_highgui.so.2.4.9
docSeg: /usr/local/lib/libopencv_imgproc.so.2.4.9
docSeg: /usr/local/lib/libopencv_flann.so.2.4.9
docSeg: /usr/local/lib/libopencv_core.so.2.4.9
docSeg: CMakeFiles/docSeg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable docSeg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/docSeg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/docSeg.dir/build: docSeg
.PHONY : CMakeFiles/docSeg.dir/build

CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/main.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/lbp.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/histogram.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/GCoptimization.cpp.o.requires
CMakeFiles/docSeg.dir/requires: CMakeFiles/docSeg.dir/LinkedBlockList.cpp.o.requires
.PHONY : CMakeFiles/docSeg.dir/requires

CMakeFiles/docSeg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/docSeg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/docSeg.dir/clean

CMakeFiles/docSeg.dir/depend:
	cd /home/jobin/docSegmentation/cpp/lbp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jobin/docSegmentation/cpp/lbp /home/jobin/docSegmentation/cpp/lbp /home/jobin/docSegmentation/cpp/lbp/build /home/jobin/docSegmentation/cpp/lbp/build /home/jobin/docSegmentation/cpp/lbp/build/CMakeFiles/docSeg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/docSeg.dir/depend

