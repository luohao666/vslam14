# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/snorker/Documents/slambook/ch7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/snorker/Documents/slambook/ch7/build

# Include any dependencies generated for this target.
include CMakeFiles/pose_estimation_3d2d.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_estimation_3d2d.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_estimation_3d2d.dir/flags.make

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o: CMakeFiles/pose_estimation_3d2d.dir/flags.make
CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o: ../pose_estimation_3d2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/snorker/Documents/slambook/ch7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o -c /home/snorker/Documents/slambook/ch7/pose_estimation_3d2d.cpp

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snorker/Documents/slambook/ch7/pose_estimation_3d2d.cpp > CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.i

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snorker/Documents/slambook/ch7/pose_estimation_3d2d.cpp -o CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.s

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.requires:

.PHONY : CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.requires

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.provides: CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.requires
	$(MAKE) -f CMakeFiles/pose_estimation_3d2d.dir/build.make CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.provides.build
.PHONY : CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.provides

CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.provides.build: CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o


# Object files for target pose_estimation_3d2d
pose_estimation_3d2d_OBJECTS = \
"CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o"

# External object files for target pose_estimation_3d2d
pose_estimation_3d2d_EXTERNAL_OBJECTS =

pose_estimation_3d2d: CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o
pose_estimation_3d2d: CMakeFiles/pose_estimation_3d2d.dir/build.make
pose_estimation_3d2d: /usr/local/lib/libopencv_videostab.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_ts.a
pose_estimation_3d2d: /usr/local/lib/libopencv_superres.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_stitching.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_contrib.so.2.4.13
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libcxsparse.so
pose_estimation_3d2d: /usr/local/lib/libceres.a
pose_estimation_3d2d: /home/snorker/Documents/slambook/3rdparty/Sophus/build/libSophus.so
pose_estimation_3d2d: /usr/local/lib/libopencv_nonfree.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_ocl.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_gpu.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_photo.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_objdetect.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_legacy.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_video.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_ml.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_calib3d.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_features2d.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_highgui.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_imgproc.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_flann.so.2.4.13
pose_estimation_3d2d: /usr/local/lib/libopencv_core.so.2.4.13
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libglog.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libspqr.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libtbb.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libcholmod.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libccolamd.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libcamd.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libcolamd.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libamd.so
pose_estimation_3d2d: /usr/lib/liblapack.so
pose_estimation_3d2d: /usr/lib/libf77blas.so
pose_estimation_3d2d: /usr/lib/libatlas.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/librt.so
pose_estimation_3d2d: /usr/lib/liblapack.so
pose_estimation_3d2d: /usr/lib/libf77blas.so
pose_estimation_3d2d: /usr/lib/libatlas.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
pose_estimation_3d2d: /usr/lib/x86_64-linux-gnu/librt.so
pose_estimation_3d2d: CMakeFiles/pose_estimation_3d2d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/snorker/Documents/slambook/ch7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pose_estimation_3d2d"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_estimation_3d2d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_estimation_3d2d.dir/build: pose_estimation_3d2d

.PHONY : CMakeFiles/pose_estimation_3d2d.dir/build

CMakeFiles/pose_estimation_3d2d.dir/requires: CMakeFiles/pose_estimation_3d2d.dir/pose_estimation_3d2d.cpp.o.requires

.PHONY : CMakeFiles/pose_estimation_3d2d.dir/requires

CMakeFiles/pose_estimation_3d2d.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_estimation_3d2d.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_estimation_3d2d.dir/clean

CMakeFiles/pose_estimation_3d2d.dir/depend:
	cd /home/snorker/Documents/slambook/ch7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/snorker/Documents/slambook/ch7 /home/snorker/Documents/slambook/ch7 /home/snorker/Documents/slambook/ch7/build /home/snorker/Documents/slambook/ch7/build /home/snorker/Documents/slambook/ch7/build/CMakeFiles/pose_estimation_3d2d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose_estimation_3d2d.dir/depend

