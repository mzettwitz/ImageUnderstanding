# ImageUnterstanding
Repo for attendand project in lecture 'Advanced Topics in Image Understanding' in summerterm 2017

## Used Libs:
 OpenCV 3.2.0 + contrib-repo

## Install instructions:
OpenCV install instructions on ubuntu:

- Download opencv-master(https://github.com/opencv/opencv) and opencv-contrib(https://github.com/opencv/opencv_contrib)
- Build and install OpenCV by command line (or GUI):
```
$ cd <opencv_build_dir>
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib/modules> <opencv_source_dir>
$ make -j4 (4 = number of cores, you can use up to the maximum including hyperthreading, e.g. make -j16)
$ sudo make install
```
- Make sure you have permissions to the folders, if you are not sure: simply use ```$ sudo chmod -R 777 <Folder containing openCV stuff(opencv, opencv_contrib, build)>```
- If errors occur while building, you may need to disable libs not found, by altering the cmake params (e.g. -DWITH_LAPACK=OFF), clear the build dir and rerun cmake and make step

## Build with CMake
```
$ cd <ImageUnderstanding>
$ cmake .
$ make
```
