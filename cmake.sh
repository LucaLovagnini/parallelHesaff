#!/bin/bash

#Directory containing
ROOT_DIR=$HOME/ParallelOpenCV
OPENCV=$ROOT_DIR/opencv
OPENCV_CONTRIB=$ROOT_DIR/opencv_contrib/modules
INSTALL="$ROOT_DIR/install"
BUILD=$OPENCV/build
IPPROOT="/opt/intel/compilers_and_libraries_2017.1.132/linux/ipp/"

ICPC_OPTIMAL_OPTS="-O3 -qopenmp -parallel -ipo -no-prec-div -ansi-alias -fma -align -finline-functions -xCORE-AVX2 -fp-model fast=2 " 

echo "ICPC_OPTIMAL_OPTS=$ICPC_OPTIMAL_OPTS"

set -x

case "$1" in
	gcc)
		echo "gcc"
		PERSONAL="-D WITH_CUDA=OFF" 
		;;
	icpc)
		echo "icpc"
		PERSONAL="-D WITH_CUDA=OFF  -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_FLAGS=\" $ICPC_OPTIMAL_OPTS \" -DCMAKE_CXX_FLAGS=\" $ICPC_OPTIMAL_OPTS \" -D CMAKE_MODULE_LINKER_FLAGS=\" $ICPC_OPTIMAL_OPTS \" -D CMAKE_SHARED_LINKER_FLAGS=\" $ICPC_OPTIMAL_OPTS \""
		;;
	gcc_cuda)
		echo "gcc_cuda"
		PERSONAL="-D WITH_CUDA=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON -D WITH_CUFFT=ON"
		;;
	*)
		echo "$0 usage: gcc|icpc|gcc_cuda"
		exit 1
esac

INSTALL="$INSTALL"_"$1"

if [ "$#" -lt 2 ] || [ "$2" -eq 1 ]; then
	#DON'T use $BUILD (otherwise if bad setting you could delete /* /.*)
	rm -rf $OPENCV/build/* $OPENCV/build/.*
fi

eval cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$INSTALL -D OPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB -DWITH_OPENCL=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_LAPACK=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DENABLE_AVX2=ON -DHAVE_MKL=ON -DMKL_WITH_TBB=OFF -D WITH_IPP=ON -DIPPROOT="$IPPROOT" -D BUILD_opencv_java=OFF -D BUILD_opencv_photo=OFF -D BUILD_opencv_python=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_stitching=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_videostab=OFF -D BUILD_opencv_viz=OFF -D BUILD_opencv_viz=off -D BUILD_opencv_world=off -D BUILD_opencv_aruco=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_cnn_3dobj=OFF -D BUILD_opencv_contrib_world=OFF -D BUILD_opencv_cvv=OFF -D BUILD_opencv_datasets=OFF -D BUILD_opencv_dnn=OFF -D BUILD_opencv_dnns_easily_fooled=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_freetype=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_hdf=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_matlab=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_reg=OFF -D BUILD_opencv_objdetect=OFF -D BUILD_opencv_plot=OFF -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D BUILD_opencv_saliency=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_stereo=OFF -D BUILD_opencv_ximgproc=OFF -D BUILD_opencv_optflow=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_text=OFF "$PERSONAL" -B$BUILD -H$OPENCV

make -C $BUILD -j16
make -C $BUILD install -j16
