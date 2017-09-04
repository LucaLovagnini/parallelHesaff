/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <iostream>
#include <fstream>
#include <omp.h>
#include <thread>
#include <chrono>

#define SLEEP 1

#include "hesaff.hpp"

int main(int argc, char **argv)
{

   //print OpenCV information
   //std::cout<<getBuildInformation()<<std::endl;     

   //omp_set_num_threads(1);
   if (argc>1)
   {
      int runs=1;
      int maxthreads=omp_get_max_threads();
      float ratioThreshold = 0.23;
      bool scalability=false;
      if (argc>2){
        maxthreads = atoi(argv[2]);
        std::cout<<"maxthreads="<<maxthreads<<std::endl;
      }
      if (argc>3){
        runs = atoi(argv[3]);
        std::cout<<"runs="<<runs<<std::endl;
      }
      if (argc>4){
    	  ratioThreshold = atoi(argv[4]);
    	  std::cout<<"ratioThreshold="<<ratioThreshold<<std::endl;
      }
      if (argc>5){
        scalability = bool(atoi(argv[5]));
        if(scalability)
	  std::cout<<"Testing scalability"<<std::endl;
      }
      const std::string fileName = argv[1];
      Mat tmp = imread(fileName);
      Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));
      
      float *out = image.ptr<float>(0);
      unsigned char *in  = tmp.ptr<unsigned char>(0); 

      for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
      {
         *out = (float(in[0]) + in[1] + in[2])/3.0f;
         out++;
         in+=3;
      }

      HessianAffineParams par;

      int initialThreads = scalability ? 1 : maxthreads;
      const std::string statsFileName = "hesaffStats.ods";
      std::ofstream statsFile;
      if(std::ifstream(statsFileName))
    	  statsFile.open(statsFileName, ios::app);
      else{
    	  statsFile.open(statsFileName, ios::out);
    	  statsFile<<"FileName"<<"\t"<<"Threads"<<"\t"<<"DescriptorTime"<<"\t"<<"ParallelTime"<<"\n";
      }
	for(int nThreads=initialThreads; nThreads<=maxthreads; nThreads++){
		omp_set_num_threads(nThreads);
		double descAvgTime = 0;
		double parallelAvgTime = 0;
		for(size_t i=0; i<runs; i++){
			std::cout<<"---------------------------------------------------"<<std::endl;
		    std::cout<<"File: "<<fileName<<std::endl;
			std::cout<<"Using "<<nThreads<<" threads"<<std::endl;
			// copy params 
			PyramidParams p;
			p.threshold = par.threshold;

			AffineShapeParams ap;
			ap.maxIterations = par.max_iter;
			ap.patchSize = par.patch_size;
			ap.mrSize = par.desc_factor;

			SIFTDescriptorParams sp;
			sp.patchSize = par.patch_size;

			cv::Mat1f descriptors;
			Wrapper wrapper(sp, ap);

			cv::Mat imageClone = image.clone();
			AffineHessianDetector detector(p, ap, sp);
			detector.SetImage(imageClone);
			auto start = startTimerHesaff();
			detector.detectPyramidKeypoints(imageClone, descriptors, ap, sp);
			descAvgTime+=stopTimerHesaff(start, "descriptor");
			const double parallelTime = detector.getParallelTime();
			std::cout<<"parallel time="<<parallelTime<<std::endl;
			parallelAvgTime+=parallelTime;
			std::cout<<"descriptors.rows="<<descriptors.rows<<std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(SLEEP));
		}
		std::cout<<"avg descriptor time="<<descAvgTime/runs<<std::endl;
		std::cout<<"avg parallel time="<<parallelAvgTime/runs<<std::endl;
		statsFile<<fileName<<"\t"<<nThreads<<"\t"<<descAvgTime<<"\t"<<parallelAvgTime<<"\n";
		statsFile.flush();
      }
	statsFile.close();
   } else {
      printf("\nUsage: hesaff image_name.ppm\nDetects Hessian Affine points and describes them using SIFT descriptor.\nThe detector assumes that the vertical orientation is preserved.\n\n");
   }
}
