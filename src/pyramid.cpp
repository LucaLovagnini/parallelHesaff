/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <vector>
#include <string.h>
#include <algorithm>
#include <tuple>
#include <thread>
#include <chrono>
#include <deque>
#include <tbb/concurrent_queue.h>
#include "pyramid.h"
#include "helpers.h"

#include "omp.h"

#define OMP

#define VERBOSE

using namespace std;

/* find blob point type from Hessian matrix H, 
   we know that:
   - if H is positive definite it is a DARK blob,
   - if H is negative definite it is a BRIGHT blob
   - det H is negative it is a SADDLE point
*/
int getHessianPointType(const float *ptr, float value)
{
   if (value < 0)
      return HessianDetector::HESSIAN_SADDLE;
   else {
      // at this point we know that 2x2 determinant is positive
      // so only check the remaining 1x1 subdeterminant
      float Lxx = (ptr[-1]-2*ptr[0]+ptr[1]);
      if (Lxx < 0)
         return HessianDetector::HESSIAN_DARK;
      else
         return HessianDetector::HESSIAN_BRIGHT;
   }      
}

bool isMax(float val, const Mat &pix, int row, int col)
{   
   for (int r = row - 1; r <= row + 1; r++)
   {
      const float *row = pix.ptr<float>(r);
      for (int c = col - 1; c <= col + 1; c++)
         if (row[c] > val)
            return false;
   }
   return true;
}
   
bool isMin(float val, const Mat &pix, int row, int col)
{
   for (int r = row - 1; r <= row + 1; r++)
   {
      const float *row = pix.ptr<float>(r);
      for (int c = col - 1; c <= col + 1; c++)
         if (row[c] < val)
            return false;
   }
   return true;
}

Mat HessianDetector::hessianResponse(const Mat &inputImage, float norm)
{
   const int rows = inputImage.rows;
   const int cols = inputImage.cols;
   const int stride = cols;

   // allocate output
   Mat outputImage(rows, cols, CV_32FC1);
   
   // setup input and output pointer to be centered at 1,0 and 1,1 resp.
   float      *out = outputImage.ptr<float>(1) + 1;
   const float *in = inputImage.ptr<float>(0);
   float norm2 = norm * norm;
   
   /* move 3x3 window and convolve */
   for (int r = 1; r < rows - 1; ++r)
   {
	  #pragma omp simd
      for (int c = 1; c < cols - 1; c++)
      {
         /* fetch remaining values (last column) */
		 float v11 = in[(r-1)*cols+(c-1)];
		 float v12 = in[(r-1)*cols+c];
		 float v13 = in[(r-1)*cols+c+1];

		 float v21 = in[r*cols+(c-1)];
		 float v22 = in[r*cols+c];
		 float v23 = in[r*cols+c+1];

		 float v31 = in[(r+1)*cols+c-1];
		 float v32 = in[(r+1)*cols+c];
		 float v33 = in[(r+1)*cols+c+1];
         
         // compute 3x3 Hessian values from symmetric differences.
         float Lxx = (v21 - 2*v22 + v23);
         float Lyy = (v12 - 2*v22 + v32);
         float Lxy = (v13 - v11 + v31 - v33)/4.0f;
         
         /* normalize and write out */
         *out = (Lxx * Lyy - Lxy * Lxy)*norm2;

         /* move input/output pointers */
         out++;
      }
      out += 2;
   }
   return outputImage;
}

// it seems 0.6 works better than 0.5 (as in DL paper)
#define MAX_SUBPIXEL_SHIFT 0.6

// we don't care about border effects
#define POINT_SAFETY_BORDER  3

void HessianDetector::localizeKeypoint(int r, int c, float curScale, float pixelDistance, Wrapper &wrapper)
{
   const int cols = wrapper.cur.cols;
   const int rows = wrapper.cur.rows;

   const float *curPtr = wrapper.cur.ptr<float>(0);
   const float *lowPtr = wrapper.low.ptr<float>(0);
   const float *highPtr = wrapper.high.ptr<float>(0);

   float b[3] = {};
   float val = 0;
   //bool converged = false;
   int nr = r, nc = c;
   
   for (int iter=0; iter<5; iter++)
   {
      // take current position
      r = nr; c = nc;
      
      float dxx = curPtr[r*cols+c-1] - 2.0f * curPtr[r*cols+c] + curPtr[r*cols+c+1];
      float dyy = curPtr[(r-1)*cols+c] - 2.0f * curPtr[r*cols+c] + curPtr[(r+1)*cols+c];
      float dss = lowPtr[r*cols+c] - 2.0f * curPtr[r*cols+c] + highPtr[r*cols+c];
         
      float dxy = 0.25f*(curPtr[(r+1)*cols+c+1] - curPtr[(r+1)*cols+c-1] - curPtr[(r-1)*cols+c+1] + curPtr[(r-1)*cols+c-1]);
      // check edge like shape of the response function in first iteration
      if (0 == iter)
      {
         float edgeScore = (dxx + dyy)*(dxx + dyy)/(dxx * dyy - dxy * dxy);
         if (edgeScore >= edgeScoreThreshold || edgeScore < 0)
            // local neighbourhood looks like an edge
            return;
      }
      float dxs = 0.25f*(highPtr[(r)*cols+c+1] - highPtr[(r)*cols+c-1] - lowPtr[(r)*cols+c+1] + lowPtr[(r)*cols+c-1]);
      float dys = 0.25f*(highPtr[(r+1)*cols+c] - highPtr[(r-1)*cols+c] - lowPtr[(r+1)*cols+c] + lowPtr[(r-1)*cols+c]);
         
      float A[9];
      A[0] = dxx; A[1] = dxy; A[2] = dxs;
      A[3] = dxy; A[4] = dyy; A[5] = dys;
      A[6] = dxs; A[7] = dys; A[8] = dss;

      float dx = 0.5f*(curPtr[r*cols+c+1] - curPtr[r*cols+c-1]);
      float dy = 0.5f*(curPtr[(r+1)*cols+c] - curPtr[(r-1)*cols+c]);
      float ds = 0.5f*(highPtr[(r)*cols+c]  - lowPtr[r*cols+c]);
         
      b[0] = - dx; b[1] = - dy; b[2] = - ds;
         
      solveLinear3x3(A, b);
         
      // check if the solution is valid
      if (isnan(b[0]) || isnan(b[1]) || isnan(b[2]))
         return;
         
      // aproximate peak value
      val = curPtr[r*cols+c] + 0.5f * (dx*b[0] + dy*b[1] + ds*b[2]);
               
      // if we are off by more than MAX_SUBPIXEL_SHIFT, update the position and iterate again      
      if (b[0] >  MAX_SUBPIXEL_SHIFT) { if (c < cols - POINT_SAFETY_BORDER) nc++; else return; }
      if (b[1] >  MAX_SUBPIXEL_SHIFT) { if (r < rows - POINT_SAFETY_BORDER) nr++; else return; }
      if (b[0] < -MAX_SUBPIXEL_SHIFT) { if (c >        POINT_SAFETY_BORDER) nc--; else return; }
      if (b[1] < -MAX_SUBPIXEL_SHIFT) { if (r >        POINT_SAFETY_BORDER) nr--; else return; }
      
      if (nr == r && nc == c) 
      {
         // converged, displacement is sufficiently small, terminate here
         // TODO: decide if we want only converged local extrema...
         //converged = true;
         break;
      }
   }
      
   // if spatial localization was all right and the scale is close enough...
   if (fabs(b[0]) > 1.5 || fabs(b[1]) > 1.5 || fabs(b[2]) > 1.5 || fabs(val) < finalThreshold /*|| octaveMap.at<unsigned char>(r,c) > 0*/)
      return;
      
   // mark we were here already
   //octaveMap.at<unsigned char>(r,c) = 1;

   // output keypoint
   float scale = curScale * pow(2.0f, b[2] / par.numberOfScales );
   // set point type according to final location
   int type = getHessianPointType(wrapper.blur.ptr<float>(r)+c, val);
   // point is now scale and translation invariant, add it...
   wrapper.findAffineShapeArgs.push_back(FindAffineShapeArgs(pixelDistance*(c + b[0]), pixelDistance*(r + b[1]), pixelDistance*scale, pixelDistance, type, val, wrapper));
   return;
   //findAffineShape(prevBlur, pixelDistance*(c + b[0]), pixelDistance*(r + b[1]), pixelDistance*scale, pixelDistance, type, val, descriptors, keys);
}

void HessianDetector::findLevelKeypoints(float curScale, float pixelDistance, Wrapper &wrapper)
{
	//auto startFindLevelKeypoints = startTimerHesaff();
   const int rows = wrapper.cur.rows;
   const int cols = wrapper.cur.cols;
   const float *curPtr = wrapper.cur.ptr<float>(0);
   for (int r = par.border; r < (rows - par.border); r++)
   {
      for (int co = par.border; co < (cols - par.border); co++)
      {
         const float val = curPtr[r*cols+co];
         if ( (val > positiveThreshold && (isMax(val, wrapper.cur, r, co) && isMax(val, wrapper.low, r, co) && isMax(val, wrapper.high, r, co))) ||
              (val < negativeThreshold && (isMin(val, wrapper.cur, r, co) && isMin(val, wrapper.low, r, co) && isMin(val, wrapper.high, r, co))) )
            // either positive -> local max. or negative -> local min.
			localizeKeypoint(r, co, curScale, pixelDistance, wrapper);
      }
   }
   //wrapper.findLevelKeypointsTime = stopTimerHesaff(startFindLevelKeypoints);
}

bool compareBlurArgsT(const blurArgsT &left, const blurArgsT &right){
	return std::get<1>(left) < std::get<1>(right);
}
   
void HessianDetector::detectPyramidKeypoints(const Mat &image, cv::Mat &descriptors, const AffineShapeParams ap, const SIFTDescriptorParams sp)
{
   float curSigma = 0.5f;
   float pixelDistance = 1.0f;
   cv::Mat octaveLayer;

   // prepare first octave input image
   if (par.initialSigma > curSigma)
   {
      float sigma = sqrt(par.initialSigma * par.initialSigma - curSigma * curSigma);
      octaveLayer = gaussianBlur(image, sigma);
   }
   
   // while there is sufficient size of image
   int minSize = 2 * par.border + 2;
   int rowsCounter = image.rows;
   int colsCounter = image.cols;
   float sigmaStep = pow(2.0f, 1.0f / (float) par.numberOfScales);
   int levels = 0;
   while (rowsCounter > minSize && colsCounter > minSize){
	   rowsCounter/=2; colsCounter/=2;
	   levels++;
   }
   int scaleCycles = par.numberOfScales+2;

   std::vector<blurArgsT> blurArgs;

   //-------------------Shared Vectors-------------------
	std::vector<Mat> blurs (scaleCycles*levels+1, Mat());
	std::vector<Mat> hessResps (levels*scaleCycles+2); //+2 because high needs an extra one
	std::vector<Wrapper> localWrappers;
   	tbb::concurrent_bounded_queue<FindAffineShapeArgs> concfindAffineShapeArgs;
	localWrappers.resize(levels*(scaleCycles-2));
	vector<float> pixelDistances;
	pixelDistances.reserve(levels);
	bool aborted = false;

	for(int i=0; i<levels; i++){
	   pixelDistances.push_back(pixelDistance);
	   pixelDistance*=2;
	}

   //compute blurs at all layers (not parallelizable)
   for(int i=0; i<levels; i++){
	   blurs[i*scaleCycles+1] = octaveLayer.clone();
	   for (int j = 1; j < scaleCycles; j++){
		   float sigma = par.sigmas[j]* sqrt(sigmaStep * sigmaStep - 1.0f);
		   blurs[j+1+i*scaleCycles] = gaussianBlur(blurs[j+i*scaleCycles], sigma);
		   if(j == par.numberOfScales)
			   octaveLayer = halfImage(blurs[j+1+i*scaleCycles]);
	   }
   }

   auto parallelRegionTime = startTimerHesaff();
   double maxHessianTime = 0, maxBarrierTime = 0, maxFindLevelKeypointTime = 0, maxHessianKeypointDetecetedTime = 0;
   int computingKeypoints;
   size_t resultIndex = 0;
   #pragma omp parallel
   {

	#pragma omp single
	computingKeypoints = omp_get_num_threads();


	auto start = startTimerHesaff();
   //compute all the hessianResponses
	#pragma omp for collapse(2) schedule(dynamic)
	for(int i=0; i<levels; i++)
		for (int j = 1; j <= scaleCycles; j++)
		{
			int scaleCyclesLevel = scaleCycles * i;
			float curSigma = par.sigmas[j];
			hessResps[j+scaleCyclesLevel] = hessianResponse(blurs[j+scaleCyclesLevel], curSigma*curSigma);
		}
	double hessianRespTime = stopTimerHesaff(start);
	std::vector<FindAffineShapeArgs> localfindAffineShapeArgs;
	start = startTimerHesaff();
	#pragma omp for collapse(2) schedule(dynamic) nowait
	for(int i=0; i<levels; i++)
		for (int j = 2; j < scaleCycles; j++){
			int scaleCyclesLevel = scaleCycles * i;
			size_t c = (scaleCycles-2) * i +j-2;
			localWrappers[c] = Wrapper(sp, ap, hessResps[j+scaleCyclesLevel-1], hessResps[j+scaleCyclesLevel], hessResps[j+scaleCyclesLevel+1],
					blurs[j+scaleCyclesLevel-1], blurs[j+scaleCyclesLevel]);
			//toDo: octaveMap is shared, need synchronization
			//if(j==1)
			//	octaveMap = Mat::zeros(blurs[scaleCyclesLevel+1].rows, blurs[scaleCyclesLevel+1].cols, CV_8UC1);
			float curSigma = par.sigmas[j];
			// find keypoints in this part of octave for curLevel
			findLevelKeypoints(curSigma, pixelDistances[i], localWrappers[c]);
			localfindAffineShapeArgs.insert(localfindAffineShapeArgs.end(), localWrappers[c].findAffineShapeArgs.begin(), localWrappers[c].findAffineShapeArgs.end());
		}
	double findLevelKeypointTime = stopTimerHesaff(start);

	#pragma omp atomic
	computingKeypoints--;

	for(size_t i=0; i<localfindAffineShapeArgs.size(); i++){
		concfindAffineShapeArgs.push(localfindAffineShapeArgs[i]);
	}

	std::vector<Result> localRes;
	Wrapper toyWrapper;//used only to initialize singleArgs
	FindAffineShapeArgs singleArg;
	start = startTimerHesaff();
	SIFTDescriptor sift(sp);
	start = startTimerHesaff();
	int keypointsCalls=0;
	float nTimes=0;
	try{
		do {
			bool hasIndex, allThreadsDone;
			concfindAffineShapeArgs.pop(singleArg);
			hessianKeypointCallback->onHessianKeypointDetected(singleArg, localRes, sift, nTimes);
			keypointsCalls ++;
		}
		while (concfindAffineShapeArgs.size()>0 || computingKeypoints>0);
	}
    catch(...){std::cout<<omp_get_thread_num()<<" exception caught!"<<std::endl;}
	double hessianKeypointDetecetedTime = stopTimerHesaff(start);

	#pragma omp critical
	{
		maxHessianTime = std::max(maxHessianTime, hessianRespTime);
		maxFindLevelKeypointTime = std::max(maxFindLevelKeypointTime, findLevelKeypointTime);
		maxHessianKeypointDetecetedTime = std::max(maxHessianKeypointDetecetedTime, hessianKeypointDetecetedTime);
		#ifdef VERBOSE
		std::cout<<omp_get_thread_num()<<": hessResp="<<hessianRespTime<<" findLevelKeypoint="<<findLevelKeypointTime<<" hessianKeypointDeteceted="<<hessianKeypointDetecetedTime<<" keypointsCalls="<<keypointsCalls<<" nTimes="<<nTimes<<std::endl;
		#endif
		for(size_t i=0; i<localRes.size(); i++){
			descriptors.push_back(localRes[i].descriptor);
			blurArgs.push_back(localRes[i].blurArgs);
		}
	}

	if(concfindAffineShapeArgs.size()<0 && !aborted && !computingKeypoints){
		concfindAffineShapeArgs.abort();
		aborted=true;
	}
   }//end of parallel region

   parallelTime = stopTimerHesaff(parallelRegionTime);

	#ifdef VERBOSE
   //sort blurArgs according to time
   std::sort(blurArgs.begin(), blurArgs.end(), compareBlurArgsT);
   int maxSide=0;
   for(size_t i=blurArgs.size()-10; i<blurArgs.size(); i++){
	   blurArgsT args = blurArgs[i];
	   std::cout<<"time="<<get<1>(args)<<" thread="<<get<0>(args)<<" smoothed="<<get<2>(args).rows<<"x"<<get<2>(args).cols<<" sigma="<<get<3>(args)<<
			   " ratio="<<(float) get<2>(args).rows/std::max(image.rows, image.cols)<<std::endl;
	   maxSide = std::max(maxSide, get<2>(args).rows);
   }
   std::cout<<"img="<<image.rows<<"x"<<image.cols<<" maxSide="<<maxSide<<" ratio="<<(float) maxSide/std::max(image.rows, image.cols)<<std::endl;
   std::cout<<"parallelRegionTime="<<parallelTime<<" totalMax="<<maxHessianTime+maxBarrierTime+maxFindLevelKeypointTime+maxHessianKeypointDetecetedTime<<std::endl;
   #endif
}

double HessianDetector::getParallelTime(){
	return parallelTime;
}

HessianDetector::~HessianDetector(){}
