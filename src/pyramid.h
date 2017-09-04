/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#ifndef __PYRAMID_H__
#define __PYRAMID_H__

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "affine.h"
#include "siftdesc.h"

using namespace cv;

struct PyramidParams
{
   // shall input image be upscaled ( > 0)
   int upscaleInputImage;
   // number of scale per octave
   int  numberOfScales;
   // amount of smoothing applied to the initial level of first octave
   float initialSigma;
   // noise dependent threshold on the response (sensitivity)
   float threshold;
   // ratio of the eigenvalues 
   float edgeEigenValueRatio;
   // number of pixels ignored at the border of image
   int  border;
   std::vector<float> sigmas;
   PyramidParams()
   	   {
         upscaleInputImage = 0;
         numberOfScales = 3;
         initialSigma = 1.6f;
         threshold = 16.0f/3.0f; //0.04f * 256 / 3;
         edgeEigenValueRatio = 10.0f;
         border = 5;
         float sigmaStep = pow(2.0f, 1.0f / (float) numberOfScales);
         sigmas.reserve(numberOfScales+2);
         sigmas.push_back(0); //neverused
         sigmas.push_back(initialSigma);
         for(int i=2; i<numberOfScales+3; i++){
        	 sigmas.push_back(sigmas[i-1]*sigmaStep);
         }
      }
};

class HessianKeypointCallback
{
public:
   virtual void onHessianKeypointDetected(const FindAffineShapeArgs &args, std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes) = 0;
   virtual ~HessianKeypointCallback(){}
};

struct HessianDetector
{
   enum {
      HESSIAN_DARK   = 0,
      HESSIAN_BRIGHT = 1,
      HESSIAN_SADDLE = 2,
   };
public:
   HessianDetector(const PyramidParams &par) :
	  par(par),
      edgeScoreThreshold((par.edgeEigenValueRatio + 1.0f)*(par.edgeEigenValueRatio + 1.0f)/par.edgeEigenValueRatio),
      // thresholds are squared, response of det H is proportional to square of derivatives!
      finalThreshold(par.threshold * par.threshold),
      positiveThreshold(0.8 * finalThreshold),
      negativeThreshold(-positiveThreshold)
      {
         hessianKeypointCallback = 0;
      }
   void setHessianKeypointCallback(HessianKeypointCallback *callback)
      {
         hessianKeypointCallback = callback;
      }
   void detectPyramidKeypoints(const Mat &image, cv::Mat &descriptors, const AffineShapeParams ap, const SIFTDescriptorParams sp);
   virtual double getParallelTime();
   ~HessianDetector();
   HessianKeypointCallback *hessianKeypointCallback;
   PyramidParams par;
protected:   
   void localizeKeypoint(int r, int c, float curScale, float pixelDistance, Wrapper &wrapper);
   void findLevelKeypoints(float curScale, float pixelDistance , Wrapper &wrapper);
   Mat hessianResponse(const Mat &inputImage, float norm);
   double parallelTime = 0;
private:
   // some constants derived from parameters
   const float edgeScoreThreshold;
   const float finalThreshold;
   const float positiveThreshold;
   const float negativeThreshold;
   // temporary arrays used by protected functions
   //Mat octaveMap;
};

#endif // __PYRAMID_H__
