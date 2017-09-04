/*
 * hesaff.hpp
 *
 *  Created on: 13 Apr 2017
 *      Author: luca
 */

#ifndef DESCRIPTORS_HESAFF_HESAFF_HPP_
#define DESCRIPTORS_HESAFF_HESAFF_HPP_

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"
#include "Wrapper.h"

using namespace cv;
using namespace std;

struct HessianAffineParams
{
   float threshold;
   int   max_iter;
   float desc_factor;
   int   patch_size;
   bool  verbose;
   HessianAffineParams()
      {
         threshold = 16.0f/3.0f;
         max_iter = 16;
         desc_factor = 3.0f*sqrt(3.0f);
         patch_size = 41;
         verbose = false;
      }
};

struct AffineHessianDetector : HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
	   Mat image;
	   SIFTDescriptorParams sp;
	   AffineShapeParams ap;
	   int patchSize;
	public:
	   AffineHessianDetector(const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) :
	      HessianDetector(par),
	      AffineShape(ap),
		  patchSize(ap.patchSize),
		  sp(sp),
		  ap(ap)
	      {
	         this->setHessianKeypointCallback(this);
	         this->setAffineShapeCallback(this);
	      }

	   void SetImage(const cv::Mat&image) { this->image = image; }

	   ~AffineHessianDetector(){}

	   void onHessianKeypointDetected(const FindAffineShapeArgs &args, std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes)
	      {
	         findAffineShape(args, res, sift, nTimes);
	      }


	   void onAffineShapeFound(
	   	  const FindAffineShapeArgs &args,
		 float a11, float a12, // affine shape matrix
		 float a21, float a22,
		 std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes)
	      {
		   	 auto start = startTimerHesaff();
	         // convert shape into a up is up frame
	         rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);
	         cv::Mat patch(patchSize, patchSize, CV_32FC1);
	         std::vector<unsigned char> workspace;
	         auto startTime = startTimerHesaff();
	         blurArgsT blurArgs;
	         bool result = normalizeAffine(image, args.x, args.y, args.s, a11, a12, a21, a22, patch, workspace, blurArgs);
	         nTimes += stopTimerHesaff(startTime);
	         // now sample the patch
	         if (!result)
	         {
	        	 res.push_back(Result(blurArgs));
	            // compute SIFT
	            sift.computeSiftDescriptor(patch.ptr<float>(0), res.back().descriptor.ptr<float>(0), patch.cols, patch.rows);
	         }
	      }
};



#endif /* DESCRIPTORS_HESAFF_HESAFF_HPP_ */
