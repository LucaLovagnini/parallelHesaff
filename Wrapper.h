/*
 * wrapper.h
 *
 *  Created on: 13 Apr 2017
 *      Author: luca
 */

#ifndef DESCRIPTORS_HESAFF_WRAPPER_H_
#define DESCRIPTORS_HESAFF_WRAPPER_H_

#include <opencv2/opencv.hpp>
#include "siftdesc.h"

#include "FindAffineShaperArgs.h"

struct Keypoint
{
	Keypoint(float x, float y, float s,
			float a11, float a12, float a21, float a22,
			float response, float type) :
			x(x), y(y), s(s), a11(a11), a12(a12),
			a21(a21), a22(a22), response(response), type(type){}
   float x, y, s;
   float a11,a12,a21,a22;
   float response;
   int type;
};

typedef std::tuple<int, float, cv::Mat, float> blurArgsT;

struct Result{
	Result(blurArgsT &blurArgs) : blurArgs(blurArgs) {
		descriptor = cv::Mat1f::zeros(1,128);
	}
	cv::Mat1f descriptor;
	blurArgsT blurArgs;
};

struct FindAffineShapeArgs;

struct AffineShapeParams
{
   // number of affine shape interations
   int maxIterations;

   // convergence threshold, i.e. maximum deviation from isotropic shape at convergence
   float convergenceThreshold;

   // widht and height of the SMM mask
   int smmWindowSize;

   // width and height of the patch
   int patchSize;

   // amount of smoothing applied to the initial level of first octave
   float initialSigma;

   // size of the measurement region (as multiple of the feature scale)
   float mrSize;

   AffineShapeParams()
      {
         maxIterations = 16;
         initialSigma = 1.6f;
         convergenceThreshold = 0.05;
         patchSize = 41;
         smmWindowSize = 19;
         mrSize = 3.0f*sqrt(3.0f);
      }
};

struct Wrapper{

	Wrapper(){}

	Wrapper(const SIFTDescriptorParams &sp, const AffineShapeParams ap) :
		ap(ap),
		sp(sp),
		low(cv::Mat()),
		cur(cv::Mat()),
		high(cv::Mat()),
		prevBlur(cv::Mat()),
		blur(cv::Mat())
	{
		//findAffineShapeArgs.reserve(500);
	}

	Wrapper(const SIFTDescriptorParams &sp, const AffineShapeParams ap, const cv::Mat &low, const cv::Mat &cur, const cv::Mat &high,
			const cv::Mat &prevBlur, const cv::Mat &blur) :
			ap(ap),
			sp(sp),
			low(low),
			cur(cur),
			high(high),
			prevBlur(prevBlur),
			blur(blur)
	{
		findAffineShapeArgs.reserve(500);
	}

	AffineShapeParams ap;
	SIFTDescriptorParams sp;

	cv::Mat high;
	cv::Mat prevBlur;
	cv::Mat blur;
	cv::Mat low;
	cv::Mat cur;

    float descTime = 0;
    float findLevelKeypointsTime = 0;

    std::vector<FindAffineShapeArgs> findAffineShapeArgs;
};


#endif /* DESCRIPTORS_HESAFF_WRAPPER_H_ */
