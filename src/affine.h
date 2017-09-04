/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#ifndef __AFFINE_H__
#define __AFFINE_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include "helpers.h"
#include "Wrapper.h"
#include "siftdesc.h"


struct AffineShapeCallback
{
   virtual void onAffineShapeFound(
	  const FindAffineShapeArgs &args,
      float a11, float a12, // affine shape matrix 
      float a21, float a22, 
      std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes) = 0;
   virtual ~AffineShapeCallback(){}
};

struct AffineShape
{
public:   
   AffineShape(const AffineShapeParams &par)
	  {
         this->par = par;
         affineShapeCallback = 0;
      }
   
   ~AffineShape()
      {
      }
   
   // computes affine shape 
   bool findAffineShape(const FindAffineShapeArgs &args, std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes);

   // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times
   bool normalizeAffine(const cv::Mat &img, float x, float y, float s, float a11, float a12, float a21, float a22, cv::Mat &patch, std::vector<unsigned char> &workspace,
		   blurArgsT &blurArgs);

   void setAffineShapeCallback(AffineShapeCallback *callback)
      {
         affineShapeCallback = callback;
      }

protected:
   AffineShapeParams par;

private:
   AffineShapeCallback *affineShapeCallback;
};

#endif // __AFFINE_H__
