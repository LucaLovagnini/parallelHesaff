/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include "affine.h"

#include "omp.h"

using namespace cv;

void computeGradient(const Mat &img, Mat &gradx, Mat &grady)
{
   const int width = img.cols;
   const int height = img.rows;
   for (int r = 0; r < height; ++r)
	  #pragma omp simd
      for (int c = 0; c < width; ++c) 
      {
         float xgrad, ygrad; 
         if (c == 0) xgrad = img.at<float>(r,c+1) - img.at<float>(r,c); else 
            if (c == width-1) xgrad = img.at<float>(r,c) - img.at<float>(r,c-1); else 
               xgrad = img.at<float>(r,c+1) - img.at<float>(r,c-1);
         
         if (r == 0) ygrad = img.at<float>(r+1,c) - img.at<float>(r,c); else 
            if (r == height-1) ygrad = img.at<float>(r,c) - img.at<float>(r-1,c); else
               ygrad = img.at<float>(r+1,c) - img.at<float>(r-1,c);
         
         gradx.at<float>(r,c) = xgrad;
         grady.at<float>(r,c) = ygrad;
      }
}

bool AffineShape::findAffineShape(const FindAffineShapeArgs &args, std::vector<Result> &res, SIFTDescriptor &sift, float &nTimes)
{
   float x = args.x;
   float y = args.y;
   float s = args.s;
   float pixelDistance = args.pixelDistance;
   int type = args.type;
   float response = args.response;
   const Wrapper &wrapper = args.wrapper.get();

   //std::cout<<"x="<<x<<" y="<<y<<" s="<<s<<" pixelDistance="<<pixelDistance<<" type="<<type<<" response="<<response<<" wrapper="<<wrapper.blur.rows<<std::endl;
   float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;
   float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, l1 = 1.0f, l2 = 1.0f;
   float lx = args.x/pixelDistance, ly = y/pixelDistance;
   float ratio = s/(wrapper.ap.initialSigma*pixelDistance);
   // kernel size...
   const int maskPixels = par.smmWindowSize * par.smmWindowSize;

   cv::Mat fx = cv::Mat(wrapper.ap.smmWindowSize, wrapper.ap.smmWindowSize, CV_32FC1);
   cv::Mat fy = cv::Mat(wrapper.ap.smmWindowSize, wrapper.ap.smmWindowSize, CV_32FC1);
   fx = cv::Scalar(0);
   fy = cv::Scalar(0);

   cv::Mat mask = cv::Mat(wrapper.ap.smmWindowSize, wrapper.ap.smmWindowSize, CV_32FC1);
   computeGaussMask(mask);

   cv::Mat img(wrapper.ap.smmWindowSize, wrapper.ap.smmWindowSize, CV_32FC1);

   for (int l = 0; l < wrapper.ap.maxIterations; l ++)
   {

      // warp input according to current shape matrix
      interpolate(wrapper.prevBlur, lx, ly, u11*ratio, u12*ratio, u21*ratio, u22*ratio, img);
      
      // compute SMM on the warped patch
      float a = 0, b = 0, c = 0;
      float *maskptr = mask.ptr<float>(0);
      float *pfx = fx.ptr<float>(0), *pfy = fy.ptr<float>(0);
      
      computeGradient(img, fx, fy);
      
      // estimate SMM
      for (int i = 0; i < maskPixels; ++i)
      {
         const float v = (*maskptr);
         const float gxx = *pfx;
         const float gyy = *pfy;
         const float gxy = gxx * gyy;
               
         a += gxx * gxx * v;
         b += gxy * v;
         c += gyy * gyy * v;
         pfx++; pfy++; maskptr++;
      }
      a /= maskPixels; b /= maskPixels; c /= maskPixels;

      // compute inverse sqrt of the SMM 
      invSqrt(a, b, c, l1, l2);
         
      // update eigen ratios
      eigen_ratio_bef = eigen_ratio_act;
      eigen_ratio_act = 1 - l2 / l1;
         
      // accumulate the affine shape matrix
      float u11t = u11, u12t = u12;
         
      u11 = a*u11t+b*u21; u12 = a*u12t+b*u22;
      u21 = b*u11t+c*u21; u22 = b*u12t+c*u22;
         
      // compute the eigen values of the shape matrix
      if (!getEigenvalues(u11, u12, u21, u22, l1, l2))
         break;
         
      // leave on too high anisotropy
      if ((l1/l2>6) || (l2/l1>6))
         break;
         
      if (eigen_ratio_act < wrapper.ap.convergenceThreshold && eigen_ratio_bef < wrapper.ap.convergenceThreshold)
      {
         if (affineShapeCallback){
            affineShapeCallback->onAffineShapeFound(args, u11, u12, u21, u22, res, sift, nTimes);
         }
         else
        	 std::cout<<"FALSE!"<<std::endl;
         return true;
      }
   }
   return false;
}

bool AffineShape::normalizeAffine(const Mat &img, float x, float y, float s, float a11, float a12, float a21, float a22, cv::Mat &patch, std::vector<unsigned char> &workspace
		, blurArgsT &blurArgs)
{
   // determinant == 1 assumed (i.e. isotropic scaling should be separated in mrScale
   assert( fabs(a11*a22-a12*a21 - 1.0f) < 0.01);
   float mrScale = ceil(s * par.mrSize); // half patch size in pixels of image

   int   patchImageSize = 2*int(mrScale)+1; // odd size
   float imageToPatchScale = float(patchImageSize) / float(par.patchSize);  // patch size in the image / patch size -> amount of down/up sampling

   //compute  patch.rows / (largest image side) rate. If larger than RATIOTHRESHOLD, then blurInPlace is going to be very expensive, so skip it
#ifdef RATIOTHRESHOLD
   if((float) patchImageSize / std::max(img.rows, img.cols) > RATIOTHRESHOLD){
	   return true;
   }
#endif

   // is patch touching boundary? if yes, ignore this feature
   if (interpolateCheckBorders(img, x, y, a11*imageToPatchScale, a12*imageToPatchScale, a21*imageToPatchScale, a22*imageToPatchScale, patch))
      return true;
   
   if (imageToPatchScale > 0.4)
   { 
      // the pixels in the image are 0.4 apart + the affine deformation      
      // leave +1 border for the bilinear interpolation 
      patchImageSize += 2; 
      size_t wss = patchImageSize*patchImageSize*sizeof(float);
      if (wss >= workspace.size())
    	  workspace.resize(wss);
      
      Mat smoothed(patchImageSize, patchImageSize, CV_32FC1, (void *)&workspace.front());
      // interpolate with det == 1
      if (!interpolate(img, x, y, a11, a12, a21, a22, smoothed))
      {
    	 auto start = startTimerHesaff();
         // smooth accordingly
         gaussianBlurInplace(smoothed, 1.5f*imageToPatchScale);
         float blurTime = stopTimerHesaff(start);
         blurArgs = blurArgsT(omp_get_thread_num(), blurTime, smoothed, 1.5f*imageToPatchScale);
         // subsample with corresponding scale
         interpolate(smoothed, (float)(patchImageSize>>1), (float)(patchImageSize>>1), imageToPatchScale, 0, 0, imageToPatchScale, patch);
      } else
         return true;      
   } else {
      // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
      a11 *= imageToPatchScale; a12 *= imageToPatchScale;
      a21 *= imageToPatchScale; a22 *= imageToPatchScale;
      // ok, do the interpolation
      interpolate(img, x, y, a11, a12, a21, a22, patch);
   }
   return false;
}
