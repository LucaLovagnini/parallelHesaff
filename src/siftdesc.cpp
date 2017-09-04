/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <vector>
#include "siftdesc.h"
#include "helpers.h"

using namespace std;
using namespace cv;

// The SIFT descriptor is subject to US Patent 6,711,293


void SIFTDescriptor::precomputeBinsAndWeights()
{
   int halfSize = par.patchSize>>1;
   float step = float(par.spatialBins+1)/(2*halfSize);
   
   // allocate maps at the same location
   precomp_bins.resize(2*par.patchSize);
   precomp_weights.resize(2*par.patchSize);
   bin1 = bin0 = &precomp_bins.front(); bin1 += par.patchSize;
   w1   =   w0 = &precomp_weights.front(); w1 += par.patchSize;

   // maps every pixel in the patch 0..patch_size-1 to appropriate spatial bin and weight
   for (int i = 0; i < par.patchSize; i++)
   {
      float x = step*i;      // x goes from <-1 ... spatial_bins> + 1
      int  xi = (int)(x);
      // bin indices
      bin0[i] = xi-1; // get real xi
      bin1[i] = xi;
      // weights
      w1[i]   = x - xi;
      w0[i]   = 1.0f - w1[i];
      // truncate weights and bins in case they reach outside of valid range
      if (bin0[i] <          0) { bin0[i] = 0;           w0[i] = 0; }
      if (bin0[i] >= par.spatialBins) { bin0[i] = par.spatialBins-1; w0[i] = 0; }
      if (bin1[i] <          0) { bin1[i] = 0;           w1[i] = 0; }
      if (bin1[i] >= par.spatialBins) { bin1[i] = par.spatialBins-1; w1[i] = 0; }
      // adjust for orientation bin skip
      bin0[i] *= par.orientationBins;
      bin1[i] *= par.orientationBins;
   }
}

void SIFTDescriptor::samplePatch(float *vec)
{
   float vec000[128] = {0};
   float vec001[128] = {0};
   float vec010[128] = {0};
   float vec011[128] = {0};
   float vec100[128] = {0};
   float vec101[128] = {0};
   float vec110[128] = {0};
   float vec111[128] = {0};

   for (int r = 0; r < par.patchSize; ++r)
   {
      const int br0 = par.spatialBins * bin0[r]; const float wr0 = w0[r];
      const int br1 = par.spatialBins * bin1[r]; const float wr1 = w1[r];
	  #pragma omp simd
      for (int c = 0; c < par.patchSize; ++c)
      {
         float val = mask.at<float>(r,c) * grad.at<float>(r,c);

         const int bc0 = bin0[c]; const float wc0 = w0[c]*val;
         const int bc1 = bin1[c]; const float wc1 = w1[c]*val;

         // ori from atan2 is in range <-pi,pi> so add 2*pi to be surely above zero
         const float o = float(par.orientationBins)*(ori.at<float>(r,c) + 2*M_PI)/(2*M_PI);

         int   bo0 = (int)o;
         const float wo1 =  o - bo0;
         bo0 %= par.orientationBins;

         int   bo1 = (bo0+1) % par.orientationBins;
         const float wo0 = 1.0f - wo1;

         // add to corresponding 8 vec...
#pragma distribute_point
         val = wr0*wc0;
         if (val>0) {
        	 vec000[br0+bc0+bo0] += val * wo0;
        	 vec001[br0+bc0+bo1] += val * wo1;
         }
#pragma distribute_point
         val = wr0*wc1;
         if (val>0) {
        	 vec010[br0+bc1+bo0] += val * wo0;
        	 vec011[br0+bc1+bo1] += val * wo1;
         }
#pragma distribute_point
         val = wr1*wc0;
         if (val>0) {
        	 vec100[br1+bc0+bo0] += val * wo0;
        	 vec101[br1+bc0+bo1] += val * wo1;
         }
#pragma distribute_point
         val = wr1*wc1;
         if (val>0) {
        	 vec110[br1+bc1+bo0] += val * wo0;
        	 vec111[br1+bc1+bo1] += val * wo1;
         }
      }
   }
   #pragma omp simd
   for(int i=0; i<128; i++)
	   vec[i]+=vec000[i]+vec001[i]+
	   	   	   vec010[i]+vec011[i]+
			   vec100[i]+vec101[i]+
			   vec110[i]+vec111[i];
}

void SIFTDescriptor::sample(float* vec)
{
   samplePatch(vec);
   cv::Mat1f matVec(1,128, vec);
   // accumulate histograms
   cv::normalize(matVec,matVec);
   // check if there are some values above threshold
   bool changed = false;
   for (size_t i = 0; i < 128; i++) if (vec[i] > par.maxBinValue) { vec[i] = par.maxBinValue; changed = true; }
   if (changed) cv::normalize(matVec,matVec);

   for (size_t i = 0; i < 128; i++)
   {
      int b  = min((int)(512.0f * vec[i]), 255);
      vec[i] = float(b);
   }
}

void SIFTDescriptor::computeSiftDescriptor(float *patch, float *vec, const int width, const int height)
{
   // photometrically normalize with weights as in SIFT gradient magnitude falloff
   float mean, var;
   float meanTest = mean;
   float varTest = var;
   //std::cout<<"mask row="<<mask.rows<<" cols="<<mask.cols<<std::endl;
   photometricallyNormalize(patch, mask.ptr<float>(0), mean, var, width, height);
   // prepare gradients
   for (int r = 0; r < height; ++r)
      for (int c = 0; c < width; ++c)
      {
         float xgrad, ygrad;
         if (c == 0) xgrad = patch[r*width+c+1] - patch[r*width+c]; else
            if (c == width-1) xgrad = patch[r*width+c] - patch[r*width+c-1]; else
               xgrad = patch[r*width+c+1] - patch[r*width+c-1];

         if (r == 0) ygrad = patch[(r+1)*width+c] - patch[r*width+c]; else
            if (r == height-1) ygrad = patch[r*width+c] - patch[(r-1)*width+c]; else
               ygrad = patch[(r+1)*width+c] - patch[(r-1)*width+c];

         grad.at<float>(r,c) = ::sqrt(xgrad * xgrad + ygrad * ygrad);
         ori.at<float>(r,c) = atan2(ygrad, xgrad);
      }
   // compute SIFT vector
   sample(vec);
}
