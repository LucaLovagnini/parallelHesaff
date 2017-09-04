/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <sys/time.h>
#include <opencv2/opencv.hpp>

void solveLinear3x3(float *A, float *b);
bool getEigenvalues(float a, float b, float c, float d, float &l1, float &l2);
void invSqrt(float &a, float &b, float &c, float &l1, float &l2);
void computeGaussMask(cv::Mat &mask);
void computeCircularGaussMask(cv::Mat &mask);
void rectifyAffineTransformationUpIsUp(float *U);
void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22);
bool interpolate(const cv::Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, cv::Mat &res);
bool interpolateCheckBorders(const cv::Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, const cv::Mat &res);
void photometricallyNormalize(float *image, const float *binaryMask, float &sum, float &var, const int width, const int height);

cv::Mat gaussianBlur(const cv::Mat input, const float sigma);
void gaussianBlurInplace(cv::Mat &inplace, float sigma);
cv::Mat doubleImage(const cv::Mat &input);
cv::Mat halfImage(const cv::Mat &input);

float timeElapsedHesaff(const struct timeval &start);
struct timeval startTimerHesaff();
float stopTimerHesaff(const struct timeval &start, const std::string &label = std::string());
void equalMats(const cv::Mat &a, const cv::Mat &b);
   
#endif // __HELPERS_H__
