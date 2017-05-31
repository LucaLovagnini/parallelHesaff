/*
 * FindAffineShaperArgs.h
 *
 *  Created on: 21 Apr 2017
 *      Author: luca
 */

#ifndef DESCRIPTORS_HESAFF_FINDAFFINESHAPERARGS_H_
#define DESCRIPTORS_HESAFF_FINDAFFINESHAPERARGS_H_

#include <functional>

struct Wrapper;

struct FindAffineShapeArgs
{
	FindAffineShapeArgs(float x, float y, float s, float pixelDistance, float type, float response, const Wrapper &wrapper) :
		x(x), y(y), s(s), pixelDistance(pixelDistance), type(type), response(response), wrapper(std::cref(wrapper)) {}

	FindAffineShapeArgs() : wrapper(*(const Wrapper *)nullptr) {
	}

	float x=0, y=0, s=0;
	float pixelDistance=0, type=0, response=0;
	std::reference_wrapper<Wrapper const> wrapper;
};

#endif /* DESCRIPTORS_HESAFF_FINDAFFINESHAPERARGS_H_ */
