/*
 * onAffineShapeFound.hpp
 *
 *  Created on: 20 May 2017
 *      Author: luca
 */

#ifndef DESCRIPTORS_HESAFF_ONAFFINESHAPEFOUND_HPP_
#define DESCRIPTORS_HESAFF_ONAFFINESHAPEFOUND_HPP_

class onAffineShapeFoundArgs{
public:
	onAffineShapeFoundArgs(float x, float y, float s, float u11, float u12, float u21, float u22):
		x(x), y(y), s(s), u11(u11), u12(u12), u21(u21), u22(u22){}
	float x,y,s;
	float u11, u12, u21, u22;
};



#endif /* DESCRIPTORS_HESAFF_ONAFFINESHAPEFOUND_HPP_ */
