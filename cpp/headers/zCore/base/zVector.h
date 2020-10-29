// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#ifndef ZSPACE_ZVECTOR_H
#define ZSPACE_ZVECTOR_H

#pragma once

#include <stdexcept>
#include <vector>
using namespace std;

#include "headers/zCore/base/zDefinitions.h"
#include "headers/zCore/base/zMatrix.h"

namespace  zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zVector
	*	\brief A 3 dimensional vector math class.
	*	\since version 0.0.1
	*/

	/** @}*/
	/** @}*/

	class ZSPACE_CORE zVector
	{
	protected:

	public:
		/*!	\brief x component				*/
		float x;

		/*!	\brief y component				*/
		float y;

		/*!	\brief z component				*/
		float z;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_x		- x component of the zVector.
		*	\param		[in]	_z		- y component of the zVector.
		*	\param		[in]	_z		- z component of the zVector.
		*	\param		[in]	_w		- w component of the zVector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector(float _x, float _y, float _z, float _w = 1.0);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	vals	- input array of values.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector(zFloat3 &_vals);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	vals	- input array of values.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector(zFloat4 &_vals);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zVector();

		//--------------------------
		//---- OPERATORS
		//--------------------------	

		/*! \brief This operator checks for equality of two zVectors.
		*
		*	\param		[in]	v1		- zVector against which the equality is checked.
		*	\return				bool	- true if vectors are equal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool operator==(const zVector &v1);

		/*! \brief This method returns the component value of the current zVector.
		*
		*	\param		[in]	index		- index. ( 0 - x component, 1 - y component, 2 - z component, 3 - w component).
		*	\return				float		- value of the component.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float  operator[](int index);

		/*! \brief This operator is used for vector addition.
		*
		*	\param		[in]	v1		- zVector which is added to the current vector.
		*	\return				zVector	- resultant vector after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator+(const zVector &v1);

		/*! \brief This operator is used for vector subtraction.
		*
		*	\param		[in]	v1		- zVector which is subtracted from the current vector.
		*	\return				zVector	- resultant vector after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator -(const zVector &v1);

		/*! \brief This operator is used for vector dot product.
		*
		*	\param		[in]	v1		- zVector which is used for the dot product with the current vector.
		*	\return				float	- resultant value after the dot product.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float operator *(const zVector &v1);

		/*! \brief This operator is used for vector cross procduct.
		*
		*	\param		[in]	v1		- zVector which is used for the cross product with the current vector.
		*	\return				zVector	- resultant vector after the cross product.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator ^(const zVector &v1);

		/*! \brief This operator is used for scalar addition of a vector.
		*
		*	\param		[in]	val		- scalar value to be added to the current vector.
		*	\return				zVector	- resultant vector after the scalar addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator +(float val);

		/*! \brief This operator is used for scalar subtraction of a vector.
		*
		*	\param		[in]	val		- scalar value to be subtracted from the current vector.
		*	\return				zVector	- resultant vector after the scalar subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator -(float val);

		/*! \brief This operator is used for scalar muliplication of a vector.
		*
		*	\param		[in]	val		- scalar value to be multiplied with the current vector.
		*	\return				zVector	- resultant vector after the scalar multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator *(float val);

		/*! \brief This operator is used for 4x4 matrix muliplication of a vector.
		*
		*	\param		[in]	inMatrix	- input 4X4 matrix to be multiplied with the current vector.
		*	\return				zVector		- resultant vector after the matrix multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator*(zMatrix4 &inMatrix);

		/*! \brief This operator is used for 3X3 matrix muliplication of a vector.
		*
		*	\param		[in]	inMatrix	- input 3X3 matrix to be multiplied with the current vector.
		*	\return				zVector		- resultant vector after the matrix multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator*(zMatrix3 &inMatrix);

		/*! \brief This operator is used for scalar division of a vector.
		*
		*	\param		[in]	val		- scalar value used to divide the current vector.
		*	\return				zVector	- resultant vector after the scalar division.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector operator /(float val);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This overloaded operator is used for vector addition and assigment of the result to the current vector.
		*
		*	\param		[in]	v1		- zVector which is added to the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator +=(const zVector &v1);

		/*! \brief This overloaded operator is used for vector subtraction and assigment of the result to the current vector.
		*
		*	\param		[in]	v1		- zVector which is subtacted from the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator -=(const zVector &v1);

		/*! \brief This overloaded operator is used for scalar addition and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be added to the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator +=(float val);

		/*! \brief This overloaded operator is used for scalar subtraction and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be subtracted from the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator -=(float val);

		/*! \brief This overloaded operator is used for scalar multiplication and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be multiplied to the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator *=(float val);

		/*! \brief This overloaded operator is used for scalar division and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value used to divide from the current vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator /=(float val);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method returns the squared length of the zVector.
		*
		*	\return				float		- value of the squared maginute of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float length2();

		/*! \brief This method returns the magnitude/length of the zVector.
		*
		*	\return				float		- value of the maginute of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float length();

		/*! \brief This method normalizes the vector to unit length.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void normalize();

		/*! \brief This method returns the squared distance between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				float		- squared value of the distance between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float squareDistanceTo(zVector &v1);

		/*! \brief This method returns the distance between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				float		- value of the distance between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float distanceTo(zVector &v1);

		/*! \brief This method returns the angle between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				float		- value of the angle between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float angle(zVector &v1);

		/*! \brief This method returns the angle between the current zVector and input zVector in the range of 0 to 360 in the plane given by the input normal..
		*
		*	\param		[in]	v1			- input vector.
		*	\param		[in]	normal		- input reference normal or axis of rotation.
		*	\return				float		- value of the angle between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float angle360(zVector &v1, zVector &normal);

		/*! \brief This method returns the dihedral angle between the two input zVectors using current zVector as edge reference.
		*
		*	\details Based on https://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleMeshDerivativesCheatSheet.pdf
		*	\param		[in]	v1			- input vector.
		*	\param		[in]	v2			- input vector.
		*	\return				float		- value of the dihedral angle between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float dihedralAngle(zVector &v1, zVector &v2);

		/*! \brief This method returns the contangent of the angle between the current and input vector.
		*
		*	\details Based on http://multires.caltech.edu/pubs/diffGeoOps.pdf and http://rodolphe-vaillant.fr/?e=69
		*	\param		[in]	v			- input vector.
		*	\return				float		- cotangent of angle between the vectors.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float cotan(zVector &v);

		/*! \brief This method gets the components as a array of doubles of the current zVector.
		*
		*	\param		[out]	vals		- output compnent values. ( 0 - x component, 1 - y component, 2 - z component).
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void getComponents(zFloat4 &_vals);


		/*! \brief This method returns the rotated vector of the current vector about an input axis by the the input angle.
		*
		*	\param		[in]		axisVec			- axis of rotation.
		*	\param		[in]		angle			- rotation angle.
		*	\return					zVector			- rotated vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector rotateAboutAxis(zVector axisVec, float angle = 0);

		/*! \brief This method returns the row matrix of the current zVector.
		*
		*	\return				zMatrix4Row		- row matrix of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void toRowMatrix4(zMatrix4Row &row);

		/*! \brief This method returns the row matrix of the current zVector.
		*
		*	\return				zMatrix3Row		- row matrix of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void toRowMatrix3(zMatrix3Row &row);

		/*! \brief This method returns the column matrix of the current zVector.
		*
		*	\param		[in]	rows			- number of rows in the output vector.
		*	\return				zMatrix4Col		- column matrix of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void toColumnMatrix4(zMatrix4Col &col);

		/*! \brief This method returns the column matrix of the current zVector.
		*
		*	\param		[in]	rows			- number of rows in the output vector.
		*	\return				zMatrix3Col		- column matrix of the vector.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void toColumnMatrix3(zMatrix3Col &col);

		/*! \brief This method returns the vector from the input row matrix.
		*
		*	\param		[in]		zMatrix4Row	- input row matrix.
		*	\return					zVector		- zVector of the row matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector fromRowMatrix4(zMatrix4Row &inMatrix);

		/*! \brief This method returns the vector from the input row matrix.
		*
		*	\param		[in]		zMatrix3Row	- input row matrix.
		*	\return					zVector		- zVector of the row matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector fromRowMatrix3(zMatrix3Row &inMatrix);

		/*! \brief This method returns the vector from the input column matrix.
		*
		*	\param		[in]		zMatrix4Col	- input column matrix.
		*	\return					zVector		- zVector of the column matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector fromColumnMatrix4(zMatrix4Col &inMatrix);

		/*! \brief This method returns the vector from the input column matrix.
		*
		*	\param		[in]		zMatrix3Col	- input column matrix.
		*	\return					zVector		- zVector of the column matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVector fromColumnMatrix3(zMatrix3Col &inMatrix);

#ifndef __CUDACC__

		/*! \brief This operator is used for 4x4 matrix muliplication of a vector.
		*
		*	\param		[in]	inMatrix	- input 4X4 zTransform to be multiplied with the current vector.
		*	\return				zVector		- resultant vector after the matrix multiplication.
		*	\since version 0.0.1
		*/
		zVector operator*(zTransform inTrans);

		//--------------------------
		//---- STREAM OPERATORS
		//--------------------------

		/*! \brief This method outputs the vector component values to the stream.
		*
		*	\param		[in]		os				- output stream.
		*	\param		[in]		zVector			- vector to be streamed.
		*	\since version 0.0.1
		*/
		friend ZSPACE_CORE ostream& operator<<(ostream& os, const zVector& vec);

#endif

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/base/zVector.cpp>
#endif

#endif