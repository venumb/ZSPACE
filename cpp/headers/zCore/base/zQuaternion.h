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

#ifndef ZSPACE_QUATERNION_H
#define ZSPACE_QUATERNION_H

#pragma once

#include<headers/zCore/base/zDefinitions.h>
#include<headers/zCore/base/zVector.h>

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

	/*! \class zQuaternion
	*	\brief A Quaternion  class.
	*	\details Adapted from Quaternion class described here https://github.com/dgpdec/course.
	*	\since version 0.0.2
	*/

	/** @}*/
	/** @}*/

	class ZSPACE_CORE zQuaternion
	{
	protected:

		/*!	\brief scalar part of quaternion			*/
		double s;		

		/*!	\brief vector part of quaternion(imaginary)		*/
		zVector v;
	

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------


		/*! \brief Default Constructor.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion();

		/*! \brief Overloaded Constructor.
		*	\param		[in]	_s			- scalar part.
		*	\param		[in]	_v			- vector part.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion(double _s, double _vi, double _vj, double _vk);

		/*! \brief Overloaded Constructor.
		*	\param		[in]	_s			- scalar part.
		*	\param		[in]	_v			- vector part.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion(double _s, zVector _v);

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator assigns scalar component of the quaternion
		*
		*	\param		[in]	_s		- scalar component.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void operator=(double _s);

		/*! \brief This operator assigns vector component of the quaternion
		*
		*	\param		[in]	_v		- vector component.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void operator=(zVector &_v);

		/*! \brief This operator returns the indexed component (0-based indexing: double, i, j, k)
		*
		*	\return			double&		- reference to the specified component (0-based indexing: double, i, j, k)
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double& operator[](int index);

		/*! \brief This operator returns the indexed component (0-based indexing: double, i, j, k)
		*
		*	\return			double&		- const reference to the specified component (0-based indexing: double, i, j, k)
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE const double& operator[](int index) const;

		/*! \brief This operator is used for quaternion addition.
		*
		*	\param		[in]	q			- quaternion which is added to the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the addition.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion operator+(const zQuaternion  &q);

		/*! \brief This operator is used for quaternion subtraction.
		*
		*	\param		[in]	q			- quaternion which is subtracted from the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the subtraction.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion operator-(const zQuaternion &q);

		/*! \brief This operator is used for quaternion multiplication with a scalar.
		*
		*	\param		[in]	c			- scalar value which is multiplied to the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the multiplication.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion operator*(double c);

		/*! \brief This operator is used for quaternion Hamiton Product.
		*
		*	\param		[in]	q			- quaternion which is multiplied with the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the multiplication.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion operator*(zQuaternion &q);

		/*! \brief This operator is used for quaternion division with a scalar.
		*
		*	\param		[in]	q			- scalar value which divides the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the division.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion operator/(double c);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This overloaded operator is used for addition and assigment of the result to the current quaternion.
		*
		*	\param		[in]	q		- quaternion which is added to the current quaternion.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator +=(const zQuaternion &q);

		/*! \brief This overloaded operator is used for subtraction and assigment of the result to the current quaternion.
		*
		*	\param		[in]	q		- quaternion which is subtracted from the current quaternion.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator -=(const zQuaternion &q);

		/*! \brief This operator is used for multiplication with a scalar and assigment of the result to the current quaternion.
		*
		*	\param		[in]	c			- scalar value which is multiplied to the current quaternion.		
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void operator*=(double c);

		/*! \brief This operator is used for quaternion Hamiton Product and assigment of the result to the current quaternion..
		*
		*	\param		[in]	q			- quaternion which is multiplied with the current quaternion.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void operator*=(zQuaternion q);

		/*! \brief This operator is used for division with a scalar and assigment of the result to the current quaternion.
		*
		*	\param		[in]	c			- scalar value which is used for division of the current quaternion.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void operator/=(double c);
			   
		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the real/scalar component of the quaternion.
		*	\return				double		- scalar component.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double getRe();

		/*! \brief This method gets the imaginary/ vector component of the quaternion.
		*	\return				zVector		- vector component.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector getIm();

		/*! \brief This method gets the conjuate of the quaternion.
		*	\return				zQuaternion		- conjugate quaternion.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion getConjugate();

		/*! \brief This method gets the inverse of the quaternion.
		*	\return				zQuaternion		- inverse quaternion.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion getInverse();

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method gets the Euclidean length of the quaternion.
		*	\return				double		- Euclidean length.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double length();

		/*! \brief This method gets the Euclidean length sqaured of the quaternion.
		*	\return				double		- Euclidean length squared.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double length2();

		/*! \brief This method normalizes the quaternion .
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void normalize();

		/*! \brief This method computes spherical-linear interpolation
		*	\param		[in]	q0			- input start quaternion.
		*	\param		[in]	q1			- input end quaternion.
		*	\param		[in]	t			- input time step.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zQuaternion slerp(zQuaternion& q0, zQuaternion& q1, double t);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
#else
#include<source/zCore/base/zQuaternion.cpp>
#endif

#endif