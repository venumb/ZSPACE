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

#ifndef ZSPACE_TRANSFORMATIONMATRIX_H
#define ZSPACE_TRANSFORMATIONMATRIX_H

#pragma once

#include <headers/zCore/base/zMatrix.h>
#include <headers/zCore/base/zVector.h>


namespace zSpace
{
		   
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zTransformationMatrix
	*	\brief A transformation matrix class.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/


	class ZSPACE_CORE zTransformationMatrix
	{

	protected:

		/*!	\brief a 4x4 matrix to store transfromation  */
		zTransform Transform;

		/*!	\brief a 4x4 matrix to store pivot position  */
		zTransform P;

		/*!	\brief a 4x4 matrix to store scale information  */
		zTransform S;

		/*!	\brief a 4x4 matrix to store translation information  */
		zTransform T;

		/*!	\brief a 4x4 matrix to store rotation information. R = Rx * Ry *Rz  */
		zTransform R;

		/*!	\brief a 4x4 matrix to store rotation information in x axis  */
		zTransform Rx;

		/*!	\brief a 4x4 matrix to store rotation information in y axis  */
		zTransform Ry;

		/*!	\brief a 4x4 matrix to store rotation information in z axis  */
		zTransform Rz;

		/*!	\brief stores rotation  values in x, y, and z in radians  */
		double3 rotation;

		/*!	\brief stores scale values in x, y, and z  */
		double3 scale;

		/*!	\brief stores translation as a vector  */
		double3 translation;

		/*!	\brief stores pivot as a vector  */
		double3 pivot;

		

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------


		/*! \brief Default Constructor 
		*	
		*	\since version 0.0.2
		*/
		zTransformationMatrix();		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		~zTransformationMatrix();

		//--------------------------
		//---- SET METHODS
		//--------------------------


		/*! \brief This method sets the transform to the input tranform.
		*
		*	\param		[in]	decompose		- decoposes the translation and rotation matrix if true.
		*	\return 			zTransform		- input transform.
		*	\since version 0.0.2
		*/
		void setTransform(zTransform &inTransform, bool decompose = true);

		/*! \brief This method sets the rotation components of the tranform.
		*
		*	\param		[in]	_rotation			- input rotations in x,y and z in degrees.
		*	\param		[in]	addValues			- true if the input values are added to the existing values.
		*	\since version 0.0.2
		*/
		void setRotation(double3 &_rotation, bool addValues = false);

		/*! \brief This method sets the scale components of the tranform.
		*
		*	\param		[in]	_scale		- input scale in x,y and z.
		*	\since version 0.0.2
		*/
		void setScale(double3 &_scale);

		/*! \brief This method sets the translation components of the tranform.
		*
		*	\param		[in]	_translation		- input translation in x,y and z.
		*	\param		[in]	addValues			- true if the input values are added to the existing values.
		*	\since version 0.0.2
		*/
		void setTranslation(double3 &_translation, bool addValues = false);

		/*! \brief This method sets the pivot components of the tranform.
		*
		*	\param		[in]	_pivot		- input pivot in x,y and z.
		*	\since version 0.0.2
		*/
		void setPivot(double3 &_pivot);

		//--------------------------
		//---- GET METHODS
		//--------------------------
			

		/*! \brief This method gets the X-Axis of the tranform.
		*
		*	\return				zVector		- output x axis vector.
		*	\since version 0.0.2
		*/
		zVector getX();

		/*! \brief This method gets the Y-Axis of the tranform.
		*
		*	\return				zVector		- output y axis vector.
		*	\since version 0.0.2
		*/
		zVector getY();

		/*! \brief This method gets the Z-Axis of the tranform.
		*
		*	\return				zVector		- output z axis vector.
		*	\since version 0.0.2
		*/
		zVector getZ();

		/*! \brief This method gets the Origin of the tranform.
		*
		*	\return				zVector		- output origin vector.
		*	\since version 0.0.2
		*/
		zVector getO();

		/*! \brief This method gets the translation components of the tranform.
		*
		*	\return				zVector		- output translation vector.
		*	\since version 0.0.2
		*/
		zVector getTranslation();

		/*! \brief This method gets the pivot components of the tranform.
		*
		*	\return				zVector		- output pivot vector.
		*	\since version 0.0.2
		*/
		zVector getPivot();

		/*! \brief This method gets the pointer to the pivot components.
		*
		*	\return			double*		- pointer to pivot component.
		*	\since version 0.0.2
		*/
		double* getRawPivot();

		/*! \brief This method gets the rotation components of the tranform.
		*
		*	\param		[out]	_rotation		- output rotations in x,y and z in degrees.
		*	\since version 0.0.2
		*/
		void getRotation(double3 &_rotation);

		/*! \brief This method gets the scale components of the tranform.
		*
		*	\param		[out]	_scale		- output scale in x,y and z.
		*	\since version 0.0.2
		*/
		void getScale(double3 &_scale);		

		//--------------------------
		//---- AS MATRIX METHODS
		//--------------------------

		/*! \brief This method returns the 4x4 matrix that describes this transformation;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asMatrix();

		/*! \brief This method returns the pointer to the 4x4 matrix that describes this transformation;
		*
		*	\return 			double*		-  pointer to the 4x4 matrix .
		*	\since version 0.0.2
		*/
		double* asRawMatrix();

		/*! \brief This method returns the inverse of the 4x4 matrix that describes this transformation;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInverseMatrix();

		/*! \brief This method returns the scale matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asScaleMatrix();

		/*! \brief This method returns the rotation matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asRotationMatrix();
		
		/*! \brief This method returns the rotation matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asPivotMatrix();

		/*! \brief This method returns the pivot translation matrix
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asPivotTranslationMatrix();

		/*! \brief This method returns the inverse of the pivot translation matrix
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInversePivotTranslationMatrix();

		/*! \brief This method returns the scaling matrix;
		*
		*	\details based on https://studylib.net/doc/5892312/scaling-relative-to-a-fixed-point-using-matrix-using-the
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asScaleTransformMatrix();

		/*! \brief This method returns the scaling matrix;
		*
		*	\details based on https://studylib.net/doc/5892312/scaling-relative-to-a-fixed-point-using-matrix-using-the
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInverseScaleTransformMatrix();

		//--------------------------
		//---- GET MATRIX METHODS
		//--------------------------

		/*! \brief This method get the tranformation to the world space.
		*		
		*	\return 			zTransform		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getWorldMatrix();

		/*! \brief This method gets the tranformation to the local space.
		*
		*	\return 			zTransform		- local transformation matrix.
		*	\since version 0.0.2
		*/		
		zTransform getLocalMatrix();
		
		/*! \brief This method gets the tranformation from current tranform to input transform.
		*
		*	\param		[in]	to			- input transform.
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getToMatrix(zTransformationMatrix &to);

		/*! \brief This method gets the tranformation to change the basis to another.
		*
		*	\param		[in]	to			- input transformation matrix.
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getBasisChangeMatrix(zTransformationMatrix &to);

		/*! \brief This method gets the input target as per the current transform as the basis.
		*
		*	\param		[in]	target		- input target transform.
		*	\return 			zTransform	- new target transform.
		*	\since version 0.0.2
		*/
		zTransform getTargetMatrix(zTransform &target);		

	protected:

		/*! \brief This method compute the transform from the indivual matricies.
		*
		*	\since version 0.0.2
		*/
		void computeTransform();

		/*! \brief This method compute the rotation matrix.
		*
		*	\since version 0.0.2
		*/
		void computeR();

		/*! \brief This method compute the rotation matrix in X.
		*
		*	\since version 0.0.2
		*/
		void computeRx();

		/*! \brief This method compute the rotation matrix in Y.
		*
		*	\since version 0.0.2
		*/
		void computeRy();

		/*! \brief This method compute the rotation matrix in Z.
		*
		*	\since version 0.0.2
		*/
		void computeRz();

		/*! \brief This method compute the translation matrix.
		*
		*	\since version 0.0.2
		*/
		void computeT();

		/*! \brief This method compute the pivot matrix.
		*
		*	\since version 0.0.2
		*/
		void computeP();

		/*! \brief This method compute the scale matrix.
		*
		*	\since version 0.0.2
		*/
		void computeS();

		/*! \brief This method decomposes the rotation matrix into Euler angles.
		*
		*	\details based on  http://www.gregslabaugh.net/publications/euler.pdf. The zTransform matrix is the the transpose of the one in the paper. 
		*	\since version 0.0.2
		*/
		void decomposeR();

		/*! \brief This method decomposes the translation matrix into ddistances in x, y and z axis.
		*
		*	\since version 0.0.2
		*/
		void decomposeT();	

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
#else
#include<source/zCore/base/zTransformationMatrix.cpp>
#endif

#endif