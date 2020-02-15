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

#ifndef ZSPACE_FN_H
#define ZSPACE_FN_H

#pragma once


#include <headers/zInterface/objects/zObj.h>

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFn
	*	\brief The base function set class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_API zFn
	{

	protected:
		
		/*!	\brief function type  */
		zFnType fnType;
		

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFn();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFn();

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------

		/*! \brief This method return the function set type.
		*
		*	\return 	zFnType			- type of function set.
		*	\since version 0.0.2
		*/
		virtual zFnType getType() = 0;

		/*! \brief This method imports the object linked to function type.
		*
		*	\param	[in]	path			- output file name including the directory path and extension.
		*	\param	[in]	type			- type of file to be imported.
		*	\param	[in]	staticGeom		- true if the object is static. Helps speed up display especially for meshes object. Default set to false.
		*	\since version 0.0.2
		*/
		virtual void from(string path, zFileTpye type, bool staticGeom = false) = 0;

		/*! \brief This method exports the object linked to function type.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported.
		*	\since version 0.0.2
		*/
		virtual void to(string path, zFileTpye type) = 0;

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\param [out]		minBB			- output minimum bounding box.
		*	\param [out]		maxBB			- output maximum bounding box.
		*	\since version 0.0.2
		*/
		virtual void getBounds(zPoint &minBB, zPoint &maxBB) = 0;

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		virtual void clear() = 0;


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the object transform to the input transform.
		*
		*	\param [in]		inTransform			- input transform.
		*	\param [in]		decompose			- decomposes transform to rotation and translation if true.
		*	\param [in]		updatePositions		- updates the object positions if true.
		*	\since version 0.0.2
		*/
		virtual void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) = 0;
	
		
		/*! \brief This method sets the scale components of the object.
		*
		*	\param [in]		scale		- input scale values.
		*	\since version 0.0.2
		*/
		virtual void setScale(zFloat4 &scale) = 0;
	

		/*! \brief This method sets the rotation components of the object.
		*
		*	\param [in]		rotation			- input rotation values.
		*	\param [in]		append		- true if the input values are added to the existing rotations.
		*	\since version 0.0.2
		*/
		virtual void setRotation(zFloat4 &rotation, bool append = false) = 0;
	

		/*! \brief This method sets the translation components of the object.
		*
		*	\param [in]		translation			- input translation vector.
		*	\param [in]		append				- true if the input values are added to the existing translation.
		*	\since version 0.0.2
		*/
		virtual void setTranslation(zVector &translation, bool append = false) = 0;
		

		/*! \brief This method sets the pivot of the object.
		*
		*	\param [in]		pivot				- input pivot position.
		*	\since version 0.0.2
		*/
		virtual void setPivot(zVector &pivot) = 0;
	
		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the object transform.
		*
		*	\since version 0.0.2
		*/
		virtual  void getTransform(zTransform &transform) = 0;

		//--------------------------
		//---- TRANSFORMATION METHODS
		//--------------------------
				
	protected:		

		/*! \brief This method scales the object with the input scale transformation matix.
		*
		*	\since version 0.0.2
		*/
		virtual void transformObject(zTransform &transform) = 0;

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFn.cpp>
#endif

#endif