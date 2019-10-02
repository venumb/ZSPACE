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

#ifndef ZSPACE_FIELD_2D_H
#define ZSPACE_FIELD_2D_H

#pragma once

#include<headers/zCore/base/zVector.h>
#include<headers/zCore/base/zColor.h>


namespace zSpace
{

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	
	/** \addtogroup zFields
	*	\brief The field classes of the library.	
	*  @{
	*/

	/*! \class zField2D
	*	\brief A template class for 2D fields - scalar and vector.
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class ZSPACE_CORE zField2D
	{	   		

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
	
		/*!	\brief stores the resolution in X direction  */
		int n_X;

		/*!	\brief stores the resolution in Y direction  */
		int n_Y;

		/*!	\brief stores the size of one unit in X direction  */
		double unit_X;

		/*!	\brief stores the size of one unit in Y direction  */
		double unit_Y;

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector minBB;

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector maxBB;

		/*!	\brief container for the field values  */
		vector<T> fieldValues;

		/*!	\brief boolean indicating if the field values size is equal to mesh vertices(true) or equal to mesh faces(false)  */
		bool valuesperVertex = true;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zField2D();

		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\since version 0.0.1
		*/
		zField2D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y);

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\since version 0.0.1
		*/		
		zField2D(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB = zVector());
		
			
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zField2D();

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/field/zField2D.cpp>
#endif

#endif