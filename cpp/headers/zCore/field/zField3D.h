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

#ifndef ZSPACE_FIELD_3D_H
#define ZSPACE_FIELD_3D_H

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

	/*! \class zField3D
	*	\brief A template class for 3D fields - scalar and vector.
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class ZSPACE_CORE zField3D
	{		
		
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
		
		/*!	\brief stores the resolution in X direction  */
		int n_X;

		/*!	\brief stores the resolution in Y direction  */
		int n_Y;

		/*!	\brief stores the resolution in Z direction  */
		int n_Z;

		/*!	\brief stores the size of one unit in X direction  */
		double unit_X;

		/*!	\brief stores the size of one unit in Y direction  */
		double unit_Y;

		/*!	\brief stores the size of one unit in Z direction  */
		double unit_Z;

		/*!	\brief stores the minimum bounds of the field  */
		zVector minBB;

		/*!	\brief stores the minimum bounds of the field  */
		zVector maxBB;

		/*!	\brief container for the field values  */
		vector<T> fieldValues;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zField3D();

		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\since version 0.0.1
		*/		
		zField3D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z);

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each voxel in x direction.
		*	\param		[in]	_unit_Y		- size of each voxel in y direction.
		*	\param		[in]	_unit_Z		- size of each voxel in yz direction.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.		
		*	\since version 0.0.1
		*/
		zField3D(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zVector _minBB = zVector());

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zField3D();
			

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/field/zField3D.cpp>
#endif

#endif