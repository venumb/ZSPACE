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

#ifndef ZSPACE_AG_TYPEDEF
#define ZSPACE_AG_TYPEDEF

#pragma once
#include <headers/zArchGeom/zAgColumn.h>
#include <headers/zArchGeom/zAgSlab.h>
#include <headers/zArchGeom/zAgWall.h>
#include <headers/zArchGeom/zAgFacade.h>
#include <headers/zArchGeom/zAgRoof.h>


namespace zSpace
{

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/*! \typedef zColumnsArray
	*	\brief A vector of zAgColumn.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgColumn> zColumnArray;

	/*! \typedef zSlabArray
	*	\brief A vector of zAgSlab.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgSlab> zSlabArray;

	/*! \typedef zSlabArray
	*	\brief A vector of zAgWall.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgWall> zWallArray;

	/*! \typedef zfacadeArray
	*	\brief A vector of zAgFacade.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgFacade> zFacadeArray;

	/*! \typedef zRoofArray
	*	\brief A vector of zAgRoof.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgRoof> zRoofArray;

	/*! \typedef zColumnsPointerArray
	*	\brief A vector of zAgColumn Pointer.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgColumn*> zColumnPointerArray;

	/*! \typedef zSlabPointerArray
	*	\brief A vector of zAgSlab Pointer.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgSlab*> zSlabPointerArray;

	/*! \typedef zSlabPointerArray
	*	\brief A vector of zAgWall Pointer.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgWall*> zWallPointerArray;

	/*! \typedef zfacadePointerArray
	*	\brief A vector of zAgFacade Pointer.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zAgFacade*> zFacadePointerArray;

	/*! \typedef zroofPointerArray
*	\brief A vector of zAgRoof Pointer.
*
*	\since version 0.0.4
*/
	typedef vector<zAgRoof*> zRoofPointerArray;

}


#endif