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

#ifndef ZSPACE_HC_TYPEDEF
#define ZSPACE_HC_TYPEDEF

#pragma once

#include <headers/zHousing/architecture/zHcStructure.h>
#include <headers/zHousing/base/zHcEnumerators.h>


namespace zSpace
{

	/** \addtogroup zHcTypeDefs
	*	\brief  The type defintions of the housing library.
	*  @{
	*/

	/*! \typedef zStructurearray
	*	\brief A vector of zHcStructure.
	*
	*	\since version 0.0.4
	*/
	typedef vector <zHcStructure> zStructureArray;

	/*! \typedef zStructurePointerArray
	*	\brief A vector of zHcStructure pointers.
	*
	*	\since version 0.0.4
	*/
	typedef vector <zHcStructure*> zStructurePointerArray;


	/*! \typedef zCellFace
	*	\brief A vector of zCellFace.
	*
	*	\since version 0.0.4
	*/
	typedef vector <zCellFace> zCellFaceArray;

}


#endif