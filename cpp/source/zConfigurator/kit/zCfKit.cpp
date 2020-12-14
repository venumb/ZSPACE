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


#include <headers/zConfigurator/kit/zCfKit.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zCfKit::zCfKit() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zCfKit::~zCfKit() {}

	//---- SET METHODS

	ZSPACE_INLINE void zCfKit::setComponentID(string& _componentID)
	{
		componentID = _componentID;
	}

	ZSPACE_INLINE void zCfKit::setUnitID(string& _unitID)
	{
		unitID = _unitID;
	}

	ZSPACE_INLINE void zCfKit::setVoxelID(string& _voxelID)
	{
		voxelID = _voxelID;
	}

	//---- GET METHODS

	ZSPACE_INLINE zObjMesh* zCfKit::getRawComponentMesh()
	{
		return &o_component;
	}
}