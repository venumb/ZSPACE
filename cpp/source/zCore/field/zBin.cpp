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

#include<headers/zCore/field/zBin.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zBin::zBin()
	{
		ids.clear();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zBin::~zBin() {}

	//---- METHODS

	ZSPACE_INLINE void  zBin::addObject()
	{
		ids.push_back(vector<int>());
	}

	ZSPACE_INLINE void zBin::addVertexIndex(int vertexId, int objectId)
	{
		if (objectId < ids.size())	ids[objectId].push_back(vertexId);
		else throw std::invalid_argument(" error: objectId out of bounds.");

	}

	ZSPACE_INLINE void zBin::clear()
	{
		ids.clear();
	}

	ZSPACE_INLINE bool zBin::contains()
	{
		return (ids.size() > 0);
	}

}