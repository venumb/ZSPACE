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


#include<headers/zHousing/architecture/zHcUnit.h>

namespace zSpace
{
	//---- CONSTRUCTORS

	ZSPACE_INLINE zHcUnit::zHcUnit(){}

	ZSPACE_INLINE zHcUnit::zHcUnit(zObjMesh&_inMeshObj, zObjMeshArray &_strcutureObjs)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(_inMeshObj);

		for (int i = 0; i < _strcutureObjs.size(); i++)
		{
			structureObjs.push_back(&_strcutureObjs[i]);			
		}
		
		createStructuralUnits();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	bool zHcUnit::createStructuralUnits()
	{
		bool success = false;
		if (!inMeshObj) return success;

		for (zItMeshFace f(*inMeshObj); !f.end(); f++)
		{
			int  id = f.getId();
			zPointArray vPositions;
			f.getVertexPositions(vPositions);

			zHcStructure tempStructure(*structureObjs[id], vPositions);
			structureUnits.push_back(tempStructure);
		}

		success = true;
		return success;
	}

	void zHcUnit::createLayoutPlan(zLayoutType layout_)
	{
		layoutType = layout_;
	}

}