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


#include<headers/zArchGeom/zAgFacade.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgFacade::zAgFacade(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgFacade::~zAgFacade() {}

	//---- SET METHODS



	zAgFacade::zAgFacade(zObjMesh & _inMeshObj, zPointArray&_vertexCorners)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);

		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;


		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

}