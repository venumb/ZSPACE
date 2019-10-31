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


#include<headers/zArchGeom/zAgWall.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgWall::zAgWall(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgWall::~zAgWall() {}

	//---- SET METHODS



	zAgWall::zAgWall(zObjMesh & _inMeshObj, zPointArray&_vertexCorners)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);

		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (int i = 0; i < _vertexCorners.size(); i++)
		{
			pointArray.push_back(_vertexCorners[i]);
			polyConnect.push_back(i);
		}
		polyCount.push_back(_vertexCorners.size());

		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

}