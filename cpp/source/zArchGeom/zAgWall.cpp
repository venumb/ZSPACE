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

	ZSPACE_INLINE zAgWall::zAgWall(zPointArray&_vertexCorners, int _faceId)
	{
		vertexCorners = _vertexCorners;
		faceId = _faceId;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgWall::~zAgWall() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zAgWall::createTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (int i = 0; i < vertexCorners.size(); i++)
		{
			pointArray.push_back(vertexCorners[i]);
			polyConnect.push_back(i);
		}
		polyCount.push_back(vertexCorners.size());

		zFnMesh fnInMesh(wallMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

	ZSPACE_INLINE void zAgWall::createRhwc()
	{
		createTimber();
	}

	//---- UPDATE METHODS

	ZSPACE_INLINE void zAgWall::updateWall(zPointArray & _vertexCorners)
	{
		vertexCorners = _vertexCorners;
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zAgWall::displayWall(bool showWall)
	{
		wallMeshObj.setDisplayObject(showWall);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zAgWall::addObjsToModel()
	{
		model->addObject(wallMeshObj);
		wallMeshObj.setDisplayElements(false, true, true);
	}

#endif
}