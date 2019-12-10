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


	ZSPACE_INLINE zAgWall::zAgWall(zPointArray&_corners, int _faceId)
	{
		corners = _corners;
		faceId = _faceId;
	}

	ZSPACE_INLINE void zAgWall::createTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (int i = 0; i < corners.size(); i++)
		{
			pointArray.push_back(corners[i]);
			polyConnect.push_back(i);
		}
		polyCount.push_back(corners.size());

		zFnMesh fnInMesh(inMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

	ZSPACE_INLINE void zAgWall::createRhwc()
	{
		createTimber();
	}

	ZSPACE_INLINE void zAgWall::updateWall(zPointArray & _corners)
	{
		corners = _corners;
	}

#ifndef ZSPACE_UNREAL_INTEROP

	ZSPACE_INLINE void zAgWall::displayWall(bool showWall)
	{
		inMeshObj.setShowObject(showWall);
	}

	ZSPACE_INLINE void zAgWall::addObjsToModel()
	{
		model->addObject(inMeshObj);
		inMeshObj.setShowElements(false, true, true);
	}

#endif
}