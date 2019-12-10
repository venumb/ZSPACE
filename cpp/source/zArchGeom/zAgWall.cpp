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


	ZSPACE_INLINE zAgWall::zAgWall(zPointArray&_vertexCorners, int _faceId)
	{
		fnInMesh = zFnMesh(inMeshObj);
		vertexCorners = _vertexCorners;
		faceId = _faceId;
	}

	ZSPACE_INLINE void zAgWall::createWallByType(zStructureType & _structureType)
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

		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

	ZSPACE_INLINE void zAgWall::updateWall(zPointArray & _vertexCorners)
	{
		vertexCorners = _vertexCorners;
	}

	ZSPACE_INLINE void zAgWall::displayWall(bool showWall)
	{
		inMeshObj.setShowObject(showWall);
	}

	ZSPACE_INLINE void zAgWall::setWallDisplayModel(zModel & _model)
	{
		model = &_model;

		model->addObject(inMeshObj);
		inMeshObj.setShowElements(false, true, true);
	}

}