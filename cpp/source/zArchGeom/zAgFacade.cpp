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



	zAgFacade::zAgFacade(zObjMesh & _inMeshObj, zPointArray&_vertexCorners, int _faceId)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);
		faceId = _faceId;
		vertexCorners = _vertexCorners;

	}

	void zAgFacade::createFacadeByType(zStructureType & _structureType)
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (auto& v : vertexCorners)
		{
			pointArray.push_back(v);
		}

		for (int i = 0; i < vertexCorners.size(); i++)
		{
			polyConnect.insert(polyConnect.begin(), i);
		}

		polyCount.push_back(vertexCorners.size());

		zObjMesh initFace;
		zFnMesh fnInit(initFace);

		fnInit.create(pointArray, polyCount, polyConnect);
		fnInit.extrudeBoundaryEdge(1.0, *inMeshObj, false);

		zVector backCenter;
		zPointArray tempVPositions;
		zPointArray backCorners;

		fnInMesh.getVertexPositions(tempVPositions);

		for (int i = tempVPositions.size() / 2; i < tempVPositions.size(); i++)
		{
			backCorners.push_back(tempVPositions[i]);
			backCenter += tempVPositions[i];
		}

		backCenter /= backCorners.size();

		for (auto& v : backCorners)
		{
			zVector dir = backCenter - v;
			dir.normalize();
			dir *= 0.6;
			v += dir;
		}

		for (int i = tempVPositions.size() / 2; i < tempVPositions.size(); i++)
		{
			tempVPositions[i] = backCorners[i - (tempVPositions.size() / 2)];
		}

		fnInMesh.setVertexPositions(tempVPositions);
		fnInMesh.extrudeMesh(0.05, *inMeshObj, false);
		//fnInMesh.smoothMesh(2, false);
	}

	void zAgFacade::updateFacade(zPointArray & _vertexCorners)
	{
		vertexCorners = _vertexCorners;
	}

}