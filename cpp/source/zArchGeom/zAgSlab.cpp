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


#include<headers/zArchGeom/zAgSlab.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgSlab::zAgSlab(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgSlab::~zAgSlab() {}

	//---- SET METHODS



	zAgSlab::zAgSlab(zObjMesh & _inMeshObj, zVector & xCenter, zVector & yCenter, zVector & center, zAgColumn &_parentColumn)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);
		parentColumn = &_parentColumn;

		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		if (!parentColumn) return;

		////////
		pointArray.push_back(parentColumn->b);
		pointArray.push_back(parentColumn->a);
		pointArray.push_back(yCenter);
		pointArray.push_back(center);

		pointArray.push_back(parentColumn->b);
		pointArray.push_back(center);
		pointArray.push_back(xCenter);
		pointArray.push_back(parentColumn->c);

		pointArray.push_back(parentColumn->a);
		pointArray.push_back(parentColumn->position);
		pointArray.push_back(yCenter);

		pointArray.push_back(parentColumn->position);
		pointArray.push_back(parentColumn->c);
		pointArray.push_back(xCenter);

		pointArray.push_back(parentColumn->position);
		pointArray.push_back(xCenter);
		pointArray.push_back(center);
		pointArray.push_back(yCenter);



		/////////
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(3);
		polyCount.push_back(3);
		polyCount.push_back(4);



		///////////
		polyConnect.push_back(0);
		polyConnect.push_back(1);
		polyConnect.push_back(2);
		polyConnect.push_back(3);

		polyConnect.push_back(4);
		polyConnect.push_back(5);
		polyConnect.push_back(6);
		polyConnect.push_back(7);

		polyConnect.push_back(8);
		polyConnect.push_back(9);
		polyConnect.push_back(10);

		polyConnect.push_back(11);
		polyConnect.push_back(12);
		polyConnect.push_back(13);

		polyConnect.push_back(14);
		polyConnect.push_back(15);
		polyConnect.push_back(16);
		polyConnect.push_back(17);


		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.smoothMesh(2, false);
	}

}