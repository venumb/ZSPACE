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


#include<headers/zHousing/architecture/zHcStructure.h>


namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zHcStructure::zHcStructure() {}


	ZSPACE_INLINE zHcStructure::zHcStructure(zObjMesh&_inMeshObj, zPointArray &faceVertexPositions)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(_inMeshObj);	

		CreateSpatialCell(faceVertexPositions);
			
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcStructure::~zHcStructure() {}

	//---- SET METHODS
	   	 

	void zHcStructure::CreateSpatialCell(zPointArray &vPositions)
	{
		zIntArray polyConnect;
		zIntArray polyCount;

		int vPositionsCount = vPositions.size();
		polyCount.push_back(vPositions.size());

		for (int i = 0; i < vPositions.size(); i++)
		{
			polyConnect.push_back(i);
		}

		zObjMesh tempObj;
		zFnMesh tempFn(tempObj);

		tempFn.create(vPositions, polyCount, polyConnect);
		tempFn.extrudeMesh(height, *inMeshObj, false);

		//zIntArray polyconnect;
		//zIntArray polyCount;

		//for (int i = 0; i < vertexPositions_.size() / 2; i++)
		//{
		//	if ((i + 1) < vertexPositions_.size() / 2)
		//	{
		//		polyconnect.push_back(i);
		//		polyconnect.push_back(i + vertexPositions_.size() / 2);
		//		polyconnect.push_back(i + (vertexPositions_.size() / 2) + 1);
		//		polyconnect.push_back(i + 1);
		//	}
		//	else if ((i + 1) == vertexPositions_.size() / 2)
		//	{
		//		polyconnect.push_back(i);
		//		polyconnect.push_back(i + vertexPositions_.size() / 2);
		//		polyconnect.push_back(vertexPositions_.size());
		//		polyconnect.push_back(0);
		//	}
		//}

		////create top faces
		//for (int i = 0; i < vertexPositions_.size()/2; i++)
		//{
		//	polyconnect.push_back(i);
		//}

		////create bottom faces
		//for (int i = 0; i < vertexPositions_.size() / 2; i++)
		//{
		//	polyconnect.push_back(i + vertexPositions_.size() / 2);
		//}

		////set wrapper faces poly count
		//for (int i = 0; i < vertexPositions_.size() / 2; i++)
		//{
		//	polyCount.push_back(4);
		//}

		//polyCount.push_back(vertexPositions_.size() / 2); //bottom face poly count
		//polyCount.push_back(vertexPositions_.size() / 2); //top face poly count

		//printf("v number: %i /n", vertexPositions_.size());
		//printf("polyCount size: %i /n", polyCount.size());
		//printf("polyConnect size: %i /n", polyconnect.size());

		//zVector x, y, z;
		//vector<zVector> testpointArray;
		//vector<int> testpolyCount;
		//vector<int> testpolyConnect;

		//x = zVector(0, 0, 0);
		//y = zVector(0, 10, 0);
		//z = zVector(10, 0, 0);

		//testpointArray.push_back(x);
		//testpointArray.push_back(y);
		//testpointArray.push_back(z);

		//testpolyConnect.push_back(0);
		//testpolyConnect.push_back(1);
		//testpolyConnect.push_back(2);

		//testpolyCount.push_back(3);

		//
		//fnInMesh.create(testpointArray, testpolyCount, testpolyConnect);
	}

	

}