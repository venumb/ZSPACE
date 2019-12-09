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



	zAgFacade::zAgFacade(zObjMesh & _inMeshObj, zPointArray&_vertexCorners, zVectorArray&_extrudeDir, int _faceId)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);
		faceId = _faceId;
		vertexCorners = _vertexCorners;
		extrudeDir = _extrudeDir;
	}

	void zAgFacade::createFacadeByType(zStructureType & _structureType)
	{
		if (_structureType == zStructureType::zRHWC) createFacadeConcrete(); ////////UPDATE FOR TEST
		else if (_structureType == zStructureType::zDigitalTimber) createFacadeTimber();


	}

	void zAgFacade::createFacadeTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;


		/*int i = 0;
		for (auto dir : extrudeDir)
		{
			dir.normalize();
			
			pointArray.push_back(vertexCorners[i] + dir);
			polyConnect.push_back(i);
			i++;
		}
		polyCount.push_back(4);*/

		zPointArray bottomCorners;
		zVector center, centerBottom;
		zIntArray indices;

		for (auto v : vertexCorners)
		{
			center += v;
		}
		center /= vertexCorners.size();

		int id = 0;
		for (auto v : vertexCorners)
		{
			if (v.z < center.z)
			{
				bottomCorners.push_back(v);
				centerBottom += v;
				indices.push_back(id);
			}

			id++;
		}
		centerBottom /= bottomCorners.size();

		//printf("boot: %i", bottomCorners.size());

		///////////////////////////////////////////////

		int numPoints = 0;
		for (int i = 0; i < bottomCorners.size(); i++)
		{
			zPointArray facadePoints;
			facadePoints.assign(15, zVector());

			zVector outDir = extrudeDir[indices[i]];
			outDir.normalize();
			zVector inDir = bottomCorners[i] - centerBottom;

			facadePoints[0] = centerBottom + outDir * -0.8;
			facadePoints[1] = centerBottom + (outDir * -0.7) + (inDir * 0.25);
			facadePoints[2] = centerBottom + (outDir * -0.6) + (inDir * 0.5);
			facadePoints[3] = centerBottom + (outDir * -0.5) + (inDir * 0.75);
			facadePoints[4] = bottomCorners[i];


			for (int u = 0 ; u < 5; u++)
			{
				facadePoints[u + 5] = facadePoints[u] + zVector(0, 0, 1);
			}
			facadePoints[7] += inDir * 0.1;

			for (int u = 5; u < 10; u++)
			{
				facadePoints[u + 5] = facadePoints[u] + zVector(0, 0, 2);
			}


			for (auto v : facadePoints)
			{
				pointArray.push_back(v);
			}

			///////////////////////
				//polyconnect and polycount quads
			int pInContour = 5; // number in each contour
			int layerCount = 0;

			for (int j = 0; j < 2 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
			{
				/*if (layerCount == 1)
				{
					layerCount++;
					continue;
				}*/

				for (int k = 0; k < pInContour - 1; k ++) // loops though faces in a given j layer
				{
					int pc_size = polyConnect.size();
					polyConnect.push_back((numPoints)+j + k);
					polyConnect.push_back((numPoints)+j + k + pInContour);
					polyConnect.push_back((numPoints)+j + k + pInContour + 1);
					polyConnect.push_back((numPoints)+j + k + 1);

					polyCount.push_back(4);
				}
				layerCount++;
			}
			numPoints = pointArray.size();


		}

		///////////


		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.smoothMesh(1, false);
		}
	}

	void zAgFacade::createFacadeConcrete()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		zPointArray bottomCorners;
		zVector center, centerBottom;
		zIntArray indices;

		for (auto v : vertexCorners)
		{
			center += v;
		}
		center /= vertexCorners.size();

		int id = 0;
		for (auto v : vertexCorners)
		{
			if (v.z < center.z)
			{
				bottomCorners.push_back(v);
				centerBottom += v;
				indices.push_back(id);
			}

			id++;
		}
		centerBottom /= bottomCorners.size();

		//printf("boot: %i", bottomCorners.size());

		///////////////////////////////////////////////

		int numPoints = 0;
		for (int i = 0; i < bottomCorners.size(); i++)
		{
			zPointArray facadePoints;
			facadePoints.assign(20, zVector());

			zVector outDir = extrudeDir[indices[i]];
			outDir.normalize();
			zVector inDir = bottomCorners[i] - centerBottom;

			facadePoints[0] = centerBottom + outDir + (inDir * 0.5);
			facadePoints[1] = centerBottom + (inDir * 0.5);
			facadePoints[2] = centerBottom + (outDir * -0.2) + (inDir * 0.9);
			facadePoints[3] = bottomCorners[i];
			facadePoints[4] = centerBottom + (outDir)+(inDir * 1);
			
			facadePoints[5] = centerBottom + outDir + (inDir * 0.9) + zVector(0,0,0.2);
			facadePoints[6] = centerBottom + (inDir * 0.9) + zVector(0, 0, 0.2);
			facadePoints[7] = centerBottom + (outDir * -0.1) + (inDir * 0.95) + zVector(0, 0, 0.2);
			facadePoints[8] = bottomCorners[i] + zVector(0, 0, 0.2);
			facadePoints[9] = centerBottom + (outDir) + (inDir * 1) + zVector(0, 0, 0.2);

			facadePoints[10] = centerBottom + outDir + (inDir * 0.9) + zVector(0, 0, 2.8);
			facadePoints[11] = centerBottom + (inDir * 0.9) + zVector(0, 0, 2.8);
			facadePoints[12] = centerBottom + (outDir * -0.1) + (inDir * 0.95) + zVector(0, 0, 2.95);
			facadePoints[13] = bottomCorners[i] + zVector(0, 0, 3);
			facadePoints[14] = centerBottom + (outDir)+(inDir * 1) + zVector(0, 0, 3); 

			facadePoints[15] = centerBottom + outDir + zVector(0, 0, 2.8);
			facadePoints[16] = centerBottom + zVector(0, 0, 2.8);
			facadePoints[17] = centerBottom + (outDir * -0.1) + zVector(0, 0, 2.95);
			facadePoints[18] = centerBottom + zVector(0, 0, 3);
			facadePoints[19] = centerBottom + outDir + zVector(0, 0, 3);
			
			for (auto v : facadePoints)
			{
				pointArray.push_back(v);
			}

			///////////////////////
				//polyconnect and polycount quads
			int pInContour = 5; // number in each contour
			int layerCount = 0;

			for (int j = 0; j < 3 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
			{
				/*if (layerCount == 1)
				{
					layerCount++;
					continue;
				}*/

				for (int k = 0; k < pInContour - 1; k++) // loops though faces in a given j layer
				{
					int pc_size = polyConnect.size();
					polyConnect.push_back((numPoints)+j + k);
					polyConnect.push_back((numPoints)+j + k + pInContour);
					polyConnect.push_back((numPoints)+j + k + pInContour + 1);
					polyConnect.push_back((numPoints)+j + k + 1);

					polyCount.push_back(4);
				}
				layerCount++;
			}
			numPoints = pointArray.size();


		}

		///////////


		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.smoothMesh(1, false);
		}
	}

	void zAgFacade::updateFacade(zPointArray & _vertexCorners)
	{
		vertexCorners = _vertexCorners;
	}

}