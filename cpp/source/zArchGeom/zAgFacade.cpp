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



	ZSPACE_INLINE zAgFacade::zAgFacade(zPointArray&_vertexCorners, zVectorArray&_extrudeDir, int _faceId)
	{
		faceId = _faceId;
		vertexCorners = _vertexCorners;
		extrudeDir = _extrudeDir;
	}


	ZSPACE_INLINE void zAgFacade::createTimber()
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

		///////////////////////////////////////////////

		int numPoints = 0;
		for (int i = 0; i < bottomCorners.size(); i++)
		{
			zPointArray facadePoints;
			facadePoints.assign(30, zVector());

			zVector outDir = extrudeDir[indices[i]];
			outDir.normalize();
			zVector inDir = bottomCorners[i] - centerBottom;

			facadePoints[0] = centerBottom + outDir * -1.1;
			facadePoints[1] = centerBottom + (outDir * -1) + (inDir * 0.25);
			facadePoints[2] = facadePoints[1];
			facadePoints[3] = centerBottom + (outDir * -0.75) + (inDir * 0.6);
			facadePoints[4] = facadePoints[3];
			facadePoints[5] = centerBottom + (outDir * -0.2) + (inDir * 0.85);
			facadePoints[6] = facadePoints[5];
			facadePoints[7] = bottomCorners[i] + (outDir * 0.5);
			facadePoints[8] = facadePoints[7];
			facadePoints[9] = bottomCorners[i] + (outDir * 1);


			for (int u = 0 ; u < 10; u++)
			{
				facadePoints[u + 10] = facadePoints[u] + zVector(0, 0, 0.6);
			}

			facadePoints[13] += inDir * 0.15;
			facadePoints[14] = facadePoints[13];

			for (int u = 0; u < 10; u++)
			{
				facadePoints[u + 20] = facadePoints[u] + zVector(0, 0, 3);
			}

			for (auto v : facadePoints)
			{
				pointArray.push_back(v);
			}

			///////////////////////
				//polyconnect and polycount quads
			int pInContour = 10; // number in each contour
			int layerCount = 0;

			for (int j = 0; j < 2 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
			{
				/*if (layerCount == 1)
				{
					layerCount++;
					continue;
				}*/

				for (int k = 0; k < pInContour - 1; k +=2) // loops though faces in a given j layer
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
			zFnMesh fnInMesh(inMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.smoothMesh(1, false);
			createMullions();
		}
	}

	ZSPACE_INLINE void zAgFacade::createRhwc()
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
			facadePoints.assign(48, zVector());

			zVector outDir = extrudeDir[indices[i]];
			outDir.normalize();
			zVector inDir = bottomCorners[i] - centerBottom;

			facadePoints[0] = centerBottom + outDir + (inDir * 0.5);
			facadePoints[1] = centerBottom + (inDir * 0.5);
			facadePoints[2] = facadePoints[1];
			facadePoints[3] = centerBottom + (outDir * -0.2) + (inDir * 0.9);
			facadePoints[4] = facadePoints[3];
			facadePoints[5] = bottomCorners[i];
			facadePoints[6] = facadePoints[5];
			facadePoints[7] = centerBottom + (outDir)+(inDir * 1);
			
			facadePoints[8] = centerBottom + outDir + (inDir * 0.9) + zVector(0,0,0.4);
			facadePoints[9] = centerBottom + (inDir * 0.9) + zVector(0, 0, 0.4);
			facadePoints[10] = facadePoints[9];
			facadePoints[11] = centerBottom + (outDir * -0.1) + (inDir * 0.95) + zVector(0, 0, 0.4);
			facadePoints[12] = facadePoints[11];
			facadePoints[13] = bottomCorners[i] + zVector(0, 0, 0.4);
			facadePoints[14] = facadePoints[13];
			facadePoints[15] = centerBottom + (outDir) + (inDir * 1) + zVector(0, 0, 0.4);

			facadePoints[16] = centerBottom + outDir + (inDir * 0.9) + zVector(0, 0, 2.6);
			facadePoints[17] = centerBottom + (inDir * 0.9) + zVector(0, 0, 2.6);
			facadePoints[18] = facadePoints[17];
			facadePoints[19] = centerBottom + (outDir * -0.1) + (inDir * 0.95) + zVector(0, 0, 2.6);
			facadePoints[20] = facadePoints[19];
			facadePoints[21] = bottomCorners[i] + zVector(0, 0, 2.6);
			facadePoints[22] = facadePoints[21];
			facadePoints[23] = centerBottom + (outDir)+(inDir * 1) + zVector(0, 0, 2.6);

			facadePoints[24] = centerBottom + outDir + (inDir * 0.5) + zVector(0, 0, 2.8);
			facadePoints[25] = centerBottom + (inDir * 0.5) + zVector(0, 0, 2.8);
			facadePoints[26] = facadePoints[25];
			facadePoints[27] = centerBottom + (outDir * -0.1) + (inDir * 0.9) + zVector(0, 0, 2.95);
			facadePoints[28] = facadePoints[27];
			facadePoints[29] = bottomCorners[i] + zVector(0, 0, 3);
			facadePoints[30] = facadePoints[29];
			facadePoints[31] = centerBottom + (outDir)+(inDir * 1) + zVector(0, 0, 3); 

			for (int u = 24; u < 32; u++) facadePoints[u + 8] = facadePoints[u];

			facadePoints[40] = centerBottom + outDir + zVector(0, 0, 2.8);
			facadePoints[41] = centerBottom + zVector(0, 0, 2.8);
			facadePoints[42] = facadePoints[41];
			facadePoints[43] = centerBottom + (outDir * -0.1) + zVector(0, 0, 2.95);
			facadePoints[44] = facadePoints[43];
			facadePoints[45] = centerBottom + zVector(0, 0, 3);
			facadePoints[46] = facadePoints[45];
			facadePoints[47] = centerBottom + outDir + zVector(0, 0, 3);
			
			for (auto v : facadePoints)
			{
				pointArray.push_back(v);
			}

			///////////////////////
				//polyconnect and polycount quads
			int pInContour = 8; // number in each contour
			int layerCount = 0;

			for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
			{
				if (layerCount == 3)
				{
					layerCount++;
					continue;
				}

				for (int k = 0; k < pInContour - 1; k+=2) // loops though faces in a given j layer
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
			zFnMesh fnInMesh(inMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
			fnInMesh.smoothMesh(2, false);

			fnInMesh.to("C:/Users/cesar.fragachan/Desktop/column.obj", zOBJ);
		}
	}

	ZSPACE_INLINE void zAgFacade::createMullions()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (zItMeshHalfEdge he(inMeshObj); !he.end(); he++)
		{
			if (he.onBoundary()) continue;

			zVector xd = he.getNext().getVector();
			xd.normalize();
			xd *= 0.05;

			zVector yd = he.getFace().getNormal();
			yd.normalize();
			yd *= 0.05;

			pointArray.push_back(he.getVertex().getPosition());
			pointArray.push_back(he.getVertex().getPosition() + yd);
			pointArray.push_back(he.getVertex().getPosition() + yd + xd);
			pointArray.push_back(he.getVertex().getPosition() + xd);

			pointArray.push_back(he.getPrev().getVertex().getPosition());
			pointArray.push_back(he.getPrev().getVertex().getPosition() + yd);
			pointArray.push_back(he.getPrev().getVertex().getPosition() + yd + xd);
			pointArray.push_back(he.getPrev().getVertex().getPosition() + xd);
		}

		for (int i = 0; i < pointArray.size() - 8; i+=8)
		{
			for (int j = 0; j < 4; j++)
			{
				int s = polyConnect.size();
				polyConnect.push_back(i + j);
				polyConnect.push_back(i + ((j + 1) % 4));
				polyConnect.push_back(i + ((j + 1) % 4) + 4);
				polyConnect.push_back(i + j + 4);

				//printf("\n polyconnect: %i %i %i %i", polyConnect[s], polyConnect[s + 1], polyConnect[s + 2], polyConnect[s + 3]);

				polyCount.push_back(4);
			}

		}

		zFnMesh fnInMesh(inMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);
	}

	ZSPACE_INLINE void zAgFacade::updateFacade(zPointArray & _vertexCorners)
	{
		vertexCorners = _vertexCorners;
	}

	ZSPACE_INLINE void zAgFacade::displayFacade(bool showFacade)
	{
		inMeshObj.setShowObject(showFacade);
	}

	ZSPACE_INLINE void zAgFacade::addObjsToModel()
	{
		model->addObject(inMeshObj);
		inMeshObj.setShowElements(false, true, true);
	}

}