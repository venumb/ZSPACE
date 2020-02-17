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

	ZSPACE_INLINE zAgSlab::zAgSlab(zVectorArray&_centerVecs, zVectorArray&_midPoints, zAgColumn&_parentColumn)
	{
		parentColumn = &_parentColumn;
		centerVecs = _centerVecs;
		midPoints = _midPoints;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgSlab::~zAgSlab() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zAgSlab::createRhwc()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		int numPoints = 0;
		for (int i = 0; i < centerVecs.size(); i++)
		{
			if (parentColumn->boundaryArray[i] == zBoundary::zInterior && parentColumn->snapSlabpoints[i].size() > 0)
			{
				for (auto v : parentColumn->snapSlabpoints[i])
				{
					v.z = parentColumn->position.z;
					pointArray.push_back(v);
				}
				/////new points 
				zPointArray newPoints;
				newPoints.assign(16, zVector());

				newPoints[0] = midPoints[i];
				newPoints[1] = centerVecs[i];
				newPoints[2] = newPoints[1];
				newPoints[3] = midPoints[(i + 1) % midPoints.size()];

				newPoints[4] = midPoints[i] + zVector(0, 0, -0.2);
				newPoints[5] = centerVecs[i] + zVector(0, 0, -0.2);
				newPoints[6] = newPoints[1] + zVector(0, 0, -0.2);
				newPoints[7] = midPoints[(i + 1) % midPoints.size()] + zVector(0, 0, -0.2);

				newPoints[8] = parentColumn->snapSlabpoints[i][4];
				newPoints[9] = parentColumn->snapSlabpoints[i][5];
				newPoints[10] = parentColumn->snapSlabpoints[i][6];
				newPoints[11] = parentColumn->snapSlabpoints[i][7];

				newPoints[12] = parentColumn->snapSlabpoints[i][0];
				newPoints[13] = parentColumn->snapSlabpoints[i][1];
				newPoints[14] = parentColumn->snapSlabpoints[i][2];
				newPoints[15] = parentColumn->snapSlabpoints[i][3];

				/////////add point to mesh
				for (auto v : newPoints)
				{
					pointArray.push_back(v);
				}


				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 4; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{
					/*if (layerCount == 1)
					{
						layerCount++;
						continue;
					}*/

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
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

			if (parentColumn->boundaryArray[i] == zBoundary::zEdge && parentColumn->snapSlabpoints[i].size() > 0)
			{
				zVector x, y; //y is edge for facade

				if (parentColumn->boundaryArray[(i + 1) % centerVecs.size()] == zBoundary::zInterior)
				{
					y = midPoints[i];
					x = midPoints[(i + 1) % midPoints.size()];
				}
				else
				{
					x = midPoints[i];
					y = midPoints[(i + 1) % midPoints.size()];
				}

				///////////////facade snap points
				zPointArray facadeSnaps;
				facadeSnaps.assign(6, zVector());

				zVector inDir = y - centerVecs[i];
				zVector outDir = centerVecs[i] - x;
				outDir.normalize();

				facadeSnaps[0] = centerVecs[i] + outDir * 0.3;
				facadeSnaps[1] = centerVecs[i] + outDir * 0.15 + (inDir * 0.25);
				facadeSnaps[2] = centerVecs[i] + (inDir * 0.5);
				facadeSnaps[3] = centerVecs[i] + (outDir * 0.3) + (inDir * 0.9);
				facadeSnaps[4] = y;
				facadeSnaps[5] = y + (outDir * -1);

				///////////////////////////////////////////
				/////new points 
				zPointArray slabPoints;
				slabPoints.assign(48, zVector());

				//
				for (int j = 0; j < parentColumn->snapSlabpoints[i].size() / 2; j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					v.z = parentColumn->position.z;
					slabPoints[j] = v;
				}

				zVector t = parentColumn->snapSlabpoints[i][10];
				t.z = parentColumn->position.z;
				slabPoints[10] = x;
				slabPoints[11] = t;

				slabPoints[12] = t;

				slabPoints[13] = facadeSnaps[1];
				slabPoints[14] = slabPoints[13];

				slabPoints[15] = facadeSnaps[2];
				slabPoints[16] = slabPoints[15];

				slabPoints[17] = facadeSnaps[3];
				slabPoints[18] = slabPoints[17];

				slabPoints[19] = facadeSnaps[4];
				slabPoints[20] = slabPoints[19];

				slabPoints[21] = facadeSnaps[5];

				slabPoints[22] = facadeSnaps[0];
				slabPoints[23] = facadeSnaps[1];

				int k = 0;
				for (int j = parentColumn->snapSlabpoints[i].size() / 2; j < parentColumn->snapSlabpoints[i].size(); j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					slabPoints[k + 24] = v;
					k++;
				}

				slabPoints[34] = facadeSnaps[0] - zVector(0, 0, 0.1);
				slabPoints[35] = parentColumn->snapSlabpoints[i][11];

				for (int j = 0; j < parentColumn->snapSlabpoints[i].size() / 2; j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					slabPoints[j + 36] = v;
				}

				t = parentColumn->snapSlabpoints[i][10];
				slabPoints[46] = x - zVector(0, 0, 0.2);;
				slabPoints[47] = t;

				/////////////////////////////////////////////
				/////////add point to mesh
				for (auto v : slabPoints)
				{
					pointArray.push_back(v);
				}

				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 12; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 3 * pInContour; j += pInContour) //jumps between layers (3 is total of layers:6 minus 1)
				{
					/*if (layerCount == 1)
					{
						layerCount++;
						continue;
					}*/

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
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
		}

		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			zFnMesh fnInMesh(slabMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.smoothMesh(1, false);
		}
	}

	ZSPACE_INLINE void zAgSlab::createTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		int numPoints = 0;
		for (int i = 0; i < centerVecs.size(); i++)
		{
			if (parentColumn->boundaryArray[i] == zBoundary::zInterior && parentColumn->snapSlabpoints[i].size() > 0)
			{
				for (auto v : parentColumn->snapSlabpoints[i])
				{
					v.z = parentColumn->position.z;
					pointArray.push_back(v);
				}
				/////new points 
				zPointArray newPoints;
				newPoints.assign(16, zVector());

				newPoints[0] = midPoints[i];
				newPoints[1] = centerVecs[i];
				newPoints[2] = newPoints[1];
				newPoints[3] = midPoints[(i + 1) % midPoints.size()];

				newPoints[4] = midPoints[i] + zVector(0,0,-0.5);
				newPoints[5] = centerVecs[i] + zVector(0, 0, -0.5);
				newPoints[6] = newPoints[1] + zVector(0, 0, -0.5);
				newPoints[7] = midPoints[(i + 1) % midPoints.size()] + zVector(0, 0, -0.5);

				newPoints[8] = parentColumn->snapSlabpoints[i][4];
				newPoints[9] = parentColumn->snapSlabpoints[i][5];
				newPoints[10] = parentColumn->snapSlabpoints[i][6];
				newPoints[11] = parentColumn->snapSlabpoints[i][7];

				newPoints[12] = parentColumn->snapSlabpoints[i][0];
				newPoints[13] = parentColumn->snapSlabpoints[i][1];
				newPoints[14] = parentColumn->snapSlabpoints[i][2];
				newPoints[15] = parentColumn->snapSlabpoints[i][3];

				/////////add point to mesh
				for (auto v : newPoints)
				{
					pointArray.push_back(v);
				}


				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 4; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{
					/*if (layerCount == 1)
					{
						layerCount++;
						continue;
					}*/

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
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

			if (parentColumn->boundaryArray[i] == zBoundary::zEdge && parentColumn->snapSlabpoints[i].size() > 0)
			{
				zVector x, y; //y is edge for facade

				if (parentColumn->boundaryArray[(i + 1) % centerVecs.size()] == zBoundary::zInterior)
				{
					y = midPoints[i];
					x = midPoints[(i + 1) % midPoints.size()];
				}
				else
				{
					x = midPoints[i];
					y = midPoints[(i + 1) % midPoints.size()];
				}

				///////////////facade snap points
				zPointArray facadeSnaps;
				facadeSnaps.assign(6, zVector());

				zVector inDir = y - centerVecs[i];
				zVector outDir = centerVecs[i] - x;
				outDir.normalize();

				facadeSnaps[0] = centerVecs[i] + outDir * 1.1;
				facadeSnaps[1] = centerVecs[i] + (outDir * 1) + (inDir * 0.25);
				facadeSnaps[2] = centerVecs[i] + (outDir * 0.75) + (inDir * 0.6);
				facadeSnaps[3] = centerVecs[i] + (outDir * 0.2) + (inDir * 0.85);
				facadeSnaps[4] = y + (outDir * -0.5);
				facadeSnaps[5] = y + (outDir * -1);

				///////////////////////////////////////////
				/////new points 
				zPointArray slabPoints;
				slabPoints.assign(48, zVector());

				//
				for (int j = 0; j < parentColumn->snapSlabpoints[i].size()/2; j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					v.z = parentColumn->position.z;
					slabPoints[j] = v;
				}

				zVector t = parentColumn->snapSlabpoints[i][10];
				t.z = parentColumn->position.z;
				slabPoints[10] = x;
				slabPoints[11] = t;

				slabPoints[12] = t;

				slabPoints[13] = facadeSnaps[1];
				slabPoints[14] = slabPoints[13];

				slabPoints[15] = facadeSnaps[2];
				slabPoints[16] = slabPoints[15];

				slabPoints[17] = facadeSnaps[3];
				slabPoints[18] = slabPoints[17];

				slabPoints[19] = facadeSnaps[4];
				slabPoints[20] = slabPoints[19];

				slabPoints[21] = facadeSnaps[5];

				slabPoints[22] = facadeSnaps[0];
				slabPoints[23] = facadeSnaps[1];

				int k = 0;
				for (int j = parentColumn->snapSlabpoints[i].size() / 2; j < parentColumn->snapSlabpoints[i].size(); j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					slabPoints[k + 24] = v;
					k++;
				}

				slabPoints[34] = facadeSnaps[0] + outDir * -0.1 - zVector(0, 0, 0.5);
				slabPoints[35] = parentColumn->snapSlabpoints[i][11];

				for (int j = 0; j < parentColumn->snapSlabpoints[i].size() / 2; j++)
				{
					zVector v = parentColumn->snapSlabpoints[i][j];
					slabPoints[j + 36] = v;
				}

				t = parentColumn->snapSlabpoints[i][10];
				slabPoints[46] = x - zVector(0, 0, 0.5);;
				slabPoints[47] = t;

				/////////////////////////////////////////////
				/////////add point to mesh
				for (auto v : slabPoints)
				{
					pointArray.push_back(v);
				}

				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 12; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 3 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{
					/*if (layerCount == 1)
					{
						layerCount++;
						continue;
					}*/

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
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
		}

		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			zFnMesh fnInMesh(slabMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.triangulate();
			//fnInMesh.smoothMesh(1, false);
		}
		
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zAgSlab::displaySlab(bool showSlab)
	{
		slabMeshObj.setDisplayObject(showSlab);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zAgSlab::addObjsToModel()
	{
		model->addObject(slabMeshObj);
		slabMeshObj.setDisplayElements(false, true, true);
	}

#endif

}