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



	zAgSlab::zAgSlab(zObjMesh&_inMeshObj, zVectorArray&_centerVecs, zVectorArray&_midPoints, zAgColumn&_parentColumn)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(*inMeshObj);
		parentColumn = &_parentColumn;
		centerVecs = _centerVecs;
		midPoints = _midPoints;
	}

	void zAgSlab::createSlabByType(zStructureType & _structureType)
	{
		structureType = _structureType;

		if (structureType == zStructureType::zRHWC) createTimberSlab(); //CHANGE FOR TEST
		else if (structureType == zStructureType::zDigitalTimber) createTimberSlab();
	}

	void zAgSlab::createRhwcSlab()
	{
		//zPointArray pointArray;
		//zIntArray polyConnect;
		//zIntArray polyCount;

		//if (!parentColumn) return;

		//////////
		//pointArray.push_back(parentColumn->b);
		//pointArray.push_back(parentColumn->a);
		//pointArray.push_back(yCenter);
		//pointArray.push_back(center);

		//pointArray.push_back(parentColumn->b);
		//pointArray.push_back(center);
		//pointArray.push_back(xCenter);
		//pointArray.push_back(parentColumn->c);

		//pointArray.push_back(parentColumn->a);
		//pointArray.push_back(parentColumn->position);
		//pointArray.push_back(yCenter);

		//pointArray.push_back(parentColumn->position);
		//pointArray.push_back(parentColumn->c);
		//pointArray.push_back(xCenter);

		//pointArray.push_back(parentColumn->position);
		//pointArray.push_back(xCenter);
		//pointArray.push_back(center);
		//pointArray.push_back(yCenter);



		///////////
		//polyCount.push_back(4);
		//polyCount.push_back(4);
		//polyCount.push_back(3);
		//polyCount.push_back(3);
		//polyCount.push_back(4);



		/////////////
		//polyConnect.push_back(0);
		//polyConnect.push_back(1);
		//polyConnect.push_back(2);
		//polyConnect.push_back(3);

		//polyConnect.push_back(4);
		//polyConnect.push_back(5);
		//polyConnect.push_back(6);
		//polyConnect.push_back(7);

		//polyConnect.push_back(8);
		//polyConnect.push_back(9);
		//polyConnect.push_back(10);

		//polyConnect.push_back(11);
		//polyConnect.push_back(12);
		//polyConnect.push_back(13);

		//polyConnect.push_back(14);
		//polyConnect.push_back(15);
		//polyConnect.push_back(16);
		//polyConnect.push_back(17);


		//fnInMesh.create(pointArray, polyCount, polyConnect);
		//fnInMesh.smoothMesh(2, false);
	}

	void zAgSlab::createTimberSlab()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		int numPoints = 0;
		for (int i = 0; i < centerVecs.size(); i++)
		{
			if (parentColumn->boundaryArray[i] == zBoundary::zInterior && parentColumn->snapSlabpoints[i].size() > 0)
			{
				printf("\n slab snap points count: %i ", parentColumn->snapSlabpoints[i].size());

				for (auto v : parentColumn->snapSlabpoints[i])
				{
					v.z = parentColumn->position.z;
					pointArray.push_back(v);
				}
				/////
				zPointArray newPoints;
				newPoints.assign(4, zVector());

				newPoints[0] = midPoints[i];
				newPoints[1] = centerVecs[i];
				newPoints[2] = newPoints[1];
				newPoints[3] = midPoints[(i + 1) % midPoints.size()];

				for (auto v : newPoints)
				{
					pointArray.push_back(v);
				}


				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 4; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 2 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
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
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//fnInMesh.extrudeMesh(-0.1, *inMeshObj, false);
		}
		
	}

	//void zAgSlab::createTimberSlab()
	//{
	//	zPointArray pointArray;
	//	zIntArray polyConnect;
	//	zIntArray polyCount;

	//	for (int i = 0; i < centerVecs.size(); i++)
	//	{

	//		if (parentColumn->axisAttributes[i])
	/*{
		pointArray.push_back(parentColumn->snapSlabpoints[i][0]);
		pointArray.push_back(midPoints[i]);
		pointArray.push_back(centerVecs[i]);
		pointArray.push_back(parentColumn->snapSlabpoints[i][1]);

		pointArray.push_back(parentColumn->snapSlabpoints[i][1]);
		pointArray.push_back(centerVecs[i]);
		pointArray.push_back(midPoints[(i + 1) % centerVecs.size()]);

		polyConnect.push_back(0 + (7 * i));
		polyConnect.push_back(1 + (7 * i));
		polyConnect.push_back(2 + (7 * i));
		polyConnect.push_back(3 + (7 * i));

		polyConnect.push_back(4 + (7 * i));
		polyConnect.push_back(5 + (7 * i));
		polyConnect.push_back(6 + (7 * i));

		polyCount.push_back(4);
		polyCount.push_back(3);
	}

			else
			{
			pointArray.push_back(parentColumn->snapSlabpoints[i][0]);
			pointArray.push_back(midPoints[i]);
			pointArray.push_back(parentColumn->snapSlabpoints[i][1]);

			pointArray.push_back(parentColumn->snapSlabpoints[i][1]);
			pointArray.push_back(midPoints[i]);
			pointArray.push_back(centerVecs[i]);
			pointArray.push_back(midPoints[(i + 1) % centerVecs.size()]);

			polyConnect.push_back(0 + (7 * i));
			polyConnect.push_back(1 + (7 * i));
			polyConnect.push_back(2 + (7 * i));

			polyConnect.push_back(3 + (7 * i));
			polyConnect.push_back(4 + (7 * i));
			polyConnect.push_back(5 + (7 * i));
			polyConnect.push_back(6 + (7 * i));

			polyCount.push_back(3);
			polyCount.push_back(4);

			}*/


	//	}

	//	fnInMesh.create(pointArray, polyCount, polyConnect);
	//	fnInMesh.extrudeMesh(-0.1, *inMeshObj, false);
	//}
}