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


#include<headers/zArchGeom/zAgRoof.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgRoof::zAgRoof(){}

	ZSPACE_INLINE  zAgRoof::zAgRoof(zPointArray&_vertexCorners, bool _isFacade)
	{
		vertexCorners = _vertexCorners;
		isFacade = _isFacade;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgRoof::~zAgRoof() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zAgRoof::createRhwc()
	{
		

		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		if (isFacade)
		{
			zFnMesh fnInMesh(roofMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
		}
		else
		{
			if (vertexCorners.size() == 0) return;

			for (int i = 0; i < vertexCorners.size(); i++)
			{
				pointArray.push_back(vertexCorners[i]);
				polyConnect.push_back(i);
			}

			polyCount.push_back(4);

			zFnMesh fnInMesh(roofMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
		}
	}

	ZSPACE_INLINE void zAgRoof::createTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		if (vertexCorners.size() == 0) return;

		if (!isFacade) 
		{
			for (int i = 0; i < vertexCorners.size(); i++)
			{
				pointArray.push_back(vertexCorners[i]);
				polyConnect.push_back(i);
			}

			polyCount.push_back(4);
		}

		else
		{
			zVector center = (vertexCorners[0] + vertexCorners[2]) / 2;
			zVector outDir = vertexCorners[1] - vertexCorners[0];

			int numPoints = 0;
			for (int i = 0;  i < vertexCorners.size(); i+=2)
			{
				zVector inDir = vertexCorners[i] - center;

				zPointArray facadePoints;
				zPointArray roofPoints;
				facadePoints.assign(9, zVector());

				facadePoints[0] = center + outDir * -1.1;
				facadePoints[1] = center + (outDir * -1) + (inDir * 0.25);
				facadePoints[2] = facadePoints[1];
				facadePoints[3] = center + (outDir * -0.75) + (inDir * 0.6);
				facadePoints[4] = facadePoints[3];
				facadePoints[5] = center + (outDir * -0.2) + (inDir * 0.85);
				facadePoints[6] = facadePoints[5];
				facadePoints[7] = vertexCorners[i] + (outDir * 0.5);
				facadePoints[8] = facadePoints[7];

				roofPoints.push_back(center + outDir);


				for (int j = 0; j < facadePoints.size(); j++)
				{
					roofPoints.push_back(facadePoints[j]);
				}

				roofPoints.push_back(vertexCorners[i] + outDir);



				for (int j = 0; j < roofPoints.size(); j++)
				{
					pointArray.push_back(roofPoints[j]);
				}

				////update with cleaner solution
				if (i == 0)
				{
					for (int j = 0; j < roofPoints.size(); j++)
					{
						polyConnect.push_back(j + numPoints);
					}
				}
				else
				{
					for (int j = roofPoints.size(); j != 0; j--)
					{
						polyConnect.push_back(j - 1 + numPoints);
					}

				}
			
				polyCount.push_back(roofPoints.size());

				numPoints = pointArray.size();
			}
		}
		
		zFnMesh fnInMesh(roofMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);
		//fnInMesh.triangulate();
		fnInMesh.extrudeMesh(0.1, roofMeshObj, false);
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zAgRoof::displayRoof(bool showRoof)
	{
		roofMeshObj.setDisplayObject(showRoof);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zAgRoof::addObjsToModel()
	{
		model->addObject(roofMeshObj);
		roofMeshObj.setDisplayElements(false, true, true);
	}

#endif
}