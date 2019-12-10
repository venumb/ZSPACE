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

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgRoof::~zAgRoof() {}

	//---- SET METHODS

	ZSPACE_INLINE  zAgRoof::zAgRoof(zPointArray&_corners, bool _isFacade)
	{
		
		corners = _corners;			
		isFacade = _isFacade;

		printf("\n corners constructor size: %i", corners.size());
	}


	ZSPACE_INLINE void zAgRoof::createRhwc()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		if (corners.size() == 0) return;

		printf("\n corners rhwc size: %i", corners.size());


		for (int i = 0; i < corners.size(); i++)
		{
			pointArray.push_back(corners[i]);
			polyConnect.push_back(i);
		}

		polyCount.push_back(4);
		printf("\n polyconnect: %i %i %i", pointArray.size(), polyConnect.size(), polyCount.size());
		zFnMesh fnInMesh(inMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);

	}

	ZSPACE_INLINE void zAgRoof::createTimber()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		if (corners.size() == 0) return;

		printf("\n corners timber size: %i", corners.size());


		if (!isFacade) 
		{
			for (int i = 0; i < corners.size(); i++)
			{
				pointArray.push_back(corners[i]);
				polyConnect.push_back(i);
			}

			polyCount.push_back(4);
		}

		else
		{
			zVector center = (corners[0] + corners[2]) / 2;
			zVector outDir = corners[1] - corners[0];

			int numPoints = 0;
			for (int i = 0;  i < corners.size(); i+=2)
			{
				zVector inDir = corners[i] - center;

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
				facadePoints[7] = corners[i] + (outDir * 0.5);
				facadePoints[8] = facadePoints[7];

				roofPoints.push_back(center + outDir);


				for (int j = 0; j < facadePoints.size(); j++)
				{
					roofPoints.push_back(facadePoints[j]);
				}

				roofPoints.push_back(corners[i] + outDir);



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
		
		zFnMesh fnInMesh(inMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);
		fnInMesh.extrudeMesh(0.1, inMeshObj, false);
	}

	ZSPACE_INLINE void zAgRoof::displayRoof(bool showRoof)
	{
		inMeshObj.setShowObject(showRoof);
	}

	ZSPACE_INLINE void zAgRoof::addObjsToModel()
	{
		model->addObject(inMeshObj);
		inMeshObj.setShowElements(false, true, true);
	}

}