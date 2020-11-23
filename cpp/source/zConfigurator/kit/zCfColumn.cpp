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


#include <headers/zConfigurator/kit/zCfColumn.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zCfColumn::zCfColumn() 
	{
		componentType = zCfComponentType::zConfigfColumn;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zCfColumn::~zCfColumn() {}

	//---- OVERRIDE METHODS
		
	ZSPACE_INLINE void zCfColumn::create(zTransformationMatrix& _transform, string type)
	{
		transform = _transform;

	}

	void zCfColumn::createSplitColumn(zTransformationMatrix& _transform, float offsetX, float offsetY)
	{
		transform = _transform;

		// points
		zPointArray positions;

		// polyconnects
		zIntArray pConnects;
		zIntArray pCounts;
			
		zFloat4 s;
		_transform.getScale(s);

		zVector X = _transform.getX();
		zVector Y = _transform.getY();
		zVector Z = zVector(0,0,1);
				

		// first column 
		zPoint origin0 = _transform.getTranslation();
		origin0 += Y * offsetY;
		origin0 -= X * offsetX;
		origin0 -= X * s[0];

		for (int i = 0; i < 2; i++)
		{
			if (i == 1)
			{
				origin0 += X * s[0];
				origin0 += X * offsetX * 2;
			}

			int vCount = positions.size();

			positions.push_back(origin0);
			positions.push_back(origin0 + X * s[0]);
			positions.push_back(origin0 + Y * s[1]);
			positions.push_back(origin0 + X * s[0] + Y * s[1]);

			positions.push_back(positions[vCount + 0] + Z * s[2]);
			positions.push_back(positions[vCount + 1] + Z * s[2]);
			positions.push_back(positions[vCount + 2] + Z * s[2]);
			positions.push_back(positions[vCount + 3] + Z * s[2]);

			pConnects.push_back(vCount + 0); pConnects.push_back(vCount + 2); pConnects.push_back(vCount + 3);  pConnects.push_back(vCount + 1);
			pConnects.push_back(vCount + 0); pConnects.push_back(vCount + 1); pConnects.push_back(vCount + 5);  pConnects.push_back(vCount + 4);
			pConnects.push_back(vCount + 1); pConnects.push_back(vCount + 3); pConnects.push_back(vCount + 7);  pConnects.push_back(vCount + 5);
			pConnects.push_back(vCount + 3); pConnects.push_back(vCount + 2); pConnects.push_back(vCount + 6);  pConnects.push_back(vCount + 7);
			pConnects.push_back(vCount + 2); pConnects.push_back(vCount + 0); pConnects.push_back(vCount + 4);  pConnects.push_back(vCount + 6);
			pConnects.push_back(vCount + 4); pConnects.push_back(vCount + 5); pConnects.push_back(vCount + 7);  pConnects.push_back(vCount + 6);

			pCounts.push_back(4); pCounts.push_back(4); pCounts.push_back(4);
			pCounts.push_back(4); pCounts.push_back(4); pCounts.push_back(4);
		}

		zFnMesh fnMesh(componentObj);
		fnMesh.create(positions, pCounts, pConnects);

	}
		
		

}