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


#include <headers/zConfigurator/kit/zCfWall.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zCfWall::zCfWall()
	{
		componentType = zCfComponentType::zConfigfWall;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zCfWall::~zCfWall() {}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zCfWall::create(zTransformationMatrix& _transform, string type)
	{
		transform = _transform;

	}

	ZSPACE_INLINE void zCfWall::createGuideGraphs(zTransformationMatrix& _transform, float offsetX, float offsetY, float spacing, zFloat2& cornerOffsets)
	{
		transform = _transform;

		o_guideGraphs.clear();
		o_guideGraphs.assign(4, zObjGraph());

		// points
		zPointArray positions;

		// edge connects
		zIntArray eConnects;
		
		zFloat4 s;
		_transform.getScale(s);

		zVector X = _transform.getX();
		zVector Y = _transform.getY();
		zVector Z = zVector(0, 0, 1);

		zPoint origin0 = _transform.getTranslation();
		origin0 += Y * offsetY;
		origin0 -= X * offsetX;
		origin0 -= X * s[0];

		// check if length of wall in along X or Y
		int numPoints = (s[0] > s[1]) ? floor(s[0] / spacing) + 1 : floor(s[1] / spacing) + 1;

		for (int j = 0; j < 4; j++)
		{
			positions.clear();
			eConnects.clear();

			for (int i = 0; i < numPoints; i++)
			{
				zPoint p = origin0;
				p += (s[0] > s[1]) ? X * i * spacing : Y * i * spacing;

				if (j == 1)
				{
					p += Z * s[2];
				}
				if (j == 2)
				{
					p += (s[0] > s[1]) ? Y * s[0] : X * s[0];
				}
				if (j == 3)
				{
					p += Z * s[2];
					p += (s[0] > s[1]) ? Y * s[0] : X * s[0];
				}
				
				positions.push_back(p);

				if (i > 0)
				{
					eConnects.push_back(i - 1);
					eConnects.push_back(i);
				}

			}

			zFnGraph fnGraph(o_guideGraphs[j]);
			fnGraph.create(positions, eConnects);
						
		}

	}

	ZSPACE_INLINE void zCfWall::createSplitColumn(zTransformationMatrix& _transform, float offsetX, float offsetY)
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
		zVector Z = zVector(0, 0, 1);


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

		zFnMesh fnMesh(o_component);
		fnMesh.create(positions, pCounts, pConnects);

	}

	//---- GET METHODS

	ZSPACE_INLINE zObjGraph* zCfWall::getRawGuideGraph(int id)
	{
		if (id >= 4 ) throw std::invalid_argument(" error: null pointer.");

		return &o_guideGraphs[id];
	}

}