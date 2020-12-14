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


#include<headers/zToolsets/geometry/zTsSDFBridge.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSDFBridge::zTsSDFBridge() {}

	ZSPACE_INLINE zTsSDFBridge::zTsSDFBridge(zObjMesh& _o_guideMesh)
	{
		o_guideMesh = &_o_guideMesh;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSDFBridge::~zTsSDFBridge() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsSDFBridge::createCutPlaneMesh(double width)
	{

		zFnMesh fnGuideMesh(*o_guideMesh);

		globalVertices.clear();

		guideEdge_globalVertex.clear();
		guideEdge_globalVertex.assign(-1, fnGuideMesh.numEdges());

		zPointArray positions;
		zIntArray pCounts, pConnects;
					

		// add face centers to global vertices

		int n_gV = 0;
	
		for (zItMeshFace f(*o_guideMesh); !f.end(); f++)
		{
			zPoint p = f.getCenter();
			zVector n = f.getNormal();

			globalVertices.push_back(zGlobalVertex());
			globalVertices[n_gV].pos = p + n * width * 0.5;;
			n_gV++;

			globalVertices.push_back(zGlobalVertex());
			globalVertices[n_gV].pos = p - n * width * 0.5;;
			n_gV++;
		}

		// add boundary edge centers to global vertices
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (e.onBoundary())
			{
				zPoint p = e.getCenter();

				zItMeshFaceArray efaces;
				e.getFaces(efaces);
				zVector n = efaces[0].getNormal();
				
				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p + n * width * 0.5;;
				n_gV++;

				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p - n * width * 0.5;;
				n_gV++;
			}

		}

		//create plane mesh

	}

}