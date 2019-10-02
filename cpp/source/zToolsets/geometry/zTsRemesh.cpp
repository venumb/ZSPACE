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


#include<headers/zToolsets/geometry/zTsRemesh.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsRemesh::zTsRemesh() {}

	ZSPACE_INLINE zTsRemesh::zTsRemesh(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsRemesh::~zTsRemesh() {}

	//---- REMESH METHODS

	ZSPACE_INLINE void zTsRemesh::splitLongEdges(double maxEdgeLength)
	{
		/*for (int i = 0; i < meshObj->mesh.halfEdges.size(); i += 2)
		{
			if (meshObj->mesh.edgeActive[i])
			{
				double eLength = fnMesh.getEdgelength(i);

				while (eLength > maxEdgeLength)
				{
					fnMesh.splitEdge(i, 0.5, true);
					eLength = fnMesh.getEdgelength(i);

				}
			}
		}*/
	}

	ZSPACE_INLINE void collapseShortEdges(double minEdgeLength, double maxEdgeLength)
	{
		//int finished = false;

		//vector<bool> edgeFinished;

		//while (!finished)
		//{
		//	for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
		//	{
		//		if (meshObj->mesh.edgeActive[i])
		//		{

		//			double eLength = fnMesh.getEdgelength(i);

		//			if (eLength < minEdgeLength)
		//			{
		//				int v1 = meshObj->mesh.edges[i].getVertex()->getVertexId();
		//				int v2 = meshObj->mesh.edges[i + 1].getVertex()->getVertexId();

		//				zVector pos = meshObj->mesh.vertexPositions[v2]; /*(meshObj->mesh.vertexPositions[v1] + meshObj->mesh.vertexPositions[v2]) * 0.5;*/

		//				vector<int> cVertsV1;
		//				fnMesh.getConnectedVertices(v1, zVertexData, cVertsV1);

		//				bool collapse_ok = true;

		//				for (int j = 0; j < cVertsV1.size(); j++)
		//				{
		//					if (pos.distanceTo(meshObj->mesh.vertexPositions[cVertsV1[j]]) > maxEdgeLength)
		//					{
		//						collapse_ok = false;
		//						break;
		//					}
		//				}

		//				if (collapse_ok)
		//				{
		//					//printf("\n working %i \n", i);
		//					fnMesh.collapseEdge(i, 0.5, false);

		//					//printMesh(inMesh);
		//				}
		//			}



		//		}
		//	}
		//}


	}

	ZSPACE_INLINE void equalizeValences()
	{
		/*for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
		{
			if (meshObj->mesh.edgeActive[i])
			{
				if (fnMesh.onBoundary(i, zEdgeData) || fnMesh.onBoundary(i + 1, zEdgeData)) continue;

				vector<int> fVerts;
				fnMesh.getVertices(fnMesh.getFaceIndex(i) , zFaceData, fVerts);

				vector<int> sym_fVerts;
				fnMesh.getVertices(fnMesh.getFaceIndex(i+1), zFaceData, sym_fVerts);

				if (fVerts.size() != 3 || sym_fVerts.size() != 3)
				{
					printf("\n Cannot flip edge %i as it not shared by two Triangles.", i);
					continue;
				}

				int v1 = fnMesh.getEndVertexIndex(i);
				int v2 = fnMesh.getEndVertexIndex(i+1);

				int nextEdge = fnMesh.getNextIndex(i);
				int symNextEdge = fnMesh.getNextIndex(i+1);

				int v3 = fnMesh.getEndVertexIndex(nextEdge);
				int v4 = fnMesh.getEndVertexIndex(symNextEdge);

				int tarVal_v1 = (fnMesh.onBoundary(v1, zVertexData)) ? 4 : 6;
				int tarVal_v2 = (fnMesh.onBoundary(v2, zVertexData)) ? 4 : 6;
				int tarVal_v3 = (fnMesh.onBoundary(v3, zVertexData)) ? 4 : 6;
				int tarVal_v4 = (fnMesh.onBoundary(v4, zVertexData)) ? 4 : 6;

				int val_v1 = fnMesh.getVertexValence(v1);
				int val_v2 = fnMesh.getVertexValence(v2);
				int val_v3 = fnMesh.getVertexValence(v3);
				int val_v4 = fnMesh.getVertexValence(v4);

				int dev_pre = abs(val_v1 - tarVal_v1) + abs(val_v2 - tarVal_v2) + abs(val_v3 - tarVal_v3) + abs(val_v4 - tarVal_v4);

				fnMesh.flipTriangleEdge(i);

				val_v1 = fnMesh.getVertexValence(v1);
				val_v2 = fnMesh.getVertexValence(v2);
				val_v3 = fnMesh.getVertexValence(v3);
				val_v4 = fnMesh.getVertexValence(v4);


				int dev_post = abs(val_v1 - tarVal_v1) + abs(val_v2 - tarVal_v2) + abs(val_v3 - tarVal_v3) + abs(val_v4 - tarVal_v4);

				if (dev_pre <= dev_post) fnMesh.flipTriangleEdge(i);
			}
		}*/
	}

	ZSPACE_INLINE void tangentialRelaxation()
	{
		/*for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (meshObj->mesh.vertexActive[i])
			{
				vector<int> cVerts;
				fnMesh.getConnectedVertices(i, zVertexData, cVerts);

				vector<zVector> cVerts_Pos;
				vector<double> weights;

				for (int j = 0; j < cVerts.size(); j++)
				{
					cVerts_Pos.push_back(fnMesh.getVertexPosition(cVerts[j]));
					weights.push_back(1.0);
				}

				zVector v_bary = coreUtils.getBaryCenter(cVerts_Pos, weights);

				zVector v_norm = fnMesh.getVertexNormal(i);
				zVector v_pos = fnMesh.getVertexPosition(i);

				double dotP = v_norm * (v_pos - v_bary);
				zVector newPos = v_bary + (v_norm * dotP);

				fnMesh.setVertexPosition(i, newPos);

			}
		}*/

	}

}