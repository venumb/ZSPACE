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


#include<headers/zToolsets/geometry/zTsVariableExtrude.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsVariableExtrude::zTsVariableExtrude() {}

	ZSPACE_INLINE zTsVariableExtrude::zTsVariableExtrude(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsVariableExtrude::~zTsVariableExtrude() {}

	ZSPACE_INLINE zObjMesh zTsVariableExtrude::getVariableFaceOffset(bool keepExistingFaces, bool assignColor, float minVal, float maxVal, bool useVertexColor)
	{

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> faceColors;

		vector<zVector> fCenters;
		fnMesh.getCenters(zFaceData, fCenters);

		// append inmesh positions
		if (keepExistingFaces)
		{
			fnMesh.getVertexPositions(positions);
		}

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{


			vector<zItMeshVertex> fVerts;
			f.getVertices(fVerts);

			zColor faceCol = f.getColor();
			float faceVal = faceCol.r;
			zColor newCol(faceCol.r, 0, 0, 1);

			double extrudeVal = coreUtils.ofMap(faceVal, 0.0f, 1.0f, minVal, maxVal);

			// get current size of positions
			int numVerts = positions.size();

			// append new positions
			for (auto &fV : fVerts)
			{
				zVector pos = fV.getPosition();
				zVector dir = fCenters[f.getId()] - pos;
				double len = dir.length();
				dir.normalize();

				if (useVertexColor)
				{
					zColor vertexCol = fV.getColor();

					extrudeVal = coreUtils.ofMap(vertexCol.r, 0.0f, 1.0f, minVal, maxVal);
				}

				zVector newPos = pos + dir * len * extrudeVal;

				positions.push_back(newPos);
			}

			// compute polyconnects and polycounts
			if (keepExistingFaces)
			{
				for (int j = 0; j < fVerts.size(); j++)
				{
					int currentId = j;
					int nextId = (j + 1) % fVerts.size();

					polyConnects.push_back(fVerts[currentId].getId());
					polyConnects.push_back(fVerts[nextId].getId());
					polyConnects.push_back(numVerts + nextId);
					polyConnects.push_back(numVerts + currentId);


					polyCounts.push_back(4);

					if (assignColor) faceColors.push_back(newCol);

				}

			}
			else
			{

				for (int j = 0; j < fVerts.size(); j++)
				{
					int currentId = j;
					polyConnects.push_back(numVerts + currentId);
				}

				if (assignColor) faceColors.push_back(newCol);

				polyCounts.push_back(fVerts.size());

			}

		}


		zObjMesh out;
		zFnMesh temp(out);

		if (positions.size() > 0)
		{
			temp.create(positions, polyCounts, polyConnects);

			if (assignColor) temp.setFaceColors(faceColors, true);
		}

		return out;
	}

	ZSPACE_INLINE zObjMesh zTsVariableExtrude::getVariableBoundaryOffset(bool keepExistingFaces, bool assignColor, float minVal, float maxVal)
	{

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> vertexColors;


		vector<zVector> fCenters;
		fnMesh.getCenters(zFaceData, fCenters);

		if (keepExistingFaces)
		{
			vector<vector<int>> inVertex_newVertex;

			for (zItMeshVertex v(*meshObj); !v.end(); v++)
			{
				vector<int> temp;

				if (v.onBoundary())
				{

					temp.push_back(positions.size());


					positions.push_back(v.getPosition());

					if (assignColor)
					{
						vertexColors.push_back(v.getColor());
					}

					float vertexVal = v.getColor().r;
					zColor newCol(v.getColor().r, 0, 0, 1);

					double extrudeVal = coreUtils.ofMap(vertexVal, 0.0f, 1.0f, minVal, maxVal);


					zItMeshHalfEdge vEdge;

					vector<zItMeshHalfEdge> cEdges;
					v.getConnectedHalfEdges(cEdges);

					for (auto &he : cEdges)
					{
						if (he.onBoundary())
						{
							vEdge = he;
						}
					}

					//if (vEdge == NULL) continue;

					zItMeshVertex next = vEdge.getVertex();;
					zItMeshVertex prev = vEdge.getPrev().getSym().getVertex();

					zVector vNorm = v.getNormal();

					zVector Ori = v.getPosition();;

					zVector v1 = Ori - prev.getPosition();
					v1.normalize();

					zVector n1 = v1 ^ vNorm;
					n1.normalize();

					zVector v2 = next.getPosition() - Ori;
					v2.normalize();

					zVector n2 = v2 ^ vNorm;
					n2.normalize();

					v1 = v1 ^ v2;
					zVector v3 = (n1 + n2);

					v3 *= 0.5;
					v3.normalize();



					double cs = v3 * v2;
					double length = extrudeVal;


					zVector a1 = v2 * cs;
					zVector a2 = v3 - a1;

					double alpha = 0;
					if (a2.length() > 0) alpha = sqrt(a2.length() * a2.length());

					if (cs < 0 && a2.length() > 0) alpha *= -1;

					if (alpha > 0) length /= alpha;

					zVector offPos = Ori + (v3 * length);

					temp.push_back(positions.size());
					positions.push_back(offPos);
					if (assignColor) vertexColors.push_back(v.getColor());



				}

				inVertex_newVertex.push_back(temp);

			}

			// poly connects 
			for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
			{
				if (he.onBoundary())
				{
					vector<int> eVerts;
					he.getVertices(eVerts);

					polyConnects.push_back(inVertex_newVertex[eVerts[0]][0]);
					polyConnects.push_back(inVertex_newVertex[eVerts[1]][0]);
					polyConnects.push_back(inVertex_newVertex[eVerts[1]][1]);
					polyConnects.push_back(inVertex_newVertex[eVerts[0]][1]);

					polyCounts.push_back(4);

				}
			}

		}


		if (!keepExistingFaces)
		{

			for (zItMeshVertex v(*meshObj); !v.end(); v++)
			{
				vector<int> temp;

				if (v.onBoundary())
				{

					float vertexVal = v.getColor().r;
					zColor newCol(v.getColor().r, 0, 0, 1);

					double extrudeVal = coreUtils.ofMap(vertexVal, 0.0f, 1.0f, minVal, maxVal);


					zItMeshHalfEdge vEdge = v.getHalfEdge();

					vector<zItMeshHalfEdge> cEdges;
					v.getConnectedHalfEdges(cEdges);

					for (auto &he : cEdges)
					{
						if (he.onBoundary())
						{
							vEdge = he;
						}
					}

					//if (vEdge == NULL) continue;


					zItMeshVertex next = vEdge.getVertex();;
					zItMeshVertex prev = vEdge.getPrev().getSym().getVertex();

					zVector vNorm = v.getNormal();

					zVector Ori = v.getPosition();

					zVector v1 = Ori - prev.getPosition();
					v1.normalize();

					zVector n1 = v1 ^ vNorm;
					n1.normalize();

					zVector v2 = next.getPosition() - Ori;
					v2.normalize();

					zVector n2 = v2 ^ vNorm;
					n2.normalize();

					v1 = v1 ^ v2;
					zVector v3 = (n1 + n2);

					v3 *= 0.5;
					v3.normalize();



					double cs = v3 * v2;
					double length = extrudeVal;


					zVector a1 = v2 * cs;
					zVector a2 = v3 - a1;

					double alpha = 0;
					if (a2.length() > 0) alpha = sqrt(a2.length() * a2.length());

					if (cs < 0 && a2.length() > 0) alpha *= -1;

					if (alpha > 0) length /= alpha;

					zVector offPos = Ori + (v3 * length);


					positions.push_back(offPos);
					if (assignColor) vertexColors.push_back(v.getColor());

				}

				else
				{
					positions.push_back(v.getPosition());
					if (assignColor) vertexColors.push_back(v.getColor());
				}


			}

			// poly connects 
			for (zItMeshFace f(*meshObj); !f.end(); f++)
			{

				vector<int> fVerts;
				f.getVertices(fVerts);

				for (int j = 0; j < fVerts.size(); j++)
				{
					polyConnects.push_back(fVerts[j]);
				}


				polyCounts.push_back(fVerts.size());


			}

		}

		zObjMesh out;
		zFnMesh tempFn(out);

		if (positions.size() > 0)
		{
			tempFn.create(positions, polyCounts, polyConnects);

			if (assignColor) tempFn.setVertexColors(vertexColors, true);
		}

		return out;
	}

	ZSPACE_INLINE zObjMesh zTsVariableExtrude::getVariableFaceThicknessExtrude(bool assignColor, float minVal, float maxVal)
	{



		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> vertexColors;


		int numVerts = fnMesh.numVertices();

		// append inmesh positions
		fnMesh.getVertexPositions(positions);

		if (assignColor)  fnMesh.getVertexColors(vertexColors);

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			double vertexVal = v.getColor().r;
			zColor newCol(v.getColor().r, 0, 0, 1);

			double extrudeVal = coreUtils.ofMap((float) vertexVal, 0.0f, 1.0f, minVal, maxVal);

			zVector vNormal = v.getNormal();
			vNormal.normalize();

			positions.push_back(v.getPosition() + vNormal * extrudeVal);

			if (assignColor) vertexColors.push_back(v.getColor());

		}

		// bottom and top face connectivity
		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> fVerts;
			f.getVertices(fVerts);

			// top face
			for (int j = 0; j < fVerts.size(); j++)
			{
				polyConnects.push_back(fVerts[j] + numVerts);
			}
			polyCounts.push_back(fVerts.size());


			// bottom face
			for (int j = fVerts.size() - 1; j >= 0; j--)
			{
				polyConnects.push_back(fVerts[j]);
			}
			polyCounts.push_back(fVerts.size());

		}

		// boundary thickness
		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{
			if (he.onBoundary())
			{
				vector<int> eVerts;
				he.getVertices(eVerts);

				polyConnects.push_back(eVerts[0]);
				polyConnects.push_back(eVerts[1]);
				polyConnects.push_back(eVerts[1] + numVerts);
				polyConnects.push_back(eVerts[0] + numVerts);

				polyCounts.push_back(4);

			}
		}


		zObjMesh out;
		zFnMesh tempFn(out);

		if (positions.size() > 0)
		{
			tempFn.create(positions, polyCounts, polyConnects);

			if (assignColor) tempFn.setVertexColors(vertexColors, true);
		}

		return out;
	}

}