#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zGraphMeshUtilities.h>
#include <headers/geometry/zMeshModifiers.h>

namespace zSpace
{
	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zVariableExtrude
	*	\brief Collection of methods for creating variable extrude on mesh based on face/ vertex color.
	*  @{
	*/

	/*! \brief This method offset extrudes the faces of the input mesh based on vertex / face color. It uses only the red channel of the color.
	*
	*	\param		[in]	inMesh						- input mesh.
	*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
	*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
	*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
	*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
	*	\since version 0.0.1
	*/
	inline zMesh variableFaceOffsetExtrude(zMesh & inMesh,bool keepExistingFaces= true, bool assignColor = true, double minVal = 0.01, double maxVal = 0.99 , bool useVertexColor =  false)
	{

		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> faceColors;

		vector<zVector> fCenters;
		getCenters(inMesh, zFaceData, fCenters);

		// append inmesh positions
		if (keepExistingFaces)
		{ 		
			positions = inMesh.vertexPositions;		
		}

		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			if (!inMesh.faceActive[i]) continue;

			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			vector<zVector> fVertPositions;
			inMesh.getVertexPositions(i, zFaceData, fVertPositions);

			double faceVal = inMesh.faceColors[i].r;	
			zColor newCol(inMesh.faceColors[i].r, 0, 0, 1);

			double extrudeVal = ofMap(faceVal, 0.0, 1.0, minVal, maxVal);

			// get current size of positions
			int numVerts = positions.size();

			// append new positions
			for (int j = 0; j < fVertPositions.size(); j++)
			{
				zVector dir = fCenters[i] - fVertPositions[j];
				double len = dir.length(); 
				dir.normalize(); 

				if (useVertexColor)
				{
					extrudeVal = ofMap(inMesh.vertexColors[fVerts[j]].r, 0.0, 1.0, minVal, maxVal);
				}

				zVector newPos = fVertPositions[j] + dir * len * extrudeVal;				

				positions.push_back(newPos);
			}

			// compute polyconnects and polycounts
			if (keepExistingFaces)
			{
				for (int j = 0; j < fVerts.size(); j++)
				{
					int currentId = j; 
					int nextId = (j + 1) % fVerts.size(); 

					polyConnects.push_back(fVerts[currentId]);
					polyConnects.push_back(fVerts[nextId]);
					polyConnects.push_back(numVerts + nextId);
					polyConnects.push_back(numVerts + currentId);
					

					polyCounts.push_back(4);

					if(assignColor) faceColors.push_back(newCol);

				}

			}
			else
			{

				for (int j = 0; j < fVertPositions.size(); j++)
				{
					int currentId = j;
					polyConnects.push_back(numVerts + currentId);
				}

				if (assignColor) faceColors.push_back(newCol);

				polyCounts.push_back(fVertPositions.size());

			}

		}

		if (positions.size() > 0)
		{
			out = zMesh(positions, polyCounts, polyConnects);

			if (assignColor) setFaceColors(out, faceColors, true);
		}
		
		return out;
	}

	/*! \brief This method offset extrudes the faces of the input mesh based on vertex / face color. It uses only the red channel of the color.
	*
	*	\param		[in]	inMesh						- input mesh.
	*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
	*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
	*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
	*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
	*	\since version 0.0.1
	*/
	inline zMesh variableFaceOffset(zMesh & inMesh, bool keepExistingFaces = true, bool assignColor = true, double minVal = 0.01, double maxVal = 0.99)
	{

		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> vertexColors;


		vector<zVector> fCenters;
		getCenters(inMesh, zFaceData, fCenters);

		if (keepExistingFaces)
		{
			vector<vector<int>> inVertex_newVertex;

			for (int i = 0; i < inMesh.vertexPositions.size(); i++)
			{
				vector<int> temp;			

				if (inMesh.onBoundary(i, zVertexData))
				{

					temp.push_back(positions.size());
					positions.push_back(inMesh.vertexPositions[i]);
					if (assignColor) vertexColors.push_back(inMesh.vertexColors[i]);

					double vertexVal = inMesh.vertexColors[i].r;
					zColor newCol(inMesh.vertexColors[i].r, 0, 0, 1);

					double extrudeVal = ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);

					
					zEdge *vEdge = inMesh.vertices[i].getEdge();

					int next = vEdge->getVertex()->getVertexId();;
					int prev = vEdge->getPrev()->getSym()->getVertex()->getVertexId();


					zVector Ori = inMesh.vertexPositions[i];;
					zVector v1 = inMesh.vertexPositions[prev] - Ori;
					v1.normalize();

					zVector v2 = inMesh.vertexPositions[next] - Ori;
					v2.normalize();

					zVector v3 = v1;

					v1 = v1 ^ v2;
					v3 = v3 + v2;

					double cs = v3 * v2;

					zVector a1 = v2 * cs;
					zVector a2 = v3 - a1;

					double alpha = sqrt(a2.length() * a2.length());
					if (cs < 0) alpha *= -1;

					double length = extrudeVal / alpha;
				
					zVector offPos = Ori + (v3 * length);

					temp.push_back(positions.size());
					positions.push_back(offPos);
					if (assignColor) vertexColors.push_back(inMesh.vertexColors[i]);

				}	

				inVertex_newVertex.push_back(temp);

			}

			// poly connects 

			for (int i = 0; i < inMesh.edgeActive.size(); i++)
			{
				if (inMesh.onBoundary(i, zEdgeData))
				{
					vector<int> eVerts;
					inMesh.getVertices(i, zEdgeData, eVerts);

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
			

			for (int i = 0; i < inMesh.vertexPositions.size(); i++)
			{
				vector<int> temp;

				if (inMesh.onBoundary(i, zVertexData))
				{

					double vertexVal = inMesh.vertexColors[i].r;
					zColor newCol(inMesh.vertexColors[i].r, 0, 0, 1);

					double extrudeVal = ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);


					zEdge *vEdge = inMesh.vertices[i].getEdge();

					int next = vEdge->getVertex()->getVertexId();;
					int prev = vEdge->getPrev()->getSym()->getVertex()->getVertexId();


					zVector Ori = inMesh.vertexPositions[i];;
					zVector v1 = inMesh.vertexPositions[prev] - Ori;
					v1.normalize();

					zVector v2 = inMesh.vertexPositions[next] - Ori;
					v2.normalize();

					zVector v3 = v1;

					v1 = v1 ^ v2;
					v3 = v3 + v2;

					double cs = v3 * v2;

					zVector a1 = v2 * cs;
					zVector a2 = v3 - a1;

					double alpha = sqrt(a2.length() * a2.length());
					if (cs < 0) alpha *= -1;

					double length = extrudeVal / alpha;

					zVector offPos = Ori + (v3 * length);

					
					positions.push_back(offPos);
					if (assignColor) vertexColors.push_back(inMesh.vertexColors[i]);

				}

				else
				{
					positions.push_back(inMesh.vertexPositions[i]);
					if (assignColor) vertexColors.push_back(inMesh.vertexColors[i]);
				}
			

			}

			// poly connects 

			for (int i = 0; i < inMesh.faceActive.size(); i++)
			{
				
					vector<int> fVerts;
					inMesh.getVertices(i, zFaceData, fVerts);

					for (int j = 0; j < fVerts.size(); j++)
					{
						polyConnects.push_back(fVerts[j]);
					}				
				

					polyCounts.push_back(fVerts.size());

				
			}

		}

		

		if (positions.size() > 0)
		{
			out = zMesh(positions, polyCounts, polyConnects);

			if (assignColor) setVertexColors(out, vertexColors, true);
		}

		return out;
	}

	/*! \brief This method extrudes the input mesh based on vertex / face color. It uses only the red channel of the color.
	*
	*	\param		[in]	inMesh						- input mesh.
	*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
	*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
	*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
	*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
	*	\since version 0.0.1
	*/
	inline zMesh variableFaceThicknessExtrude(zMesh & inMesh, bool assignColor, double minVal, double maxVal)
	{

		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zColor> vertexColors;
				

		int numVerts = inMesh.vertexPositions.size();

		// append inmesh positions
		positions = inMesh.vertexPositions;
		
		if(assignColor) vertexColors = inMesh.vertexColors;

		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			double vertexVal = inMesh.vertexColors[i].r;
			zColor newCol(inMesh.vertexColors[i].r, 0, 0, 1);

			double extrudeVal = ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);

			zVector vNormal = inMesh.vertexNormals[i];
			vNormal.normalize();

			positions.push_back(inMesh.vertexPositions[i] + vNormal * extrudeVal);

			if (assignColor) vertexColors.push_back( inMesh.vertexColors[i]);

		}
	
		// bottom and top face connectivity
		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			if (!inMesh.faceActive[i]) continue;

			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			// top face
			for (int j = 0; j < fVerts.size(); j++)
			{
				polyConnects.push_back(fVerts[j] + numVerts);
			}
			polyCounts.push_back(fVerts.size());


			// bottom face
			for (int j = fVerts.size() -1; j >= 0; j--)
			{
				polyConnects.push_back(fVerts[j]);
			}
			polyCounts.push_back(fVerts.size());

		}

		// boundary thickness
		for (int i = 0; i < inMesh.edgeActive.size(); i++)
		{
			if (inMesh.onBoundary(i, zEdgeData))
			{
				vector<int> eVerts;
				inMesh.getVertices(i, zEdgeData, eVerts);

				polyConnects.push_back(eVerts[0]);
				polyConnects.push_back(eVerts[1]);
				polyConnects.push_back(eVerts[1] + numVerts);
				polyConnects.push_back(eVerts[0] + numVerts);

				polyCounts.push_back(4);

			}
		}


		if (positions.size() > 0)
		{
			out = zMesh(positions, polyCounts, polyConnects);

			if (assignColor) setVertexColors(out, vertexColors, true);
		}

		return out;
	}
	

	/** @}*/

	/** @}*/
}