#pragma once

#include <headers/geometry/zMesh.h>


namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief  The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryUtilities
	*	\brief Collection of utility methods for graphs, meshes and scalarfields.
	*  @{
	*/

	/** \addtogroup zMeshUtilities
	*	\brief Collection of utility methods for meshes.
	*  @{
	*/

	//--------------------------
	//--- SET METHODS 
	//--------------------------

	void setVertexColor(zMesh &inMesh, zColor col, bool setFaceColor = false)
	{

		for (int i = 0; i < inMesh.vertexColors.size(); i++)
		{
			inMesh.vertexColors[i] = col;
		}

		if (setFaceColor) inMesh.computeFaceColorfromVertexColor();
	}


	/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the mesh.
	*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/

	void setVertexColors(zMesh &inMesh, vector<zColor>& col, bool setFaceColor = false)
	{
		if (col.size() != inMesh.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

		for (int i = 0; i < inMesh.vertexColors.size(); i++)
		{
			inMesh.vertexColors[i] = col[i];
		}

		if (setFaceColor) inMesh.computeFaceColorfromVertexColor();
	}

	/*! \brief This method sets face color of all the faces to the input color.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the face color if true.
	*	\since version 0.0.1
	*/

	void setFaceColor(zMesh &inMesh, zColor col, bool setVertexColor = false)
	{

		for (int i = 0; i < inMesh.faceColors.size(); i++)
		{
			inMesh.faceColors[i] = col;
		}

		if (setVertexColor) inMesh.computeVertexColorfromFaceColor();
	}

	/*! \brief This method sets face color of all the faces to the input color contatiner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	col				- input color contatiner. The size of the contatiner should be equal to number of faces in the mesh.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the face color if true.
	*	\since version 0.0.1
	*/

	void setFaceColors(zMesh &inMesh, vector<zColor>& col, bool setVertexColor = false)
	{
		if (col.size() != inMesh.faceColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

		for (int i = 0; i < inMesh.faceColors.size(); i++)
		{
			inMesh.faceColors[i] = col[i];
		}

		if (setVertexColor) inMesh.computeVertexColorfromFaceColor();
	}


	/*! \brief This method sets face normals of all the faces to the input normal.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	fNormal			- input normal.
	*	\since version 0.0.1
	*/

	void setFaceNormals(zMesh &inMesh, zVector &fNormal)
	{	

		inMesh.faceNormals.clear();

		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			inMesh.faceNormals.push_back(fNormal);
		}

		// compute normals per face based on vertex normals and store it in faceNormals
		inMesh.computeVertexNormalfromFaceNormal();
	}

	/*! \brief This method sets face normals of all the faces to the input normals contatiner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	fNormals		- input normals contatiner. The size of the contatiner should be equal to number of faces in the mesh.
	*	\since version 0.0.1
	*/

	void setFaceNormals(zMesh &inMesh, vector<zVector> &fNormals)
	{

		if (inMesh.faceActive.size() != fNormals.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

		inMesh.faceNormals.clear();

		for (int i = 0; i < fNormals.size(); i++)
		{
			inMesh.faceNormals.push_back(fNormals[i]);
		}

		// compute normals per face based on vertex normals and store it in faceNormals
		 inMesh.computeVertexNormalfromFaceNormal();	
	}


	//--------------------------
	//--- GET METHODS 
	//--------------------------

	/*! \brief This method computes the local curvature of the mesh vertices.
	*
	*	\param		[in]	inMesh				- input mesh.
	*	\param		[out]	vertexCurvature		- container of vertex curvature.
	*	\since version 0.0.1
	*/
	
	void getPrincipalCurvature(zMesh &inMesh, vector<zCurvature> &vertexCurvatures)
	{
		for (int j = 0; j < inMesh.numVertices(); j++)
		{

			if (inMesh.vertexActive[j])
			{
				double angleSum = 0;
				double cotangentSum = 0;
				double areaSum = 0;
				double areaSumMixed = 0;
				double edgeLengthSquare = 0;
				float gaussianCurv = 0;
				float gaussianAngle = 0;

				zCurvature curv;
				curv.k1 = 0;
				curv.k2 = 0;

				zVector meanCurvNormal;

				if (!inMesh.onBoundary(j, zVertexData))
				{
					vector<int> connectedvertices;
					inMesh.getConnectedVertices(j, zVertexData, connectedvertices);

					zVector pt = inMesh.vertexPositions[j];

					float multFactor = 0.125;
					for (int i = 0; i < connectedvertices.size(); i++)
					{
						int next = (i + 1) % connectedvertices.size();
						int prev = (i + connectedvertices.size() - 1) % connectedvertices.size();

						zVector pt1 = inMesh.vertexPositions[connectedvertices[i]];
						zVector pt2 = inMesh.vertexPositions[connectedvertices[next]];
						zVector pt3 = inMesh.vertexPositions[connectedvertices[prev]];

						zVector cr = (pt1 - pt) ^ (pt2 - pt);

						float ang = (pt1 - pt).angle(pt2 - pt);
						angleSum += ang;
						cotangentSum += (((pt2 - pt)*(pt1 - pt)) / cr.length());


						float e_Length = (pt1 - pt2).length();

						edgeLengthSquare += (e_Length * e_Length);

						zVector cr_alpha = (pt - pt1) ^ (pt2 - pt1);
						zVector cr_beta = (pt - pt1) ^ (pt3 - pt1);

						float coTan_alpha = (((pt - pt1)*(pt2 - pt1)) / cr_alpha.length());
						float coTan_beta = (((pt - pt1)*(pt3 - pt1)) / cr_beta.length());

						// check if triangle is obtuse
						if ((pt1 - pt).angle(pt2 - pt) <= 90 && (pt - pt1).angle(pt2 - pt1) <= 90 && (pt1 - pt2).angle(pt - pt2) <= 90)
						{
							areaSumMixed += (coTan_alpha + coTan_beta) * edgeLengthSquare * 0.125;
						}
						else
						{

							double triArea = (((pt1 - pt) ^ (pt2 - pt)).length()) / 2;

							if ((ang) <= 90) areaSumMixed += triArea * 0.25;
							else areaSumMixed += triArea * 0.5;

						}

						meanCurvNormal += ((pt - pt1)*(coTan_alpha + coTan_beta));
					}

					meanCurvNormal /= (2 * areaSumMixed);

					gaussianCurv = (360 - angleSum) / ((0.5 * areaSum) - (multFactor * cotangentSum * edgeLengthSquare));
					//outGauss.push_back(gaussianCurv);

					////// Based on Discrete Differential-Geometry Operators for Triangulated 2-Manifolds

					//gaussianCurv = (360 - angleSum) / areaSumMixed;

					double meanCurv = (meanCurvNormal.length() / 2);
					//if (meanCurv <0.001) meanCurv = 0;

					double deltaX = (meanCurv*meanCurv) - gaussianCurv;
					if (deltaX < 0) deltaX = 0;


					curv.k1 = meanCurv + sqrt(deltaX);
					curv.k2 = meanCurv - sqrt(deltaX);

				}

				vertexCurvatures.push_back(curv);
			}

			else
			{
				zCurvature curv;

				curv.k1 = -1;
				curv.k2 = -1;

				vertexCurvatures.push_back(curv);
			}
		}
	}

	/*! \brief This method computes the centers of a zEdge or zFace.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	type			- zEdgeData or zFaceData.
	*	\param		[out]	centers			- vector of centers of type zVector.
	*	\since version 0.0.1
	*/
	
	void getCenters(zMesh &inMesh, zHEData type, vector<zVector> &centers)
	{
		// Mesh Edge 
		if (type == zEdgeData)
		{
			vector<zVector> edgeCenters;

			edgeCenters.clear();

			for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
			{
				if (inMesh.edgeActive[i])
				{
					vector<int> eVerts;
					inMesh.getVertices(i, zEdgeData, eVerts);

					zVector cen = (inMesh.vertexPositions[eVerts[0]] + inMesh.vertexPositions[eVerts[1]]) * 0.5;

					edgeCenters.push_back(cen);
					edgeCenters.push_back(cen);
				}
				else
				{
					edgeCenters.push_back(zVector());
					edgeCenters.push_back(zVector());
				}
			}

			centers = edgeCenters;
		}

		// Mesh Face 
		else if (type == zFaceData)
		{
			vector<zVector> faceCenters;
			faceCenters.clear();

			for (int i = 0; i < inMesh.faceActive.size(); i++)
			{
				if (inMesh.faceActive[i])
				{
					vector<int> fVerts;
					inMesh.getVertices(i, zFaceData, fVerts);
					zVector cen;

					for (int j = 0; j < fVerts.size(); j++) cen += inMesh.vertexPositions[fVerts[j]];

					cen /= fVerts.size();
					faceCenters.push_back(cen);
				}
				else
				{
					faceCenters.push_back(zVector());
				}
			}

			centers = faceCenters;
		}
		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	/*! \brief This method computes the lengths of the edges of a zMesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[out]	edgeLengths			- vector of edge lengths.
	*	\since version 0.0.1
	*/
	void getEdgeLengths(zMesh &inMesh, vector<double> &edgeLengths)
	{
		vector<double> out;

		for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
		{
			if (inMesh.edgeActive[i])
			{
				int v1 = inMesh.edges[i].getVertex()->getVertexId();
				int v2 = inMesh.edges[i].getSym()->getVertex()->getVertexId();

				zVector e = inMesh.vertexPositions[v1] - inMesh.vertexPositions[v2];
				double e_len = e.length();

				out.push_back(e_len);
				out.push_back(e_len);
			}
			else
			{
				out.push_back(0);
				out.push_back(0);

			}


		}

		edgeLengths = out;
	}


	/*! \brief This method computes the dihedral angle per edge of zMesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[out]	dihedralAngles		- vector of edge dihedralAngles.
	*	\since version 0.0.1
	*/
	
	void getEdgeDihedralAngles(zMesh &inMesh, vector<double> &dihedralAngles)
	{
		vector<double> out;

		if(inMesh.faceNormals.size() != inMesh.faceActive.size()) throw std::invalid_argument(" error: invalid zHEData type");

		for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
		{
			if (inMesh.edgeActive[i])
			{
				if (!inMesh.onBoundary(i, zEdgeData) && !inMesh.onBoundary(i + 1, zEdgeData))
				{
					// get connected face to edge
					vector<int> cFaces;
					inMesh.getFaces(i, zEdgeData, cFaces);

					zVector n0 = inMesh.faceNormals[cFaces[0]];
					zVector n1 = inMesh.faceNormals[cFaces[1]];

					zVector e = inMesh.vertexPositions[inMesh.edges[i].getVertex()->getVertexId()] - inMesh.vertexPositions[inMesh.edges[i + 1].getVertex()->getVertexId()];

					double di_ang;
					di_ang = e.dihedralAngle(n0, n1);

					// per half edge
					out.push_back(di_ang);
					out.push_back(di_ang);
					
				}
				else
				{
					// per half edge
					out.push_back(90);
					out.push_back(90);
				}
			}
			else
			{
				// per half edge
				out.push_back(90);
				out.push_back(90);
			}


		}

		dihedralAngles = out;
	}

	
	/*! \brief This method computes the area around every vertex of a zMesh based on face centers.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	faceCenters		- vector of face centers of type zVector.
	*	\param		[in]	edgeCenters		- vector of edge centers of type zVector.
	*	\param		[out]	vertexAreas		- vector of vertex Areas.
	*	\return				double			- total area of the mesh.
	*	\since version 0.0.1
	*/
	
	double getVertexArea(zMesh &inMesh, vector<zVector> &faceCenters, vector<zVector> &edgeCenters, vector<double> &vertexAreas)
	{
		vector<double> out;

		double totalArea = 0;

		for (int i = 0; i < inMesh.vertexActive.size(); i++)
		{
			double vArea = 0;


			if (inMesh.vertexActive[i])
			{
				vector<int> cEdges;
				inMesh.getConnectedEdges(i, zVertexData, cEdges);

				for (int j = 0; j < cEdges.size(); j++)
				{

					int currentEdge = cEdges[j];
					int nextEdge = cEdges[(j + 1) % cEdges.size()];

					if (!inMesh.edges[currentEdge].getFace() || !inMesh.edges[nextEdge].getSym()->getFace()) continue;

					if (inMesh.edges[currentEdge].getFace()->getFaceId() != inMesh.edges[nextEdge].getSym()->getFace()->getFaceId()) continue;

					zVector vPos = inMesh.vertexPositions[i];
					zVector fCen = faceCenters[inMesh.edges[currentEdge].getFace()->getFaceId()];
					zVector currentEdge_cen = edgeCenters[currentEdge];
					zVector nextEdge_cen = edgeCenters[nextEdge];

					double Area1 = triangleArea(vPos, currentEdge_cen, fCen);
					vArea += (Area1);

					double Area2 = triangleArea(vPos, nextEdge_cen, fCen);
					vArea += (Area2);

				}

			}

		
			out.push_back(vArea);

			totalArea += vArea;

		}

		//printf("\n totalArea : %1.4f ",  totalArea);

		vertexAreas = out;

		return totalArea;
	}

	/*! \brief This method returns the minimum and maximum edge lengths in the mesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[out]	minVal		- minimum edge length in the mesh.
	*	\param		[out]	maxVal		- maximum edge length in the mesh.
	*	\since version 0.0.1
	*/

	void getEdgeLengthDomain(zMesh &inMesh, double &minVal, double &maxVal)
	{
		minVal = 10000;
		maxVal = 0;

		for (int i = 0; i <inMesh.edgeActive.size(); i += 2)
		{
			if (inMesh.edgeActive[i])
			{
				vector<int> eVerts;
				inMesh.getVertices(i, zEdgeData, eVerts);

				double len = inMesh.vertexPositions[eVerts[0]].distanceTo(inMesh.vertexPositions[eVerts[1]]);

				if (len < minVal) minVal = len;
				if (len > maxVal) maxVal = len;
			}
		}

		printf("\n Edgelength min : %1.2f max : %1.2f ", minVal, maxVal);
	}

	/*! \brief This method sets vertex color of all the vertices to the input color.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/


	

	/** @}*/

	/** @}*/

	/** @}*/
}