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

	/*! \brief This method sets vertex color of all the vertices to the input color.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/
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



	//--------------------------
	//--- UTILITY METHODS 
	//--------------------------

	/*! \brief This method stores input mesh connectivity information in the input containers
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[out]	polyConnects	- stores list of polygon connection with vertex ids for each face.
	*	\param		[out]	polyCounts		- stores number of vertices per polygon.
	*	\since version 0.0.1
	*/

	void computePolyConnects_PolyCount(zMesh &inMesh, vector<int>(&polyConnects), vector<int>(&polyCounts))
	{
		polyConnects.clear();
		polyCounts.clear();

		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			if (!inMesh.faceActive[i]) continue;

			vector<int> facevertices;
			inMesh.getVertices(i, zFaceData, facevertices);

			polyCounts.push_back(facevertices.size());

			for (int j = 0; j < facevertices.size(); j++)
			{
				polyConnects.push_back(facevertices[j]);
			}
		}
	}


	/*! \brief This method combines the two disjoint meshes to one mesh.
	*
	*	\param		[in]	m1				- input mesh 1.
	*	\param		[in]	m2				- input mesh 2.
	*	\retrun				zMesh			- combined mesh.
	*	\since version 0.0.1
	*/
	
	zMesh combineDisjointMesh(zMesh &m1, zMesh &m2)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;


		for (int i = 0; i < m1.vertexPositions.size(); i++)
		{
			positions.push_back(m1.vertexPositions[i]);
		}

		for (int i = 0; i < m2.vertexPositions.size(); i++)
		{
			positions.push_back(m2.vertexPositions[i]);
		}

		computePolyConnects_PolyCount(m1,polyConnects, polyCounts);

		vector<int>temp_polyConnects;
		vector<int>temp_polyCounts;
		computePolyConnects_PolyCount(m2,temp_polyConnects, temp_polyCounts);

		for (int i = 0; i < temp_polyConnects.size(); i++)
		{
			polyConnects.push_back(temp_polyConnects[i] + m1.vertexPositions.size());
		}

		for (int i = 0; i < temp_polyCounts.size(); i++)
		{
			polyCounts.push_back(temp_polyCounts[i]);
		}

		out = zMesh(positions, polyCounts, polyConnects);;

		return out;
	}


	/*! \brief This method returns an extruded mesh from the input mesh.
	*
	*	\param		[in]	inMesh				- input mesh.
	*	\param		[in]	extrudeThickness	- extrusion thickness.
	*	\param		[in]	thicknessTris		- true if the cap needs to be triangulated.
	*	\retrun				zMesh				- extruded mesh.
	*	\since version 0.0.1
	*/

	zMesh extrudeMesh(zMesh &inMesh, float extrudeThickness, bool thicknessTris = false)
	{
		if (inMesh.faceNormals.size() == 0 || inMesh.faceNormals.size() != inMesh.faceActive.size()) inMesh.computeMeshNormals();

		zMesh out;

		vector<zVector> positions;
		vector<int> polyCounts;
		vector<int> polyConnects;

		

		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			positions.push_back(inMesh.vertexPositions[i]);
		}

		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			positions.push_back(inMesh.vertexPositions[i] + (inMesh.vertexNormals[i] * extrudeThickness));
		}

		for (int i = 0; i < inMesh.numPolygons(); i++)
		{
			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			for (int j = 0; j < fVerts.size(); j++)
			{
				polyConnects.push_back(fVerts[j]);
			}

			polyCounts.push_back(fVerts.size());

			for (int j = fVerts.size() - 1; j >= 0; j--)
			{
				polyConnects.push_back(fVerts[j] + inMesh.vertexPositions.size());
			}

			polyCounts.push_back(fVerts.size());

		}


		for (int i = 0; i < inMesh.numEdges(); i++)
		{
			if (inMesh.onBoundary(i, zEdgeData))
			{
				vector<int> eVerts;
				inMesh.getVertices(i, zEdgeData, eVerts);

				if (thicknessTris)
				{
					polyConnects.push_back(eVerts[1]);
					polyConnects.push_back(eVerts[0]);
					polyConnects.push_back(eVerts[0] + inMesh.vertexPositions.size());

					polyConnects.push_back(eVerts[0] + inMesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1] + inMesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1]);

					polyCounts.push_back(3);
					polyCounts.push_back(3);
				}
				else
				{
					polyConnects.push_back(eVerts[1]);
					polyConnects.push_back(eVerts[0]);
					polyConnects.push_back(eVerts[0] + inMesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1] + inMesh.vertexPositions.size());


					polyCounts.push_back(4);
				}


			}
		}

		out = zMesh(positions, polyCounts, polyConnects);

		return out;
	}

	/*! \brief This method returns the offset positions of a polygon of the input mesh.
	*
	*	\details	beased on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
	*	\param		[in]	inMesh				- input mesh.
	*	\param		[in]	faceIndex			- face index.
	*	\param		[in]	offset				- offset distance.
	*	\param		[out]	offsetPositions		- container with the offset positions.
	*	\since version 0.0.1
	*/
	
	void offsetMeshFace(zMesh &inMesh, int faceIndex, double offset, vector<zVector>& offsetPositions)
	{
		vector<zVector> out;

		vector<int> fVerts;
		inMesh.getVertices(faceIndex, zFaceData, fVerts);

		for (int j = 0; j < fVerts.size(); j++)
		{
			int next = (j + 1) % fVerts.size();
			int prev = (j - 1 + fVerts.size()) % fVerts.size();


			zVector Ori = inMesh.vertexPositions[fVerts[j]];;
			zVector v1 = inMesh.vertexPositions[fVerts[prev]] - inMesh.vertexPositions[fVerts[j]];
			v1.normalize();

			zVector v2 = inMesh.vertexPositions[fVerts[next]] - inMesh.vertexPositions[fVerts[j]];
			v2.normalize();

			zVector v3 = v1;

			v1 = v1 ^ v2;
			v3 = v3 + v2;

			double cs = v3 * v2;

			zVector a1 = v2 * cs;
			zVector a2 = v3 - a1;

			double alpha = sqrt(a2.length() * a2.length());
			if (cs < 0) alpha *= -1;

			double length = offset / alpha;

			zVector mPos = inMesh.vertexPositions[fVerts[j]];
			zVector offPos = mPos + (v3 * length);

			out.push_back(offPos);

		}

		offsetPositions = out;

	}

	/*! \brief This method returns the vartiable offset positions of a polygon of the input mesh.
	*	
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	faceIndex				- face index.
	*	\param		[in]	offsets					- offset distance from each edge of the mesh.
	*	\param		[in]	faceCenter				- center of polygon.
	*	\param		[in]	faceNormal				- normal of polygon.
	*	\param		[out]	intersectionPositions	- container with the intersection positions.
	*	\since version 0.0.1
	*/

	void offsetMeshFace_Variable(zMesh &m, int faceIndex, vector<double>& offsets, zVector& faceCenter, zVector& faceNormal, vector<zVector>& intersectionPositions)
	{
		vector<zVector> offsetPoints;
		vector<int> fEdges;
		m.getEdges(faceIndex, zFaceData, fEdges);

		for (int j = 0; j < fEdges.size(); j++)
		{
			zVector p2 = m.vertexPositions[m.edges[fEdges[j]].getVertex()->getVertexId()];
			zVector p1 = m.vertexPositions[m.edges[fEdges[j]].getSym()->getVertex()->getVertexId()];

			zVector norm1 = ((p1 - p2) ^ faceNormal);
			norm1.normalize();
			if ((faceCenter - p1) * norm1 < 0) norm1 *= -1;


			offsetPoints.push_back(p1 + norm1 * offsets[j]);
			offsetPoints.push_back(p2 + norm1 * offsets[j]);

		}


		for (int j = 0; j < fEdges.size(); j++)
		{
			int prevId = (j - 1 + fEdges.size()) % fEdges.size();

			zVector a0 = offsetPoints[j * 2];
			zVector a1 = offsetPoints[j * 2 + 1];

			zVector b0 = offsetPoints[prevId * 2];
			zVector b1 = offsetPoints[prevId * 2 + 1];


			double uA = -1;
			double uB = -1;
			bool intersect = line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

			if (intersect)
			{
				//printf("\n %i working!! ", j);

				zVector closestPt;

				if (uA >= uB)
				{
					zVector dir = a1 - a0;
					double len = dir.length();
					dir.normalize();

					if (uA < 0) dir *= -1;
					closestPt = a0 + dir * len * uA;
				}
				else
				{
					zVector dir = b1 - b0;
					double len = dir.length();
					dir.normalize();

					if (uB < 0) dir *= -1;

					closestPt = b0 + dir * len * uB;
				}


				intersectionPositions.push_back(closestPt);
			}

		}
	}


	/*! \brief This method transforms the input mesh by the input transform matrix.
	*
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	transform				- transfrom matrix.
	*	\since version 0.0.1
	*/

	void transformMesh(zMesh &inMesh, zMatrixd& transform)
	{
		for (int j = 0; j < inMesh.vertexPositions.size(); j++)
		{			
			zVector newPos = inMesh.vertexPositions[j] * transform;
			inMesh.vertexPositions[j] = newPos;		
		}
	}


	/** @}*/

	/** @}*/

	/** @}*/
}