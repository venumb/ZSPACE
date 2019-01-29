#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zGraphMeshUtilities.h>
#include <headers/geometry/zMeshModifiers.h>

#include <headers/dynamics/zParticle.h>

#include <headers/IO/zExchange.h>

namespace zSpace
{	
	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zPolytopal
	*	\brief Collection of methods for polytopal mesh creation.
	*  @{
	*/


	/*! \brief This method creates the center line graph based on the input volume meshes.
	*
	*	\param		[in]	inputVolumeMeshes			- input volume mesh container.
	*	\param		[in]	offset						- input offset value container.
	*	\param		[out]	volumeFace_Vertex			- input volume face vertex container. vertices to edgeId map. Used to check if edge exists with the haskey being the vertex sequence.
	*	\param		[out]	graphPosition_volumeMesh	- input graph position volume mesh container.
	*	\param		[out]	graphPosition_volumeFace	- input graph position volume face container.
	*	\param		[out]	graphVertex_Offset			- input graph vertex offset container.
	*	\return				zGraph						- centerline graph.
	*	\since version 0.0.1
	*/
	zGraph createCentreLineGraph(vector<zMesh> &inputVolumeMeshes, vector<double> &offset, unordered_map <string, int> &volumeFace_Vertex, vector<int> &graphPosition_volumeMesh, vector<int> &graphPosition_volumeFace, vector<double> &graphVertex_Offset)
	{
		zGraph out;

		vector<zVector>positions;
		vector<int>edgeConnects;
		unordered_map <string, int> positionVertex;

		
		volumeFace_Vertex.clear();

		graphPosition_volumeMesh.clear();
	
		graphPosition_volumeFace.clear();

		graphVertex_Offset.clear();

		for (int j = 0; j < inputVolumeMeshes.size(); j++)
		{
			int n_v = inputVolumeMeshes[j].numVertices();
			int n_e = inputVolumeMeshes[j].numEdges();
			int n_f = inputVolumeMeshes[j].numPolygons();

			zVector volCenter;

			for (int i = 0; i < n_v; i++)
			{
				volCenter += inputVolumeMeshes[j].vertexPositions[i];
			}

			volCenter /= n_v;

			double factor = pow(10, 2);
			double x1 = round(volCenter.x *factor) / factor;
			double y1 = round(volCenter.y *factor) / factor;
			double z1 = round(volCenter.z *factor) / factor;

			string hashKey = (to_string(x1) + "," + to_string(y1) + "," + to_string(z1));

			int vId_volCenter = -1;
			bool chkExists = existsInMap(hashKey, positionVertex, vId_volCenter);

			if (!chkExists)
			{
				positionVertex[hashKey] = positions.size();
				vId_volCenter = positions.size();
				positions.push_back(volCenter);

				graphVertex_Offset.push_back(offset[j]);

				graphPosition_volumeFace.push_back(-1);
				graphPosition_volumeFace.push_back(-1);

				graphPosition_volumeMesh.push_back(j);
				graphPosition_volumeMesh.push_back(-1);
			}

			string hashKey_volface = (to_string(j) + "," + to_string(-1));
			volumeFace_Vertex[hashKey_volface] = vId_volCenter;

			vector<zVector> fCenters;
			getCenters(inputVolumeMeshes[j], zFaceData, fCenters);

			for (int i = 0; i < fCenters.size(); i++)
			{
				double factor = pow(10, 2);
				double x = round(fCenters[i].x *factor) / factor;
				double y = round(fCenters[i].y *factor) / factor;
				double z = round(fCenters[i].z *factor) / factor;

				
				string hashKey_f = (to_string(x) + "," + to_string(y) + "," + to_string(z));

				int vId_fCenter = -1;
				bool chkExists_f = existsInMap(hashKey_f, positionVertex, vId_fCenter);

				if (!chkExists_f)
				{
					positionVertex[hashKey_f] = positions.size();
					vId_fCenter = positions.size();
					positions.push_back(fCenters[i]);

					graphVertex_Offset.push_back(offset[j]);

					graphPosition_volumeFace.push_back(i);
					graphPosition_volumeFace.push_back(-1);

					graphPosition_volumeMesh.push_back(j);
					graphPosition_volumeMesh.push_back(-2);
				}
				else
				{
					graphPosition_volumeFace[(vId_fCenter * 2) + 1] = i;
					graphPosition_volumeMesh[(vId_fCenter * 2) + 1] = j;
				}
				string hashKey_volface = (to_string(j) + "," + to_string(i));
				volumeFace_Vertex[hashKey_volface] = vId_fCenter;

				edgeConnects.push_back(vId_volCenter);
				edgeConnects.push_back(vId_fCenter);
			}
		}

		out = zGraph(positions, edgeConnects);;

		for (int i = 0; i < out.numEdges(); i += 2)
		{
			zColor col(1, 1, 0, 1);
			

			out.edgeColors[i] = (col);
			out.edgeColors[i + 1] = (col);
		}

		// compute intersection point
		for (int i = 0; i < out.numVertices(); i++)
		{
			if (graphPosition_volumeFace[i * 2] >= -1 && graphPosition_volumeFace[(i * 2) + 1] != -1)
			{
				int volMesh_1 = graphPosition_volumeMesh[i * 2];
				int f1 = graphPosition_volumeFace[i * 2];

				zVector normF1 = inputVolumeMeshes[volMesh_1].faceNormals[f1];
				zVector currentPos = out.vertexPositions[i];

				vector<int> cVerts;
				out.getConnectedVertices(i, zVertexData, cVerts);

				if (cVerts.size() == 2)
				{
					zVector p1 = out.vertexPositions[cVerts[0]];
					zVector p2 = out.vertexPositions[cVerts[1]];

					graphVertex_Offset[i] = (graphVertex_Offset[cVerts[0]] + graphVertex_Offset[cVerts[1]])*0.5;

					zVector interPt;
					bool chkIntersection = line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						out.vertexPositions[i] = (interPt);

						double distTOp1 = interPt.distanceTo(p1);
						double distTOp2 = interPt.distanceTo(p2);
						double distp12 = p1.distanceTo(p2);

						double wt1 = distTOp1 / distp12;
						double wt2 = distTOp2 / distp12;

						graphVertex_Offset[i] = (graphVertex_Offset[cVerts[0]] * wt1) + (graphVertex_Offset[cVerts[1]] * wt2);

					}
				}
			}
		}

		return out;
	}


	/*! \brief This method creates the polytopal mesh based on the input volume mesh and its center line graph.
	*
	*	\param		[in]	inputVolumeMeshes		- input volume mesh.
	*	\param		[in]	volMeshId				- input volume mesh id.
	*	\param		[in]	centerLinegraph			- input center line graph.
	*	\param		[in]	volumeFace_Vertex		- input volume face vertex.
	*	\param		[in]	graphVertex_Offset		- input graph vertex offset container.
	*	\return				zMesh					- polytopal mesh.
	*	\since version 0.0.1
	*/
	zMesh createPolytopalMesh(zMesh &inputVolumeMesh, int &volMeshId, zGraph &centerLinegraph, unordered_map <string, int> &volumeFace_Vertex, vector<double> &graphVertex_Offset)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		int n_v = inputVolumeMesh.numVertices();
		int n_e = inputVolumeMesh.numEdges();
		int n_f = inputVolumeMesh.numPolygons();

		/*zVector minBB, maxBB;
		inputVolumeMesh.computeBoundingBox(minBB, maxBB);
		zVector volCenter = (minBB + maxBB) / 2;*/

		zVector volCenter;

		for (int i = 0; i < n_v; i++)
		{
			volCenter += inputVolumeMesh.vertexPositions[i];
		}

		volCenter /= n_v;

		vector<zVector> fCenters;
		getCenters(inputVolumeMesh, zFaceData, fCenters);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eFaces;
			inputVolumeMesh.getFaces(i, zEdgeData, eFaces);
			vector<int> eVertices;
			inputVolumeMesh.getVertices(i, zEdgeData, eVertices);

			zVector pos0 = inputVolumeMesh.vertexPositions[eVertices[1]];
			zVector pos1 = inputVolumeMesh.vertexPositions[eVertices[0]];

			if (eFaces.size() == 2)
			{
				int numVerts = positions.size();

				for (int j = 0; j < eFaces.size(); j++)
				{


					string hashKey_f = (to_string(volMeshId) + "," + to_string(eFaces[j]));
					int vId_fCenter = -1;
					bool chkExists_f = existsInMap(hashKey_f, volumeFace_Vertex, vId_fCenter);
					double boundaryOffset = graphVertex_Offset[vId_fCenter];


					zVector fCenter = centerLinegraph.vertexPositions[vId_fCenter];/*fCenters.getValue(eFaces[j])*/;
					zVector dir_fCenter_0 = pos0 - fCenter;
					double len0 = dir_fCenter_0.length();
					dir_fCenter_0.normalize();

					positions.push_back(fCenter + (dir_fCenter_0 * len0 *boundaryOffset));


					zVector dir_fCenter_1 = pos1 - fCenter;
					double len1 = dir_fCenter_1.length();
					dir_fCenter_1.normalize();

					positions.push_back(fCenter + (dir_fCenter_1 * len1 *boundaryOffset));
				}

				zVector dir_volCenter_0 = pos0 - volCenter;
				double len0 = dir_volCenter_0.length();
				dir_volCenter_0.normalize();

				string hashKey_v = (to_string(volMeshId) + "," + to_string(-1));
				int vId_vCenter = -1;
				bool chkExists_f = existsInMap(hashKey_v, volumeFace_Vertex, vId_vCenter);
				double boundaryOffset = graphVertex_Offset[vId_vCenter];

				double centerOffset = graphVertex_Offset[vId_vCenter];

				positions.push_back(volCenter + (dir_volCenter_0 * len0 *centerOffset));


				zVector dir_volCenter_1 = pos1 - volCenter;
				double len1 = dir_volCenter_1.length();
				dir_volCenter_1.normalize();

				positions.push_back(volCenter + (dir_volCenter_1 * len1 *centerOffset));

				polyConnects.push_back(numVerts);
				polyConnects.push_back(numVerts + 4);
				polyConnects.push_back(numVerts + 5);
				polyConnects.push_back(numVerts + 1);
				polyCounts.push_back(4);

				polyConnects.push_back(numVerts + 5);
				polyConnects.push_back(numVerts + 4);
				polyConnects.push_back(numVerts + 2);
				polyConnects.push_back(numVerts + 3);
				polyCounts.push_back(4);

			}
		}

		out = zMesh(positions, polyCounts, polyConnects);;
		return out;
	}


	/*! \brief This method remeshes the smoothed polytopal mesh to have rulings in ony one direction.
	*
	*	\param		[in]	smoothPolytopalMesh		- input smooth polytopal mesh.
	*	\param		[in]	SUBDIVS					- input number of subdivisions.
	*	\return				zMesh					- remeshed smooothed polytopal mesh.
	*	\since version 0.0.1
	*/
	zMesh remeshSmoothPolytopalMesh(zMesh &smoothPolytopalMesh, int SUBDIVS = 1)
	{
		

		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		int n_v_lowPoly = smoothPolytopalMesh.numVertices();

		smoothMesh(smoothPolytopalMesh, SUBDIVS);

		int n_v = smoothPolytopalMesh.numVertices();
		int n_e = smoothPolytopalMesh.numEdges();
		int n_f = smoothPolytopalMesh.numPolygons();

		for (int i = 0; i < n_v_lowPoly; i += 6)
		{
			int vert0 = i;
			int vert1 = i + 1;
			int edge0, edge1;

			vector<int> cEdges0;
			smoothPolytopalMesh.getConnectedEdges(vert0, zVertexData, cEdges0);

			for (int j = 0; j < cEdges0.size(); j++)
			{
				if (!smoothPolytopalMesh.onBoundary(cEdges0[j], zEdgeData))
				{
					edge0 = smoothPolytopalMesh.edges[cEdges0[j]].getSym()->getEdgeId();
				}
			}

			vector<int> cEdges1;
			smoothPolytopalMesh.getConnectedEdges(vert1, zVertexData, cEdges1);

			for (int j = 0; j < cEdges1.size(); j++)
			{
				if (smoothPolytopalMesh.onBoundary(cEdges1[j], zEdgeData))
				{
					edge1 = cEdges1[j];
				}
			}

			positions.push_back(smoothPolytopalMesh.vertexPositions[vert0]);
			positions.push_back(smoothPolytopalMesh.vertexPositions[vert1]);

			//while (smoothPolytopalMesh.edges[edge0].getVertex()->getVertexId() != i + 2)
			for (int k = 0; k < pow(2, (SUBDIVS + 1)); k++)
			{
				int numVerts = positions.size();

				int vert2 = smoothPolytopalMesh.edges[edge0].getSym()->getVertex()->getVertexId();
				int vert3 = smoothPolytopalMesh.edges[edge1].getVertex()->getVertexId();

				positions.push_back(smoothPolytopalMesh.vertexPositions[vert2]);
				positions.push_back(smoothPolytopalMesh.vertexPositions[vert3]);

				polyConnects.push_back(numVerts - 2);
				polyConnects.push_back(numVerts);
				polyConnects.push_back(numVerts + 1);
				polyConnects.push_back(numVerts - 1);
				polyCounts.push_back(4);

				int tempEdge0 = smoothPolytopalMesh.edges[edge0].getPrev()->getEdgeId();
				int tempEdge1 = smoothPolytopalMesh.edges[edge1].getNext()->getEdgeId();

				//vert0 = vert2;
				//vert1 = vert3;

				edge0 = tempEdge0;
				edge1 = tempEdge1;
			}
		}

		out = zMesh(positions, polyCounts, polyConnects);;
		return out;
	}

	/*! \brief This method computes the ruling intersetions.
	*
	*	\param		[in]	smoothPolytopalMesh		- input smooth polytopal mesh.
	*	\param		[in]	v0						- input vertex index 0.
	*	\param		[in]	v1						- input vertex index 1.
	*	\param		[out]	closestPt				- stores closest point if there is intersection bettwen the two ruling edges.
	*	\return				bool					- true if there is a intersection else false.
	*	\since version 0.0.1
	*/
	bool computeRulingIntersection(zMesh &smoothPolytopalMesh, int v0, int v1, zVector &closestPt)
	{
		bool out = false;

		int e0 = -1;
		int e1 = -1;

		vector<int> cEdges0;
		smoothPolytopalMesh.getConnectedEdges(v0, zVertexData, cEdges0);
		if (cEdges0.size() == 3)
		{
			for (int i = 0; i < cEdges0.size(); i++)
			{
				if (!smoothPolytopalMesh.onBoundary(cEdges0[i], zEdgeData))
				{
					e0 = cEdges0[i];
					break;
				}
			}
		}

		vector<int> cEdges1;
		smoothPolytopalMesh.getConnectedEdges(v1, zVertexData, cEdges1);
		if (cEdges1.size() == 3)
		{
			for (int i = 0; i < cEdges1.size(); i++)
			{
				if (!smoothPolytopalMesh.onBoundary(cEdges1[i], zEdgeData))
				{
					e1 = cEdges1[i];
					break;
				}
			}
		}

		if (e0 != -1 && e1 != -1)
		{
			int v2 = (v0 % 2 == 0) ? v0 + 1 : v0 - 1;
			int v3 = (v1 % 2 == 0) ? v1 + 1 : v1 - 1;

			zVector a0 = smoothPolytopalMesh.vertexPositions[v2];
			zVector a1 = smoothPolytopalMesh.vertexPositions[v0];

			zVector b0 = smoothPolytopalMesh.vertexPositions[v3];
			zVector b1 = smoothPolytopalMesh.vertexPositions[v1];

			double uA = -1;
			double uB = -1;
			out = line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

			if (out)
			{
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


			}

		}
		return out;
	}

	/*! \brief This method closes the smooth polytopal mesh.	
	*
	*	\param		[in]	inputVolumeMesh			- input volume mesh.
	*	\param		[in]	smoothPolytopalMesh		- input smooth polytopal mesh.
	*	\param		[in]	SUBDIVS					- input number of subdivisions.
	*	\since version 0.0.1
	*/
	void closePolytopalMesh(zMesh &inputVolumeMesh, zMesh &smoothPolytopalMesh, int SUBDIVS = 1)
	{
		int n_v = inputVolumeMesh.numVertices();
		int n_e = inputVolumeMesh.numEdges();
		int n_f = inputVolumeMesh.numPolygons();

		int n_v_smooth = smoothPolytopalMesh.numVertices();
		int n_e_smooth = smoothPolytopalMesh.numEdges();
		int n_f_smooth = smoothPolytopalMesh.numPolygons();

		int numVertsPerStrip = floor(n_v_smooth / (0.5 * n_e));
		int half_NumVertsPerStrip = floor(numVertsPerStrip / 2);


		vector<bool> vertVisited;

		for (int i = 0; i < n_v_smooth; i++)
		{
			vertVisited.push_back(false);
		}

		for (int i = 0; i < n_e; i += 2)
		{
			int eStripId = floor(i / 2);


			//-- Prev  Edge	

			int ePrev = inputVolumeMesh.edges[i].getPrev()->getEdgeId();
			int ePrevStripId = floor(ePrev / 2);


			if (ePrev % 2 == 0)
			{
				for (int j = 1, k = 0; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip - 2; j += 2, k += 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = ePrevStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = (cPt);
							smoothPolytopalMesh.vertexPositions[v1] = (cPt);
						}

						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}



				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = 0; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip - 2; j -= 2, k += 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = ePrevStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = (cPt);
							smoothPolytopalMesh.vertexPositions[v1] = cPt;

						}
						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}
				}
			}

			//-- Next Edge		

			int eNext = inputVolumeMesh.edges[i].getNext()->getEdgeId();
			int eNextStripId = floor(eNext / 2);

			if (eNext % 2 == 0)
			{
				for (int j = 0, k = 1; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip; j += 2, k += 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eNextStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}

						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}

				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = 1; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip; j -= 2, k += 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eNextStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}

						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}

				}
			}


			//-- SYM Prev  Edge	

			int eSymPrev = inputVolumeMesh.edges[i].getSym()->getPrev()->getEdgeId();
			int eSymPrevStripId = floor(eSymPrev / 2);


			if (eSymPrev % 2 == 0)
			{
				for (int j = 1, k = numVertsPerStrip - 1; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip; j += 2, k -= 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eSymPrevStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}

						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}
				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = numVertsPerStrip - 1; j > half_NumVertsPerStrip - 1, k > half_NumVertsPerStrip; j -= 2, k -= 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eSymPrevStripId * numVertsPerStrip + j;

					if (!vertVisited[v0] && !vertVisited[v1])
					{

						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}
						vertVisited[v0] = true;
						vertVisited[v1] = true;

					}
				}
			}


			//--SYM Next Edge		
			int eSymNext = inputVolumeMesh.edges[i].getSym()->getNext()->getEdgeId();
			int eSymNextStripId = floor(eSymNext / 2);

			if (eSymNext % 2 == 0)
			{
				for (int j = 0, k = numVertsPerStrip - 2; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip - 1; j += 2, k -= 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eSymNextStripId * numVertsPerStrip + j;


					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}
						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}
				}
			}
			else
			{
				for (int j = numVertsPerStrip - 1, k = numVertsPerStrip - 2; j > half_NumVertsPerStrip, k > half_NumVertsPerStrip - 1; j -= 2, k -= 2)
				{
					int v0 = eStripId * numVertsPerStrip + k;
					int v1 = eSymNextStripId * numVertsPerStrip + j;


					if (!vertVisited[v0] && !vertVisited[v1])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

						if (intersectChk)
						{
							smoothPolytopalMesh.vertexPositions[v0] = cPt;
							smoothPolytopalMesh.vertexPositions[v1] = cPt;
						}
						vertVisited[v0] = true;
						vertVisited[v1] = true;
					}

				}
			}
		}

		for (int i = 0; i < n_v; i++)
		{
			vector<int> cEdges;
			inputVolumeMesh.getConnectedEdges(i, zVertexData, cEdges);

			vector<int> smoothMeshVerts;
			vector<zVector> intersectPoints;

			// get connected edge strips
			for (int j = 0; j < cEdges.size(); j++)
			{
				int eStripId = floor(cEdges[j] / 2);
				int vertId = eStripId * numVertsPerStrip + half_NumVertsPerStrip;

				if (cEdges[j] % 2 == 0) vertId -= 1;
				smoothMeshVerts.push_back(vertId);
			}

			// comput smooth mesh vertices
			for (int j = 0; j < smoothMeshVerts.size(); j++)
			{
				int v0 = smoothMeshVerts[j];
				int v1 = smoothMeshVerts[(j + 1) % smoothMeshVerts.size()];

				vertVisited[v0] = true;

				zVector cPt;
				bool intersectChk = computeRulingIntersection(smoothPolytopalMesh, v0, v1, cPt);

				if (intersectChk)
				{
					intersectPoints.push_back(cPt);
				}
			}

			// get average intersection point
			zVector avgIntersectPoint;

			for (int j = 0; j < intersectPoints.size(); j++)
			{
				avgIntersectPoint += intersectPoints[j];
			}

			avgIntersectPoint = avgIntersectPoint / intersectPoints.size();

			//// update positions
			for (int j = 0; j < smoothMeshVerts.size(); j++)
			{
				smoothPolytopalMesh.vertexPositions[smoothMeshVerts[j]] = avgIntersectPoint;
			}

		}


		//for (int i = 0; i < vertVisited.size(); i++)
		//{
		//	if (!vertVisited[i])
		//	{
		//		printf("\n %i ", i);
		//	}
		//}
	}
	   
	/*! \brief This method explodes the input volume meshes. 
	*
	*	\param		[in]	inputVolumeMeshes			- input volume meshes container.
	*	\param		[in]	centerlineGraph				- input center line graph.
	*	\param		[in]	graphPosition_volumeMesh	- input graph position volume mesh container.
	*	\param		[in]	scaleFactor					- input scale factor.
	*	\since version 0.0.1
	*/
	void explodePolytopalMeshes(vector<zMesh> & inputVolumeMeshes, zGraph &centerlineGraph, vector<int> &graphPosition_volumeMesh, double scaleFactor = 1)
	{
		zVector g_minBB, g_maxBB;
		//centerlineGraph.computeBoundingBox(g_minBB, g_maxBB);
		getBounds(centerlineGraph.vertexPositions, g_minBB, g_maxBB);

		zVector scalecCenter = (g_minBB + g_maxBB)*0.5;

		for (int i = 0; i < centerlineGraph.numVertices(); i++)
		{
			if (graphPosition_volumeMesh[(i * 2) + 1] == -1)
			{
				zVector dir = centerlineGraph.vertexPositions[i] - scalecCenter;
				double len = dir.length();
				dir.normalize();

				zVector newCenter = scalecCenter + (dir * len * scaleFactor);

				zVector translateVec = newCenter - centerlineGraph.vertexPositions[i];

				int volMeshId = graphPosition_volumeMesh[(i * 2)];

				for (int j = 0; j < inputVolumeMeshes[volMeshId].numVertices(); j++)
				{
					zVector newPos = inputVolumeMeshes[volMeshId].vertexPositions[j];
					newPos += translateVec;

					inputVolumeMeshes[volMeshId].vertexPositions[j] = newPos;
				}

			}

		}
	}

	/*! \brief This method aligns the centerlinegraph to the compression network. 
	*
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf)
	*	\param		[in]	inputVolumeMeshes			- input volume meshes container.
	*	\param		[in]	centerlineGraph				- input center line graph.
	*	\param		[in]	graphVertex_volumeMesh		- input center line graph vertexID to volume meshID map. 
	*	\param		[in]	graphVertex_volumeFace		- input center line graph vertexID to local volume mesh faceID map. 
	*	\param		[in]	graphParticles				- input graph particle container.
	*	\param		[in]	dT							- time step.
	*	\param		[in]	type						- integration type.
	*	\return				bool						- true if the compression graph is reached. 
	*	\since version 0.0.1
	*/
	bool computeCompressionGraph(vector<zMesh> & inputVolumeMeshes, zGraph &centerlineGraph, vector<double> &original_edgeLengths,  vector<zVector> &original_positions,  vector<int> &graphVertex_volumeMesh,  vector<int> &graphVertex_volumeFace , vector<zParticle> &graphParticles, double dT, zIntergrationType type = zEuler)
	{
		if (graphParticles.size() != centerlineGraph.vertexActive.size()) fromGRAPH(graphParticles, centerlineGraph);

		bool out = true;

		vector<vector<zVector>> volmesh_fCenters;
		for (int i = 0; i < inputVolumeMeshes.size(); i++)
		{
			vector<zVector> fCenters;
			getCenters(inputVolumeMeshes[i], zFaceData, fCenters);

			for (int j = 0; j < fCenters.size(); j++)volmesh_fCenters.push_back(fCenters);
		}
			   	
		vector<double> edgeLengths;
		getEdgeLengths(centerlineGraph, edgeLengths);
		
	
		// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
		for (int i = 0; i < graphParticles.size(); i++)
		{
			if (!centerlineGraph.vertexActive[i]) continue;
			if (graphParticles[i].fixed) continue;
			
			// get position of vertex
			zVector V = centerlineGraph.vertexPositions[i];

			int volId_V = graphVertex_volumeMesh[i * 2];
			int faceId_V = graphVertex_volumeFace[i * 2];

			// get connected vertices
			vector<int> cEdges;
			centerlineGraph.getConnectedEdges(i, zVertexData, cEdges);

			zVector forceV;

			// perpenducalrity force
			for (int j = 0; j < cEdges.size(); j++)
			{
				// get edge 
				int v1_ID = centerlineGraph.edges[cEdges[j]].getVertex()->getVertexId();
				zVector V1 = centerlineGraph.vertexPositions[v1_ID];

				zVector e = V - V1;
				double len = edgeLengths[cEdges[j]];
				e.normalize();
				
				// get volume and face Id of the connected Vertex				
				int volId =  graphVertex_volumeMesh[v1_ID * 2];

				int faceId = graphVertex_volumeFace[v1_ID * 2];
				if (faceId == -1) faceId = faceId_V;
			

				// get face normal
				zVector fNorm = inputVolumeMeshes[volId].faceNormals[faceId];
				fNorm.normalize();
		

				// flipp if edge and face normal are in opposite direction
				if (e*fNorm < 0) fNorm *= -1;

				// get projected position of vertex along face normal 
				zVector projV = V1 + (fNorm * len);

				zVector f = (projV - V) * 0.5;					

				forceV += f;


			}

			printf("\n v %i forceV1 %1.4f %1.4f %1.4f  l %1.4f", i, forceV.x, forceV.y, forceV.z, forceV.length());

			// keep it in plane force
			// get volume and face Id of the connected Vertex		
		
			if (faceId_V != -1 )
			{
				// add force to keep point in the plane of the face

				zVector fNorm = inputVolumeMeshes[volId_V].faceNormals[faceId_V];
				fNorm.normalize();
				
				zVector e = V - volmesh_fCenters[volId_V][faceId_V];

				double minDist = minDist_Point_Plane(V, volmesh_fCenters[volId_V][faceId_V], fNorm);

				if (minDist > 0)
				{
					if (e*fNorm >= 0) fNorm *= -1;
					forceV += (fNorm * minDist);				
				}

				//forceV += (e * -1);

			}		
					

			printf("\n v %i forceV2 %1.4f %1.4f %1.4f  l %1.4f", i, forceV.x, forceV.y, forceV.z, forceV.length());

			if (forceV.length() > 0.001)
			{			
				
				printf("\n working!");
				out = false;
			}

			// add forces to particle
			graphParticles[i].addForce(forceV);

		}	


		// update Particles
		for (int i = 0; i < graphParticles.size(); i++)
		{
			graphParticles[i].integrateForces(dT, type);
			graphParticles[i].updateParticle(true);
		}

		return out;

	}

	/** @}*/

	/** @}*/
}

