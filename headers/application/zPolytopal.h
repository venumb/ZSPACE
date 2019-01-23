#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zGraphMeshUtilities.h>

namespace zSpace
{
			
	/*! \brief This methods creates the center line graph based on the input volume meshes.
	*
	*	\param		[in]	inputVolumeMeshes		- input volume mesh.
	*	\param		[in]	polyCounts				- container of type integer with number of vertices per polygon.
	*	\param		[in]	polyConnects			- polygon connection list with vertex ids for each face.
	*	\param		[in]	computeNormals			- computes the face and vertex normals if true.
	*	\return				zGraph					- centerline graph.
	*	\since version 0.0.1
	*/
	zGraph createCentreLineGraph(vector<zMesh> & inputVolumeMeshes, vector<double> &volColor, unordered_map <string, int> &volumeFace_Vertex, vector<int> &graphPosition_volumeMesh, vector<int> &graphPosition_volumeFace, vector<double> &graphVertex_Offset)
	{
		zGraph out;

		vector<zVector>positions;
		vector<int>edgeConnects;
		unordered_map <string, int> positionVertex;

		//zAttributeUnorderedMap <string, int> volumeFace_Vertex;
		volumeFace_Vertex.clear();

		//vector<int> graphPosition_volumeMesh;
		graphPosition_volumeMesh.clear();
		//vector<int> graphPosition_volumeFace;
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



			string hashKey = (to_string(volCenter.x) + "," + to_string(volCenter.y) + "," + to_string(volCenter.z));

			int vId_volCenter = -1;
			bool chkExists = existsInMap(hashKey, positionVertex, vId_volCenter);

			if (!chkExists)
			{
				positionVertex[hashKey] = positions.size();
				vId_volCenter = positions.size();
				positions.push_back(volCenter);

				graphVertex_Offset.push_back(volColor[j]);

				graphPosition_volumeFace.push_back(-1);
				graphPosition_volumeFace.push_back(-1);

				graphPosition_volumeMesh.push_back(j);
				graphPosition_volumeMesh.push_back(-1);

			}

			string hashKey_volface = (to_string(j) + "," + to_string(-1));
			volumeFace_Vertex[hashKey_volface] = vId_volCenter;

			//printf("\n %i %i | %i ", j, -1, vId_volCenter);

			vector<zVector> fCenters;
			getCenters(inputVolumeMeshes[j], zFaceData, fCenters);

			for (int i = 0; i < fCenters.size(); i++)
			{
				string hashKey_f = (to_string(fCenters[i].x) + "," + to_string(fCenters[i].y) + "," + to_string(fCenters[i].z));

				//printf("\n %i %i hashkey : %1.2f, %1.2f, %1.2f", j, i,fCenters.values[i].x, fCenters.values[i].y, fCenters.values[i].z);
				//printf("\n %i %i hashkey : %s", j, i, hashKey_f.c_str());

				int vId_fCenter = -1;
				bool chkExists_f = existsInMap(hashKey_f, positionVertex, vId_fCenter);

				if (!chkExists_f)
				{
					positionVertex[hashKey_f] = positions.size();
					vId_fCenter = positions.size();
					positions.push_back(fCenters[i]);

					graphVertex_Offset.push_back(volColor[j]);

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

				//printf("\n %i %i | %i ", j, i, vId_fCenter);

				edgeConnects.push_back(vId_volCenter);
				edgeConnects.push_back(vId_fCenter);
			}

		}

		out = zGraph(positions, edgeConnects);;
		printf("\n polytopalGraph: %i %i ", out.numVertices(), out.numEdges());

		for (int i = 0; i < out.numEdges(); i += 2)
		{
			zColor col(1, 1, 0, 1);
			//out.setAttribute <zColor>(col, i, zColorAttribute, zGraphVertex);

			out.edgeColors[i] = (col);
			out.edgeColors[i + 1] = (col);
		}

		// compute intersection point
		for (int i = 0; i < out.numVertices(); i++)
		{
			printf("\n %i ,volMesh %i, %i | volFace %i, %i", i, graphPosition_volumeMesh[i * 2], graphPosition_volumeMesh[(i * 2) + 1], graphPosition_volumeFace[i * 2], graphPosition_volumeFace[(i * 2) + 1]);

			printf("\n %i ,point %1.2f, %1.2f, %1.2f", i, out.vertexPositions[i].x, out.vertexPositions[i].y, out.vertexPositions[i].z);


			if (graphPosition_volumeFace[i * 2] >= -1 && graphPosition_volumeFace[(i * 2) + 1] != -1)
			{
				int volMesh_1 = graphPosition_volumeMesh[i * 2];
				int f1 = graphPosition_volumeFace[i * 2];

				zVector normF1 = inputVolumeMeshes[volMesh_1].faceNormals[f1];
				zVector currentPos = out.vertexPositions[i];

				vector<int> cVerts;
				out.getConnectedVertices(i, zVertexData, cVerts);

				printf("\n %i cVerts : %i ", i, cVerts.size());

				if (cVerts.size() == 2)
				{
					zVector p1 = out.vertexPositions[cVerts[0]];
					zVector p2 = out.vertexPositions[cVerts[1]];

					graphVertex_Offset[i] = (graphVertex_Offset[cVerts[0]] + graphVertex_Offset[cVerts[1]])*0.5;

					zVector interPt;
					bool chkIntersection = line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						printf("\n %i old | %1.2f %1.2f %1.2f ", i, currentPos.x, currentPos.y, currentPos.z);

						out.vertexPositions[i] = (interPt);

						double distTOp1 = interPt.distanceTo(p1);
						double distTOp2 = interPt.distanceTo(p2);
						double distp12 = p1.distanceTo(p2);

						double wt1 = distTOp1 / distp12;
						double wt2 = distTOp2 / distp12;

						graphVertex_Offset[i] = (graphVertex_Offset[cVerts[0]] * wt1) + (graphVertex_Offset[cVerts[1]] * wt2);


						printf("\n %i new | %1.2f %1.2f %1.2f ", i, interPt.x, interPt.y, interPt.z);
					}
				}


			}

			printf("\n %i : %1.2f ", i, graphVertex_Offset[i]);

		}

		std::cout << std::endl;
		for (auto it = positionVertex.begin(); it != positionVertex.end(); ++it)
			std::cout << " " << it->first << ":" << it->second;
		std::cout << std::endl;




		return out;
	}


	zMesh createPolytopalMesh(zMesh &inputVolumeMesh, int &volMeshId, zGraph &centerLinegraph, unordered_map <string, int> &volumeFace_Vertex, vector<double> &graphVertex_Offset);

	zMesh remeshSmoothPolytopalMesh(zMesh &lowPolytopalMesh, zMesh &smoothPolytopalMesh, int SUBDIVS = 1);

	void closePolytopalMesh(zMesh &inputVolumeMesh, zMesh &smoothPolytopalMesh, int SUBDIVS = 1);

	bool computeRulingIntersection(zMesh &smoothPolytopalMesh, int v0, int v1, zVector &closestPt);

	
	void explodePolytopalMeshes(vector<zMesh> & inputVolumeMeshes, zGraph &centerlineGraph, vector<int> &graphPosition_volumeMesh, double scaleFactor = 1);

}