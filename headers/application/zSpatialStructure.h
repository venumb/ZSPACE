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

	/** \addtogroup zSpatialStructure
	*	\brief Collection of methods for creating spatial strctures from volume meshes.
	*  @{
	*/

	//--------------------------
	//---- SPATIAL DIAGRAM 
	//--------------------------

	/*! \brief This method creates the spatial graph based on the input volume meshes.
	*
	*	\param		[in]	inputVolumeMeshes					- input force volume mesh container.
	*	\param		[in]	offsets								- input offsets value container.
	*	\param		[out]	volumeFace_spatialGraphVertex		- input force volume mesh faceID to form graph vertexID map.
	*	\param		[out]	spatialGraphVertex_volumeMesh		- input form graph vertexID to volume meshID map.
	*	\param		[out]	spatialGraphVertex_volumeFace		- input form graph vertexID to local volume mesh faceID map.
	*	\param		[out]	spatialGraphVertex_Offset			- input graph vertex offset container.
	*	\return				zGraph								- spatial graph.
	*	\since version 0.0.1
	*/
	inline zGraph createSpatialGraph(vector<zMesh> &inputVolumeMeshes, vector<double> &offsets, unordered_map <string, int> &volumeFace_spatialGraphVertex, vector<int> &spatialGraphVertex_volumeMesh, vector<int> &spatialGraphVertex_volumeFace, vector<double> &spatialGraphVertex_Offset)
	{
		zGraph out;

		vector<zVector>positions;
		vector<int>edgeConnects;
		unordered_map <string, int> positionVertex;


		volumeFace_spatialGraphVertex.clear();

		spatialGraphVertex_volumeMesh.clear();

		spatialGraphVertex_volumeFace.clear();

		spatialGraphVertex_Offset.clear();

		for (int j = 0; j < inputVolumeMeshes.size(); j++)
		{
			int n_v = inputVolumeMeshes[j].numVertices();
			int n_e = inputVolumeMeshes[j].numEdges();
			int n_f = inputVolumeMeshes[j].numPolygons();
			

			zVector volCenter;
			getCenter_PointCloud(inputVolumeMeshes[j].vertexPositions, volCenter);			

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

				spatialGraphVertex_Offset.push_back(offsets[j]);

				spatialGraphVertex_volumeFace.push_back(-1);
				spatialGraphVertex_volumeFace.push_back(-1);

				spatialGraphVertex_volumeMesh.push_back(j);
				spatialGraphVertex_volumeMesh.push_back(-1);
			}

			string hashKey_volface = (to_string(j) + "," + to_string(-1));
			volumeFace_spatialGraphVertex[hashKey_volface] = vId_volCenter;

			vector<zVector> fCenters;
			getCenters(inputVolumeMeshes[j], zFaceData, fCenters);

			for (int i = 0; i < fCenters.size(); i++)
			{
				if (inputVolumeMeshes[j].faceColors[i].r != 1.0)
				{
					int vId_fCenter = -2;
					string hashKey_volface = (to_string(j) + "," + to_string(i));
					volumeFace_spatialGraphVertex[hashKey_volface] = vId_fCenter;

				}
				else
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

						spatialGraphVertex_Offset.push_back(offsets[j]);

						spatialGraphVertex_volumeFace.push_back(i);
						spatialGraphVertex_volumeFace.push_back(-1);

						spatialGraphVertex_volumeMesh.push_back(j);
						spatialGraphVertex_volumeMesh.push_back(-2);
					}
					else
					{
						spatialGraphVertex_volumeFace[(vId_fCenter * 2) + 1] = i;
						spatialGraphVertex_volumeMesh[(vId_fCenter * 2) + 1] = j;
					}
					string hashKey_volface = (to_string(j) + "," + to_string(i));
					volumeFace_spatialGraphVertex[hashKey_volface] = vId_fCenter;

					edgeConnects.push_back(vId_volCenter);
					edgeConnects.push_back(vId_fCenter);

				}
				
				
			}
		}

		out = zGraph(positions, edgeConnects);;

		for (int i = 0; i < out.numEdges(); i += 2)
		{
			zColor col(0.75, 0.5, 0, 1);


			out.edgeColors[i] = (col);
			out.edgeColors[i + 1] = (col);

			
		}	   
		
		return out;
	}

	//--------------------------
	//---- POLYTOPAL MESH
	//--------------------------

	inline zMesh extrudeConnectionFaces(zMesh &inMesh, int &volMeshId, zGraph &spatialGraph, unordered_map <string, int> &volumeFace_spatialGraphVertex, vector<double> &spatialGraphVertex_Offset)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zVector> fCenters;
		getCenters(inMesh, zFaceData, fCenters);

		positions = inMesh.vertexPositions;


		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			double faceVal = inMesh.faceColors[i].r;

			if (!inMesh.faceActive[i] ) continue;

			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			if (faceVal != 1.0)
			{
				for (int j = 0; j < fVerts.size(); j++) polyConnects.push_back(fVerts[j]);

				polyCounts.push_back(fVerts.size());
			}
			else
			{
				vector<zVector> fVertPositions;
				inMesh.getVertexPositions(i, zFaceData, fVertPositions);

				string hashKey_f = (to_string(volMeshId) + "," + to_string(i));
				int vId_fCenter = -1;
				bool chkExists_f = existsInMap(hashKey_f, volumeFace_spatialGraphVertex, vId_fCenter);
				
				double boundaryOffset = spatialGraphVertex_Offset[vId_fCenter];

				// get current size of positions
				int numVerts = positions.size();

				// append new positions
				for (int j = 0; j < fVertPositions.size(); j++)
				{
					zVector dir = spatialGraph.vertexPositions[vId_fCenter] - fVertPositions[j];
					double len = dir.length();
					dir.normalize();				

					zVector newPos = fVertPositions[j] + dir * len * boundaryOffset;

					positions.push_back(newPos);
				}

				// compute polyconnects and polycounts
				for (int j = 0; j < fVerts.size(); j++)
				{
					int currentId = j;
					int nextId = (j + 1) % fVerts.size();

					polyConnects.push_back(fVerts[currentId]);
					polyConnects.push_back(fVerts[nextId]);
					polyConnects.push_back(numVerts + nextId);
					polyConnects.push_back(numVerts + currentId);

					polyCounts.push_back(4);
				
				}

			}

			

		}

		if (positions.size() > 0)
		{
			out = zMesh(positions, polyCounts, polyConnects);			
		}

		return out;
		
	}

	/*! \brief This method creates the polytopal mesh based on the input volume mesh and its center line graph.
	*
	*	\param		[in]	inputVolumeMesh					- input volume mesh.
	*	\param		[in]	volMeshId						- input volume mesh id.
	*	\param		[in]	spatialGraph					- input spatial graph.
	*	\param		[in]	volumeFace_spatialGraphVertex	- input volume mesh faceID to form graph vertexID map.
	*	\param		[in]	spatialGraphVertex_Offset		- input spatial graph vertex offset container.
	*	\return				zMesh							- spatial mesh.
	*	\since version 0.0.1
	*/
	inline zMesh createSpatialMesh(zMesh &inputVolumeMesh, int &volMeshId, zGraph &spatialGraph, unordered_map <string, int> &volumeFace_spatialGraphVertex, vector<double> &spatialGraphVertex_Offset)
	{
		

			
		string hashKey_volcenter = (to_string(volMeshId) + "," + to_string(-1));
		int vId_volCenter = -1;
		bool chkExists_f = existsInMap(hashKey_volcenter, volumeFace_spatialGraphVertex, vId_volCenter);


		zVector scaleCenter = spatialGraph.vertexPositions[vId_volCenter];
		double volCenterOffset = spatialGraphVertex_Offset[vId_volCenter];

		zMesh out = extrudeConnectionFaces(inputVolumeMesh, volMeshId, spatialGraph, volumeFace_spatialGraphVertex, spatialGraphVertex_Offset) ;


		// scale original points from scale center

		for (int i = 0; i < out.numVertices(); i++)
		{
			if (out.onBoundary(i, zVertexData)) continue;


			zVector dir = scaleCenter - out.vertexPositions[i];
			double len = dir.length();
			dir.normalize();

			zVector newPos = out.vertexPositions[i] + dir * len * volCenterOffset;

			out.vertexPositions[i] = newPos;

		}
		
		return out;
	}


	//--------------------------
	//---- UTILITIES
	//--------------------------

	/*! \brief This method computes the face centers of the input volume mesh container and stores it in a 2 Dimensional Container.
	*
	*	\param		[in]	inputVolumeMeshes			- input volume meshes container.
	*	\param		[out]	volmesh_fCenters			- 2 Dimensional Container of face centers of volumeMesh.
	*	\since version 0.0.1
	*/
	inline void getVolumeMesh_FaceCenters(vector<zMesh> & inputVolumeMeshes, vector<vector<zVector>> &volmesh_fCenters)
	{
		for (int i = 0; i < inputVolumeMeshes.size(); i++)
		{
			vector<zVector> fCenters;
			getCenters(inputVolumeMeshes[i], zFaceData, fCenters);

			volmesh_fCenters.push_back(fCenters);
		}
	}

	/*! \brief This method computes the form graph edge weights based on the force volume mesh face areas.
	*
	*	\param		[in]	inputVolumeMeshes			- input volume meshes container.
	*	\param		[in]	spatialGraph				- input spatial graph.
	*	\param		[in]	spatialGraphVertex_volumeMesh		- input form graph vertexID to volume meshID map.
	*	\param		[in]	spatialGraphVertex_volumeFace		- input form graph vertexID to local volume mesh faceID map.
	*	\param		[in]	minWeight					- minimum weight of the edge.
	*	\param		[in]	maxWeight					- maximum weight of the edge.
	*	\since version 0.0.1
	*/
	inline void computeFormGraph_EdgeWeights(vector<zMesh> & inputVolumeMeshes, zGraph &spatialGraph, vector<int> &spatialGraphVertex_volumeMesh, vector<int> &spatialGraphVertex_volumeFace, double minWeight = 2.0, double maxWeight = 10.0)
	{
		//compute edgeWeights
		vector<vector<double>> volMesh_fAreas;

		double minArea = 10000, maxArea = -100000;
		zColor maxCol(0.784, 0, 0.157, 1);
		zColor minCol(0.027, 0, 0.157, 1);

		for (int i = 0; i < inputVolumeMeshes.size(); i++)
		{
			vector<double> fAreas;
			getPlanarFaceAreas(inputVolumeMeshes[i], fAreas);

			double temp_MinArea = zMin(fAreas);
			minArea = (temp_MinArea < minArea) ? temp_MinArea : minArea;

			double temp_maxArea = zMax(fAreas);
			maxArea = (temp_maxArea > maxArea) ? temp_maxArea : maxArea;

			volMesh_fAreas.push_back(fAreas);
		}

		for (int i = 0; i < spatialGraph.numVertices(); i++)
		{
			if (spatialGraphVertex_volumeFace[i * 2] == -1 && spatialGraphVertex_volumeFace[(i * 2) + 1] == -1) continue;

			vector<int> cEdges;
			spatialGraph.getConnectedEdges(i, zVertexData, cEdges);

			int volID = spatialGraphVertex_volumeMesh[i * 2];
			int faceID = spatialGraphVertex_volumeFace[i * 2];

			double fArea = volMesh_fAreas[volID][faceID];

			for (int j = 0; j < cEdges.size(); j++)
			{
				double val = ofMap(fArea, minArea, maxArea, minWeight, maxWeight);

				zColor col = blendColor(fArea, minArea, maxArea, minCol, maxCol, zRGB);

				int symEdge = (cEdges[j] % 2 == 0) ? cEdges[j] + 1 : cEdges[j] - 1;

				spatialGraph.edgeWeights[cEdges[j]] = val;
				spatialGraph.edgeWeights[symEdge] = val;

				spatialGraph.edgeColors[cEdges[j]] = col;
				spatialGraph.edgeColors[symEdge] = col;



			}

		}
	}


	//--------------------------
	//---- UPDATE SPATIAL GRAPH METHODS
	//--------------------------

	/*! \brief This method updates the form diagram.
	*
	*	\param		[in]	inputVolumeMeshes				- input volume meshes container.
	*	\param		[in]	spatialGraph					- input spatial graph.
	*	\param		[in]	spatialGraphVertex_volumeMesh	- input form graph vertexID to volume meshID map.
	*	\param		[in]	spatialGraphVertex_volumeFace	- input form graph vertexID to local volume mesh faceID map.
	*	\param		[in]	graphParticles					- input graph particle container.
	*	\param		[in]	dT								- time step.
	*	\param		[in]	type							- integration type.
	*	\return				bool							- true if the compression graph is reached.
	*	\since version 0.0.1
	*/
	inline bool updateSpatialGraph(vector<zMesh> & inputVolumeMeshes, zGraph &spatialGraph, vector<vector<zVector>> &volmesh_fCenters, vector<int> &spatialGraphVertex_volumeMesh, vector<int> &spatialGraphVertex_volumeFace, vector<zParticle> &spatialGraphParticles, double dT, zIntergrationType type = zEuler)
	{
		if (spatialGraphParticles.size() != spatialGraph.vertexActive.size()) fromGRAPH(spatialGraphParticles, spatialGraph);

		bool out = true;

		vector<double> edgeLengths;
		getEdgeLengths(spatialGraph, edgeLengths);

		// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
		for (int i = 0; i < spatialGraphParticles.size(); i++)
		{
			if (!spatialGraph.vertexActive[i]) continue;
			if (spatialGraphParticles[i].fixed) continue;

			// get position of vertex
			zVector V = spatialGraph.vertexPositions[i];

			int volId_V = spatialGraphVertex_volumeMesh[i * 2];
			int faceId_V = spatialGraphVertex_volumeFace[i * 2];

			// get connected vertices
			vector<int> cEdges;
			spatialGraph.getConnectedEdges(i, zVertexData, cEdges);

			zVector forceV;

			// perpenducalrity force
			for (int j = 0; j < cEdges.size(); j++)
			{
				// get edge 
				int v1_ID = spatialGraph.edges[cEdges[j]].getVertex()->getVertexId();

				// get volume and face Id of the connected Vertex
				int volId = spatialGraphVertex_volumeMesh[v1_ID * 2];
				int faceId = spatialGraphVertex_volumeFace[v1_ID * 2];
				if (faceId == -1) faceId = faceId_V;


				if (spatialGraph.checkVertexValency(v1_ID, 2)) v1_ID = spatialGraph.edges[cEdges[j]].getNext()->getVertex()->getVertexId();
				zVector V1 = spatialGraph.vertexPositions[v1_ID];

				zVector e = V - V1;
				double len = e.length();;
				e.normalize();



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


			// keep it in plane force
			// get volume and face Id of the connected Vertex		

			if (faceId_V != -1 && spatialGraphVertex_volumeFace[i * 2 + 1] == -1)
			{
				// add force to keep point in the plane of the face

				zVector fNorm = inputVolumeMeshes[volId_V].faceNormals[faceId_V];
				fNorm.normalize();

				zVector e = V - volmesh_fCenters[volId_V][faceId_V];

				double minDist = minDist_Point_Plane(V, volmesh_fCenters[volId_V][faceId_V], fNorm);

				if (minDist > 0)
				{
					if (e * fNorm >= 0)	fNorm *= -1;


					forceV += (fNorm * minDist);
				}

				//forceV += (e * -1);

			}

			// common face vertex between volumne centers
			if (faceId_V != -1 && spatialGraphVertex_volumeFace[i * 2 + 1] != -1)
			{
				spatialGraphParticles[i].fixed = true;
				spatialGraph.vertexColors[i] = zColor(1, 0, 0, 1);

				forceV = zVector();
			}


			if (forceV.length() > 0.001)
			{

				out = false;
			}

			// add forces to particle
			spatialGraphParticles[i].addForce(forceV);

		}


		// update Particles
		for (int i = 0; i < spatialGraphParticles.size(); i++)
		{
			spatialGraphParticles[i].integrateForces(dT, type);
			spatialGraphParticles[i].updateParticle(true);
		}

		// update fixed particle positions ( ones with a common face) 
		for (int i = 0; i < spatialGraphParticles.size(); i++)
		{
			if (!spatialGraphParticles[i].fixed) continue;

			vector<int> cVerts;
			spatialGraph.getConnectedVertices(i, zVertexData, cVerts);


			int volId_V = spatialGraphVertex_volumeMesh[i * 2];
			int faceId_V = spatialGraphVertex_volumeFace[i * 2];

			zVector normF1 = inputVolumeMeshes[volId_V].faceNormals[faceId_V];
			zVector currentPos = spatialGraph.vertexPositions[i];

			if (cVerts.size() == 2)
			{
				zVector p1 = spatialGraph.vertexPositions[cVerts[0]];
				zVector p2 = spatialGraph.vertexPositions[cVerts[1]];

				zVector interPt;
				bool chkIntersection = line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

				if (chkIntersection)
				{
					spatialGraph.vertexPositions[i] = (interPt);
				}
			}

		}

		return out;

	}


	/** @}*/

	/** @}*/
}
