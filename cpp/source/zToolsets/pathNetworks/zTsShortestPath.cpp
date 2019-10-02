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


#include<headers/zToolsets/pathNetworks/zTsShortestPath.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	template<typename T, typename U>
	ZSPACE_INLINE zTsShortestPath<T,U>::zTsShortestPath() {}

	//---- graph specilization for zTsShortestPath constructor
	template<>
	ZSPACE_INLINE  zTsShortestPath<zObjGraph, zFnGraph>::zTsShortestPath(zObjGraph & _graph)
	{
		heObj = &_graph;
		fnHE = zFnGraph(_graph);
	}

	//---- mesh specilization for zTsShortestPath constructor
	template<>
	ZSPACE_INLINE  zTsShortestPath<zObjMesh, zFnMesh>::zTsShortestPath(zObjMesh & _mesh)
	{
		heObj = &_mesh;
		fnHE = zFnMesh(_mesh);
	}

	//---- DESTRUCTOR

	template<typename T, typename U>
	ZSPACE_INLINE zTsShortestPath<T, U>::~zTsShortestPath() {}

	//--- WALK METHODS 

	//---- graph specilization for shortestDistance
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::shortestDistance(int index, vector<float> &dist, vector<int> &parent)
	{
		if (index > fnHE.numVertices()) throw std::invalid_argument("index out of bounds.");

		float maxDIST = 100000;

		dist.clear();
		parent.clear();

		vector<bool> sptSet;

		// Initialize all distances as INFINITE and stpSet[] as false 
		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			dist.push_back(maxDIST);
			sptSet.push_back(false);

			parent.push_back(-2);
		}

		// Distance of source vertex from itself is always 0 
		dist[index] = 0;
		parent[index] = -1;

		// Find shortest path for all vertices 
		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			// Pick the minimum distance vertex from the set of vertices not 
			// yet processed. u is always equal to src in the first iteration. 
			zItGraphVertex u(*heObj, minDistance(dist, sptSet));

			// Mark the picked vertex as processed 
			sptSet[u.getId()] = true;

			// Update dist value of the adjacent vertices of the picked vertex. 

			vector<zItGraphVertex> cVerts;
			u.getConnectedVertices(cVerts);

			for (auto &v : cVerts)
			{

				zVector uPos = u.getPosition();

				zVector vPos = v.getPosition();

				float distUV = uPos.distanceTo(vPos);


				if (!sptSet[v.getId()] && dist[u.getId()] != maxDIST && dist[u.getId()] + distUV < dist[v.getId()])
				{
					dist[v.getId()] = dist[u.getId()] + distUV;
					parent[v.getId()] = u.getId();
				}
			}

		}


	}

	//---- mesh specilization for shortestDistance
	template<>
	ZSPACE_INLINE  void  zTsShortestPath<zObjMesh, zFnMesh>::shortestDistance(int index, vector<float> &dist, vector<int> &parent)
	{
		float maxDIST = 100000;

		dist.clear();
		parent.clear();

		vector<bool> sptSet;

		// Initialize all distances as INFINITE and stpSet[] as false 
		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			dist.push_back(maxDIST);
			sptSet.push_back(false);

			parent.push_back(-2);
		}

		// Distance of source vertex from itself is always 0 
		dist[index] = 0;
		parent[index] = -1;

		// Find shortest path for all vertices 
		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			// Pick the minimum distance vertex from the set of vertices not 
			// yet processed. u is always equal to src in the first iteration. 
			zItMeshVertex u(*heObj, minDistance(dist, sptSet));

			// Mark the picked vertex as processed 
			sptSet[u.getId()] = true;

			// Update dist value of the adjacent vertices of the picked vertex. 

			vector<zItMeshVertex> cVerts;
			u.getConnectedVertices(cVerts);

			for (auto &v : cVerts)
			{

				zVector uPos = u.getPosition();

				zVector vPos = v.getPosition();

				float distUV = uPos.distanceTo(vPos);


				if (!sptSet[v.getId()] && dist[u.getId()] != maxDIST && dist[u.getId()] + distUV < dist[v.getId()])
				{
					dist[v.getId()] = dist[u.getId()] + distUV;
					parent[v.getId()] = u.getId();
				}
			}

		}


	}

	//---- graph specilization for shortestPath
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::shortestPath_DistanceParent(int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
	{
		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId >= 0)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.halfEdgeExists(id, nextId, eId);

				if (chkEdge)
				{
					// update edge visits
					tempEdgePath.push_back(eId);
				}
			}


			id = nextId;

		} while (id >= 0);


		if (id == -1)
		{
			if (type == zEdgeVisited)
			{
				for (int i = 0; i < tempEdgePath.size(); i++)
				{
					edgePath[tempEdgePath[i]] ++;

					zItGraphHalfEdge he(*heObj, tempEdgePath[i]);

					int symEdge = he.getSym().getId();
					edgePath[symEdge] ++;
				}
			}

			if (type == zEdgePath)
			{
				edgePath.clear();
				edgePath = tempEdgePath;
			}

		}
		else
		{
			printf("\n shortest path between %i & %i doesnt exists as they are part of disconnected graph. ");
		}
	}

	//---- mesh specilization for shortestPath
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::shortestPath_DistanceParent(int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
	{
		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId >= 0)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.halfEdgeExists(id, nextId, eId);

				if (chkEdge)
				{
					// update edge visits
					tempEdgePath.push_back(eId);
				}
			}


			id = nextId;

		} while (id >= 0);


		if (id == -1)
		{
			if (type == zEdgeVisited)
			{
				for (int i = 0; i < tempEdgePath.size(); i++)
				{
					edgePath[tempEdgePath[i]] ++;

					zItMeshHalfEdge he(*heObj, tempEdgePath[i]);

					int symEdge = he.getSym().getId();
					edgePath[symEdge] ++;
				}
			}

			if (type == zEdgePath)
			{
				edgePath.clear();
				edgePath = tempEdgePath;
			}

		}
		else
		{
			printf("\n shortest path between %i & %i doesnt exists as they are part of disconnected mesh. ");
		}
	}

	//---- graph specilization for shortestPath
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::shortestPath(int indexA, int indexB, zWalkType type, vector<int> &edgePath)
	{

		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(indexA, dists, parent);

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId >= 0)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.halfEdgeExists(id, nextId, eId);

				if (chkEdge)
				{
					// update edge visits
					tempEdgePath.push_back(eId);
				}
			}


			id = nextId;

		} while (id >= 0);


		if (id == -1)
		{
			if (type == zEdgeVisited)
			{
				for (int i = 0; i < tempEdgePath.size(); i++)
				{
					zItGraphHalfEdge he(*heObj, tempEdgePath[i]);

					int symEdge = he.getSym().getId();
					edgePath[symEdge] ++;
				}
			}

			if (type == zEdgePath)
			{
				edgePath.clear();
				edgePath = tempEdgePath;
			}

		}
		else
		{
			printf("\n shortest path between %i & %i doesnt exists as they are part of disconnected graph. ");
		}
	}

	//---- mesh specilization for shortestPath
	template<>
	ZSPACE_INLINE  void  zTsShortestPath<zObjMesh, zFnMesh>::shortestPath(int indexA, int indexB, zWalkType type, vector<int> &edgePath)
	{
		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(indexA, dists, parent);

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId != -1)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.halfEdgeExists(id, nextId, eId);

				if (chkEdge)
				{
					// update edge visits
					tempEdgePath.push_back(eId);
				}
			}


			id = nextId;

		} while (id != -1);

		if (id == -1)
		{
			if (type == zEdgeVisited)
			{
				for (int i = 0; i < tempEdgePath.size(); i++)
				{
					edgePath[tempEdgePath[i]] ++;

					zItMeshHalfEdge he(*heObj, tempEdgePath[i]);

					int symEdge = he.getSym().getId();
					edgePath[symEdge] ++;
				}
			}

			if (type == zEdgePath)
			{
				edgePath.clear();
				edgePath = tempEdgePath;
			}

		}
		else
		{
			printf("\n shortest path between %i & %i doesnt exists as they are part of disconnected mesh. ");
		}

	}

	//---- graph specilization for getShortestPathGraph
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::getShortestPathGraph(int indexA, int indexB, zObjGraph & outGraph)
	{
		vector<int> edgePath;
		shortestPath(indexA, indexB, zEdgePath, edgePath);

		vector<zVector> positions;
		vector<int> vertexIds;
		vector<int> edgeConnects;

		unordered_map <string, int> positionVertex;


		for (int i = 0; i < edgePath.size(); i++)
		{
			zItGraphHalfEdge he(*heObj, edgePath[i]);



			zVector v1_pos = he.getVertex().getPosition();
			zVector v0_pos = he.getStartVertex().getPosition();

			int v0;
			bool chk0 = coreUtils.vertexExists(positionVertex, v0_pos, 3, v0);

			if (!chk0)
			{
				v0 = positions.size();
				coreUtils.addToPositionMap(positionVertex, v0_pos, v0, 3);
				positions.push_back(v0_pos);
			}

			int v1;
			bool chk1 = coreUtils.vertexExists(positionVertex, v1_pos, 3, v1);

			if (!chk1)
			{
				v1 = positions.size();
				coreUtils.addToPositionMap(positionVertex, v1_pos, v1, 3);
				positions.push_back(v1_pos);
			}

			edgeConnects.push_back(v0);
			edgeConnects.push_back(v1);

		}

		zFnGraph tempFn(outGraph);
		tempFn.create(positions, edgeConnects, false);

	}

	//---- mesh specilization for getShortestPathGraph
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::getShortestPathGraph(int indexA, int indexB, zObjGraph & outGraph)
	{
		vector<int> edgePath;
		shortestPath(indexA, indexB, zEdgePath, edgePath);

		vector<zVector> positions;
		vector<int> vertexIds;
		vector<int> edgeConnects;

		unordered_map <string, int> positionVertex;


		for (int i = 0; i < edgePath.size(); i++)
		{
			zItMeshHalfEdge he(*heObj, edgePath[i]);

			zVector v1_pos = he.getVertex().getPosition();
			zVector v0_pos = he.getStartVertex().getPosition();

			int v0;
			bool chk0 = coreUtils.vertexExists(positionVertex, v0_pos, 3, v0);

			if (!chk0)
			{
				v0 = positions.size();
				coreUtils.addToPositionMap(positionVertex, v0_pos, v0, 3);
				positions.push_back(v0_pos);
			}

			int v1;
			bool chk1 = coreUtils.vertexExists(positionVertex, v1_pos, 3, v1);

			if (!chk1)
			{
				v1 = positions.size();
				coreUtils.addToPositionMap(positionVertex, v1_pos, v1, 3);
				positions.push_back(v1_pos);
			}

			edgeConnects.push_back(v0);
			edgeConnects.push_back(v1);

		}

		zFnGraph tempFn(outGraph);
		tempFn.create(positions, edgeConnects, false);

	}

	//---- graph specilization for shortestPathWalks
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks(vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numHalfEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(i, dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), i);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent(i, j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::shortestPathWalks(vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numHalfEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(i, dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), i);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent(i, j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- graph specilization for shortestPathWalks_SourceToAll 
	template<>
	ZSPACE_INLINE  void  zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks_SourceToAll(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{

		// initialise edge visits to 0
		if (edgeVisited.size() == 0 || edgeVisited.size() < fnHE.numHalfEdges())
		{
			edgeVisited.clear();
			for (int i = 0; i < fnHE.numHalfEdges(); i++)
			{
				edgeVisited.push_back(0);
			}
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			// compute shortes path from all vertices to current vertex 
			for (int j = 0; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent(sourceVertices[i], j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks_SourceToAll 
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::shortestPathWalks_SourceToAll(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		// initialise edge visits to 0
		if (edgeVisited.size() == 0 || edgeVisited.size() < fnHE.numHalfEdges())
		{
			edgeVisited.clear();
			for (int i = 0; i < fnHE.numHalfEdges(); i++)
			{
				edgeVisited.push_back(0);
			}
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			// compute shortes path from all vertices to current vertex 
			for (int j = 0; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent(sourceVertices[i], j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- graph specilization for shortestPathWalks_SourceToOtherSource 
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks_SourceToOtherSource(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		vector<bool> computeDone;

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numHalfEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), sourceVertices[i]);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < sourceVertices.size(); j++)
			{
				shortestPath_DistanceParent(sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks_SourceToOtherSource 
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::shortestPathWalks_SourceToOtherSource(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		vector<bool> computeDone;

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numHalfEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), sourceVertices[i]);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < sourceVertices.size(); j++)
			{
				shortestPath_DistanceParent(sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- graph specilization for walking distance sources
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnGraph>::walk_DistanceFromSources(vector<int>& sourceVertices, vector<double>& vertexDistances)
	{
		float maxDIST = 100000;

		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vertexDistances.push_back(maxDIST);
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			for (int j = 0; j < dists.size(); j++)
			{
				if (dists[j] < vertexDistances[j]) vertexDistances[j] = dists[j];
			}
		}

	}

	//---- mesh specilization for walking distance sources
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::walk_DistanceFromSources(vector<int>& sourceVertices, vector<double>& vertexDistances)
	{
		float maxDIST = 100000;

		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vertexDistances.push_back(maxDIST);
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance(sourceVertices[i], dists, parent);

			for (int j = 0; j < dists.size(); j++)
			{
				if (dists[j] < vertexDistances[j]) vertexDistances[j] = dists[j];
			}
		}

	}

	//---- graph specilization for walking distance sources
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjGraph, zFnMesh>::walk_Animate(double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
	{
		currentWalkingEdges.clear();
		walkedEdges.clear();

		for (zItGraphEdge e(*heObj); !e.end(); e++)
		{
			zItGraphVertex v1 = e.getHalfEdge(0).getVertex();
			zItGraphVertex v2 = e.getHalfEdge(1).getVertex();

			zVector v1Pos = v1.getPosition();

			zVector v2Pos = v2.getPosition();

			if (vertexDistances[v1.getId()] <= MaxDistance && vertexDistances[v2.getId()] <= MaxDistance)
			{


				walkedEdges.push_back(v1Pos);
				walkedEdges.push_back(v2Pos);
			}

			else if (vertexDistances[v1.getId()] <= MaxDistance && vertexDistances[v2.getId()] > MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v1.getId()];

				zVector dir = v2Pos - v1Pos;
				dir.normalize();

				zVector pos = v1Pos + (dir * remainingDist);

				double TempDist = pos.distanceTo(v1Pos);
				double TempDist2 = v2Pos.distanceTo(v1Pos);

				if (TempDist < TempDist2)
				{
					currentWalkingEdges.push_back(v1Pos);
					currentWalkingEdges.push_back(pos);
				}
				else
				{
					walkedEdges.push_back(v1Pos);
					walkedEdges.push_back(v2Pos);
				}

			}

			else if (vertexDistances[v1.getId()] > MaxDistance && vertexDistances[v2.getId()] <= MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v2.getId()];

				zVector dir = v1Pos - v2Pos;
				dir.normalize();

				zVector pos = v2Pos + (dir * remainingDist);

				double TempDist = pos.distanceTo(v2Pos);
				double TempDist2 = v1Pos.distanceTo(v2Pos);

				if (TempDist < TempDist2)
				{
					currentWalkingEdges.push_back(v2Pos);
					currentWalkingEdges.push_back(pos);
				}
				else
				{
					walkedEdges.push_back(v1Pos);
					walkedEdges.push_back(v2Pos);
				}

			}
		}
	}

	//---- mesh specilization for walking distance sources
	template<>
	ZSPACE_INLINE  void zTsShortestPath<zObjMesh, zFnMesh>::walk_Animate(double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
	{

		currentWalkingEdges.clear();
		walkedEdges.clear();

		for (zItMeshEdge e(*heObj); !e.end(); e++)
		{
			zItMeshVertex v1 = e.getHalfEdge(0).getVertex();
			zItMeshVertex v2 = e.getHalfEdge(1).getVertex();

			zVector v1Pos = v1.getPosition();

			zVector v2Pos = v2.getPosition();

			if (vertexDistances[v1.getId()] <= MaxDistance && vertexDistances[v2.getId()] <= MaxDistance)
			{


				walkedEdges.push_back(v1Pos);
				walkedEdges.push_back(v2Pos);
			}

			else if (vertexDistances[v1.getId()] <= MaxDistance && vertexDistances[v2.getId()] > MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v1.getId()];

				zVector dir = v2Pos - v1Pos;
				dir.normalize();

				zVector pos = v1Pos + (dir * remainingDist);

				double TempDist = pos.distanceTo(v1Pos);
				double TempDist2 = v2Pos.distanceTo(v1Pos);

				if (TempDist < TempDist2)
				{
					currentWalkingEdges.push_back(v1Pos);
					currentWalkingEdges.push_back(pos);
				}
				else
				{
					walkedEdges.push_back(v1Pos);
					walkedEdges.push_back(v2Pos);
				}

			}

			else if (vertexDistances[v1.getId()] > MaxDistance && vertexDistances[v2.getId()] <= MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v2.getId()];

				zVector dir = v1Pos - v2Pos;
				dir.normalize();

				zVector pos = v2Pos + (dir * remainingDist);

				double TempDist = pos.distanceTo(v2Pos);
				double TempDist2 = v1Pos.distanceTo(v2Pos);

				if (TempDist < TempDist2)
				{
					currentWalkingEdges.push_back(v2Pos);
					currentWalkingEdges.push_back(pos);
				}
				else
				{
					walkedEdges.push_back(v1Pos);
					walkedEdges.push_back(v2Pos);
				}

			}
		}
	}

	//--- SPANNING TREE METHODS 

	template<typename T, typename U>	
	ZSPACE_INLINE int zTsShortestPath<T, U>::minDistance(vector<float> &dist, vector<bool> &sptSet)
	{
		if (dist.size() != sptSet.size()) throw std::invalid_argument("input container sizes are not equal.");

		// Initialize min value 
		int min = 100000, min_index;

		for (int i = 0; i < dist.size(); i++)
		{
			if (!sptSet[i] && dist[i] <= min)
			{
				min = dist[i];
				min_index = i;
			}
		}

		return min_index;
	}



#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zTsShortestPath<zObjMesh, zFnMesh>;

	template class zTsShortestPath<zObjGraph, zFnGraph>;

#endif

}