#pragma once

#include <headers/geometry/zGraph.h>
#include <headers/geometry/zMesh.h>

namespace zSpace
{

/** \addtogroup zGeometry
*	\brief  The geometry classes, modifier and utility methods of the library.
*  @{
*/

/** \addtogroup zGeometryUtilities
*	\brief Collection of utility methods for graphs, meshes and fields.
*  @{
*/

/** \addtogroup zGraphMeshUtilities
*	\brief Collection of utility methods common for meshes and graphs.
*  @{
*/

	//--------------------------
	//--- GET METHODS 
	//--------------------------

	/*! \brief This method computes the centers of a all edges or faces of a zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	type					- zEdgeData or zFaceData.
	*	\param		[out]	centers					- vector of centers of type zVector.
	*	\since version 0.0.1
	*/
	template<typename T>
	void getCenters(T &inHEDataStructure, zHEData type, vector<zVector> &centers);

	/*! \brief This method computes the lengths of the edges of a zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[out]	edgeLengths				- vector of edge lengths.
	*	\since version 0.0.1
	*/
	template<typename T>
	void getEdgeLengths(T &inHEDataStructure, vector<double> &edgeLengths);

	//--------------------------
	//--- WALK METHODS 
	//--------------------------

	/*! \brief This method computes the shortest path from the source vertex to all vertices of the zGraph/zMesh.
	*
	*	\details based on Dijkstra’s shortest path algorithm (https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/)
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	index					- source vertex index.
	*	\param		[out]	dist					- container of distance to each vertex from source.
	*	\param		[out]	parent					- container of parent vertex index of each to each vertex. Required to get the path information.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestDistance(T &inHEDataStructure, int index, vector<float> &dist, vector<int> &parent);

	/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh. The distance and parent containers need to be computed before using the shortest distance method.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	indexA					- source vertex index.
	*	\param		[in]	indexB					- destination vertex index.
	*	\param		[in]	dist					- container of shortest distances to each vertex from the source. To be computed using the shortest distance method.
	*	\param		[in]	parent					- container of parent to each vertex. To be computed using the shortest distance method.
	*	\param		[in]	type					- zWalkType - zEdgePath or zEdgeVisited.
	*	\param		[out]	edgePath				- container of edges of the shortest path.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPath_DistanceParent(T &inHEDataStructure, int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath);

	/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	indexA					- source vertex index.
	*	\param		[in]	indexB					- destination vertex index.
	*	\param		[out]	edgePath				- container of edges of the shortest path.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPath(T &inHEDataStructure, int indexA, int indexB, zWalkType type, vector<int> &edgePath);

	/*! \brief This method computes the shortest path from the all vertices to all vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[out]	edgeVisited				- container of number of times edge is visited.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPathWalks(T &inHEDataStructure, vector<int> &edgeVisited);

	/*! \brief This method computes the shortest path from the all input source vertices to all other vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	sourceVertices			- input container of source vertex indicies.
	*	\param		[out]	edgeVisited				- container of number of times edge is visited.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPathWalks_SourceToAll(T &inHEDataStructure, vector<int> &sourceVertices, vector<int> &edgeVisited);

	/*! \brief This method computes the shortest path from the all input source vertices to all other input source vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	sourceVertices			- input container of source vertex indicies.
	*	\param		[out]	edgeVisited				- container of number of times edge is visited.
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPathWalks_SourceToOtherSource(T &inHEDataStructure,vector<int> &sourceVertices, vector<int> &edgeVisited);

	/*! \brief This method computes the edges in the zGraph/zMesh. within the input distance of the source vertex.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	index					- source vertex index.
	*	\param		[in]	distance				- max distance from source vertex.
	*	\param		[out]	edgeVisited				- container of number of times edge is visited.
	*	\since version 0.0.1
	*/
	template<typename T>
	void walkingDistance(T &inHEDataStructure, int index, double distance, vector<int>& edgeVisited);


/** @}*/

/** @}*/

/** @}*/

}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

//--------------------------
//---- TEMPLATE SPECIALIZATION DEFINITIONS 
//--------------------------

//---------------//

//---- graph specilization for getCenters
template<>
void zSpace::getCenters(zGraph &inGraph, zHEData type, vector<zVector> &centers)
{
	// Edge 
	if (type == zEdgeData)
	{
		vector<zVector> edgeCenters;

		edgeCenters.clear();

		for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
		{
			if (inGraph.edgeActive[i])
			{
				vector<int> eVerts;
				inGraph.getVertices(i, zEdgeData, eVerts);

				zVector cen = (inGraph.vertexPositions[eVerts[0]] + inGraph.vertexPositions[eVerts[1]]) * 0.5;

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
	else throw std::invalid_argument(" error: invalid zHEData type");
}

//---- mesh specilization for getCenters
template<>
void zSpace::getCenters(zMesh &inMesh, zHEData type, vector<zVector> &centers)
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

//---------------//

//---- graph specilization for getEdgeLengths
template<>
void zSpace::getEdgeLengths(zGraph &inGraph, vector<double> &edgeLengths)
{
	vector<double> out;

	for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
	{
		if (inGraph.edgeActive[i])
		{
			int v1 = inGraph.edges[i].getVertex()->getVertexId();
			int v2 = inGraph.edges[i].getSym()->getVertex()->getVertexId();

			zVector e = inGraph.vertexPositions[v1] - inGraph.vertexPositions[v2];
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

//---- mesh specilization for getEdgeLengths
template<>
void zSpace::getEdgeLengths(zMesh &inMesh, vector<double> &edgeLengths)
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

//---------------//

//---- graph specilization for shortestDistance
template<>
void zSpace::shortestDistance(zGraph &inGraph, int index, vector<float> &dist, vector<int> &parent)
{
	if (index >inGraph.vertexActive.size()) throw std::invalid_argument("index out of bounds.");

	float maxDIST = 100000;

	dist.clear();
	parent.clear();

	vector<bool> sptSet;

	// Initialize all distances as INFINITE and stpSet[] as false 
	for (int i = 0; i < inGraph.vertexActive.size(); i++)
	{
		dist.push_back(maxDIST);
		sptSet.push_back(false);

		parent.push_back(-2);
	}

	// Distance of source vertex from itself is always 0 
	dist[index] = 0;
	parent[index] = -1;

	// Find shortest path for all vertices 
	for (int i = 0; i < inGraph.vertexActive.size(); i++)
	{
		if (!inGraph.vertexActive[i]) continue;

		// Pick the minimum distance vertex from the set of vertices not 
		// yet processed. u is always equal to src in the first iteration. 
		int u = minDistance(dist, sptSet);

		// Mark the picked vertex as processed 
		sptSet[u] = true;

		// Update dist value of the adjacent vertices of the picked vertex. 

		vector<int> cVerts;
		inGraph.getConnectedVertices(u, zVertexData, cVerts);

		for (int j = 0; j < cVerts.size(); j++)
		{
			int v = cVerts[j];
			float distUV = inGraph.vertexPositions[u].distanceTo(inGraph.vertexPositions[v]);

			if (!sptSet[v] && dist[u] != maxDIST && dist[u] + distUV < dist[v])
			{
				dist[v] = dist[u] + distUV;
				parent[v] = u;
			}
		}

	}


}

//---- mesh specilization for shortestDistance
template<>
void  zSpace::shortestDistance(zMesh &inMesh, int index, vector<float> &dist, vector<int> &parent)
{
	float maxDIST = 100000;

	dist.clear();
	parent.clear();

	vector<bool> sptSet;

	// Initialize all distances as INFINITE and stpSet[] as false 
	for (int i = 0; i < inMesh.vertexActive.size(); i++)
	{
		dist.push_back(maxDIST);
		sptSet.push_back(false);

		parent.push_back(-2);
	}

	// Distance of source vertex from itself is always 0 
	dist[index] = 0;
	parent[index] = -1;

	// Find shortest path for all vertices 
	for (int i = 0; i < inMesh.vertexActive.size(); i++)
	{
		if (!inMesh.vertexActive[i]) continue;

		// Pick the minimum distance vertex from the set of vertices not 
		// yet processed. u is always equal to src in the first iteration. 
		int u = minDistance(dist, sptSet);

		// Mark the picked vertex as processed 
		sptSet[u] = true;

		// Update dist value of the adjacent vertices of the picked vertex. 

		vector<int> cVerts;
		inMesh.getConnectedVertices(u, zVertexData, cVerts);

		for (int j = 0; j < cVerts.size(); j++)
		{
			int v = cVerts[j];
			float distUV = inMesh.vertexPositions[u].distanceTo(inMesh.vertexPositions[v]);

			if (!sptSet[v] && dist[u] != maxDIST && dist[u] + distUV < dist[v])
			{
				dist[v] = dist[u] + distUV;
				parent[v] = u;
			}
		}

	}


}

//---------------//

//---- graph specilization for shortestPath
template<>
void zSpace::shortestPath_DistanceParent(zGraph &inGraph, int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
{
	vector<int> tempEdgePath;

	if (indexA >inGraph.vertexActive.size()) throw std::invalid_argument("indexA out of bounds.");
	if (indexB >inGraph.vertexActive.size()) throw std::invalid_argument("indexB out of bounds.");

	int id = indexB;

	do
	{
		int nextId = parent[id];
		if (nextId >= 0)
		{
			// get the edge if it exists
			int eId;
			bool chkEdge = inGraph.edgeExists(id, nextId, eId);

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

				int symEdge = inGraph.edges[tempEdgePath[i]].getSym()->getEdgeId();
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
void zSpace::shortestPath_DistanceParent(zMesh &inMesh, int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
{
	vector<int> tempEdgePath;

	if (indexA >inMesh.vertexActive.size()) throw std::invalid_argument("indexA out of bounds.");
	if (indexB >inMesh.vertexActive.size()) throw std::invalid_argument("indexB out of bounds.");

	int id = indexB;

	do
	{
		int nextId = parent[id];
		if (nextId >= 0)
		{
			// get the edge if it exists
			int eId;
			bool chkEdge = inMesh.edgeExists(id, nextId, eId);

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

				int symEdge = inMesh.edges[tempEdgePath[i]].getSym()->getEdgeId();
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

//---------------//

//---- graph specilization for shortestPath
template<>
void zSpace::shortestPath(zGraph &inGraph, int indexA, int indexB, zWalkType type, vector<int> &edgePath )
{
	

	vector<int> tempEdgePath;

	if (indexA >inGraph.vertexActive.size()) throw std::invalid_argument("indexA out of bounds.");
	if (indexB >inGraph.vertexActive.size()) throw std::invalid_argument("indexB out of bounds.");

	vector<float> dists;
	vector<int> parent;

	// get Dijkstra shortest distance spanning tree
	shortestDistance(inGraph, indexA, dists, parent);

	int id = indexB;

	do
	{
		int nextId = parent[id];
		if (nextId >= 0)
		{
			// get the edge if it exists
			int eId;
			bool chkEdge = inGraph.edgeExists(id, nextId, eId);

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

				int symEdge = inGraph.edges[tempEdgePath[i]].getSym()->getEdgeId();
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
void  zSpace::shortestPath(zMesh &inMesh, int indexA, int indexB, zWalkType type, vector<int> &edgePath)
{
	vector<int> tempEdgePath;

	if (indexA >inMesh.vertexActive.size()) throw std::invalid_argument("indexA out of bounds.");
	if (indexB >inMesh.vertexActive.size()) throw std::invalid_argument("indexB out of bounds.");

	vector<float> dists;
	vector<int> parent;

	// get Dijkstra shortest distance spanning tree
	shortestDistance(inMesh, indexA, dists, parent);

	int id = indexB;

	do
	{
		int nextId = parent[id];
		if (nextId != -1)
		{
			// get the edge if it exists
			int eId;
			bool chkEdge = inMesh.edgeExists(id, nextId, eId);

			if (chkEdge)
			{
				// update edge visits
				edgePath.push_back(eId);
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

				int symEdge = inMesh.edges[tempEdgePath[i]].getSym()->getEdgeId();
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

//---------------//

//---- graph specilization for shortestPathWalks
template<>
void zSpace::shortestPathWalks(zGraph &inGraph, vector<int> &edgeVisited)
{
	edgeVisited.clear();		

	// initialise edge visits to 0
	for (int i = 0; i < inGraph.numEdges(); i++)
	{
		edgeVisited.push_back(0);
	}


	for (int i = 0; i < inGraph.numVertices(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inGraph, i, dists, parent);

		printf("\n total: %i source: %i ", inGraph.numVertices(), i);

		// compute shortes path from all vertices to current vertex 
		for (int j = i+1; j < inGraph.numVertices(); j++)
		{
			shortestPath_DistanceParent(inGraph, i, j, dists,parent, zEdgeVisited,edgeVisited );
		}
	}
}

//---- mesh specilization for shortestPathWalks
template<>
void zSpace::shortestPathWalks(zMesh &inMesh, vector<int> &edgeVisited)
{
	edgeVisited.clear();

	// initialise edge visits to 0
	for (int i = 0; i < inMesh.numEdges(); i++)
	{
		edgeVisited.push_back(0);
	}


	for (int i = 0; i < inMesh.numVertices(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inMesh, i, dists, parent);

		printf("\n total: %i source: %i ", inMesh.numVertices(), i);

		// compute shortes path from all vertices to current vertex 
		for (int j = i + 1; j < inMesh.numVertices(); j++)
		{
			shortestPath_DistanceParent(inMesh, i, j, dists, parent, zEdgeVisited, edgeVisited);
		}
	}
}

//---------------//

//---- graph specilization for shortestPathWalks_SourceToAll 
template<>
void zSpace::shortestPathWalks_SourceToAll(zGraph &inGraph, vector<int> &sourceVertices, vector<int> &edgeVisited)
{		

	// initialise edge visits to 0
	if (edgeVisited.size() == 0 || edgeVisited.size() < inGraph.numEdges())
	{
		edgeVisited.clear();
		for (int i = 0; i < inGraph.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}
	}

	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inGraph, sourceVertices[i], dists, parent);

		// compute shortes path from all vertices to current vertex 
		for (int j = 0; j < inGraph.numVertices(); j++)
		{
			shortestPath_DistanceParent(inGraph, sourceVertices[i], j,dists,parent,zEdgeVisited, edgeVisited);
		}
	}
}

//---- mesh specilization for shortestPathWalks_SourceToAll 
template<>
void zSpace::shortestPathWalks_SourceToAll(zMesh &inMesh, vector<int> &sourceVertices, vector<int> &edgeVisited)
{
	// initialise edge visits to 0
	if (edgeVisited.size() == 0 || edgeVisited.size() < inMesh.numEdges())
	{
		edgeVisited.clear();
		for (int i = 0; i < inMesh.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}
	}

	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inMesh, sourceVertices[i], dists, parent);

		// compute shortes path from all vertices to current vertex 
		for (int j = 0; j < inMesh.numVertices(); j++)
		{
			shortestPath_DistanceParent(inMesh, sourceVertices[i], j,dists,parent,zEdgeVisited, edgeVisited);
		}
	}
}

//---------------//

//---- graph specilization for shortestPathWalks_SourceToOtherSource 
template<>
void zSpace::shortestPathWalks_SourceToOtherSource(zGraph &inGraph, vector<int> &sourceVertices, vector<int> &edgeVisited)
{
	edgeVisited.clear();

	vector<bool> computeDone;

	// initialise edge visits to 0
	for (int i = 0; i < inGraph.numEdges(); i++)
	{
		edgeVisited.push_back(0);
	}

	
	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inGraph, sourceVertices[i], dists, parent);

		printf("\n total: %i source: %i ", inGraph.numVertices(), sourceVertices[i]);

		// compute shortes path from all vertices to current vertex 
		for (int j = sourceVertices[i] + 1; j < sourceVertices.size(); j++)
		{
			shortestPath_DistanceParent(inGraph, sourceVertices[i], sourceVertices[j],dists,parent,zEdgeVisited, edgeVisited);			
		}
	}
}

//---- mesh specilization for shortestPathWalks_SourceToOtherSource 
template<>
void zSpace::shortestPathWalks_SourceToOtherSource(zMesh &inMesh, vector<int> &sourceVertices, vector<int> &edgeVisited)
{
	edgeVisited.clear();

	vector<bool> computeDone;

	// initialise edge visits to 0
	for (int i = 0; i < inMesh.numEdges(); i++)
	{
		edgeVisited.push_back(0);
	}


	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inMesh, sourceVertices[i], dists, parent);

		printf("\n total: %i source: %i ", inMesh.numVertices(), sourceVertices[i]);

		// compute shortes path from all vertices to current vertex 
		for (int j = sourceVertices[i] + 1; j < sourceVertices.size(); j++)
		{
			shortestPath_DistanceParent(inMesh, sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
		}
	}
}

//---------------//

//---- graph specilization for walking distance
template<>
void zSpace::walkingDistance(zGraph& inGraph, int index, double distance, vector<int>& edgeVisited)
{
	vector<float> dists;
	vector<int> parent;

	// get Dijkstra shortest distance spanning tree
	shortestDistance(inGraph, index, dists, parent);

	for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
	{
		if (!inGraph.edgeActive[i]) continue;

		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

		if (dists[v1] < distance && dists[v2] < distance)
		{
			edgeVisited.push_back(i);
			edgeVisited.push_back(i + 1);
		}
	}

}

//---- graph specilization for walking distance
template<>
void zSpace::walkingDistance(zMesh& inMesh, int index, double distance, vector<int>& edgeVisited)
{
	vector<float> dists;
	vector<int> parent;

	// get Dijkstra shortest distance spanning tree
	shortestDistance(inMesh, index, dists, parent);

	for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
	{
		if (!inMesh.edgeActive[i]) continue;

		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();

		if (dists[v1] < distance && dists[v2] < distance)
		{
			edgeVisited.push_back(i);
			edgeVisited.push_back(i + 1);
		}
	}

}

//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */