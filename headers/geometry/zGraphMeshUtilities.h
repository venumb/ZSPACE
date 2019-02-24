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

	/*! \brief This method computes the edge length of the input edge of zGraph/zMesh.
	*
	*	\tparam				T				- Type to work with zGraph and zMesh.
	*	\param		[in]	inMesh			- input graph or mesh.
	*	\param		[out]	index			- edge index.
	*	\return				double			- edge length.
	*	\since version 0.0.1
	*/
	template<typename T>
	double getEdgelength(T &inHEDataStructure, int index);

	/*! \brief This method computes the lengths of the edges of a zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[out]	edgeLengths				- vector of edge lengths.
	*	\return				double					- total edge lengths.
	*	\since version 0.0.1
	*/
	template<typename T>
	double getEdgeLengths(T &inHEDataStructure, vector<double> &edgeLengths);

	/*! \brief This method computes the edge vector of the input edge of zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[out]	index					- edge index.
	*	\return				zVector					- edge vector.
	*	\since version 0.0.1
	*/
	template<typename T>
	zVector getEdgeVector(T &inHEDataStructure, int index);

	/*! \brief This method sets edge color of of the input zGraph/zMesh edge and its symmetry edge to the input color.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	index					- input edge index.
	*	\param		[in]	col						- input color.
	*	\since version 0.0.1
	*/
	template<typename T>
	void setEdgeColor(T & inHEDataStructure, int index, zColor col);
	

	/*! \brief This method gets the ring neighbours of the input vertex of a zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	numRings				- number of rings.
	*	\param		[out]	ringNeighbours			- contatiner of neighbour vertex indicies.
	*	\since version 0.0.1
	*/
	template<typename T>
	void getNeighbourhoodRing(T &inHEDataStructure, int numRings, vector<int> &ringNeighbours);

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
	*	\param		[out]	edgeContainer			- container of edges of the shortest path(zEdgePath) or number of times an edge is visited(zEdgeVisited).
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPath_DistanceParent(T &inHEDataStructure, int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgeContainer);

	/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	indexA					- source vertex index.
	*	\param		[in]	indexB					- destination vertex index.
	*	\param		[in]	type					- zWalkType - zEdgePath or zEdgeVisited.
	*	\param		[out]	edgeContainer			- container of edges of the shortest path(zEdgePath) or number of times an edge is visited(zEdgeVisited).
	*	\since version 0.0.1
	*/
	template<typename T>
	void shortestPath(T &inHEDataStructure, int indexA, int indexB, zWalkType type, vector<int> &edgeContainer);

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

	/*! \brief This method computes the vertex distance to all the zGraph/zMesh vertices from the input vertex source indicies.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	sourceVertices			- container of source vertex indicies.
	*	\param		[out]	vertexDistances			- container of distance to each vertex from the source.
	*	\since version 0.0.1
	*/
	template<typename T>
	void walk_DistanceFromSources(T &inHEDataStructure, vector<int>& sourceVertices, vector<double>& vertexDistances);

	/*! \brief This method computes the vertex distance to all the zGraph/zMesh vertices from the input vertex source indicies.
	*
	*	\tparam				T						- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEDataStructure		- input graph or mesh.
	*	\param		[in]	MaxDistance				- maximum walk distance.
	*	\param		[in]	vertexDistances			- container of distance to each vertex from the source. To be computed using the method walkingDistance_Sources.
	*	\param		[out]	walkedEdges				- container of edges already walked - stores both the start and end positions of the edge.
	*	\param		[out]	currentWalkingEdges		- container of edges not completely walked - stores both the start and end positions.
	*	\since version 0.0.1
	*/
	template<typename T>
	void walk_Animate(T &inHEDataStructure, double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges);


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
double zSpace::getEdgelength(zGraph &inGraph, int index)
{
	if (index > inGraph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
	if (!inGraph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

	int v1 = inGraph.edges[index].getVertex()->getVertexId();
	int v2 = inGraph.edges[index].getSym()->getVertex()->getVertexId();

	double out = inGraph.vertexPositions[v1].distanceTo(inGraph.vertexPositions[v2]);

	return out;
}

//---- mesh specilization for getEdgeLength
template<>
double zSpace::getEdgelength(zMesh &inMesh, int index)
{
	if (index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
	if (!inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

	int v1 = inMesh.edges[index].getVertex()->getVertexId();
	int v2 = inMesh.edges[index].getSym()->getVertex()->getVertexId();

	double out = inMesh.vertexPositions[v1].distanceTo(inMesh.vertexPositions[v2]);

	return out;
}

//---------------//

//---- graph specilization for getEdgeLengths
template<>
double zSpace::getEdgeLengths(zGraph &inGraph, vector<double> &edgeLengths)
{
	double total = 0.0;

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

			total += e_len;
		}
		else
		{
			out.push_back(0);
			out.push_back(0);

		}


	}

	edgeLengths = out;

	return total;
}

//---- mesh specilization for getEdgeLengths
template<>
double zSpace::getEdgeLengths(zMesh &inMesh, vector<double> &edgeLengths)
{
	double total = 0.0;

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

			total += e_len;
		}
		else
		{
			out.push_back(0);
			out.push_back(0);

		}


	}

	edgeLengths = out;

	return total;
}

//---------------//

//---- graph specilization for getEdgeVector
template<>
zSpace::zVector zSpace::getEdgeVector(zGraph &inGraph, int index)
{
	if (index > inGraph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
	if (!inGraph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

	int v1 = inGraph.edges[index].getVertex()->getVertexId();
	int v2 = inGraph.edges[index].getSym()->getVertex()->getVertexId();

	zVector out = inGraph.vertexPositions[v1] - (inGraph.vertexPositions[v2]);

	return out;
}

//---- mesh specilization for getEdgeVector
template<>
zSpace::zVector zSpace::getEdgeVector(zMesh &inMesh, int index)
{
	if (index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
	if (!inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

	int v1 = inMesh.edges[index].getVertex()->getVertexId();
	int v2 = inMesh.edges[index].getSym()->getVertex()->getVertexId();

	zVector out = inMesh.vertexPositions[v1] - (inMesh.vertexPositions[v2]);

	return out;
}

//---------------//

//---- graph specilization for setEdgeColor
template<>
void zSpace::setEdgeColor(zGraph & inGraph, int index, zColor col)
{

	inGraph.edgeColors[index] = col;

	int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

	inGraph.edgeColors[symEdge] = col;

}

//---- mesh specilization for setEdgeColor
template<>
void zSpace::setEdgeColor(zMesh & inMesh, int index, zColor col)
{

	inMesh.edgeColors[index] = col;

	int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

	inMesh.edgeColors[symEdge] = col;

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
		for (int j = i + 1; j < sourceVertices.size(); j++)
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
		for (int j = i + 1; j < sourceVertices.size(); j++)
		{
			shortestPath_DistanceParent(inMesh, sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
		}
	}
}

//---------------//

////---- graph specilization for walking distance
//template<>
//void zSpace::walkingDistance(zGraph& inGraph, int index, double distance, vector<int>& edgeVisited)
//{
//	vector<float> dists;
//	vector<int> parent;
//
//	// get Dijkstra shortest distance spanning tree
//	shortestDistance(inGraph, index, dists, parent);
//
//	for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
//	{
//		if (!inGraph.edgeActive[i]) continue;
//
//		int v1 = inGraph.edges[i].getVertex()->getVertexId();
//		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();
//
//		if (dists[v1] < distance && dists[v2] < distance)
//		{
//			edgeVisited.push_back(i);
//			edgeVisited.push_back(i + 1);
//		}
//	}
//
//}
//
////---- mesh specilization for walking distance
//template<>
//void zSpace::walkingDistance(zMesh& inMesh, int index, double distance, vector<int>& edgeVisited)
//{
//	vector<float> dists;
//	vector<int> parent;
//
//	// get Dijkstra shortest distance spanning tree
//	shortestDistance(inMesh, index, dists, parent);
//
//	for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
//	{
//		if (!inMesh.edgeActive[i]) continue;
//
//		int v1 = inMesh.edges[i].getVertex()->getVertexId();
//		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();
//
//		if (dists[v1] < distance && dists[v2] < distance)
//		{
//			edgeVisited.push_back(i);
//			edgeVisited.push_back(i + 1);
//		}
//	}
//
//}

//---------------//

//---- graph specilization for walking distance sources
template<>
void zSpace::walk_DistanceFromSources(zGraph& inGraph, vector<int>& sourceVertices, vector<double>& vertexDistances)
{
	float maxDIST = 100000;

	for (int i = 0; i < inGraph.numVertices(); i++)
	{
		vertexDistances.push_back(maxDIST);
	}
	
	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inGraph, sourceVertices[i], dists, parent);

		for (int j = 0; j < dists.size(); j++)
		{
			if (dists[j] < vertexDistances[j]) vertexDistances[j] = dists[j];
		}
	}

}

//---- mesh specilization for walking distance sources
template<>
void zSpace::walk_DistanceFromSources(zMesh& inMesh, vector<int>& sourceVertices, vector<double>& vertexDistances)
{
	float maxDIST = 100000;

	for (int i = 0; i < inMesh.numVertices(); i++)
	{
		vertexDistances.push_back(maxDIST);
	}

	for (int i = 0; i < sourceVertices.size(); i++)
	{
		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inMesh, sourceVertices[i], dists, parent);

		for (int j = 0; j < dists.size(); j++)
		{
			if (dists[j] < vertexDistances[j]) vertexDistances[j] = dists[j];
		}
	}

}

//---------------//

//---- graph specilization for walking distance sources
template<>
void zSpace::walk_Animate(zGraph &inGraph, double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
{
	currentWalkingEdges.clear();
	walkedEdges.clear();

	for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
	{
		if (!inGraph.edgeActive[i]) continue;

		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

		if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] <= MaxDistance)
		{
			walkedEdges.push_back(inGraph.vertexPositions[v1]);
			walkedEdges.push_back(inGraph.vertexPositions[v2]);
		}

		else if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] > MaxDistance)
		{


			double remainingDist = MaxDistance - vertexDistances[v1];

			zVector dir = inGraph.vertexPositions[v2] - inGraph.vertexPositions[v1];
			dir.normalize();

			zVector pos = inGraph.vertexPositions[v1] + (dir * remainingDist);

			double TempDist = pos.distanceTo(inGraph.vertexPositions[v1]);
			double TempDist2 = inGraph.vertexPositions[v2].distanceTo(inGraph.vertexPositions[v1]);

			if (TempDist< TempDist2)
			{
				currentWalkingEdges.push_back(inGraph.vertexPositions[v1]);
				currentWalkingEdges.push_back(pos);
			}
			else
			{
				walkedEdges.push_back(inGraph.vertexPositions[v1]);
				walkedEdges.push_back(inGraph.vertexPositions[v2]);
			}

		}

		else if (vertexDistances[v1] > MaxDistance && vertexDistances[v2] <= MaxDistance)
		{


			double remainingDist = MaxDistance - vertexDistances[v2];

			zVector dir = inGraph.vertexPositions[v1] - inGraph.vertexPositions[v2];
			dir.normalize();

			zVector pos = inGraph.vertexPositions[v2] + (dir * remainingDist);

			double TempDist = pos.distanceTo(inGraph.vertexPositions[v2]);
			double TempDist2 = inGraph.vertexPositions[v1].distanceTo(inGraph.vertexPositions[v2]);

			if (TempDist < TempDist2)
			{
				currentWalkingEdges.push_back(inGraph.vertexPositions[v2]);
				currentWalkingEdges.push_back(pos);
			}
			else
			{
				walkedEdges.push_back(inGraph.vertexPositions[v1]);
				walkedEdges.push_back(inGraph.vertexPositions[v2]);
			}

		}
	}
}

//---- mesh specilization for walking distance sources
template<>
void zSpace::walk_Animate(zMesh &inMesh, double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
{
	currentWalkingEdges.clear();
	walkedEdges.clear();

	for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
	{
		if (!inMesh.edgeActive[i]) continue;

		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();

		if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] <= MaxDistance)
		{
			walkedEdges.push_back(inMesh.vertexPositions[v1]);
			walkedEdges.push_back(inMesh.vertexPositions[v2]);
		}

		else if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] > MaxDistance)
		{


			double remainingDist = MaxDistance - vertexDistances[v1];

			zVector dir = inMesh.vertexPositions[v2] - inMesh.vertexPositions[v1];
			dir.normalize();

			zVector pos = inMesh.vertexPositions[v1] + (dir * remainingDist);

			double TempDist = pos.distanceTo(inMesh.vertexPositions[v1]);
			double TempDist2 = inMesh.vertexPositions[v2].distanceTo(inMesh.vertexPositions[v1]);

			if (TempDist< TempDist2)
			{
				currentWalkingEdges.push_back(inMesh.vertexPositions[v1]);
				currentWalkingEdges.push_back(pos);
			}
			else
			{
				walkedEdges.push_back(inMesh.vertexPositions[v1]);
				walkedEdges.push_back(inMesh.vertexPositions[v2]);
			}

		}

		else if (vertexDistances[v1] > MaxDistance && vertexDistances[v2] <= MaxDistance)
		{


			double remainingDist = MaxDistance - vertexDistances[v2];

			zVector dir = inMesh.vertexPositions[v1] - inMesh.vertexPositions[v2];
			dir.normalize();

			zVector pos = inMesh.vertexPositions[v2] + (dir * remainingDist);

			double TempDist = pos.distanceTo(inMesh.vertexPositions[v2]);
			double TempDist2 = inMesh.vertexPositions[v1].distanceTo(inMesh.vertexPositions[v2]);

			if (TempDist < TempDist2)
			{
				currentWalkingEdges.push_back(inMesh.vertexPositions[v2]);
				currentWalkingEdges.push_back(pos);
			}
			else
			{
				walkedEdges.push_back(inMesh.vertexPositions[v1]);
				walkedEdges.push_back(inMesh.vertexPositions[v2]);
			}

		}
	}
}


//---------------//

//---- graph specilization for walking distance sources
//template<>
//void zSpace::getNeighbourhoodRing(zGraph &inGraph, int numRings, vector<int> &ringNeighbours)
//{
//	
//}

//---------------//
#endif /* DOXYGEN_SHOULD_SKIP_THIS */