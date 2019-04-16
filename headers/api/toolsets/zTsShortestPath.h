#pragma once

#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>

namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/*! \class zTsShortestPath
	*	\brief A walk toolset for doing shortest paths on graphs and meshes.
	*	\tparam				T			- Type to work with zObjMesh or zObjGraph.
	*	\tparam				U			- Type to work with zFnMesh or zFnGraph.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	
	template<typename T, typename U>
	class zTsShortestPath
	{
	protected:
		/*!	\brief pointer to half edge Object  */		
		T *heObj;

	public:
		
		/*!	\brief half edge function set  */
		U fnHE;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsShortestPath() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_heObj			- input half edge obj.
		*	\since version 0.0.2
		*/
		zTsShortestPath(T &_heObj);
		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsShortestPath() {}

		//--------------------------
		//--- WALK METHODS 
		//--------------------------

		/*! \brief This method computes the shortest path from the source vertex to all vertices of the zGraph/zMesh.
		*
		*	\details based on Dijkstras shortest path algorithm (https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/)
		*	\param		[in]	index					- source vertex index.
		*	\param		[out]	dist					- container of distance to each vertex from source.
		*	\param		[out]	parent					- container of parent vertex index of each to each vertex. Required to get the path information.
		*	\since version 0.0.2
		*/
		void shortestDistance(int index, vector<float> &dist, vector<int> &parent);

		/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh. The distance and parent containers need to be computed before using the shortest distance method.
		*
		*	\param		[in]	indexA					- source vertex index.
		*	\param		[in]	indexB					- destination vertex index.
		*	\param		[in]	dist					- container of shortest distances to each vertex from the source. To be computed using the shortest distance method.
		*	\param		[in]	parent					- container of parent to each vertex. To be computed using the shortest distance method.
		*	\param		[in]	type					- zWalkType - zEdgePath or zEdgeVisited.
		*	\param		[out]	edgeContainer			- container of edges of the shortest path(zEdgePath) or number of times an edge is visited(zEdgeVisited).
		*	\since version 0.0.2
		*/
		void shortestPath_DistanceParent( int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgeContainer);

		/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh.
		*
		*	\param		[in]	indexA					- source vertex index.
		*	\param		[in]	indexB					- destination vertex index.
		*	\param		[in]	type					- zWalkType - zEdgePath or zEdgeVisited.
		*	\param		[out]	edgeContainer			- container of edges of the shortest path(zEdgePath) or number of times an edge is visited(zEdgeVisited).
		*	\since version 0.0.2
		*/
		void shortestPath(int indexA, int indexB, zWalkType type, vector<int> &edgeContainer);

		/*! \brief This method computes the shortest path from the all vertices to all vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
		*
		*	\param		[out]	edgeVisited				- container of number of times edge is visited.
		*	\since version 0.0.2
		*/
		void shortestPathWalks(vector<int> &edgeVisited);

		/*! \brief This method computes the shortest path from the all input source vertices to all other vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
		*
		*	\param		[in]	sourceVertices			- input container of source vertex indicies.
		*	\param		[out]	edgeVisited				- container of number of times edge is visited.
		*	\since version 0.0.2
		*/
		void shortestPathWalks_SourceToAll(vector<int> &sourceVertices, vector<int> &edgeVisited);

		/*! \brief This method computes the shortest path from the all input source vertices to all other input source vertices of a zGraph/zMesh and returns the number of times an edge is visited in those walks.
		*
		*	\param		[in]	sourceVertices			- input container of source vertex indicies.
		*	\param		[out]	edgeVisited				- container of number of times edge is visited.
		*	\since version 0.0.2
		*/

		void shortestPathWalks_SourceToOtherSource(vector<int> &sourceVertices, vector<int> &edgeVisited);

		/*! \brief This method computes the vertex distance to all the zGraph/zMesh vertices from the input vertex source indicies.
		*
		*	\param		[in]	sourceVertices			- container of source vertex indicies.
		*	\param		[out]	vertexDistances			- container of distance to each vertex from the source.
		*	\since version 0.0.2
		*/
		void walk_DistanceFromSources(vector<int>& sourceVertices, vector<double>& vertexDistances);

		/*! \brief This method computes the vertex distance to all the zGraph/zMesh vertices from the input vertex source indicies.
		*
		*	\param		[in]	MaxDistance				- maximum walk distance.
		*	\param		[in]	vertexDistances			- container of distance to each vertex from the source. To be computed using the method walkingDistance_Sources.
		*	\param		[out]	walkedEdges				- container of edges already walked - stores both the start and end positions of the edge.
		*	\param		[out]	currentWalkingEdges		- container of edges not completely walked - stores both the start and end positions.
		*	\since version 0.0.2
		*/
		void walk_Animate( double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges);

		//--------------------------
		//--- SPANNING TREE METHODS 
		//--------------------------

		/*! \brief This method returns the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree. To be used with shortestDistance method for mesh/graph.
		*
		*	\details based on Dijkstra’s shortest path algorithm (https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/)
		*	\param		[out]	dist					- container of distance to each vertex from source.
		*	\param		[out]	sptSet					- container of shortest path tree for each vertex..
		*	\since version 0.0.2
		*/
		int minDistance(vector<float> &dist, vector<bool> &sptSet)
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

	};


#ifndef DOXYGEN_SHOULD_SKIP_THIS

	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- graph specilization for zTsShortestPath constructor
	template<>
	inline zTsShortestPath<zObjGraph, zFnGraph>::zTsShortestPath(zObjGraph & _graph)
	{
		heObj = &_graph;
		fnHE = zFnGraph(_graph);
	}

	//---- mesh specilization for zTsShortestPath constructor
	template<>
	inline zTsShortestPath<zObjMesh, zFnMesh>::zTsShortestPath(zObjMesh & _mesh)
	{
		heObj = &_mesh;
		fnHE = zFnMesh(_mesh);
	}

	//---------------//	

	//---- graph specilization for shortestDistance
	template<>
	inline void zTsShortestPath<zObjGraph, zFnGraph>::shortestDistance(int index, vector<float> &dist, vector<int> &parent)
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
			int u = minDistance(dist, sptSet);

			// Mark the picked vertex as processed 
			sptSet[u] = true;

			// Update dist value of the adjacent vertices of the picked vertex. 

			vector<int> cVerts;
			fnHE.getConnectedVertices(u, zVertexData, cVerts);

			for (int j = 0; j < cVerts.size(); j++)
			{
				int v = cVerts[j];
				
				zVector uPos = 	fnHE.getVertexPosition(u);

				zVector vPos  = fnHE.getVertexPosition(v);

				float distUV = uPos.distanceTo(vPos);
				

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
	inline void  zTsShortestPath<zObjMesh,zFnMesh>::shortestDistance(int index, vector<float> &dist, vector<int> &parent)
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
			int u = minDistance(dist, sptSet);

			// Mark the picked vertex as processed 
			sptSet[u] = true;

			// Update dist value of the adjacent vertices of the picked vertex. 

			vector<int> cVerts;
			fnHE.getConnectedVertices(u, zVertexData, cVerts);

			for (int j = 0; j < cVerts.size(); j++)
			{
				int v = cVerts[j];

				zVector uPos =	fnHE.getVertexPosition(u);

				zVector vPos = fnHE.getVertexPosition(v);

				float distUV = uPos.distanceTo(vPos);				

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
	inline void zTsShortestPath<zObjGraph, zFnGraph>::shortestPath_DistanceParent(int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
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
				bool chkEdge = fnHE.edgeExists(id, nextId, eId);

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

					int symEdge = fnHE.getSymIndex(tempEdgePath[i]);
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
	inline void zTsShortestPath<zObjMesh, zFnMesh>::shortestPath_DistanceParent(int indexA, int indexB, vector<float> &dist, vector<int> &parent, zWalkType type, vector<int> &edgePath)
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
				bool chkEdge = fnHE.edgeExists(id, nextId, eId);

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

					int symEdge = fnHE.getSymIndex(tempEdgePath[i]);
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
	inline void zTsShortestPath<zObjGraph, zFnGraph>::shortestPath( int indexA, int indexB, zWalkType type, vector<int> &edgePath)
	{


		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance( indexA, dists, parent);

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId >= 0)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.edgeExists(id, nextId, eId);

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

					int symEdge = fnHE.getSymIndex(tempEdgePath[i]);
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
	inline void  zTsShortestPath<zObjMesh, zFnMesh>::shortestPath(int indexA, int indexB, zWalkType type, vector<int> &edgePath)
	{
		vector<int> tempEdgePath;

		if (indexA > fnHE.numVertices()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB > fnHE.numVertices()) throw std::invalid_argument("indexB out of bounds.");

		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance( indexA, dists, parent);

		int id = indexB;

		do
		{
			int nextId = parent[id];
			if (nextId != -1)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = fnHE.edgeExists(id, nextId, eId);

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

					int symEdge = fnHE.getSymIndex(tempEdgePath[i]);
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
	inline void zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks(vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		// initialise edge visits to 0
		for (int i = 0; i <fnHE.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( i, dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), i);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent( i, j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks
	template<>
	inline void zTsShortestPath<zObjMesh, zFnMesh>::shortestPathWalks(vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < fnHE.numVertices(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( i, dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), i);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent( i, j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---------------//

	//---- graph specilization for shortestPathWalks_SourceToAll 
	template<>
	inline void  zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks_SourceToAll(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{

		// initialise edge visits to 0
		if (edgeVisited.size() == 0 || edgeVisited.size() < fnHE.numEdges())
		{
			edgeVisited.clear();
			for (int i = 0; i < fnHE.numEdges(); i++)
			{
				edgeVisited.push_back(0);
			}
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( sourceVertices[i], dists, parent);

			// compute shortes path from all vertices to current vertex 
			for (int j = 0; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent( sourceVertices[i], j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks_SourceToAll 
	template<>
	inline void zTsShortestPath<zObjMesh,zFnMesh>::shortestPathWalks_SourceToAll(vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		// initialise edge visits to 0
		if (edgeVisited.size() == 0 || edgeVisited.size() < fnHE.numEdges())
		{
			edgeVisited.clear();
			for (int i = 0; i < fnHE.numEdges(); i++)
			{
				edgeVisited.push_back(0);
			}
		}

		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( sourceVertices[i], dists, parent);

			// compute shortes path from all vertices to current vertex 
			for (int j = 0; j < fnHE.numVertices(); j++)
			{
				shortestPath_DistanceParent( sourceVertices[i], j, dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---------------//

	//---- graph specilization for shortestPathWalks_SourceToOtherSource 
	template<>
	inline void zTsShortestPath<zObjGraph, zFnGraph>::shortestPathWalks_SourceToOtherSource( vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		vector<bool> computeDone;

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( sourceVertices[i], dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), sourceVertices[i]);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < sourceVertices.size(); j++)
			{
				shortestPath_DistanceParent( sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---- mesh specilization for shortestPathWalks_SourceToOtherSource 
	template<>
	inline void zTsShortestPath<zObjMesh, zFnMesh>::shortestPathWalks_SourceToOtherSource( vector<int> &sourceVertices, vector<int> &edgeVisited)
	{
		edgeVisited.clear();

		vector<bool> computeDone;

		// initialise edge visits to 0
		for (int i = 0; i < fnHE.numEdges(); i++)
		{
			edgeVisited.push_back(0);
		}


		for (int i = 0; i < sourceVertices.size(); i++)
		{
			vector<float> dists;
			vector<int> parent;

			// get Dijkstra shortest distance spanning tree
			shortestDistance( sourceVertices[i], dists, parent);

			printf("\n total: %i source: %i ", fnHE.numVertices(), sourceVertices[i]);

			// compute shortes path from all vertices to current vertex 
			for (int j = i + 1; j < sourceVertices.size(); j++)
			{
				shortestPath_DistanceParent( sourceVertices[i], sourceVertices[j], dists, parent, zEdgeVisited, edgeVisited);
			}
		}
	}

	//---------------//

	//---- graph specilization for walking distance sources
	template<>
	inline void zTsShortestPath<zObjGraph, zFnGraph>::walk_DistanceFromSources( vector<int>& sourceVertices, vector<double>& vertexDistances)
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
			shortestDistance( sourceVertices[i], dists, parent);

			for (int j = 0; j < dists.size(); j++)
			{
				if (dists[j] < vertexDistances[j]) vertexDistances[j] = dists[j];
			}
		}

	}

	//---- mesh specilization for walking distance sources
	template<>
	inline void zTsShortestPath<zObjMesh, zFnMesh>::walk_DistanceFromSources( vector<int>& sourceVertices, vector<double>& vertexDistances)
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

	//---------------//

	//---- graph specilization for walking distance sources
	template<>
	inline void zTsShortestPath<zObjGraph, zFnMesh>::walk_Animate(double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
	{
		currentWalkingEdges.clear();
		walkedEdges.clear();

		for (int i = 0; i < fnHE.numEdges(); i += 2)
		{
			int v1 = fnHE.getEndVertexIndex(i); 
			int v2 = fnHE.getEndVertexIndex(i+1);

			zVector v1Pos = fnHE.getVertexPosition(v1);

			zVector v2Pos = fnHE.getVertexPosition(v2);

			if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] <= MaxDistance)
			{
				

				walkedEdges.push_back(v1Pos);
				walkedEdges.push_back(v2Pos);
			}

			else if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] > MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v1];

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

			else if (vertexDistances[v1] > MaxDistance && vertexDistances[v2] <= MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v2];

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
	inline void zTsShortestPath<zObjMesh,zFnMesh>::walk_Animate(double MaxDistance, vector<double>& vertexDistances, vector<zVector>& walkedEdges, vector<zVector>& currentWalkingEdges)
	{

		currentWalkingEdges.clear();
		walkedEdges.clear();

		for (int i = 0; i < fnHE.numEdges(); i += 2)
		{
			int v1 = fnHE.getEndVertexIndex(i);
			int v2 = fnHE.getEndVertexIndex(i + 1);

			zVector v1Pos = fnHE.getVertexPosition(v1);

			zVector v2Pos = fnHE.getVertexPosition(v2);

			if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] <= MaxDistance)
			{


				walkedEdges.push_back(v1Pos);
				walkedEdges.push_back(v2Pos);
			}

			else if (vertexDistances[v1] <= MaxDistance && vertexDistances[v2] > MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v1];

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

			else if (vertexDistances[v1] > MaxDistance && vertexDistances[v2] <= MaxDistance)
			{


				double remainingDist = MaxDistance - vertexDistances[v2];

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
	
	//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/*! \typedef zTsShortestPathMesh
	*	\brief A shortest path object for meshes.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsShortestPath<zObjMesh, zFnMesh> zTsShortestPathMesh;

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/*! \typedef zTsShortestPathGraph
	*	\brief A shortest path object for graphs.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsShortestPath<zObjGraph, zFnGraph> zTsShortestPathGraph;
	

}

