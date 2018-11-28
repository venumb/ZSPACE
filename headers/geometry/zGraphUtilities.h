#pragma once

#include <headers/geometry/zGraph.h>


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

	/** \addtogroup zGraphUtilities
	*	\brief Collection of utility methods for graphs.
	*  @{
	*/

	/*! \brief This method returns the total edge length of the graph
	*
	*	\param		[in]	inGraph	- input graph.
	*	\since version 0.0.1
	*/
	double totalEdgeLength(zGraph &inGraph)
	{
		double out = 0.0;

		for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
		{
			if (inGraph.edgeActive[i])
			{
				int v1 = inGraph.edges[i].getVertex()->getVertexId();
				int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

				zVector posV2 = inGraph.vertexPositions[v2];
				double dist = inGraph.vertexPositions[v1].distanceTo(posV2);
				//printf("\n %1.2f ", dist);

				out += dist;
			}
		}

		return out;
	}
	
	/*! \brief This method sets vertex color of all the vertices to the input color. 
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.	
	*	\since version 0.0.1
	*/
	
	void setVertexColor(zGraph &ingraph, zColor col, bool setEdgeColor)
	{

		for (int i = 0; i < ingraph.vertexColors.size(); i++)
		{
			ingraph.vertexColors[i] = col;
		}

		if (setEdgeColor) ingraph.computeEdgeColorfromVertexColor();

	}

	/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the graph.
	*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/

	void setVertexColors(zGraph &ingraph, vector<zColor>& col, bool setEdgeColor)
	{
		if (col.size() != ingraph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < ingraph.vertexColors.size(); i++)
		{
			ingraph.vertexColors[i] = col[i];
		}

		if (setEdgeColor) ingraph.computeEdgeColorfromVertexColor();
	}

	/*! \brief This method sets edge color of all the edges to the input color.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/

	void setEdgeColor(zGraph & ingraph, zColor col, bool setVertexColor)
	{
		for (int i = 0; i < ingraph.edgeColors.size(); i++)
		{
			ingraph.edgeColors[i] = col;
		}

		if (setVertexColor) ingraph.computeVertexColorfromEdgeColor();

	}

	/*! \brief This method sets edge color of all the vertices with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of edges in the graph.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/

	void setEdgeColors(zGraph & ingraph, vector<zColor>& col, bool setVertexColor)
	{
		if (col.size() != ingraph.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph edges.");

		for (int i = 0; i < ingraph.edgeColors.size(); i++)
		{
			ingraph.edgeColors[i] = col[i];
		}

		if (setVertexColor) ingraph.computeVertexColorfromEdgeColor();
	}
	

	//--------------------------
	//--- WALK METHODS 
	//--------------------------


	/*! \brief This method computes the shortest path from the source vertex to all vertices of the graph.
	*
	*	\details based on Dijkstra’s shortest path algorithm (https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/)
	*	\param		[in]	inGraph					- input graph.
	*	\param		[in]	index					- source vertex index.
	*	\param		[out]	dist					- container of distance to each vertex from source.
	*	\param		[out]	parent					- container of parent vertex index of each to each vertex. Required to get the path information.
	*	\since version 0.0.1
	*/
	void shortestDistance(zGraph &inGraph, int index, vector<float> &dist, vector<int> &parent)
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


	/*! \brief This method computes the shortest path from the source vertex to destination vertex of the graph.
	*	
	*	\param		[in]	inGraph					- input graph.
	*	\param		[in]	indexA					- source vertex index.
	*	\param		[in]	indexB					- destination vertex index.
	*	\param		[out]	edgePath				- container of edges of the shortest path.	
	*	\since version 0.0.1
	*/
	void shortestPath(zGraph &inGraph, int indexA, int indexB, vector<int> &edgePath)
	{
		edgePath.clear();

		if(indexA >inGraph.vertexActive.size()) throw std::invalid_argument("indexA out of bounds.");
		if (indexB >inGraph.vertexActive.size()) throw std::invalid_argument("indexB out of bounds.");

		vector<float> dists;
		vector<int> parent;

		// get Dijkstra shortest distance spanning tree
		shortestDistance(inGraph, indexA, dists, parent);

		int id = indexB;
		
		do
		{
			int nextId = parent[id];
			if (nextId != -1)
			{
				// get the edge if it exists
				int eId;
				bool chkEdge = inGraph.edgeExists(id, nextId, eId);

				if (chkEdge)
				{
					// update edge visits
					edgePath.push_back(eId);
				}
			}


			id = nextId;

		} while (id != -1);	
		
	}

	/*! \brief This method computes the shortest path from the all vertices to all vertices of a graph and returns the number of times an edge is visited in those walks.
	*
	*	\param		[in]	inGraph					- input graph.
	*	\param		[out]	edgeVisited				- container of number of times edge is visited.
	*	\since version 0.0.1
	*/
	void shortestPathWalks(zGraph &inGraph, vector<int> &edgeVisited)
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

			// compute shortes path from all vertices to current vertex 
			for (int j = 0; j < inGraph.numVertices(); j++)
			{
				if (j == i) continue;

				vector<int> edgePath;
				shortestPath(inGraph, i, j, edgePath);

				for (int  k = 0; k < edgePath.size(); k++)
				{
					// update edge visits
					edgeVisited[edgePath[k]]++;

					// adding to the other half edge
					(edgePath[k] % 2 == 0) ? edgeVisited[edgePath[k] + 1]++ : edgeVisited[edgePath[k] - 1]++;
				}
								
			}
		}
	}


	


	/** @}*/
	
	/** @}*/

	/** @}*/
}