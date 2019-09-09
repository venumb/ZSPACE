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

#ifndef ZSPACE_TS_PATHNETWORK_SHORTESTPATH_H
#define ZSPACE_TS_PATHNETWORK_SHORTESTPATH_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

namespace zSpace
{

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

	template<typename T, typename U>
	class ZSPACE_TOOLS zTsShortestPath
	{
	protected:
		/*!	\brief pointer to half edge Object  */		
		T *heObj;
		
		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;


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
		zTsShortestPath();

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
		~zTsShortestPath();

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

		/*! \brief This method computes the shortest path from the source vertex to destination vertex of the zGraph/zMesh and returns a graph of the path.
		*
		*	\param		[in]	indexA					- source vertex index.
		*	\param		[in]	indexB					- destination vertex index.
		*	\param		[out]	outGraph				- output graph of the shortest path.
		*	\since version 0.0.3
		*/
		void getShortestPathGraph(int indexA, int indexB, zObjGraph &outGraph);

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
		int minDistance(vector<float> &dist, vector<bool> &sptSet);

	};


	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/*! \typedef zTsMeshShortestPath
	*	\brief A shortest path object for meshes.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	typedef zTsShortestPath<zObjMesh, zFnMesh> zTsMeshShortestPath;

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/*! \typedef zTsGraphShortestPath
	*	\brief A shortest path object for graphs.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	typedef zTsShortestPath<zObjGraph, zFnGraph> zTsGraphShortestPath;
	

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/pathNetworks/zTsShortestPath.cpp>
#endif

#endif