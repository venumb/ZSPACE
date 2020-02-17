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

#ifndef ZSPACE_FN_GRAPH_H
#define ZSPACE_FN_GRAPH_H

#pragma once


#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/functionsets/zFnMesh.h>

#include<headers/zInterface/iterators/zItGraph.h>

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnGraph
	*	\brief A graph function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zFnGraph : protected zFn
	{	

	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;	

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
		/*!	\brief boolean true is its a planar graph  */
		bool planarGraph;

		/*!	\brief stores normal of the the graph if planar  */
		zVector graphNormal;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnGraph();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.2
		*/
		zFnGraph(zObjGraph &_graphObj, bool  _planarGraph = false, zVector _graphNormal = zVector(0, 0, 1));


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnGraph();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		zFnType getType() override;

		void from(string path, zFileTpye type, bool staticGeom = false) override;

		void to(string path, zFileTpye type) override;

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		void clear() override;

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a graph from the input containers.
		*
		*	\param		[in]	_positions		- container of type zPoint containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void create(zPointArray(&_positions), zIntArray(&edgeConnects), bool staticGraph = false);

		/*! \brief his method creates a graphfrom the input containers.
		*
		*	\param		[in]	_positions		- container of type zPoint containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	graphNormal		- normal of the plane of the graph.
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void create(zPointArray(&_positions), zIntArray(&edgeConnects), zVector &graphNormal, bool staticGraph = false);

		/*! \brief This method creates a graph from a mesh.
		*
		*	\param		[in]	graphObj			- input mesh object.	
		*	\param		[in]	staticGraph			- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\param		[in]	excludeBoundary		- excludes the boundary edge of the input mesh.
		*	\since version 0.0.4
		*/
		void createFromMesh(zObjMesh &meshObj, bool excludeBoundary = false, bool staticGraph = false);

		/*! \brief This method adds a vertex to the graph.
		*
		*	\param		[in]	_pos				- zPoint holding the position information of the vertex.
		*	\param		[in]	checkDuplicates		- checks for duplicates if true.
		*	\param		[out]	vertex				- vertex iterator of the new vertex or existing if it is a duplicate.
		*	\return				bool				- true if the vertex container is resized.
		*	\note	 The vertex pointers will need to be computed/ set.
		*	\since version 0.0.2
		*/
		bool addVertex(zPoint &_pos, bool checkDuplicates, zItGraphVertex &vertex);

		/*! \brief This method adds an edge and its symmetry edge to the graph.
		*
		*	\param		[in]	v1			- start vertex index of the edge.
		*	\param		[in]	v2			- end vertex index of the edge.
		*	\param		[out]	halfEdge	- hafedge iterator of the new halfedge or existing if it is a duplicate.
		*	\return				bool		- true if the edges container is resized.
		*	\note	 The half edge pointers will need to be computed/ set.
		*	\since version 0.0.2
		*/
		bool addEdges(int &v1, int &v2, bool checkDuplicates, zItGraphHalfEdge &halfEdge);

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method returns the number of vertices in the graph.
		*	\return				number of vertices.
		*	\since version 0.0.2
		*/
		int numVertices();

		/*! \brief This method returns the number of edges in the graph.
		*	\return				number of edges.
		*	\since version 0.0.2
		*/
		int numEdges();

		/*! \brief This method returns the number of half edges in the graph .
		*	\return				number of half edges.
		*	\since version 0.0.2
		*/
		int numHalfEdges();

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos					- position to be checked.
		*	\param		[out]		outVertexId			- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor		- input precision factor.
		*	\return		[out]		bool				- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zPoint pos, zItGraphVertex &outVertex, int precisionfactor = 6);

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1					-  vertexId 1.
		*	\param		[in]	v2					-  vertexId 2.
		*	\param		[out]	outHalfEdgeId		-  half edge id.
		*	\return		[out]	bool			-  true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool halfEdgeExists(int v1, int v2, int &outHalfEdgeId);

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			-  vertexId 1.
		*	\param		[in]	v2			-  vertexId 2.
		*	\param		[out]	outEdgeId	-  half edge iterator if edge exists.
		*	\return		[out]	bool		-  true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool halfEdgeExists(int v1, int v2, zItGraphHalfEdge &outHalfEdge);
	

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		/*! \brief This method computes the Edge colors based on the vertex colors.
		*
		*	\since version 0.0.2
		*/
		void computeEdgeColorfromVertexColor();

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.2
		*/
		void computeVertexColorfromEdgeColor();

		/*! \brief This method averages the positions of vertex except for the ones on the boundary.
		*
		*	\param		[in]	numSteps	- number of times the averaging is carried out.
		*	\since version 0.0.2
		*/
		void averageVertices(int numSteps = 1);

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData .
		*	\since version 0.0.2
		*/
		void removeInactiveElements(zHEData type);
		
		/*! \brief This method makes the graph a fixed. Computes the static edge vertex positions if true.
		*
		*	\since version 0.0.2
		*/
		void makeStatic();

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets vertex positions of all the vertices.
		*
		*	\param		[in]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void setVertexPositions(zPointArray& pos);


		/*! \brief This method sets vertex color of all the vertices to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColor(zColor col, bool setEdgeColor = false);

		/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the graph.
		*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColors(zColorArray& col, bool setEdgeColor = false);
	
		/*! \brief This method sets edge color of all the edges to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColor(zColor col, bool setVertexColor = false);

		/*! \brief This method sets edge color of all the edges with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of half edges in the graph.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColors(zColorArray& col, bool setVertexColor);
		
		/*! \brief This method sets edge weight of all the edges to the input weight.
		*
		*	\param		[in]	wt				- input weight.
		*	\since version 0.0.2
		*/
		void setEdgeWeight(double wt);

		/*! \brief This method sets edge weights of all the edges with the input weight contatiner.
		*
		*	\param		[in]	wt				- input weight  contatiner. The size of the contatiner should be equal to number of half edges in the mesh.
		*	\since version 0.0.2
		*/
		void setEdgeWeights(zDoubleArray& wt);
		
		//--------------------------
		//--- GET METHODS 
		//--------------------------

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexPositions(zPointArray& pos);

		/*! \brief This method gets pointer to the internal vertex positions container.
		*
		*	\return				zPoint*					- pointer to internal vertex position container.
		*	\since version 0.0.2
		*/
		zPoint* getRawVertexPositions();

		/*! \brief This method gets vertex color of all the vertices.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexColors(zColorArray& col);

		/*! \brief This method gets pointer to the internal vertex color container.
		*
		*	\return				zColor*					- pointer to internal vertex color container.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColors();
	
		/*! \brief This method gets edge color of all the edges.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getEdgeColors(zColorArray& col);

		/*! \brief This method gets pointer to the internal edge color container.
		*
		*	\return				zColor*					- pointer to internal edge color container.
		*	\since version 0.0.2
		*/
		zColor* getRawEdgeColors();

		/*! \brief This method computes the center the graph.
		*
		*	\return		zPoint					- center .
		*	\since version 0.0.2
		*/
		zPoint getCenter();

		/*! \brief This method computes the centers of a all edges of the graph.
		*
		*	\param		[in]	type					- zEdgeData or zHalfEdgeData.
		*	\param		[out]	centers					- vector of centers of type zPoint.
		*	\since version 0.0.2
		*/
		void getCenters(zHEData type, zPointArray &centers);

		/*! \brief This method computes the lengths of all the half edges of a the graph.
		*
		*	\param		[out]	halfEdgeLengths				- vector of halfedge lengths.
		*	\return				double						- total edge lengths.
		*	\since version 0.0.2
		*/
		double getHalfEdgeLengths(zDoubleArray &halfEdgeLengths);

		/*! \brief This method computes the lengths of all the  edges of a the graph.
		*
		*	\param		[out]	EdgeLengths		- vector of edge lengths.
		*	\return				double				- total edge lengths.
		*	\since version 0.0.2
		*/
		double getEdgeLengths(zDoubleArray &edgeLengths);

		/*! \brief This method stores graph edge connectivity information in the input containers
		*
		*	\param		[out]	edgeConnects	- stores list of esdge connection with vertex ids for each edge.
		*	\since version 0.0.2
		*/
		void getEdgeData(zIntArray &edgeConnects);

		/*! \brief This method creates a duplicate of the input graph.
		*
		*	\param		[in]	planarGraph		- true if input graph is planar.
		*	\param		[in]	graphNormal		- graph normal if the input graph is planar.
		*	\return				zGraph			- duplicate graph.
		*	\since version 0.0.2
		*/
		zObjGraph getDuplicate(bool planarGraph = false, zVector graphNormal = zVector(0, 0, 1));

		/*! \brief This method returns the mesh created from planar input graph.
		*
		*	\param		[out]	zObjMesh				- output mesh object
		*	\param		[in]	inGraph					- input graph.
		*	\param		[in]	width					- offset width from the graph.
		*	\param		[in]	graphNormal				- normal of the plane of the graph.		
		*	\since version 0.0.2
		*/
		void getGraphMesh(zObjMesh &out, double width = 0.5, zVector graphNormal = zVector(0, 0, 1));

		/*! \brief This method computes the graph eccentricity.
		*
		*	\details based on https://codeforces.com/blog/entry/17974
		*	\param		[out]	zItGraphVertexArray		- array of eccentric graph centers
		*	\since version 0.0.4
		*/
		void getGraphEccentricityCenter(zItGraphVertexArray &outV);

		//--------------------------
		//---- TOPOLOGY MODIFIER METHODS
		//--------------------------
		
		/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
		*
		*	\param		[in]	edge			- iterator of the edge to be split.
		*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
		*	\return				zItGraphVertex	- iterator of the new vertex added after splitinng the edge.
		*	\since version 0.0.2
		*/
		zItGraphVertex splitEdge(zItGraphEdge &edge, double edgeFactor = 0.5);
		
		//--------------------------
		//---- TRANSFORM METHODS OVERRIDES
		//--------------------------

		void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) override;

		void setScale(zFloat4 &scale) override;

		void setRotation(zFloat4 &rotation, bool appendRotations = false) override;

		void setTranslation(zVector &translation, bool appendTranslations = false) override;

		void setPivot(zVector &pivot) override;

		void getTransform(zTransform &transform) override;

	protected:

		//--------------------------
		//---- PROTECTED OVERRIDE METHODS
		//--------------------------	
		void transformObject(zTransform &transform) override;

		//--------------------------
		//---- PROTECTED REMOVE INACTIVE
		//--------------------------

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData .
		*	\since version 0.0.2
		*/
		void removeInactive(zHEData type);

		//--------------------------
		//---- PROTECTED FACTORY METHODS
		//--------------------------

		/*! \brief This method imports zGraph from a TXT file.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\return 		bool			- true if the file was read succesfully.
		*	\since version 0.0.2
		*/
		bool fromTXT(string infilename);

		/*! \brief This method imports zGraph from a TXT file from Maya.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\return 		bool				- true if the file was read succesfully.
		*	\since version 0.0.2
		*/
		bool fromMAYATXT(string infilename);

		/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		inGraph				- graph created from the JSON file.
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\return 		bool			- true if the file was read succesfully.
		*	\since version 0.0.2
		*/
		bool fromJSON(string infilename);


		/*! \brief This method exports zGraph to a TXT file format.
		*
		*	\param [in]		inGraph				- input graph.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toTXT(string outfilename);

		/*! \brief This method exports zGraph to a JSON file format using JSON Modern Library.
		*
		*	\param [in]		inGraph				- input graph.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\param [in]		vColors				- export vertex color information if true.
		*	\since version 0.0.2
		*/
		void toJSON(string outfilename);

	private:

		//--------------------------
		//---- PRIVATE METHODS
		//--------------------------

		/*! \brief This method sets the edge vertex position conatiners for static meshes.
		*
		*	\since version 0.0.2
		*/
		void setStaticContainers();

		//--------------------------
		//---- DEACTIVATE AND REMOVE METHODS
		//--------------------------

		/*! \brief This method adds input half edge and corresponding symmety edge from the halfEdgesMap.
		*
		*	\param		[in]	he			- half edge iterator.
		*	\since version 0.0.2
		*/
		void addToHalfEdgesMap(zItGraphHalfEdge &he);

		/*! \brief This method removes input half edge and corresponding symmety edge from the halfEdgesMap.
		*
		*	\param		[in]	he			- half edge iterator.
		*	\since version 0.0.2
		*/
		void removeFromHalfEdgesMap(zItGraphHalfEdge &he);
	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnGraph.cpp>
#endif

#endif