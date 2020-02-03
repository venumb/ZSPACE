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


#ifndef ZSPACE_HE_GRAPH_H
#define ZSPACE_HE_GRAPH_H

#pragma once

#include <headers/zCore/base/zVector.h>
#include <headers/zCore/base/zMatrix.h>
#include <headers/zCore/base/zColor.h>
#include <headers/zCore/base/zTypeDef.h>

#include <headers/zCore/utilities/zUtilsCore.h>

#include <headers/zCore/geometry/zHEGeomTypes.h>

namespace zSpace
{

	/*! \struct connectedEdgesPerVerts
	*	\brief A  struct defined to  hold temporary connected edges of a vertex of a graph.
	*	\since version 0.0.1
	*/
	struct connectedEdgesPerVerts { int vertId; vector<int> temp_connectedEdges; };

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zGraph
	*	\brief A half edge graph class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zGraph
	{
		
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of active vertices */
		int n_v;

		/*!	\brief stores number of active  edges */
		int n_e;

		/*!	\brief stores number of active half edges */
		int n_he;

		/*!	\brief vertex container */
		zVertexArray vertices;

		/*!	\brief edge container	*/
		zHalfEdgeArray halfEdges;

		/*!	\brief edge container	*/
		zEdgeArray edges;

		/*!	\brief container which stores vertex positions.			*/
		zPointArray vertexPositions;

		/*!	\brief vertices to edgeId map. Used to check if edge exists with the haskey being the vertex sequence.	 */
		unordered_map <string, int> existingHalfEdges;

		/*!	\brief position to vertexId map. Used to check if vertex exists with the haskey being the vertex position.	 */
		unordered_map <string, int> positionVertex;	

		/*!	\brief container which stores vertex colors.	*/
		zColorArray vertexColors;

		/*!	\brief container which stores edge colors.	*/
		zColorArray edgeColors;

		/*!	\brief container which stores vertex weights.	*/
		zDoubleArray vertexWeights;

		/*!	\brief container which stores edge weights.	*/
		zDoubleArray edgeWeights;

		/*!	\brief stores vertex handles. Used for container resizing only  */
		vector<zVertexHandle> vHandles;

		/*!	\brief stores edge handles. Used for container resizing only  */
		vector<zEdgeHandle> eHandles;

		/*!	\brief stores half edge handles. Used for container resizing only  */
		vector<zHalfEdgeHandle> heHandles;


		/*!	\brief stores the start vertex ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_VertexId;	

		/*!	\brief stores the start edge ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_EdgeId;		

		/*!	\brief stores the start vertex color ID in the VBO, when attache to the zBufferObject.	*/
		int VBO_VertexColorId;		
		
		/*!	\brief boolean indicating if the geometry is static or not. Default its false.	*/
		bool staticGeometry = false;

		/*! \brief container of edge vertices . Used for display if it is a static geometry */
		vector<zIntArray> edgeVertices;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/		
		zGraph();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/		
		~zGraph();
			

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the graph from the input contatiners.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	staticGraph		- boolean indicating if the geometry is static or not. Default its false.
		*	\since version 0.0.1
		*/
		void create(zPointArray(&_positions), zIntArray(&edgeConnects), bool staticGraph = false);

		/*! \brief This methods creates the graph from the input contatiners for planar graphs.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	graphNormal		- normal of the plane of the graph.
		*	\param		[in]	sortReference	- reference vector for sorting edges.
		*	\since version 0.0.1
		*/
		void create(zPointArray(&_positions), zIntArray(&edgeConnects), zVector &graphNormal, zVector &sortReference);

		/*! \brief This methods clears all the graph containers.
		*
		*	\since version 0.0.2
		*/
		void clear();

		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zPoint holding the position information of the vertex.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zPoint &pos);

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\return					bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zPoint pos, int &outVertexId, int precisionfactor = 6);

		/*! \brief This method sets the number of vertices in zGraph  the input value.
		*	\param		[in]		_n_v	- number of vertices.
		*	\param		[in]		setMax	- if true, sets max vertices as amultiple of _n_v.
		*	\since version 0.0.1
		*/
		void setNumVertices(int _n_v, bool setMax = true);

		//--------------------------
		//---- MAP METHODS
		//--------------------------

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/		
		void addToPositionMap(zPoint &pos, int index, int precisionfactor = 6);

		/*! \brief This method removes the position given by input vector from the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/
		void removeFromPositionMap(zPoint &pos, int precisionfactor = 6);

		/*! \brief This method adds both the half-edges given by input vertex indices to the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\param		[in]		index	- input edge index in the edge container.
		*	\since version 0.0.1
		*/
		void addToHalfEdgesMap(int v1, int v2, int index);

		/*! \brief This method removes both the half-edges given given by vertex input indices from the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\since version 0.0.1
		*/
		void removeFromHalfEdgesMap(int v1, int v2);

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outEdgeId	- stores edgeId if the edge exists else it is -1.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool halfEdgeExists(int v1, int v2, int &outEdgeId);


		//--------------------------
		//---- EDGE METHODS
		//--------------------------

		/*! \brief This method adds an edge and its symmetry edge to the edges array.
		*
		*	\param		[in]	v1			- start zVertex of the edge.
		*	\param		[in]	v2			- end zVertex of the edge.
		*	\return				bool		- true if the edges container is resized.
		*	\since version 0.0.1
		*/
		bool addEdges(int &v1, int &v2);		

		/*! \brief This method sets the number of edges to  the input value.
		*	\param		[in]		_n_e	- number of edges.
		*	\param		[in]		setMax	- if true, sets max edges as amultiple of _n_he.
		*	\since version 0.0.1
		*/	
		void setNumEdges(int _n_e, bool setMax = true);


		/*! \brief This method sets the static edge vertices if the graph is static.
		*	\param		[in]		_edgeVertices	- input container of edge Vertices.
		*	\since version 0.0.2
		*/
		void setStaticEdgeVertices(vector<zIntArray> &_edgeVertices);
		

		/*! \brief This method sorts edges cyclically around a given vertex using a bestfit plane.
		*
		*	\param		[in]	unsortedEdges		- vector of type zEdge holding edges to be sorted
		*	\param		[in]	center				- zVertex to which all the above edges are connected.
		*	\param		[in]	sortReferenceId		- local index in the unsorted edge which is the start reference for sorting.
		*	\param		[out]	sortedEdges			- vector of zVertex holding the sorted edges.
		*	\since version 0.0.1
		*/
		void cyclic_sortEdges(zIntArray &unSortedEdges, zVector &center, int sortReferenceIndex, zIntArray &sortedEdges);
		
		/*! \brief This method sorts edges cyclically around a given vertex using a bestfit plane.
		*
		*	\param		[in]	unsortedEdges		- vector of type zEdge holding edges to be sorted
		*	\param		[in]	center				- zVertex to which all the above edges are connected.
		*	\param		[in]	referenceDir		- reference vector to sort.
		*	\param		[in]	norm				- reference normal to sort.
		*	\param		[out]	sortedEdges			- vector of zVertex holding the sorted edges.
		*	\since version 0.0.1
		*/
		void cyclic_sortEdges(zIntArray &unSortedEdges, zVector &center, zVector& referenceDir, zVector& norm, zIntArray &sortedEdges);
		
		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method assigns a unique index per graph element.
		*
		*	\param		[in]	type				- zVertexData or zEdgeData or zHalfEdgeData.
		*	\since version 0.0.1
		*/
		void indexElements(zHEData type);

		/*! \brief This method resizes the array connected with the input type to the specified newSize.
		*
		*	\param		[in]	type			- zVertexData or zHalfEdgeData or zEdgeData.
		*	\param		[in]	newSize			- new size of the array.
		*	\since version 0.0.1
		*/
		void resizeArray(zHEData type, int newSize);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/geometry/zGraph.cpp>
#endif

#endif