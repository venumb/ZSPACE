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

#ifndef ZSPACE_HE_MESH_H
#define ZSPACE_HE_MESH_H

#pragma once

#include <headers/zCore/geometry/zGraph.h>
#include <set>

namespace zSpace
{

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zMesh
	*	\brief A half edge mesh class.
	*	\since version 0.0.1
	*/
	

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zMesh : public zGraph
	{

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
		
		/*!	\brief stores number of active faces  */
		int n_f;

		/*! \brief face container			*/
		zFaceArray faces;

		/*!	\brief container which stores vertex normals.	*/
		zVectorArray vertexNormals;

		/*!	\brief container which stores face normals. 	*/
		zVectorArray faceNormals;

		/*!	\brief container which stores face colors. 	*/
		zColorArray faceColors;

		/*!	\brief storesface handles. Used for container resizing only  */
		vector<zFaceHandle> fHandles;

		/*!	\brief stores the start face ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_FaceId;

		/*! \brief container of face vertices . Used for display if it is a static geometry */
		vector<zIntArray> faceVertices;
		

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zMesh();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zMesh();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------
		/*! \brief This method creates a mesh from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	polyCounts		- container of type integer with number of vertices per polygon.
		*	\param		[in]	polyConnects	- polygon connection list with vertex ids for each face.
		*	\since version 0.0.1
		*/
		void create(zPointArray(&_positions), zIntArray(&polyCounts), zIntArray(&polyConnects));

		/*! \brief This methods clears all the mesh containers.
		*
		*	\since version 0.0.2
		*/
		void clear();

		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zVector &pos);

		//--------------------------
		//---- EDGE METHODS
		//--------------------------

		/*! \brief This method computes number of edges from the polygon related containers.
		*
		*	\param		[in]	polyCounts		- container of type integer with number of vertices per polygon.
		*	\param		[in]	polyConnects	- polygon connection list with vertex ids for each face.
		*	\since version 0.0.1
		*/
		int computeNumEdges(zIntArray(&polyCounts), zIntArray(&polyConnects));

		/*! \brief This method adds an edge and its symmetry edge to the edges array.
		*
		*	\param		[in]	v1			- start zVertex of the edge.
		*	\param		[in]	v2			- end zVertex of the edge.
		*	\return				bool		- true if the edges container is resized.
		*	\since version 0.0.1
		*/
		bool addEdges(int &v1, int &v2);

		/*! \brief This method sets the static edge vertices if the graph is static.
		*	\param		[in]		_edgeVertices	- input container of edge Vertices.
		*	\since version 0.0.2
		*/
		void setStaticEdgeVertices(vector<zIntArray> &_edgeVertices);

		//--------------------------
		//---- FACE METHODS
		//--------------------------


		/*! \brief This method adds a face with null edge pointer to the faces array.
		*	
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addPolygon();

		/*! \brief This method adds a face to the faces array and updates the pointers of vertices, edges and polygons of the mesh based on face vertices.
		*
		*	\param		[in]	fVertices	- array of ordered vertices that make up the polygon.	
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addPolygon(zIntArray &fVertices);

		/*! \brief This method sets the number of faces in zMesh  the input value.
		*	\param		[in]	_n_f	-	number of faces.
		*	\param		[in]	setMax	- if true, sets max edges as amultiple of _n_e.
		*	\since version 0.0.1
		*/		
		void setNumPolygons(int _n_f, bool setMax = true);
		
		/*! \brief This method gets the edges of a zFace.
		*
		*	\param		[in]	index			- index in the face container.
		*	\param		[in]	type			- zFaceData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.1
		*/
		void getFaceEdges(int index, zIntArray &edgeIndicies);	

		/*!	\brief This method gets the vertices attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge/face container.
		*	\param		[in]	type			- zEdgeData or zFaceData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.1
		*/
		void getFaceVertices(int index, zIntArray &vertexIndicies);

		/*! \brief This method sets the static face verticies if the mesh is static.
		*	\param		[in]		_faceVertices	- input container of face vertices.
		*	\since version 0.0.2
		*/
		void setStaticFaceVertices(vector<zIntArray> &_faceVertices);
	
		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method assigns a unique index per mesh element.
		*
		*	\param		[in]	type				- zVertexData or zEdgeData or zHalfEdgeData or zFaceData.
		*	\since version 0.0.1
		*/
		void indexElements(zHEData type);
		
		/*! \brief This method updates the pointers for boundary Edges.
		*
		*	\param		[in]	numEdges		- number of edges in the mesh.
		*	\since version 0.0.1
		*/
		void update_BoundaryEdgePointers();

		/*! \brief This method resizes the array connected with the input type to the specified newSize.
		*
		*	\param		[in]	type			- zVertexData or zHalfEdgeData or zEdgeData or zFaceData.
		*	\param		[in]	newSize			- new size of the array.		
		*	\since version 0.0.1
		*/
		void resizeArray(zHEData type, int newSize);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/geometry/zMesh.cpp>
#endif

#endif