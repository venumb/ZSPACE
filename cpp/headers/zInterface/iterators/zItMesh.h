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

#ifndef ZSPACE_ITERATOR_MESH_H
#define ZSPACE_ITERATOR_MESH_H

#pragma once

#include<headers/zInterface/iterators/zIt.h>
#include<headers/zInterface/objects/zObjMesh.h>

namespace zSpace
{
	
	class ZSPACE_API zItMeshEdge;	
	class ZSPACE_API zItMeshVertex;	
	class ZSPACE_API zItMeshFace;
	class ZSPACE_API zItMeshHalfEdge;


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/** \addtogroup Container
	*	\brief  The container typedef of the library.
	*  @{
	*/

	/*! \typedef zItMeshEdgeArray
	*	\brief A vector of zItMeshEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItMeshEdge> zItMeshEdgeArray;

	/*! \typedef zItMeshHalfEdgeArray
	*	\brief A vector of zItMeshHalfEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItMeshHalfEdge> zItMeshHalfEdgeArray;

	/*! \typedef  zItMeshVertexArray
	*	\brief A vector of zItMeshVertex.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItMeshVertex> zItMeshVertexArray;

	/*! \typedef  zItMeshFaceArray
	*	\brief A vector of zItMeshFace.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItMeshFace> zItMeshFaceArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/
	
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshVertex
	*	\brief The mesh vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class ZSPACE_API zItMeshVertex : public zIt
	{
	protected:

		/*!	\brief iterator of core mesh vertex object  */
		zItVertex iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshVertex();
	
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshVertex(zObjMesh &_meshObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh vertex container.
		*	\since version 0.0.3
		*/
		zItMeshVertex(zObjMesh &_meshObj, int _index);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void begin() override;

		void operator++(int) override;

		void operator--(int) override;

		bool end() override;

		void reset() override;		

		int size() override;

		void deactivate() override;

		//--------------------------
		//---- TOPOLOGY QUERY METHODS
		//--------------------------
	
		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge iterator.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(zItMeshHalfEdgeArray & halfedges);
		

		/*! \brief This method gets the indicies of halfedges connected to the iterator.
		*
		*	\param		[out]	halfedgeIndicies	- vector of halfedge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(zIntArray& halfedgeIndicies);		


		/*! \brief This method gets the edges connected to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge iterator.
		*	\since version 0.0.3
		*/
		void getConnectedEdges(zItMeshEdgeArray& edges);


		/*! \brief This method gets the indicies of edges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedEdges(zIntArray& edgeIndicies);

		/*! \brief This method gets the vertices connected to the iterator.
		*
		*	\param		[out]	verticies	- vector of verticies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(zItMeshVertexArray& verticies);
		

		/*! \brief This method gets the indicies of vertices connected to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(zIntArray& vertexIndicies);		

		/*! \brief This method gets the faces connected to the iterator.
		*
		*	\param		[out]	faces	- vector of faces.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(zItMeshFaceArray& faces);

		/*! \brief This method gets the indicies of faces connected to the iterator.
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(zIntArray& faceIndicies);

		/*!	\brief This method determines if  the element is on the boundary.
		*
		*	\return				bool	- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary();
		

		/*!	\brief This method calculate the valency of the vertex.
		*		
		*	\return				int		- valency of input vertex input valence.
		*	\since version 0.0.3
		*/
		int getValence();
		

		/*!	\brief This method determines if vertex valency is equal to the input valence number.
		*		
		*	\param		[in]	valence	- input valence value.
		*	\return				bool	- true if valency is equal to input valence.
		*	\since version 0.0.3
		*/
		bool checkValency(int valence = 1);		

		/*! \brief This method computes the principal curvatures of the vertex.
		*
		*	\return			zCurvature		- vertex curvature.
		*	\since version 0.0.3
		*/
		zCurvature getPrincipalCurvature();

		/*! \brief This method computes the voronoi area of the vertex.
		*
		*	\return			double		- vertex area.
		*	\since version 0.0.3
		*/
		double getArea();
		
	
		//--------------------------
		//---- GET METHODS
		//--------------------------
		
		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the half edge attached to the vertex.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge();	

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItVertex		- raw iterator
		*	\since version 0.0.3
		*/
		zItVertex  getRawIter();

		/*! \brief This method gets position of the vertex.
		*
		*	\return				zVector					- vertex position.
		*	\since version 0.0.3
		*/
		zVector getPosition();

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zVector* getRawPosition();

		/*! \brief This method gets normal of the vertex.
		*
		*	\return				zVector					- vertex normal.
		*	\since version 0.0.3
		*/
		zVector getNormal();

		/*! \brief This method gets pointer to the normal of the vertex.
		*
		*	\return				zVector*				- pointer to internal vertex normal.
		*	\since version 0.0.3
		*/
		zVector* getRawNormal();


		/*! \brief This method gets normalss of the vertex for all faces.
		*
		*	\param		[out]	vNormals					- input vertex index.
		*	\since version 0.0.3
		*/
		void getNormals(vector<zVector> &vNormals);
		

		/*! \brief This method gets color of the vertex.
		*
		*	\return				zColor					- vertex color.
		*	\since version 0.0.3
		*/
		zColor getColor();

		/*! \brief This method gets pointer to the color of the vertex.
		*
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.3
		*/
		zColor* getRawColor();
		
		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id);

		/*! \brief This method sets the vertex half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItMeshHalfEdge &he);

		/*! \brief This method sets position of the vertex.
		*
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.3
		*/
		void setPosition(zVector &pos);

		/*! \brief This method sets color of the vertex.
		*
		*	\param		[in]	col						- vertex color.
		*	\since version 0.0.3
		*/
		void setColor(zColor col);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------
		
		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive();		

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItMeshVertex &other);

		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshVertex &other);		

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshEdge
	*	\brief The mesh  edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItMeshEdge : public zIt
	{
	protected:

		zItEdge iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItMeshEdge();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshEdge(zObjMesh &_meshObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh edge list.
		*	\since version 0.0.3
		*/
		zItMeshEdge(zObjMesh &_meshObj, int _index);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void begin() override;

		void operator++(int) override;

		void operator--(int) override;

		bool end() override;

		void reset() override;

		int size() override;

		void deactivate() override;

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	verticies	- vector of vertex iterators.
		*	\since version 0.0.3
		*/
		void getVertices(zItMeshVertexArray &verticies);

		/*!	\brief This method gets the indicies of the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zIntArray &vertexIndicies);

		/*!	\brief TThis method gets the vertex positions attached to the iterator.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions);

		/*! \brief This method gets the faces attached to the iterator
		*
		*	\param		[out]	faces	- vector of face iterators.
		*	\since version 0.0.3
		*/
		void getFaces(zItMeshFaceArray &faces);

		/*! \brief This method gets the indicies of faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(zIntArray &faceIndicies);

		/*! \brief This method checks if the edge is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary();

		/*! \brief This method computes the centers of a the edge.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter();

		/*! \brief This method gets the vector of the edge.
		*
		*	\return				zVector					- edge vector.
		*	\since version 0.0.3
		*/
		zVector getVector();

		/*! \brief This method gets the edge length of the edge.
		*
		*	\return				double			- edge length.
		*	\since version 0.0.3
		*/
		double getLength();

		/*! \brief This method computes the dihedral angle of the edge.
		*
		*	\return			double		- dihedral angle in degrees.
		*	\since version 0.0.2
		*/
		double getDihedralAngle();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the half edge attached to the edge.
		*
		*	\param		[in]	_index				- input index ( 0 or 1).
		*	\return				zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge(int _index);

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItEdge		- raw iterator
		*	\since version 0.0.3
		*/
		zItEdge  getRawIter();

		/*! \brief This method gets color of the edge.
		*
		*	\return				zColor					- edge color.
		*	\since version 0.0.3
		*/
		zColor getColor();

		/*! \brief This method gets pointer to the color of the edge.
		*
		*	\return				zColor*					- edge color.
		*	\since version 0.0.3
		*/
		zColor* getRawColor();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id);

		/*! \brief This method sets the edge half edge to the input half edge.
		*
		*	\param		[in]	he		- input half edge iterator
		*	\param		[in]	_index	- input index ( 0 or 1).
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItMeshHalfEdge &he, int _index);

		/*! \brief This method sets color of the edge.
		*
		*	\param		[in]	col						- input color.
		*	\since version 0.0.3
		*/
		void setColor(zColor col);

		/*! \brief This method sets display weight of the edge.
		*
		*	\param		[in]	wt						- input weight.
		*	\since version 0.0.3
		*/
		void setWeight(double wt);


		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive();

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItMeshEdge &other);

		/*! \brief This operator checks for non equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshEdge &other);



	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshFace
	*	\brief The mesh  face iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItMeshFace : public zIt
	{
	protected:

		zItFace iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItMeshFace();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshFace(zObjMesh &_meshObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh face list.
		*	\since version 0.0.3
		*/
		zItMeshFace(zObjMesh &_meshObj, int _index);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void begin() override;

		void operator++(int) override;

		void operator--(int) override;

		bool end() override;

		void reset() override;

		int size() override;

		void deactivate() override;

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the half edges of the face.
		*
		*	\param		[out]	halfedges	- vector of halfedge interators.
		*	\since version 0.0.3
		*/
		void getHalfEdges(zItMeshHalfEdgeArray &halfedges);

		/*! \brief This method gets the indicies of half edges of the face.
		*
		*	\param		[out]	halfedgeIndicies	- vector of halfedge indicies.
		*	\since version 0.0.3
		*/
		void getHalfEdges(zIntArray &halfedgeIndicies);

		/*!	\brief This method gets the the vertices of the face.
		*
		*	\param		[out]	verticies	- vector of vertex iterators.
		*	\since version 0.0.3
		*/
		void getVertices(zItMeshVertexArray &verticies);

		/*!	\brief This method gets the indicies of the vertices of the face.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zIntArray &vertexIndicies);

		/*!	\brief TThis method gets the vertex positions of the face.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions);


		/*! \brief This method gets the faces connected to the iterator.
		*
		*	\param		[out]	faces	- vector of faces.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(zItMeshFaceArray& faces);

		/*! \brief This method gets the indicies of faces connected to the iterator.
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(zIntArray& faceIndicies);

		/*! \brief This method checks if the face is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary();

		/*! \brief This method computes the center of a the face.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter();

		/*! \brief This method gets the number of vertices in the face.
		*
		*	\return				int				- number of vertices in the face.
		*	\since version 0.0.3
		*/
		int getNumVertices();

		/*! \brief This method computes the input face triangulations using ear clipping algorithm.
		*
		*	\details based on  https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf & http://abitwise.blogspot.co.uk/2013/09/triangulating-concave-and-convex.html
		*	\param		[out]	numTris			- number of triangles in the input polygon.
		*	\param		[out]	tris			- index array of each triangle associated with the face.
		*	\since version 0.0.2
		*/
		void getTriangles(int &numTris, zIntArray &tris);


		/*! \brief This method computes the volume of the polyhedras formed by the face vertices and the face center of the input indexed face of the mesh.
		*
		*	\param		[in]	index			- input face index.
		*	\param		[in]	faceTris		- container of index array of each triangle associated per face. It will be computed if the container is empty.
		*	\param		[in]	fCenters		- center of associated face.  It will be computed if the point to origin.
		*	\param		[in]	absoluteVolumes	- will make all the volume value positive if true.
		*	\return				double			- volume of the polyhedras formed by the face vertices and the face center.
		*	\since version 0.0.2
		*/
		double getVolume(zIntArray &faceTris, zVector &fCenter, bool absoluteVolume = true);

		/*! \brief This method computes the area of the face . It works only for if the faces are planar.
		*
		*	\details	Based on http://geomalgorithms.com/a01-_area.html.
		*	\return				double			- area of the face.
		*	\since version 0.0.2
		*/
		double getPlanarFaceArea();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the half edge attached to the face.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge();

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItFace		- raw iterator
		*	\since version 0.0.3
		*/
		zItFace  getRawIter();

		/*! \brief This method gets the offset positions of the face.
		*
		*	\details	beased on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
		*	\param		[in]	offset				- offset distance.
		*	\param		[out]	offsetPositions		- container with the offset positions.
		*	\since version 0.0.3
		*/
		void getOffsetFacePositions(double offset, vector<zVector>& offsetPositions);

		/*! \brief This method returns the vartiable offset positions of the face.
		*
		*	\param		[in]	offsets					- offset distance from each edge of the mesh.
		*	\param		[in]	faceCenter				- center of polygon.
		*	\param		[in]	faceNormal				- normal of polygon.
		*	\param		[out]	intersectionPositions	- container with the intersection positions.
		*	\since version 0.0.3
		*/
		void getOffsetFacePositions_Variable(vector<double>& offsets, zVector& faceCenter, zVector& faceNormal, vector<zVector>& intersectionPositions);

		/*! \brief This method gets normal of the face.
		*
		*	\return				zVector					- face normal.
		*	\since version 0.0.3
		*/
		zVector getNormal();

		/*! \brief This method gets pointer to the normal of the face.
		*
		*	\return				zVector*			- pointer to internal face normal.
		*	\since version 0.0.3
		*/
		zVector* getRawNormal();

		/*! \brief This method gets color of the face.
		*
		*	\return				zColor					- face color.
		*	\since version 0.0.3
		*/
		zColor getColor();

		/*! \brief This method gets pointer to the color of the face.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal face color.
		*	\since version 0.0.3
		*/
		zColor* getRawColor();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id);

		/*! \brief This method sets the edge half edge to the input half edge.
		*
		*	\param		[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItMeshHalfEdge &he);

		/*! \brief This method sets color of the face.
		*
		*	\param		[in]	col						- input color.
		*	\since version 0.0.3
		*/
		void setColor(zColor col);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method gets if the face is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive();

		/*! \brief This method returns a boolean if the input point is in the half space.
		*
		*	\details	beased on https://www.codeproject.com/Articles/1065730/Point-Inside-Convex-Polygon-in-Cplusplus
		*	\return		bool				- true if point is in the half space.
		*	\param		[in]	pt			- point to check for.
		*	\since version 0.0.4
		*/
		bool checkPointInHalfSpace(zPoint &pt);

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two face iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItMeshFace &other);

		/*! \brief This operator checks for non equality of two face iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshFace &other);

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshHalfEdge
	*	\brief The mesh half edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItMeshHalfEdge : public zIt
	{
	protected:

		zItHalfEdge iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge(zObjMesh &_meshObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh halfedge list.
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge(zObjMesh &_meshObj, int _index);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void begin() override;

		void operator++(int) override;

		void operator--(int) override;

		bool end() override;

		void reset() override;

		int size() override;

		void deactivate() override;

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the vertex pointed by the symmetry of input iterator.
		*	
		*	\return				zItMeshVertex	- iterator to vertex.
		*	\since version 0.0.3
		*/
		zItMeshVertex getStartVertex();
	
		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zItMeshVertexArray &verticies);

		/*!	\brief This method gets the indicies of the vertices attached to halfedge.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zIntArray &vertexIndicies);

		/*!	\brief TThis method gets the vertex positions attached to the iterator and its symmetry.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions);

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(zItMeshHalfEdgeArray& edgeIndicies);

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(zIntArray& edgeIndicies);

		/*! \brief This method gets the faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(zItMeshFaceArray &faceIndicies);		

		/*! \brief This method gets the indicies of faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(zIntArray &faceIndicies);		

		/*! \brief This method checks if the half edge is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary();

		/*! \brief This method computes the centers of a the half edge.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter();

		/*! \brief This method gets the vector of the half edge.
		*
		*	\return				zVector					- edge vector.
		*	\since version 0.0.3
		*/
		zVector getVector();

		/*! \brief This method computes the edge length of the half edge.
		*	
		*	\return				double			- half edge length.
		*	\since version 0.0.3
		*/
		double getLength();


		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the symmetry half edge attached to the halfedge.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getSym();

		/*! \brief This method gets the next half edge attached to the halfedge.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getNext();

		/*! \brief This method gets the prev half edge attached to the halfedge.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getPrev();

		/*! \brief This method gets the vertex attached to the halfedge.
		*
		*	\return			zItMeshVertex		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshVertex getVertex();
		
		
		/*! \brief This method gets the face attached to the halfedge.
		*
		*	\return			zItMeshFace		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshFace getFace();
	
		/*! \brief This method gets the edge attached to the halfedge.
		*
		*	\return			zItMeshEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshEdge getEdge();
		
		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItHalfEdge		- raw iterator
		*	\since version 0.0.3
		*/
		zItHalfEdge  getRawIter();

		/*! \brief This method gets color of the halfedge.
		*
		*	\return				zColor					- halfedge color.
		*	\since version 0.0.3
		*/
		zColor getColor();

		/*! \brief This method gets pointer to the color of the halfedge.
		*
		*	\return				zColor*					- halfedge color.
		*	\since version 0.0.3
		*/
		zColor* getRawColor();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id);

		/*! \brief This method sets the symmetry half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setSym(zItMeshHalfEdge &he);

		/*! \brief This method sets the next half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setNext(zItMeshHalfEdge &he);

		/*! \brief This method sets the previous half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setPrev(zItMeshHalfEdge &he);

		/*! \brief This method sets the half edge vertex to the input vertex.
		*
		*	\param	[in]	v		- input vertex iterator
		*	\since version 0.0.3
		*/
		void setVertex(zItMeshVertex &v);

		/*! \brief This method sets the halfedge edge to the input edge.
		*
		*	\param	[in]	e		- input edge iterator
		*	\since version 0.0.3
		*/
		void setEdge(zItMeshEdge &e);

		/*! \brief This method sets the halfedge face to the input edge.
		*
		*	\param	[in]	f		- input face iterator
		*	\since version 0.0.3
		*/
		void setFace(zItMeshFace &f);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method checks if the half edge is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive();	

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItMeshHalfEdge &other);

		/*! \brief This operator checks for non equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshHalfEdge &other);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/iterators/zItMesh.cpp>
#endif

#endif