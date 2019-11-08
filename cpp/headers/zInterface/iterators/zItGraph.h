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

#ifndef ZSPACE_ITERATOR_GRAPH_H
#define ZSPACE_ITERATOR_GRAPH_H

#pragma once

#include<headers/zInterface/iterators/zIt.h>
#include<headers/zInterface/objects/zObjGraph.h>

namespace zSpace
{
	
	class ZSPACE_API zItGraphEdge;
	class ZSPACE_API zItGraphVertex;	
	class ZSPACE_API zItGraphHalfEdge;

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

	/*! \typedef zItGraphEdgeArray
	*	\brief A vector of zItGraphEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItGraphEdge> zItGraphEdgeArray;

	/*! \typedef zItGraphHalfEdgeArray
	*	\brief A vector of zItGraphHalfEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItGraphHalfEdge> zItGraphHalfEdgeArray;

	/*! \typedef  zItGraphVertexArray
	*	\brief A vector of zItGraphVertex.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItGraphVertex> zItGraphVertexArray;

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

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphVertex
	*	\brief The graph vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItGraphVertex : public zIt
	{
	protected:

		zItVertex iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphVertex();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphVertex(zObjGraph &_graphObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	_index				- input index in graph vertex list.
		*	\since version 0.0.3
		*/
		zItGraphVertex(zObjGraph &_graphObj, int _index);

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
		void getConnectedHalfEdges(zItGraphHalfEdgeArray& halfedges);


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
		void getConnectedEdges(zItGraphEdgeArray& edges);


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
		void getConnectedVertices(zItGraphVertexArray& verticies);


		/*! \brief This method gets the indicies of vertices connected to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(zIntArray& vertexIndicies);
			   		 
		/*!	\brief This method calculate the valency of the vertex.
		*
		*	\return				int		- valency of input vertex input valence.
		*	\since version 0.0.3
		*/
		int getValence();

		/*! \brief This method returns a container holding the BSF if the input ItGraph.
		*
		*	\param		[in]	index	- index to start from.
		*	\param		[out]	bsf		- container holding the BSF zItGraphVertex.
		*	\since version 0.0.4
		*/
		void getBSF(zItGraphVertexArray& bsf);

		/*! \brief This method returns a container holding the BSF if the input ItGraph.
		*
		*	\param		[in]	index	- index to start from.
		*	\param		[out]	bsf		- container holding the BSF IDs.
		*	\since version 0.0.4
		*/
		void getBSF(zIntArray& bsf);

		/*!	\brief This method determines if vertex valency is equal to the input valence number.
		*
		*	\param		[in]	valence	- input valence value.
		*	\return				bool	- true if valency is equal to input valence.
		*	\since version 0.0.3
		*/
		bool checkValency(int valence = 1);

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
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getHalfEdge();

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItVertex		- raw iterator
		*	\since version 0.0.3
		*/
		zItVertex  getRawIter();

		/*! \brief This method gets position of the vertex.
		*
		*	\return				zPoint					- vertex position.
		*	\since version 0.0.3
		*/
		zPoint getPosition();

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\return				zPoint*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zPoint* getRawPosition();
		
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
		void setHalfEdge(zItGraphHalfEdge &he);

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
		bool operator==(zItGraphVertex &other);


		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphVertex &other);

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphEdge
	*	\brief The graph  edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItGraphEdge : public zIt
	{
	protected:

		zItEdge iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphEdge();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphEdge(zObjGraph &_graphObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	_index				- input index in graph edge list.
		*	\since version 0.0.3
		*/
		zItGraphEdge(zObjGraph &_graphObj, int _index);

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
		void getVertices(zItGraphVertexArray &verticies);

		/*!	\brief This method gets the indicies of the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zIntArray &vertexIndicies);

		/*!	\brief This method gets the vertex positions attached to the iterator.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions);

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
		*	\return				zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getHalfEdge(int _index);

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
		void setHalfEdge(zItGraphHalfEdge &he, int _index);

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
		bool operator==(zItGraphEdge &other);

		/*! \brief This operator checks for non equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphEdge &other);

	};


	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphHalfEdge
	*	\brief The graph half edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItGraphHalfEdge : public zIt
	{
	protected:

		zItHalfEdge iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge(zObjGraph &_graphObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	_index				- input index in graph halfedge list.
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge(zObjGraph &_graphObj, int _index);

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
		*	\return				zItGraphVertex	- iterator to vertex.
		*	\since version 0.0.3
		*/
		zItGraphVertex getStartVertex();

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(zItGraphVertexArray &verticies);

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
		void getConnectedHalfEdges(zItGraphHalfEdgeArray& edgeIndicies);

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(zIntArray& edgeIndicies);		

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
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getSym();

		/*! \brief This method gets the next half edge attached to the halfedge.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getNext();

		/*! \brief This method gets the prev half edge attached to the halfedge.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getPrev();

		/*! \brief This method gets the vertex attached to the halfedge.
		*
		*	\return			zItGraphVertex		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphVertex getVertex();

		/*! \brief This method gets the edge attached to the halfedge.
		*
		*	\return			zItGraphEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphEdge getEdge();

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
		void setSym(zItGraphHalfEdge &he);

		/*! \brief This method sets the next half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setNext(zItGraphHalfEdge &he);

		/*! \brief This method sets the previous half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setPrev(zItGraphHalfEdge &he);

		/*! \brief This method sets the half edge vertex to the input vertex.
		*
		*	\param	[in]	v		- input vertex iterator
		*	\since version 0.0.3
		*/
		void setVertex(zItGraphVertex &v);

		/*! \brief This method sets the halfedge edge to the input edge.
		*
		*	\param	[in]	e		- input edge iterator
		*	\since version 0.0.3
		*/
		void setEdge(zItGraphEdge &e);

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
		bool operator==(zItGraphHalfEdge &other);

		/*! \brief This operator checks for non equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphHalfEdge &other);
	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/iterators/zItGraph.cpp>
#endif

#endif