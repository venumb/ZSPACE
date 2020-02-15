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


#ifndef ZSPACE_HE_GEOMTYPES_H
#define ZSPACE_HE_GEOMTYPES_H

#pragma once

#include <vector>
using namespace std;

#include<headers/zCore/base/zInline.h>

namespace zSpace
{	
	
	class ZSPACE_CORE zEdge;
	class ZSPACE_CORE zHalfEdge;
	class ZSPACE_CORE zVertex;
	class ZSPACE_CORE zFace;

	typedef std::vector<zVertex>::iterator			zItVertex;
	typedef std::vector<zHalfEdge>::iterator		zItHalfEdge;
	typedef std::vector<zEdge>::iterator			zItEdge;
	typedef std::vector<zFace>::iterator			zItFace;
	
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zHEHandles
	*	\brief The half edge geometry handle classes of the library.
	*  @{
	*/

	/*! \class zEdge
	*	\brief An edge class to hold edge information of a half-edge data structure.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zEdge
	{
	protected:
		
		/*!	\brief stores of the edge. */
		int index;

		/*!	\brief pointer to 2 half edges of the edge. */
		zHalfEdge* he[2];

	public:		
	
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zEdge();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/

		ZSPACE_CUDA_CALLABLE ~zEdge();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the edgeId of current edge.
		*
		*	\return				int -  edge index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getId();

		/*! \brief This method sets the edgeId of current edge to the input value.
		*
		*	\param		[in]	edgeId - input edge index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setId(int _edgeId);

		/*! \brief This method gets the half edge pointer of current edge at the input index.
		*
		*	\param		[in]	_index		- input index - 0 or 1.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getHalfEdge(int _index);

		/*! \brief This method sets the half edge pointer of current edge at the input index.
		*
		*	\param		[in]	_he			- input half edge pointer.
		*	\param		[in]	_index		- input index - 0 or 1.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setHalfEdge(zHalfEdge* _he, int _index);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method checks if the current element is active.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool isActive() const;
		
		/*! \brief This method resets the pointers of the current edge to null.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void reset();

	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zHEHandles
	*	\brief The half edge geometry handle classes of the library.
	*  @{
	*/

	/*! \class zHalfEdge
	*	\brief An half edge class to hold half edge information of a half-edge data structure.
	*	\since version 0.0.2
	*/

	/** @}*/ 

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zHalfEdge
	{
	protected:
		
		/*!	\brief iterator	in vertex list. */
		zVertex* v;

		/*!	\brief iterator in face list. */
		zFace* f;

		/*!	\brief iterator in edge list. */
		zEdge* e;

		/*!	\brief iterator to previous  halfedge */
		zHalfEdge* prev;

		/*!	\brief iterator to next halfedge */
		zHalfEdge* next;

		/*!	\brief iterator to symmerty/twin half edge.			*/
		zHalfEdge* sym;

		/*!	\brief index of half edge.			*/
		int index;


	public:	

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zHalfEdge();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the index of current half edge.
		*
		*	\return				int - half edge index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getId();

		/*! \brief This method sets the index of current half edge to the the input value.
		*
		*	\param		[in]	_ edgeId - input half edge index
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setId(int _edgeId);

		/*! \brief This method returns the symmetry half edge of current half edge.
		*
		*	\return				zHalfEdge - symmetry halfedge pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getSym();

		/*! \brief This method sets the symmetry edge of current zEdge to the the input edge
		*
		*	\param		[in]	_sym - symmetry half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setSym(zHalfEdge* _sym);

		/*! \brief This method returns the previous half edge of current zEdge.
		*
		*	\return				zHalfEdge - previous halfedge pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getPrev();

		/*! \brief This method sets the previous half edge of current half edge to the the input half edge.
		*
		*	\param		[in]	_prev		- previous halfedge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setPrev(zHalfEdge* _prev);

		/*! \brief This method returns the next edge of current halfedge.
		*
		*	\return				zHalfEdge - next halfedge pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getNext();

		/*! \brief This method sets the next edge of current half edge to the the input half edge
		*
		*	\param		[in]	_next		- next half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setNext(zHalfEdge* _next);

		/*! \brief This method returns the vertex pointed to by the current zEdge.
		*
		*	\return				zVertex - vertex pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVertex* getVertex();

		/*! \brief This method sets the vertex pointed to by the current half edge to the the input vertex.
		*
		*	\param		[in]	_v - input vertex.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setVertex(zVertex* _v);

		/*! \brief This method returns the face pointed to by the current half edge.
		*
		*	\return				zEdge - face pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zFace* getFace();

		/*! \brief This method sets the face pointed to by the current half edge to the the input face.
		*
		*	\param		[in]	 _f - input face.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setFace(zFace* _f);

		/*! \brief This method gets the edge pointed to by the current half edge.
		*
		*	\return				zEdge - edge pointed by the half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zEdge* getEdge();

		/*! \brief This method sets the edge pointed to by the current halfedge to the the input edge.
		*
		*	\param		[in]	 _e - input edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setEdge(zEdge* _e);

		/*! \brief This method resets the pointers of the half edge to null.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool isActive() const;

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zHEHandles
	*	\brief The half edge geometry handle classes of the library.
	*  @{
	*/

	/*! \class zVertex
	*	\brief A vertex struct to  hold vertex information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zVertex
	{
	protected:
		
		/*!	\brief index of vertex.			*/
		int index;

		/*!	\brief pointer to zHalfEdge starting at the current zVertex.		*/
		zHalfEdge* he;

	public:				

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zVertex();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zVertex();
		
		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method gets the index of current vertex.
		*
		*	\return				int - vertex index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getId();

		/*! \brief This method sets the index of current vertex to the the input value.
		*
		*	\param		[in]	_vertexId - input index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setId(int _vertexId);

		/*! \brief This method gets the half edge pointed by the current Vertex.
		*
		*	\return				zHalfEdge - half edge pointed to by the vertex.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getHalfEdge();

		/*! \brief This method sets the half edge of the current vertex to the the input half edge
		*
		*	\param		[in]	_he - input half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setHalfEdge(zHalfEdge* _he);

		/*! \brief This method makes the pointers of the current zVertex to null.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool isActive() const;

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zHEHandles
	*	\brief The half edge geometry handle classes of the library.
	*  @{
	*/

	/*! \class zFace
	*	\brief A face struct to  hold polygonal information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zFace
	{
	protected:
		
		/*!	\brief stores index of the face. 	*/
		int index;

		/*!	\brief pointer to one of the zHalfEdge contained in the polygon.		*/
		zHalfEdge* he;

	public:	
				
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zFace();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zFace();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method gets the index of current face.
		*
		*	\return				int - face index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getId();

		/*! \brief This method sets the index of current face to the the input value.
		*
		*	\param		[in]	_faceId - input face index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setId(int _faceId);

		/*! \brief This method returns the associated edge of current zFace.
		*
		*	\return				zHalfEdge -  half edge pointed to by the face.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zHalfEdge* getHalfEdge();

		/*! \brief This method sets the associated edge of current zFace to the the input edge
		*
		*	\param		[in]	_he - input half edge.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setHalfEdge(zHalfEdge* _he);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method makes the resets pointers of the current face to null.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool isActive() const;

	};	


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/geometry/zHEGeomTypes.cpp>
#endif

#endif