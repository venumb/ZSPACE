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


#ifndef ZSPACE_VOL_GEOMTYPES_H
#define ZSPACE_VOL_GEOMTYPES_H

#pragma once

#include <vector>
using namespace std;

#include<headers/zCore/base/zInline.h>

namespace zSpace
{	
	
	class ZSPACE_CORE zVolEdge;
	class ZSPACE_CORE zVolHalfEdge;
	class ZSPACE_CORE zVolVertex;
	class ZSPACE_CORE zVolFace;
	class ZSPACE_CORE zVolCell;

	typedef std::vector<zVolVertex>::iterator		zItVolVertex;
	typedef std::vector<zVolHalfEdge>::iterator		zItVolHalfEdge;
	typedef std::vector<zVolEdge>::iterator			zItVolEdge;
	typedef std::vector<zVolFace>::iterator			zItVolFace;
	typedef std::vector<zVolFace>::iterator			zItVolCell;
	
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zVolHandles
	*	\brief The half-face geometry handle classes of the library.
	*  @{
	*/

	/*! \class zVolEdge
	*	\brief An edge class to hold edge information of a half-face data structure.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zVolEdge
	{
	protected:
		
		/*!	\brief stores of the edge. */
		int index;

		/*!	\brief pointer to 2 half edges of the edge. */
		zItVolHalfEdge he0;
		zItVolHalfEdge he1;

	public:		
	
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zVolEdge();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/

		~zVolEdge();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the edgeId of current edge.
		*
		*	\return				int -  edge index.
		*	\since version 0.0.1
		*/
		int getId();

		/*! \brief This method sets the edgeId of current edge to the input value.
		*
		*	\param		[in]	edgeId - input edge index.
		*	\since version 0.0.1
		*/
		void setId(int _edgeId);

		/*! \brief This method gets the half edge pointer of current edge at the input index.
		*
		*	\param		[in]	_index		- input index - 0 or 1.
		*	\since version 0.0.1
		*/
		zItVolHalfEdge getHalfEdge(int _index);

		/*! \brief This method sets the half edge pointer of current edge at the input index.
		*
		*	\param		[in]	_he			- input half edge pointer.
		*	\param		[in]	_index		- input index - 0 or 1.
		*	\since version 0.0.1
		*/
		void setHalfEdge(zItVolHalfEdge &_he, int _index);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method checks if the current element is active.
		*
		*	\since version 0.0.1
		*/
		bool isActive() const;
		
		/*! \brief This method resets the pointers of the current edge to null.
		*
		*	\since version 0.0.1
		*/
		void reset();	

	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zVolHandles
	*	\brief The half-face geometry handle classes of the library.
	*  @{
	*/

	/*! \class zVolHalfEdge
	*	\brief An half edge class to hold half edge information of a half-face data structure.
	*	\since version 0.0.4
	*/

	/** @}*/ 

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zVolHalfEdge
	{
	protected:
		
		/*!	\brief iterator	in vertex list. */
		zItVolVertex v;

		/*!	\brief iterator in face list. */
		vector<zItVolFace> faces;

		/*!	\brief iterator in edge list. */
		zItVolEdge e;

		/*!	\brief iterator to symmerty/twin half edge.			*/
		zItVolHalfEdge sym;

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
		zVolHalfEdge();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zVolHalfEdge();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the index of current half edge.
		*
		*	\return				int - half edge index.
		*	\since version 0.0.1
		*/
		int getId();

		/*! \brief This method sets the index of current half edge to the the input value.
		*
		*	\param		[in]	_ edgeId - input half edge index
		*	\since version 0.0.1
		*/
		void setId(int _edgeId);

		/*! \brief This method returns the symmetry half edge of current half edge.
		*
		*	\return				zHalfEdge - symmetry halfedge pointed by the half edge.
		*	\since version 0.0.1
		*/
		zItVolHalfEdge getSym();

		/*! \brief This method sets the symmetry edge of current zEdge to the the input edge
		*
		*	\param		[in]	_sym - symmetry half edge.
		*	\since version 0.0.1
		*/
		void setSym(zItVolHalfEdge &_sym);
		
		/*! \brief This method returns the vertex pointed to by the current zEdge.
		*
		*	\return				zVertex - vertex pointed by the half edge.
		*	\since version 0.0.1
		*/
		zVertex* getVertex();

		/*! \brief This method sets the vertex pointed to by the current half edge to the the input vertex.
		*
		*	\param		[in]	_v - input vertex.
		*	\since version 0.0.1
		*/
		void setVertex(zVertex* _v);

		/*! \brief This method returns the face pointed to by the current half edge.
		*
		*	\return				zEdge - face pointed by the half edge.
		*	\since version 0.0.1
		*/
		zFace* getFace();

		/*! \brief This method sets the face pointed to by the current half edge to the the input face.
		*
		*	\param		[in]	 _f - input face.
		*	\since version 0.0.1
		*/
		void setFace(zFace* _f);

		/*! \brief This method gets the edge pointed to by the current half edge.
		*
		*	\return				zEdge - edge pointed by the half edge.
		*	\since version 0.0.1
		*/
		zEdge* getEdge();

		/*! \brief This method sets the edge pointed to by the current halfedge to the the input edge.
		*
		*	\param		[in]	 _e - input edge.
		*	\since version 0.0.1
		*/
		void setEdge(zEdge* _e);

		/*! \brief This method resets the pointers of the half edge to null.
		*
		*	\since version 0.0.1
		*/
		void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		bool isActive() const;

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zVolHandles
	*	\brief The half-face geometry handle classes of the library.
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
		zVertex();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zVertex();
		
		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method gets the index of current vertex.
		*
		*	\return				int - vertex index.
		*	\since version 0.0.1
		*/
		int getId();

		/*! \brief This method sets the index of current vertex to the the input value.
		*
		*	\param		[in]	_vertexId - input index.
		*	\since version 0.0.1
		*/
		void setId(int _vertexId);

		/*! \brief This method gets the half edge pointed by the current Vertex.
		*
		*	\return				zHalfEdge - half edge pointed to by the vertex.
		*	\since version 0.0.1
		*/
		zHalfEdge* getHalfEdge();

		/*! \brief This method sets the half edge of the current vertex to the the input half edge
		*
		*	\param		[in]	_he - input half edge.
		*	\since version 0.0.1
		*/
		void setHalfEdge(zHalfEdge* _he);

		/*! \brief This method makes the pointers of the current zVertex to null.
		*
		*	\since version 0.0.1
		*/
		void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		bool isActive() const;

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zVolHandles
	*	\brief The half-face geometry handle classes of the library.
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
		zFace();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zFace();

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method gets the index of current face.
		*
		*	\return				int - face index.
		*	\since version 0.0.1
		*/
		int getId();

		/*! \brief This method sets the index of current face to the the input value.
		*
		*	\param		[in]	_faceId - input face index.
		*	\since version 0.0.1
		*/
		void setId(int _faceId);

		/*! \brief This method returns the associated edge of current zFace.
		*
		*	\return				zHalfEdge -  half edge pointed to by the face.
		*	\since version 0.0.1
		*/
		zHalfEdge* getHalfEdge();

		/*! \brief This method sets the associated edge of current zFace to the the input edge
		*
		*	\param		[in]	_he - input half edge.
		*	\since version 0.0.1
		*/
		void setHalfEdge(zHalfEdge* _he);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method makes the resets pointers of the current face to null.
		*
		*	\since version 0.0.1
		*/
		void reset();

		/*! \brief This method checks if the current element is active.
		*
		*	\return				bool - true if active else false.
		*	\since version 0.0.1
		*/
		bool isActive() const;

	};	


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/geometry/zVolGeomTypes.cpp>
#endif

#endif