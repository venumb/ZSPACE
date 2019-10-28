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

#ifndef ZSPACE_OBJ_GRAPH_H
#define ZSPACE_OBJ_GRAPH_H

#pragma once

#include <headers/zInterface/objects/zObj.h>
#include <headers/zCore/geometry/zGraph.h>

#include <vector>
using namespace std;

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjGraph
	*	\brief The graph object class.
	*	\since version 0.0.2
	*/

	/** @}*/
	
	/** @}*/

	class ZSPACE_API zObjGraph : public zObj
	{
	private:
		/*! \brief boolean for displaying the vertices */
		bool showVertices;

		/*! \brief boolean for displaying the edges */
		bool showEdges;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief graph */
		zGraph graph;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjGraph();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjGraph();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show vertices, edges and face booleans.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showVerts, bool _showEdges);

		/*! \brief This method sets show vertices boolean.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\since version 0.0.2
		*/
		void setShowVertices(bool _showVerts);

		/*! \brief This method sets show edges boolean.
		*
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowEdges(bool _showEdges);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the vertex VBO Index .
		*
		*	\return			int				- vertex VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexID();

		/*! \brief This method gets the edge VBO Index .
		*
		*	\return			int				- edge VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_EdgeID();

		/*! \brief This method gets the vertex color VBO Index .
		*
		*	\return			int				- vertex color VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexColorID();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

#ifndef ZSPACE_UNREAL_INTEROP
		void draw() override;
#endif

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zGraph.
		*
		*	\since version 0.0.2
		*/
		void drawGraph();


		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends graph to the buffer.
		*
		*	\since version 0.0.1
		*/
		void appendToBuffer();

	};

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

	/*! \typedef zObjGraphArray
	*	\brief A vector of zObjGraph.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjGraph> zObjGraphArray;

	/*! \typedef zObjGraphPointerArray
	*	\brief A vector of zObjGraph pointers.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjGraph*>zObjGraphPointerArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjGraph.cpp>
#endif

#endif