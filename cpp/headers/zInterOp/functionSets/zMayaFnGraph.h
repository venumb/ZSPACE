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

#if defined(ZSPACE_MAYA_INTEROP) 

#ifndef ZSPACE_MAYA_FN_GRAPH_H
#define ZSPACE_MAYA_FN_GRAPH_H

#pragma once

#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/functionsets/zFnGraph.h>
#include<headers/zInterface/iterators/zItGraph.h>

#include <headers/zInterOp/include/zMayaInclude.h>


namespace zSpace
{
	/** \addtogroup zInterOp
	*	\brief classes and function sets for inter operability between maya, rhino and zspace.
	*  @{
	*/

	/** \addtogroup zMayaFunctionSets
	*	\brief Collection of function set classes for intergration with Maya using the Maya C++ API.
	*  @{
	*/

	/*! \class zMayaFnGraph
	*	\brief A Maya curve to graph interOp function set.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_MAYA zMayaFnGraph : public zFnGraph
	{

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zMayaFnGraph();

		/*! \brief Overloaded constructor.
		*
		*	\param	[in]	_zspace_graphObj			- input zspace graph object.
		*	\since version 0.0.4
		*/
		zMayaFnGraph(zObjGraph &_zspace_graphObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zMayaFnGraph();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace graph from the input maya curves.
		*
		*	\param	[in]		maya_inCurves	- input array of maya curve objects.
		*	\param	[in]		onlyCorners		- graph create only with first and last CV point if true.
		*	\since version 0.0.4
		*/
		void fromMayaCurves(MObjectArray &maya_inCurves , bool onlyCorners);

		/*! \brief This method creates the zspace mesh from the input maya points array, by creating a edge for every consecutive pair of vertices.
		*
		*	\param	[in]		maya_Points	- input array of maya points .
		*	\since version 0.0.4
		*/
		void fromMayaPoints(MPointArray &maya_Points);

		/*! \brief This method creates the maya curves from the zspace graph.
		*
		*	\since version 0.0.4
		*/
		void toMayaCurves(MObjectArray &maya_inCurves);

		/*! \brief This method creates the maya curves on screen directly from the zspace graph.
		*
		*	\since version 0.0.4
		*/
		void toMayaCurves();

	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/functionSets/zMayaFnGraph.cpp>
#endif

#endif

#endif
