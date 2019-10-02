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

#if defined(ZSPACE_RHINO_INTEROP) 

#ifndef ZSPACE_RHINO_FN_GRAPH_H
#define ZSPACE_RHINO_FN_GRAPH_H

#pragma once

#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/functionsets/zFnGraph.h>
#include<headers/zInterface/iterators/zItGraph.h>

#include <headers/zInterOp/include/zRhinoInclude.h>


namespace zSpace
{
	/** \addtogroup zInterOp
	*	\brief classes and function sets for inter operability between maya, rhino and zspace.
	*  @{
	*/

	/** \addtogroup zRhinoFunctionSets
	*	\brief Collection of function set classes for intergration with Rhinoceros using the Rhino C++ API.
	*  @{
	*/

	/*! \class zFnRhinoGraph
	*	\brief A Rhino curve to graph interOp function set.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_RHINO zRhinoFnGraph : public zFnGraph
	{

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zRhinoFnGraph();

		/*! \brief Overloaded constructor.
		*
		*	\param	[in]	_zspace_meshObj			- input zspace mesh object.
		*	\since version 0.0.4
		*/
		zRhinoFnGraph(zObjGraph &_zspace_graphObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zRhinoFnGraph();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace graph from the input rhino curves.
		*
		*	\param	[in]		rhino_curves	- input array of rhino curve object.
		*	\param	[in]		onlyCorners		- graph create only with first and last CV point if true.
		*	\since version 0.0.4
		*/
		void fromRhinoCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves, bool onlyCorners);


		/*! \brief This method creates the rhino curves from the zspace graph.
		*
		*	\param	[out]		rhino_curves	- output array of rhino curve object.
		*	\since version 0.0.4
		*/
		void toRhinoCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves);


	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/functionSets/zRhinoFnGraph.cpp>
#endif

#endif

#endif