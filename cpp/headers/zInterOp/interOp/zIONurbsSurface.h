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

#if defined(ZSPACE_MAYA_INTEROP)  && defined(ZSPACE_RHINO_INTEROP)

#ifndef ZSPACE_INTEROP_NURBS_SURFACE_H
#define ZSPACE_INTEROP_NURBS_SURFACE_H

#pragma once

#include <headers/zInterOp/functionsets/zMayaFnGraph.h>
#include <headers/zInterOp/functionsets/zRhinoFnGraph.h>

namespace zSpace
{

	/** \addtogroup zInterOp
	*	\brief classes and function sets for inter operability between maya, rhino and zspace.
	*  @{
	*/

	/** \addtogroup zIO
	*	\brief classes for inter operability between maya, rhino and zspace geometry classes.
	*  @{
	*/

	/*! \class zIONurbsCurve
	*	\brief A NurbsCurve interOp class.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_INTEROP  zIONurbsSurface
	{
	protected:

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zIONurbsSurface();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zIONurbsSurface();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the rhino nurbs surface from the input maya nurbs surface.
		*
		*	\param		[in]		maya_curve		- input maya nurbs curve object.
		*	\param		[out]		ON_Mesh			- output rhino nurbs curve object.
		*	\since version 0.0.4
		*/
		bool toRhinoSurface(MObject &maya_surface, ON_NurbsSurface &rhino_surface);

		/*! \brief This method creates the rhino nurbs surface from the input maya nurbs surface.
		*
		*	\param		[in]		rhino_curve		- input rhino nurbs curve object.
		*	\param		[out]		maya_curve		- output maya nurbs curve object.
		*	\since version 0.0.4
		*/
		bool toMayaSurface(ON_NurbsSurface &rhino_surface, MObject &maya_surface);

	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/interOp/zIONurbsSurface.cpp>
#endif

#endif

#endif