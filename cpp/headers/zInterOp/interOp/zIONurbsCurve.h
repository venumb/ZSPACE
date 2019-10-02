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

#ifndef ZSPACE_INTEROP_NURBS_CURVE_H
#define ZSPACE_INTEROP_NURBS_CURVE_H

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
	
	class ZSPACE_INTEROP  zIONurbsCurve
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
		zIONurbsCurve();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zIONurbsCurve();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace graph from the input maya nurbs curve.
		*
		*	\param		[in]		maya_curves		- input array maya nurbscurve objects.
		*	\param		[in]		onlyCorners		- graph create only with first and last CV point if true.
		*	\param		[out]		zspace_graphObj	- output zspace graph object.
		*	\since version 0.0.4
		*/
		bool toZSpaceCurves(MObjectArray &maya_curves, bool onlyCorners, zObjGraph &zspace_graphObj);

		/*! \brief This method creates the zspace graph from the input rhino nurbs curve.
		*
		*	\param		[in]		rhino_curves	- input array rhino nurbscurve objects.
		*	\param		[in]		onlyCorners		- graph create only with first and last CV point if true.
		*	\param		[out]		zspace_graphObj	- output zspace graph object.
		*	\since version 0.0.4
		*/
		bool toZSpaceCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves, bool onlyCorners, zObjGraph &zspace_graphObj);

		/*! \brief This method creates the zspace mesh from the input maya mesh.
		*
		*	\param		[in]		zspace_graphObj	- input zspace graph object.
		*	\param		[out]		rhino_curve		- output rhino nurbs curve object.
		*	\since version 0.0.4
		*/
		bool toRhinoCurves(zObjGraph &zspace_graphObj, ON_ClassArray<ON_NurbsCurve> &rhino_curves);

		/*! \brief This method creates the rhino curve from the input maya curve.
		*
		*	\param		[in]		maya_curve		- input maya nurbs curve object.
		*	\param		[out]		rhino_curve		- output rhino nurbs curve object.
		*	\since version 0.0.4
		*/
		bool toRhinoCurve(MObject &maya_curve, ON_NurbsCurve &rhino_curve);

		/*! \brief This method creates the maya curves from the input zspace graph.
		*
		*	\param		[in]		zspace_graphObj	- input zspace graph object.
		*	\param		[out]		maya_curves		- output array of maya curve objects.
		*	\since version 0.0.4
		*/
		bool toMayaCurves(zObjGraph &zspace_graphObj, MObjectArray &maya_curves);

		/*! \brief This method creates the rhino nurbs curve from the input maya nurbs curve.
		*
		*	\param		[in]		rhino_curve		- input rhino nurbs curve object.
		*	\param		[out]		maya_curve		- output maya nurbs curve object.
		*	\since version 0.0.4
		*/
		bool toMayaCurve(ON_NurbsCurve &rhino_curve, MObject &maya_curve);

	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/interOp/zIONurbsCurve.cpp>
#endif

#endif

#endif