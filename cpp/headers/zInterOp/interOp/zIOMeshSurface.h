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

#ifndef ZSPACE_INTEROP_MESH_SURFACE_H
#define ZSPACE_INTEROP_MESH_SURFACE_H

#pragma once

#include <headers/zInterOp/core/zRhinoCore.h>

#include <headers/zInterOp/interOp/zIOMesh.h>
#include <headers/zInterOp/interOp/zIONurbsCurve.h>
#include <headers/zInterOp/interOp/zIONurbsSurface.h>

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

	/*! \class zIOMeshSurface
	*	\brief A Maya Mesh to Rhino Surface interOp class.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_INTEROP  zIOMeshSurface
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to a zspace mesh object  */
		zObjMesh *zspace_meshObj;

		/*!	\brief pointer to a maya mesh object  */
		MObject *maya_meshObj;

		/*!	\brief a boolean to indicate if the input mesh is quad mesh or not  */
		bool inputQuadMesh = true;		

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief zspace mesh function set  */
		zFnMesh zspace_FnMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zIOMeshSurface();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_maya_meshObj			- input maya mesh object.
		*	\param		[in]	_zspace_meshObj			- input zspace mesh object.
		*	\since version 0.0.4
		*/
		zIOMeshSurface(MObject &_maya_meshObj, zObjMesh &_zspace_meshObj );

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zIOMeshSurface();


		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method checks if the input maya mesh is a quad Mesh, if not will sundivide by 1 division.
		*
		*	\param		[in]	subdivs					- input number of subdivsions.
		*	\param		[out]	rhino_surface			- out container of rhino nurbs surface.
		*	\since version 0.0.4
		*/
		void toRhinoSurface(int subdivs, ON_ClassArray<ON_NurbsCurve> &rhino_nurbsCurve, ON_ClassArray<ON_NurbsSurface> &rhino_surface);

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------

	protected:

		/*! \brief This method checks if the input maya mesh is a quad Mesh, if not will sundivide by 1 division.
		*
		*	\param		[in]	inMeshObj			- input maya mesh object.		
		*	\since version 0.0.4
		*/
		bool makeQuadMesh(MObject &inMeshObj);

		/*! \brief This method gets the vertices of the smooth mesh which correponds to the low-poly edge.
		*
		*	\param		[in]	inMeshObj			- input maya mesh object.
		*	\since version 0.0.4
		*/
		void getVertsforCurve(zItMeshHalfEdge &smooth_he, int nV_lowPoly, zItMeshVertexArray &crvVerts);
	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/interOp/zIOMeshSurface.cpp>
#endif

#endif

#endif