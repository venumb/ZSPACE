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

#ifndef ZSPACE_INTEROP_MESH_H
#define ZSPACE_INTEROP_MESH_H

#pragma once

#include <headers/zInterOp/functionsets/zMayaFnMesh.h>
#include <headers/zInterOp/functionsets/zRhinoFnMesh.h>

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

	/*! \class zIOMesh
	*	\brief A mesh interOp class.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	
	class ZSPACE_INTEROP  zIOMesh
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
		zIOMesh();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zIOMesh();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace mesh from the input maya mesh.
		*
		*	\param		[in]		maya_meshObj	- input maya mesh object.
		*	\param		[out]		zspace_meshObj	- output zspace mesh object.
		*	\since version 0.0.4
		*/
		bool toZSpaceMesh(MObject &maya_meshObj, zObjMesh &zspace_meshObj);

		/*! \brief This method creates the zspace mesh from the input rhino mesh.
		*
		*	\param		[in]		rhino_meshObj	- input maya mesh object.
		*	\param		[out]		zspace_meshObj	- output zspace mesh object.
		*	\since version 0.0.4
		*/
		bool toZSpaceMesh(ON_Mesh &rhino_meshObj, zObjMesh &zspace_meshObj);

		/*! \brief This method creates the rhino mesh from the input zspace mesh.
		*
		*	\param		[in]		zspace_meshObj	- input zspace mesh object.
		*	\param		[out]		ON_Mesh			- output rhino mesh object.
		*	\since version 0.0.4
		*/
		bool toRhinoMesh(zObjMesh &zspace_meshObj, ON_Mesh &rhino_meshObj);

		/*! \brief This method creates the rhino mesh from the input maya mesh.
		*
		*	\param		[in]		maya_meshObj	- input maya mesh object.
		*	\param		[out]		ON_Mesh			- output rhino mesh object.
		*	\since version 0.0.4
		*/
		bool toRhinoMesh(MObject &maya_meshObj, ON_Mesh &rhino_meshObj);

		/*! \brief This method creates the maya mesh from the input zspace mesh.
		*
		*	\param		[in]		zspace_meshObj	- input zspace mesh object.
		*	\param		[out]		maya_meshObj	- output maya mesh object.
		*	\since version 0.0.4
		*/
		bool toMayaMesh(zObjMesh &zspace_meshObj, MObject &maya_meshObj);

		/*! \brief This method creates the maya mesh from the input rhino mesh.
		*
		*	\param		[in]		rhino_meshObj	- input maya mesh object.
		*	\param		[out]		maya_meshObj	- output maya mesh object.
		*	\since version 0.0.4
		*/
		bool toMayaMesh(ON_Mesh &rhino_meshObj, MObject &maya_meshObj);

	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/interOp/zIOMesh.cpp>
#endif

#endif

#endif