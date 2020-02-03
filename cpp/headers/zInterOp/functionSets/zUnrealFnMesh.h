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

#if defined(ZSPACE_UNREAL_INTEROP) 

#ifndef ZSPACE_UNREAL_FN_MESH_H
#define ZSPACE_UNREAL_FN_MESH_H

#pragma once

#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/functionsets/zFnMesh.h>
#include<headers/zInterface/iterators/zItMesh.h>

#include <headers/zInterOp/include/zUnrealInclude.h>


namespace zSpace
{
	/** \addtogroup zInterOp
	*	\brief classes and function sets for inter operability between maya, rhino, unreal and zspace.
	*  @{
	*/

	/** \addtogroup zUnrealFunctionSets
	*	\brief Collection of function set classes for intergration with Unreal.
	*  @{
	*/

	/*! \class zFnUnrealMesh
	*	\brief An Unreal mesh interOp function set.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_RHINO zUnrealFnMesh : public zFnMesh
	{

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zUnrealFnMesh();

		/*! \brief Overloaded constructor.
		*
		*	\param	[in]	_zspace_meshObj			- input zspace mesh object.
		*	\since version 0.0.4
		*/
		zUnrealFnMesh(zObjMesh &_zspace_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zUnrealFnMesh();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace mesh from the input rhino mesh.
		*
		*	\param	[in]		rhino_mesh	- input rhino mesh.
		*	\since version 0.0.4
		*/
		void fromUnrealMesh(FDynamicMesh3 &unreal_mesh);

		/*! \brief This method creates the rhino mesh from the zspace mesh.
		*
		*	\param	[out]		rhino_mesh	- output rhino mesh object.
		*	\since version 0.0.4
		*/
		void toUnrealMesh(FDynamicMesh3 &unreal_mesh);

	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/functionSets/zUnrealFnMesh.cpp>
#endif

#endif

#endif