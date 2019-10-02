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

#ifndef ZSPACE_RHINO_FN_MESH_H
#define ZSPACE_RHINO_FN_MESH_H

#pragma once

#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/functionsets/zFnMesh.h>
#include<headers/zInterface/iterators/zItMesh.h>

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

	/*! \class zFnRhinoMesh
	*	\brief A Rhino mesh interOp function set.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_RHINO zRhinoFnMesh : public zFnMesh
	{

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zRhinoFnMesh();

		/*! \brief Overloaded constructor.
		*
		*	\param	[in]	_zspace_meshObj			- input zspace mesh object.
		*	\since version 0.0.4
		*/
		zRhinoFnMesh(zObjMesh &_zspace_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zRhinoFnMesh();

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace mesh from the input rhino mesh.
		*
		*	\param	[in]		rhino_mesh	- input rhino mesh.
		*	\since version 0.0.4
		*/
		void fromRhinoMesh(ON_Mesh &rhino_mesh);

		/*! \brief This method creates the rhino mesh from the zspace mesh.
		*
		*	\param	[out]		rhino_mesh	- output rhino mesh object.
		*	\since version 0.0.4
		*/
		void toRhinoMesh(ON_Mesh &rhino_mesh);


	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/functionSets/zRhinoFnMesh.cpp>
#endif

#endif

#endif