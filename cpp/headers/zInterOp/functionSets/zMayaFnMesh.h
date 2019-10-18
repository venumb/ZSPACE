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

#ifndef ZSPACE_MAYA_FN_MESH_H
#define ZSPACE_MAYA_FN_MESH_H

#pragma once

#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/functionsets/zFnMesh.h>
#include<headers/zInterface/iterators/zItMesh.h>

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

	/*! \class zFnMayaMesh
	*	\brief A Maya mesh interOp function set.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_MAYA zMayaFnMesh : public zFnMesh
	{
	private:

		zDoubleArray creaseEdgeData;
		zIntArray creaseEdgeIndex;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zMayaFnMesh();

		/*! \brief Overloaded constructor.
		*
		*	\param	[in]	_zspace_meshObj			- input zspace mesh object.
		*	\since version 0.0.4
		*/
		zMayaFnMesh(zObjMesh &_zspace_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zMayaFnMesh();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		zFnType getType() override;

		void from(string path, zFileTpye type, bool staticGeom = false) override;

		void to(string path, zFileTpye type) override;

		void clear() override;

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------

		/*! \brief This method creates the zspace mesh from the input maya mesh.
		*
		*	\param	[in]		maya_meshObj	- input maya mesh object.
		*	\since version 0.0.4
		*/
		void fromMayaMesh(MObject &maya_meshObj);

		/*! \brief This method creates the zspace mesh from the input maya DAG object.
		*
		*	\param	[in]		maya_dagpath	- input maya DAG object.
		*	\since version 0.0.4
		*/
		void fromMayaMesh(MDagPath &maya_dagpath);

		/*! \brief This method creates the maya mesh from the zspace mesh.
		*
		*	\param	[out]		maya_meshObj	- output maya mesh object.
		*	\since version 0.0.4
		*/
		void toMayaMesh(MObject &maya_meshObj);

		/*! \brief This method updates the maya outmesh attribute.
		*
		*	\param	[out]		data		- input maya data block.
		*	\param	[out]		outMesh		- output maya object.
		*	\since version 0.0.4			
		*/
		void updateMayaOutmesh(MDataBlock & data, MObject & outMesh, bool updateVertexColor = false, bool updateFaceColor = false );

		//--------------------------
		//---- INTEROP METHODS
		//--------------------------
	private:

		void setCreaseDataJSON(string outfilename);

		void getCreaseDataJSON(string infilename);
	};

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterOp/functionSets/zMayaFnMesh.cpp>
#endif

#endif

#endif