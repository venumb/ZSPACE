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

#ifndef ZSPACE_TS_STATICS_POLYTOPAL_H
#define ZSPACE_TS_STATICS_POLYTOPAL_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


#include <depends/alglib/cpp/src/ap.h>
#include <depends/alglib/cpp/src/linalg.h>
#include <depends/alglib/cpp/src/optimization.h>
using namespace alglib;

namespace zSpace
{
	
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \struct zGlobalVertex
	*	\brief A struct for to hold global vertex attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zGlobalVertex
	{
		zPoint pos;
		
		/*!	\brief container of conincedent vertex indicies  */
		vector<int> coincidentVertices;

	};
	
	
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	
	/*! \class zTsSDFBridge
	*	\brief A toolset for 3D graphics and poytopal meshes.
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf) and Freeform Developable Spatial Structures (https://www.ingentaconnect.com/content/iass/piass/2016/00002016/00000003/art00010 )
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsSDFBridge
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to input guide mesh Object  */
		zObjMesh *o_guideMesh;

		/*!	\brief cut plane mesh Object  */
		zObjMesh* o_planeMesh;

		/*!	\brief map between a coincident plane vertex and global vertex  */
		zIntArray planeVertex_globalVertex;

		/*!	\brief map between a guide edge to global vertex  */
		zIntArray guideEdge_globalVertex;

		vector<zGlobalVertex> globalVertices;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zTsSDFBridge();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_o_guideMesh		- input guide mesh object.
		*	\since version 0.0.4
		*/
		zTsSDFBridge(zObjMesh& _o_guideMesh);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zTsSDFBridge();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param		[in]	width			- input plane width.
		*	\since version 0.0.2
		*/
		void createCutPlaneMesh(double width);

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsSDFBridge.cpp>
#endif

#endif