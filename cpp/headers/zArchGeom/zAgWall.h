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

#ifndef ZSPACE_AG_WALL_H
#define ZSPACE_AG_WALL_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zHousing/base/zHcEnumerators.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zAgWall
	*	\brief A toolset for creating walls in housing units
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_AG zAgWall
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to input mesh Object  */
		zObjMesh *inMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnInMesh;

		/*!	\brief input vertex corners */
		zPointArray vertexCorners;

		/*!	\brief id of the parent's face its parented to */
		int faceId;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgWall();

		/*! \brief overload constructor.
		*
		*	\param		[in]	_inMeshObj					- input mesh object.
		*	\param		[in]	_vertexcorners				- input vertex corners.
		*	\since version 0.0.4
		*/
		zAgWall(zObjMesh&_inMeshObj, zPointArray&_vertexCorners, int _faceId);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgWall();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		void createWallByType(zStructureType&_structureType);

		//--------------------------
		//---- UPDATE METHODS
		//--------------------------

		void updateWall(zPointArray&_vertexCorners);

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zArchGeom/zAgWall.cpp>
#endif

#endif