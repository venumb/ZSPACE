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

#ifndef ZSPACE_AG_SLAB_H
#define ZSPACE_AG_SLAB_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>
#include "zAgColumn.h";

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zAgSlab
	*	\brief A toolset for creating slabs in housing units
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_AG zAgSlab
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		zStructureType structureType;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to the column this object is parented to */
		zAgColumn* parentColumn;

		/*!	\brief pointer to input mesh Object  */
		zObjMesh *inMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnInMesh;

		zVector xCenter, yCenter, center;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgSlab();

		/*! \brief overload constructor.
		*
		*	\param		[in]	_position					- input show forces booelan.
		*	\param		[in]	_forceScale					- input scale of forces.
		*	\param		[in]	_position					- input show forces booelan.
		*	\param		[in]	_forceScale					- input scale of forces.
		*	\since version 0.0.4
		*/
		zAgSlab(zObjMesh&_inMeshObj, zVector&_xCenter, zVector&_yCenter, zVector&_center, zAgColumn&_parenColumn);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgSlab();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates slabs by a structural type
		*
		*	\param		[in]	_structureType					- input set structure type
		*	\since version 0.0.4
		*/
		void createSlabByType(zStructureType&_structureType);

		/*! \brief This method creates a robotic hotwire cut slab
		*
		*	\since version 0.0.4
		*/
		void createRhwcSlab();

		/*! \brief This method creates a timber slab
		*
		*	\since version 0.0.4
		*/
		void createTimberSlab();

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zArchGeom/zAgSlab.cpp>
#endif

#endif