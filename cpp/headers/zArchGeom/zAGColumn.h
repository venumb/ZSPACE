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

#ifndef ZSPACE_AG_COLUMN_H
#define ZSPACE_AG_COLUMN_H

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

	/*! \class zAgColumn
	*	\brief A toolset for creating columns in housing units
	*	\since version 0.0.4
	*/


	class ZSPACE_AG zAgColumn
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
		float height;
		float nodeHeight = 1;
		float nodeDepth = 0.8;
		float beamA_Height = 0.5;
		float beamB_Height = 0.3;

		zVector position, a, b, c;
		zVector x, y, z;

		/*!	\brief pointer to input mesh Object  */
		zObjMesh *inMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnInMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgColumn();

		/*! \brief overloaded constructor.
		*	
		*	\param		[in]	_showForces					- input show forces booelan.
		*	\since version 0.0.4
		*/
		zAgColumn(zObjMesh&_inMeshObj, zVector&_position, zVector &_x, zVector & _y, zVector & _z, float&_height);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgColumn();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates columns by a structural type
		*
		*	\param		[in]	_structureType					- input set structure type
		*	\since version 0.0.4
		*/
		void createColumnByType(zStructureType&_structureType);

		/*! \brief This method creates a robotic hotwire cut column
		*
		*	\since version 0.0.4
		*/
		void createRhwcColumn();

		/*! \brief This method creates a robotic hotwire cut column
		*
		*	\since version 0.0.4
		*/
		void createTimberColumn();


	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zArchGeom/zAgColumn.cpp>
#endif

#endif