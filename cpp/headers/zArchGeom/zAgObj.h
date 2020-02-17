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

#ifndef ZSPACE_AG_OBJ
#define ZSPACE_AG_OBJ

#pragma once

#include <headers/zCore/base/zExtern.h>
#include <headers/zInterface/model/zModel.h>

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


	class ZSPACE_AG zAgObj
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief input structure type */
		zStructureType structureType;

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*!	\brief pointer to display model  */
		zModel *model;
#endif

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgObj();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgObj();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates columns by a structural type
		*
		*	\param		[in]	_structureType					- input set structure type
		*	\since version 0.0.4
		*/
		void createByType(zStructureType&_structureType);

		/*! \brief This method creates a robotic hotwire cut arch geom
		*
		*	\since version 0.0.4
		*/
		virtual void createRhwc();

		/*! \brief This method creates a timber cut arch geom
		*
		*	\since version 0.0.4
		*/
		virtual void createTimber();


		//--------------------------
		//---- SET METHODS
		//--------------------------


#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void setModel(zModel&_model);

		/*! \brief This method adds objects to the model.
		*
		*	\since version 0.0.4
		*/
		virtual void addObjsToModel() = 0;
#endif

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zArchGeom/zAgObj.cpp>
#endif

#endif