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

#ifndef ZSPACE_AG_ROOF_H
#define ZSPACE_AG_ROOF_H

#pragma once

#include <headers/zInterface/model/zModel.h>
#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zArchGeom/zAgObj.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zAgSlab
	*	\brief A toolset for creating roofs in housing units
	*	\since version 0.0.4
	*/
	/** @}*/

	class ZSPACE_AG zAgRoof : public zAgObj
	{
	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief roof mesh Object  */
		zObjMesh roofMeshObj;

		/*!	\brief input center vectors for oposite corner */
		zPointArray vertexCorners; 
			
		/*!	\brief input boolean condition if is in facade */
		bool isFacade;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgRoof();

		/*! \brief overload constructor.
		*
		*	\param		[in]	_vertexCorners				- input vertex corners.
		*	\param		[in]	_isFacade					- input scale of forces.
		*	\since version 0.0.4
		*/
		zAgRoof(zPointArray&_vertexCorners, bool _isFacade);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgRoof();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------


		/*! \brief This method creates a robotic hotwire cut slab
		*
		*	\since version 0.0.4
		*/
		void createRhwc() override;

		/*! \brief This method creates a timber slab
		*
		*	\since version 0.0.4
		*/
		void createTimber() override;

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the mesh associated with this obj
		*
		*	\since version 0.0.4
		*/
		void displayRoof(bool showRoof);

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void addObjsToModel() override;
#endif
	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zArchGeom/zAgRoof.cpp>
#endif

#endif