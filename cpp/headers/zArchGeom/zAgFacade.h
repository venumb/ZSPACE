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

#ifndef ZSPACE_AG_FACADE_H
#define ZSPACE_AG_FACADE_H

#pragma once

#include <headers/zInterface/model/zModel.h>
#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zHousing/base/zHcEnumerators.h>

#include <headers/zArchGeom/zAgObj.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zAgFacade
	*	\brief A toolset for creating facades in housing units
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_AG zAgFacade : public zAgObj
	{

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief facade mesh Object array */
		zObjMeshArray facadeObjs;

		/*!	\brief input vertex corners */
		zPointArray vertexCorners;

		/*!	\brief input direction of extrusion */
		zVectorArray extrudeDir;

		/*!	\brief id of the parent's face its parented to */
		int faceId;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgFacade();

		/*! \brief overload constructor.
		*
		*	\param		[in]	_vertexcorners			- input vertex corners.
		*	\param		[in]	_extrudeDir				- input extrude direction.
		*	\param		[in]	_vertexcorners			- input face id of parent mesh.
		*	\since version 0.0.4
		*/
		zAgFacade(zPointArray&_vertexCorners, zVectorArray&_extrudeDir, int _faceId);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgFacade();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createTimber() override;

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createTimberMullions(zObjMesh&_timberMullions);

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createTimberGlass(zObjMesh&_timberGlass);

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createTimberPanels(zObjMesh&_timberPanels);


		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createRhwc() override;

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createRwhcMullions(zObjMesh&_RwhcMullions);

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createRwhcGlass(zObjMesh&_RwhcGlass);

		/*! \brief This method creates a robotic hotwire cut facade
		*
		*	\since version 0.0.4
		*/
		void createRwhcPanels(zObjMesh&_RwhcPanels);

		/*! \brief This method creates mullions
		*
		*	\since version 0.0.4
		*/
		void createMullions(zObjMesh&_Mullions);

		//--------------------------
		//---- UPDATE METHODS
		//--------------------------

		void updateFacade(zPointArray&_vertexCorners);

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the mesh associated with this obj
		*
		*	\since version 0.0.4
		*/
		void displayFacade(bool showFacade);


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
#include<source/zArchGeom/zAgFacade.cpp>
#endif

#endif