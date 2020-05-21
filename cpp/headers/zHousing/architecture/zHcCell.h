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

#ifndef ZSPACE_HC_CELL_H
#define ZSPACE_HC_CELL_H

#pragma once

#include <headers/zInterface/model/zModel.h>

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zHousing/base/zHcEnumerators.h>
#include <headers/zHousing/base/zHcTypeDef.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zHcCell
	*	\brief A toolset for housing units grid cells
	*	\since version 0.0.4
	*/

	/** @}*/

	class ZSPACE_AG zHcCell
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		int level = -1;

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

/*!	\brief pointer to display model  */
		zModel *model;
#endif

		/*!	\stores current cell estate  */
		zGridCellEstate cellEstate;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------


		/*!	\brief pointer to input mesh Object  */
		zObjMesh cellObj;

		bool isSelected = false;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zHcCell();

		/*! \brief overload constructor.
		*
		*	\param		[in]	verticesPos					- vertex positions
		*	\param		[in]	cellEstate					- grid cell Estate
		*	\since version 0.0.4
		*/
		zHcCell(zGridCellEstate _cellEstate);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zHcCell();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief set level.
		*
		*	\param		[in]	level					- level position
		*	\since version 0.0.4
		*/
		zGridCellEstate getEstate();


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief set level.
		*
		*	\param		[in]	level					- level position
		*	\since version 0.0.4
		*/
		void setLevel(int _level);

		/*! \brief set level.
		*
		*	\param		[in]	level					- level position
		*	\since version 0.0.4
		*/
		void setEstate(zGridCellEstate estate);

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief create cell Mesh Objects.
		*
		*	\param		[in]	verticesPos					- vertex positions
		*	\param		[in]	cellEstate					- grid cell Estate
		*	\since version 0.0.4
		*/
		void createCellMesh(zPointArray vertexPositions);
		

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showColumns					- display column objs
		*	\since version 0.0.4
		*/
		void displayCell(bool&_showAll, int&_showLevel);

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void setColor(zColor color);

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void setModel(zModel&_model);

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void AddObjToModel();
#endif

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcCell.cpp>
#endif

#endif