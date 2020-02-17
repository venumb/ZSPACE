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

#ifndef ZSPACE_HC_UNIT_H
#define ZSPACE_HC_UNIT_H

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

	/*! \class zHcUnit
	*	\brief A toolset for housing units configuration
	*	\since version 0.0.4
	*/

	/** @}*/

	class ZSPACE_AG zHcUnit
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

/*!	\brief pointer to display model  */
		zModel *model;
#endif

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------


		/*!	\brief pointer to input mesh Object  */
		zObjMesh *inUnitMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnUnitMesh;

		/*!	\brief pointer to input mesh Object  */
		vector<zObjMeshArray> layoutMeshObjs;

		/*!	\brief house layout option */
		zLayoutType layoutType;

		/*!	\brief house function option */
		zFunctionType funcType;

		/*!	\brief house structure option */
		zStructureType structureType;

		/*!	\brief structure unit */
		zHcStructure structureUnit;

		/*!	\brief store edge attributes: primary (true) secondary (false)  */
		zBoolArray edgeAttributes;

		/*!	\brief store boundary attributes */
		zBoolArray eBoundaryAttributes;

		/*!	\brief store structure height array */
		zFloatArray structureHeight;

		/*!	\brief interface manager, handles an input path directory*/
		zUtilsCore core;
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zHcUnit();


		/*! \brief Default constructor.
		*
		*	\param		[in]	_inMeshObj 					- input mesh
		*	\param		[in]	_structureType 				- input desired function type
		*	\param		[in]	_structureType 				- input desired structure type
		*	\since version 0.0.4
		*/
		zHcUnit(zObjMesh&_inMeshObj, zFunctionType&_funcType, zStructureType&_structureType);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zHcUnit();


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the edges attributes: primary or secundary and boundary conditions
		*
		*	\since version 0.0.4
		*/
		void setCellAttributes();

		/*! \brief This method sets the layout by specified type.
		*
		*	\param		[in]	_layoutType					- input desired layout option
		*	\since version 0.0.4
		*/
		void setLayoutByType(zLayoutType&_layout);

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates structural units zHcStructure.
		*
		*	\param		[in]	_structureType 				- input desired structure type
		*	\since version 0.0.4
		*/
		bool createStructuralUnits(zStructureType _structureType);

		//--------------------------
		//---- IMPORT METHODS
		//--------------------------

		/*! \brief This method creates internal layout.
		*
		*	\param		[in]	_layoutType					- input desired layout option
		*	\since version 0.0.4
		*/
		void importLayoutsFromPath(vector<string>_paths);

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showColumns					- display column objs
		*	\since version 0.0.4
		*/
		void displayLayout(int&_index, bool&_show);

		/*! \brief This method sets the zModel pointer.
		*
		*	\since version 0.0.4
		*/
		void setUnitDisplayModel(zModel&_model);
#endif
		
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcUnit.cpp>
#endif

#endif