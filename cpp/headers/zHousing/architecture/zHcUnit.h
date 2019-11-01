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

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zInterface/model/zModel.h>

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

		/*!	\brief pointer to display model  */
		zModel *model;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to input mesh Object  */
		zObjMesh *inMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnInMesh;

		/*!	\brief house layout option */
		zLayoutType layoutType;

		/*!	\brief house function option */
		zFunctionType funcType;

		/*!	\brief house structure option */
		zStructureType structureType;

		/*!	\brief structure units/cells array */
		zStructurePointerArray structureUnits;

		/*!	\brief store edge attributes: primary (true) secondary (false)  */
		zBoolArray edgeAttributes;

		/*!	\brief store boundary attributes */
		zBoolArray eBoundaryAttributes;
		
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
		*	\since version 0.0.4
		*/
		zHcUnit(zModel&_model, zObjMesh& inMeshObj_, zFunctionType&_funcType);

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

		/*! \brief This method creates structural units zHcStructure.
		*
		*	\param		[in]	_columnObjs 				- input pointer of column mesh obj
		*	\param		[in]	_slabObjs					- input pointer of slab mesh obj
		*	\since version 0.0.4
		*/
		bool createStructuralUnits();

		/*! \brief This method creates internal layout.
	*
	*	\since version 0.0.4
	*/
		void setEdgesAttributes();

		/*! \brief This method creates internal layout.
		*
		*	\param		[in]	_layoutType					- input desired layout option
		*	\since version 0.0.4
		*/
		void createLayoutPlan(zLayoutType layout_);

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcUnit.cpp>
#endif

#endif