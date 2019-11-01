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

#ifndef ZSPACE_HC_AGGREGATION_H
#define ZSPACE_HC_AGGREGATION_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zInterface/model/zModel.h>

#include <headers/zHousing/architecture/zHcUnit.h>
#include <headers/zHousing/base/zHcEnumerators.h>
//#include <headers/zHousing/base/zHcTypeDef.h>
//#include <headers/zApp/include/zHousing.h>
//#include <headers/zApp/include/zArchGeom.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zHcAggregation
	*	\brief A toolset for housing units aggegation
	*	\since version 0.0.4
	*/

	/** @}*/

	class ZSPACE_AG zHcAggregation
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

		/*!	\brief input mesh Object  */
		zObjMeshArray inMeshObjs;

		/*!	\brief input mesh function set  */
		vector<zFnMesh> fnInMeshArray;

		/*!	\brief 2D container of structure obj meshes  */
		zObjMeshArray testObjs;

		/*!	\brief container of housing units pointer  */
		vector<zHcUnit*> unitArray;

		/*!	\brief interface manager, handles an input path directory*/
		zUtilsCore core;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zHcAggregation();


		/*! \brief overload constructor.
		*
		*	\since version 0.0.4
		*/
		zHcAggregation(zModel &_model);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zHcAggregation();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method creates housing units.
		*
		*	\param		[in]	_index				- input housing unit index
		*	\since version 0.0.4
		*/
		void createHousingUnits();

		/*! \brief This method creates housing units from an imported mesh.
		*
		*	\param		[in]	_path				- directory of files
		*	\param		[in]	_type				- file type
		*	\since version 0.0.4
		*/
		void importMeshFromDirectory(string&_path, zFileTpye type);


		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		void drawHousing();
	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcAggregation.cpp>
#endif

#endif