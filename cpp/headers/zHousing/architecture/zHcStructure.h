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

#ifndef ZSPACE_HC_STRUCTURE_H
#define ZSPACE_HC_STRUCTURE_H

#pragma once

#include <headers/zInterface/model/zModel.h>

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>


#include <headers/zHousing/base/zHcEnumerators.h>
#include <headers/zArchGeom/zAgTypedef.h>


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zHcStructure
	*	\brief A toolset for deploying structural elements on housing units
	*	\since version 0.0.4
	*/

	/** @}*/


	class ZSPACE_AG zHcStructure
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to display model  */
		zModel *model;

		/*!	\brief structure type (i.e. rhwc, timber)  */
		zStructureType structureType;

		/*!	\brief structure type (i.e. rhwc, timber)  */
		zFunctionType functionType;


	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------


		/*!	\brief pointer to mesh Object  */
		zObjMesh* inStructObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnStruct;

		/*!	\brief mesh Object array */
		zObjMeshArray cellObjs;

		/*!	\brief column array per cell set  */
		zColumnArray columnArray;

		/*!	\brief slabs array per cell set  */
		zSlabArray slabArray;

		/*!	\brief slabs array per cell set  */
		zWallArray wallArray;

		/*!	\brief slabs array per cell set  */
		zFacadeArray facadeArray;

		/*!	\brief structure height float array ( per face ) */
		zFloatArray heightArray;

		/*!	\brief pointer to output mesh Object  */
		zObjMesh outMeshObj;

		/*!	\brief output mesh function set  */
		zFnMesh fnOutMesh;

		/*!	\brief pointer container of columns pointers  */
		zObjMeshArray columnObjs;

		/*!	\brief pointer container of slabs pointers  */
		zObjMeshArray slabObjs;

		/*!	\brief pointer container of walls pointers  */
		zObjMeshArray wallObjs;

		/*!	\brief pointer container of facade pointers  */
		zObjMeshArray facadeObjs;

		/*! \brief container to top edges attributes */
		zBoolArray edgesAttributes;

		/*! \brief container to edge boundary attributes */
		zBoolArray boundaryAttributes;

		/*! \brief container to cell faces attributes */
		vector <zCellFace> cellFaceArray;


		

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zHcStructure();

		/*! \brief Overloaded constructor.
		*
		*	\since version 0.0.4
		*/
		zHcStructure(zObjMesh&_inStructObj, zFunctionType&_funcType, zStructureType&_structureType, zFloatArray _heightArray, zBoolArray&_edgesAttributes, zBoolArray&_boundaryAttributes);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zHcStructure();


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method creates a spatial cell in which the architectural elements live in
		*
		*	\since version 0.0.4
		*/
		void setCellFacesAttibutes();


		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a spatial cell in which the architectural elements live in
		*
		*	\param		[in]	_structuretype					- structural type 
		*	\since version 0.0.4
		*/
		void createStructureByType(zStructureType&_structureType);

		/*! \brief This method updates a spatial cell 
		*
		*	\param		[in]	_height							- cell height
		*	\since version 0.0.4
		*/
		void updateStructure(zFloatArray _heightArray);

		/*! \brief This method creates a spatial cell in which the architectural elements live in
		*
		*	\param		[in]	_structuretype					- structural type
		*	\since version 0.0.4
		*/
		void updateArchComponents(zStructureType&_structureType);


		/*! \brief This method creates the columns that live in this structure cell object
		*
		*	\since version 0.0.4
		*/
		bool createColumns();

		/*! \brief This method creates the slabs that live in this structure cell object
		*
		*	\since version 0.0.4
		*/
		bool createSlabs();

		/*! \brief This method creates the walls that live in this structure cell object
		*
		*	\since version 0.0.4
		*/
		bool createWalls();

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\since version 0.0.4
		*/
		bool createFacades();

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showColumns					- display column objs
		*	\since version 0.0.4
		*/
		void displayColumns(bool showColumns);

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showSlabs						- display slab objs
		*	\since version 0.0.4
		*/
		void displaySlabs(bool showSlabs);

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showWalls					- display wall objs
		*	\since version 0.0.4
		*/
		void displayWalls( bool showWalls);

		/*! \brief This method creates the facades that live in this structure cell object
		*
		*	\param		[in]	_showFacade					- display facade objs
		*	\since version 0.0.4
		*/
		void displayFacade( bool showFacade);


#ifndef ZSPACE_UNREAL_INTEROP
		/*! \brief This method sets the model display not for Unreal.
		*
		*	\param		[in]	_model				- pointer to display model
		*	\since version 0.0.2
		*/
		void setStructureDisplayModel(zModel&_model);
#endif

	};

	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcStructure.cpp>
#endif

#endif