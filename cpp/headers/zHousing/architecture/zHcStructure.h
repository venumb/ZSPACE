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

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zInterface/model/zModel.h>

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

		bool displayColumns = true;

		/*!	\brief pointer to display model  */
		zModel *model;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief mesh Object  */
		zObjMesh* cellObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnCell;

		/*!	\brief column array per cell set  */
		zColumnArray columnArray;

		/*!	\brief slabs array per cell set  */
		zSlabArray slabArray;

		/*!	\brief slabs array per cell set  */
		zWallArray wallArray;

		/*!	\brief slabs array per cell set  */
		zFacadeArray facadeArray;

		/*!	\brief structure height float default: 300cm */
		float height = 3;

		/*!	\brief pointer to output mesh Object  */
		zObjMesh outMeshObj;

		/*!	\brief output mesh function set  */
		zFnMesh fnOutMesh;

		/*!	\brief pointer container of columns pointers  */
		zObjMeshPointerArray columnObjs;

		/*!	\brief pointer container of slabs pointers  */
		zObjMeshPointerArray slabObjs;

		/*!	\brief pointer container of walls pointers  */
		zObjMeshPointerArray wallObjs;

		/*!	\brief pointer container of facade pointers  */
		zObjMeshPointerArray facadeObjs;

		/*! \brief container to top edges attributes */
		zBoolArray cellEdgesAttributes;

		/*! \brief container to edge boundary attributes */
		zBoolArray cellBoundaryAttributes;

		/*! \brief container to cell faces attributes */
		vector <zCellFace> cellFaceArray;

		/*! \brief pointer to housing unit that this cell is attached to*/
		//class zHcUnit* parentUnit;

		

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
		zHcStructure(zModel&_model, zPointArray &faceVertexPositions, zBoolArray&_cellEdgesAttributes, zBoolArray&_cellBoundaryAttributes, zFunctionType&_funcType);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zHcStructure();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a spatial cell in which the architectural elements live in
		*
		*	\param		[in]	_vertexpositions_					- ordered set of vertices up and down.
		*	\since version 0.0.4
		*/
		void createStructureCell(zPointArray &vertexPositions_);

		/*! \brief This method creates a spatial cell in which the architectural elements live in
		*
		*	\since version 0.0.4
		*/
		void setCellFacesAttibutes();

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
		//---- SET METHODS
		//--------------------------


		/*! \brief This method sets show vertices boolean.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\since version 0.0.2
		*/
		void setShowColumns(bool _showCols);

	};

	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcStructure.cpp>
#endif

#endif