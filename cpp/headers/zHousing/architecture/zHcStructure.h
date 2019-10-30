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
#include <headers/zInterface/functionsets/zFnParticle.h>

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

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to input mesh Object  */
		zObjMesh* inMeshObj;

		/*!	\brief input mesh function set  */
		zFnMesh fnInMesh;

		/*!	\brief column array per cell set  */
		zColumnArray columnArray;

		/*!	\brief slabs array per cell set  */
		zSlabArray slabArray;

		/*!	\brief structure height float default: 300cm */
		float height = 3;

		/*!	\brief pointer to output mesh Object  */
		zObjMesh outMeshObj;

		/*!	\brief output mesh function set  */
		zFnMesh fnOutMesh;

		/*!	\brief pointer container to structure Object  */
		zObjMeshPointerArray columnObjs;

		/*!	\brief pointer container to structure Object  */
		zObjMeshPointerArray slabObjs;

		/*! \brief container to top edges attributes */
		zBoolArray cellEdgesAttributes;

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
		zHcStructure(zObjMesh &_inMeshObj, zPointArray &faceVertexPositions, zObjMeshPointerArray&_columnObjs, zObjMeshPointerArray&_slabObjs, zBoolArray&_cellEdgesAttributes);

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

	};

	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zHousing/architecture/zHcStructure.cpp>
#endif

#endif