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

#ifndef ZSPACE_AG_COLUMN_H
#define ZSPACE_AG_COLUMN_H

#pragma once

#include <headers/zCore/base/zExtern.h>
#include <headers/zInterface/model/zModel.h>

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

	/*! \class zAgColumn
	*	\brief A toolset for creating columns in housing units
	*	\since version 0.0.4
	*/


	class ZSPACE_AG zAgColumn : public zAgObj
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief stores x and y directions */
		zVector x, y, z;


	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to column mesh Object  */
		zObjMesh columnMeshObj;

		/*!	\brief input height  */
		float height;

		/*!	\brief input axis x and y */
		zVectorArray axis;

		/*!	\brief input avis attributes primary or secondary  */
		zBoolArray axisAttributes;

		/*!	\brief input mesh snap points for slab creation  */
		vector<zVectorArray> snapSlabpoints;

		/*!	\brief input array of neighbouring cell condition */
		vector<zBoundary> boundaryArray;

		/*!	\brief stores position point*/
		zVector position;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgColumn();

		/*! \brief overloaded constructor.
		*	
		*	\param		[in]	_position					- world position.
		*	\param		[in]	_showForces					- array of axis vectors.
		*	\param		[in]	_showForces					- array of axis attributes.
		*	\param		[in]	_showForces					- array on boundary attributes.
		*	\param		[in]	_showForces					- input height.
		*	\since version 0.0.4
		*/
		zAgColumn(zVector&_position, zVectorArray&_axis, zBoolArray&_axisAttributes, vector<zBoundary>&_boundaryArray, float _height);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgColumn();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a robotic hotwire cut column
		*
		*	\since version 0.0.4
		*/
		void createRhwc() override;

		/*! \brief This method creates a Timber column
		*
		*	\since version 0.0.4
		*/
		void createTimber() override;


		/*! \brief This method creates structural extrusions from wireframe mesh
		*
		*	\since version 0.0.4
		*/
		void createFrame();

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the mesh associated with this obj
		*
		*	\since version 0.0.4
		*/
		void displayColumn(bool showColumn);

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		/*! \brief This method displays the mesh associated with this obj
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
#include<source/zArchGeom/zAgColumn.cpp>
#endif

#endif