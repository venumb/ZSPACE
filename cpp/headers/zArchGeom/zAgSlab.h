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

#ifndef ZSPACE_AG_SLAB_H
#define ZSPACE_AG_SLAB_H

#pragma once

#include <headers/zCore/base/zExtern.h>
#include <headers/zInterface/model/zModel.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zArchGeom/zAgObj.h>
#include <headers/zArchGeom/zAgColumn.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{

	/*! \class zAgSlab
	*	\brief A toolset for creating slabs in housing units
	*	\since version 0.0.4
	*/
	/** @}*/

	class ZSPACE_AG zAgSlab : public zAgObj
	{

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to the column this object is parented to */
		zAgColumn* parentColumn;

		/*!	\brief slab mesh Object  */
		zObjMesh slabMeshObj;

		/*!	\brief input center vectors for oposite corner */
		zVectorArray centerVecs; 
			
		/*!	\brief input vector axis */
		zVectorArray midPoints;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zAgSlab();

		/*! \brief overload constructor.
		*
		*	\param		[in]	_centerVecs					- input corners corresponding to centers of slab.
		*	\param		[in]	_midPoints					- input corners corresponding to midpoint of bay.
		*	\param		[in]	_parentColumn				- pointer to parent column.
		*	\since version 0.0.4
		*/
		zAgSlab(zVectorArray&_centerVecs, zVectorArray&_midPoints, zAgColumn&_parentColumn);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zAgSlab();

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
		void displaySlab(bool showSlab);


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
#include<source/zArchGeom/zAgSlab.cpp>
#endif

#endif