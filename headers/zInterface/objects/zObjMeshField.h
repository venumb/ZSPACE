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

#ifndef ZSPACE_OBJ_MESHFIELD_H
#define ZSPACE_OBJ_MESHFIELD_H

#pragma once

#include <headers/zInterface/objects/zObjMesh.h>
#include <headers/zCore/field/zField2D.h>

#include <vector>
using namespace std;

namespace zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjMeshField
	*	\brief A template 2D field mesh object class.
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	template<typename T>
	class ZSPACE_API zObjMeshField : public zObjMesh
	{

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField2D<T> field;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjMeshField();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjMeshField();		

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override;
			
		void getBounds(zVector &minBB, zVector &maxBB) override;

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \typedef zObjMeshScalarField
	*	\brief A 2D scalar field mesh object.
	*
	*	\since version 0.0.2
	*/
	typedef zObjMeshField<zScalar> zObjMeshScalarField;

	/*! \typedef zObjMeshVectorField
	*	\brief A 2D scalar field mesh object.
	*
	*	\since version 0.0.2
	*/
	typedef zObjMeshField<zVector> zObjMeshVectorField;

	/** @}*/

	/** @}*/
}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjMeshField.cpp>
#endif

#endif
