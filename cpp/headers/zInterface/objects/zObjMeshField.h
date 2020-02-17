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

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		void draw() override;
#endif
			
		void getBounds(zPoint &minBB, zPoint &maxBB) override;

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

	
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/** \addtogroup Container
	*	\brief  The container typedef of the library.
	*  @{
	*/

	/*! \typedef zObjMeshScalarFieldArray
	*	\brief A vector of zObjMeshScalarField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjMeshScalarField> zObjMeshScalarFieldArray;

	/*! \typedef zObjMeshVectorFieldArray
	*	\brief A vector of zObjMeshVectorField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjMeshVectorField> zObjMeshVectorFieldArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/
}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjMeshField.cpp>
#endif

#endif
