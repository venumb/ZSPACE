// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//

#ifndef ZSPACE_OBJ_POINTFIELD_H
#define ZSPACE_OBJ_POINTFIELD_H

#pragma once

#include <headers/zInterface/objects/zObjPointCloud.h>
#include <headers/zCore/field/zField3D.h>

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

	/*! \class zObjPointField
	*	\brief A template 3D point field object class.
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	template<typename T>
	class ZSPACE_API zObjPointField : public zObjPointCloud
	{
	private:
		
	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField3D<T> field;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjPointField();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjPointField();
	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		void draw() override;
#endif

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \typedef zObjPointScalarField
	*	\brief A 2D scalar field mesh object.
	*
	*	\since version 0.0.2
	*/
	typedef zObjPointField<zScalar> zObjPointScalarField;

	/*! \typedef zObjPointVectorField
	*	\brief A 2D scalar field mesh object.
	*
	*	\since version 0.0.2
	*/
	typedef zObjPointField<zVector> zObjPointVectorField;

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

	/*! \typedef zObjPointScalarFieldArray
	*	\brief A vector of zObjPointScalarField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjPointScalarField> zObjPointScalarFieldArray;

	/*! \typedef zObjPointVectorFieldArray
	*	\brief A vector of zObjPointVectorField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjPointVectorField> zObjPointVectorFieldArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjPointField.cpp>
#endif

#endif

