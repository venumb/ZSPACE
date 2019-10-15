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

#ifndef ZSPACE_TYPEDEF
#define ZSPACE_TYPEDEF

#pragma once
#include <headers/zCore/base/zVector.h>
#include <headers/zCore/base/zMatrix.h>
#include <headers/zCore/base/zColor.h>

#include <headers/zCore/geometry/zHEGeomTypes.h>


namespace zSpace
{
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

	/*! \typedef zIntArray
	*	\brief A vector of integers.
	*
	*	\since version 0.0.4
	*/
	typedef vector<int> zIntArray;

	/*! \typedef zDoubleArray
	*	\brief A vector of double.
	*
	*	\since version 0.0.4
	*/
	typedef vector<double> zDoubleArray;

	/*! \typedef zFloatArray
	*	\brief A vector of double.
	*
	*	\since version 0.0.4
	*/
	typedef vector<float> zFloatArray;

	/*! \typedef zFloatArray
	*	\brief A vector of double.
	*
	*	\since version 0.0.4
	*/
	typedef vector<string> zStringArray;

	/*! \typedef zBoolArray
	*	\brief A vector of bool.
	*
	*	\since version 0.0.4
	*/
	typedef vector<bool> zBoolArray;

	/*! \typedef zInt2DArray
	*	\brief A 2 dimensional vector of int.
	*
	*	\since version 0.0.4
	*/
	typedef vector<vector<int>> zInt2DArray;

	/*! \typedef zInt3DArray
	*	\brief A 3 dimensional vector of int.
	*
	*	\since version 0.0.4
	*/
	typedef vector<vector<vector<int>>> zInt3DArray;

	/*! \typedef zVector3DArray
	*	\brief A 3 dimensional vector of zVector.
	*
	*	\since version 0.0.4
	*/
	typedef vector<vector<vector<zVector>>> zVector3DArray;

	/*! \typedef zPointArray
	*	\brief A vector of zVector representing points.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zVector> zPointArray;

	/*! \typedef zVectorArray
	*	\brief A vector of zVector representing vectors.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zVector> zVectorArray;

	/*! \typedef zColorArray
	*	\brief A vector of zColor.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zColor> zColorArray;

	/*! \typedef zVertexArray
	*	\brief A vector of zVertex.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zVertex> zVertexArray;

	/*! \typedef zHalfEdgeArray
	*	\brief A vector of zHalfEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zHalfEdge> zHalfEdgeArray;

	/*! \typedef zEdgeArray
	*	\brief A vector of zEdge.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zEdge> zEdgeArray;

	/*! \typedef zFaceArray
	*	\brief A vector of zFace.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zFace> zFaceArray;

	/*! \typedef zCurvatureArray
	*	\brief A vector of zCurvature.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zCurvature> zCurvatureArray;

	/*! \typedef zScalarArray
	*	\brief A vector of zScalar.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zScalar> zScalarArray;

	/*! \typedef zIntPairArray
	*	\brief A vector of integer pairs.
	*
	*	\since version 0.0.4
	*/
	typedef vector<pair<int, int>> zIntPairArray;

	/** @}*/

	/*! \typedef zPoint
	*	\brief A 3D position.
	*
	*	\since version 0.0.4
	*/
	typedef zVector zPoint;


	/** @}*/

	/** \addtogroup Pair
	*	\brief  The pair typedef of the library.
	*  @{
	*/

	/*! \typedef zIntPair
	*	\brief A pair of integers.
	*
	*	\since version 0.0.4
	*/
	typedef pair<int, int> zIntPair;
	
	/** @}*/

	/** @}*/
	/** @}*/	

}


#endif