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

#ifndef ZSPACE_TS_GEOMETRY_VARIABLE_EXTRUDE_H
#define ZSPACE_TS_GEOMETRY_VARIABLE_EXTRUDE_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>


namespace zSpace
{

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsVariableExtrude
	*	\brief A function set for extrusions based on color attributes.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsVariableExtrude
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnMesh fnMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsVariableExtrude();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsVariableExtrude(zObjMesh &_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsVariableExtrude();

		/*! \brief This method offset extrudes the faces of the input mesh based on vertex / face color. It uses only the red channel of the color.
		*
		
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableFaceOffset(bool keepExistingFaces = true, bool assignColor = true, float minVal = 0.01, float maxVal = 0.99, bool useVertexColor = false);

		/*! \brief This method offsets the boundary faces of the input mesh based on vertex color. It uses only the red channel of the color.
		*
		*	\details	face offset based on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableBoundaryOffset(bool keepExistingFaces = true, bool assignColor = true, float minVal = 0.01, float maxVal = 0.99);


		/*! \brief This method extrudes the input mesh based on vertex / face color. It uses only the red channel of the color.
		*
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableFaceThicknessExtrude(bool assignColor, float minVal, float maxVal);

	};
	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsVariableExtrude.cpp>
#endif

#endif