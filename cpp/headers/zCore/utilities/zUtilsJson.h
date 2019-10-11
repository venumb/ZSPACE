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

#ifndef ZSPACE_UTILS_JSON_H
#define ZSPACE_UTILS_JSON_H

#pragma once

#include<headers/zCore/base/zInline.h>

#include <depends/modernJSON/json.hpp>
using json = nlohmann::json;;

using namespace std;

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \struct zUtilsJsonHE
	*	\brief A json utility struct for storing half edge datastructure.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	struct ZSPACE_CORE zUtilsJsonHE
	{
			
		/*!	\brief container of vertex data  */
		vector<int> vertices;

		/*!	\brief container of half edge data */
		vector<zIntArray> halfedges;

		/*!	\brief container of face data*/
		vector<int> faces;

		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<zDoubleArray> vertexAttributes;

		/*!	\brief container of half edge attribute data - color*/
		vector<zDoubleArray> halfedgeAttributes;

		/*!	\brief container of face attribute data - normals, colors.*/
		vector<zDoubleArray> faceAttributes;
		
		/*!	\brief container of edge attribute data - crease*/
		zDoubleArray edgeCreaseData;
		
	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \struct zUtilsJsonRobot
	*	\brief A json utility struct for storing robot data.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE zUtilsJsonRobot
	{
		/*!	\brief container of DH_d values  */
		double scale;

		/*!	\brief container of DH_d values  */
		zDoubleArray d;

		/*!	\brief container of DH_a values  */
		zDoubleArray a;

		/*!	\brief container of DH_alpha values  */
		zDoubleArray alpha;

		/*!	\brief container of DH_alpha values  */
		zDoubleArray theta;

		/*!	\brief container of jointRotations_home values  */
		zDoubleArray jointRotations_home;

		/*!	\brief container of jointRotations_minimum values  */
		zDoubleArray jointRotations_minimum;

		/*!	\brief container of jointRotations_maximum values  */
		zDoubleArray jointRotations_maximum;

		/*!	\brief container of jointRotations_pulse values  */
		zDoubleArray jointRotations_pulse;

		/*!	\brief container of jointRotations_mask values  */
		zDoubleArray jointRotations_mask;

		/*!	\brief container of jointRotations_offset values  */
		zDoubleArray jointRotations_offset;

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \struct zUtilsJsonTopOpt
	*	\brief A json utility struct for storing topOpt data.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE zUtilsJsonTopOpt
	{
		/*!	\brief container of edge data */
		vector<zIntArray> edges;

		/*!	\brief container of face data */
		vector<zIntArray> faces;

		/*!	\brief container of vertex attribute data - positions, normals*/
		vector<zDoubleArray> vertexAttributes;

		/*!	\brief container of face attribute data - normals*/
		vector<zDoubleArray> faceAttributes;

		/*!	\brief container of SPC*/
		zBoolArray SPC;

		/*!	\brief container of design space*/
		zBoolArray designSpace;

		/*!	\brief container of Load point data*/
		vector<zDoubleArray> loads;

		/*!	\brief container of Load point data*/
		vector<zIntArray> loadPoints;

		/*!	\brief container of pattern grouping data  */
		vector< zDoubleArray > patternGrouping;

		/*!	\brief container of material data  */
		zDoubleArray material;
				
	};

}


#endif