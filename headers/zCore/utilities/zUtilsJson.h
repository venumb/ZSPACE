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
		vector<vector<int>> halfedges;

		/*!	\brief container of face data*/
		vector<int> faces;

		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<vector<double>> vertexAttributes;

		/*!	\brief container of half edge attribute data - color*/
		vector<vector<double>> halfedgeAttributes;

		/*!	\brief container of face attribute data - normals, colors.*/
		vector<vector<double>> faceAttributes;	
		
		
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
		vector<double> d;

		/*!	\brief container of DH_a values  */
		vector<double> a;

		/*!	\brief container of DH_alpha values  */
		vector<double> alpha;

		/*!	\brief container of DH_alpha values  */
		vector<double> theta;

		/*!	\brief container of jointRotations_home values  */
		vector<double> jointRotations_home;

		/*!	\brief container of jointRotations_minimum values  */
		vector<double> jointRotations_minimum;

		/*!	\brief container of jointRotations_maximum values  */
		vector<double> jointRotations_maximum;

		/*!	\brief container of jointRotations_pulse values  */
		vector<double> jointRotations_pulse;

		/*!	\brief container of jointRotations_mask values  */
		vector<double> jointRotations_mask;

		/*!	\brief container of jointRotations_offset values  */
		vector<double> jointRotations_offset;		

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
		vector<vector<int>> edges;

		/*!	\brief container of face data */
		vector<vector<int>> faces;

		/*!	\brief container of vertex attribute data - positions, normals*/
		vector<vector<double>> vertexAttributes;

		/*!	\brief container of face attribute data - normals*/
		vector<vector<double>> faceAttributes;

		/*!	\brief container of SPC*/
		vector<bool> SPC;

		/*!	\brief container of design space*/
		vector<bool> designSpace;

		/*!	\brief container of Load point data*/
		vector<vector<double>> loads;

		/*!	\brief container of Load point data*/
		vector<vector<int>> loadPoints;

		/*!	\brief container of pattern grouping data  */
		vector< vector<double>> patternGrouping;

		/*!	\brief container of material data  */
		vector<double> material;
				
	};

}


#endif