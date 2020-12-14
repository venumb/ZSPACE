#pragma once
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

#ifndef ZSPACE_CONFIG_ENUMERATORS
#define ZSPACE_CONFIG_ENUMERATORS


namespace zSpace
{

	/** \addtogroup zConfigurator
	*	\brief Collection of tool sets for configurator.
	*  @{
	*/

	/** \addtogroup Kit
	*	\brief Kit tool sets for configurator.
	*  @{
	*/

	/** \addtogroup zCfEnumerators
	*	\brief  The enumerators of the library.
	*  @{
	*/

	/*! \enum	zConnectivityType
	*	\brief	connectivty matrix types.
	*	\since	version 0.0.4
	*/
	enum zCfComponentType { zConfigfColumn = 1000, zConfigBeam, zConfigfWall };

	/** @}*/

	/** @}*/

	/** @}*/
}

#endif