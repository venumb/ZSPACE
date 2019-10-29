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

#ifndef ZSPACE_HC_ENUMERATORS
#define ZSPACE_HC_ENUMERATORS

	/*!	\namespace	zSpace
	*	\brief		namespace for the library.
	*	\since		version 0.0.4
	*/
namespace zSpace
{
	/** \addtogroup zHcEnumerators
	*	\brief  The enumerators of the library.
	*  @{
	*/

	/*! \enum	zLayout
	*	\brief	Housing Layout options for users to choose from
	*	\since	version 0.0.4
	*/	
	enum zLayoutType { zOpenPlan, zPrivate};

	/*! \enum	zStructureType
	*	\brief	Manufacturing Technology associated with the construction / Tectonics
	*	\since	version 0.0.4
	*/
	enum zStructureType { zRHWC, zDigitalTimber };

	/*! \enum	zFunctionType
	*	\brief	Programmatic use of the housing unit configurator
	*	\since	version 0.0.4
	*/	
	enum zFunctionType { zPublic, zStudio, zLoft, zLandscape, zTType, zLType };






	/** @}*/

	/** @}*/

	/** @}*/
}

#endif