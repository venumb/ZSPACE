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

#ifndef ZSPACE_CONFIG_DEFINITION
#define ZSPACE_CONFIG_DEFINITION

#pragma once

namespace zSpace
{
	/** \addtogroup zConfigurator
	*	\brief Collection of tool sets for configurator.
	*  @{
	*/
	
	/** \addtogroup base
	*	\brief base tool sets for configurator.
	*  @{
	*/

	/** \addtogroup zCfDefinitions
	*	\brief  The defintions of the configurator.
	*  @{
	*/

	//--------------------------
	//---- UNIT TYPE DEFINITIONS
	//--------------------------

	/*!
	*	\brief Defines the string of Unit_Base.
	*/
	#ifndef Unit_Base
	#define Unit_Base "base" 
	#endif

	/*!
	*	\brief Defines the string of Unit_Wide.
	*/
	#ifndef Unit_Wide
	#define Unit_Wide "wide" 
	#endif

	/*!
	*	\brief Defines the string of Unit_Tall.
	*/
	#ifndef Unit_Tall
	#define Unit_Tall "tall" 
	#endif

	/*!
	*	\brief Defines the string of Unit_Side.
	*/
	#ifndef Unit_Side
	#define Unit_Side "side" 
	#endif

	//--------------------------
	//---- VOXEL LOCATION DEFINITIONS
	//--------------------------

	/*!
	*	\brief Defines the string of VL_Interior.
	*/
	#ifndef VL_Interior
	#define VL_Interior "Interior" 
	#endif

	/*!
	*	\brief Defines the string of VL_Terrace.
	*/
	#ifndef VL_Terrace
	#define VL_Terrace "Terrace" 
	#endif

	/*!
	*	\brief Defines the string of VL_Entrance.
	*/
	#ifndef VL_Entrance
	#define VL_Entrance "Entrance" 
	#endif

	//--------------------------
	//---- VOXEL FACE GEOMETRY TYPES DEFINITIONS
	//--------------------------

	/*!
	*	\brief Defines the string of VFG_Interior_Ceiling_Arched.
	*/
	#ifndef VFG_Interior_Ceiling_Arched
	#define VFG_Interior_Ceiling_Arched "Interior_Ceiling_Arched" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Ceiling_Arched.
	*/
	#ifndef VFG_Nosing_Ceiling_Arched
	#define VFG_Nosing_Ceiling_Arched "Nosing_Ceiling_Arched" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Interior_Wall_Void.
	*/
	#ifndef VFG_Interior_Wall_Void
	#define VFG_Interior_Wall_Void "Interior_Wall_Void" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Wall_Void.
	*/
	#ifndef VFG_Nosing_Wall_Void
	#define VFG_Nosing_Wall_Void "Nosing_Wall_Void" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Interior_Wall_LongSolid.
	*/
	#ifndef VFG_Interior_Wall_LongSolid
	#define VFG_Interior_Wall_LongSolid "Interior_Wall_LongSolid" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Wall_LongSolid.
	*/
	#ifndef VFG_Nosing_Wall_LongSolid
	#define VFG_Nosing_Wall_LongSolid "Nosing_Wall_LongSolid" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Wall_LipSolid.
	*/
	#ifndef VFG_Nosing_Wall_LipSolid
	#define VFG_Nosing_Wall_LipSolid "Nosing_Wall_LipSolid" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Interior_Wall_ShortSolid.
	*/
	#ifndef VFG_Interior_Wall_ShortSolid
	#define VFG_Interior_Wall_ShortSolid "Interior_Wall_ShortSolid" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Wall_ShortSolid.
	*/
	#ifndef VFG_Nosing_Wall_ShortSolid
	#define VFG_Nosing_Wall_ShortSolid "Nosing_Wall_ShortSolid" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Interior_Wall_LongGlazed.
	*/
	#ifndef VFG_Interior_Wall_LongGlazed
	#define VFG_Interior_Wall_LongGlazed "Interior_Wall_LongGlazed" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Wall_ShortCurveGlazed.
	*/
	#ifndef VFG_Nosing_Wall_ShortCurveGlazed
	#define VFG_Nosing_Wall_ShortCurveGlazed "Nosing_Wall_ShortCurveGlazed" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Interior_Floor_Flat.
	*/
	#ifndef VFG_Interior_Floor_Flat
	#define VFG_Interior_Floor_Flat "Interior_Floor_Flat" 
	#endif

	/*!
	*	\brief Defines the string of VFG_Nosing_Floor_Flat.
	*/
	#ifndef VFG_Nosing_Floor_Flat
	#define VFG_Nosing_Floor_Flat "Nosing_Floor_Flat" 
	#endif

	/** @}*/
	/** @}*/
	/** @}*/

}

#endif