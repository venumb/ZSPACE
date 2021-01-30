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

#if defined(ZSPACE_RHINO_INTEROP) 

#ifndef ZSPACE_INTEROP_RHINOCORE_H
#define ZSPACE_INTEROP_RHINOCORE_H

#pragma once
#include "headers/zCore/base/zInline.h"

namespace zSpace
{

	/** \addtogroup zInterOp
	*	\brief classes and function sets for inter operability between maya, rhino and zspace.
	*  @{
	*/

	/** \addtogroup zIO
	*	\brief classes for inter operability between maya, rhino and zspace geometry classes.
	*  @{
	*/

	/*! \class zRhinoCore
	*	\brief A Rhjino Core Object Class for loading required Rhino module fo interop.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/



	class ZSPACE_RHINO zRhinoCore
	{		

	public:

		HINSTANCE hinstLib;
		BOOL fFreeResult, fRunTimeLinkSuccess = FALSE;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zRhinoCore();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zRhinoCore();

		/*! \brief Method for startup process
		*
		*	\since version 0.0.4
		*/
		void startUp();

		/*! \brief Method for shutdown process
		*
		*	\since version 0.0.4
		*/
		void shutDown();
	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
//#include <source/zInterOp/core/zRhinoCore.cpp>
#endif


#endif

#endif