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

#ifndef ZSPACE_ZCOLOR_H
#define ZSPACE_ZCOLOR_H

#pragma once

#include <stdexcept>

#include<headers/zCore/base/zInline.h>
#include<headers/zCore/base/zEnumerators.h>

using namespace std;

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
	
	/*! \class zColor
	*	\brief	A color math class. 
	*
	*	Currently supports two types of color data structures - RGBA and HSV.
	*	\since	version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class ZSPACE_CORE zColor
	{
	public:

		/*!	\brief red component				*/
		float r; 

		/*!	\brief green component			*/
		float g; 

		/*!	\brief blue component			*/
		float b; 

		/*!	\brief alpha component			*/
		float a;  

		/*!	\brief hue component				*/
		float h; 

		/*!	\brief saturation component		*/
		float s;

		/*!	\brief value component			*/
		float v;  
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief	 Default constructor sets by default r,g,b,h,s,v to 0 and a to 1.
		*	\since	 version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor();

		/*! \brief	Overloaded constructor sets RGB_A components of the color and uses it to compute HSV components.
		*
		*	\param		[in]	_r		- red component of zColor.
		*	\param		[in]	_g		- green component of zColor.
		*	\param		[in]	_b		- blue component of zColor.
		*	\param		[in]	_a		- alpha component of zColor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor(float _r, float _g, float _b, float _a);
		

		// overloaded constructor HSV
		/*! \brief Overloaded constructor using HSV components of the color and uses it to compute RGB_A components.
		*
		*	\param		[in]	_h		- hue component of zColor.
		*	\param		[in]	_s		- saturation component of zColor.
		*	\param		[in]	_v		- value component of zColor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor(float _h, float _s, float _v);
		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zColor();

		//--------------------------
		//---- METHODS
		//--------------------------

		
		/*! \brief This methods calculates the HSV components based on the RGB_A components of color.
		*	\since version 0.0.1
		*/		
		ZSPACE_CUDA_CALLABLE void toHSV();
		

		/*! \brief This methods calculates the RGB_A components based on the HSV components of color.
		*	\since version 0.0.1
		*/		
		ZSPACE_CUDA_CALLABLE void toRGB();
			

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This operator checks for equality of two zColors.
		*
		*	\param		[in]	c1		- input color.
		*	\return				bool	- true if colors are the same.
		*	\since version 0.0.1
		*/				
		ZSPACE_CUDA_CALLABLE bool operator==(zColor &c1);
		

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/base/zColor.cpp>
#endif

#endif