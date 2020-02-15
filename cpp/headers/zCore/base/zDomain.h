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

#ifndef ZSPACE_ZDOMAIN_H
#define ZSPACE_ZDOMAIN_H

#pragma once

#include<headers/zCore/base/zDefinitions.h>
#include<headers/zCore/base/zMatrix.h>
#include<headers/zCore/base/zVector.h>
#include<headers/zCore/base/zColor.h>
#include<headers/zCore/base/zDate.h>

namespace  zSpace
{

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \struct zDomain
	*	\brief A struct for storing domain values.
	*	\since version 0.0.2
	*/

	/** @}*/
	/** @}*/

	template <typename T>
	class ZSPACE_CORE zDomain
	{
	public:
		/*!	\brief stores the minimum value of the domain*/
		T min;

		/*!	\brief stores the maximum value of the domain*/
		T max;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zDomain();
		
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_min	- minimum value of the domain.
		*	\param		[in]	_max	- maximum value of the domain.		
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zDomain(T _min, T _max);

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE ~zDomain();
		
	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zDomainTypedef
	*	\brief  The domain typedef of the library.
	*  @{
	*/

	/*! \typedef zDomainInt
	*	\brief A domain  of integers.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<int> zDomainInt;

	/*! \typedef zDomainDouble
	*	\brief A domain  of double.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<double> zDomainDouble;

	/*! \typedef zDomainFloat
	*	\brief A domain  of float.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<float> zDomainFloat;

	/*! \typedef zDomainColor
	*	\brief A domain  of color.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<zColor> zDomainColor;

	/*! \typedef zDomainVector
	*	\brief A domain  of vectors.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<zVector> zDomainVector;

	/*! \typedef zDomainDate
	*	\brief A domain  of dates.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<zDate> zDomainDate;


	/** @}*/
	/** @}*/
	/** @}*/
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/base/zDomain.cpp>
#endif

#endif