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

#ifndef ZSPACE_ZDATE_H
#define ZSPACE_ZDATE_H

#pragma once
#include <vector>

#include <headers/zCore/base/zInline.h>

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

	/** \addtogroup zStructs
	*	\brief  The structs of the library.
	*  @{
	*/


	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zDate
	{
	public:

		int year = 0;
		int month, day, hour;
		int minute = 0;
		int monthDays[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

		zDate();

		zDate(int _month, int _day, int _hour);
		

		zDate(int _year, int _month, int _day, int _hour, int _minute);
	

		bool operator==(const zDate &other) const;

		// This function counts number of leap years before the 
		// given date 
		int countLeapYears(zDate d);

		// This function returns number of days between two given 
		// dates 
		int getDifference(zDate dt1, zDate dt2);
		
	};

}


// Create Hashkey for zDate

template <>
struct ZSPACE_CORE std::hash<zSpace::zDate>
{
	std::size_t operator()(const zSpace::zDate& k) const;

};

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/base/zDate.cpp>
#endif

#endif


