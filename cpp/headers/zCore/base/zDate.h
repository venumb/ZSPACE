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
#include <stdexcept>
#include <vector>
#include <ostream>
#include <time.h> 
using namespace std;

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

	/*! \class zDate
	*	\brief A date class which extend the c++ time struct tm .
	*	\since version 0.0.4
	*/

	/** @}*/
	/** @}*/

	class ZSPACE_CORE zDate : public tm
	{

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE zDate();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_month		- month [values between 1-12].
		*	\param		[in]	_day		- day [values between 1-31].
		*	\param		[in]	_hour		- hour [values between 0-23].
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE zDate(int _month, int _day, int _hour);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_year		- year.
		*	\param		[in]	_month		- month [values between 1-12].
		*	\param		[in]	_day		- day [values between 1-31].
		*	\param		[in]	_hour		- hour [values between 0-23].
		*	\param		[in]	_minute		- minute [values between 0-59].
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE zDate(int _year, int _month, int _day, int _hour, int _minute);	

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zDate();

		//--------------------------
		//---- OPERATORS
		//--------------------------	

		/*! \brief This operator checks for equality of two zDates.
		*
		*	\param		[in]	other		- zDates against which the equality is checked.
		*	\return				bool		- true if vectors are equal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool operator==(const zDate &other) const;

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method converts unix time to Gregorian date.
		*
		*	\param		[in]	unix			- input unix time.
		*	\return				zDate			- output Gregorian date.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void fromUnix(time_t unix);

		/*! \brief This method converts Julian date to Gregorian date.
		*
		*	\details https://stackoverflow.com/questions/7136385/calculate-day-number-from-an-unix-timestamp-in-a-math-way
		*	\param		[in]	date			- input Julian date.
		*	\return				zDate			- output Gregorian date.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void fromJulian(double julian);

		/*! \brief This method converts Gregorian date to Julian date.
		*
		*	\param		[in]	date			- input Gregorian date.
		*	\return				double			- output Julian date.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE double toJulian();

		/*! \brief This method converts unix time to Julian date.
		*
		*	\param		[in]	unix			- input unix time.
		*	\return				double			- output Julian date.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE double toJulian(time_t unix);

		/*! \brief This method converts Julian date to unix time.
		*
		*	\param		[in]	unix			- input Julian date.
		*	\return				time_t			- output unix time.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE time_t  toUnix(double  julian);

		/*! \brief This method converts Gregorian date to unix time.
		*
		*	\param		[in]	date			- input Gregorian date.
		*	\return				time_t			- output unix time.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE time_t  toUnix();

#ifndef __CUDACC__

		//--------------------------
		//---- STREAM OPERATORS
		//--------------------------

		/*! \brief This method outputs the dates component values to the stream.
		*
		*	\param		[in]		os				- output stream.
		*	\param		[in]		zDate			- date to be streamed.
		*	\since version 0.0.1
		*/

		friend ZSPACE_CORE ostream& operator<<(ostream& os, const zDate& date);		

#endif
		
	};

}


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \struct hash<zSpace::zDate>
	*	\brief A hash operator for the zDate class.
	*	\since version 0.0.4
	*/

	/** @}*/
	/** @}*/

template <>
struct ZSPACE_CORE std::hash<zSpace::zDate>
{
	ZSPACE_CUDA_CALLABLE std::size_t operator()(const zSpace::zDate& k) const;

};

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/base/zDate.cpp>
#endif

#endif


