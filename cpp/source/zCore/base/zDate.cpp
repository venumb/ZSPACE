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


#include<headers/zCore/base/zDate.h>

namespace zSpace
{

	ZSPACE_INLINE zDate::zDate()
	{
		year = month = day = hour = minute = 0;
	}

	ZSPACE_INLINE  zDate::zDate(int _month, int _day, int _hour)
	{
		month = _month;
		day = _day;
		hour = _hour;
	}

	ZSPACE_INLINE  zDate::zDate(int _year, int _month, int _day, int _hour, int _minute)
	{
		year = _year;
		month = _month;
		day = _day;
		hour = _hour;
		minute = _minute;
	}

	ZSPACE_INLINE bool zDate::operator==(const zDate &other) const
	{
		return (month == other.month
			&& day == other.day
			&& hour == other.hour);
	}

	ZSPACE_INLINE int zDate::countLeapYears(zDate d)
	{
		int years = d.year;

		// Check if the current year needs to be considered 
		// for the count of leap years or not 
		if (d.month <= 2)
			years--;

		// An year is a leap year if it is a multiple of 4, 
		// multiple of 400 and not a multiple of 100. 
		return years / 4 - years / 100 + years / 400;
	}

	ZSPACE_INLINE int zDate::getDifference(zDate dt1, zDate dt2)
	{
		// COUNT TOTAL NUMBER OF DAYS BEFORE FIRST DATE 'dt1' 

		// initialize count using years and day 
		long int n1 = dt1.year * 365 + dt1.day;

		// Add days for months in given date 
		for (int i = 0; i < dt1.month - 1; i++)
			n1 += monthDays[i];

		// Since every leap year is of 366 days, 
		// Add a day for every leap year 
		n1 += countLeapYears(dt1);

		// SIMILARLY, COUNT TOTAL NUMBER OF DAYS BEFORE 'dt2' 

		long int n2 = dt2.year * 365 + dt2.day;
		for (int i = 0; i < dt2.month - 1; i++)
			n2 += monthDays[i];
		n2 += countLeapYears(dt2);

		// return difference between two counts 
		return (n2 - n1);
	}

}



ZSPACE_INLINE std::size_t std::hash<zSpace::zDate>::operator()(const zSpace::zDate & k) const
{
	using std::size_t;
	using std::hash;
	using std::string;


	return ((hash<int>()(k.month)
		^ (hash<int>()(k.day) << 1)) >> 1)
		^ (hash<int>()(k.hour) << 1);

}

