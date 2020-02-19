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

	//---- CONSTRUCTOR

	ZSPACE_INLINE zDate::zDate()
	{
		tm_year = 1970;

		tm_mon = 1;
		tm_mday = 1;
		tm_hour = 0;

		tm_min = 0;
		tm_sec = 0;
	}

	ZSPACE_INLINE  zDate::zDate(int _month, int _day, int _hour)
	{
		tm_year = 1970;

		tm_mon = _month;
		tm_mday = _day;
		tm_hour = _hour;

		tm_min = 0;
		tm_sec = 0;
		
	}

	ZSPACE_INLINE  zDate::zDate(int _year, int _month, int _day, int _hour, int _minute)
	{
		tm_year = _year;
		tm_mon = _month;
		tm_mday = _day;
		tm_hour = _hour;
		tm_min = _minute;

		tm_sec = 0;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zDate::~zDate() {}

	//---- OPERATORS

	ZSPACE_INLINE bool zDate::operator==(const zDate &other) const
	{
		return (tm_mon == other.tm_mon
			&& tm_mday == other.tm_mday
			&& tm_hour == other.tm_hour);
	}

	//---- DATE  METHODS

	ZSPACE_INLINE void zDate::fromUnix(time_t unixSecs)
	{
		
		int z = unixSecs / 86400 + 719468;
		int era = (z >= 0 ? z : z - 146096) / 146097;
		unsigned doe = static_cast<unsigned>(z - era * 146097);
		unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
		int y = static_cast<int>(yoe) + era * 400;
		unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
		unsigned mp = (5 * doy + 2) / 153;
		unsigned d = doy - (153 * mp + 2) / 5 + 1;
		unsigned m = mp + (mp < 10 ? 3 : -9);
		y += (m <= 2);

		zDate temp(y, m, d, 0, 0);
		time_t unix_dayStart = temp.toUnix();

		int remainder = unixSecs - unix_dayStart;
		int h = (int)remainder / 3600;
		int min = fmod((float)(remainder / 60.0), (float) 60.0);

		*this = zDate(y, m, d, h, min);


	}

	ZSPACE_INLINE void zDate::fromJulian(double julian)
	{
		time_t unixTime = toUnix(julian);
		this->fromUnix(unixTime);
	}

	ZSPACE_INLINE double zDate::toJulian()
	{


		double y, m, d, b;

		if (tm_mon > 2)
		{
			y = tm_year;
			m = tm_mon;
		}
		else
		{
			y = tm_year - 1;
			m = tm_mon + 12;
		}


		d = tm_mday;
		if (tm_hour != 0)  d += (double)(tm_hour / 24.0);
		if (tm_min != 0) d += (double)(tm_min / 1440.0);
		if (tm_sec != 0) d += (double)(tm_sec / 86400.0);
		b = 2 - floor(y / 100) + floor(y / 400);

		return floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + d + b - 1524.5;


	}

	ZSPACE_INLINE double zDate::toJulian(time_t unixSecs)
	{
		return (unixSecs / 86400.0) + 2440587.5;
	}

	ZSPACE_INLINE time_t zDate::toUnix(double julian)
	{
		return (julian - 2440587.5) * 86400;;
	}

	ZSPACE_INLINE time_t zDate::toUnix()
	{
		double jd = toJulian();
		return toUnix(jd);
	}

#ifndef __CUDACC__
	
	//---- STREAM OPERATORS

	ZSPACE_INLINE ostream& operator<<(ostream & os, const zDate & date)
	{
		os << " [ " << date.tm_year << ',' << date.tm_mon << ',' << date.tm_mday << ',' << date.tm_hour << ',' << date.tm_min << " ]";
		return os;
	}

#endif

}



ZSPACE_INLINE std::size_t std::hash<zSpace::zDate>::operator()(const zSpace::zDate & k) const
{
	using std::size_t;
	using std::hash;
	using std::string;


	return ((hash<int>()(k.tm_mon)
		^ (hash<int>()(k.tm_mday) << 1)) >> 1)
		^ (hash<int>()(k.tm_hour) << 1);

}

