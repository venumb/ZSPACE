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


#include<headers/zCore/base/zColor.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zColor::zColor()
	{
		r = g = b = 0;
		a = 1;

		h = s = v = 0;
	}

	ZSPACE_INLINE zColor::zColor(float _r, float _g, float _b, float _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;

		// compute HSV
		toHSV();
	}

	ZSPACE_INLINE zColor::zColor(float _h, float _s, float _v)
	{
		h = _h;
		s = _s;
		v = _v;


		// compute RGB
		toRGB();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zColor::~zColor() {}

	//---- METHODS

	ZSPACE_INLINE void zColor::toHSV()
	{

		float      min, max, delta;

		min = r < g ? r : g;
		min = min < b ? min : b;

		max = r > g ? r : g;
		max = max > b ? max : b;

		v = max;                                // v
		delta = max - min;
		if (delta < 0.00001)
		{
			s = 0;
			h = 0; // undefined, maybe nan?

		}

		if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
			s = (delta / max);                  // s
		}
		else {
			// if max is 0, then r = g = b = 0              
			// s = 0, h is undefined
			s = 0.0;
			h = NAN;                            // its now undefined

		}

		if (r >= max)                           // > is bogus, just keeps compilor happy
			h = (g - b) / delta;        // between yellow & magenta
		else
			if (g >= max)
				h = 2.0 + (b - r) / delta;  // between cyan & yellow
			else
				h = 4.0 + (r - g) / delta;  // between magenta & cyan

		h *= 60.0;                              // degrees

		if (h < 0.0)
			h += 360.0;

	}

	ZSPACE_INLINE void zColor::toRGB()
	{
		float      hh, p, q, t, ff;
		long        i;


		if (s <= 0.0) {       // < is bogus, just shuts up warnings
			r = v;
			g = v;
			b = v;

			return;
		}

		hh = h;

		if (hh >= 360.0) hh = 0.0;
		if (hh != 0.0) hh /= 60.0;

		i = (long)hh;
		ff = hh - i;
		p = v * (1.0 - s);
		q = v * (1.0 - (s * ff));
		t = v * (1.0 - (s * (1.0 - ff)));

		switch (i) {
		case 0:
			r = v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = v;
			b = p;
			break;
		case 2:
			r = p;
			g = v;
			b = t;
			break;

		case 3:
			r = p;
			g = q;
			b = v;
			break;
		case 4:
			r = t;
			g = p;
			b = v;
			break;
		case 5:
		default:
			r = v;
			g = p;
			b = q;
			break;
		}

	}

	//---- OVERLOADED OPERATORS

	ZSPACE_INLINE bool zColor::operator==(zColor &c1)
	{
		return (r == c1.r && g == c1.g && b == c1.b);
	}
		


}
