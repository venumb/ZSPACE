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

#include<headers/managed_zCore/base/zColor.h>

namespace zSpaceManaged
{
	//---- CONSTRUCTOR

	zColor::zColor() : zManagedObj(new zSpace::zColor()) {}	

	zColor::zColor(double _r, double _g, double _b, double _a) : zManagedObj(new zSpace::zColor(_r, _g, _b, _a)) {}	

	zColor::zColor(double _h, double _s, double _v) : zManagedObj(new zSpace::zColor(_h, _s, _v)) {}
	
	zColor::zColor(const zColor ^& c1) : zManagedObj(new zSpace::zColor())
	{
		m_zInstance = c1->m_zInstance;
	}

	//---- METHODS

	void zColor::toHSV()
	{
		m_zInstance->toHSV();
	}

	void zColor::toRGB()
	{
		m_zInstance->toRGB();
	}

	bool zColor::operator==(zColor c1)
	{
		return m_zInstance->operator==(*c1.m_zInstance);
	}

}