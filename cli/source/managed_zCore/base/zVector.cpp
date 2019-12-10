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

#include<headers/managed_zCore/base/zVector.h>

namespace zSpaceManaged
{
	//---- CONSTRUCTOR

	zVector::zVector() : zManagedObj(new zSpace::zVector()) {}
	
	zVector::zVector(double _x, double _y, double _z) : zManagedObj(new zSpace::zVector(_x, _y, _z)) {}

	//---- OPERATOR

	bool zVector::operator==(zVector c1)
	{
		return m_zInstance->operator==(*c1.m_zInstance);
	}

	double zVector::operator[](int index)
	{
		m_zInstance->operator[](index);
	}

	zVector zVector::operator+(zVector c1)
	{
		zVector out;
		out.m_zInstance =  &m_zInstance->operator+(*c1.m_zInstance);

		return out;
	}

}