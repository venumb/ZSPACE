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

	zVector::zVector(const zVector ^& v1) : zManagedObj(new zSpace::zVector())
	{
		m_zInstance = v1->m_zInstance;			
	}


	//---- OPERATOR

	bool zVector::operator==(zVector ^v1)
	{
		return m_zInstance->operator==(*v1->m_zInstance);
	}

	double zVector:: operator[](int index)
	{
		return m_zInstance->operator[](index);
	}

	zVector^ zVector::operator+(const zVector ^v1, const zVector ^v2)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance =  &v1->m_zInstance->operator+(*v2->m_zInstance);		

		return out/*gcnew zVector(out->x, out->y, out->z)*/;
	}

	zVector^ zVector::operator-(const zVector ^v1, const zVector ^v2)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator-(*v2->m_zInstance);

		return out;
	}

	double zVector::operator*(const zVector ^v1, const zVector ^v2)
	{
		return v1->m_zInstance->operator*(*v2->m_zInstance);
	}

	zVector^ zVector::operator^(const zVector ^v1, const zVector ^v2)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator^(*v2->m_zInstance);

		return out;
	}

	zVector^ zVector::operator+(const zVector ^v1, double val)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator+(val);

		return out;
	}

	zVector^ zVector::operator-(const zVector ^v1, double val)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator-(val);

		return out;
	}

	zVector^ zVector::operator*(const zVector ^v1, double val)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator*(val);

		return out;
	}

	zVector^ zVector::operator/(const zVector ^v1, double val)
	{
		zVector^ out = gcnew zVector();
		out->m_zInstance = &v1->m_zInstance->operator/(val);

		return out;
	}

}