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

	zVector::zVector() : zManagedObj(new zSpace::zVector()) { }
	
	zVector::zVector(double _x, double _y, double _z) : zManagedObj(new zSpace::zVector(_x, _y, _z)) {}
	
	zVector::zVector(zSpace::zVector &v1) : zManagedObj(new zSpace::zVector())
	{
		m_zInstance->x = v1.x;
		m_zInstance->y = v1.y;
		m_zInstance->z = v1.z;
	}

	//---- OPERATOR

	bool zVector::operator==(zVector ^v1)
	{
		return m_zInstance->operator==(*v1->m_zInstance);
	}

	double zVector::operator[](int index)
	{
		return m_zInstance->operator[](index);
	}

	zVector^ zVector::operator+( zVector ^v1, const zVector ^v2)
	{
		return gcnew zVector(v1->m_zInstance->operator+(*v2->m_zInstance));
	}

	zVector^ zVector::operator-( zVector ^v1, const zVector ^v2)
	{	
		return gcnew zVector(v1->m_zInstance->operator-(*v2->m_zInstance));
	}

	double zVector::operator*( zVector ^v1, const zVector ^v2)
	{
		return v1->m_zInstance->operator*(*v2->m_zInstance);
	}

	zVector^ zVector::operator^( zVector ^v1, const zVector ^v2)
	{
		return gcnew zVector(v1->m_zInstance->operator^(*v2->m_zInstance));
	}

	zVector^ zVector::operator+( zVector ^v1, double val)
	{
		return gcnew zVector(v1->m_zInstance->operator+(val));
	}

	zVector^ zVector::operator-( zVector ^v1, double val)
	{
		return gcnew zVector(v1->m_zInstance->operator-(val));
	}

	zVector^ zVector::operator*( zVector ^v1, double val)
	{
		return gcnew zVector(v1->m_zInstance->operator*(val));
	}

	zVector^ zVector::operator/( zVector ^v1, double val)
	{
		return gcnew zVector(v1->m_zInstance->operator/(val));
	}

	//---- OVERLOADED OPERATOR

	zVector^ zVector::operator+=(const zVector ^v1)
	{
		m_zInstance->operator+=(*v1->m_zInstance);
		return this;
	}

	zVector^ zVector::operator-=(const zVector ^v1)
	{
		m_zInstance->operator-=(*v1->m_zInstance);
		return this;
	}

	zVector^ zVector::operator+=(double val)
	{
		m_zInstance->operator+=(val);
		return this;
	}

	zVector^ zVector::operator-=(double val)
	{
		m_zInstance->operator-=(val);
		return this;
	}

	zVector^ zVector::operator*=(double val)
	{
		m_zInstance->operator*=(val);
		return this;
	}

	zVector^ zVector::operator/=(double val)
	{
		m_zInstance->operator/=(val);
		return this;
	}
	
	//---- METHODS

	double zVector::length2()
	{
		return m_zInstance->length2();
	}

	double zVector::length()
	{
		return m_zInstance->length();
	}
	
	void zVector::normalize()
	{
		m_zInstance->normalize();
		
	}

	double zVector::squareDistanceTo(zVector ^v1)
	{
		return m_zInstance->squareDistanceTo(*v1->m_zInstance);
	}

	double zVector::distanceTo(zVector ^v1)
	{
		return m_zInstance->distanceTo(*v1->m_zInstance);
	}

	double zVector::angle(zVector ^v1)
	{
		return m_zInstance->angle(*v1->m_zInstance);
	}

	double zVector::angle360(zVector ^v1, zVector ^normal)
	{
		return m_zInstance->angle360(*v1->m_zInstance, *normal->m_zInstance);
	}

	double zVector::dihedralAngle(zVector ^v1, zVector ^v2)
	{
		return m_zInstance->dihedralAngle(*v1->m_zInstance, *v2->m_zInstance);
	}

	double zVector::cotan(zVector ^v1)
	{
		return m_zInstance->cotan(*v1->m_zInstance);
	}

	zVector^ zVector::rotateAboutAxis(zVector ^axisVec, double angle)
	{
		return gcnew zVector(m_zInstance->rotateAboutAxis(*axisVec->m_zInstance, angle));
	}

}