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


#include<headers/zCore/base/zQuaternion.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zQuaternion::zQuaternion()
	{
		s = 0.0;
		v = zVector();
	}

	ZSPACE_INLINE zQuaternion::zQuaternion(float _s, float _vi, float _vj, float _vk)
	{
		s = _s;
		v = zVector(_vi, _vj, _vk);
	}

	ZSPACE_INLINE zQuaternion::zQuaternion(float _s, zVector _v)
	{
		s = _s;
		v = _v;
	}

	//---- OPERATORS

	ZSPACE_INLINE void zQuaternion::operator=(float _s)
	{
		s = _s;
		v = zVector();		
	}

	ZSPACE_INLINE void zQuaternion::operator=(zVector &_v)
	{
		s = 0.0;
		v = _v;
	}

	ZSPACE_INLINE float& zQuaternion::operator[](int index)
	{
		return (&s)[index];
	}

	ZSPACE_INLINE const float& zQuaternion::operator[](int index) const
	{
		return (&s)[index];
	}

	ZSPACE_INLINE zQuaternion zQuaternion::operator+(const zQuaternion  &q)
	{
		return zQuaternion(s + q.s, v + q.v);
	}

	ZSPACE_INLINE zQuaternion zQuaternion::operator-(const zQuaternion &q)
	{
		return zQuaternion(s - q.s, v - q.v);
	}

	ZSPACE_INLINE zQuaternion zQuaternion::operator*(float c)
	{
		return zQuaternion(s*c, v *c);
	}

	ZSPACE_INLINE zQuaternion zQuaternion::operator*(zQuaternion &q)
	{
		return zQuaternion(s*q.s - (v * q.v), ((q.v *s) + (v * q.s)) + (v^q.v));
	}

	ZSPACE_INLINE zQuaternion zQuaternion::operator/(float c)
	{
		return zQuaternion(s / c, v / c);
	}

	//---- OVERLOADED OPERATORS

	ZSPACE_INLINE void zQuaternion::operator +=(const zQuaternion &q)
	{
		s += q.s;
		v += q.v;
	}

	ZSPACE_INLINE void zQuaternion::operator -=(const zQuaternion &q)
	{
		s -= q.s;
		v -= q.v;
	}

	ZSPACE_INLINE void zQuaternion::operator*=(float c)
	{
		s *= c;
		v *= c;
	}

	ZSPACE_INLINE void zQuaternion::operator*=(zQuaternion q)
	{
		*this = (*this * q);
	}

	ZSPACE_INLINE void zQuaternion::operator/=(float c)
	{
		s /= c;
		v /= c;
	}

	//---- GET METHODS

	ZSPACE_INLINE float zQuaternion::getRe()
	{
		return s;
	}

	ZSPACE_INLINE zVector zQuaternion::getIm()
	{
		return v;
	}

	ZSPACE_INLINE zQuaternion zQuaternion::getConjugate()
	{
		return zQuaternion(s, v * -1);
	}

	ZSPACE_INLINE zQuaternion zQuaternion::getInverse()
	{
		return (this->getConjugate()) / this->length2();
	}

	//---- METHODS

	ZSPACE_INLINE float zQuaternion::length()
	{
		return sqrt(s*s + v.x*v.x + v.y*v.y + v.z*v.z);
	}

	ZSPACE_INLINE float zQuaternion::length2()
	{
		return ((s*s) + (v*v));
	}

	ZSPACE_INLINE void zQuaternion::normalize()
	{
		*this /= length2();
	}

	ZSPACE_INLINE zQuaternion zQuaternion::slerp(zQuaternion& q0, zQuaternion& q1, float t)
	{
		// interpolate length
		float m0 = q0.length();
		float m1 = q1.length();
		float m = (1 - t)*m0 + t * m1;

		// interpolate direction
		zQuaternion p0 = q0 / m0;
		zQuaternion p1 = q1 / m1;
		float theta = acos((p0.getConjugate()*p1).getRe());
		zQuaternion p = p0 * (sin((1 - t)*theta)) + (p1*sin(t*theta)) / sin(theta);

		return p * m;
	}

}