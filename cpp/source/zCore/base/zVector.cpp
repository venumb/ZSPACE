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


#include "headers/zCore/base/zVector.h"


namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zVector::zVector()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	ZSPACE_INLINE zVector::zVector(float _x, float _y, float _z, float _w)
	{
		x = _x;
		y = _y;
		z = _z;

	}

	ZSPACE_INLINE zVector::zVector(zFloat3 &_vals)
	{
		x = _vals[0];
		y = _vals[1];
		z = _vals[2];
	}

	ZSPACE_INLINE zVector::zVector(zFloat4 &_vals)
	{
		x = _vals[0];
		y = _vals[1];
		z = _vals[2];
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zVector::~zVector() {}

	//---- OPERATORS

	ZSPACE_INLINE bool zVector::operator==( zVector &v1)
	{		
		return (this->distanceTo(v1) < distanceTolerance);
	}

	ZSPACE_INLINE float  zVector::operator[](int index)
	{
		return 1.0/**vals[index]*/;
	}

	ZSPACE_INLINE zVector zVector::operator+(const zVector &v1)
	{
		return zVector(x + v1.x, y + v1.y, z + v1.z);
	}

	ZSPACE_INLINE zVector zVector::operator -(const zVector &v1)
	{
		return zVector(x - v1.x, y - v1.y, z - v1.z);
	}

	ZSPACE_INLINE float zVector::operator *(const zVector &v1)
	{
		return (x * v1.x + y * v1.y + z * v1.z);
	}

	ZSPACE_INLINE zVector zVector::operator ^(const zVector &v1)
	{
		return zVector(y * v1.z - z * v1.y, z*v1.x - x * v1.z, x*v1.y - y * v1.x);
	}

	ZSPACE_INLINE zVector zVector::operator +(float val)
	{
		return  zVector(x + val, y + val, z + val);
	}

	ZSPACE_INLINE zVector zVector::operator -(float val)
	{
		return  zVector(x - val, y - val, z - val);
	}

	ZSPACE_INLINE zVector zVector::operator*(float val)
	{
		return  zVector(x * val, y * val, z * val);
	}

	ZSPACE_INLINE zVector zVector::operator*(zMatrix3 & inMatrix)
	{
		zMatrix3Col inVecMatrix; 
		toColumnMatrix3(inVecMatrix);

		zMatrix3Col outVecMatrix;
		inMatrix.multiply(inVecMatrix, outVecMatrix);

		return fromColumnMatrix3(outVecMatrix);
	}

	ZSPACE_INLINE zVector zVector::operator*(zMatrix4 & inMatrix)
	{
		zMatrix4Col inVecMatrix;
		toColumnMatrix4(inVecMatrix);

		zMatrix4Col outVecMatrix;
		inMatrix.multiply(inVecMatrix, outVecMatrix);

		return fromColumnMatrix4(outVecMatrix);
	}

	ZSPACE_INLINE zVector zVector::operator/(float val)
	{
		if (val == 0)
		{

#ifndef __CUDACC__
			throw std::invalid_argument("val can't be zero");
#endif
			return zVector();
		}
		else return  zVector(x / val, y / val, z / val);
	}

	//---- OVERLOADED OPERATORS

	ZSPACE_INLINE void zVector::operator +=(const zVector &v1)
	{
		x += v1.x;
		y += v1.y;
		z += v1.z;

	}

	ZSPACE_INLINE void zVector::operator -=(const zVector &v1)
	{
		x -= v1.x;
		y -= v1.y;
		z -= v1.z;

	}

	ZSPACE_INLINE void zVector::operator +=(float val)
	{
		x += val;
		y += val;
		z += val;
	}

	ZSPACE_INLINE void zVector::operator -=(float val)
	{
		x -= val;
		y -= val;
		z -= val;
	}

	ZSPACE_INLINE void zVector::operator *=(float val)
	{
		x *= val;
		y *= val;
		z *= val;

	}

	ZSPACE_INLINE void zVector::operator /=(float val)
	{
		if (abs(x) >= EPS) x /= val;
		if (abs(y) >= EPS) y /= val;
		if (abs(z) >= EPS) z /= val;

	}

	//---- METHODS

	ZSPACE_INLINE float zVector::length2()
	{
		return (x*x + y * y + z * z);
	}

	ZSPACE_INLINE float zVector::length()
	{
		return sqrt(x*x + y * y + z * z);
	}

	ZSPACE_INLINE void zVector::normalize()
	{
		float length = this->length();

		x /= length;
		y /= length;
		z /= length;
	}
	

	ZSPACE_INLINE float zVector::squareDistanceTo(zVector &v1)
	{
		return (this->operator- (v1)) * (this->operator- (v1));
	}

	ZSPACE_INLINE float zVector::distanceTo(zVector &v1)
	{
		return sqrt(squareDistanceTo(v1));
	}

	ZSPACE_INLINE float zVector::angle(zVector &v1)
	{
		// check if they are parallel
		zVector a(x, y, z);
		zVector b(v1.x, v1.y, v1.z);

		a.normalize();
		b.normalize();

		if (a*b == 1) return 0;
		else if (a*b == -1) return 180;
		else
		{

			float dotProduct = a * b;
			float angle = acos(dotProduct);

			return angle * RAD_TO_DEG;
		}

	}

	ZSPACE_INLINE float zVector::angle360(zVector &v1, zVector &normal)
	{
		// check if they are parallel
		zVector a(x, y, z);
		zVector b(v1.x, v1.y, v1.z);

		a.normalize();
		b.normalize();

		if (a*b == 1) return 0;
		else if (a*b == -1) return 180;
		else
		{
			float length = this->length();
			float length1 = v1.length();

			float dot = a * b;

			zVector cross = a ^ b;
			float det = normal * (cross);

			float angle = atan2(det, dot);
			if (angle < 0) angle += TWO_PI;


			return angle * RAD_TO_DEG;
		}



	}

	ZSPACE_INLINE float zVector::dihedralAngle(zVector &v1, zVector &v2)
	{
		zVector e(x, y, z);

		v1.normalize();
		v2.normalize();
		float dot = v1 * v2;

		zVector cross = v1 ^ v2;
		float  dtheta = atan2(e * cross, dot);

		return(dtheta * (180.0 / PI));
	}

	ZSPACE_INLINE float zVector::cotan(zVector &v)
	{
		zVector u(x, y, z);

		float dot = u * v;

		//using http://multires.caltech.edu/pubs/diffGeoOps.pdf
		//float denom = (u*u) * (v*v) - (dot*dot);	
		//if (denom == 0) return 0.0;
		//else return dot / sqrt(denom);

		//using http://rodolphe-vaillant.fr/?e=69
		float denom = (u ^ v).length();

		if (denom == 0) return 0.0;
		else return dot / denom;
	}

	ZSPACE_INLINE void zVector::getComponents(zFloat4 &_vals)
	{
		_vals[0] = x;
		_vals[1] = y;
		_vals[2] = z;
		_vals[3] = 1;
	}

	ZSPACE_INLINE zVector zVector::rotateAboutAxis(zVector axisVec, float angle)
	{
		axisVec.normalize();

		float theta = DEG_TO_RAD * angle;

		zMatrix3 rotationMatrix; ;

		rotationMatrix(0, 0) = cos(theta) + (axisVec.x * axisVec.x) * (1 - cos(theta));
		rotationMatrix(1, 0) = (axisVec.x * axisVec.y) * (1 - cos(theta)) - (axisVec.z) * (sin(theta));
		rotationMatrix(2, 0) = (axisVec.x * axisVec.z) * (1 - cos(theta)) + (axisVec.y) * (sin(theta));

		rotationMatrix(0, 1) = (axisVec.x * axisVec.y) * (1 - cos(theta)) + (axisVec.z) * (sin(theta));
		rotationMatrix(1, 1) = cos(theta) + (axisVec.y * axisVec.y) * (1 - cos(theta));
		rotationMatrix(2, 1) = (axisVec.y * axisVec.z) * (1 - cos(theta)) - (axisVec.x) * (sin(theta));

		rotationMatrix(0, 2) = (axisVec.x * axisVec.z) * (1 - cos(theta)) - (axisVec.y) * (sin(theta));
		rotationMatrix(1, 2) = (axisVec.z * axisVec.y) * (1 - cos(theta)) + (axisVec.x) * (sin(theta));
		rotationMatrix(2, 2) = cos(theta) + (axisVec.z * axisVec.z) * (1 - cos(theta));


		zVector out = this->operator* (rotationMatrix);

		return out;
	}

	ZSPACE_INLINE void zVector::toRowMatrix4(zMatrix4Row &row)
	{
		row[0] = x; row[1] = y; row[2] = z; row[3] = 1;
	}

	ZSPACE_INLINE void zVector::toRowMatrix3(zMatrix3Row &row)
	{
		row[0] = x; row[1] = y; row[2] = z;
	}

	ZSPACE_INLINE void zVector::toColumnMatrix4(zMatrix4Col &col)
	{
		col[0] = x; col[1] = y; col[2] = z; col[3] = 1;
	}

	ZSPACE_INLINE void zVector::toColumnMatrix3(zMatrix3Col &col)
	{
		col[0] = x; col[1] = y; col[2] = z;
	}

	ZSPACE_INLINE zVector zVector::fromRowMatrix4(zMatrix4Row &row)
	{
		return zVector(row[0], row[1], row[2], row[3]);
	}

	ZSPACE_INLINE zVector zVector::fromRowMatrix3(zMatrix3Row &row)
	{
		return zVector(row[0], row[1], row[2]);
	}

	ZSPACE_INLINE zVector zVector::fromColumnMatrix4(zMatrix4Col &col)
	{
		return zVector(col[0], col[1], col[2], col[3]);
	}

	ZSPACE_INLINE zVector zVector::fromColumnMatrix3(zMatrix3Col &col)
	{
		return zVector(col[0], col[1], col[2]);
	}
	


#ifndef __CUDACC__


	ZSPACE_INLINE zVector zVector::operator*(zTransform inTrans)
	{
		Vector4f p(x, y, z, 1);

		Vector4f newP = inTrans * p;

		zVector out(newP(0), newP(1), newP(2), newP(3));

		return out;
	}


	//---- STREAM OPERATORS

	ZSPACE_INLINE ostream & operator<<(ostream & os, const zVector & vec)
	{
		os << " [ " << vec.x << ',' << vec.y << ',' << vec.z << " ]";
		return os;
	}

#endif

}



