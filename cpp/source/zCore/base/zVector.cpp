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


#include<headers/zCore/base/zVector.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zVector::zVector()
	{
		x = 0;
		y = 0;
		z = 0;

		vals[0] = &x;
		vals[1] = &y;
		vals[2] = &z;
	}

	ZSPACE_INLINE zVector::zVector(double _x, double _y, double _z)
	{

		x = _x;
		y = _y;
		z = _z;

		vals[0] = &x;
		vals[1] = &y;
		vals[2] = &z;

	}

	ZSPACE_INLINE zVector::zVector(zDouble3 &_vals)
	{

		x = _vals[0];
		y = _vals[1];
		z = _vals[2];

		vals[0] = &x;
		vals[1] = &y;
		vals[2] = &z;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zVector::~zVector() {}

	//---- OPERATORS

	ZSPACE_INLINE bool zVector::operator==(const zVector &v1)
	{
		bool out = false;
		if (x == v1.x && y == v1.y && z == v1.z) out = true;

		return out;
	}

	ZSPACE_INLINE double  zVector::operator[](int index)
	{
		if (index >= 0 && index <= 2)return *vals[index];
	}

	ZSPACE_INLINE zVector zVector::operator+(const zVector &v1)
	{
		return zVector(x + v1.x, y + v1.y, z + v1.z);
	}

	ZSPACE_INLINE zVector zVector::operator -(const zVector &v1)
	{
		return zVector(x - v1.x, y - v1.y, z - v1.z);
	}

	ZSPACE_INLINE double zVector::operator *(const zVector &v1)
	{
		return (x * v1.x + y * v1.y + z * v1.z);
	}

	ZSPACE_INLINE zVector zVector::operator ^(const zVector &v1)
	{
		return zVector(y * v1.z - z * v1.y, z*v1.x - x * v1.z, x*v1.y - y * v1.x);
	}

	ZSPACE_INLINE zVector zVector::operator +(double val)
	{
		return  zVector(x + val, y + val, z + val);
	}

	ZSPACE_INLINE zVector zVector::operator -(double val)
	{
		return  zVector(x - val, y - val, z - val);
	}

	ZSPACE_INLINE zVector zVector::operator*(double val)
	{
		return  zVector(x * val, y * val, z * val);
	}

	ZSPACE_INLINE zVector zVector::operator*(zMatrixd inMatrix)
	{
		if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
		if (inMatrix.getNumCols() < 3 || inMatrix.getNumCols() > 4) 	throw std::invalid_argument("input Matrix is not a 3X3 or 4X4 matrix.");

		zMatrixd vecMatrix = this->toColumnMatrix(inMatrix.getNumCols());

		zMatrixd outVecMatrix = inMatrix * vecMatrix;

		return this->fromColumnMatrix(outVecMatrix);
	}

	ZSPACE_INLINE zVector zVector::operator*(zTransform inTrans)
	{
		Vector4d p(x, y, z, 1);

		Vector4d newP = inTrans * p;

		zVector out(newP(0), newP(1), newP(2));

		return out;
	}

	ZSPACE_INLINE zVector zVector::operator/(double val)
	{
		if (val == 0)
			throw std::invalid_argument("val can't be zero");

		return  zVector(x / val, y / val, z / val);
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

	ZSPACE_INLINE void zVector::operator +=(double val)
	{
		x += val;
		y += val;
		z += val;
	}

	ZSPACE_INLINE void zVector::operator -=(double val)
	{
		x -= val;
		y -= val;
		z -= val;
	}

	ZSPACE_INLINE void zVector::operator *=(double val)
	{
		x *= val;
		y *= val;
		z *= val;

	}

	ZSPACE_INLINE void zVector::operator /=(double val)
	{
		if (abs(x) >= EPS) x /= val;
		if (abs(y) >= EPS) y /= val;
		if (abs(z) >= EPS) z /= val;

	}

	//---- METHODS

	ZSPACE_INLINE double zVector::length2()
	{
		return (x*x + y * y + z * z);
	}

	ZSPACE_INLINE double zVector::length()
	{
		return sqrt(x*x + y * y + z * z);
	}

	ZSPACE_INLINE void zVector::normalize()
	{
		double length = this->length();

		x /= length;
		y /= length;
		z /= length;
	}

	ZSPACE_INLINE double zVector::squareDistanceTo(zVector &v1)
	{
		return (this->operator- (v1)) * (this->operator- (v1));
	}

	ZSPACE_INLINE double zVector::distanceTo(zVector &v1)
	{
		return sqrt(squareDistanceTo(v1));
	}

	ZSPACE_INLINE double zVector::angle(zVector &v1)
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

			double dotProduct = a * b;
			double angle = acos(dotProduct);

			return angle * RAD_TO_DEG;
		}

	}

	ZSPACE_INLINE double zVector::angle360(zVector &v1, zVector &normal)
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
			double length = this->length();
			double length1 = v1.length();

			double dot = a * b;

			zVector cross = a ^ b;
			double det = normal * (cross);

			double angle = atan2(det, dot);
			if (angle < 0) angle += TWO_PI;


			return angle * RAD_TO_DEG;
		}



	}

	ZSPACE_INLINE double zVector::dihedralAngle(zVector &v1, zVector &v2)
	{
		zVector e(x, y, z);

		v1.normalize();
		v2.normalize();
		double dot = v1 * v2;

		zVector cross = v1 ^ v2;
		double  dtheta = atan2(e * cross, dot);

		return(dtheta * (180.0 / PI));
	}

	ZSPACE_INLINE double zVector::cotan(zVector &v)
	{
		zVector u(x, y, z);

		double dot = u * v;

		//using http://multires.caltech.edu/pubs/diffGeoOps.pdf
		//double denom = (u*u) * (v*v) - (dot*dot);	
		//if (denom == 0) return 0.0;
		//else return dot / sqrt(denom);

		//using http://rodolphe-vaillant.fr/?e=69
		double denom = (u ^ v).length();

		if (denom == 0) return 0.0;
		else return dot / denom;
	}

	ZSPACE_INLINE void zVector::getComponents(zDouble3 &_vals)
	{
		_vals[0] = x;
		_vals[1] = y;
		_vals[2] = z;
	}

	ZSPACE_INLINE double* zVector::getRawComponents()
	{
		return *vals;
	}

	ZSPACE_INLINE zMatrixd zVector::toRowMatrix(int cols)
	{
		vector<double> vals;

		if (cols == 4) vals = { x,y,z,1 };
		if (cols == 3) vals = { x,y,z };

		return zMatrixd(1, cols, vals);
	}

	ZSPACE_INLINE zMatrixd zVector::toColumnMatrix(int rows)
	{
		vector<double> vals = { x,y,z,1 };

		if (rows == 4) vals = { x,y,z,1 };
		if (rows == 3) vals = { x,y,z };
		return zMatrixd(rows, 1, vals);
	}

	ZSPACE_INLINE zVector zVector::fromRowMatrix(zMatrixd &inMatrix)
	{
		if (inMatrix.getNumRows() != 1) throw std::invalid_argument("input Matrix is not a row matrix.");
		if (inMatrix.getNumCols() < 3 || inMatrix.getNumCols() > 4) throw std::invalid_argument("cannot convert row matrix to vector.");

		return zVector(inMatrix(0, 0), inMatrix(0, 1), inMatrix(0, 2));
	}

	ZSPACE_INLINE zVector zVector::fromColumnMatrix(zMatrixd &inMatrix)
	{
		if (inMatrix.getNumCols() != 1) throw std::invalid_argument("input Matrix is not a column matrix.");
		if (inMatrix.getNumRows() < 3 || inMatrix.getNumRows() > 4) throw std::invalid_argument("cannot convert column matrix to vector.");

		return zVector(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
	}

	ZSPACE_INLINE zVector zVector::rotateAboutAxis(zVector axisVec, double angle)
	{
		axisVec.normalize();

		double theta = DEG_TO_RAD * angle;

		zMatrixd rotationMatrix = zMatrixd(3, 3);

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

	//---- STREAM OPERATORS

	ZSPACE_INLINE ostream & operator<<(ostream & os, const zVector & vec)
	{
		os << "\n" <<vec.x << ',' << vec.y << ',' << vec.z;
		return os;
	}

}