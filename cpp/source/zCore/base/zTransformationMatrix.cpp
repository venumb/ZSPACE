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


#include<headers/zCore/base/zTransformationMatrix.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTransformationMatrix::zTransformationMatrix()
	{

		Transform.setIdentity();

		P.setIdentity();

		S.setIdentity();

		R.setIdentity();

		T.setIdentity();

		pivot[0] = pivot[1] = pivot[2] = 0;
		computeP();

		rotation[0] = rotation[1] = rotation[2] = 0;
		computeRx();
		computeRy();
		computeRz();
		computeR();

		scale[0] = scale[1] = scale[2] = 1;
		computeS();

		translation[0] = translation[1] = translation[2] = 0;
		computeT();

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTransformationMatrix::~zTransformationMatrix()
	{
	}

	//---- SET METHODS

	ZSPACE_INLINE void zTransformationMatrix::setTransform(zTransform &inTransform, bool decompose)
	{
		if (decompose)
		{
			// set scale
			S(0, 0) = zVector(inTransform(0, 0), inTransform(0, 1), inTransform(0, 2)).length(); 
			S(1, 1) = zVector(inTransform(1, 0), inTransform(1, 1), inTransform(1, 2)).length();
			S(2, 2) = zVector(inTransform(2, 0), inTransform(2, 1), inTransform(2, 2)).length();

			// set translation
			T(3, 0) = inTransform(3, 0); T(3, 1) = inTransform(3, 1); T(3, 2) = inTransform(3, 2);

			// set rotation 
			R(0, 0) = inTransform(0, 0); R(0, 1) = inTransform(0, 1); R(0, 2) = inTransform(0, 2);
			R(1, 0) = inTransform(1, 0); R(1, 1) = inTransform(1, 1); R(1, 2) = inTransform(1, 2);
			R(2, 0) = inTransform(2, 0); R(2, 1) = inTransform(2, 1); R(2, 2) = inTransform(2, 2);

			// compute components
			decomposeR();
			decomposeT();
			decomposeS();
		}

		Transform = inTransform;

	}

	ZSPACE_INLINE void zTransformationMatrix::setRotation(zFloat4 &_rotation, bool addValues)
	{
		if (addValues)
		{
			rotation[0] += DEG_TO_RAD * _rotation[0];
			rotation[1] += DEG_TO_RAD * _rotation[1];
			rotation[2] += DEG_TO_RAD * _rotation[2];
			rotation[3] = _rotation[3];
		}
		else
		{
			rotation[0] = DEG_TO_RAD * _rotation[0];
			rotation[1] = DEG_TO_RAD * _rotation[1];
			rotation[2] = DEG_TO_RAD * _rotation[2];
			rotation[3] = _rotation[3];
		}



		computeRx();
		computeRy();
		computeRz();

		computeR();
		computeTransform();
	}

	ZSPACE_INLINE void zTransformationMatrix::setScale(zFloat4 &_scale)
	{
		scale[0] = _scale[0];
		if (scale[0] == 0)scale[0] = scaleZero;

		scale[1] = _scale[1];
		if (scale[1] == 0)scale[1] = scaleZero;

		scale[2] = _scale[2];
		if (scale[2] == 0)scale[2] = scaleZero;

		scale[3] = _scale[3];
		if (scale[3] == 0)scale[3] = _scale[3];

		computeS();

	}

	ZSPACE_INLINE void zTransformationMatrix::setTranslation(zFloat4 &_translation, bool addValues)
	{
		if (addValues)
		{
			translation[0] += _translation[0];
			translation[1] += _translation[1];
			translation[2] += _translation[2];
			translation[3] = _translation[3];
		}
		else
		{
			translation[0] = _translation[0];
			translation[1] = _translation[1];
			translation[2] = _translation[2];
			translation[3] = _translation[3];
		}

		computeT();
		computeTransform();
	}

	ZSPACE_INLINE void zTransformationMatrix::setPivot(zFloat4 &_pivot)
	{
		pivot[0] = _pivot[0];
		pivot[1] = _pivot[1];
		pivot[2] = _pivot[2];
		pivot[3] = _pivot[3];

		computeP();

	}

	//---- GET METHODS

	ZSPACE_INLINE zVector zTransformationMatrix::getX()
	{
		zVector x = zVector(Transform(0, 0), Transform(0, 1), Transform(0, 2));
		x.normalize();
		return x;
	}

	ZSPACE_INLINE zVector zTransformationMatrix::getY()
	{
		zVector y = zVector(Transform(1, 0), Transform(1, 1), Transform(1, 2));;
		y.normalize();
		return y;
	}

	ZSPACE_INLINE zVector zTransformationMatrix::getZ()
	{
		zVector z = zVector(Transform(2, 0), Transform(2, 1), Transform(2, 2));
		z.normalize();
		return z;
	}

	ZSPACE_INLINE zVector zTransformationMatrix::getO()
	{
		return zVector(Transform(3, 0), Transform(3, 1), Transform(3, 2));
	}

	ZSPACE_INLINE zVector zTransformationMatrix::getTranslation()
	{
		return translation;
	}

	ZSPACE_INLINE zVector zTransformationMatrix::getPivot()
	{
		return pivot;
	}

	ZSPACE_INLINE float* zTransformationMatrix::getRawPivot()
	{
		return &pivot[0];
	}

	ZSPACE_INLINE void zTransformationMatrix::getRotation(zFloat4 &_rotation)
	{
		_rotation[0] = RAD_TO_DEG * rotation[0];
		_rotation[1] = RAD_TO_DEG * rotation[1];
		_rotation[2] = RAD_TO_DEG * rotation[2];
		_rotation[3] = rotation[3];

	}

	ZSPACE_INLINE void zTransformationMatrix::getScale(zFloat4 &_scale)
	{
		_scale[0] = scale[0];
		if (scale[0] == scaleZero)_scale[0] = 0;

		_scale[1] = scale[1];
		if (scale[1] == scaleZero)_scale[1] = 0;

		_scale[2] = scale[2];
		if (scale[2] == scaleZero)_scale[2] = 0;

		_scale[2] = scale[2];
		if (scale[2] == scaleZero)_scale[2] = 1;
	}

	//---- AS MATRIX METHODS

	ZSPACE_INLINE zTransform zTransformationMatrix::asMatrix()
	{
		return Transform;
	}

	ZSPACE_INLINE float* zTransformationMatrix::asRawMatrix()
	{
#ifndef __CUDACC__
		return Transform.data();
#else

		return Transform.getRawMatrixValues();
#endif
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asInverseMatrix()
	{
#ifndef __CUDACC__
		return Transform.inverse();
#else
		zTransform out;
		Transform.inverse(out);
		return out;
#endif
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asScaleMatrix()
	{
		return S;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asRotationMatrix()
	{
		return R;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asPivotMatrix()
	{
		return P;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asPivotTranslationMatrix()
	{
		zTransform out;
		out.setIdentity();


		out(0, 3) = -1 * pivot[0];
		out(1, 3) = -1 * pivot[1];
		out(2, 3) = -1 * pivot[2];


		return out;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asInversePivotTranslationMatrix()
	{

#ifndef __CUDACC__
		return asPivotTranslationMatrix().inverse();
#else
		zTransform out;
		asPivotTranslationMatrix().inverse(out);
		return out;
#endif

	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asScaleTransformMatrix()
	{
		zTransform out;
		out.setIdentity();

		out(0, 0) = scale[0];
		out(1, 1) = scale[1];
		out(2, 2) = scale[2];

		out(0, 3) = (1 - scale[0]) * pivot[0];
		out(1, 3) = (1 - scale[1]) * pivot[1];
		out(2, 3) = (1 - scale[2]) * pivot[2];


		return out;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::asInverseScaleTransformMatrix()
	{
#ifndef __CUDACC__
		return asScaleTransformMatrix().inverse();
#else
		zTransform out;
		asScaleTransformMatrix().inverse(out);
		return out;
#endif

	}

	//---- GET MATRIX METHODS

	ZSPACE_INLINE zTransform zTransformationMatrix::getWorldMatrix()
	{

		zTransform outMatrix;
		outMatrix.setIdentity();

		zVector X = getX();
		zVector Y = getY();
		zVector Z = getZ();
		zVector O = getO();


		outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
		outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
		outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = O.x; outMatrix(1, 3) = O.y; outMatrix(2, 3) = O.z;

		return outMatrix;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::getLocalMatrix()
	{
		zTransform outMatrix;
		outMatrix.setIdentity();

		zVector X = getX();
		zVector Y = getY();
		zVector Z = getZ();
		zVector O = getO();

		zVector orig(0, 0, 0);
		zVector d = O - orig;

		outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
		outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
		outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);

		return outMatrix;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::getToMatrix(zTransformationMatrix &to)
	{
		zTransform world = to.getWorldMatrix();
		zTransform local = this->getLocalMatrix();



		return world * local;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::getBasisChangeMatrix(zTransformationMatrix &to)
	{
		zTransform local = to.getLocalMatrix();
		zTransform world = this->getWorldMatrix();

		return local * world;
	}

	ZSPACE_INLINE zTransform zTransformationMatrix::getTargetMatrix(zTransform &target)
	{
#ifndef __CUDACC__
		zTransform C_inverse = Transform.inverse();
#else
		zTransform C_inverse;
		Transform.inverse(C_inverse);
#endif
				

		zTransform targ_newbasis;
		targ_newbasis.setIdentity();

		targ_newbasis = C_inverse * target;

		return targ_newbasis;
	}

	//---- PROTECTED METHODS

	ZSPACE_INLINE void zTransformationMatrix::computeTransform()
	{
		Transform.setIdentity();

		zTransform PS = P * S;

#ifndef __CUDACC__
		zTransform PSInverse = PS.inverse();
#else
		zTransform PSInverse;
		Transform.inverse(PSInverse);
#endif

		Transform = PS * PSInverse * R * T;

	}

	ZSPACE_INLINE void zTransformationMatrix::computeR()
	{
		R.setIdentity();
		R = Rx * Ry *Rz;
	}

	ZSPACE_INLINE void zTransformationMatrix::computeRx()
	{
		Rx.setIdentity();

		Rx(1, 1) = cos(rotation[0]);
		Rx(1, 2) = sin(rotation[0]);

		Rx(2, 1) = -sin(rotation[0]);
		Rx(2, 2) = cos(rotation[0]);

	}

	ZSPACE_INLINE void zTransformationMatrix::computeRy()
	{
		Ry.setIdentity();

		Ry(0, 0) = cos(rotation[1]);
		Ry(0, 2) = -sin(rotation[1]);

		Ry(2, 0) = sin(rotation[1]);
		Ry(2, 2) = cos(rotation[1]);

	}

	ZSPACE_INLINE void zTransformationMatrix::computeRz()
	{
		Rz.setIdentity();

		Rz(0, 0) = cos(rotation[2]);
		Rz(0, 1) = sin(rotation[2]);

		Rz(1, 0) = -sin(rotation[2]);
		Rz(1, 1) = cos(rotation[2]);
	}

	ZSPACE_INLINE void zTransformationMatrix::computeT()
	{
		T.setIdentity();

		T(3, 0) = translation[0];
		T(3, 1) = translation[1];
		T(3, 2) = translation[2];
	}

	ZSPACE_INLINE void zTransformationMatrix::computeP()
	{
		P.setIdentity();

		P(3, 0) = pivot[0];
		P(3, 1) = pivot[1];
		P(3, 2) = pivot[2];
	}

	ZSPACE_INLINE void zTransformationMatrix::computeS()
	{
		S.setIdentity();

		S(0, 0) = scale[0];
		S(1, 1) = scale[1];
		S(2, 2) = scale[2];
	}

	ZSPACE_INLINE void zTransformationMatrix::decomposeR()
	{
		zFloat3 rot0;
		zFloat3 rot1;

		if (R(0, 2) != 1 && R(0, 2) != -1)
		{
			rot0[1] = asin(R(0, 2)) * -1;
			rot1[1] = PI - rot0[1];

			rot0[0] = atan2(R(1, 2) / cos(rot0[1]), R(2, 2) / cos(rot0[1]));
			rot1[1] = atan2(R(1, 2) / cos(rot1[1]), R(2, 2) / cos(rot1[1]));

			rot0[2] = atan2(R(0, 1) / cos(rot0[1]), R(0, 0) / cos(rot0[1]));
			rot1[2] = atan2(R(0, 1) / cos(rot1[1]), R(0, 0) / cos(rot1[1]));

		}
		else
		{
			rot0[2] = rot1[2] = 0;

			if (R(0, 2) == -1)
			{
				rot0[1] = rot1[1] = HALF_PI;
				rot0[0] = rot1[0] = atan2(R(1, 0), R(2, 0));
			}
			else
			{
				rot0[1] = rot1[1] = HALF_PI * -1;
				rot0[0] = rot1[0] = atan2(R(1, 0)* -1, R(2, 0)* -1);
			}
		}

		rotation[0] = rot0[0];
		rotation[1] = rot0[1];
		rotation[2] = rot0[2];

		computeRx(); computeRy(); computeRz();
	}

	ZSPACE_INLINE void zTransformationMatrix::decomposeT()
	{
		translation[0] = T(3, 0);
		translation[1] = T(3, 1);
		translation[2] = T(3, 2);
	}

	ZSPACE_INLINE void zTransformationMatrix::decomposeS()
	{
		scale[0] = S(0, 0);
		scale[1] = S(1, 1);
		scale[2] = S(2, 2);
	}

}