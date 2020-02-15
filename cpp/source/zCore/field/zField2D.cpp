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

#include<headers/zCore/field/zField2D.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	template <typename T>
	ZSPACE_INLINE zField2D<T>::zField2D()
	{
		fieldValues.clear();
	}

	template <typename T>
	ZSPACE_INLINE zField2D<T>::zField2D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y)
	{
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		minBB = _minBB;
		maxBB = _maxBB;
		n_X = _n_X;
		n_Y = _n_Y;

		unit_X = (maxBB.x - minBB.x) / (n_X - 1);
		unit_Y = (maxBB.y - minBB.y) / (n_Y - 1);

		zVector unitVec = zVector(unit_X, unit_Y, 0);
		zVector startPt = minBB;

		fieldValues.clear();

		T defaultValue;
		fieldValues.assign(n_X*n_Y, defaultValue);

	}

	template <typename T>
	ZSPACE_INLINE zField2D<T>::zField2D(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB )
	{
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unit_X = _unit_X;
		unit_Y = _unit_Y;
		n_X = _n_X;
		n_Y = _n_Y;

		minBB = _minBB;
		maxBB = _minBB + zVector(unit_X * n_X, unit_Y * n_Y, 0);


		zVector unitVec = zVector(unit_X, unit_Y, 0);
		zVector startPt = _minBB;

		fieldValues.clear();

		T defaultValue;
		fieldValues.assign(n_X*n_Y, defaultValue);

	}

	//---- DESTRUCTOR

	template <typename T>
	ZSPACE_INLINE zField2D<T>::~zField2D() {}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zField2D<zVector>;

	template class zField2D<float>;

#endif
}