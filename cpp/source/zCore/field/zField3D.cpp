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

#include<headers/zCore/field/zField3D.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	template <typename T>
	ZSPACE_INLINE zField3D<T>::zField3D()
	{
		fieldValues.clear();
	}

	template <typename T>
	ZSPACE_INLINE zField3D<T>::zField3D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z)
	{
		minBB = _minBB;
		maxBB = _maxBB;

		n_X = _n_X;
		n_Y = _n_Y;
		n_Z = _n_Z;

		unit_X = (maxBB.x - minBB.x) / n_X;
		unit_Y = (maxBB.y - minBB.y) / n_Y;
		unit_Z = (maxBB.z - minBB.z) / n_Z;

		zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
		zVector startPt = minBB;

		fieldValues.clear();

		printf("unit_X : %1.2f unit_Y : %1.2f unit_Z : %1.2f ", unit_X, unit_Y, unit_Z);

		for (int i = 0; i < n_X; i++)
		{
			for (int j = 0; j < n_Y; j++)
			{

				for (int k = 0; k < n_Z; k++)
				{
					T defaultValue;
					fieldValues.push_back(defaultValue);
				}

			}
		}

	}

	template <typename T>
	ZSPACE_INLINE zField3D<T>::zField3D(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zVector _minBB )
	{
		unit_X = _unit_X;
		unit_Y = _unit_Y;
		unit_Z = _unit_Z;

		n_X = _n_X;
		n_Y = _n_Y;
		n_Z = _n_Z;

		maxBB = minBB + zVector(unit_X * n_X, unit_Y * n_Y, unit_Z * n_Z);

		zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
		zVector startPt = minBB;


		for (int i = 0; i < n_X; i++)
		{
			for (int j = 0; j < n_Y; j++)
			{

				for (int k = 0; k < n_Z; k++)
				{

					T defaultValue;
					fieldValues.push_back(defaultValue);
				}
			}
		}


	}

	//---- DESTRUCTOR

	template <typename T>
	ZSPACE_INLINE zField3D<T>::~zField3D() {}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zField3D<zVector>;

	template class zField3D<float>;

#endif
}