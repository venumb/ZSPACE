// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zInterface/objects/zObjPointField.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	template<typename T>
	ZSPACE_INLINE zObjPointField<T>::zObjPointField()
	{
#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayVertices = false;
	}

	//---- DESTRUCTOR

	template<typename T>
	ZSPACE_INLINE zObjPointField<T>::~zObjPointField() {}

	//---- OVERRIDE METHODS

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
	template<typename T>
	ZSPACE_INLINE void zObjPointField<T>::draw()
	{
		zObjPointCloud::draw();
	}

#endif


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zObjPointField<zVector>;

	template class zObjPointField<zScalar>;

#endif
}