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


#include<headers/zInterface/objects/zObj.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zObj::zObj()
	{
		displayObject = true;
		displayObjectTransform = false;

		transformationMatrix = zTransformationMatrix();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObj::~zObj() {}

	//---- VIRTUAL METHODS

	ZSPACE_INLINE void zObj::draw() {};

	ZSPACE_INLINE void zObj::getBounds(zPoint &minBB, zPoint &maxBB) {};

	//---- SET METHODS

	ZSPACE_INLINE void zObj::setDisplayObject(bool _displayObject)
	{
		displayObject = _displayObject;
	}

	ZSPACE_INLINE void zObj::setDisplayTransform(bool _displayObjectTransform)
	{
		displayObjectTransform = _displayObjectTransform;
	}

	//---- GET METHODS

	ZSPACE_INLINE bool zObj::getDisplayObject()
	{
		return displayObject;
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zObj::setUtils(zUtilsDisplay &_displayUtils)
	{
		displayUtils = &_displayUtils;
	}

#endif 

}