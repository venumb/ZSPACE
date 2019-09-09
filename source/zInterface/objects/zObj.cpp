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


#include<headers/zInterface/objects/zObj.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zObj::zObj()
	{
		showObject = true;
		showObjectTransform = false;

		transformationMatrix = zTransformationMatrix();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObj::~zObj() {}

	//---- VIRTUAL METHODS

	ZSPACE_INLINE void zObj::draw() {};

	ZSPACE_INLINE void zObj::getBounds(zVector &minBB, zVector &maxBB) {};

	//---- SET METHODS

	ZSPACE_INLINE void zObj::setShowObject(bool _showObject)
	{
		showObject = _showObject;
	}

	ZSPACE_INLINE void zObj::setShowTransform(bool _showObjectTransform)
	{
		showObjectTransform = _showObjectTransform;
	}

	ZSPACE_INLINE void zObj::setUtils(zUtilsDisplay &_displayUtils, zUtilsCore &_coreUtils)
	{
		displayUtils = &_displayUtils;
		coreUtils = &_coreUtils;
	}

}