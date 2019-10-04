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


#include<headers/zInterface/objects/zObjParticle.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjParticle::zObjParticle()
	{
		displayUtils = nullptr;

		showForces = false;

		forceScale = 1.0;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjParticle::~zObjParticle() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjParticle::setShowElements(bool _showForces, double _forceScale)
	{
		showForces = _showForces;
		forceScale = _forceScale;
	}

	//---- OVERRIDE METHODS

#ifndef ZSPACE_OPENGL

	ZSPACE_INLINE void zObjParticle::draw()
	{
		if (showObject)
		{
			if (showForces) drawForces();
		}

		if (showObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zObjParticle::drawForces()
	{
		zVector p = *particle.s.p;
		zVector p1 = p + particle.f;

		displayUtils->drawPoint(p1);

		displayUtils->drawLine(p, p1, zColor(0, 1, 0, 1), 1.0);

	}

#endif // !ZSPACE_OPENGL
}