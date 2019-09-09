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


#include<headers/zCore/dynamics/zParticle.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zParticle::zParticle()
	{
		m = 1.0;
		fixed = false;

		s.p = nullptr;
	}

	ZSPACE_INLINE zParticle::zParticle(zVector &_p, bool _fixed, double _m, zVector _v, zVector _f)
	{
		s.p = &_p;
		s.v = _v;
		f = _f;
		m = _m;

		fixed = _fixed;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zParticle::~zParticle() {}
}