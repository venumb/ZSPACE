// This file is part of zspace, a simple C++ collection of geometr_y data-structures & algorithms, 
// data anal_ysis & visualization framework.
//
// Cop_yright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a cop_y of the MIT License was not distributed with this file, _you can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//


#include<headers/zArchGeom/zAgObj.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgObj::zAgObj(){ }


	//---- DESTRUCTOR

	ZSPACE_INLINE zAgObj::~zAgObj() {}

	//---- CREATE METHODS


	ZSPACE_INLINE void zAgObj::createByType(zStructureType&_structureType)
	{
		structureType = _structureType;

		if (structureType == zStructureType::zRHWC) createRhwc(); 
		else if (structureType == zStructureType::zDigitalTimber) createTimber();
	}

	ZSPACE_INLINE void zAgObj::createRhwc(){}

	ZSPACE_INLINE void zAgObj::createTimber(){}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else
	   
	ZSPACE_INLINE void zAgObj::setModel(zModel & _model)
	{
		model = &_model;
	}

#endif

}