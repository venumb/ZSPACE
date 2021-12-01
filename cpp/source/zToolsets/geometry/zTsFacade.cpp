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


#include<headers/zToolsets/geometry/zTsFacade.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsFacade::zTsFacade(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsFacade::~zTsFacade() {}

	//---- CREATE METHODS


	//--- SET METHODS 

	ZSPACE_INLINE void zTsFacade::setGuideMesh(zObjMesh& _o_guideMesh)
	{
		o_guideMesh = &_o_guideMesh;		
	}

	//---- GET METHODS

	
	//---- COMPUTE METHODS
		

	//---- UTILITY METHODS
	


}