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


#include<headers/zHousing/architecture/zHcAggregation.h>

namespace zSpace
{
	//---- CONSTRUCTORS

	ZSPACE_INLINE zHcAggregation::zHcAggregation(){}


	//---- DESTRUCTOR

	ZSPACE_INLINE zHcAggregation::~zHcAggregation() {}


	//---- SET METHODS

	ZSPACE_INLINE void zHcAggregation::createHousingUnits(zStructureType&_structureType)
	{
		for (int i = 0; i < inMeshObjs.size(); i++)
		{
			zFunctionType funcType;

			zItMeshVertex v(inMeshObjs[i]);
			if (v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0) funcType = zFunctionType::zPublic;
			else if (v.getColor().r == 0 && v.getColor().g == 1 && v.getColor().b == 0) funcType = zFunctionType::zLandscape;
			else funcType = zFunctionType::zPublic;

			zHcUnit* tempHcUnit = new zHcUnit(inMeshObjs[i], funcType, _structureType);
			unitArray.push_back(tempHcUnit);
		}
	}

	ZSPACE_INLINE void zHcAggregation::importMeshFromDirectory(string & _path, zFileTpye type)
	{
		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);

		inMeshObjs.assign(pathsArray.size(), zObjMesh());

		for (int i = 0; i < pathsArray.size(); i++)
		{
			zFnMesh tempFnMesh = zFnMesh(inMeshObjs[i]);
			tempFnMesh.from(pathsArray[i], zJSON);
			fnInMeshArray.push_back(tempFnMesh);
		}

	
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zHcAggregation::setDisplayModel(zModel&_model)
	{
		model = &_model;

		if (unitArray.size() == 0) return;

		for (auto& unit : unitArray)
		{
			unit->setUnitDisplayModel(_model);
		}
	}

	ZSPACE_INLINE void zHcAggregation::drawHousing()
	{
		
	}
}