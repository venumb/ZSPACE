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


#include<headers/zHousing/architecture/zHcUnit.h>


namespace zSpace
{
	//---- CONSTRUCTORS

	ZSPACE_INLINE zHcUnit::zHcUnit() {}

	ZSPACE_INLINE zHcUnit::zHcUnit(zObjMesh _inMeshObj, zHcUnitType _funcType)
	{

		inUnitMeshObj = _inMeshObj;
		funcType = _funcType;

		cout << "unit" << endl;
		

		if(funcType == zHcUnitType::zFlat) cout << "flat" << endl;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	ZSPACE_INLINE void zHcUnit::setPosition(zVector _position)
	{
		position = _position;
		
		zFnMesh fnMesh(inUnitMeshObj);
		fnMesh.setTranslation(position);
		fnMesh.setPivot(position);
	}

	ZSPACE_INLINE void zHcUnit::setRotation(zVector _rotation)
	{
		rotation = _rotation;
		zFloat4 m = { rotation.x, rotation.y, rotation.z };
		
		zFnMesh fnMesh(inUnitMeshObj);
		fnMesh.setRotation(m);
	}

	ZSPACE_INLINE void zHcUnit::setScale(zVector _scale)
	{
		scale = _scale;
		zFloat4 m = { scale.x, scale.y, scale.z };

		zFnMesh fnMesh(inUnitMeshObj);
		fnMesh.setScale(m);
	}

	ZSPACE_INLINE void zHcUnit::setColor(zColor color)
	{
		zFnMesh fnMesh(inUnitMeshObj);
		fnMesh.setVertexColor(color, true);
	}

	ZSPACE_INLINE void zHcUnit::setLayoutByType(zLayoutType&_layout)
	{
		layoutType = _layout;


	}

	//---- CREATE METHODS

	ZSPACE_INLINE bool zHcUnit::createStructuralUnits(zStructureType _structureType)
	{
		
	}

	//---- IMPORT METHODS

	

	//---- DISPLAY METHODS

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zHcUnit::displayLayout(int&_index, bool&_show)
	{
		for (int i = 0; i < layoutMeshObjs.size(); i++)
		{
			for (int j = 0; j < layoutMeshObjs[i].size(); j++)
			{
				_index == i ? layoutMeshObjs[i][j].setDisplayObject(_show) : layoutMeshObjs[i][j].setDisplayObject(false);

			}
		}
	}

	ZSPACE_INLINE void zHcUnit::setUnitDisplayModel(zModel&_model)
	{
		model = &_model;

		model->addObject(inUnitMeshObj);
		inUnitMeshObj.setDisplayElements(false, true, true);
			
	}

#endif


}