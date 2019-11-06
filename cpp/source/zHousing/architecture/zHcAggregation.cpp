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
		unitArray.assign(unitObjs.size(), zHcUnit());

	

		for (int i = 0; i < unitObjs.size(); i++)
		{
			zFunctionType funcType;

			zItMeshVertex v(unitObjs[i]);
			if (v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0)	funcType = zFunctionType::zPublic;		
			else if (v.getColor().r == 0 && v.getColor().g == 1 && v.getColor().b == 0) funcType = zFunctionType::zFlat;
			else if (v.getColor().r == 0 && v.getColor().g == 0 && v.getColor().b == 1) funcType = zFunctionType::zLandscape;
			else if (v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 1) funcType = zFunctionType::zVertical;
			else funcType = zFunctionType::zPublic;

			unitArray[i] = zHcUnit(unitObjs[i], funcType, _structureType);
			unitArray[i].createStructuralUnits(_structureType);
			//unitArray[i].createLayoutByType(zLayoutType::zStudio, );
		}

		

	}

	ZSPACE_INLINE void zHcAggregation::importMeshFromDirectory(string & _path, zFileTpye type)
	{
		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);

		unitObjs.assign(pathsArray.size(), zObjMesh());
		fnUnitMeshArray.assign(pathsArray.size(), zFnMesh());

		for (int i = 0; i < pathsArray.size(); i++)
		{
			fnUnitMeshArray[i] = zFnMesh(unitObjs[i]);
			fnUnitMeshArray[i].from(pathsArray[i], zJSON);			
		}	
	}

	//---- UPDATE METHODS

	void zHcAggregation::updateStructureType(zStructureType & _structureType)
	{
		for (auto& hc : unitArray)
		{
			for (auto& s : hc.structureUnits)
			{
				s.updateStructure(_structureType);
			}
		}
	}

	void zHcAggregation::updateLayout(int unitId, zLayoutType & _layoutType, bool flip)
	{
		unitArray[unitId].createLayoutByType(_layoutType, flip);
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zHcAggregation::setDisplayModel(zModel&_model)
	{
		model = &_model;

		if (unitArray.size() == 0) return;

		for (auto& u : unitObjs)
		{
			model->addObject(u);
			u.setShowElements(true, true, false);

		}

		for (auto& unit : unitArray)
		{
			unit.setUnitDisplayModel(_model);

		}
	}


	ZSPACE_INLINE void zHcAggregation::showColumns(bool showColumn)
	{
		for (auto& hc : unitArray)
		{
			for (auto& s : hc.structureUnits)
			{
				
					s.displayColumns(showColumn);			
				
			}
		}
	}

	ZSPACE_INLINE void zHcAggregation::showSlabs(bool showSlab)
	{
		for (auto& hc : unitArray)
		{
			for (auto& s : hc.structureUnits) s.displaySlabs(showSlab);
		}
	}

	ZSPACE_INLINE void zHcAggregation::showWalls(bool showWall)
	{
		for (auto& hc : unitArray)
		{
			for (auto& s : hc.structureUnits) s.displayWalls(showWall);
		}
	}

	ZSPACE_INLINE void zHcAggregation::showFacade(bool showFacade)
	{
		for (auto& hc : unitArray)
		{
			for (auto& s : hc.structureUnits) s.displayFacade(showFacade);
		}
	}
	ZSPACE_INLINE  void zHcAggregation::showLayout(bool showLayout)
	{
		for (auto& hc : unitArray)
		{
			hc.displayLayout(showLayout);
		}
	}
}