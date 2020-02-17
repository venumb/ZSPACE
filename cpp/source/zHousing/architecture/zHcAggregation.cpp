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

	ZSPACE_INLINE zHcAggregation::zHcAggregation() { }


	//---- DESTRUCTOR

	ZSPACE_INLINE zHcAggregation::~zHcAggregation() {}


	//---- CREATE METHODS

	ZSPACE_INLINE void zHcAggregation::createHousingUnits(zStructureType _structureType)
	{
		unitArray.assign(unitObjs.size(), zHcUnit());

		for (int i = 0; i < unitObjs.size(); i++)
		{
			zFunctionType funcType;

			zItMeshVertex v(unitObjs[i]);

			for (zItMeshFace f(unitObjs[i]); !f.end(); f++)
			{
				if (f.getColor().r == 1 && f.getColor().g == 0 && f.getColor().b == 0)
				{
					funcType = zFunctionType::zPublic;
					break;
				}
				else if (f.getColor().r == 0 && f.getColor().g == 1 && f.getColor().b == 0)
				{
					funcType = zFunctionType::zFlat;
					break;

				}
				else if (f.getColor().r == 0 && f.getColor().g == 0 && f.getColor().b == 1)
				{
					funcType = zFunctionType::zLandscape;
					break;
				}
				else if (f.getColor().r == 1 && f.getColor().g == 0 && f.getColor().b == 1)
				{
					funcType = zFunctionType::zVertical;
					break;
				}

				else
				{
					funcType = zFunctionType::zPublic;
				}
			}
			
			unitArray[i] = zHcUnit(unitObjs[i], funcType, _structureType);
			unitArray[i].createStructuralUnits(_structureType);
		}

		

	}

	//---- IMPORT METHODS

	ZSPACE_INLINE void zHcAggregation::importMeshesFromDirectory(string&_path)
	{
		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);

		unitObjs.assign(pathsArray.size(), zObjMesh());
		fnUnitMeshArray.assign(pathsArray.size(), zFnMesh());

		for (int i = 0; i < pathsArray.size(); i++)
		{
			fnUnitMeshArray[i] = zFnMesh(unitObjs[i]);
			fnUnitMeshArray[i].from(pathsArray[i], zJSON);
			//fnUnitMeshArray[i].triangulate();
		}	
	}

	ZSPACE_INLINE void zHcAggregation::importLayoutMeshesFromDirectory(vector<string>_pathFlats, vector<string>_pathVerticals, vector<string>_pathLandscapes)
	{ 

		int uId = 0;
		for (auto &h : unitArray)
		{
			if (h.funcType == zFunctionType::zFlat)
			{
				h.importLayoutsFromPath(_pathFlats);
			}
			else if (h.funcType == zFunctionType::zVertical)
			{
				h.importLayoutsFromPath(_pathVerticals);
			}
			else if (h.funcType == zFunctionType::zLandscape)
			{			
				h.importLayoutsFromPath(_pathLandscapes);
			}

			uId++;
		}
	}

	//---- UPDATE METHODS

	ZSPACE_INLINE void zHcAggregation::updateStructureType(zStructureType _structureType)
	{
		for (auto& hc : unitArray)
		{
			hc.structureUnit.updateArchComponents(_structureType);
		}
	}

	ZSPACE_INLINE void zHcAggregation::updateLayout(int unitId, zLayoutType & _layoutType, bool flip)
	{
		unitArray[unitId].setLayoutByType(_layoutType);
	}

	
#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	//---- DISPLAY SET METHODS

	ZSPACE_INLINE void zHcAggregation::setDisplayModel(zModel&_model)
	{
		model = &_model;

		if (unitArray.size() == 0) return;


		for (auto& u : unitObjs)
		{
			model->addObject(u);
			u.setDisplayElements(true, true, false);
		}


		for (auto& unit : unitArray)
		{
			unit.setUnitDisplayModel(_model);

		}
	}

	//---- DISPLAY SET METHODS

	ZSPACE_INLINE void zHcAggregation::showArchGeom(bool&_showColumn, bool&_showSlab, bool&_showWall, bool&_showFacade, bool&_showRoof)
	{
		for (auto& hc : unitArray)
		{
			hc.structureUnit.displayArchComponents(_showColumn, _showSlab, _showWall, _showFacade, _showRoof);			
		}
	}

	ZSPACE_INLINE  void zHcAggregation::showLayout(int&_index, bool&_showLayout)
	{
		for (auto& hc : unitArray)
		{
			hc.displayLayout(_index, _showLayout);
		}
	}

#endif
}