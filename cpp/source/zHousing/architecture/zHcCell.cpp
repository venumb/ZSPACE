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


#include<headers/zHousing/architecture/zHcCell.h>


namespace zSpace
{
	//---- CONSTRUCTORS

	ZSPACE_INLINE zHcCell::zHcCell() {}


	ZSPACE_INLINE zHcCell::zHcCell(zGridCellEstate _cellEstate)
	{
		cellEstate = _cellEstate;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcCell::~zHcCell() {}

	//---- SET METHODS

	ZSPACE_INLINE zGridCellEstate zHcCell::getEstate()
	{
		return cellEstate;
	}

	ZSPACE_INLINE void zHcCell::setLevel(int _level)
	{
		level = _level;
	}

	ZSPACE_INLINE void zHcCell::setEstate(zGridCellEstate estate)
	{
		cellEstate = estate;
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zHcCell::createCellMesh(zPointArray vertexPositions)
	{
		zIntArray polyCounts, polyConnects;

		polyCounts.push_back(vertexPositions.size());
		for (int i = 0; i < vertexPositions.size(); i++) polyConnects.push_back(i);

		zFnMesh fnMesh(cellObj);
		fnMesh.create(vertexPositions, polyCounts, polyConnects);
		fnMesh.extrudeMesh(2.f, cellObj, false);

		zFloat4 m = { 0.95,0.95,0.95 };
		zVector pivot = fnMesh.getCenter();
		fnMesh.setPivot(pivot);
		fnMesh.setScale(m);

	}


	//---- DISPLAY METHODS

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zHcCell::displayCell(bool&_showAll, int&_showLevel)
	{
		//////////////

		if (_showAll)
		{
			cellObj.setDisplayObject(true);
		}
		else
		{
			if (level == _showLevel)
				cellObj.setDisplayObject(true);
			else
				cellObj.setDisplayObject(false);
		}

		///////////////////

		if (cellEstate == zGridCellEstate::zAvailable)
		{
			cellObj.setDisplayElements(false, true, true);
			zColor color = (isSelected) ? zColor(1, 1, 0, 1) : zColor(1, 0, 1, 1);
			setColor(color);
		}

		else if (cellEstate == zGridCellEstate::zLocked)
		{
			cellObj.setDisplayElements(false, true, false);
			setColor(zColor(0.2, 0.2, 0.2, 1));
		}
		else if (cellEstate == zGridCellEstate::zReserved)
		{
			cellObj.setDisplayObject(false);
		}

	}

	ZSPACE_INLINE void zHcCell::setColor(zColor color)
	{
		zFnMesh fnMesh(cellObj);

		if(cellEstate == zGridCellEstate::zAvailable)
			fnMesh.setVertexColor(color, true);
	}

	ZSPACE_INLINE void zHcCell::setModel(zModel&_model)
	{
		model = &_model;
	}

	ZSPACE_INLINE void zHcCell::AddObjToModel()
	{
		model->addObject(cellObj);
	}

#endif


}