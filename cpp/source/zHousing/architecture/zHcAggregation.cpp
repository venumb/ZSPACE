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

	ZSPACE_INLINE zHcAggregation::zHcAggregation(zObjMeshArray&_inMeshObs, vector<zObjMeshArray> &_strcutureObjs, vector<zObjMeshArray> &_columnObjs, vector<zObjMeshArray> &_slabObjs, vector<zObjMeshArray> &_wallObjs, vector<zObjMeshArray> &_facadeObjs)
	{
		for (auto& m : _inMeshObs)
		{
			inMeshObjs.push_back(&m);
		}

		vector<zObjMesh*> temp;
		structureObjs.assign(_inMeshObs.size(), temp);
		columnObjs.assign(_inMeshObs.size(), temp);
		slabObjs.assign(_inMeshObs.size(), temp);
		wallObjs.assign(_inMeshObs.size(), temp);
		facadeObjs.assign(_inMeshObs.size(), temp);
		//printf("\n numer of col: %i", _columnObjs[0].size());
		//printf("\n numer of slabs: %i", _slabObjs[1].size());
		//printf("\n numer of col container: %i", _columnObjs.size());

		for (int i = 0; i < _strcutureObjs.size(); i++)
		{
			for (int j = 0; j < _strcutureObjs[i].size(); j++)
			{
				structureObjs[i].push_back(&_strcutureObjs[i][j]);

			}
		}

		for (int i = 0; i < _columnObjs.size(); i++)
		{
			for (int j = 0; j < _columnObjs[i].size(); j++)
			{
				columnObjs[i].push_back(&_columnObjs[i][j]);

			}
		}

		for (int i = 0; i < _slabObjs.size(); i++)
		{
			for (int j = 0; j < _slabObjs[i].size(); j++)
			{
				slabObjs[i].push_back(&_slabObjs[i][j]);

			}
		}
		for (int i = 0; i < _wallObjs.size(); i++)
		{
			for (int j = 0; j < _wallObjs[i].size(); j++)
			{
				wallObjs[i].push_back(&_wallObjs[i][j]);

			}
		}
		for (int i = 0; i < _facadeObjs.size(); i++)
		{
			for (int j = 0; j < _facadeObjs[i].size(); j++)
			{
				facadeObjs[i].push_back(&_facadeObjs[i][j]);

			}
		}

		//printf("\n numer of items obj: %i", inMeshObjs.size());

		for (int i = 0; i < _inMeshObs.size(); i++)
		{
			setupHousingUnits(i);
		}


	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcAggregation::~zHcAggregation() {}

	//---- SET METHODS

	

	void zHcAggregation::setupHousingUnits(int index)
	{
		zFunctionType funcType;

		zItMeshVertex v(*inMeshObjs[index]);
		if (v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0) funcType = zFunctionType::zPublic;
		else if (v.getColor().r == 0 && v.getColor().g == 1 && v.getColor().b == 0) funcType = zFunctionType::zLandscape;
		else funcType = zFunctionType::zPublic;

		zHcUnit tempHcUnit = zHcUnit(*inMeshObjs[index], structureObjs[index], columnObjs[index], slabObjs[index], wallObjs[index], facadeObjs[index], funcType);
		unitArray.push_back(&tempHcUnit);

	}


}