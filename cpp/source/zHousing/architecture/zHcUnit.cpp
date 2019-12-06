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

	ZSPACE_INLINE zHcUnit::zHcUnit(zObjMesh&_inMeshObj, zFunctionType&_funcType, zStructureType&_structureType)
	{
		inUnitMeshObj = &_inMeshObj;
		fnUnitMesh = zFnMesh(_inMeshObj);
		funcType = _funcType;

		setCellAttributes();

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	bool zHcUnit::createStructuralUnits(zStructureType&_structureType)
	{
		bool success = false;
		if (!inUnitMeshObj) return success;

		zFloatArray heightArray;
		heightArray.assign(fnUnitMesh.numPolygons(), 3);

		//create and initialise a structure obj and add it to container
		structureUnit = zHcStructure(*inUnitMeshObj, funcType, _structureType, heightArray, edgeAttributes, eBoundaryAttributes);
		//structureUnit.createStructuralCell();
		structureUnit.createStructureByType(_structureType);

		success = true;
		return success;
	}


	ZSPACE_INLINE void zHcUnit::setCellAttributes()
	{
		if (!inUnitMeshObj) return;

		for (zItMeshEdge e(*inUnitMeshObj); !e.end(); e++)
		{
			zItMeshHalfEdge he = e.getHalfEdge(0);
			zItMeshFace f = he.getFace();

			zIntArray indices;
			f.getHalfEdges(indices);
			for (int i = 0; i < indices.size(); i++)
			{
				if (indices[i] == he.getId())
				{
					if (i % 2 == 0) edgeAttributes.push_back(true);
					else edgeAttributes.push_back(false);
				}
			}

			if (e.onBoundary()) eBoundaryAttributes.push_back(true);
			else eBoundaryAttributes.push_back(false);
		}
	}

	ZSPACE_INLINE void zHcUnit::setUnitDisplayModel(zModel&_model)
	{
		model = &_model;

		structureUnit.setStructureDisplayModel(_model);

		for (auto& layout : layoutMeshObjs)
		{
			model->addObject(layout);
			layout.setShowElements(false, true, true);
		}
	}

	ZSPACE_INLINE void zHcUnit::setLayoutByType(zLayoutType&_layout)
	{
		layoutType = _layout;

		/*if (layoutType == zLayoutType::zStudio) createStudioLayout(flip);
		else if (layoutType == zLayoutType::zOneBed) createOneBedLayout(flip);
		else if (layoutType == zLayoutType::zTwoBed) createTwoBedLayout(flip);
		else if (layoutType == zLayoutType::zLoft) createLoftLayout(flip);
		
		updateStructureUnits();*/
	}

	ZSPACE_INLINE void zHcUnit::updateStructureUnits()
	{

		//for (zItMeshFace f(*inUnitMeshObj); !f.end(); f++)
		//{
		//	int  id = f.getId();
		//	zPointArray vPositions;
		//	f.getVertexPositions(vPositions);

		//	//set cell edges attributes
		//	zBoolArray cellEdgeAttributes;
		//	zIntArray heIndices;
		//	f.getHalfEdges(heIndices);
		//	for (int heInt : heIndices)
		//	{
		//		int edgeIndex = inUnitMeshObj->mesh.halfEdges[heInt].getEdge()->getId();
		//		cellEdgeAttributes.push_back(edgeAttributes[edgeIndex]);
		//	}

		//	structureUnits[id].updateStructure(structureHeight[id]);
		//}
	}

	////////////////////////lsyout creation

	ZSPACE_INLINE void zHcUnit::importLayoutFromPath(string & _path)
	{

		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);

		layoutMeshObjs.assign(pathsArray.size(), zObjMesh());
		fnLayoutMeshArray.assign(pathsArray.size(), zFnMesh());

		for (int i = 0; i < pathsArray.size(); i++)
		{
			fnLayoutMeshArray[i] = zFnMesh(layoutMeshObjs[i]);
			fnLayoutMeshArray[i].from(pathsArray[i], zJSON);
		}

	}

	////////////////////////display methods

	ZSPACE_INLINE void zHcUnit::displayLayout(int&_index, bool&_show)
	{
		for (int i = 0; i < layoutMeshObjs.size(); i++)
		{
			_index == i ? layoutMeshObjs[i].setShowObject(_show) : layoutMeshObjs[i].setShowObject(false);
		}
	}
}