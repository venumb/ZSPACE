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

	ZSPACE_INLINE zHcUnit::zHcUnit(){}

	ZSPACE_INLINE zHcUnit::zHcUnit(zObjMesh&_inMeshObj, zFunctionType&_funcType, zStructureType&_structureType)
	{
		inUnitMeshObj = &_inMeshObj;
		fnUnitMesh = zFnMesh(_inMeshObj);
		funcType = _funcType;

		setEdgesAttributes();
		createStructuralUnits(_structureType);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	bool zHcUnit::createStructuralUnits(zStructureType&_structureType)
	{
		bool success = false;
		if (!inUnitMeshObj) return success;

		//create structure obj per face
		for (zItMeshFace f(*inUnitMeshObj); !f.end(); f++)
		{
			int  id = f.getId();
			zPointArray vPositions;
			f.getVertexPositions(vPositions);

			//set cell edges attributes
			zBoolArray cellBoundaryAttributes;
			zBoolArray cellEdgeAttributes;
			zIntArray heIndices;
			f.getHalfEdges(heIndices);
			for (int heInt : heIndices)
			{
				int edgeIndex = inUnitMeshObj->mesh.halfEdges[heInt].getEdge()->getId();
				cellEdgeAttributes.push_back(edgeAttributes[edgeIndex]);
				cellBoundaryAttributes.push_back(eBoundaryAttributes[edgeIndex]);
			}

			//create and initialise a structure obj and add it to container
			zHcStructure* tempStructure = new zHcStructure(vPositions, cellEdgeAttributes, cellBoundaryAttributes, funcType, _structureType);
			structureUnits.push_back(tempStructure);
		}

		success = true;
		return success;
	}

	void zHcUnit::setEdgesAttributes()
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

	void zHcUnit::setUnitDisplayModel(zModel&_model)
	{
		model = &_model;
		model->addObject(*inUnitMeshObj);
		inUnitMeshObj->setShowElements(true, true, false);

		if (structureUnits.size() == 0) return;

		for (auto& structure : structureUnits)
		{
			structure->setStructureDisplayModel(_model);
		}
	}

	void zHcUnit::createLayoutPlan(zLayoutType layout_)
	{
		layoutType = layout_;
	}

}