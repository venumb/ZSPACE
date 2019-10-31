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

	ZSPACE_INLINE zHcUnit::zHcUnit(zObjMesh&_inMeshObj, zObjMeshPointerArray &_strcutureObjs, zObjMeshPointerArray &_columnObjs, zObjMeshPointerArray &_slabObjs, zObjMeshPointerArray &_wallObjs, zObjMeshPointerArray &_facadeObjs, zFunctionType&_funcType)
	{
		inMeshObj = &_inMeshObj;
		fnInMesh = zFnMesh(_inMeshObj);
		funcType = _funcType;

		for (int i = 0; i < _strcutureObjs.size(); i++)
		{
			structureObjs.push_back(_strcutureObjs[i]);			
		}
		
		setEdgesAttributes();
		createStructuralUnits(_columnObjs, _slabObjs, _wallObjs, _facadeObjs);
	/*	printf("\n numer of edges in obj: %i", fnInMesh.numEdges());
		printf("\n edgesattrib size: %i", edgeAttributes.size());
		printf("\n boundary attrib size: %i", eBoundaryAttributes.size());*/

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	bool zHcUnit::createStructuralUnits(zObjMeshPointerArray &_columnObjs, zObjMeshPointerArray &_slabObjs, zObjMeshPointerArray &_wallObjs, zObjMeshPointerArray &_facadeObjs)
	{
		bool success = false;
		if (!inMeshObj) return success;

		//create structure obj per face
		for (zItMeshFace f(*inMeshObj); !f.end(); f++)
		{
			int  id = f.getId();
			zPointArray vPositions;
			f.getVertexPositions(vPositions);

			//set new container of corresponding column and slab obj according to vertices num
			zObjMeshPointerArray tempColumnObjs;
			zObjMeshPointerArray tempSlabObjs;
			zObjMeshPointerArray tempWallObjs;
			zObjMeshPointerArray tempFacadeObjs;

			for (int i = 0; i < vPositions.size(); i++)
			{
				tempColumnObjs.push_back(_columnObjs[id * vPositions.size() + i]);
				tempSlabObjs.push_back(_slabObjs[id * vPositions.size() + i]);
				tempWallObjs.push_back(_wallObjs[id * vPositions.size() + i]);
				tempFacadeObjs.push_back(_facadeObjs[id * vPositions.size() + i]);
			}

			//set cell edges attributes
			zBoolArray cellBoundaryAttributes;
			zBoolArray cellEdgeAttributes;
			zIntArray heIndices;
			f.getHalfEdges(heIndices);
			for (int heInt : heIndices)
			{
				int edgeIndex = inMeshObj->mesh.halfEdges[heInt].getEdge()->getId();
				cellEdgeAttributes.push_back(edgeAttributes[edgeIndex]);
				cellBoundaryAttributes.push_back(eBoundaryAttributes[edgeIndex]);
			}

			//printf("\n number of slabObjs: %i", tempSlabObjs.size());

			//create and initialise a structure obj and add it to container
			zHcStructure tempStructure(*structureObjs[id], vPositions, tempColumnObjs, tempSlabObjs, tempWallObjs, tempFacadeObjs, cellEdgeAttributes, cellBoundaryAttributes, funcType);
			structureUnits.push_back(tempStructure);
		}

		success = true;
		return success;
	}

	void zHcUnit::setEdgesAttributes()
	{
		if (!inMeshObj) return;

		for (zItMeshEdge e(*inMeshObj); !e.end(); e++)
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

	void zHcUnit::createLayoutPlan(zLayoutType layout_)
	{
		layoutType = layout_;
	}

}