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

		layoutMeshObjs.assign(5, zObjMesh());

		setCellAttributes();
		//createStructuralUnits(_structureType);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcUnit::~zHcUnit() {}

	//---- SET METHODS

	bool zHcUnit::createStructuralUnits(zStructureType&_structureType)
	{
		bool success = false;
		if (!inUnitMeshObj) return success;

		structureUnits.assign(fnUnitMesh.numPolygons(), zHcStructure());

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
			structureUnits[f.getId()] = zHcStructure(vPositions, cellEdgeAttributes, cellBoundaryAttributes, funcType, _structureType);
			structureUnits[f.getId()].createStructuralCell(vPositions);
			structureUnits[f.getId()].createStructureByType(_structureType);
		}

		success = true;
		return success;
	}


	void zHcUnit::setCellAttributes()
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

		if (structureUnits.size() == 0) return;

		for (auto& structure : structureUnits)
		{
			structure.setStructureDisplayModel(_model);
		}

		for (auto& layout : layoutMeshObjs)
		{
			model->addObject(layout);
			layout.setShowElements(false, true, true);
		}
	}

	void zHcUnit::createLayoutByType(zLayoutType&_layout, bool flip)
	{
		layoutType = _layout;

		if (layoutType == zLayoutType::zStudio) createStudioLayout(flip);
		else if (layoutType == zLayoutType::zOneBed) createOneBedLayout(flip);

	}

	void zHcUnit::displayLayout(bool showlayout)
	{
		for (auto &l : layoutMeshObjs)
		{
			l.setShowObject(showlayout);
		}
	}

	/////////////////
	void zHcUnit::createStudioLayout(bool flip)
	{
		zObjMesh unitMeshTemp = *inUnitMeshObj;
		zFnMesh fnUnitTemp(unitMeshTemp);

		zPointArray pointArray;
		zIntArray polyCount;
		zIntArray polyConnect;

		if (funcType == zFunctionType::zFlat)
		{

			zItMeshEdge e0, e1, e2, e0n, e1n;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && e.getHalfEdge(0).getNext().getNext().getEdge().onBoundary()) //TODO update with facade / entrance check attribute
				{
					e0 = e;
					e1 = e.getHalfEdge(0).getNext().getNext().getEdge();
					e0n = e.getHalfEdge(0).getNext().getEdge();
					e1n = e.getHalfEdge(0).getNext().getNext().getNext().getEdge();

					e2 = e0n.getHalfEdge(1).getSym().getNext().getNext().getEdge();

				}
			}

			zPointArray e0Vs, e1Vs, e2Vs, e0nVs, e1nVs;
			e0.getVertexPositions(e0Vs);
			e1.getVertexPositions(e1Vs);
			e2.getVertexPositions(e2Vs);
			e0n.getVertexPositions(e0nVs);
			e1n.getVertexPositions(e1nVs);

			zVector newV0, newV1, newV2, newV3;

			newV0 = fnUnitTemp.splitEdge(e0, 0.33).getPosition();
			newV1 = fnUnitTemp.splitEdge(e1, 0.66).getPosition();

			if (flip)
			{
				newV2 = fnUnitTemp.splitEdge(e1n, 0.25).getPosition();
				zVector dir = newV2 - e1nVs[0];
				newV3 = newV0 + dir;
			}
			else
			{
				newV2 = fnUnitTemp.splitEdge(e1n, 0.75).getPosition();
				zVector dir = newV2 - e1nVs[1];
				newV3 = newV1 + dir;
			}

			//////////1st volumne
			

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e1nVs[0]);
				pointArray.push_back(newV2);
				pointArray.push_back(newV3);
				pointArray.push_back(newV0);
			}
			else
			{
				pointArray.push_back(e1nVs[1]);
				pointArray.push_back(newV1);
				pointArray.push_back(newV3);
				pointArray.push_back(newV2);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);


			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(3, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(0, 1, 0, 1), true);

			////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e1nVs[1]);
				pointArray.push_back(newV1);
				pointArray.push_back(newV3);
				pointArray.push_back(newV2);
			}
			else
			{
				pointArray.push_back(e1nVs[0]);
				pointArray.push_back(newV2);
				pointArray.push_back(newV3);
				pointArray.push_back(newV0);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(1, 1, 0, 1), true);


			////////3rd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			pointArray.push_back(newV0);
			pointArray.push_back(newV1);
			pointArray.push_back(e2Vs[0]);
			pointArray.push_back(e2Vs[1]);

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp2(layoutMeshObjs[2]);

			layoutTemp2.create(pointArray, polyCount, polyConnect);
			layoutTemp2.extrudeMesh(3, layoutMeshObjs[2], false);
			layoutTemp2.setVertexColor(zColor(0, 1, 1, 1), true);

		}
	}

	////////////////////////

	void zHcUnit::createOneBedLayout(bool flip)
	{
		zObjMesh unitMeshTemp = *inUnitMeshObj;
		zFnMesh fnUnitTemp(unitMeshTemp);

		zPointArray pointArray;
		zIntArray polyCount;
		zIntArray polyConnect;

		if (funcType == zFunctionType::zFlat)
		{
			zItMeshEdge e0, e1, e2, e0n, e1n;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && e.getHalfEdge(0).getNext().getNext().getEdge().onBoundary()) //TODO update with facade / entrance check attribute
				{
					e0 = e;
					e1 = e.getHalfEdge(0).getNext().getNext().getEdge();
					e0n = e.getHalfEdge(0).getNext().getEdge();
					e1n = e.getHalfEdge(0).getNext().getNext().getNext().getEdge();
					e2 = e0n.getHalfEdge(1).getSym().getNext().getNext().getEdge();
				}
			}

			zPointArray e0Vs, e1Vs, e2Vs, e0nVs, e1nVs;
			e0.getVertexPositions(e0Vs);
			e1.getVertexPositions(e1Vs);
			e2.getVertexPositions(e2Vs);
			e0n.getVertexPositions(e0nVs);
			e1n.getVertexPositions(e1nVs);

			zVector newV0, newV1, newV2, newV3, newV4;

			newV0 = fnUnitTemp.splitEdge(e0, 0.66).getPosition();
			newV1 = fnUnitTemp.splitEdge(e1, 0.33).getPosition();

			if (flip)
			{
				newV2 = fnUnitTemp.splitEdge(e1n, 0.25).getPosition();
				zVector dir = newV2 - e1nVs[0];
				newV3 = newV0 + dir;
				newV4 = e0nVs[0] + dir;
			}
			else
			{
				newV2 = fnUnitTemp.splitEdge(e1n, 0.75).getPosition();
				zVector dir = newV2 - e1nVs[1];
				newV3 = newV1 + dir;
				newV4 = e0nVs[1] + dir;
			}

			//////////1st volumne

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e1nVs[0]);
				pointArray.push_back(newV2);
				pointArray.push_back(newV3);
				pointArray.push_back(newV0);
			}
			else
			{
				pointArray.push_back(e1nVs[1]);
				pointArray.push_back(newV1);
				pointArray.push_back(newV3);
				pointArray.push_back(newV2);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(3, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(1, 0, 1, 1), true);

			////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e1nVs[1]);
				pointArray.push_back(e0nVs[1]);
				pointArray.push_back(newV4);
				pointArray.push_back(newV2);
			}
			else
			{
				pointArray.push_back(e1nVs[0]);
				pointArray.push_back(newV2);
				pointArray.push_back(newV4);
				pointArray.push_back(e0nVs[0]);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(1, 1, 0, 1), true);

			////////3nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e0nVs[0]);
				pointArray.push_back(newV0);
				pointArray.push_back(newV3);
				pointArray.push_back(newV4);
			}
			else
			{
				pointArray.push_back(e0nVs[1]);
				pointArray.push_back(newV4);
				pointArray.push_back(newV3);
				pointArray.push_back(newV1);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp2(layoutMeshObjs[2]);

			layoutTemp2.create(pointArray, polyCount, polyConnect);
			layoutTemp2.extrudeMesh(3, layoutMeshObjs[2], false);
			layoutTemp2.setVertexColor(zColor(0, 1, 0, 1), true);

			////////4rd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			pointArray.push_back(e0nVs[0]);
			pointArray.push_back(e0nVs[1]);
			pointArray.push_back(e2Vs[0]);
			pointArray.push_back(e2Vs[1]);

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp3(layoutMeshObjs[3]);

			layoutTemp3.create(pointArray, polyCount, polyConnect);
			layoutTemp3.extrudeMesh(3, layoutMeshObjs[3], false);
			layoutTemp3.setVertexColor(zColor(0, 1, 1, 1), true);

		}
	}
}