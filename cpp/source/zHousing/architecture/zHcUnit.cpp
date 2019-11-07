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

			structureHeight.push_back(3.0);

			//create and initialise a structure obj and add it to container
			structureUnits[f.getId()] = zHcStructure(vPositions, cellEdgeAttributes, cellBoundaryAttributes, funcType, _structureType, structureHeight[id]);
			structureUnits[f.getId()].createStructuralCell(vPositions);
			structureUnits[f.getId()].createStructureByType(_structureType);
		}

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

	ZSPACE_INLINE void zHcUnit::createLayoutByType(zLayoutType&_layout, bool flip)
	{
		layoutType = _layout;

		if (layoutType == zLayoutType::zStudio) createStudioLayout(flip);
		else if (layoutType == zLayoutType::zOneBed) createOneBedLayout(flip);
		else if (layoutType == zLayoutType::zTwoBed) createTwoBedLayout(flip);
		else if (layoutType == zLayoutType::zLoft) createLoftLayout(flip);
		
		updateStructureUnits();
	}

	ZSPACE_INLINE void zHcUnit::updateStructureUnits()
	{

		for (zItMeshFace f(*inUnitMeshObj); !f.end(); f++)
		{
			int  id = f.getId();
			zPointArray vPositions;
			f.getVertexPositions(vPositions);

			//set cell edges attributes
			zBoolArray cellEdgeAttributes;
			zIntArray heIndices;
			f.getHalfEdges(heIndices);
			for (int heInt : heIndices)
			{
				int edgeIndex = inUnitMeshObj->mesh.halfEdges[heInt].getEdge()->getId();
				cellEdgeAttributes.push_back(edgeAttributes[edgeIndex]);
			}

			structureUnits[id].updateStructure(vPositions, cellEdgeAttributes, structureHeight[id]);
		}
	}

	////////////////////////lsyout creation

	ZSPACE_INLINE void zHcUnit::createStudioLayout(bool flip)
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


	ZSPACE_INLINE void zHcUnit::createOneBedLayout(bool flip)
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

		if (funcType == zFunctionType::zLandscape)
		{
			zItMeshEdge e0, e1, e2, e0f, e1f, e2f;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && e.getHalfEdge(0).getNext().getEdge().onBoundary() && edgeAttributes[e.getId()] == 1)//TODO update with facade / entrance check attribute
				{
					e0 = e;
					e1 = e.getHalfEdge(0).getNext().getNext().getEdge();
					e2 = e1.getHalfEdge(0).getNext().getNext().getEdge();

					e0f = e.getHalfEdge(0).getPrev().getSym().getPrev().getEdge();
					e1f = e1.getHalfEdge(0).getPrev().getSym().getPrev().getEdge();
					e2f = e2.getHalfEdge(0).getNext().getSym().getNext().getEdge();
				}
			}

			zPointArray e0Vs, e1Vs, e2Vs, e0fVs, e1fVs, e2fVs;
			e0.getVertexPositions(e0Vs);
			e1.getVertexPositions(e1Vs);
			e2.getVertexPositions(e2Vs);
			e0f.getVertexPositions(e0fVs);
			e1f.getVertexPositions(e1fVs);
			e2f.getVertexPositions(e2fVs);

			zVector newV0, newV1;

			if (flip)
			{
				newV0 = fnUnitTemp.splitEdge(e0f, 0.75).getPosition();
				newV1 = fnUnitTemp.splitEdge(e1f, 0.75).getPosition();
			}
			else
			{
				newV0 = fnUnitTemp.splitEdge(e0, 0.25).getPosition();
				newV1 = fnUnitTemp.splitEdge(e1, 0.25).getPosition();
			}

			//////////1st volumne

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e2Vs[0]);
				pointArray.push_back(e2Vs[1]);
				pointArray.push_back(e0Vs[0]);
				pointArray.push_back(e0Vs[1]);
			}
			else
			{
				pointArray.push_back(e2fVs[0]);
				pointArray.push_back(e2fVs[1]);
				pointArray.push_back(e0fVs[0]);
				pointArray.push_back(e0fVs[1]);
			}

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(3, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(0, 1, 1, 1), true);

			//////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e2fVs[0]);
				pointArray.push_back(e2fVs[1]);
				pointArray.push_back(e1fVs[0]);
				pointArray.push_back(e1fVs[1]);
			}
			else
			{
				pointArray.push_back(e2Vs[0]);
				pointArray.push_back(e2Vs[1]);
				pointArray.push_back(e1Vs[0]);
				pointArray.push_back(e1Vs[1]);
			}

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(1, 0, 1, 1), true);

			//////////3nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(newV0);
				pointArray.push_back(e0fVs[1]);
				pointArray.push_back(e1fVs[1]);
				pointArray.push_back(newV1);
			}
			else
			{
				pointArray.push_back(e0Vs[0]);
				pointArray.push_back(newV0);
				pointArray.push_back(newV1);
				pointArray.push_back(e1Vs[0]);
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


			if (flip)
			{
				pointArray.push_back(e0fVs[0]);
				pointArray.push_back(newV0);
				pointArray.push_back(newV1);
				pointArray.push_back(e1fVs[0]);

			}
			else
			{
				pointArray.push_back(newV0);
				pointArray.push_back(e0Vs[1]);
				pointArray.push_back(e1Vs[1]);
				pointArray.push_back(newV1);

			}

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp3(layoutMeshObjs[3]);

			layoutTemp3.create(pointArray, polyCount, polyConnect);
			layoutTemp3.extrudeMesh(3, layoutMeshObjs[3], false);
			layoutTemp3.setVertexColor(zColor(1, 1, 0, 1), true);

		}

		
	}

	ZSPACE_INLINE void zHcUnit::createTwoBedLayout(bool flip)
	{
		zObjMesh unitMeshTemp = *inUnitMeshObj;
		zFnMesh fnUnitTemp(unitMeshTemp);

		zPointArray pointArray;
		zIntArray polyCount;
		zIntArray polyConnect;

		if (funcType == zFunctionType::zLandscape)
		{
			zItMeshEdge e0, e1, e2, e0f, e1f, e2f;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && e.getHalfEdge(0).getNext().getEdge().onBoundary() && edgeAttributes[e.getId()] == 1)//TODO update with facade / entrance check attribute
				{
					e0 = e;
					e1 = e.getHalfEdge(0).getNext().getNext().getEdge();
					e2 = e1.getHalfEdge(0).getNext().getNext().getEdge();

					e0f = e.getHalfEdge(0).getPrev().getSym().getPrev().getEdge();
					e1f = e1.getHalfEdge(0).getPrev().getSym().getPrev().getEdge();
					e2f = e2.getHalfEdge(0).getNext().getSym().getNext().getEdge();
				}
			}

			zPointArray e0Vs, e1Vs, e2Vs, e0fVs, e1fVs, e2fVs;
			e0.getVertexPositions(e0Vs);
			e1.getVertexPositions(e1Vs);
			e2.getVertexPositions(e2Vs);
			e0f.getVertexPositions(e0fVs);
			e1f.getVertexPositions(e1fVs);
			e2f.getVertexPositions(e2fVs);

			zVector newV0, newV1;
			zItMeshEdge e0p, e0n, e0fp, e0fn;

			e0p = e0.getHalfEdge(0).getPrev().getEdge();
			e0n = e0.getHalfEdge(0).getNext().getEdge();
			e0fp = e0f.getHalfEdge(0).getPrev().getEdge();
			e0fn = e0f.getHalfEdge(0).getNext().getEdge();

			if (flip)
			{
				newV0 = fnUnitTemp.splitEdge(e0fp, 0.33).getPosition();
				newV1 = fnUnitTemp.splitEdge(e0fn, 0.33).getPosition();
			}
			else
			{
				newV0 = fnUnitTemp.splitEdge(e0p, 0.33).getPosition();
				newV1 = fnUnitTemp.splitEdge(e0n, 0.66).getPosition();
			}

			//////////1st volumne

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e2Vs[0]);
				pointArray.push_back(e2Vs[1]);
				pointArray.push_back(e0Vs[0]);
				pointArray.push_back(e0Vs[1]);
			}
			else
			{
				pointArray.push_back(e2fVs[0]);
				pointArray.push_back(e2fVs[1]);
				pointArray.push_back(e0fVs[0]);
				pointArray.push_back(e0fVs[1]);
			}

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(3, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(0, 1, 1, 1), true);

			//////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e2fVs[0]);
				pointArray.push_back(e2fVs[1]);
				pointArray.push_back(e1fVs[0]);
				pointArray.push_back(e1fVs[1]);
			}
			else
			{
				pointArray.push_back(e2Vs[0]);
				pointArray.push_back(e2Vs[1]);
				pointArray.push_back(e1Vs[0]);
				pointArray.push_back(e1Vs[1]);
			}

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(1, 0, 1, 1), true);

			//////////3nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			if (flip)
			{
				pointArray.push_back(e1fVs[0]);
				pointArray.push_back(newV1);
				pointArray.push_back(newV0);
				pointArray.push_back(e1fVs[1]);

			}
			else
			{
				pointArray.push_back(e1Vs[0]);
				pointArray.push_back(newV1);
				pointArray.push_back(newV0);
				pointArray.push_back(e1Vs[1]);
			}
			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp2(layoutMeshObjs[2]);

			layoutTemp2.create(pointArray, polyCount, polyConnect);
			layoutTemp2.extrudeMesh(3, layoutMeshObjs[2], false);
			layoutTemp2.setVertexColor(zColor(0, 1, 0, 1), true);

			//////////4rd volume
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();


			if (flip)
			{
				pointArray.push_back(newV1);
				pointArray.push_back(e0fVs[0]);
				pointArray.push_back(e0fVs[1]);
				pointArray.push_back(newV0);

			}
			else
			{
				pointArray.push_back(e0Vs[1]);
				pointArray.push_back(newV0);
				pointArray.push_back(newV1);
				pointArray.push_back(e0Vs[0]);

			}

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp3(layoutMeshObjs[3]);

			layoutTemp3.create(pointArray, polyCount, polyConnect);
			layoutTemp3.extrudeMesh(3, layoutMeshObjs[3], false);
			layoutTemp3.setVertexColor(zColor(1, 0, 1, 1), true);

		}

		if (funcType == zFunctionType::zVertical)
		{
			zItMeshEdge upe0, upe1, upe2, upe0n, upe1n, upe2n;
			zItMeshEdge entranceUp, entranceDown;

			bool checkFirst = false;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && !e.getHalfEdge(0).getNext().getNext().getEdge().onBoundary()) //TODO update with facade / entrance check attribute
				{
					if (!checkFirst)
					{
						entranceUp = e;
						checkFirst = true;
					}
					else
					{
						if (e.getCenter().z > entranceUp.getCenter().z) entranceUp = e;
					}
				}
			}

			upe0 = entranceUp;
			upe1 = entranceUp.getHalfEdge(0).getNext().getNext().getEdge();
			upe2 = upe1.getHalfEdge(1).getNext().getNext().getEdge();
			upe1n = upe1.getHalfEdge(1).getNext().getEdge();
			upe2n = upe2.getHalfEdge(0).getNext().getEdge();


			zPointArray e0Vs, e1Vs, e2Vs, e0nVs, e1nVs, e2nVs;
			upe0.getVertexPositions(e0Vs);
			upe1.getVertexPositions(e1Vs);
			upe2.getVertexPositions(e2Vs);

			zVector newV0, newV1, newV2, newV3, newV4;

			newV0 = fnUnitTemp.splitEdge(upe1n, 0.33).getPosition();
			newV1 = fnUnitTemp.splitEdge(upe2n, 0.66).getPosition();

			/*if (flip)
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
			}*/

			//////////1st volumne

			polyCount.push_back(4);

			pointArray.push_back(e1Vs[0]);
			pointArray.push_back(e1Vs[1]);
			pointArray.push_back(e0Vs[0]);
			pointArray.push_back(e0Vs[1]);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(3, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(1, 0, 1, 1), true);

			////////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			pointArray.push_back(e1Vs[1]);
			pointArray.push_back(e1Vs[0]);
			pointArray.push_back(newV1);
			pointArray.push_back(newV0);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(0, 1, 0, 1), true);

			////////////3nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			pointArray.push_back(newV0);
			pointArray.push_back(newV1);
			pointArray.push_back(e2Vs[0]);
			pointArray.push_back(e2Vs[1]);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp2(layoutMeshObjs[2]);

			layoutTemp2.create(pointArray, polyCount, polyConnect);
			layoutTemp2.extrudeMesh(3, layoutMeshObjs[2], false);
			layoutTemp2.setVertexColor(zColor(1, 0, 1, 1), true);

			//////////4rd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			zVector te1 = e0Vs[1] + zVector(0, 0, -3);
			zVector te0 = e0Vs[0] + zVector(0, 0, -3);
			zVector n1 = newV1 + zVector(0, 0, -3);
			zVector n0 = newV0 + zVector(0, 0, -3);

			pointArray.push_back(te0);

			pointArray.push_back(te1);
			pointArray.push_back(n1);
			pointArray.push_back(n0);

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp3(layoutMeshObjs[3]);

			layoutTemp3.create(pointArray, polyCount, polyConnect);
			layoutTemp3.extrudeMesh(3, layoutMeshObjs[3], false);
			layoutTemp3.setVertexColor(zColor(0, 1, 1, 1), true);

			//////////5th volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			te1 = e2Vs[1] + zVector(0, 0, -3);
			te0 = e2Vs[0] + zVector(0, 0, -3);
			n0 = newV0 + zVector(0, 0, -3);
			n1 = newV1 + zVector(0, 0, -3);

			pointArray.push_back(n0);
			pointArray.push_back(n1);
			pointArray.push_back(te0);
			pointArray.push_back(te1);


			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp4(layoutMeshObjs[4]);

			layoutTemp4.create(pointArray, polyCount, polyConnect);
			layoutTemp4.extrudeMesh(3, layoutMeshObjs[4], false);
			layoutTemp4.setVertexColor(zColor(0, 1, 0, 1), true);

			structureHeight[2] = 3.0f;
			structureHeight[0] = 3.0f;


		}
	}


	ZSPACE_INLINE void zHcUnit::createLoftLayout(bool flip)
	{
		zObjMesh unitMeshTemp = *inUnitMeshObj;
		zFnMesh fnUnitTemp(unitMeshTemp);

		zPointArray pointArray;
		zIntArray polyCount;
		zIntArray polyConnect;

		if (funcType == zFunctionType::zVertical)
		{
			zItMeshEdge upe0, upe1, upe2, upe0n, upe1n, upe2n;
			zItMeshEdge entranceUp, entranceDown;

			bool checkFirst = false;
			for (zItMeshEdge e(unitMeshTemp); !e.end(); e++)
			{
				if (e.onBoundary() && !e.getHalfEdge(0).getNext().getNext().getEdge().onBoundary()) //TODO update with facade / entrance check attribute
				{
					if (!checkFirst)
					{
						entranceUp = e;
						checkFirst = true;
					}
					else 
					{
						if (e.getCenter().z > entranceUp.getCenter().z) entranceUp = e;
					}
				}
			}
			
			upe0 = entranceUp;
			upe1 = entranceUp.getHalfEdge(0).getNext().getNext().getEdge();
			upe2 = upe1.getHalfEdge(1).getNext().getNext().getEdge();
			upe1n = upe1.getHalfEdge(1).getNext().getEdge();
			upe2n = upe2.getHalfEdge(0).getNext().getEdge();
			

			zPointArray e0Vs, e1Vs, e2Vs, e0nVs, e1nVs, e2nVs;
			upe0.getVertexPositions(e0Vs);
			upe1.getVertexPositions(e1Vs);
			upe2.getVertexPositions(e2Vs);

			zVector newV0, newV1, newV2, newV3, newV4;

			newV0 = fnUnitTemp.splitEdge(upe1n, 0.66).getPosition();
			newV1 = fnUnitTemp.splitEdge(upe2n, 0.33).getPosition();

			/*if (flip)
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
			}*/

			//////////1st volumne

			polyCount.push_back(4);

			pointArray.push_back(e1Vs[0]);
			pointArray.push_back(e1Vs[1]);
			pointArray.push_back(e0Vs[0]);
			pointArray.push_back(e0Vs[1]);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp0(layoutMeshObjs[0]);

			layoutTemp0.create(pointArray, polyCount, polyConnect);
			layoutTemp0.extrudeMesh(6, layoutMeshObjs[0], false);
			layoutTemp0.setVertexColor(zColor(0, 1, 1, 1), true);

			////////////2nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			pointArray.push_back(e1Vs[1]);
			pointArray.push_back(e1Vs[0]);
			pointArray.push_back(newV1);
			pointArray.push_back(newV0);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp1(layoutMeshObjs[1]);

			layoutTemp1.create(pointArray, polyCount, polyConnect);
			layoutTemp1.extrudeMesh(3, layoutMeshObjs[1], false);
			layoutTemp1.setVertexColor(zColor(1, 0, 1, 1), true);

			////////////3nd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			polyCount.push_back(4);

			pointArray.push_back(newV0);
			pointArray.push_back(newV1);
			pointArray.push_back(e2Vs[0]);
			pointArray.push_back(e2Vs[1]);


			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp2(layoutMeshObjs[2]);

			layoutTemp2.create(pointArray, polyCount, polyConnect);
			layoutTemp2.extrudeMesh(3, layoutMeshObjs[2], false);
			layoutTemp2.setVertexColor(zColor(0, 1, 0, 1), true);

			//////////4rd volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			zVector te1 = e1Vs[1] + zVector(0, 0, -3);
			zVector te0 = e1Vs[0] + zVector(0, 0, -3);
			zVector n0 = newV0+ zVector(0, 0, -3);
			zVector n1 = newV1 + zVector(0, 0, -3);

			pointArray.push_back(te1);
			pointArray.push_back(te0);
			pointArray.push_back(n1);
			pointArray.push_back(n0);

			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp3(layoutMeshObjs[3]);

			layoutTemp3.create(pointArray, polyCount, polyConnect);
			layoutTemp3.extrudeMesh(3, layoutMeshObjs[3], false);
			layoutTemp3.setVertexColor(zColor(0, 1, 1, 1), true);

			//////////5th volumne
			pointArray.clear();
			polyCount.clear();
			polyConnect.clear();

			te1 = e2Vs[1] + zVector(0, 0, -3);
			te0 = e2Vs[0] + zVector(0, 0, -3);
			n0 = newV0 + zVector(0, 0, -3);
			n1 = newV1 + zVector(0, 0, -3);

			pointArray.push_back(n0);
			pointArray.push_back(n1);
			pointArray.push_back(te0);
			pointArray.push_back(te1);


			polyCount.push_back(4);

			for (int i = 0; i < pointArray.size(); i++) polyConnect.push_back(i);

			zFnMesh layoutTemp4(layoutMeshObjs[4]);

			layoutTemp4.create(pointArray, polyCount, polyConnect);
			layoutTemp4.extrudeMesh(3, layoutMeshObjs[4], false);
			layoutTemp4.setVertexColor(zColor(1, 1, 0, 1), true);


			/////sets the new cell heights 
			//TODO update to more efficient method removing faces

			structureHeight[2] = 6.0f;
			structureHeight[0] = 0.0f;



		}
	}

	////////////////////////display methods

	ZSPACE_INLINE void zHcUnit::displayLayout(bool showlayout)
	{
		for (auto &l : layoutMeshObjs)
		{
			l.setShowObject(showlayout);
		}
	}
}