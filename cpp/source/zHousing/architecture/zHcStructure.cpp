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

#include<headers/zHousing/architecture/zHcStructure.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zHcStructure::zHcStructure() {}

	ZSPACE_INLINE zHcStructure::zHcStructure(zObjMesh&_inStructObj, zFunctionType&_funcType, zStructureType&_structureType, zFloatArray _heightArray, zBoolArray&_edgesAttributes, zBoolArray&_boundaryAttributes)
	{
		inStructObj = &_inStructObj;
		fnStruct = zFnMesh(*inStructObj);	
		structureType = _structureType;
		functionType = _funcType;

		heightArray = _heightArray;

		int columnNum = 0;
		int slabNum = 0;
		int wallNum = 0;
		int facadeNum = 0;
		int roofNum = 0;



		for (zItMeshVertex v(*inStructObj); !v.end(); v++)
		{
			if (!v.onBoundary() && v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0)
			{
				columnNum++;
				slabNum++;
			}
		}


		for (zItMeshFace f(*inStructObj); !f.end(); f++)
		{

			if (f.getColor().r == 1 && f.getColor().g == 1 && f.getColor().b == 1) facadeNum++;
			else if (f.getColor().r == 0 && f.getColor().g == 0 && f.getColor().b == 0)wallNum++;
			else if ((f.getColor().r != f.getColor().g || f.getColor().r != f.getColor().b) && f.getNormal().z > 0) roofNum++;
		}

		columnArray.assign(columnNum, zAgColumn());
		slabArray.assign(slabNum, zAgSlab());
		wallArray.assign(wallNum, zAgWall());
		facadeArray.assign(facadeNum, zAgFacade());
		roofArray.assign(roofNum, zAgRoof());


		printf("\n column array: %i", columnArray.size());

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcStructure::~zHcStructure() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zHcStructure::createStructureByType(zStructureType & _structureType)
	{
		structureType = _structureType;

		if (functionType == zFunctionType::zPublic)
		{
			createColumns();
			createSlabs();
		}
		else
		{
			createWalls();
			createFacades();
			createRoofs();
		}
	}

	ZSPACE_INLINE bool zHcStructure::createColumns()
	{
		int count = 0;
		for (zItMeshVertex v(*inStructObj); !v.end(); v++)
		{
			if (!v.onBoundary() && v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0)
			{

				//set vector axis for column
				zVectorArray axis;
				//set axis attibutes primary or secondary direction
				zBoolArray axisAttributes;


				vector<zBoundary> boundaryArray;

				zItMeshHalfEdgeArray heArray;
				v.getConnectedHalfEdges(heArray);

				for (zItMeshHalfEdge he : heArray)
				{
					int nextValence = he.getNext().getVertex().getValence();
					if (nextValence == 2) boundaryArray.push_back(zBoundary::zCorner);
					else if (nextValence == 3) boundaryArray.push_back(zBoundary::zEdge);
					else  boundaryArray.push_back(zBoundary::zInterior);

					axis.push_back(he.getVector());
				}

				//TODO Update
				if (v.getHalfEdge().getId() % 2 == 0)
				{
					axisAttributes.push_back(true);
					axisAttributes.push_back(false);
					axisAttributes.push_back(true);
					axisAttributes.push_back(false);
				}

				else
				{
					axisAttributes.push_back(false);
					axisAttributes.push_back(true);
					axisAttributes.push_back(false);
					axisAttributes.push_back(true);
				}


				/////
				zVector z = zVector(0, 0, -1);
				zVector pos = v.getPosition();

				/////
				columnArray[count] = zAgColumn(pos, axis, axisAttributes, boundaryArray, 3.0f);
				columnArray[count].createByType(structureType);

				count++;
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createSlabs()
	{
		int heCount = 0;
		for (zItMeshVertex v(*inStructObj); !v.end(); v++)
		{
			if (!v.onBoundary() && v.getColor().r == 1 && v.getColor().g == 0 && v.getColor().b == 0)
			{
	
				zVectorArray centerarray;//set center array
				zVectorArray axis;//set mid points axis array
				zBoolArray axisAttributes;//set axis attributes primary or secondary

				zItMeshHalfEdgeArray heArray;
				v.getConnectedHalfEdges(heArray);

				for (zItMeshHalfEdge he : heArray)
				{

					int nextValence = he.getNext().getVertex().getValence();

					if (nextValence == 2)
					{
						centerarray.push_back(he.getNext().getVertex().getPosition());
						axis.push_back(he.getVertex().getPosition());
					}
					else if (nextValence == 3)
					{
						if (he.getNext().getSym().onBoundary())
						{
							centerarray.push_back(he.getNext().getCenter());
							axis.push_back(he.getVertex().getPosition());
						}
						else
						{
							centerarray.push_back(he.getNext().getNext().getCenter());
							axis.push_back(he.getCenter());
						}
					}
					else
					{
						centerarray.push_back(he.getFace().getCenter());
						axis.push_back(he.getCenter());
					}
					
				}

				slabArray[heCount] = zAgSlab(centerarray, axis, columnArray[heCount]);
				slabArray[heCount].createByType(structureType);

				heCount++;

			}
		}
		

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createWalls()
	{
		int count = 0;
		for (zItMeshFace f(*inStructObj); !f.end(); f++)
		{
			if (f.getColor().r == 0 && f.getColor().g == 0 && f.getColor().b == 0)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);
				
				wallArray[count] = zAgWall(vCorners, f.getId());
				wallArray[count].createByType(structureType);

				count++;
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createFacades()
	{

		int count = 0;
		for (zItMeshFace f(*inStructObj); !f.end(); f++)
		{
			if (f.getColor().r == 1 && f.getColor().g == 1 && f.getColor().b == 1)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zVectorArray extrudeDir;

				zItMeshVertexArray vInFace;
				f.getVertices(vInFace);

				//loop to get vertex not in face
				for (auto vf : vInFace)
				{
					zItMeshHalfEdgeArray heArray;
					vf.getConnectedHalfEdges(heArray);

					for (auto he: heArray)
					{

						zVector dir = he.getVector();
						dir.normalize();
						zVector n = f.getNormal();
						if (fabs(n * dir) > 0.5) extrudeDir.push_back(dir);
					}
				}

				facadeArray[count] = zAgFacade(vCorners, extrudeDir, f.getId());
				facadeArray[count].createByType(structureType);

				count++;
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createRoofs()
	{
		int count = 0;
		for (zItMeshFace f(*inStructObj); !f.end(); f++)
		{
			if ((f.getColor().r != f.getColor().g || f.getColor().r != f.getColor().b) && f.getNormal().z > 0)
			{
				zPointArray corners;
				f.getVertexPositions(corners);
				bool isFacade = false;

				zItMeshHalfEdgeArray heArray;
				f.getHalfEdges(heArray);

				for (auto he : heArray)
				{
					if (he.getSym().onBoundary()) continue;
					if (he.getSym().getFace().getColor().r == 1 && he.getSym().getFace().getColor().g == 1 && he.getSym().getFace().getColor().b == 1)
					{
						//jumping zigzAG
						corners[0] = he.getPrev().getVertex().getPosition();
						corners[1] = he.getNext().getNext().getVertex().getPosition();

						corners[2] = he.getVertex().getPosition();
						corners[3] = he.getNext().getVertex().getPosition();

						isFacade = true;
					}
				}

				roofArray[count] = zAgRoof(corners, isFacade);
				roofArray[count].createByType(structureType);

				count++;
			}
		}

		

		return true;
	}

	//---- UPDATE METHODS

	ZSPACE_INLINE void zHcStructure::updateArchComponents(zStructureType & _structureType)
	{
		structureType = _structureType;

		if (functionType == zFunctionType::zPublic)
		{
			for (auto& column : columnArray) column.createByType(structureType);
			for (auto& slab : slabArray) slab.createByType(structureType);
		}
		else
		{
			for (auto& wall : wallArray) wall.createByType(structureType);
			for (auto& facade : facadeArray) facade.createByType(structureType);
			for (auto& roof : roofArray) roof.createByType(structureType);
		}

	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	//---- SET METHODS

	ZSPACE_INLINE void zHcStructure::displayArchComponents(bool showColumn, bool showSlabs, bool showWalls, bool showFacade, bool showRoof)
	{
		for (auto& c : columnArray) c.displayColumn(showColumn);
	
		for (auto& s : slabArray) s.displaySlab(showSlabs);

		for (auto& w : wallArray) w.displayWall(showWalls);
		
		for (auto& f : facadeArray) f.displayFacade(showFacade);

		for (auto& r : roofArray) r.displayRoof(showRoof);

	}

	//---- DISPLAY METHODS
	ZSPACE_INLINE void zHcStructure::setStructureDisplayModel(zModel & _model)
	{
		model = &_model;

		for (auto& c : columnArray)
		{
			c.setModel(_model);
			c.addObjsToModel();
		}

		for (auto& s : slabArray)
		{
			s.setModel(_model);
			s.addObjsToModel();
		}

		for (auto& w : wallArray)
		{
			w.setModel(_model);
			w.addObjsToModel();
		}

		for (auto& f : facadeArray)
		{
			f.setModel(_model);
			f.addObjsToModel();
		}

		for (auto& r : roofArray)
		{
			r.setModel(_model);
			r.addObjsToModel();
		}
	}
#endif

}