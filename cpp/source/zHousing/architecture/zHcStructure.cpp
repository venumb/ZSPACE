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
//#include <headers/zHousing/architecture/zHcUnit.h>



namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zHcStructure::zHcStructure() {}


		/*! \brief container to cell faces attributes */
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
			//printf("\n color: %f", f.getColor().a);

			if (f.getColor().r == 1 && f.getColor().g == 1 && f.getColor().b == 1) facadeNum++;
			else if (f.getColor().r == 0 && f.getColor().g == 0 && f.getColor().b == 0)wallNum++;
			else if ((f.getColor().r != f.getColor().g  || f.getColor().r != f.getColor().b) && f.getNormal().z > 0) roofNum++;
		}

		//printf("\n wall numn: %i", wallNum);
		printf("\n roof numn: %i", roofNum);



		columnObjs.assign(columnNum, zObjMesh());
		slabObjs.assign(slabNum, zObjMesh());
		wallObjs.assign(wallNum, zObjMesh());
		facadeObjs.assign(facadeNum, zObjMesh());
		roofObjs.assign(roofNum, zObjMesh());
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

	ZSPACE_INLINE void zHcStructure::updateStructure (zFloatArray _heightArray)
	{
		//if (_height == 0)
		//{
		//	for (auto&c : columnObjs) c = zObjMesh();
		//	for (auto&s : slabObjs) s = zObjMesh();
		//	for (auto&w : wallObjs) w = zObjMesh();
		//	for (auto&f : facadeObjs) f = zObjMesh();

		//}
		/*else
		{*/
			heightArray = _heightArray;

			//createStructuralCell();
			updateArchComponents(structureType);
		//}
	}

	ZSPACE_INLINE void zHcStructure::updateArchComponents(zStructureType & _structureType)
	{
		structureType = _structureType;

		for (auto& column : columnArray) column.createColumnByType(structureType);
		for (auto& slab : slabArray) slab.createSlabByType(structureType);
		for (auto& wall : wallArray) wall.createWallByType(structureType);
		for (auto& facade : facadeArray) facade.createFacadeByType(structureType);
		for (auto& roof : roofArray) roof.createRoofByType(structureType);
	}

	ZSPACE_INLINE bool zHcStructure::createColumns()
	{
		int cellCount = 0;
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
				zAgColumn tempColumn (columnObjs[cellCount], pos, axis, axisAttributes, boundaryArray, 3.0f);
				tempColumn.createColumnByType(structureType);
				columnArray.push_back(tempColumn);

				cellCount++;
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


				zAgSlab tempSlab(slabObjs[heCount], centerarray, axis, columnArray[heCount]);
				tempSlab.createSlabByType(structureType);

				slabArray.push_back(tempSlab);

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
				
				zAgWall tempWall = zAgWall(wallObjs[count], vCorners, f.getId());
				tempWall.createWallByType(structureType);
				wallArray.push_back(tempWall);

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

				//printf("dir count : %i", extrudeDir.size());

				zAgFacade tempFacade = zAgFacade(facadeObjs[count], vCorners, extrudeDir, f.getId());
				tempFacade.createFacadeByType(structureType);
				facadeArray.push_back(tempFacade);

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

				zAgRoof tempRoof = zAgRoof(roofObjs[count], corners, isFacade);
				tempRoof.createRoofByType(structureType);
				roofArray.push_back(tempRoof);

				count++;
			}
		}

		return true;
	}



	//---- SET METHODS

	ZSPACE_INLINE void zHcStructure::setStructureDisplayModel(zModel & _model)
	{
		model = &_model;

		for (auto& c : columnObjs)
		{
			model->addObject(c);
			c.setShowElements(false, true, true);			
			
		}
		for (auto& s : slabObjs)
		{
			model->addObject(s);
			s.setShowElements(false, true, true);
		}
		for (auto& w : wallObjs)
		{
			model->addObject(w);
			w.setShowElements(false, true, true);
		}
		for (auto& f : facadeObjs)
		{
			model->addObject(f);
			f.setShowElements(false, true, true);
		}
		for (auto& r : roofObjs)
		{
			model->addObject(r);
			r.setShowElements(false, true, true);
		}
	}


	//---- DISPLAY METHODS

	ZSPACE_INLINE void zHcStructure::displayColumns(bool showColumns)
	{
		for (auto &c : columnObjs)
		{
			c.setShowObject(showColumns);	
		}
	}

	ZSPACE_INLINE void zHcStructure::displaySlabs(bool showSlabs)
	{
		for (auto &s : slabObjs)
		{
			s.setShowObject(showSlabs);
		}
	}

	ZSPACE_INLINE void zHcStructure::displayWalls(bool showWalls)
	{
		for (auto &w : wallObjs) w.setShowObject(showWalls);
	}

	ZSPACE_INLINE void zHcStructure::displayFacade(bool showFacade)
	{
		for (auto &f : facadeObjs) f.setShowObject(showFacade);
	}

	ZSPACE_INLINE void zHcStructure::displayRoof(bool showRoof)
	{
		for (auto &r : roofObjs) r.setShowObject(showRoof);
	}

}