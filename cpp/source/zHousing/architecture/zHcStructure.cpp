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
		}
		//printf("\n wall numn: %i", wallNum);
		//printf("\n facde numn: %i", facadeNum);

		cellObjs.assign(fnStruct.numPolygons(), zObjMesh());
		columnObjs.assign(columnNum, zObjMesh());
		slabObjs.assign(slabNum, zObjMesh());
		wallObjs.assign(wallNum, zObjMesh());
		facadeObjs.assign(facadeNum, zObjMesh());

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcStructure::~zHcStructure() {}

	//---- CREATE METHODS
	   	 

	ZSPACE_INLINE void zHcStructure::createStructuralCell()
	{
		//int cellCount = 0;
		//for (zItMeshFace f(*inStructObj); !f.end(); f++)
		//{
		//	zPointArray pointArray;
		//	zIntArray polyConnect;
		//	zIntArray polyCount;

		//	polyCount.push_back(f.getNumVertices());
		//	f.getVertexPositions(pointArray);

		//	for (int i = 0; i < f.getNumVertices(); i++)
		//	{
		//		polyConnect.push_back(i);
		//	}

		//	zObjMesh tempObj;
		//	zFnMesh tempFn(tempObj);

		//	tempFn.create(pointArray, polyCount, polyConnect);
		//	tempFn.extrudeMesh(- heightArray[cellCount], cellObjs[cellCount], false);

		//	setCellFacesAttibutes(); ////CHECK!!!!!!!
		//	cellCount++;
		//}

		
	}

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

			createStructuralCell();
			updateArchComponents(structureType);
		//}
	}

	ZSPACE_INLINE void zHcStructure::updateArchComponents(zStructureType & _structureType)
	{

		//structureType = _structureType;

		//for (auto& column : columnArray) column.createColumnByType(structureType);
		//for (auto& slab : slabArray) slab.createSlabByType(structureType);
		//for (auto& wall : wallArray) wall.createWallByType(structureType);


		//for (auto& facade : facadeArray)
		//{
		//	zItMeshFace f(cellObj, facade.faceId);
		//	zPointArray vCorners;
		//	f.getVertexPositions(vCorners);

		//	facade.updateFacade(vCorners);
		//	facade.createFacadeByType(structureType);
		//}

		//for (auto& wall : wallArray)
		//{
		//	zItMeshFace f(cellObj, wall.faceId);
		//	zPointArray vCorners;
		//	f.getVertexPositions(vCorners);

		//	wall.updateWall(vCorners);
		//	wall.createWallByType(structureType);
		//}
	
	}

	ZSPACE_INLINE void zHcStructure::setCellFacesAttibutes()
	{
	/*	for (zItMeshFace f(*inStructObj);  !f.end(); f++)
		{
			if (f.getColor().r > 0.5 && f.getColor().g > 0.5 && f.getColor().b > 0.5)
			{
				cellFaceArray.push_back(zCellFace::zIntWall);
			}
			else if (f.getColor().r == 0 && f.getColor().g == 0 && f.getColor().b == 0)
			{
				cellFaceArray.push_back(zCellFace::zIntWall);
			}
		}

		for (int i = 0; i < cellBoundaryAttributes.size(); i++)
		{
			if (cellBoundaryAttributes[i] == true && cellEdgesAttributes[i] == true) cellFaceArray.push_back(zCellFace::zFacade);
			else cellFaceArray.push_back(zCellFace::zIntWall);
		}*/

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
				//set neighbour faces valencies
				zIntArray valenceArray;

				zItMeshHalfEdgeArray heArray;
				v.getConnectedHalfEdges(heArray);

				for (zItMeshHalfEdge he : heArray)
				{
					//he.getEdge().getId() % 2 == 0 ? axisAttributes.push_back(true) : axisAttributes.push_back(false);

					int nextValence = he.getNext().getVertex().getValence();
					valenceArray.push_back(nextValence);
					
					axis.push_back(he.getVector());
				}
				
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
				zAgColumn tempColumn (columnObjs[cellCount], pos, axis, axisAttributes, valenceArray, 3.0f);
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
					//he.getEdge().getId() % 2 == 0 ? axisAttributes.push_back(true) : axisAttributes.push_back(false);

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

				zAgSlab tempSlab(slabObjs[heCount], centerarray, axis, axisAttributes, columnArray[heCount]);
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

		//for (auto w : wallObjs)
		//{
		//	zFnMesh temp(w);
		//	//temp.numPolygons();
		//	//printf("\n poly num: %i", temp.numPolygons());
		//}
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

				zAgFacade tempFacade = zAgFacade(facadeObjs[count], vCorners, f.getId());
				tempFacade.createFacadeByType(structureType);
				facadeArray.push_back(tempFacade);

				count++;
			}
		}

	/*	int count = 0;
		for (zItMeshFace f(cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zFacade)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zAgFacade tempFacade = zAgFacade(facadeObjs[count], vCorners, f.getId());
				tempFacade.createFacadeByType(structureType);
				facadeArray.push_back(tempFacade);

				count++;
			}
		}
		*/

		return true;
	}



	//---- SET METHODS

	ZSPACE_INLINE void zHcStructure::setStructureDisplayModel(zModel & _model)
	{
		model = &_model;

		for (auto& cell : cellObjs)
		{
			model->addObject(cell);
			cell.setShowElements(false, true, false);
		}

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
		for (auto &s : slabObjs) s.setShowObject(showSlabs);
	}

	ZSPACE_INLINE void zHcStructure::displayWalls(bool showWalls)
	{
		for (auto &w : wallObjs) w.setShowObject(showWalls);
	}

	ZSPACE_INLINE void zHcStructure::displayFacade(bool showFacade)
	{
		for (auto &f : facadeObjs) f.setShowObject(showFacade);
	}

}