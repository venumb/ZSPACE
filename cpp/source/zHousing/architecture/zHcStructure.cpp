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
	ZSPACE_INLINE zHcStructure::zHcStructure(zPointArray &faceVertexPositions, zBoolArray&_cellEdgesAttributes, zBoolArray&_cellBoundaryAttributes, zFunctionType&_funcType, zStructureType&_structureType)
	{
		//cellObj = new zObjMesh();
		fnCell = zFnMesh(cellObj);	
		structureType = _structureType;
		functionType = _funcType;

		columnObjs.assign(faceVertexPositions.size(), zObjMesh());
		slabObjs.assign(faceVertexPositions.size(), zObjMesh());
		wallObjs.assign(faceVertexPositions.size(), zObjMesh());
		facadeObjs.assign(faceVertexPositions.size(),zObjMesh());

		cellEdgesAttributes = _cellEdgesAttributes;
		cellBoundaryAttributes = _cellBoundaryAttributes;

		//createStructuralCell(faceVertexPositions);
		//setCellFacesAttibutes();

		//createStructureByType(structureType);
		
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcStructure::~zHcStructure() {}

	//---- CREATE METHODS
	   	 

	ZSPACE_INLINE void zHcStructure::createStructuralCell(zPointArray &vPositions)
	{
		zIntArray polyConnect;
		zIntArray polyCount;

		int vPositionsCount = vPositions.size();
		polyCount.push_back(vPositions.size());

		for (int i = 0; i < vPositions.size(); i++)
		{
			polyConnect.push_back(i);
		}

		zObjMesh tempObj;
		zFnMesh tempFn(tempObj);

		tempFn.create(vPositions, polyCount, polyConnect);
		tempFn.extrudeMesh(-height, cellObj, false);


		setCellFacesAttibutes();
	}

	void zHcStructure::createStructureByType(zStructureType & _structureType)
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

	void zHcStructure::updateStructureType(zStructureType & _structureType)
	{
		if (columnArray.size() == 0) return;
		if (slabArray.size() == 0) return;
		//if (wallArray.size() == 0) return;
		//if (facadeArray.size() == 0) return;

		structureType = _structureType;

		for (auto& column : columnArray) column.createColumnByType(structureType);
		for (auto& slab : slabArray) slab.createSlabByType(structureType);

	}

	ZSPACE_INLINE void zHcStructure::setCellFacesAttibutes()
	{
		for (zItMeshFace f(cellObj);  !f.end(); f++)
		{
			if (f.getId() == 0) cellFaceArray.push_back(zCellFace::zRoof);
			else if (f.getId() == 1) cellFaceArray.push_back(zCellFace::zFloor);
			else break;
		}

		for (int i = 0; i < cellBoundaryAttributes.size(); i++)
		{
			if (cellBoundaryAttributes[i] == true && cellEdgesAttributes[i] == true) cellFaceArray.push_back(zCellFace::zFacade);
			else cellFaceArray.push_back(zCellFace::zIntWall);
		}
	}

	ZSPACE_INLINE bool zHcStructure::createColumns()
	{
		for (zItMeshFace f(cellObj); !f.end(); f++)
		{
			

			if (cellFaceArray[f.getId()] == zCellFace::zRoof)
			{
				
				zItMeshHalfEdgeArray hEdges;
				f.getHalfEdges(hEdges);
				
				int heCount = 0;
				for (auto he : hEdges)
				{
									
					//set column position
					zVector columnPos = he.getStartVertex().getPosition();

					//set direction according to edge attributes
					zVector x, y;
					if (cellEdgesAttributes[heCount])
					{
						x = he.getVector();
						y = he.getPrev().getVector() * -1;
					}
					else
					{
						y = he.getVector();
						x = he.getPrev().getVector()* -1;
					}

					zVector z = zVector(0, 0, -1);

					zAgColumn tempColumn (columnObjs[heCount], columnPos, x, y, z, height);
					tempColumn.createColumnByType(structureType);
					columnArray.push_back(tempColumn);

					heCount++;
				}
			}
		}
		
		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createSlabs()
	{
		if (columnArray.size() == 0)
		{			
			return false;
		}

		for (zItMeshFace f(cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zRoof)
			{				
				zItMeshHalfEdgeArray hEdges;
				f.getHalfEdges(hEdges);

				int heCount = 0;
				for (auto &he : hEdges)
				{
					//set slab center
					zVector center = f.getCenter();

					//set xCenter and yCenter according to edge attributes
					zVector xCenter, yCenter;
					if (cellEdgesAttributes[heCount])
					{
						xCenter = he.getCenter();
						yCenter = he.getPrev().getCenter();
					}
					else
					{
						yCenter = he.getCenter();
						xCenter = he.getPrev().getCenter();
					}

					zAgSlab tempSlab(slabObjs[heCount], xCenter, yCenter, center, columnArray[heCount]);
					tempSlab.createSlabByType(structureType);
					
					slabArray.push_back(tempSlab);

					heCount++;
				}
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createWalls()
	{
		int count = 0;
		for (zItMeshFace f(cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zIntWall)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zAgWall tempWall = zAgWall(wallObjs[count], vCorners);
				wallArray.push_back(tempWall);

				count++;
			}
		}

		printf("\n num of walls placed: %i", count);

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createFacades()
	{
		int count = 0;
		for (zItMeshFace f(cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zFacade)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zAgFacade tempFacade = zAgFacade(facadeObjs[count], vCorners);
				facadeArray.push_back(tempFacade);
				count++;
			}
		}

		printf("\n num of walls placed: %i", count);

		return true;
	}



	//---- SET METHODS

	ZSPACE_INLINE void zHcStructure::setStructureDisplayModel(zModel & _model)
	{
		model = &_model;
		//model->addObject(cellObj);
		cellObj.setShowElements(false, true, false);

		for (auto& c : columnObjs)
		{
			model->addObject(c);
			c.setShowElements(true, true, false);			
			
		}
		for (auto& s : slabObjs)
		{
			model->addObject(s);
			s.setShowElements(false, true, true);
		}
		for (auto& w : wallObjs)
		{
			model->addObject(w);
			w.setShowElements(false, true, false);
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