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
#include <headers/zHousing/architecture/zHcUnit.h>



namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zHcStructure::zHcStructure() {}


		/*! \brief container to cell faces attributes */
	ZSPACE_INLINE zHcStructure::zHcStructure(zModel&_model, zPointArray &faceVertexPositions, zBoolArray&_cellEdgesAttributes, zBoolArray&_cellBoundaryAttributes, zFunctionType&_funcType)
	{
		model = &_model;
		cellObj = new zObjMesh();
		fnCell = zFnMesh(*cellObj);	


		for (int i = 0; i < faceVertexPositions.size(); i++)
		{
			zObjMesh* tempColumn = new zObjMesh();
			zObjMesh* tempSlab = new zObjMesh();
			zObjMesh* tempWall = new zObjMesh();
			zObjMesh* tempFacade = new zObjMesh();

			columnObjs.push_back(tempColumn);
			slabObjs.push_back(tempSlab);
			wallObjs.push_back(tempWall);
			facadeObjs.push_back(tempFacade);
		}
		//columnObjs.assign(faceVertexPositions.size(), new zObjMesh());
		//slabObjs.assign(faceVertexPositions.size(), zObjMesh());
		//wallObjs.assign(faceVertexPositions.size(), zObjMesh());
		//facadeObjs.assign(faceVertexPositions.size(), zObjMesh());

		cellEdgesAttributes = _cellEdgesAttributes;
		cellBoundaryAttributes = _cellBoundaryAttributes;

		createStructureCell(faceVertexPositions);
		setCellFacesAttibutes();

		if (_funcType == zFunctionType::zPublic)
		{
			createColumns();
			createSlabs();
		}
		else
		{
			createWalls();
			createFacades();
		}

		
		printf("\n column container : %i", columnObjs.size());
		model->addObject(*cellObj);
		cellObj->setShowElements(false, true, false);

		for (auto& c : columnObjs)
		{
			zFnMesh temp(*c);
			printf("\n num of polys in column: %i", temp.numPolygons());
			model->addObject(*c);
			c->setShowElements(false, true, true);
		}
		for (auto& s : slabObjs)
		{
			model->addObject(*s);
			s->setShowElements(false, true, true);
		}
		for (auto& w : wallObjs)
		{
			zFnMesh check = zFnMesh(*w);
			if (check.numPolygons() == 0) continue;

			model->addObject(*w);
			w->setShowElements(false, true, false);
		}
		for (auto& f : facadeObjs)
		{
			zFnMesh check = zFnMesh(*f);
			if (check.numPolygons() == 0) continue;

			model->addObject(*f);
			f->setShowElements(false, true, true);
		}
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHcStructure::~zHcStructure() {}

	//---- CREATE METHODS
	   	 

	ZSPACE_INLINE void zHcStructure::createStructureCell(zPointArray &vPositions)
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
		tempFn.extrudeMesh(-height, *cellObj, false);
	}

	ZSPACE_INLINE void zHcStructure::setCellFacesAttibutes()
	{
		for (zItMeshFace f(*cellObj);  !f.end(); f++)
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
		for (zItMeshFace f(*cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zRoof)
			{
				zIntArray heIndices;
				f.getHalfEdges(heIndices);
				
				int heCount = 0;
				for (int i : heIndices)
				{
					zItMeshHalfEdge he(*cellObj, i);
					
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

					zAgColumn tempColumn = zAgColumn(*columnObjs[heCount], columnPos, x, y, z, height);
					columnArray.push_back(tempColumn);

					heCount++;
				}
			}
		}
		
		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createSlabs()
	{
		if (columnArray.size() == 0) return false;

		for (zItMeshFace f(*cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zRoof)
			{
				zIntArray heIndices;
				f.getHalfEdges(heIndices);

				int heCount = 0;
				for (int i : heIndices)
				{
					zItMeshHalfEdge he(*cellObj, i);

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

					zAgSlab tempSlab = zAgSlab(*slabObjs[heCount], xCenter, yCenter, center, columnArray[heCount]);
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
		for (zItMeshFace f(*cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zIntWall)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zAgWall tempWall = zAgWall(*wallObjs[count], vCorners);

				count++;
			}
		}

		printf("\n num of walls placed: %i", count);

		return true;
	}

	ZSPACE_INLINE bool zHcStructure::createFacades()
	{
		int count = 0;
		for (zItMeshFace f(*cellObj); !f.end(); f++)
		{
			if (cellFaceArray[f.getId()] == zCellFace::zFacade)
			{
				zPointArray vCorners;
				f.getVertexPositions(vCorners);

				zAgFacade tempWall = zAgFacade(*facadeObjs[count], vCorners);

				count++;
			}
		}

		printf("\n num of walls placed: %i", count);

		return true;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zHcStructure::setShowColumns(bool _showCols)
	{
		//for (auto &m : columnObjs) m.setShowObject(_showCols);
	}

}