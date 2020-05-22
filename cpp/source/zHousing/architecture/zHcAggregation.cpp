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


#include<headers/zHousing/architecture/zHcAggregation.h>

namespace zSpace
{
	//---- CONSTRUCTORS

	ZSPACE_INLINE zHcAggregation::zHcAggregation() { }


	//---- DESTRUCTOR

	ZSPACE_INLINE zHcAggregation::~zHcAggregation() { }


	//---- CREATE METHODS

	ZSPACE_INLINE void zHcAggregation::setSelectedCell(int layer, int id)
	{
		for (int i = 0; i < gridObjs.size(); i++)
		{
			for (int j = 0; j < cellArray[i].size(); j++)
				cellArray[i][j].isSelected = false;
		}

		cellArray[layer][id].isSelected = true;
		userSelection = make_pair(layer, id);
	}

	
	ZSPACE_INLINE void zHcAggregation::createCells()
	{
		for (int i = 0; i < gridObjs.size(); i++)
		{
			int count = 0;
			for (zItMeshFace f(gridObjs[i]); !f.end(); f++)
			{
				zGridCellEstate cellEstate;

				zPointArray vpositions;
				zIntArray vertices;
				f.getVertices(vertices);
				f.getVertexPositions(vpositions);

				bool isAccess = false;

				for (int vId : vertices)
				{
					zColor color = gridObjs[i].mesh.vertexColors[vId];
					if (color.r == 1 && color.g == 0 && color.b == 0)
					{
						isAccess = true;
					}
				}

				cellEstate = (isAccess) ? zGridCellEstate::zAvailable : zGridCellEstate::zLocked;

				cellArray[i][count] = zHcCell(cellEstate);
				cellArray[i][count].createCellMesh(vpositions);
				cellArray[i][count].setLevel(i);

				count++;
			}
		}
	}

	ZSPACE_INLINE void zHcAggregation::createUnit( zHcUnitType _unitType)
	{
		int layer = userSelection.first;
		int id = userSelection.second;

		if (cellArray[layer][id].getEstate() != zGridCellEstate::zAvailable) return;

		zHcUnit* unit;

		if (_unitType == zHcUnitType::zFlat)
		{
			unit = new zHcUnit(unitTypeObjs[0], zHcUnitType::zFlat);
			createFlat(*unit);
		}

		if (_unitType == zHcUnitType::zTwin)
		{
			unit = new zHcUnit(unitTypeObjs[1], zHcUnitType::zTwin);
			createTwin(*unit);
		}

		if (_unitType == zHcUnitType::zLandscape)
		{
			unit = new zHcUnit(unitTypeObjs[2], zHcUnitType::zLandscape);
			createLandscape(*unit);
		}

		unitArray.push_back(unit);
	}

	void zHcAggregation::setUnitTransform(zHcUnit&_unit, zVector pos, zVector orient, float xScale)
	{
		zPoint p = pos;

		//set scale
		float d = xScale;
		_unit.setScale(zVector(d / 5, 1, 1.7)); //5 is the length of the default unit

		//set unit rotation (bug when done after translation)
		zVector o = orient;
		zVector dir = o - p;
		zVector x(1, 0, 0);
		float zAngle = dir.angle(x);

		_unit.setRotation(zVector(0, 0, zAngle));

		//set unit position
		_unit.setPosition(p);
	}


	ZSPACE_INLINE void zHcAggregation::createFlat(zHcUnit& _unit)
	{
		int layer = userSelection.first;
		int id = userSelection.second;

		cellArray[layer][id].setEstate(zGridCellEstate::zReserved);

		zItMeshFace f(gridObjs[layer], id);
		zItMeshHalfEdge he = f.getHalfEdge();

		int vCount = f.getNumVertices();
		for (int i = 0; i < vCount; i++)
		{
			zColor vc1 = he.getVertex().getColor();
			zColor vc2 = he.getPrev().getVertex().getColor();
			if (vc1 == vc2 && vc1.r == 1 && vc1.g == 0 && vc1.b == 0)
			{
				zPoint p = he.getCenter();

				zVector s = he.getNext().getNext().getSym().getNext().getNext().getCenter();
				float d = (s - p).length();

				zVector o = he.getNext().getNext().getCenter();

				//setTransform
				setUnitTransform(_unit, p, o, d);

				int nFaceId = he.getNext().getNext().getSym().getFace().getId();
				cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

				break;
			}

			he++;
		}

		//set unit color
		zColor color(1, 1, 1, 1);
		_unit.setColor(color);
	}

	void zHcAggregation::createTwin(zHcUnit & _unit)
	{
		int layer = userSelection.first;
		int id = userSelection.second;

		cellArray[layer][id].setEstate(zGridCellEstate::zReserved);

		zItMeshFace f(gridObjs[layer], id);
		zItMeshHalfEdge he = f.getHalfEdge();

		int vCount = f.getNumVertices();

		for (int i = 0; i < vCount; i++)
		{

			zColor vc1 = he.getVertex().getColor();
			zColor vc2 = he.getPrev().getVertex().getColor();


			if (vc1 == vc2 && vc1.r == 1 && vc1.g == 0 && vc1.b == 0)
			{
				//check neighbourhood availability
				int tempFaceNext = he.getNext().getSym().getFace().getId();
				int tempFacePrev = he.getPrev().getSym().getFace().getId();


				if (cellArray[layer][tempFaceNext].getEstate() == zGridCellEstate::zAvailable)
				{
					zPoint p = he.getVertex().getPosition();

					//set scale
					zVector s = he.getNext().getNext().getSym().getNext().getVertex().getPosition();
					float d = (s - p).length();

					//set unit rotation (bug when done after translation)
					zVector o = he.getNext().getVertex().getPosition();

					///
					setUnitTransform(_unit, p, o, d);

					int nFaceId = he.getNext().getNext().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					nFaceId = he.getNext().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					nFaceId = he.getNext().getSym().getPrev().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					break;
				}





				else if (cellArray[layer][tempFacePrev].getEstate() == zGridCellEstate::zAvailable)
				{
					zPoint p = he.getPrev().getVertex().getPosition();

					//set scale
					zVector s = he.getNext().getNext().getSym().getNext().getNext().getVertex().getPosition();
					float d = (s - p).length();

					//set unit rotation (bug when done after translation)
					zVector o = he.getNext().getNext().getVertex().getPosition();

					//
					setUnitTransform(_unit, p, o, d);

					int nFaceId = he.getNext().getNext().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					nFaceId = he.getPrev().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					nFaceId = he.getNext().getNext().getSym().getPrev().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					break;
				}

				else break;
			}

			he++;
		}

		//set unit color
		zColor color(1, 1, 1, 1);
		_unit.setColor(color);
	}

	void zHcAggregation::createLandscape(zHcUnit & _unit)
	{
		int layer = userSelection.first;
		int id = userSelection.second;

		cellArray[layer][id].setEstate(zGridCellEstate::zReserved);

		zItMeshFace f(gridObjs[layer], id);
		zItMeshHalfEdge he = f.getHalfEdge();

		int vCount = f.getNumVertices();

		for (int i = 0; i < vCount; i++)
		{
			zColor vc1 = he.getVertex().getColor();
			zColor vc2 = he.getPrev().getVertex().getColor();

			if (vc1 == vc2 && vc1.r == 1 && vc1.g == 0 && vc1.b == 0)
			{
				//check neighbourhood availability
				int tempFaceNext = he.getNext().getSym().getFace().getId();
				int tempFacePrev = he.getPrev().getSym().getFace().getId();


				if (cellArray[layer][tempFaceNext].getEstate() == zGridCellEstate::zAvailable)
				{
					zPoint p = he.getVertex().getPosition();

					//set scale
					zVector s = he.getNext().getVertex().getPosition();
					float d = (s - p).length() * 2; //FIX *2 

					//set unit rotation (bug when done after translation)
					zVector o = he.getNext().getVertex().getPosition();

					///
					setUnitTransform(_unit, p, o, d);

					int nFaceId = he.getNext().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					break;
				}


				else if (cellArray[layer][tempFacePrev].getEstate() == zGridCellEstate::zAvailable)
				{
					zPoint p = he.getPrev().getVertex().getPosition();

					//set scale
					zVector s = he.getNext().getNext().getVertex().getPosition();
					float d = (s - p).length() * 2; //FIX *2

					//set unit rotation (bug when done after translation)
					zVector o = he.getNext().getNext().getVertex().getPosition();

					//
					setUnitTransform(_unit, p, o, d);

					int nFaceId = he.getPrev().getSym().getFace().getId();
					cellArray[layer][nFaceId].setEstate(zGridCellEstate::zReserved);

					break;
				}

				else break;
			}

			he++;
		}

		//set unit color
		zColor color(1, 1, 1, 1);
		_unit.setColor(color);
	}

	//---- IMPORT METHODS


	ZSPACE_INLINE void zHcAggregation::importGridFromDirectory(string&_path)
	{
		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);
		gridObjs.assign(pathsArray.size(), zObjMesh());
		cellArray.assign(pathsArray.size(), vector<zHcCell>());

		int count = 0;
		for (auto& path : pathsArray)
		{
			zFnMesh fnMesh(gridObjs[count]);
			fnMesh.from(path, zJSON);
			count++;
		}

		for (int i = 0; i < gridObjs.size(); i++)
		{
			zFnMesh fnMesh(gridObjs[i]);
			cellArray[i].assign(fnMesh.numPolygons(), zHcCell());
		}
	}

	ZSPACE_INLINE void zHcAggregation::importUnitsFromDirectory(string & _path)
	{
		zStringArray pathsArray;
		core.getFilesFromDirectory(pathsArray, _path, zJSON);
		unitTypeObjs.assign(pathsArray.size(), zObjMesh());

		int count = 0;
		for (auto& path : pathsArray)
		{
			zFnMesh fnMesh(unitTypeObjs[count]);
			fnMesh.from(path, zJSON);
			count++;
		}

		cout << "unit types size: " << unitTypeObjs.size() << endl;
	}

	ZSPACE_INLINE void zHcAggregation::importLayoutMeshesFromDirectory(vector<string>_pathFlats, vector<string>_pathVerticals, vector<string>_pathLandscapes)
	{ 
	}

	//---- UPDATE METHODS

	ZSPACE_INLINE void zHcAggregation::updateStructureType(zStructureType _structureType)
	{
	}

	ZSPACE_INLINE void zHcAggregation::updateLayout(int unitId, zLayoutType & _layoutType, bool flip)
	{
		//unitArray[unitId].setLayoutByType(_layoutType);
	}

	
#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	//---- DISPLAY SET METHODS

	ZSPACE_INLINE void zHcAggregation::setDisplayModel(zModel&_model)
	{
		model = &_model;

		for (int i = 0; i < gridObjs.size(); i++)
		{
			for (int j = 0; j < cellArray[i].size(); j++)
			{
				cellArray[i][j].setModel(_model);
				cellArray[i][j].AddObjToModel();
			}
		}
			
	}

	void zHcAggregation::setUnitDisplayModel(zModel & _model)
	{
		for (auto& h : unitArray)
			h->setUnitDisplayModel(_model);
	}

	//---- DISPLAY SET METHODS

	ZSPACE_INLINE void zHcAggregation::showArchGeom(bool&_showColumn, bool&_showSlab, bool&_showWall, bool&_showFacade, bool&_showRoof)
	{
	/*	for (auto& hc : unitArray)
			hc.structureUnit.displayArchComponents(_showColumn, _showSlab, _showWall, _showFacade, _showRoof);	*/		
	}

	ZSPACE_INLINE  void zHcAggregation::showLayout(int&_index, bool&_showLayout)
	{
		//for (auto& hc : unitArray)
		//	hc.displayLayout(_index, _showLayout);
	}

	ZSPACE_INLINE void zHcAggregation::showCells(bool & _showAll, int &_level)
	{
		for (int i = 0; i < gridObjs.size(); i++)
		{
			for (int j = 0; j < cellArray[i].size(); j++)
				cellArray[i][j].displayCell(_showAll, _level);
		}
			
	}

#endif
}