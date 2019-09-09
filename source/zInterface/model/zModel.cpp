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


#include<headers/zInterface/model/zModel.h>

namespace zSpace
{


	//---- CONSTRUCTOR

	zModel::zModel()
	{

		showBufPointColors = false;
		showBufPoints = false;

		showBufLinesColors = false;
		showBufLines = false;

		showBufTrisColors = false;
		showBufTris = false;

		showBufQuadsColors = false;
		showBufQuads = false;
	}

	zModel::zModel(int _buffersize)
	{
		displayUtils.bufferObj = zObjBuffer(_buffersize);

		showBufPointColors = false;
		showBufPoints = false;

		showBufLinesColors = false;
		showBufLines = false;

		showBufTrisColors = false;
		showBufTris = false;

		showBufQuadsColors = false;
		showBufQuads = false;
	}

	//---- DESTRUCTOR
	
	zModel::~zModel() {}

	//---- OBJECT METHODS

	void zModel::addObject(zObj &obj)
	{
		sceneObjects.push_back(&obj);

		obj.setUtils(displayUtils, coreUtils);


	}

	void zModel::addObjects(vector<zObj> &objs)
	{
		for (int i = 0; i < objs.size(); i++)
		{
			sceneObjects.push_back(&objs[i]);

			objs[i].setUtils(displayUtils, coreUtils);
		}


	}

	//---- DRAW METHODS

	void zModel::draw()
	{
		for (auto obj : sceneObjects)
		{
			obj->draw();
		}

		// buffer display

		if (showBufPoints) displayUtils.drawPointsFromBuffer(showBufPointColors);

		if (showBufLines) displayUtils.drawLinesFromBuffer(showBufLinesColors);

		if (showBufTris) displayUtils.drawTrianglesFromBuffer(showBufTrisColors);

		if (showBufQuads) displayUtils.drawQuadsFromBuffer(showBufQuadsColors);

	}

	//---- SET METHODS

	void zModel::setShowBufPoints(bool _showBufPoints, bool showColors)
	{
		showBufPoints = _showBufPoints;

		showBufPointColors = showColors;
	}

	void zModel::setShowBufLines(bool _showBufLines, bool showColors)
	{
		showBufLines = _showBufLines;

		showBufLinesColors = showColors;
	}

	void zModel::setShowBufTris(bool _showBufTris, bool showColors)
	{
		showBufTris = _showBufTris;

		showBufTrisColors = showColors;
	}

	void zModel::setShowBufQuads(bool _showBufQuads, bool showColors)
	{
		showBufQuads = _showBufQuads;

		showBufQuadsColors = showColors;
	}

}