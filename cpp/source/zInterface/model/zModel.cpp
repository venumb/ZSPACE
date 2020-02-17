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

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

namespace zSpace
{


	//---- CONSTRUCTOR

	ZSPACE_INLINE zModel::zModel()
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

	ZSPACE_INLINE zModel::zModel(int _buffersize)
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
	
	ZSPACE_INLINE zModel::~zModel() {}

	//---- OBJECT METHODS

	ZSPACE_INLINE void zModel::addObject(zObj &obj)
	{
		sceneObjects.push_back(&obj);

		obj.setUtils(displayUtils);
	}

	ZSPACE_INLINE void zModel::addObjects(zObjArray &objs)
	{
		for (auto &obj : objs) addObject(obj);		
	}

	//---- DRAW METHODS

	ZSPACE_INLINE void zModel::draw()
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

	ZSPACE_INLINE void zModel::setShowBufPoints(bool _showBufPoints, bool showColors)
	{
		showBufPoints = _showBufPoints;

		showBufPointColors = showColors;
	}

	ZSPACE_INLINE void zModel::setShowBufLines(bool _showBufLines, bool showColors)
	{
		showBufLines = _showBufLines;

		showBufLinesColors = showColors;
	}

	ZSPACE_INLINE void zModel::setShowBufTris(bool _showBufTris, bool showColors)
	{
		showBufTris = _showBufTris;

		showBufTrisColors = showColors;
	}

	ZSPACE_INLINE void zModel::setShowBufQuads(bool _showBufQuads, bool showColors)
	{
		showBufQuads = _showBufQuads;

		showBufQuadsColors = showColors;
	}

}

#endif