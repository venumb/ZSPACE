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


#include<headers/zInterface/iterators/zItPointCLoud.h>

//---- ZIT_POINTCLOUD_VERTEX ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItPointCloudVertex::zItPointCloudVertex()
	{
		pointsObj = nullptr;
	}

	ZSPACE_INLINE zItPointCloudVertex::zItPointCloudVertex(zObjPointCloud &_pointsObj)
	{
		pointsObj = &_pointsObj;

		iter = pointsObj->pCloud.vertices.begin();
	}

	ZSPACE_INLINE zItPointCloudVertex::zItPointCloudVertex(zObjPointCloud &_pointsObj, int _index)
	{
		pointsObj = &_pointsObj;

		iter = pointsObj->pCloud.vertices.begin();

		if (_index < 0 && _index >= pointsObj->pCloud.vertices.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItPointCloudVertex::begin()
	{
		iter = pointsObj->pCloud.vertices.begin();
	}

	ZSPACE_INLINE void zItPointCloudVertex::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItPointCloudVertex::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItPointCloudVertex::end()
	{
		return (iter == pointsObj->pCloud.vertices.end()) ? true : false;
	}

	ZSPACE_INLINE void zItPointCloudVertex::reset()
	{
		iter = pointsObj->pCloud.vertices.begin();
	}

	ZSPACE_INLINE int zItPointCloudVertex::size()
	{

		return pointsObj->pCloud.vertices.size();
	}

	ZSPACE_INLINE void zItPointCloudVertex::deactivate()
	{
		iter->reset();
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItPointCloudVertex::getId()
	{
		return iter->getId();
	}

	ZSPACE_INLINE zItVertex  zItPointCloudVertex::getRawIter()
	{
		return iter;
	}

	ZSPACE_INLINE zPoint zItPointCloudVertex::getPosition()
	{
		return pointsObj->pCloud.vertexPositions[getId()];
	}

	ZSPACE_INLINE zPoint* zItPointCloudVertex::getRawPosition()
	{
		return &pointsObj->pCloud.vertexPositions[getId()];
	}

	ZSPACE_INLINE zColor zItPointCloudVertex::getColor()
	{
		return pointsObj->pCloud.vertexColors[getId()];
	}

	ZSPACE_INLINE zColor* zItPointCloudVertex::getRawColor()
	{
		return &pointsObj->pCloud.vertexColors[getId()];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItPointCloudVertex::setId(int _id)
	{
		iter->setId(_id);
	}

	ZSPACE_INLINE void zItPointCloudVertex::setPosition(zPoint &pos)
	{
		pointsObj->pCloud.vertexPositions[getId()] = pos;
	}

	ZSPACE_INLINE void zItPointCloudVertex::setColor(zColor col)
	{
		pointsObj->pCloud.vertexColors[getId()] = col;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zItPointCloudVertex::isActive()
	{
		return iter->isActive();
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItPointCloudVertex::operator==(zItPointCloudVertex &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItPointCloudVertex::operator!=(zItPointCloudVertex &other)
	{
		return (getId() != other.getId());
	}

}