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


#include<headers/zCore/geometry/zPointCloud.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zPointCloud::zPointCloud()
	{
		n_v = 0;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zPointCloud::~zPointCloud() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zPointCloud::create(zPointArray &_positions)
	{
		// clear containers
		clear();

		vertices.reserve(_positions.size());

		for (int i = 0; i < _positions.size(); i++)	addVertex(_positions[i]);

	}

	ZSPACE_INLINE void zPointCloud::clear()
	{
		vertices.clear();
		vertexPositions.clear();
		vertexColors.clear();
		vertexWeights.clear();


		n_v = 0;
	}

	//---- VERTEX METHODS

	ZSPACE_INLINE bool zPointCloud::addVertex(zPoint &pos)
	{
		bool out = false;


		addToPositionMap(pos, n_v);

		zItVertex newV = vertices.insert(vertices.end(), zVertex());
		newV->setId(n_v);


		vertexPositions.push_back(pos);
		n_v++;



		// default Attibute values			
		vertexColors.push_back(zColor(1, 0, 0, 1));
		vertexWeights.push_back(2.0);

		return out;
	}

	ZSPACE_INLINE bool zPointCloud::vertexExists(zPoint &pos, int &outVertexId, int precisionfactor)
	{
		bool out = false;;
		outVertexId = -1;

		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


		if (got != positionVertex.end())
		{
			out = true;
			outVertexId = got->second;
		}


		return out;
	}

	ZSPACE_INLINE void zPointCloud::setNumVertices(int _n_v)
	{
		n_v = _n_v;
	}

	//---- MAP METHODS

	ZSPACE_INLINE void zPointCloud::addToPositionMap(zPoint &pos, int index, int precisionfactor)
	{
		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		positionVertex[hashKey] = index;
	}

	ZSPACE_INLINE void zPointCloud::removeFromPositionMap(zPoint &pos, int precisionfactor)
	{
		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string removeHashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		positionVertex.erase(removeHashKey);
	}

}