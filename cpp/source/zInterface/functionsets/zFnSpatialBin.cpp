// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Authors : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Federico Borello <federico.borello@zaha-hadid.com>
//


#include<headers/zInterface/functionsets/zFnSpatialBin.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnSpatialBin::zFnSpatialBin()
	{
		binObj = nullptr;
	}

	ZSPACE_INLINE zFnSpatialBin::zFnSpatialBin(zObjSpatialBin &_binObj)
	{
		binObj = &_binObj;
		fnPoints = zFnPointCloud(_binObj);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnSpatialBin::~zFnSpatialBin() {}

	//---- FACTORY METHODS

	ZSPACE_INLINE void zFnSpatialBin::clear()
	{
		binObj->bins.clear();
		binObj->field.fieldValues.clear();
		fnPoints.clear();
	}

	//---- CREATE METHOD

	ZSPACE_INLINE void zFnSpatialBin::create(const zPoint &_minBB, const zPoint &_maxBB , int _res )
	{
		binObj->field = zField3D<zScalar>(_minBB, _maxBB, _res, _res, _res);

		// initialise bins
		binObj->bins.assign(_res*_res*_res, zBin());

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		for (int i = 0; i < numBins(); i++)
		{
			zIntArray temp_ringNeighbour;
			getNeighbourhoodRing(i, 1, temp_ringNeighbour);
			ringNeighbours.push_back(temp_ringNeighbour);

			zIntArray temp_adjacentNeighbour;
			getNeighbourAdjacents(i, temp_adjacentNeighbour);
			adjacentNeighbours.push_back(temp_adjacentNeighbour);
		}

		// create field points
		createPointCloud();

	}

	//---- GET METHODS

	ZSPACE_INLINE int zFnSpatialBin::numObjects()
	{
		return objects.size();
	}

	ZSPACE_INLINE int zFnSpatialBin::numBins()
	{
		return binObj->bins.size();
	}

	ZSPACE_INLINE void zFnSpatialBin::getNeighbourhoodRing(int index, int numRings, zIntArray &ringNeighbours)
	{
		ringNeighbours.clear();


		int idX = floor(index / (binObj->field.n_Y * binObj->field.n_Z));
		int idY = floor((index - (idX *binObj->field.n_Z *binObj->field.n_Y)) / (binObj->field.n_Z));
		int idZ = index % binObj->field.n_Z;

		//printf("\n%i : %i %i %i ", index, idX, idY, idZ);

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == binObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == binObj->field.n_Y - 1) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == binObj->field.n_Z - 1) endIdZ = 0;


		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{

				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;



					int newId = (newId_X * (binObj->field.n_Y* binObj->field.n_Z)) + (newId_Y * binObj->field.n_Z) + newId_Z;


					if (newId < numBins() && newId >= 0) ringNeighbours.push_back(newId);
				}

			}

		}


	}

	ZSPACE_INLINE void zFnSpatialBin::getNeighbourAdjacents(int index, zIntArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		//printf("\n working numRings : %i ", numRings);

		int idX = floor(index / (binObj->field.n_Y *binObj->field.n_Z));
		int idY = floor((index - (idX *binObj->field.n_Z *binObj->field.n_Y)) / (binObj->field.n_Z));
		int idZ = index % binObj->field.n_Z;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == binObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == binObj->field.n_Y) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == binObj->field.n_Z) endIdZ = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;

					int newId = (newId_X * (binObj->field.n_Y*binObj->field.n_Z)) + (newId_Y * binObj->field.n_Z) + newId_Z;


					if (newId < numBins())
					{
						if (i == 0 || j == 0 || k == 0) adjacentNeighbours.push_back(newId);
					}
				}
			}

		}



	}

	ZSPACE_INLINE bool zFnSpatialBin::getIndex(int index_X, int index_Y, int index_Z, int &index)
	{
		bool out = true;

		if (index_X > (binObj->field.n_X - 1) || index_X <  0 || index_Y >(binObj->field.n_Y - 1) || index_Y < 0 || index_Z >(binObj->field.n_Z - 1) || index_Z < 0) out = false;

		index = index_X * (binObj->field.n_Y *binObj->field.n_Z) + (index_Y * binObj->field.n_Z) + index_Z;

		return out;
	}

	ZSPACE_INLINE bool zFnSpatialBin::getIndex(zPoint &pos, int &index)
	{

		int index_X = floor((pos.x - binObj->field.minBB.x) / binObj->field.unit_X);
		int index_Y = floor((pos.y - binObj->field.minBB.y) / binObj->field.unit_Y);
		int index_Z = floor((pos.z - binObj->field.minBB.z) / binObj->field.unit_Z);

		bool out = getIndex(index_X, index_Y, index_Z, index);

		return out;

	}

	ZSPACE_INLINE bool zFnSpatialBin::getIndices(zPoint &pos, int &index_X, int &index_Y, int &index_Z)
	{
		index_X = floor((pos.x - binObj->field.minBB.x) / binObj->field.unit_X);
		index_Y = floor((pos.y - binObj->field.minBB.y) / binObj->field.unit_Y);
		index_Z = floor((pos.z - binObj->field.minBB.z) / binObj->field.unit_Z);

		bool out = true;
		if (index_X > (binObj->field.n_X - 1) || index_X <  0 || index_Y >(binObj->field.n_Y - 1) || index_Y < 0 || index_Z >(binObj->field.n_Z - 1) || index_Z < 0) out = false;

		return out;
	}

	//---- METHODS

	ZSPACE_INLINE void zFnSpatialBin::clearBins()
	{
		binObj->bins.clear();
	}

	//---- PROTECTED METHODS
	   
	ZSPACE_INLINE void zFnSpatialBin::partitionToBin(zPoint &inPos, int pointId, int objectId)
	{

		int index;
		bool check = getIndex(inPos, index);

		if (check)
		{
			if (objectId >= objects.size()) throw std::invalid_argument(" error: object index out of bounds.");

			else binObj->bins[index].addVertexIndex(pointId, objectId);
		}

	}

	ZSPACE_INLINE void zFnSpatialBin::createPointCloud()
	{
		vector<zVector>positions;

		zVector minBB = binObj->field.minBB;
		zVector maxBB = binObj->field.maxBB;

		double unit_X = binObj->field.unit_X;
		double unit_Y = binObj->field.unit_Y;
		double unit_Z = binObj->field.unit_Z;

		int n_X = binObj->field.n_X;
		int n_Y = binObj->field.n_Y;
		int n_Z = binObj->field.n_Z;

		zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
		zVector startPt = minBB;

		for (int i = 0; i < n_X; i++)
		{
			for (int j = 0; j < n_Y; j++)
			{

				for (int k = 0; k < n_Z; k++)
				{

					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;
					pos.z = startPt.z + k * unitVec.z;

					positions.push_back(pos);

				}
			}
		}

		fnPoints.create(positions);

	}
}