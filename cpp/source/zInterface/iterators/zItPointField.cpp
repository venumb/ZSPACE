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


#include<headers/zInterface/iterators/zItPointField.h>

//---- ZIT_MESHFIELD_SCALAR ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItPointScalarField::zItPointScalarField()
	{
		fieldObj = nullptr;
	}

	ZSPACE_INLINE zItPointScalarField::zItPointScalarField(zObjPointScalarField &_fieldObj)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

	}

	ZSPACE_INLINE zItPointScalarField::zItPointScalarField(zObjPointScalarField &_fieldObj, int _index)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	ZSPACE_INLINE zItPointScalarField::zItPointScalarField(zObjPointScalarField &_fieldObj, zVector &_pos, bool closestIndex)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index_X = floor((_pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int _index_Y = floor((_pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);
		int _index_Z = floor((_pos.z - fieldObj->field.minBB.z) / fieldObj->field.unit_Z);

		int _index = _index_X * (fieldObj->field.n_Y *fieldObj->field.n_Z) + (_index_Y * fieldObj->field.n_Z) + _index_Z;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);

		if (closestIndex)
		{
			zPointArray rNeighbourPos;
			zIntArray rNeighbours;

			getNeighbourPosition_Ring(1, rNeighbourPos);
			getNeighbour_Ring(1, rNeighbours);

			int idOut = coreUtils.getClosest_PointCloud(_pos, rNeighbourPos);

			_index = rNeighbours[idOut];
			advance(iter, _index);
		}
	}

	ZSPACE_INLINE zItPointScalarField::zItPointScalarField(zObjPointScalarField &_fieldObj, int _index_X, int _index_Y, int _index_Z)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index = _index_X * (fieldObj->field.n_Y *fieldObj->field.n_Z) + (_index_Y * fieldObj->field.n_Z) + _index_Z;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");

		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItPointScalarField::begin()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE void zItPointScalarField::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItPointScalarField::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItPointScalarField::end()
	{
		return (iter == fieldObj->field.fieldValues.end()) ? true : false;
	}

	ZSPACE_INLINE void zItPointScalarField::reset()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE int zItPointScalarField::size()
	{

		return fieldObj->field.fieldValues.size();
	}

	ZSPACE_INLINE void zItPointScalarField::deactivate()
	{
	}

	//--- FIELD TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItPointScalarField::getNeighbour_Ring(int numRings, zItPointScalarFieldArray &ringNeighbours)
	{
		ringNeighbours.clear();

		int index = getId();

		int idX = floor(index / (fieldObj->field.n_Y * fieldObj->field.n_Z));
		int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		int idZ = index % fieldObj->field.n_Z;

		//printf("\n%i : %i %i %i ", index, idX, idY, idZ);

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == fieldObj->field.n_Z - 1) endIdZ = 0;


		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{

				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;

					int newId = (newId_X * (fieldObj->field.n_Y* fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;

					if (newId < size() && newId >= 0) ringNeighbours.push_back(zItPointScalarField(*fieldObj, newId));
				}
			}

		}
	}

	ZSPACE_INLINE void zItPointScalarField::getNeighbour_Ring(int numRings, zIntArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItPointScalarFieldArray rNeighbours;
		this->getNeighbour_Ring(numRings,rNeighbours);

		for(auto n : rNeighbours) ringNeighbours.push_back(n.getId());
	
	}

	ZSPACE_INLINE void zItPointScalarField::getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItPointScalarFieldArray rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fScalar : rNeighbour)
		{
			ringNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE void zItPointScalarField::getNeighbour_Adjacents(zItPointScalarFieldArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / (fieldObj->field.n_Y *fieldObj->field.n_Z));
		int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		int idZ = index % fieldObj->field.n_Z;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == fieldObj->field.n_Z) endIdZ = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;

					int newId = (newId_X * (fieldObj->field.n_Y*fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;


					if (newId < size())
					{
						if (i == 0 || j == 0 || k == 0) adjacentNeighbours.push_back(zItPointScalarField(*fieldObj, newId));
					}

				}
			}

		}
	}

	ZSPACE_INLINE void zItPointScalarField::getNeighbour_Adjacents(zIntArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItPointScalarFieldArray aNeighbours;
		this->getNeighbour_Adjacents(aNeighbours);

		for (auto n : aNeighbours) adjacentNeighbours.push_back(n.getId());
	}

	ZSPACE_INLINE void zItPointScalarField::getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItPointScalarFieldArray aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fScalar : aNeighbour)
		{
			adjacentNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE zVector zItPointScalarField::getPosition()
	{

		zItPointCloudVertex v(*fieldObj, getId());
		return v.getPosition();		
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItPointScalarField::getId()
	{
		return distance(fieldObj->field.fieldValues.begin(), iter);
	}

	ZSPACE_INLINE void zItPointScalarField::getIndices(int &index_X, int& index_Y, int& index_Z)
	{
		int index = getId();

		index_X = floor(index / (fieldObj->field.n_Y * fieldObj->field.n_Z));
		index_Y = floor((index - (index_X *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		index_Z = index % fieldObj->field.n_Z;

	}

	ZSPACE_INLINE float zItPointScalarField::getValue()
	{
		return *iter;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItPointScalarField::setValue(zScalar val, bool append)
	{
		if (!append) *iter = val;
		else *iter += val;
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItPointScalarField::operator==(zItPointScalarField &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItPointScalarField::operator!=(zItPointScalarField &other)
	{
		return (getId() != other.getId());
	}


}

//---- ZIT_MESHFIELD_VECTOR ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItPointVectorField::zItPointVectorField()
	{
		fieldObj = nullptr;
	}

	ZSPACE_INLINE zItPointVectorField::zItPointVectorField(zObjPointVectorField &_fieldObj)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

	}

	ZSPACE_INLINE zItPointVectorField::zItPointVectorField(zObjPointVectorField &_fieldObj, int _index)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	ZSPACE_INLINE zItPointVectorField::zItPointVectorField(zObjPointVectorField &_fieldObj, zVector &_pos, bool closestIndex)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index_X = floor((_pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int _index_Y = floor((_pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	ZSPACE_INLINE zItPointVectorField::zItPointVectorField(zObjPointVectorField &_fieldObj, int _index_X, int _index_Y)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");


		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItPointVectorField::begin()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE void zItPointVectorField::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItPointVectorField::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItPointVectorField::end()
	{
		return (iter == fieldObj->field.fieldValues.end()) ? true : false;
	}

	ZSPACE_INLINE void zItPointVectorField::reset()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE int zItPointVectorField::size()
	{

		return fieldObj->field.fieldValues.size();
	}

	ZSPACE_INLINE void zItPointVectorField::deactivate()
	{		
	}

	//--- FIELD TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItPointVectorField::getNeighbour_Ring(int numRings, zItPointVectorFieldArray &ringNeighbours)
	{
		ringNeighbours.clear();

		int index = getId();

		int idX = floor(index / (fieldObj->field.n_Y * fieldObj->field.n_Z));
		int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		int idZ = index % fieldObj->field.n_Z;

		//printf("\n%i : %i %i %i ", index, idX, idY, idZ);

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == fieldObj->field.n_Z - 1) endIdZ = 0;


		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{

				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;

					int newId = (newId_X * (fieldObj->field.n_Y* fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;

					if (newId < size() && newId >= 0) ringNeighbours.push_back(zItPointVectorField(*fieldObj, newId));
				}
			}

		}
	}

	ZSPACE_INLINE void zItPointVectorField::getNeighbour_Ring(int numRings, zIntArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItPointVectorFieldArray rNeighbours;
		this->getNeighbour_Ring(numRings, rNeighbours);

		for (auto n : rNeighbours) ringNeighbours.push_back(n.getId());

	}

	ZSPACE_INLINE void zItPointVectorField::getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItPointVectorFieldArray rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fScalar : rNeighbour)
		{
			ringNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE void zItPointVectorField::getNeighbour_Adjacents(zItPointVectorFieldArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / (fieldObj->field.n_Y *fieldObj->field.n_Z));
		int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		int idZ = index % fieldObj->field.n_Z;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int startIdZ = -numRings;
		if (idZ == 0) startIdZ = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		int endIdZ = numRings;
		if (idZ == fieldObj->field.n_Z) endIdZ = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				for (int k = startIdZ; k <= endIdZ; k++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;
					int newId_Z = idZ + k;

					int newId = (newId_X * (fieldObj->field.n_Y*fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;


					if (newId < size())
					{
						if (i == 0 || j == 0 || k == 0) adjacentNeighbours.push_back(zItPointVectorField(*fieldObj, newId));
					}

				}
			}

		}
	}

	ZSPACE_INLINE void zItPointVectorField::getNeighbour_Adjacents(zIntArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItPointVectorFieldArray aNeighbours;
		this->getNeighbour_Adjacents(aNeighbours);

		for (auto n : aNeighbours) adjacentNeighbours.push_back(n.getId());
	}

	ZSPACE_INLINE void zItPointVectorField::getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItPointVectorFieldArray aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fScalar : aNeighbour)
		{
			adjacentNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE zVector zItPointVectorField::getPosition()
	{

		zItPointCloudVertex v(*fieldObj, getId());
		return v.getPosition();
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItPointVectorField::getId()
	{
		return distance(fieldObj->field.fieldValues.begin(), iter);
	}

	ZSPACE_INLINE void zItPointVectorField::getIndices(int &index_X, int& index_Y, int& index_Z)
	{
		int index = getId();

		index_X = floor(index / (fieldObj->field.n_Y * fieldObj->field.n_Z));
		index_Y = floor((index - (index_X *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
		index_Z = index % fieldObj->field.n_Z;

	}

	ZSPACE_INLINE zVector zItPointVectorField::getValue()
	{
		return *iter;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItPointVectorField::setValue(zVector &val, bool append)
	{
		if (!append) *iter = val;
		else *iter += val;
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItPointVectorField::operator==(zItPointVectorField &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItPointVectorField::operator!=(zItPointVectorField &other)
	{
		return (getId() != other.getId());
	}

}