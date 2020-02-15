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


#include<headers/zInterface/iterators/zItMeshField.h>

//---- ZIT_MESHFIELD_SCALAR ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshScalarField::zItMeshScalarField()
	{
		fieldObj = nullptr;
	}

	ZSPACE_INLINE zItMeshScalarField::zItMeshScalarField(zObjMeshScalarField &_fieldObj)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

	}

	ZSPACE_INLINE zItMeshScalarField::zItMeshScalarField(zObjMeshScalarField &_fieldObj, int _index)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	ZSPACE_INLINE zItMeshScalarField::zItMeshScalarField(zObjMeshScalarField &_fieldObj, zVector &_pos, bool closestIndex)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index_X = floor((_pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int _index_Y = floor((_pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

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

	ZSPACE_INLINE zItMeshScalarField::zItMeshScalarField(zObjMeshScalarField &_fieldObj, int _index_X, int _index_Y)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");


		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItMeshScalarField::begin()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE void zItMeshScalarField::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItMeshScalarField::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItMeshScalarField::end()
	{
		return (iter == fieldObj->field.fieldValues.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshScalarField::reset()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE int zItMeshScalarField::size()
	{

		return fieldObj->field.fieldValues.size();
	}

	ZSPACE_INLINE void zItMeshScalarField::deactivate()
	{
	}

	//--- FIELD TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItMeshScalarField::getNeighbour_Ring(int numRings, zItMeshScalarFieldArray&ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(zItMeshScalarField(*fieldObj, newId));
			}

		}
	}

	ZSPACE_INLINE void zItMeshScalarField::getNeighbour_Ring(int numRings, zIntArray &ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(newId);
			}

		}
	}

	ZSPACE_INLINE void zItMeshScalarField::getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItMeshScalarFieldArray rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fScalar : rNeighbour)
		{
			ringNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE void zItMeshScalarField::getNeighbour_Adjacents(zItMeshScalarFieldArray&adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(zItMeshScalarField(*fieldObj, newId));
				}
			}

		}
	}

	ZSPACE_INLINE void zItMeshScalarField::getNeighbour_Adjacents(zIntArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(newId);
				}
			}

		}
	}

	ZSPACE_INLINE void zItMeshScalarField::getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItMeshScalarFieldArray aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fScalar : aNeighbour)
		{
			adjacentNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE zVector zItMeshScalarField::getPosition()
	{
		if (fieldObj->field.valuesperVertex)
		{
			zItMeshVertex v(*fieldObj, getId());
			return v.getPosition();
		}
		else
		{
			zItMeshFace f(*fieldObj, getId());
			return f.getCenter();
		}
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshScalarField::getId()
	{
		return distance(fieldObj->field.fieldValues.begin(), iter);
	}

	ZSPACE_INLINE void zItMeshScalarField::getIndices(int &index_X, int& index_Y)
	{
		int index = getId();
		index_X = floor(index / fieldObj->field.n_Y);
		index_Y = index % fieldObj->field.n_Y;
				
	}

	ZSPACE_INLINE float zItMeshScalarField::getValue()
	{
		return *iter;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshScalarField::setValue(zScalar val, bool append)
	{
		if (!append) *iter = val;
		else *iter += val;
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshScalarField::operator==(zItMeshScalarField &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshScalarField::operator!=(zItMeshScalarField &other)
	{
		return (getId() != other.getId());
	}


}

//---- ZIT_MESHFIELD_VECTOR ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshVectorField::zItMeshVectorField()
	{
		fieldObj = nullptr;
	}

	ZSPACE_INLINE zItMeshVectorField::zItMeshVectorField(zObjMeshVectorField &_fieldObj)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

	}

	ZSPACE_INLINE zItMeshVectorField::zItMeshVectorField(zObjMeshVectorField &_fieldObj, int _index)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	ZSPACE_INLINE zItMeshVectorField::zItMeshVectorField(zObjMeshVectorField &_fieldObj, zVector &_pos, bool closestIndex)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index_X = floor((_pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int _index_Y = floor((_pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

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

	ZSPACE_INLINE zItMeshVectorField::zItMeshVectorField(zObjMeshVectorField &_fieldObj, int _index_X, int _index_Y)
	{
		fieldObj = &_fieldObj;
		iter = fieldObj->field.fieldValues.begin();

		int _index = _index_X * fieldObj->field.n_Y + _index_Y;

		if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");


		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItMeshVectorField::begin()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE void zItMeshVectorField::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItMeshVectorField::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItMeshVectorField::end()
	{
		return (iter == fieldObj->field.fieldValues.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshVectorField::reset()
	{
		iter = fieldObj->field.fieldValues.begin();
	}

	ZSPACE_INLINE int zItMeshVectorField::size()
	{

		return fieldObj->field.fieldValues.size();
	}

	ZSPACE_INLINE void zItMeshVectorField::deactivate()
	{		
	}

	//--- FIELD TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItMeshVectorField::getNeighbour_Ring(int numRings, zItMeshVectorFieldArray &ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(zItMeshVectorField(*fieldObj, newId));
			}

		}
	}

	ZSPACE_INLINE void zItMeshVectorField::getNeighbour_Ring(int numRings, zIntArray &ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(newId);
			}

		}
	}

	ZSPACE_INLINE void zItMeshVectorField:: getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours)
	{
		ringNeighbours.clear();

		zItMeshVectorFieldArray rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fScalar : rNeighbour)
		{
			ringNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE void zItMeshVectorField::getNeighbour_Adjacents(zItMeshVectorFieldArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(zItMeshVectorField(*fieldObj, newId));
				}
			}

		}
	}

	ZSPACE_INLINE void zItMeshVectorField::getNeighbour_Adjacents(zIntArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(newId);
				}
			}

		}
	}

	ZSPACE_INLINE void zItMeshVectorField::getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		zItMeshVectorFieldArray aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fScalar : aNeighbour)
		{
			adjacentNeighbours.push_back(fScalar.getPosition());
		}
	}

	ZSPACE_INLINE zVector zItMeshVectorField::getPosition()
	{
		if (fieldObj->field.valuesperVertex)
		{
			zItMeshVertex v(*fieldObj, getId());
			return v.getPosition();
		}
		else
		{
			zItMeshFace f(*fieldObj, getId());
			return f.getCenter();
		}
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshVectorField::getId()
	{
		return distance(fieldObj->field.fieldValues.begin(), iter);
	}

	ZSPACE_INLINE void zItMeshVectorField::getIndices(int &index_X, int& index_Y)
	{
		int index = getId();
		index_X = floor(index / fieldObj->field.n_Y);
		index_Y = index % fieldObj->field.n_Y;

	}

	ZSPACE_INLINE zVector zItMeshVectorField::getValue()
	{
		return *iter;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshVectorField::setValue(zVector &val, bool append)
	{
		if (!append) *iter = val;
		else *iter += val;
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshVectorField::operator==(zItMeshVectorField &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshVectorField::operator!=(zItMeshVectorField &other)
	{
		return (getId() != other.getId());
	}

}