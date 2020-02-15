// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Authors : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com> , Leo Bieling <leo.bieling.zaha-hadid.com>
//


#include<headers/zInterface/functionsets/zFnMeshField.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	template<typename T>
	ZSPACE_INLINE zFnMeshField<T>::zFnMeshField()
	{
		fnType = zFnType::zMeshFieldFn;
		fieldObj = nullptr;

	}

	template<typename T>
	ZSPACE_INLINE zFnMeshField<T>::zFnMeshField(zObjMeshField<T> &_fieldObj)
	{
		fieldObj = &_fieldObj;

		fnType = zFnType::zMeshFieldFn;
		fnMesh = zFnMesh(_fieldObj);

	}

	//---- DESTRUCTOR

	template<typename T>
	ZSPACE_INLINE zFnMeshField<T>::~zFnMeshField() {}

	//---- FACTORY METHODS

	template<typename T>
	ZSPACE_INLINE zFnType zFnMeshField<T>::getType()
	{
		return zFnType::zMeshFieldFn;
	}

	//---- zScalar &  zVector specilization for from

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::from(string path, zFileTpye type, bool _setValuesperVertex, bool _trimesh)
	{
		setValuesperVertex = _setValuesperVertex;

		if (!setValuesperVertex) _trimesh = false;
		triMesh = _trimesh;

		if (type == zBMP) fromBMP(path);

		else if (type == zOBJ) fnMesh.from(path, type, true);
		else if (type == zJSON) fnMesh.from(path, type, true);

		else throw std::invalid_argument(" error: invalid zFileTpye type");

	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::from(string path, zFileTpye type, bool _setValuesperVertex, bool _trimesh)
	{
		setValuesperVertex = _setValuesperVertex;

		if (!setValuesperVertex) _trimesh = false;
		triMesh = _trimesh;		

		if (type == zOBJ) fnMesh.from(path, type, true);
		else if (type == zJSON) fnMesh.from(path, type, true);

		else throw std::invalid_argument(" error: invalid zFileTpye type");

	}

	//---- zScalar &  zVector specilization for from

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::to(string path, zFileTpye type)
	{
		if (type == zBMP) toBMP(path);
		else if (type == zOBJ) fnMesh.to(path, type);
		else if (type == zJSON) fnMesh.to(path, type);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::to(string path, zFileTpye type)
	{
		if (type == zOBJ) fnMesh.to(path, type);
		else if (type == zJSON) fnMesh.to(path, type);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		minBB = fieldObj->field.minBB;
		maxBB = fieldObj->field.maxBB;
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::clear()
	{
		ringNeighbours.clear();
		adjacentNeighbours.clear();
		fieldObj->field.fieldValues.clear();
		fnMesh.clear();
	}

	//---- CREATE METHODS

	//---- zScalar &  zVector specilization for create

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _NR, bool _setValuesperVertex, bool _triMesh)
	{
		setValuesperVertex = _setValuesperVertex;
		if (!_setValuesperVertex) _triMesh = false;
		triMesh = _triMesh;


		fieldObj->field = zField2D<zScalar>(_minBB, _maxBB, _n_X, _n_Y);
		fieldObj->field.valuesperVertex = setValuesperVertex;

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();


		ringNeighbours.assign(numFieldValues(), vector<int>());
		adjacentNeighbours.assign(numFieldValues(), vector<int>());

		int i = 0;
		for (zItMeshScalarField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		createFieldMesh();

		updateColors();

	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _NR, bool _setValuesperVertex, bool _triMesh)
	{
		setValuesperVertex = _setValuesperVertex;
		if (!_setValuesperVertex) _triMesh = false;
		triMesh = _triMesh;


		fieldObj->field = zField2D<zVector>(_minBB, _maxBB, _n_X, _n_Y);
		fieldObj->field.valuesperVertex = setValuesperVertex;

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();


		ringNeighbours.assign(numFieldValues(), vector<int>());
		adjacentNeighbours.assign(numFieldValues(), vector<int>());

		int i = 0;
		for (zItMeshVectorField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		createFieldMesh();		

	}

	//---- zScalar &  zVector specilization for create

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zPoint _minBB, int _NR, bool _setValuesperVertex, bool _triMesh)
	{
		setValuesperVertex = _setValuesperVertex;
		if (!_setValuesperVertex) _triMesh = false;
		triMesh = _triMesh;

		fieldObj->field = zField2D<zScalar>(_unit_X, _unit_Y, _n_X, _n_Y, _minBB);
		fieldObj->field.valuesperVertex = setValuesperVertex;

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		ringNeighbours.assign(numFieldValues(), vector<int>());
		adjacentNeighbours.assign(numFieldValues(), vector<int>());

		int i = 0; 
		for( zItMeshScalarField s(*fieldObj); !s.end(); s++, i++)		
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		createFieldMesh();

		updateColors();
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zPoint _minBB, int _NR, bool _setValuesperVertex, bool _triMesh)
	{
		setValuesperVertex = _setValuesperVertex;
		if (!_setValuesperVertex) _triMesh = false;
		triMesh = _triMesh;

		fieldObj->field = zField2D<zVector>(_unit_X, _unit_Y, _n_X, _n_Y, _minBB);
		fieldObj->field.valuesperVertex = setValuesperVertex;

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		ringNeighbours.assign(numFieldValues(), vector<int>());
		adjacentNeighbours.assign(numFieldValues(), vector<int>());

		int i = 0;
		for (zItMeshVectorField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		createFieldMesh();		
	}

	//---- zVector specilization for createVectorFieldFromScalarField

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::createVectorFromScalarField(zObjMeshField<zScalar> &scalarFieldObj)
	{
		zFnMeshField<zScalar> fnScalarField(scalarFieldObj);

		zVector minBB, maxBB;
		fnScalarField.getBoundingBox(minBB, maxBB);

		int n_X, n_Y;
		fnScalarField.getResolution(n_X, n_Y);

		vector<zVector> gradients = fnScalarField.getGradients();

		create(minBB, maxBB, n_X, n_Y, fnScalarField.getValuesPerVertexBoolean(), fnScalarField.getTriMeshBoolean());
		setFieldValues(gradients);
	}

	//---- QUERIES

	//---- zScalar &  zVector specilization for getNeighbour_Contained
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getNeighbour_Contained(zPoint &pos, vector<int> &containedNeighbour)
	{
		containedNeighbour.clear();

		zItMeshScalarField s(*fieldObj, pos);

		int index = s.getId();	

		int numRings = 1;
		//printf("\n working numRings : %i ", numRings);

		int idX, idY;
		s.getIndices(idX, idY);		

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);
			}

		}

		// case 1
		if (checkBounds_X(idX - 1) && checkBounds_Y(idY + 1))
		{
			zItMeshScalarField g1(*fieldObj, idX - 1, idY);

			zItMeshScalarField g2(*fieldObj, idX, idY);

			zItMeshScalarField g3(*fieldObj, idX, idY + 1);

			zItMeshScalarField g4(*fieldObj, idX - 1, idY + 1);

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}



		}

		// case 2
		if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY + 1))
		{

			zItMeshScalarField g1(*fieldObj, idX, idY);

			zItMeshScalarField g2(*fieldObj, idX + 1, idY);

			zItMeshScalarField g3(*fieldObj, idX + 1, idY + 1);

			zItMeshScalarField g4(*fieldObj, idX, idY + 1);

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}

		}

		// case 3
		if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY - 1))
		{
			zItMeshScalarField g1(*fieldObj, idX, idY - 1);

			zItMeshScalarField g2(*fieldObj, idX + 1, idY - 1);

			zItMeshScalarField g3(*fieldObj, idX + 1, idY);

			zItMeshScalarField g4(*fieldObj, idX, idY);

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}
		}


		// case 4
		if (containedNeighbour.size() == 0 && checkBounds_X(idX - 1) && checkBounds_Y(idY - 1))
		{

			zItMeshScalarField g1(*fieldObj, idX - 1, idY - 1);

			zItMeshScalarField g2(*fieldObj, idX, idY - 1);

			zItMeshScalarField g3(*fieldObj, idX, idY);

			zItMeshScalarField g4(*fieldObj, idX - 1, idY);


			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}
		}


	}
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::getNeighbour_Contained(zPoint &pos, vector<int> &containedNeighbour)
	{
		containedNeighbour.clear();

		zItMeshVectorField s(*fieldObj, pos);

		int index = s.getId();

		int numRings = 1;
		//printf("\n working numRings : %i ", numRings);

		int idX, idY;
		s.getIndices(idX, idY);

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);
			}

		}

		// case 1
		if (checkBounds_X(idX - 1) && checkBounds_Y(idY + 1))
		{
			zItMeshVectorField g1(*fieldObj, idX - 1, idY);

			zItMeshVectorField g2(*fieldObj, idX, idY);

			zItMeshVectorField g3(*fieldObj, idX, idY + 1);

			zItMeshVectorField g4(*fieldObj, idX -1, idY + 1);

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}



		}

		// case 2
		if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY + 1))
		{

			zItMeshVectorField g1(*fieldObj, idX, idY);

			zItMeshVectorField g2(*fieldObj, idX + 1, idY);

			zItMeshVectorField g3(*fieldObj, idX + 1, idY + 1);

			zItMeshVectorField g4(*fieldObj, idX, idY + 1);		

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}

		}

		// case 3
		if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY - 1))
		{
			zItMeshVectorField g1(*fieldObj, idX, idY - 1);

			zItMeshVectorField g2(*fieldObj, idX + 1, idY - 1);

			zItMeshVectorField g3(*fieldObj, idX + 1, idY);

			zItMeshVectorField g4(*fieldObj, idX, idY );		

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}
		}


		// case 4
		if (containedNeighbour.size() == 0 && checkBounds_X(idX - 1) && checkBounds_Y(idY - 1))
		{

			zItMeshVectorField g1(*fieldObj, idX -1, idY - 1);

			zItMeshVectorField g2(*fieldObj, idX , idY - 1);

			zItMeshVectorField g3(*fieldObj, idX, idY);

			zItMeshVectorField g4(*fieldObj, idX - 1, idY);
		

			zVector minBB_temp = g1.getPosition();
			if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			zVector maxBB_temp = g3.getPosition();
			if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

			bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

			if (check)
			{
				containedNeighbour.push_back(g1.getId());
				containedNeighbour.push_back(g2.getId());
				containedNeighbour.push_back(g3.getId());
				containedNeighbour.push_back(g4.getId());
			}
		}


	}

	//---- zScalar &  zVector specilization for getNeighbourPosition_Contained
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getNeighbourPosition_Contained(zPoint &pos, zPointArray &containedNeighbour)
	{
		containedNeighbour.clear();

		vector<int> cNeighbourIndex;
		getNeighbour_Contained(pos, cNeighbourIndex);

		for (int i = 0; i < cNeighbourIndex.size(); i++)
		{
			zItMeshScalarField s(*fieldObj, cNeighbourIndex[i]);

			containedNeighbour.push_back(s.getPosition());
		}

	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::getNeighbourPosition_Contained(zPoint &pos, zPointArray &containedNeighbour)
	{
		containedNeighbour.clear();

		vector<int> cNeighbourIndex;
		getNeighbour_Contained(pos, cNeighbourIndex);

		for (int i = 0; i < cNeighbourIndex.size(); i++)
		{
			zItMeshVectorField s(*fieldObj, cNeighbourIndex[i]);

			containedNeighbour.push_back(s.getPosition());
		}

	}
	
	//---- GET METHODS
	
	template<typename T>
	ZSPACE_INLINE int zFnMeshField<T>::numFieldValues()
	{
		return fieldObj->field.fieldValues.size();
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getResolution(int &_n_X, int &_n_Y)
	{
		_n_X = fieldObj->field.n_X;
		_n_Y = fieldObj->field.n_Y;
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getUnitDistances(double &_unit_X, double &_unit_Y)
	{
		_unit_X = fieldObj->field.unit_X;
		_unit_Y = fieldObj->field.unit_Y;

	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getBoundingBox(zPoint &_minBB, zPoint &_maxBB)
	{
		_minBB = fieldObj->field.minBB;
		_maxBB = fieldObj->field.maxBB;
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getPositions(zPointArray &positions)
	{
		if (setValuesperVertex) 	fnMesh.getVertexPositions(positions);
		else fnMesh.getCenters(zFaceData, positions);
	}

	//---- zScalar and zVector specilization for getFieldValue

	template<>
	ZSPACE_INLINE bool zFnMeshField<zScalar>::getFieldValue(zPoint &samplePos, zFieldValueType type, zScalar& fieldValue)
	{

		bool out = false;

		zItMeshScalarField s(*fieldObj, samplePos);

		int index = s.getId();

		if (type == zFieldIndex)
		{
			zScalar fVal;

			fVal = fieldObj->field.fieldValues[index];

			fieldValue = fVal;
		}

		else if (type == zFieldNeighbourWeighted)
		{
			zScalar fVal = 0;

			zItMeshScalarField s(*fieldObj, index);

			vector<zItMeshScalarField> ringNeighbours;
			s.getNeighbour_Ring( 1, ringNeighbours);

			zPointArray positions;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				positions.push_back(ringNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				zScalar val = 	ringNeighbours[i].getValue();
				fVal += val * weights[i];
				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldAdjacentWeighted)
		{
			zScalar fVal = 0;

			zItMeshScalarField s(*fieldObj, index);

			vector<zItMeshScalarField> adjNeighbours;
			s.getNeighbour_Adjacents(adjNeighbours);

			zPointArray positions;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				positions.push_back(adjNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				zScalar val =  adjNeighbours[i].getValue();
				fVal += val * weights[i];

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldContainedWeighted)
		{		
			zScalar fVal = 0;

			vector<int> containedNeighbours;
			getNeighbour_Contained(samplePos, containedNeighbours);

			zPointArray positions;
			for (int i = 0; i < containedNeighbours.size(); i++)
			{
				zItMeshScalarField s(*fieldObj, containedNeighbours[i]);

				positions.push_back(s.getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0.0;
			for (int i = 0; i < containedNeighbours.size(); i++)
			{		
				zItMeshScalarField s(*fieldObj, containedNeighbours[i]);
				zScalar val = s.getValue();
			
				fVal += (val * weights[i]);

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;

		}

		else throw std::invalid_argument(" error: invalid zFieldValueType.");

		return true;
	}

	template<>
	ZSPACE_INLINE bool zFnMeshField<zVector>::getFieldValue(zPoint &samplePos, zFieldValueType type, zVector& fieldValue)
	{

		bool out = false;

		zItMeshVectorField s(*fieldObj, samplePos);

		int index = s.getId();

		if (type == zFieldIndex)
		{
			zVector fVal;

			fVal = fieldObj->field.fieldValues[index];

			fieldValue = fVal;
		}

		else if (type == zFieldNeighbourWeighted)
		{
			zVector fVal;

			zItMeshVectorField s(*fieldObj, index);

			vector<zItMeshVectorField> ringNeighbours;
			s.getNeighbour_Ring(1, ringNeighbours);

			zPointArray positions;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				positions.push_back(ringNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				zVector val = ringNeighbours[i].getValue();
				fVal += (val * weights[i]);
				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldAdjacentWeighted)
		{
			zVector fVal;

			zItMeshVectorField s(*fieldObj, index);

			vector<zItMeshVectorField> adjNeighbours;
			s.getNeighbour_Adjacents(adjNeighbours);

			zPointArray positions;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				positions.push_back(adjNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				zVector val = adjNeighbours[i].getValue();
				fVal += val * weights[i];

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldContainedWeighted)
		{

			zVector fVal;

			vector<int> containedNeighbours;
			getNeighbour_Contained(samplePos, containedNeighbours);

			zPointArray positions;
			for (int i = 0; i < containedNeighbours.size(); i++)
			{
				zItMeshVectorField s(*fieldObj, containedNeighbours[i]);

				positions.push_back(s.getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w = 0.0;
			for (int i = 0; i < containedNeighbours.size(); i++)
			{				

				zItMeshVectorField s(*fieldObj, containedNeighbours[i]);
				zVector val = s.getValue();

				fVal += (val * weights[i]);

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;

		}

		else throw std::invalid_argument(" error: invalid zFieldValueType.");

		return true;
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValues(vector<T>& fieldValues)
	{
		fieldValues = fieldObj->field.fieldValues;
	}

	template<typename T>
	ZSPACE_INLINE T* zFnMeshField<T>::getRawFieldValues()
	{
		if (numFieldValues() == 0) throw std::invalid_argument(" error: null pointer.");

		return &fieldObj->field.fieldValues[0];
	}

	//---- zScalar specilization for getFieldValue
	template<>
	ZSPACE_INLINE zVector zFnMeshField<zScalar>::getGradient(zItMeshScalarField &s, float epsilon )
	{
		
		bool out = true;
		
		zVector samplePos = s.getPosition();

		int id_X, id_Y;
		s.getIndices(id_X, id_Y);

		if (id_X == 0 || id_Y == 0 || id_X == fieldObj->field.n_X - 1 || id_Y == fieldObj->field.n_Y - 1)
		{
			return zVector();
		}

		zItMeshScalarField s1(*fieldObj, id_X + 1, id_Y);
		zVector samplePos1 = s1.getPosition();
		zScalar fieldVal1 = s1.getValue();
		
		zItMeshScalarField s2(*fieldObj, id_X, id_Y + 1);
		zVector samplePos2 = s2.getPosition();
		zScalar fieldVal2 = s2.getValue();


		zScalar fieldVal = s.getValue();

		zScalar gX = coreUtils.ofMap(samplePos.x + epsilon, samplePos.x, samplePos1.x, fieldVal, fieldVal1) - fieldVal;
		zScalar gY = coreUtils.ofMap(samplePos.y + epsilon, samplePos.y, samplePos2.y, fieldVal, fieldVal2) - fieldVal;

		zVector gradient = zVector(gX, gY, 0);
		gradient /= (2.0 * epsilon);

		return gradient;
	}

	//---- zScalar specilization for getFieldValue
	template<>
	ZSPACE_INLINE vector<zVector> zFnMeshField<zScalar>::getGradients(float epsilon)
	{
		vector<zVector> out;

		for (zItMeshScalarField s(*fieldObj); !s.end(); s++)
		{
			out.push_back(getGradient(s, epsilon));
		}

		return out;
	}

	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::getValuesPerVertexBoolean()
	{
		return setValuesperVertex;
	}

	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::getTriMeshBoolean()
	{
		return triMesh;
	}

	//---- SET METHODS

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::setFieldColorDomain(zDomainColor &colDomain)
	{
		fieldColorDomain = colDomain;
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::setBoundingBox(zPoint &_minBB, zPoint &_maxBB)
	{
		fieldObj->field.minBB = _minBB;
		fieldObj->field.maxBB = _maxBB;
	}
	
	//---- zScalar and zVector specilization for setFieldValues

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::setFieldValues(zScalarArray& fValues)
	{
		if (fValues.size() == numFieldValues())
		{			
			int i = 0;
			for (zItMeshScalarField s(*fieldObj); !s.end(); s++, i++)
			{
				s.setValue(fValues[i]);
			}

			updateColors();
		}

		else throw std::invalid_argument("input fValues size not field scalars.");
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::setFieldValues(vector<zVector>& fValues)
	{
		if (fValues.size() == numFieldValues())
		{
			int i = 0;
			for (zItMeshVectorField s(*fieldObj); !s.end(); s++, i++)
			{
				s.setValue(fValues[i]);
			}

		}

		else throw std::invalid_argument("input fValues size not field vectors.");
	}

	//----  2D IDW FIELD METHODS

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjMesh &inMeshObj, T meshValue, double influence, double power, bool normalise)
	{
		fieldValues.clear();
		zFnMesh inFnMesh(inMeshObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnMesh.numVertices(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				if (r < tempDist)
				{
					double w = pow(r, power);
					wSum += w;
					double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

					d = meshValue * val;

					tempDist = r;
				}

			}

			if (wSum > 0) d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)
		{
			normliseValues(fieldValues);
		}

	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjGraph &inGraphObj, T graphValue, double influence, double power, bool normalise )
	{
		fieldValues.clear();

		zFnGraph inFngraph(inGraphObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFngraph.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFngraph.numVertices(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				if (r < tempDist)
				{
					double w = pow(r, power);
					wSum += w;
					double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

					d = graphValue * val;

					tempDist = r;
				}

			}

			if (wSum > 0) d /= wSum;
			else d = T();

			fieldValues.push_back(d);


		}

		if (normalise)
		{
			normliseValues(fieldValues);
		}


	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, T value, double influence, double power, bool normalise)
	{

		fieldValues.clear();
		zFnPointCloud fnPoints(inPointsObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < fnPoints.numVertices(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

				d += (value * val);
			}


			if (wSum > 0) d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);



	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, vector<T> &values, vector<double>& influences, double power, bool normalise)
	{
		fieldValues.clear();
		zFnPointCloud fnPoints(inPointsObj);

		if (fnPoints.numVertices() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
		if (fnPoints.numVertices() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < fnPoints.numVertices(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influences[j]) / (w)) : 0.0;;

				d += (values[j] * val);
			}


			if (wSum > 0) d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);



	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, T value, double influence, double power, bool normalise)
	{

		fieldValues.clear();


		zVector *meshPositions = fnMesh.getRawVertexPositions();


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

				d += (value * val);
			}


			if (wSum > 0) d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);



	}
	
	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, vector<T> &values, zDoubleArray& influences, double power, bool normalise)
	{
		if (inPositions.size() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
		if (inPositions.size() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

		zVector *meshPositions = fnMesh.getRawVertexPositions();


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influences[j]) / (w)) : 0.0;;

				d += (values[j] * val);
			}


			if (wSum > 0)	d /= wSum;			
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);



	}

	//----  2D SCALAR FIELD METHODS

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zPointArray &inPositions, bool normalise)
	{
		scalars.clear();;

		vector<double> distVals;
		double dMin = 100000;
		double dMax = 0;;

		zVector *meshPositions = fnMesh.getRawVertexPositions();


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			distVals.push_back(100000000);
		}

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			for (int j = 0; j < inPositions.size(); j++)
			{


				double dist = meshPositions[i].squareDistanceTo(inPositions[j]);



				if (dist < distVals[i])
				{
					distVals[i] = dist;
				}
			}
		}

		dMin = coreUtils.zMin(distVals);
		dMax = coreUtils.zMax(distVals);


		for (int j = 0; j < fnMesh.numVertices(); j++)
		{
			double val = coreUtils.ofMap(distVals[j], dMin, dMax, 0.0, 1.0);
			scalars.push_back(val);
		}


		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zPointArray &inPositions, double a, double b, bool normalise)
	{
		scalars.clear();;

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = meshPositions[i].squareDistanceTo(inPositions[j]);

				if (r < tempDist)
				{
					d = F_of_r(r, a, b);
					tempDist = r;
				}

			}

			scalars.push_back(d);
		}

		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, bool normalise)
	{
		scalars.clear();;

		zFnPointCloud fnPoints(inPointsObj);

		vector<double> distVals;
		double dMin = 100000;
		double dMax = 0;;

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			distVals.push_back(10000);
		}

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			for (int j = 0; j < fnPoints.numVertices(); j++)
			{


				double dist = meshPositions[i].squareDistanceTo(inPositions[j]);

				if (dist < distVals[j])
				{
					distVals[j] = dist;
				}
			}
		}

		for (int i = 0; i < distVals.size(); i++)
		{
			dMin = coreUtils.zMin(dMin, distVals[i]);
			dMax = coreUtils.zMax(dMax, distVals[i]);
		}

		for (int j = 0; j < fnMesh.numVertices(); j++)
		{
			double val = coreUtils.ofMap(distVals[j], dMin, dMax, 0.0, 1.0);
			scalars.push_back(val);
		}


		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, double a, double b, bool normalise)
	{
		scalars.clear();;

		zFnPointCloud fnPoints(inPointsObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < fnPoints.numVertices(); j++)
			{
				double r = meshPositions[i].squareDistanceTo(inPositions[j]);

				if (r < tempDist)
				{
					d = F_of_r(r, a, b);
					tempDist = r;
				}

			}

			scalars.push_back(d);
		}

		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise)
	{
		scalars.clear();

		zFnMesh inFnMesh(inMeshObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnMesh.numVertices(); j++)
			{
				double r = meshPositions[i].squareDistanceTo(inPositions[j]);

				if (r < tempDist)
				{
					d = F_of_r(r, a, b);
					tempDist = r;
				}

			}

			scalars.push_back(d);
		}

		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFnGraph.getRawVertexPositions();

		// update values from meta balls

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnGraph.numVertices(); j++)
			{
				double r = meshPositions[i].squareDistanceTo(inPositions[j]);

				if (r < tempDist)
				{
					d = F_of_r(r, a, b);
					//printf("\n F_of_r:  %1.4f ", F_of_r(r, a, b));
					tempDist = r;
				}

			}

			scalars.push_back(d);
		}

		if (normalise)
		{
			normliseValues(scalars);
		}

	}
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise)
	{
		scalars.clear();
		zFnMesh inFnMesh(inMeshObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (zItMeshEdge e(inMeshObj); !e.end(); e++)
			{

				int e0 = e.getHalfEdge(0).getVertex().getId();
				int e1 = e.getHalfEdge(0).getStartVertex().getId();

				zVector closestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], closestPt);


				if (r < tempDist)
				{

					d = F_of_r(r, a, b);

					tempDist = r;
				}
			}

			scalars.push_back(d);

		}

		if (normalise)
		{
			normliseValues(scalars);
		}


	}
	
	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector *meshPositions = fnMesh.getRawVertexPositions();
		zVector *inPositions = inFnGraph.getRawVertexPositions();

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (zItGraphEdge e(inGraphObj); !e.end(); e++)
			{

				int e0 = e.getHalfEdge(0).getVertex().getId();
				int e1 = e.getHalfEdge(0).getStartVertex().getId();

				zVector closestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], closestPt);


				if (r < tempDist)
				{

					d = F_of_r(r, a, b);

					tempDist = r;
				}
			}

			scalars.push_back(d);

		}

		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	//----  2D SD SCALAR FIELD METHODS

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Circle(zScalarArray &scalars, zVector &cen, float r, double annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (annularVal == 0) scalars[i] = getScalar_Circle(cen, meshPositions[i], r);
			else scalars[i] = abs(getScalar_Circle(cen, meshPositions[i], r) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Line(zScalarArray &scalars, zVector &v0, zVector &v1, double annularVal, bool normalise )
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (annularVal == 0) scalars[i] = getScalar_Line(meshPositions[i], v0, v1);
			else scalars[i] = abs(getScalar_Line(meshPositions[i], v0, v1) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Square(zScalarArray &scalars, zVector &dimensions, float annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			zVector p = meshPositions[i];
			if (annularVal == 0) scalars[i] = getScalar_Square(p, dimensions);
			else scalars[i] = abs(getScalar_Square(p, dimensions) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Trapezoid(zScalarArray &scalars, float r1, float r2, float he, float annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			zVector p = meshPositions[i];
			if (annularVal == 0) scalars[i] = getScalar_Trapezoid(p, r1, r2, he);
			else scalars[i] = abs(getScalar_Trapezoid(p, r1, r2, he) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}

	//--- COMPUTE METHODS 
	
	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::checkPositionBounds(zPoint &pos, int &index)
	{
		bool out = true;

		int index_X = floor((pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int index_Y = floor((pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

		if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0) out = false;

		return out;
	}

	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::checkBounds_X(int index_X)
	{
		return (index_X < fieldObj->field.n_X && index_X >= 0);
	}

	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::checkBounds_Y(int index_Y)
	{
		return (index_Y < fieldObj->field.n_Y && index_Y >= 0);
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::computeDomain(vector<T> &values, zDomain <T> &domain)
	{
		domain.min = coreUtils.zMin(values);
		domain.max = coreUtils.zMax(values);
	}

	//---- zScalar and zVector specilization for normliseValues

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::normliseValues(zScalarArray &fieldValues)
	{
		zDomainFloat d;
		computeDomain(fieldValues, d);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = d.max - fieldValues[i];

		computeDomain(fieldValues, d);

		zDomainFloat out(-1.0, 1.0);
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = coreUtils.ofMap(fieldValues[i], d, out);
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}

	//---- zScalar specilization for smoothField

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::smoothField(int numSmooth, double diffuseDamp, zDiffusionType type)
	{
		for (int k = 0; k < numSmooth; k++)
		{
			vector<float> tempValues;

			for( zItMeshScalarField s(*fieldObj); !s.end(); s++)			
			{
				float lapA = 0;

				vector<zItMeshScalarField> ringNeigbours;
				s.getNeighbour_Ring( 1, ringNeigbours);

				for (int j = 0; j < ringNeigbours.size(); j++)
				{
					int id = ringNeigbours[j].getId();
					zScalar val = ringNeigbours[j].getValue();
					
					if (type == zLaplacian)
					{
						if (id != s.getId()) lapA += (val * 1);
						else lapA += (val * -8);
					}
					else if (type == zAverage)
					{
						lapA += (val * 1);
					}
				}



				if (type == zLaplacian)
				{
					float val1 = s.getValue();
					
					float newA = val1 + (lapA * diffuseDamp);
					tempValues.push_back(newA);
				}
				else if (type == zAverage)
				{
					if (lapA != 0) lapA /= (ringNeigbours.size());

					tempValues.push_back(lapA);
				}

			}

			setFieldValues(tempValues);

		}

		updateColors();
	}

	//---- zScalar & zVector specilization for computePositionsInFieldIndex

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::computePositionsInFieldIndex(zPointArray &positions, vector<zPointArray> &fieldIndexPositions)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<zVector> temp;
			fieldIndexPositions.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItMeshScalarField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();			

			fieldIndexPositions[fieldIndex].push_back(positions[i]);
		}
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::computePositionsInFieldIndex(zPointArray &positions, vector<zPointArray> &fieldIndexPositions)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<zVector> temp;
			fieldIndexPositions.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItMeshVectorField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositions[fieldIndex].push_back(positions[i]);
		}
	}

	//---- zScalar & zVector specilization for computePositionIndicesInFieldIndex

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::computePositionIndicesInFieldIndex(zPointArray &positions, vector<zIntArray> &fieldIndexPositionIndicies)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<int> temp;
			fieldIndexPositionIndicies.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItMeshScalarField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositionIndicies[fieldIndex].push_back(i);
		}
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::computePositionIndicesInFieldIndex(zPointArray &positions, vector<zIntArray> &fieldIndexPositionIndicies)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<int> temp;
			fieldIndexPositionIndicies.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItMeshVectorField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositionIndicies[fieldIndex].push_back(i);
		}
	}


	template<typename T>
	ZSPACE_INLINE double zFnMeshField<T>::F_of_r(double &r, double &a, double &b)
	{
		if (0 <= r && r <= b / 3.0)return (a * (1.0 - (3.0 * r * r) / (b*b)));
		if (b / 3.0 <= r && r <= b) return (3 * a / 2 * pow(1.0 - (r / b), 2.0));
		if (b <= r) return 0;


	}

	//---- SCALAR BOOLEAN METHODS

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_union(zScalarArray& scalars0, zScalarArray& scalars1, zScalarArray& scalarsResult, bool normalise)
	{
		vector<float> out;

		for (int i = 0; i < scalars0.size(); i++)
		{
			out.push_back(coreUtils.zMin(scalars0[i], scalars1[i]));
		}

		if (normalise) normliseValues(out);

		scalarsResult = out;
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_subtract(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
	{
		vector<float> out;

		for (int i = 0; i < fieldValues_A.size(); i++)
		{
			out.push_back(coreUtils.zMax(fieldValues_A[i], -1 * fieldValues_B[i]));
		}

		if (normalise) normliseValues(out);

		fieldValues_Result = out;
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_intersect(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
	{
		vector<float> out;

		for (int i = 0; i < fieldValues_A.size(); i++)
		{
			out.push_back(coreUtils.zMax(fieldValues_A[i], fieldValues_B[i]));
		}

		if (normalise) normliseValues(out);

		fieldValues_Result = out;
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_difference(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
	{
		vector<float> AUnionB;
		boolean_union(fieldValues_A, fieldValues_B, AUnionB, normalise);

		vector<float> AIntersectB;
		boolean_intersect(fieldValues_B, fieldValues_A, AIntersectB, normalise);

		vector<float> out;
		boolean_subtract(AUnionB, AIntersectB, out, normalise);

		if (normalise) normliseValues(out);

		fieldValues_Result = out;
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_clipwithPlane(zScalarArray& scalars, zMatrix4& clipPlane)
	{
		int i = 0;

		for (zItMeshVertex v(*fieldObj); !v.end(); v++, i++)
		{
			zVector O = coreUtils.fromMatrix4Column(clipPlane, 3);
			zVector Z = coreUtils.fromMatrix4Column(clipPlane, 2);

			zVector A = v.getPosition() - O;
			double minDist_Plane = A * Z;
			minDist_Plane /= Z.length();

			// printf("\n dist %1.2f ", minDist_Plane);

			if (minDist_Plane > 0)
			{
				scalars[i] = 1;
			}
		}

	}

	//----  UPDATE METHODS

	//---- zScalar specilization for updateColors

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::updateColors()
	{

		vector<float> scalars;
		getFieldValues(scalars);

		if (fnMesh.numVertices() == scalars.size() || fnMesh.numPolygons() == scalars.size())
		{

			computeDomain(scalars, contourValueDomain);

			//convert to HSV

			if (contourValueDomain.min == contourValueDomain.max)
			{
				zColor col(0.5, 0.5, 0.5);

				if (fnMesh.numVertices() == scalars.size()) fnMesh.setVertexColor(col);
				else fnMesh.setFaceColor(col);

				if (fnMesh.numPolygons() == scalars.size())
				{
					contourVertexValues.clear();

					fnMesh.computeVertexColorfromFaceColor();

					for (zItMeshVertex v(*fieldObj); !v.end(); v++)
					{
						vector<int> cFaces;
						v.getConnectedFaces(cFaces);

						double val;

						for (int j = 0; j < cFaces.size(); j++)
						{
							val += scalars[cFaces[j]];
						}

						val /= cFaces.size();

						contourVertexValues.push_back(val);
					}

					computeDomain(contourVertexValues, contourValueDomain);

				}

				if (fnMesh.numVertices() == scalars.size())
				{
					contourVertexValues = scalars;

					fnMesh.computeFaceColorfromVertexColor();
				}



				return;
			}

			else
			{
				fieldColorDomain.min.toHSV(); fieldColorDomain.max.toHSV();

				zColor* cols = fnMesh.getRawVertexColors();
				if (fnMesh.numPolygons() == scalars.size()) cols = fnMesh.getRawFaceColors();

				for (int i = 0; i < scalars.size(); i++)
				{

					if (scalars[i] < contourValueDomain.min) cols[i] = fieldColorDomain.min;
					else if (scalars[i] > contourValueDomain.max) cols[i] = fieldColorDomain.max;
					else
					{
						cols[i] = coreUtils.blendColor(scalars[i], contourValueDomain, fieldColorDomain, zHSV);
					}

				}

				if (fnMesh.numPolygons() == scalars.size())
				{
					contourVertexValues.clear();

					fnMesh.computeVertexColorfromFaceColor();

					for (zItMeshVertex v(*fieldObj); !v.end(); v++)
					{
						vector<int> cFaces;
						v.getConnectedFaces(cFaces);

						double val;

						for (int j = 0; j < cFaces.size(); j++)
						{
							val += scalars[cFaces[j]];
						}

						val /= cFaces.size();

						contourVertexValues.push_back(val);
					}

					computeDomain(contourVertexValues, contourValueDomain);

				}

				if (fnMesh.numVertices() == scalars.size())
				{
					contourVertexValues = scalars;

					fnMesh.computeFaceColorfromVertexColor();
				}
			}




		}

	}

	//---- CONTOUR METHODS
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsocontour(zObjGraph &coutourGraphObj, float inThreshold)
	{
		if (contourVertexValues.size() == 0) return;
		if (contourVertexValues.size() != numFieldValues())
		{
			throw std::invalid_argument(" error: invalid contour condition. Call updateColors method. ");
			return;
		}

		float threshold = coreUtils.ofMap(inThreshold, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max);

		vector<zVector> pos;
		vector<int> edgeConnects;

		vector<int> edgetoIsoGraphVertexId;

		zVector *positions = fnMesh.getRawVertexPositions();

		// compute positions
		int i = 0;
		for (zItMeshEdge e(*fieldObj); !e.end(); e++)
		{
			edgetoIsoGraphVertexId.push_back(-1);
			edgetoIsoGraphVertexId.push_back(-1);


			int eV0 = e.getHalfEdge(0).getVertex().getId();
			int eV1 = e.getHalfEdge(0).getStartVertex().getId();



			float scalar_lower = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV0] : contourVertexValues[eV1];
			float scalar_higher = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV1] : contourVertexValues[eV0];;

			bool chkSplitEdge = (scalar_lower <= threshold && scalar_higher > threshold) ? true : false;

			if (chkSplitEdge)
			{
				// calculate split point

				int scalar_lower_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV0 : eV1;
				int scalar_higher_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV1 : eV0;

				zVector scalar_lower_vertPos = positions[scalar_lower_vertId];
				zVector scalar_higher_vertPos = positions[scalar_higher_vertId];

				float scaleVal = coreUtils.ofMap(threshold, scalar_lower, scalar_higher, 0.0f, 1.0f);

				zVector e = scalar_higher_vertPos - scalar_lower_vertPos;
				double eLen = e.length();

				e.normalize();

				zVector newPos = scalar_lower_vertPos + (e * eLen * scaleVal);
				pos.push_back(newPos);

				// map edge to isographVertex
				edgetoIsoGraphVertexId[i] = pos.size() - 1;
				edgetoIsoGraphVertexId[i + 1] = pos.size() - 1;

			}

			i += 2;
		}

		// compute edgeConnects
		for (zItMeshFace f(*fieldObj); !f.end(); f++)
		{

			vector<int> fEdges;
			f.getHalfEdges(fEdges);
			vector<int> tempConnects;

			for (int j = 0; j < fEdges.size(); j++)
			{
				if (edgetoIsoGraphVertexId[fEdges[j]] != -1)
					tempConnects.push_back(edgetoIsoGraphVertexId[fEdges[j]]);
			}

			//printf("\n face %i | %i ", i, tempConnects.size());

			if (tempConnects.size() == 2)
			{
				edgeConnects.push_back(tempConnects[0]);
				edgeConnects.push_back(tempConnects[1]);
			}

		}

		zFnGraph tempFn(coutourGraphObj);
		tempFn.clear(); // clear memory if the mobject exists.


		tempFn.create(pos, edgeConnects);
		printf("\n %i %i ", tempFn.numVertices(), tempFn.numEdges());
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsolineMesh(zObjMesh &coutourMeshObj, float inThreshold, bool invertMesh)
	{
		if (contourVertexValues.size() == 0) return;
		if (contourVertexValues.size() != numFieldValues())
		{
			throw std::invalid_argument(" error: invalid contour condition.  Call updateColors method.");
			return;
		}

		zFnMesh tempFn(coutourMeshObj);
		tempFn.clear(); // clear memory if the mobject exists.

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unordered_map <string, int> positionVertex;

		float threshold = coreUtils.ofMap(inThreshold, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max);


		for (zItMeshFace f(*fieldObj); !f.end(); f++)
		{
			getIsolinePoly(f, positions, polyConnects, polyCounts, positionVertex, threshold, invertMesh);
		}


		tempFn.create(positions, polyCounts, polyConnects);;
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsobandMesh(zObjMesh &coutourMeshObj, float inThresholdLow, float inThresholdHigh, bool invertMesh)
	{
		if (contourVertexValues.size() == 0) return;

		if (contourVertexValues.size() != numFieldValues())
		{
			throw std::invalid_argument(" error: invalid contour condition.  Call updateColors method.");
			return;
		}

		zFnMesh tempFn(coutourMeshObj);
		tempFn.clear(); // clear memory if the mobject exists.

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unordered_map <string, int> positionVertex;

		float thresholdLow = coreUtils.ofMap(inThresholdLow, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max);
		float thresholdHigh = coreUtils.ofMap(inThresholdHigh, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max);

		if (invertMesh)
		{
			//zObjMesh m1 = getIsolineMesh((thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, false);
			//zObjMesh m2 = getIsolineMesh( (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow, true);

			///*if (m1.numVertices() > 0 && m2.numVertices() > 0)
			//{
			//	out = combineDisjointMesh(m1, m2);

			//	return out;
			//}

			//else*/ if (m1.numVertices() > 0) return m1;

			//else if (m2.numVertices() > 0) return m2;


		}

		if (!invertMesh)
		{
			for (zItMeshFace f(*fieldObj); !f.end(); f++)
			{
				getIsobandPoly(f, positions, polyConnects, polyCounts, positionVertex, (thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow);
			}

			tempFn.create(positions, polyCounts, polyConnects);;


		}


	}

	//---- PROTECTED FACTORY METHODS

	//---- zScalar specilization for toBMP

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::toBMP(string outfilename)
	{
		int resX = fieldObj->field.n_X;
		int resY = fieldObj->field.n_Y;

		zUtilsBMP bmp(resX, resY);

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		zColor *col;
		if (setValuesperVertex) col = fnMesh.getRawVertexColors();
		else col = fnMesh.getRawFaceColors();

		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{

				if (setValuesperVertex)
				{
					
					zItMeshScalarField s(*fieldObj, x, y);
					int vertexId = s.getId();

					//printf("\n %i %i %i ", x, y, faceId);

					// blue
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = col[vertexId].b * 255;;

					// green
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = col[vertexId].g * 255;;

					// red
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = col[vertexId].r * 255;;


					// alpha
					if (channels == 4)
					{
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = col[vertexId].a * 255;;

					}

				}

				else
				{
					zItMeshScalarField s(*fieldObj, x, y);
					int faceId = s.getId();

					// blue
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = col[faceId].b * 255;

					// green
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = col[faceId].g * 255;

					// red
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = col[faceId].r * 255;

					// alpha
					if (channels == 4)
					{
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = col[faceId].a * 255;
					}
				}

			}

		}

		bmp.write(outfilename.c_str());
	}

	//---- zScalar specilization for fromBMP

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::fromBMP(string infilename)
	{
		zUtilsBMP bmp(infilename.c_str());

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		int resX = bmp.bmp_info_header.width;
		int resY = bmp.bmp_info_header.height;

		if (resX == 0 || resY == 0) return;


		if (numFieldValues() != resX * resY) create(1, 1, resX, resY, zVector(), 1, setValuesperVertex);


		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{

				double r = (double)bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] / 255;
	
				zItMeshScalarField s(*fieldObj, x, y);
				s.setValue(r);

				

			}
		}

		updateColors();
	}

	template<typename T>
	ZSPACE_INLINE void zFnMeshField<T>::createFieldMesh()
	{
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		zVector minBB, maxBB;
		double unit_X, unit_Y;
		int n_X, n_Y;

		getUnitDistances(unit_X, unit_Y);
		getResolution(n_X, n_Y);

		int resX = n_X;
		int resY = n_Y;

		if (!setValuesperVertex)
		{
			resX++;
			resY++;
		}

		getBoundingBox(minBB, maxBB);

		zVector unitVec = zVector(unit_X, unit_Y, 0);
		zVector startPt = minBB;;

		if (!setValuesperVertex)startPt -= (unitVec * 0.5);

		for (int i = 0; i < resX; i++)
		{
			for (int j = 0; j < resY; j++)
			{
				zVector pos;
				pos.x = startPt.x + i * unitVec.x;
				pos.y = startPt.y + j * unitVec.y;

				positions.push_back(pos);

			}
		}

		/// poly connects

		for (int i = 0; i < resX - 1; i++)
		{
			for (int j = 0; j < resY - 1; j++)
			{
				int v0 = (i * resY) + j;
				int v1 = ((i + 1) * resY) + j;

				int v2 = v1 + 1;
				int v3 = v0 + 1;

				if (triMesh)
				{
					polyConnects.push_back(v0);
					polyConnects.push_back(v1);
					polyConnects.push_back(v3);
					polyCounts.push_back(3);

					polyConnects.push_back(v1);
					polyConnects.push_back(v2);
					polyConnects.push_back(v3);
					polyCounts.push_back(3);

				}
				else
				{
					polyConnects.push_back(v0);
					polyConnects.push_back(v1);
					polyConnects.push_back(v2);
					polyConnects.push_back(v3);

					polyCounts.push_back(4);
				}

			}
		}

		fnMesh.create(positions, polyCounts, polyConnects, true);

		printf("\n fieldmesh: v %i e %i f %i", fnMesh.numVertices(), fnMesh.numEdges(), fnMesh.numPolygons());
	}

	//---- PROTECTED SCALAR METHODS

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Circle(zPoint &cen, zPoint &p, float r)
	{
		return ((p - cen).length() - r);
	}

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Line(zPoint &p, zPoint &v0, zPoint &v1)
	{
		zVector pa = p - v0;
		zVector ba = v1 - v0;

		float h = coreUtils.ofClamp((pa* ba) / (ba* ba), 0.0f, 1.0f);

		zVector out = pa - (ba*h);


		return (out.length());
	}

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Square(zPoint &p, zVector &dimensions)
	{
		p.x = abs(p.x); p.y = abs(p.y); p.z = abs(p.z);

		zVector out;
		out.x = coreUtils.zMax<float>(coreUtils.zMax<float>(p.x - dimensions.x, 0), coreUtils.zMax<float>(p.y - dimensions.y, 0));
		out.y = 0;
		out.z = 0;

		return(out.length());
	}

	template<>
	ZSPACE_INLINE double zFnMeshField<zScalar>::getScalar_Trapezoid(zPoint &p, float &r1, float &r2, float &he)
	{
		zVector k1 = zVector(r2, he, 0);
		zVector k2 = zVector((r2 - r1), (2.0 * he), 0);

		p.x = abs(p.x);
		zVector ca = zVector(p.x - coreUtils.zMin(p.x, (p.y < 0.0) ? r1 : r2), abs(p.y) - he, 0.0);
		zVector cb = p - k1 + k2 * coreUtils.ofClamp(((k1 - p) * k2) / (k2*k2), 0.0f, 1.0f);

		double s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;

		double out = s * sqrt(coreUtils.zMin((ca * ca), (cb * cb)));

		return out;
	}

	template<>
	ZSPACE_INLINE int zFnMeshField<zScalar>::getIsolineCase(bool vertexBinary[4])
	{
		int out = -1;

		if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 0;

		if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 1;

		if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 2;

		if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 3;

		if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 4;

		if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 5;

		if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 6;

		if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 7;

		if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 8;

		if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 9;

		if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 10;

		if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 11;

		if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 12;

		if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 13;

		if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 14;

		if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 15;

		return out;
	}

	template<>
	ZSPACE_INLINE int zFnMeshField<zScalar>::getIsobandCase(int vertexTernary[4])
	{
		int out = -1;

		// No Contour
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 0;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 1;


		// Single Triangle

		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 2;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 3;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 4;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 5;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 6;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 7;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 8;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 9;

		// Single Trapezoid

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 10;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 11;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 12;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 13;


		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 14;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 15;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 16;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 17;

		// Single Rectangle

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 18;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 19;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 20;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 21;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 22;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 23;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 24;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 25;


		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 26;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 27;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 28;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 29;

		// Single Square

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 30;

		// Single Pentagon

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 31;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 32;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 33;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 34;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 35;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 36;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 37;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 38;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 39;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 40;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 41;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 42;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 43;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 44;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 45;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 46;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 47;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 48;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 49;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 50;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 51;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 52;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 53;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 54;

		// Single Hexagon

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 55;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 56;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 57;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 58;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 59;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 60;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 61;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 62;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 63;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 64;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 65;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 66;


		// Saddles:  1 or 2 polygon

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 67;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 68;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 69;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 70;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 71;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 72;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 73;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 74;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 75;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 76;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 77;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 78;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 79;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 80;

		return out;
	}

	template<>
	ZSPACE_INLINE zVector zFnMeshField<zScalar>::getContourPosition(float &threshold, zVector& vertex_lower, zVector& vertex_higher, float& thresholdLow, float& thresholdHigh)
	{

		float scaleVal = coreUtils.ofMap(threshold, thresholdLow, thresholdHigh, 0.0f, 1.0f);

		zVector e = vertex_higher - vertex_lower;
		double edgeLen = e.length();
		e.normalize();

		return (vertex_lower + (e * edgeLen *scaleVal));
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsolinePoly(zItMeshFace& f , zPointArray &positions, zIntArray &polyConnects, zIntArray &polyCounts, unordered_map <string, int> &positionVertex, float &threshold, bool invertMesh)
	{
		vector<zItMeshVertex> fVerts;
		f.getVertices(fVerts);

		if (fVerts.size() != 4) return;

		// chk if all the face vertices are below the threshold
		bool vertexBinary[4];
		float averageScalar = 0;

		for (int j = 0; j < fVerts.size(); j++)
		{
			if (contourVertexValues[fVerts[j].getId()] < threshold)
			{
				vertexBinary[j] = (invertMesh) ? false : true;
			}
			else vertexBinary[j] = (invertMesh) ? true : false;

			averageScalar += contourVertexValues[fVerts[j].getId()];
		}

		averageScalar /= fVerts.size();

		int MS_case = getIsolineCase(vertexBinary);

		vector<zVector> newPositions;
		vector<zVector> newPositions2;


		// CASE 0
		if (MS_case == 0)
		{
			for (int j = 0; j < fVerts.size(); j++)
			{
				newPositions.push_back(fVerts[j].getPosition());
			}

		}

		// CASE 1
		if (MS_case == 1)
		{
			zVector v0 = fVerts[0].getPosition();
			float s0 = fVerts[0].getColor().r;

			zVector v1 = fVerts[1].getPosition();
			float s1 = fVerts[1].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			newPositions.push_back(fVerts[2].getPosition());

			newPositions.push_back(fVerts[3].getPosition());

			v1 = fVerts[3].getPosition();
			s1 = fVerts[3].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);


		}

		// CASE 2
		if (MS_case == 2)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = fVerts[0].getColor().r;

			zVector v1 = fVerts[1].getPosition();
			float s1 = fVerts[1].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[2].getPosition();
			s0 = fVerts[2].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[2].getPosition());

			newPositions.push_back(fVerts[3].getPosition());

		}

		// CASE 3
		if (MS_case == 3)
		{
			zVector v0 = fVerts[3].getPosition();
			float s0 = fVerts[3].getColor().r;

			zVector v1 = fVerts[0].getPosition();
			float s1 = fVerts[0].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[2].getPosition();
			s0 = fVerts[2].getColor().r;

			v1 = fVerts[1].getPosition();
			s1 = fVerts[1].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[2].getPosition());

			newPositions.push_back(fVerts[3].getPosition());


		}

		// CASE 4
		if (MS_case == 4)
		{
			newPositions.push_back(fVerts[0].getPosition());

			newPositions.push_back(fVerts[1].getPosition());

			zVector v0 = fVerts[1].getPosition();
			float s0 = fVerts[1].getColor().r;

			zVector v1 = fVerts[2].getPosition();
			float s1 = fVerts[2].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[3].getPosition();
			s0 = fVerts[3].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[3].getPosition());

		}

		// CASE 5
		if (MS_case == 5)
		{
			// SADDLE CASE 

			int SaddleCase = 1/*(averageScalar < threshold) ? 0 : 1*/;
			if (invertMesh) SaddleCase = 0/*(averageScalar < threshold) ? 1 : 0*/;

			if (SaddleCase == 0)
			{
				// hex

				zVector v0 = fVerts[1].getPosition();
				float s0 = fVerts[1].getColor().r;

				zVector v1 = fVerts[0].getPosition();
				float s1 = fVerts[0].getColor().r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[1].getPosition());

				v1 = fVerts[2].getPosition();
				s1 = fVerts[2].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[3].getPosition();
				s0 = fVerts[3].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[3].getPosition());

				v1 = fVerts[0].getPosition();
				s1 = fVerts[0].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}
			if (SaddleCase == 1)
			{
				// tri 1
				zVector v0 = fVerts[1].getPosition();
				float s0 = fVerts[1].getColor().r;

				zVector v1 = fVerts[0].getPosition();
				float s1 = fVerts[0].getColor().r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[1].getPosition());


				v1 = fVerts[2].getPosition();
				s1 = fVerts[2].getColor().r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2
				v0 = fVerts[3].getPosition();
				s0 = fVerts[3].getColor().r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fVerts[3].getPosition());

				v1 = fVerts[0].getPosition();
				s1 = fVerts[0].getColor().r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 6
		if (MS_case == 6)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = fVerts[0].getColor().r;
			zVector v1 = fVerts[1].getPosition();
			float s1 = fVerts[1].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[3].getPosition();
			s0 = fVerts[3].getColor().r;
			v1 = fVerts[2].getPosition();
			s1 = fVerts[2].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[3].getPosition());

		}


		// CASE 7
		if (MS_case == 7)
		{
			zVector v0 = fVerts[3].getPosition();
			float s0 = fVerts[3].getColor().r;
			zVector v1 = fVerts[2].getPosition();
			float s1 = fVerts[2].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[3].getPosition());

			v1 = fVerts[0].getPosition();
			s1 = fVerts[0].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 8
		if (MS_case == 8)
		{
			newPositions.push_back(fVerts[0].getPosition());

			newPositions.push_back(fVerts[1].getPosition());

			newPositions.push_back(fVerts[2].getPosition());

			zVector v0 = fVerts[2].getPosition();
			float s0 = fVerts[2].getColor().r;
			zVector v1 = fVerts[3].getPosition();
			float s1 = fVerts[3].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[0].getPosition();
			s0 = fVerts[0].getColor().r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 9
		if (MS_case == 9)
		{
			zVector v0 = fVerts[1].getPosition();
			float s0 = fVerts[1].getColor().r;
			zVector v1 = fVerts[0].getPosition();
			float s1 = fVerts[0].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			newPositions.push_back(fVerts[2].getPosition());

			v0 = fVerts[2].getPosition();
			s0 = fVerts[2].getColor().r;
			v1 = fVerts[3].getPosition();
			s1 = fVerts[3].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 10
		if (MS_case == 10)
		{
			int SaddleCase = 1 /*(averageScalar < threshold) ? 0 : 1*/;
			if (invertMesh) SaddleCase = 0 /*(averageScalar < threshold) ? 1 : 0*/;

			if (SaddleCase == 0)
			{
				// hex
				newPositions.push_back(fVerts[0].getPosition());

				zVector v0 = fVerts[0].getPosition();
				float s0 = fVerts[0].getColor().r;
				zVector v1 = fVerts[1].getPosition();
				float s1 = fVerts[1].getColor().r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[2].getPosition();
				s0 = fVerts[2].getColor().r;
				v1 = fVerts[1].getPosition();
				s1 = fVerts[1].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[2].getPosition());

				v1 = fVerts[3].getPosition();
				s1 = fVerts[3].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[0].getPosition();
				s0 = fVerts[0].getColor().r;
				v1 = fVerts[3].getPosition();
				s1 = fVerts[3].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);
			}

			if (SaddleCase == 1)
			{
				// tri 1

				newPositions.push_back(fVerts[0].getPosition());

				zVector v0 = fVerts[0].getPosition();
				float s0 = fVerts[0].getColor().r;
				zVector v1 = fVerts[1].getPosition();
				float s1 = fVerts[1].getColor().r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v1 = fVerts[3].getPosition();
				s1 = fVerts[3].getColor().r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2

				v0 = fVerts[2].getPosition();
				s0 = fVerts[2].getColor().r;
				v1 = fVerts[1].getPosition();
				s1 = fVerts[1].getColor().r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fVerts[2].getPosition());

				v1 = fVerts[3].getPosition();
				s1 = fVerts[3].getColor().r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 11
		if (MS_case == 11)
		{
			zVector v0 = fVerts[2].getPosition();
			float s0 = fVerts[2].getColor().r;
			zVector v1 = fVerts[1].getPosition();
			float s1 = fVerts[1].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);


			newPositions.push_back(fVerts[2].getPosition());

			v1 = fVerts[3].getPosition();
			s1 = fVerts[3].getColor().r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);



		}

		// CASE 12
		if (MS_case == 12)
		{
			newPositions.push_back(fVerts[0].getPosition());

			newPositions.push_back(fVerts[1].getPosition());

			zVector v0 = fVerts[1].getPosition();
			float s0 = fVerts[1].getColor().r;
			zVector v1 = fVerts[2].getPosition();
			float s1 = fVerts[2].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[0].getPosition();
			s0 = fVerts[0].getColor().r;
			v1 = fVerts[3].getPosition();
			s1 = fVerts[3].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 13
		if (MS_case == 13)
		{
			zVector v0 = fVerts[1].getPosition();
			float s0 = fVerts[1].getColor().r;
			zVector v1 = fVerts[0].getPosition();
			float s1 = fVerts[0].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			v1 = fVerts[2].getPosition();
			s1 = fVerts[2].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 14
		if (MS_case == 14)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = fVerts[0].getColor().r;
			zVector v1 = fVerts[1].getPosition();
			float s1 = fVerts[1].getColor().r;

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v1 = fVerts[3].getPosition();
			s1 = fVerts[3].getColor().r;

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 15
		if (MS_case == 15)
		{
			// No Veritices to be added 

		}


		// check for edge lengths


		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				int next = (i + 1) % newPositions.size();

				if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
				{
					newPositions.erase(newPositions.begin() + i);
				}
			}
		}


		// compute poly 
		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				zVector p0 = newPositions[i];
				int v0;

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions.size());
		}


		// only if there are 2 tris Case : 5,10

		// Edge Length Check

		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				int next = (i + 1) % newPositions2.size();


				if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
				{
					newPositions2.erase(newPositions.begin() + i);
				}

			}
		}


		// compute poly 
		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				zVector p0 = newPositions2[i];
				int v0;

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions2.size());
		}



	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsobandPoly(zItMeshFace& f, zPointArray &positions, zIntArray &polyConnects, zIntArray &polyCounts, unordered_map <string, int> &positionVertex, float &thresholdLow, float &thresholdHigh)
	{
		vector<zItMeshVertex> fVerts;
		f.getVertices(fVerts);

		//printf("\n fVs: %i ", fVerts.size());

		if (fVerts.size() != 4) return;

		// chk if all the face vertices are below the threshold
		int vertexTernary[4];
		float averageScalar = 0;

		for (int j = 0; j < fVerts.size(); j++)
		{
			if (contourVertexValues[fVerts[j].getId()] <= thresholdLow)
			{
				vertexTernary[j] = 0;
			}

			else if (contourVertexValues[fVerts[j].getId()] >= thresholdHigh)
			{
				vertexTernary[j] = 2;
			}
			else vertexTernary[j] = 1;

			averageScalar += contourVertexValues[fVerts[j].getId()];
		}

		averageScalar /= fVerts.size();

		int MS_case = getIsobandCase(vertexTernary);

		vector<zVector> newPositions;
		vector<zVector> newPositions2;

		//printf("\n MS_case : %i ", MS_case);
		//if (MS_case < 72 || MS_case > 80) return;




		// No Contours CASE 0 & 1
		if (MS_case == 0 || MS_case == 1)
		{
			// NO Vertices to be added
		}

		// Single Triangle CASE 2 to 9 
		if (MS_case >= 2 && MS_case <= 9)
		{
			int startID = -1;
			float threshold = thresholdHigh;

			if (MS_case == 2 || MS_case == 6)startID = 0;
			if (MS_case == 3 || MS_case == 7)startID = 1;
			if (MS_case == 4 || MS_case == 8)startID = 2;
			if (MS_case == 5 || MS_case == 9)startID = 3;

			if (MS_case > 5) threshold = thresholdLow;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();


			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;

			zVector v1 = fVerts[prevID].getPosition();
			float s1 = fVerts[prevID].getColor().r;


			zVector pos0 = (getContourPosition(threshold, v0, v1, s0, s1));

			zVector pos1 = fVerts[startID].getPosition();


			v1 = fVerts[nextID].getPosition();
			s1 = fVerts[nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));


			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);

		}


		// Single Trapezoid CASE 10 to 17

		if (MS_case >= 10 && MS_case <= 17)
		{
			int startID = -1;

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case == 10 || MS_case == 14) startID = 0;
			if (MS_case == 11 || MS_case == 15) startID = 1;
			if (MS_case == 12 || MS_case == 16) startID = 2;
			if (MS_case == 13 || MS_case == 17) startID = 3;

			if (MS_case > 13)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;

			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;


			zVector pos0 = (getContourPosition(threshold0, v0, v1, s0, s1));

			zVector pos1 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;


			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

			zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));


			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);

		}


		// Single Rectangle CASE 18 to 29
		if (MS_case >= 18 && MS_case <= 25)
		{
			int startID = -1;
			float threshold = thresholdLow;
			if (MS_case > 21) threshold = thresholdHigh;

			if (MS_case == 18 || MS_case == 22) startID = 0;
			if (MS_case == 19 || MS_case == 23) startID = 1;
			if (MS_case == 20 || MS_case == 24) startID = 2;
			if (MS_case == 21 || MS_case == 25) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();
			zVector pos1 = fVerts[nextID].getPosition();

			zVector v0 = fVerts[nextID].getPosition();
			float s0 = fVerts[nextID].getColor().r;

			zVector v1 = fVerts[next_nextID].getPosition();
			float s1 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;
			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);

		}

		// Single Rectangle CASE 26 to 29
		if (MS_case >= 26 && MS_case <= 29)
		{
			int startID = -1;

			if (MS_case == 26) startID = 0;
			if (MS_case == 27) startID = 1;
			if (MS_case == 28) startID = 2;
			if (MS_case == 29) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;

			zVector v1 = fVerts[prevID].getPosition();
			float s1 = fVerts[prevID].getColor().r;

			zVector pos0 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

			v0 = fVerts[nextID].getPosition();
			s0 = fVerts[nextID].getColor().r;
			v1 = fVerts[next_nextID].getPosition();
			s1 = fVerts[next_nextID].getColor().r;


			zVector pos1 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
			zVector pos2 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
		}

		// Single Square CASE 30
		if (MS_case == 30)
		{
			for (int j = 0; j < fVerts.size(); j++)
				newPositions.push_back(fVerts[j].getPosition());
		}

		// Single Pentagon CASE 31 to 38
		if (MS_case >= 31 && MS_case <= 38)
		{
			int startID = -1;

			float threshold = thresholdHigh;
			if (MS_case > 34)threshold = thresholdLow;

			if (MS_case == 31 || MS_case == 35) startID = 0;
			if (MS_case == 32 || MS_case == 36) startID = 1;
			if (MS_case == 33 || MS_case == 37) startID = 2;
			if (MS_case == 34 || MS_case == 38) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();
			zVector pos1 = fVerts[nextID].getPosition();

			zVector v0 = fVerts[nextID].getPosition();
			float s0 = fVerts[nextID].getColor().r;

			zVector v1 = fVerts[next_nextID].getPosition();
			float s1 = fVerts[next_nextID].getColor().r;

			zVector pos2 = getContourPosition(threshold, v0, v1, s0, s1);

			v0 = fVerts[prevID].getPosition();
			s0 = fVerts[prevID].getColor().r;

			zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

			zVector pos4 = fVerts[prevID].getPosition();;

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Pentagon CASE 39 to 46
		if (MS_case >= 39 && MS_case <= 46)
		{
			int startID = -1;

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case > 42)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 39 || MS_case == 43) startID = 3;
			if (MS_case == 40 || MS_case == 44) startID = 2;
			if (MS_case == 41 || MS_case == 45) startID = 1;
			if (MS_case == 42 || MS_case == 46) startID = 0;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;

			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;
			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;

			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Pentagon CASE 47 to54
		if (MS_case >= 47 && MS_case <= 54)
		{
			int startID = -1;

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case > 50)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 47 || MS_case == 51) startID = 3;
			if (MS_case == 48 || MS_case == 52) startID = 2;
			if (MS_case == 49 || MS_case == 53) startID = 1;
			if (MS_case == 50 || MS_case == 54) startID = 0;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;
			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;
			v1 = fVerts[next_nextID].getPosition();
			s1 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;
			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos4 = getContourPosition(threshold0, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Hexagon CASE 55 to 62
		if (MS_case >= 55 && MS_case <= 62)
		{
			int startID = -1;

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case > 58)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 55 || MS_case == 59) startID = 0;
			if (MS_case == 56 || MS_case == 60) startID = 1;
			if (MS_case == 57 || MS_case == 61) startID = 2;
			if (MS_case == 58 || MS_case == 62) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();
			zVector pos1 = fVerts[nextID].getPosition();

			zVector v0 = fVerts[nextID].getPosition();
			float s0 = fVerts[nextID].getColor().r;
			zVector v1 = fVerts[next_nextID].getPosition();
			float s1 = fVerts[next_nextID].getColor().r;

			zVector pos2 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fVerts[prevID].getPosition();
			s0 = fVerts[prevID].getColor().r;

			zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));

			v1 = fVerts[startID].getPosition();
			s1 = fVerts[startID].getColor().r;

			zVector pos5 = getContourPosition(threshold0, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
			newPositions.push_back(pos5);
		}

		// Single Hexagon CASE 63 to 66
		if (MS_case >= 63 && MS_case <= 66)
		{
			int startID = -1;


			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case % 2 == 0)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			if (MS_case == 63 || MS_case == 64) startID = 0;
			if (MS_case == 65 || MS_case == 66) startID = 1;



			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;
			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));

			zVector pos3 = fVerts[next_nextID].getPosition();

			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;

			zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
			newPositions.push_back(pos5);
		}

		// SADDLE CASE 67 to 68 : 8 Sides
		if (MS_case >= 67 && MS_case <= 68)
		{

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case % 2 == 0)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int SaddleCase = -1;;

			if (averageScalar < thresholdLow) SaddleCase = 0;
			else if (averageScalar > thresholdHigh) SaddleCase = 2;
			else SaddleCase = 1;

			int startID = 0;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;
			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos3 = getContourPosition(threshold0, v0, v1, s0, s1);

			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));
			zVector pos5 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;

			zVector pos6 = getContourPosition(threshold1, v0, v1, s0, s1);
			zVector pos7 = (getContourPosition(threshold0, v0, v1, s0, s1));


			if (SaddleCase == 0)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

				// quad 2
				newPositions2.push_back(pos4);
				newPositions2.push_back(pos5);
				newPositions2.push_back(pos6);
				newPositions2.push_back(pos7);
			}


			if (SaddleCase == 2)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos6);
				newPositions.push_back(pos7);

				// quad 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
				newPositions2.push_back(pos5);
			}


			if (SaddleCase == 1)
			{
				// octagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);
				newPositions.push_back(pos7);
			}

		}

		// SADDLE CASE 69 to 72 : 6 Sides
		if (MS_case >= 69 && MS_case <= 72)
		{

			float threshold = thresholdLow;

			if (MS_case > 70)
			{
				threshold = thresholdHigh;
			}

			int SaddleCase = -1;;

			if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
			else SaddleCase = 0;

			int startID = (MS_case % 2 == 0) ? 1 : 0;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fVerts[startID].getPosition();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;
			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos1 = getContourPosition(threshold, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));

			zVector pos3 = fVerts[next_nextID].getPosition();

			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos4 = (getContourPosition(threshold, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;

			zVector pos5 = getContourPosition(threshold, v0, v1, s0, s1);


			if (SaddleCase == 0)
			{
				// tri 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos5);

				// tri 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
			}


			if (SaddleCase == 1)
			{
				// hexagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);

			}

		}

		// SADDLE CASE 73 to 80 : 7 Sides
		if (MS_case >= 73 && MS_case <= 80)
		{

			float threshold0 = thresholdLow;
			float threshold1 = thresholdHigh;

			if (MS_case > 76)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int SaddleCase = -1;;

			if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
			else SaddleCase = 0;

			int startID = -1;

			if (MS_case == 73 || MS_case == 77) startID = 0;
			if (MS_case == 74 || MS_case == 78) startID = 2;
			if (MS_case == 75 || MS_case == 79) startID = 1;
			if (MS_case == 76 || MS_case == 80) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector v0 = fVerts[startID].getPosition();
			float s0 = fVerts[startID].getColor().r;
			zVector v1 = fVerts[nextID].getPosition();
			float s1 = fVerts[nextID].getColor().r;

			zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fVerts[next_nextID].getPosition();
			s0 = fVerts[next_nextID].getColor().r;

			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

			zVector pos3 = fVerts[next_nextID].getPosition();

			v1 = fVerts[prevID].getPosition();
			s1 = fVerts[prevID].getColor().r;

			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fVerts[startID].getPosition();
			s0 = fVerts[startID].getColor().r;

			zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);
			zVector pos6 = getContourPosition(threshold0, v0, v1, s0, s1);

			if (SaddleCase == 0)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);

				// tri 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
			}


			if (SaddleCase == 1)
			{
				// heptagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);

			}

		}

		// check for edge lengths

		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				int next = (i + 1) % newPositions.size();

				if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
				{
					newPositions.erase(newPositions.begin() + i);

				}
			}
		}


		// compute poly 
		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				zVector p0 = newPositions[i];
				int v0;

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions.size());
		}


		// only if there are 2 tris Case : 5,10

		// Edge Length Check


		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				int next = (i + 1) % newPositions2.size();


				if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
				{
					newPositions2.erase(newPositions.begin() + i);


				}

			}
		}


		// compute poly 
		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				zVector p0 = newPositions2[i];
				int v0;

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions2.size());
		}

	}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zFnMeshField<zVector>;

	template class zFnMeshField<zScalar>;

#endif
}