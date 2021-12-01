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

			fVal = (fVal == 0.0) ? 0.0 : fVal / w;

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

			fVal = (fVal == 0.0) ? 0.0 : fVal / w;

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

			fVal = (fVal == 0.0) ? 0.0 : fVal / w;

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


	template<>
	ZSPACE_INLINE bool zFnMeshField<zScalar>::getScalarValue(zScalarArray& scalars, zPoint& samplePos, zFieldValueType type, zScalar& fieldValue)
	{
		if (scalars.size() != numFieldValues())
		{
			 throw std::invalid_argument(" error: scalars and field value size dont match.");
			return false;
		}


		bool out = false;

		zItMeshScalarField s(*fieldObj, samplePos);

		int index = s.getId();

		if (type == zFieldIndex)
		{
			
			fieldValue = scalars[index];
		}

		else if (type == zFieldNeighbourWeighted)
		{
			zScalar fVal = 0;

			zItMeshScalarField s(*fieldObj, index);

			vector<zItMeshScalarField> ringNeighbours;
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
				zScalar val = scalars[ringNeighbours[i].getId()];
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
				zScalar val = scalars[adjNeighbours[i].getId()];
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
				zScalar val = scalars[s.getId()];

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

			//printf("\n working  update colors %i ", contourVertexValues.size());;
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zPointArray &inPositions, float offset, bool normalise)
	{
		scalars.clear();;

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = meshPositions[i].distanceTo(inPositions[j]);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;
					//d = F_of_r(r, a, b);
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, float offset,  bool normalise)
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
				double r = meshPositions[i].distanceTo(inPositions[j]);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;
					//d = F_of_r(r, a, b);
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjMesh &inMeshObj, float offset,  bool normalise)
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
				double r = meshPositions[i].distanceTo(inPositions[j]);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;
					//d = F_of_r(r, a, b);
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjGraph &inGraphObj, float offset, bool normalise)
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
				double r = meshPositions[i].distanceTo(inPositions[j]);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;
					//d = F_of_r(r, a, b);
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjMesh &inMeshObj,float offset, bool normalise)
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

				float r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], closestPt);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;
					//d = F_of_r(r, a, b);
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
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjGraph &inGraphObj, float offset, bool normalise)
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

				if (e.getLength() < EPS) continue;

				zVector closestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], closestPt);

				r = r - offset;

				if (r < tempDist)
				{
					d = r;					
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
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_depthPolygon(zScalarArray& scalars, zScalarArray& polygonScalars, zObjGraph& inGraphObj, int startVertexId, float offset, zVector planeNorm , bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = (startV.getHalfEdge());

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			//double d = 0.0;
			double tempDist = 10000;

			zPoint closestPt;
			zItGraphHalfEdge closestHe;

			for (auto& orientedHE : orientedHalfEdges)
			{

				int e1 = orientedHE.getVertex().getId();
				int e0 = orientedHE.getStartVertex().getId();

				if (orientedHE.getLength() < EPS) continue;

				zVector tempClosestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], tempClosestPt);

				if (r < tempDist)
				{
					tempDist = r;
					closestPt = tempClosestPt;
					closestHe = orientedHE;
				}
			}

			// compute if mesh point below or above closestHE
			zVector X = closestHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = Z ^ X;
			Y.normalize();

			double planeMinDist = coreUtils.minDist_Point_Plane(meshPositions[i], closestPt, Y);
			double dSign = coreUtils.zSign(planeMinDist);

			zPoint p0 = closestPt;
			zPoint p1 = (dSign > 0) ? closestPt + (Y * 1) : closestPt + (Y * -1);

			// ray march https://michaelwalczyk.com/blog-ray-marching.html
			zVector ray = p1 - p0;
			ray.normalize();

			zPoint samplePoint = closestPt;
			float sdfValue;
			getFieldValue( samplePoint, zFieldValueType::zFieldContainedWeighted, sdfValue);

			float step = abs(sdfValue) * 0.75;;
			float distanceTravelled = 0.0;

			bool rayExit = false;

			do
			{
				samplePoint += ray * step;
				distanceTravelled += step;

				printf("\n%i  %1.3f ",i, distanceTravelled);

				getFieldValue(samplePoint, zFieldValueType::zFieldContainedWeighted, sdfValue);

				step = abs(sdfValue) * 0.75;;

				if (abs(sdfValue) < distanceTolerance) rayExit = true; // hit
				
				if (distanceTravelled > 1.0) // miss
				{
					rayExit = true;
				}

				
			} while (!rayExit);

			float d = distanceTravelled - offset;
			printf("\n %1.3f ", d);
			scalars.push_back(distanceTravelled - offset);

		}


		if (normalise)
		{
			normliseValues(scalars);
		}


	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_VariableDepth(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, zDomainFloat offset1, zDomainFloat offset2, zVector planeNorm , bool normalise)
	{

		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = (startV.getHalfEdge());

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;
		
		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			zPoint closestPt;
			zItGraphHalfEdge closestHe;

			for (auto &orientedHE : orientedHalfEdges)
			{

				int e0 = orientedHE.getVertex().getId();
				int e1 = orientedHE.getStartVertex().getId();

				if (orientedHE.getLength() < EPS) continue;

				zVector tempClosestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], tempClosestPt);

				if (r < tempDist)
				{					
					tempDist = r;
					closestPt = tempClosestPt;
					closestHe = orientedHE;
				}
			}

			// compute if mesh point below or above closestHE
			zVector X = closestHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = Z ^ X;
			Y.normalize();
			
			double dSign = coreUtils.zSign(coreUtils.minDist_Point_Plane(meshPositions[i], closestPt, Y));

			zDomainFloat offset = (dSign > 0) ? offset1 : offset2;

			int heStartV = closestHe.getStartVertex().getId();
			float distFromStart = closestPt.distanceTo(inPositions[heStartV]);
			distFromStart += vertex_distanceFromStart[heStartV];

			float mappedoffset = coreUtils.ofMap(distFromStart, inDomain, offset);
			d = tempDist - mappedoffset;

			scalars.push_back(d);

		}


		if (normalise)
		{
			normliseValues(scalars);
		}
	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_VariableDepth(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, vector<zDomainFloat>& thicknessOffsets1, vector<zDomainFloat>& thicknessOffsets2, vector<zDomainFloat> &intervals, zVector planeNorm, bool normalise)
	{

		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = (startV.getHalfEdge());

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			zPoint closestPt;
			zItGraphHalfEdge closestHe;

			for (auto& orientedHE : orientedHalfEdges)
			{

				int e1 = orientedHE.getVertex().getId();
				int e0 = orientedHE.getStartVertex().getId();

				if (orientedHE.getLength() < EPS) continue;

				zVector tempClosestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], tempClosestPt);

				if (r < tempDist)
				{
					tempDist = r;
					closestPt = tempClosestPt;
					closestHe = orientedHE;
				}
			}

			// compute if mesh point below or above closestHE
			zVector X = closestHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = Z ^ X;
			Y.normalize();

			double planeMinDist = coreUtils.minDist_Point_Plane(meshPositions[i], closestPt, Y);
			double dSign = coreUtils.zSign(planeMinDist);

			int heStartV = closestHe.getStartVertex().getId();
			float distFromStart = closestPt.distanceTo(inPositions[heStartV]);
			distFromStart += vertex_distanceFromStart[heStartV];

			// get Interval index
			float intervalVal = distFromStart / graphLen;
			if (intervalVal < 0 || intervalVal > 1) cout << endl << intervalVal;
			int arrayIndex = -1;
			zDomainFloat interval;
			for (int k = 0; k < intervals.size(); k++)
			{
				if (intervalVal >= intervals[k].min && intervalVal <= intervals[k].max)
				{
					arrayIndex = k;
					interval = intervals[k];
				}
					
			}
			

			zDomainFloat offset = (dSign > 0) ? thicknessOffsets1[arrayIndex] : thicknessOffsets2[arrayIndex];

			

			float mappedoffset = coreUtils.ofMap(intervalVal, interval, offset);
			//d = abs(planeMinDist) - mappedoffset;

			d = tempDist - mappedoffset;

			scalars.push_back(d);

		}


		if (normalise)
		{
			normliseValues(scalars);
		}
	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_VariableDepth(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, zObjGraph& inThkGraphObj, zVector planeNorm, bool normalise)
	{

		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = (startV.getHalfEdge());

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		// update values from edge distance
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			zPoint closestPt;
			zItGraphHalfEdge closestHe;

			for (auto& orientedHE : orientedHalfEdges)
			{

				int e1 = orientedHE.getVertex().getId();
				int e0 = orientedHE.getStartVertex().getId();

				if (orientedHE.getLength() < EPS) continue;

				zVector tempClosestPt;

				double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], tempClosestPt);

				if (r < tempDist)
				{
					tempDist = r;
					closestPt = tempClosestPt;
					closestHe = orientedHE;
				}
			}

			// compute if mesh point below or above closestHE
			zVector X = closestHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = Z ^ X;
			Y.normalize();

			double planeMinDist = coreUtils.minDist_Point_Plane(meshPositions[i], closestPt, Y);
			double dSign = coreUtils.zSign(planeMinDist);

			zPoint p0 = closestPt;
			zPoint p1 = (dSign > 0) ? closestPt + (Y * 1) : closestPt + (Y * -1);

			// find interection 

			//int heStartV = closestHe.getStartVertex().getId();
			//float distFromStart = closestPt.distanceTo(inPositions[heStartV]);
			//distFromStart += vertex_distanceFromStart[heStartV];

			//// get Interval index
			//float intervalVal = distFromStart / graphLen;
			//if (intervalVal < 0 || intervalVal > 1) cout << endl << intervalVal;
			//int arrayIndex = -1;
			//zDomainFloat interval;
			//for (int k = 0; k < intervals.size(); k++)
			//{
			//	if (intervalVal >= intervals[k].min && intervalVal <= intervals[k].max)
			//	{
			//		arrayIndex = k;
			//		interval = intervals[k];
			//	}

			//}


			//zDomainFloat offset = (dSign > 0) ? thicknessOffsets1[arrayIndex] : thicknessOffsets2[arrayIndex];



			//float mappedoffset = coreUtils.ofMap(intervalVal, interval, offset);
			////d = abs(planeMinDist) - mappedoffset;

			//d = tempDist - mappedoffset;

			//scalars.push_back(d);

		}


		if (normalise)
		{
			normliseValues(scalars);
		}
	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_SineInfill(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, zDomainFloat offset1, zDomainFloat offset2, int numTriangles, float transY, zVector planeNorm, bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = startV.getHalfEdge();

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		//float wave_period = 0.03;
		//int numLayers = floor(graphLen / wave_period);		

		//printf("\n nT %i ", numTriangles);
		int numLayers = ((numTriangles + 1) * 4) + 1 /*31*/;
		float wave_period = graphLen / numLayers;

		zPointArray positions;
		zIntArray edgeConnects;

		//positions.push_back(startV.getPosition());

		zPoint pOnCurve = startV.getPosition();
		walkHe = startV.getHalfEdge();



		int end = (floor(numLayers / 4) * 4) - 1;

		//printf(" %1.4f , %i  %i ,  %1.2f %1.2f %1.2f", graphLen, numLayers, end, pOnCurve.x, pOnCurve.y, pOnCurve.z);


		exit = false;

		for (int j = 0; j < end - 1; j++)
		{
			//if (exit) continue;

			zPoint eEndPoint = walkHe.getVertex().getPosition();
			float distance_increment = /*( j ==0 ) ? wave_period * 2 :*/  wave_period;


			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				walkHe = walkHe.getNext();
				eEndPoint = walkHe.getVertex().getPosition();

			}

			int heStartV = walkHe.getStartVertex().getId();

			//printf("\n counter %i  %1.4f %1.4f ", counter, distance_increment, vertex_distanceFromStart[heStartV]);
			zVector X = walkHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = X ^ Z;
			Y.normalize();

			//O
			zPoint O = pOnCurve + X * distance_increment;

			if (j == numLayers - 1) O = walkHe.getVertex().getPosition();


			if (j >= 0)
			{


				float distFromStart = O.distanceTo(inPositions[heStartV]);
				distFromStart += vertex_distanceFromStart[heStartV];

				if (j == numLayers - 1) distFromStart = graphLen;

				float n = (distFromStart / wave_period) + 0.5;

				float a = (n) * (HALF_PI);


				float d = sin(a);
				d = coreUtils.zSign(d);
				zDomainFloat offset = (d > 0) ? offset2 : offset1;

				//printf("\n %1.2f  %1.4f %1.4f ", n, a, d);

				float mappedoffset = coreUtils.ofMap(distFromStart, inDomain, offset);

				if (positions.size() > 0)
				{
					edgeConnects.push_back(positions.size() - 1);
					edgeConnects.push_back(positions.size());
				}

				float trans = -transY/*0.1*/;
				zPoint addP = O + (Y * trans);

				addP += (coreUtils.zSign(d) > 0) ? Y * d * (mappedoffset + (trans * -0.5)) : (Y * d * mappedoffset);

				if (positions.size() % 4 == 2) addP += (X * (-0.5 * wave_period));

				positions.push_back(addP);

				//if (graphLen - distFromStart < 3.5 * wave_period) exit = true;
			}



			pOnCurve = O;


		}


		zObjGraph tempGraph;
		zFnGraph tempFn(tempGraph);

		tempFn.create(positions, edgeConnects);

		getScalarsAsEdgeDistance(scalars, tempGraph, 0.01, normalise);

	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_SineInfill(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, vector<zDomainFloat>& thicknessOffsets1, vector<zDomainFloat>& thicknessOffsets2, vector<zDomainFloat>& intervals, int numTriangles, float transY, zVector planeNorm, bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = startV.getHalfEdge();

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		//float wave_period = 0.03;
		//int numLayers = floor(graphLen / wave_period);		
		
		//printf("\n nT %i ", numTriangles);
		int numLayers = ((numTriangles + 1)*4) + 1 /*31*/;
		float wave_period = graphLen / numLayers;

		zPointArray positions;
		zIntArray edgeConnects;

		//positions.push_back(startV.getPosition());

		zPoint pOnCurve = startV.getPosition();
		walkHe = startV.getHalfEdge();

		

		int end = (floor(numLayers / 4) * 4) - 1;

		//printf(" %1.4f , %i  %i ,  %1.2f %1.2f %1.2f", graphLen, numLayers, end, pOnCurve.x, pOnCurve.y, pOnCurve.z);


		exit = false;

		for (int j = 0; j < end; j++)
		{
			//if (exit) continue;

			zPoint eEndPoint = walkHe.getVertex().getPosition();
			float distance_increment = /*( j ==0 ) ? wave_period * 2 :*/  wave_period;
			
			
			while(pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				walkHe =  walkHe.getNext();
				eEndPoint = walkHe.getVertex().getPosition();
				
			} 

			int heStartV = walkHe.getStartVertex().getId();

			//printf("\n counter %i  %1.4f %1.4f ", counter, distance_increment, vertex_distanceFromStart[heStartV]);
			zVector X = walkHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm : planeNorm * -1;
			Z.normalize();

			zVector Y = X ^ Z;
			Y.normalize();

			//O
			zPoint O = pOnCurve + X * distance_increment;

			if (j == numLayers - 1) O = walkHe.getVertex().getPosition();
					

			if (j > 0)
			{

				
				float distFromStart = O.distanceTo(inPositions[heStartV]);
				distFromStart += vertex_distanceFromStart[heStartV];

				if (j == numLayers - 1) distFromStart = graphLen;

				float n = (distFromStart / wave_period) + 0.5;

				float a = (n) * (PI * 0.5);
				//printf("\n n %1.2f a %1.2f  %1.2f ",n, a, sin(a));


				float d = sin(a);
				d = coreUtils.zSign(d);
				//zDomainFloat offset = (d > 0) ? offset2 : offset1;


				// get Interval index
				float intervalVal = distFromStart / graphLen;
				if (intervalVal < 0 || intervalVal > 1) cout << endl << intervalVal;
				int arrayIndex = -1;
				zDomainFloat interval;
				for (int k = 0; k < intervals.size(); k++)
				{
					if (intervalVal >= intervals[k].min && intervalVal <= intervals[k].max)
					{
						arrayIndex = k;
						interval = intervals[k];
					}

				}


				zDomainFloat offset = (d > 0) ? thicknessOffsets2[arrayIndex] : thicknessOffsets1[arrayIndex];


				float mappedoffset = coreUtils.ofMap(intervalVal, interval, offset);


				//printf("\n %1.2f  %1.4f %1.4f ", n, a, d);

				//float mappedoffset = coreUtils.ofMap(distFromStart, inDomain, offset);

				if (positions.size() > 0)
				{
					edgeConnects.push_back(positions.size() - 1);
					edgeConnects.push_back(positions.size());
				}

				float trans = -transY/*0.1*/;
				zPoint addP = O /*+ (Y * trans)*/;

				addP += /*(coreUtils.zSign(d) > 0) ? Y * d * (mappedoffset + (trans * -0.5)) : */(Y * d * mappedoffset);

				if (positions.size() % 4 == 0) addP += (X * (-0.5 * wave_period));

				positions.push_back(addP);

				//if (graphLen - distFromStart < 3.5 * wave_period) exit = true;
			}
			


			pOnCurve = O;

			
		}
			

		zObjGraph tempGraph;
		zFnGraph tempFn(tempGraph);

		tempFn.create(positions, edgeConnects);

		getScalarsAsEdgeDistance(scalars, tempGraph, 0.01, normalise);

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_SineInfill(zScalarArray& scalars, zScalarArray& polygonScalars, zObjGraph& inGraphObj, zObjGraph& inPolyObj, int startVertexId, float numTriangles,float offset, zVector planeNorm , bool normalise)
	{
		//setFieldValues(polygonScalars);
		
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = startV.getHalfEdge();

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		/*zVectorArray vNormals;

		for (zItGraphVertex v(inGraphObj); !v.end(); v++)
		{
			zItGraphHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges); 

			zVector n;

			if (cHEdges.size() == 1)
			{
				zVector X = cHEdges[0].getVector();
				X.normalize();

				zVector Z = (X * worldX > 0) ? planeNorm * -1 : planeNorm * 1;
				Z.normalize();

				n = X ^ Z;
				n.normalize();
			}
			else
			{
				for (auto& cHE : cHEdges)
				{
					n += cHE.getVector();
				}

				n /= cHEdges.size();
				n.normalize();
				
			}

			vNormals.push_back(n);
		}*/

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);

		zDomainFloat inDomain(0.0, graphLen);

		//float wave_period = 0.03;
		//int numLayers = floor(graphLen / wave_period);		

		//printf("\n nT %i ", numTriangles);
		int numLayers = ((numTriangles + 1) * 4) + 2 /*31*/;
		float wave_period = graphLen / numLayers;

		zPointArray positions;
		zIntArray edgeConnects;

		//positions.push_back(startV.getPosition());

		zPoint pOnCurve = startV.getPosition();
		walkHe = startV.getHalfEdge();



		int end = (floor(numLayers / 4) * 4) - 1;

		//printf(" %1.4f , %i  %i ,  %1.2f %1.2f %1.2f", graphLen, numLayers, end, pOnCurve.x, pOnCurve.y, pOnCurve.z);

		zFnGraph fnPolgGraph(inPolyObj);
		zItGraphHalfEdge start(inPolyObj, 0);

		exit = false;

		for (int j = 0; j < end; j++)
		{
			//if (exit) continue;

			zPoint eEndPoint = walkHe.getVertex().getPosition();
			float distance_increment = ( j % 2  ==1 ) ? wave_period * 1.0 :  wave_period * 1.0;

			if (numTriangles == 2.25)
			{
				// for 8 | 3.25  , 9 | 4.00 , 10 | 4.40  11 | 3.5,  0,3| 1.25, 4 | 1.5 , 1, 2, | 1.5,   14,15,|0.75, 17 |0.6,  16 |0.55
				if (j == 0)
				{
					distance_increment = wave_period * 1.5;
				}
				//for 0 | 3.0 ,  1 | 1.5, 2 | 2.0, 10,11 | 0.65,  4 |2.5 ,3 | 3.0  , 8 |2.0 14 | 1.5, 15,|2.0 ,  16 | 3.65, 9 | 1.1, 17 |3.0
				if (j == 5)
				{
					distance_increment = wave_period * 2.5;
				}
			}
			

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				walkHe = walkHe.getNext();
				eEndPoint = walkHe.getVertex().getPosition();

			}

			int heStartV = walkHe.getStartVertex().getId();
			int heEndV = walkHe.getVertex().getId();

			//printf("\n counter %i  %1.4f %1.4f ", counter, distance_increment, vertex_distanceFromStart[heStartV]);
			zVector X = walkHe.getVector();
			X.normalize();

			zVector Z = (X * worldX > 0) ? planeNorm * -1 : planeNorm * 1;
			Z.normalize();

			zVector Y = X ^ Z;
			Y.normalize();



			//O
			zPoint O = pOnCurve + X * distance_increment;

			if (j == numLayers - 1) O = walkHe.getVertex().getPosition();


			/*float eLen = walkHe.getLength();
			

			float factor = eLen / distFromStart;

			zVector Y = vNormals[heStartV] * (1 - factor) + vNormals[heEndV] * (factor);
			Y.normalize();*/

			if (j > 0)
			{


				float distFromStart = O.distanceTo(inPositions[heStartV]);
				distFromStart += vertex_distanceFromStart[heStartV];

				if (j == numLayers - 1) distFromStart = graphLen;

				float n = positions.size() + 2 /*(distFromStart / wave_period)*/ ;

				float a =   (n + 0.5) * (PI * 0.5);
				//printf("\n n %1.2f a %1.2f  %1.2f ",n, a, sin(a));


				float d = sin(a);
				d = coreUtils.zSign(d);
				//zDomainFloat offset = (d > 0) ? offset2 : offset1;

				zPoint p0 = O;
				zPoint p1 = ((d > 0)) ? O + (Y * 1) : O + (Y * -1);
				zVector ray = p1 - p0;
				ray.normalize();
				
				// sdf ray march https://michaelwalczyk.com/blog-ray-marching.html
									

				//zPoint samplePoint = O;
				//float sdfValue;
				//getFieldValue(samplePoint, zFieldValueType::zFieldContainedWeighted, sdfValue);
				//
				//cout << samplePoint;
				//printf("\n in %i sdf  %1.3f   ", j, sdfValue);

				//float step = abs(sdfValue) * 0.75;;
				//float distanceTravelled = 0.0;

				//bool rayExit = false;

				//do
				//{
				//	printf("\n in %i  %1.3f %1.3f  ", j, distanceTravelled,step);

				//	samplePoint += ray * step;
				//	distanceTravelled += step;

				//	

				//	getFieldValue(samplePoint, zFieldValueType::zFieldContainedWeighted, sdfValue);

				//	step = abs(sdfValue) * 0.75;;

				//	if (abs(sdfValue) < distanceTolerance) rayExit = true; // hit

				//	if (distanceTravelled > 1.0) // miss
				//	{
				//		rayExit = true;
				//	}


				//} while (!rayExit);

				//float mappedoffset = distanceTravelled;
				//printf("\n out %1.3f ", mappedoffset);


				// ray intersections

				zItGraphHalfEdge he = start;
				float intDist;
				bool rayExit = false;
				do
				{
					zPoint e0 = he.getStartVertex().getPosition();
					zPoint e1 = he.getVertex().getPosition();

					double uA, uB;
					bool check = coreUtils.line_lineClosestPoints(p0, p1, e0, e1,uA, uB);
					if (check)
					{
						if (uB >= 0.0 && uB <= 1.0 && uA >= 0.0 && uA <= 1.0)
						{
							zPoint intPt = p0 + ray * uA;
							intDist = p0.distanceTo(intPt);

							rayExit = true;
						}						
											
						
					}

					he = he.getNext();

					if (he == start)rayExit = true;;

				} while (!rayExit);

				float mappedoffset = intDist;
				//printf("\n out %1.3f ", mappedoffset);

				if (positions.size() > 0)
				{
					edgeConnects.push_back(positions.size() - 1);
					edgeConnects.push_back(positions.size());
				}

			
				zPoint addP = O /*+ (Y * trans)*/;

				if (d > 0) mappedoffset -= offset;

				addP += (Y * d * mappedoffset);

				//if (positions.size() % 4 == 0) addP += (X * (-0.5 * wave_period));

				positions.push_back(addP);
				
			}



			pOnCurve = O;


		}


		zObjGraph tempGraph;
		zFnGraph tempFn(tempGraph);

		tempFn.create(positions, edgeConnects);

		getScalarsAsEdgeDistance(scalars, tempGraph, 0.01, normalise);

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_Infill(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth,  bool normalise)
	{
		
		scalars.clear();

		zVector norm(0,0,1);
		zVector worldX(1, 0, 0);
		

		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		float topIncrement = topLength / (numTriangles);
		
		int topNumPts = numTriangles * 3 + numTriangles * 2;
		zIntArray top_indexInContainer;
		top_indexInContainer.assign(topNumPts, -1);

		zPointArray topPts;

		int topID = 0;
		zItGraphHalfEdge tHe = topHE[topID];
		topPts.push_back(tHe.getStartVertex().getPosition());

		
		zPoint pOnCurve = topPts[topID];
		topID++;	

		float pWidth = 0.95 * printWidth;

		int counter =0;
		for (int i = 0; i < numTriangles * 3; i++)
		{
			float distance_increment = topIncrement * 0.5;
			

			if (i % 3 == 0) distance_increment  = (i == 1) ? pWidth * 1.0 : pWidth;
			else if(i % 3 == 1) distance_increment -=  (i == 1) ? pWidth : pWidth *0.5 ;
			else if (i % 3 == 2) distance_increment -= (i == numTriangles * 3 -1) ? pWidth * 1.0 : pWidth * 0.5;

			//length += distance_increment;

			zPoint eEndPoint = tHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tHe = topHE[topID];
				eEndPoint = tHe.getVertex().getPosition();
				topID++;
			}

			zVector he_vec = tHe.getVector();
			he_vec.normalize();

			//O
			pOnCurve = pOnCurve + he_vec * distance_increment;

			/*if (i == numTriangles * 3 - 1)
			{
				pOnCurve = topHE[topHE.size() - 1].getVertex().getPosition();
				he_vec = topHE[topHE.size() - 1].getVector();
				he_vec.normalize();
			}*/

			if (i % 3 == 0 || i % 3 == 2)
			{
				zVector X = he_vec;

				zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
				Z.normalize();

				zVector Y = X ^ Z;
				Y.normalize();

				top_indexInContainer[counter] = topPts.size();
				if( i == numTriangles * 3 -1) topPts.push_back(pOnCurve + Y * 2.5 * printWidth);
				else topPts.push_back(pOnCurve + Y * 2.5 * printWidth);

				counter++;
			}
			

			top_indexInContainer[counter] = topPts.size();
			topPts.push_back(pOnCurve);
				
			counter++;
		}
		
		/// BOTTOM B WITH GUIDE

		float bottomIncrement = bottomLength / (numTriangles);
		float d = (bottomIncrement - maxTriangleLength) * 0.5;
		float dTest = printWidth * 1.5;
		float bottomOverlapLength = (d < dTest) ? dTest : d;

		//printf("\n incr %1.4f %1.4f  %1.4f %1.4f", bottomLength, bottomIncrement, bottomOverlapLength, maxTriangleLength);

		int bottomNumPts = numTriangles * 3;
		zIntArray bottom_indexInContainer;
		bottom_indexInContainer.assign(bottomNumPts, -1);


		zPointArray bottomPts;

		int bottomID = 0;
		zItGraphHalfEdge bHe = bottomHE[bottomID];
		bottomPts.push_back(bHe.getStartVertex().getPosition());

		pOnCurve = bottomPts[bottomID];
		bottomID++;
		zColor red(1, 0, 0, 1);

		for (int i = 0; i < numTriangles; i++)
		{
			// find red vertex
			bool redVertex = false;
			
			while (!redVertex)
			{
				if (bHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					bHe = bottomHE[bottomID];
					bottomID++;
				}
			}
				
			// first point
			float distance_increment = bottomOverlapLength;

			zItGraphHalfEdge tempHe = bHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();
			
			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();
			
			pOnCurve = pOnCurve + he_vec * distance_increment;

			bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = bHe.getVertex().getPosition();

			bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// Third Point
			
			distance_increment = bottomOverlapLength;
			tempHe = bHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			//
			bHe = bottomHE[bottomID];
			bottomID++;
		}

		
		/// BOTTOM A NO GUIDE

		//float bottomIncrement = bottomLength / (numTriangles);
		//float d = (bottomIncrement - maxTriangleLength )* 0.5;
		//float dTest = printWidth * 1.5;
		//float bottomOverlapLength = (d < dTest) ? dTest : d;

		////printf("\n incr %1.4f %1.4f  %1.4f %1.4f", bottomLength, bottomIncrement, bottomOverlapLength, maxTriangleLength);

		//int bottomNumPts = numTriangles * 3;
		//zIntArray bottom_indexInContainer;
		//bottom_indexInContainer.assign(bottomNumPts, -1);
		//	

		//zPointArray bottomPts;

		//int bottomID = 0;
		//zItGraphHalfEdge bHe = bottomHE[bottomID];
		//bottomPts.push_back(bHe.getStartVertex().getPosition());

		//pOnCurve = bottomPts[bottomID];
		//bottomID++;
		//
		//for (int i = 0; i < bottomNumPts; i++)
		//{
		//	float distance_increment = bottomIncrement;
		//	if (i == 0) distance_increment = bottomIncrement / 2;

		//	if (i % 3 == 0) distance_increment -= (i == 0) ? bottomOverlapLength : bottomOverlapLength * 2;
		//	else distance_increment = bottomOverlapLength;


		//	zPoint eEndPoint = bHe.getVertex().getPosition();			

		//	while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
		//	{
		//		distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
		//		pOnCurve = eEndPoint;

		//		bottomPts.push_back(pOnCurve);

		//		bHe = bottomHE[bottomID];
		//		eEndPoint = bHe.getVertex().getPosition();
		//		bottomID++;
		//	}

		//	zVector he_vec = bHe.getVector();
		//	he_vec.normalize();

		//	//O
		//	pOnCurve = pOnCurve + he_vec * distance_increment;

		//	bottom_indexInContainer[i] = bottomPts.size();
		//	bottomPts.push_back(pOnCurve);			
		//	
		//}


		for (int i = 0; i < numTriangles; i++)
		{
			zPointArray gPositions;
			
			gPositions.push_back(topPts[top_indexInContainer[i * 5+ 0]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 5 + 1]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 5 + 2]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 5 + 4]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 5 + 3]]);

			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 1]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 0]]);

			zIntArray gEdgeCOnnects;
			
			for (int i = 0; i < gPositions.size(); i++)
			{
				int next = (i + 1) % gPositions.size();

				gEdgeCOnnects.push_back(i);
				gEdgeCOnnects.push_back(next);
			}

			zObjGraph tempGraph;
			zFnGraph fnTempGraph(tempGraph);
			fnTempGraph.create(gPositions, gEdgeCOnnects);

			zScalarArray triField;
			getScalars_Polygon(triField, tempGraph, false);

			if (i == 0) scalars = triField;
			else
			{
				zScalarArray result;
				boolean_union(scalars, triField, result, false);

				scalars.clear();
				scalars = result;

			}
		}




		/*zItGraphHalfEdge topHe;
		zItGraphHalfEdge bottomHe;
		
		for (int i = 0; i < 2; i++)
		{
			zItGraphVertex v(inPolyObj, startVerts[i]);
			
			zItGraphHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);
						
			if (i == 0)
			{
				bottomHe = (cHEdges[0].getVertex().getId() == startVerts[1]) ? cHEdges[1] : cHEdges[0];
			}

			if (i == 1)
			{
				topHe = (cHEdges[0].getVertex().getId() == startVerts[0]) ? cHEdges[1] : cHEdges[0];
			}

		}

		zBoolArray vertVisited;
		vertVisited.assign(inFnGraph.numVertices(), false);

		vertVisited[startVerts[0]] = true;
		vertVisited[startVerts[1]] = true;

		bool exit = false;
		int counter = 0;
		
		while (!exit)
		{
			if (bottomHe.getLength() < 0.05)
			{
				vertVisited[bottomHe.getVertex().getId()] = true;
				bottomHe = bottomHe.getNext();
			}
			if (topHe.getLength() < 0.05)
			{
				vertVisited[topHe.getVertex().getId()] = true;
				topHe = topHe.getNext();
			}

			

			bottomHe.getEdge().setColor(zColor(1, 0, 0, 1));
			topHe.getEdge().setColor(zColor(1, 0, 0, 1));

			zPoint bottom_start = bottomHe.getStartVertex().getPosition();
			zVector bottom_vec = bottomHe.getVector();
			bottom_vec.normalize();
			float bottom_len = bottomHe.getLength();

			zPoint top_start = topHe.getStartVertex().getPosition();
			zVector top_vec = topHe.getVector();
			top_vec.normalize();
			float top_len = topHe.getLength();

			int numTrianglesperEdge = ceil (top_len / maxTriangleLength);

			float increment = 1.0 / (2 * numTrianglesperEdge);			

			for (int i = 0; i < numTrianglesperEdge; i++)
			{
				zPoint p0 = bottom_start + (bottom_vec * (i * 2 * increment * bottom_len));
				
				float fac = ((i + 1) * increment) + (i * increment);
				zPoint p1 = top_start + (top_vec * (fac * top_len));
				zPoint p2 = bottom_start + (bottom_vec * ((i + 1) * 2 * increment * bottom_len));

				zPointArray gPositions;
				gPositions.push_back(p0);
				gPositions.push_back(p1);
				gPositions.push_back(p2);

				zIntArray gEdgeCOnnects = { 0,1,1,2,2,0 };

				zObjGraph tempGraph;
				zFnGraph fnTempGraph(tempGraph);
				fnTempGraph.create(gPositions, gEdgeCOnnects);

				zScalarArray triField;
				getScalars_Polygon(triField, tempGraph, false);

				if (counter == 0) scalars = triField;
				else
				{
					zScalarArray result;
					boolean_union(scalars, triField, result, false);

					scalars.clear();
					scalars = result;

				}

				counter++;
			}

			

			
			
			vertVisited[bottomHe.getVertex().getId()] = true;
			vertVisited[topHe.getVertex().getId()] = true;

			bottomHe = bottomHe.getNext();
			topHe = topHe.getNext();

			if (vertVisited[bottomHe.getVertex().getId()]) exit = true;
			if (vertVisited[topHe.getVertex().getId()]) exit = true;

			zItGraphVertex v = bottomHe.getVertex();
			if (topHe.getStartVertex() == v) exit = true;
			if (topHe.getVertex() == v) exit = true;
		}*/

		
	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_InfillBoundary(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth, bool normalise)
	{

		scalars.clear();

		zVector norm(0, 0, 1);
		zVector worldX(1, 0, 0);


		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();


		/// TOP B WITH GUIDE
		zColor red(1, 0, 0, 1);
		float dTest = printWidth * 1.0;
		float topOverlapLength = dTest;

		int topNumPts = numTriangles * 3 ;
		zIntArray top_indexInContainer;
		top_indexInContainer.assign(topNumPts, -1);

		zPointArray topPts;

		int topID = 0;
		zItGraphHalfEdge tHe = topHE[topID];
		topPts.push_back(tHe.getStartVertex().getPosition());


		zPoint pOnCurve = topPts[topID];
		topID++;

		int counter = 0;
		for (int i = 0; i < numTriangles ; i++)
		{
			// find red vertex
			bool redVertex = false;

			while (!redVertex)
			{
				if (tHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					tHe = topHE[topID];
					topID++;
				}
			}

			// first point
			float distance_increment = (i ==2 || i == 3 ) ?  topOverlapLength * 5.5 : topOverlapLength;
			//if (i == 3) distance_increment = topOverlapLength * 5.5;

			zItGraphHalfEdge tempHe = tHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			top_indexInContainer[(i * 3) + 0] = topPts.size();
			topPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = tHe.getVertex().getPosition();

			top_indexInContainer[(i * 3) + 1] = topPts.size();
			topPts.push_back(pOnCurve);

			// Third Point

			distance_increment = (i == 0) ? topOverlapLength * 4.5 : topOverlapLength * 2.5;

			tempHe = tHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			top_indexInContainer[(i * 3) + 2] = topPts.size();
			topPts.push_back(pOnCurve);

			//
			tHe = topHE[topID];
			topID++;
		}

		/// BOTTOM B WITH GUIDE

		dTest = printWidth * 2.5;
		float bottomOverlapLength =  dTest ;

		//printf("\n incr %1.4f %1.4f  %1.4f %1.4f", bottomLength, bottomIncrement, bottomOverlapLength, maxTriangleLength);

		int bottomNumPts = (numTriangles + 1) * 3;
		zIntArray bottom_indexInContainer;
		bottom_indexInContainer.assign(bottomNumPts, -1);


		zPointArray bottomPts;

		int bottomID = 0;
		zItGraphHalfEdge bHe = bottomHE[bottomID];
		bottomPts.push_back(bHe.getStartVertex().getPosition());

		pOnCurve = bottomPts[bottomID];
		bottomID++;
		

		for (int i = 0; i < numTriangles + 1 ; i++)
		{
			if (i == 0)
			{
				
				float distance_increment = dTest * 0.5;
				zItGraphHalfEdge tempHe = bottomHE[0];

				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

			}
			else if (i == numTriangles)
			{
				float distance_increment = dTest;

				zItGraphHalfEdge tempHe = bottomHE[bottomHE.size() -  1].getSym();
				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);
			}
			else
			{
				// find red vertex
				bool redVertex = false;

				while (!redVertex)
				{
					if (bHe.getVertex().getColor() == red) redVertex = true;
					else
					{
						bHe = bottomHE[bottomID];
						bottomID++;
					}
				}

				// first point
				float distance_increment = bottomOverlapLength;
								
				zItGraphHalfEdge tempHe = bHe.getSym();
				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				// second Point 
				pOnCurve = bHe.getVertex().getPosition();

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				// Third Point

				distance_increment = (i == 3)? bottomOverlapLength * 4.0 :   bottomOverlapLength;
				tempHe = bHe.getNext();
				pOnCurve = tempHe.getStartVertex().getPosition();

				eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				//
				bHe = bottomHE[bottomID];
				bottomID++;
			}

			
		}

				
		/// POLYGON SCALARS

		/*for (int i = 0; i < numTriangles ; i++)
		{
			zPointArray gPositions;

			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 1]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 2]]);
			
			gPositions.push_back(bottomPts[bottom_indexInContainer[ ((i + 1) * 3) + 0]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);
			
			

			zIntArray gEdgeCOnnects;

			for (int i = 0; i < gPositions.size(); i++)
			{
				int next = (i + 1) % gPositions.size();

				gEdgeCOnnects.push_back(i);
				gEdgeCOnnects.push_back(next);
			}

			zObjGraph tempGraph;
			zFnGraph fnTempGraph(tempGraph);
			fnTempGraph.create(gPositions, gEdgeCOnnects);

			zScalarArray triField;
			getScalars_Polygon(triField, tempGraph, false);

			if (i == 0) scalars = triField;
			else
			{
				zScalarArray result;
				boolean_union(scalars, triField, result, false);

				scalars.clear();
				scalars = result;

			}
		}*/



		/// GRAPH SCALARS
		zPointArray gPositions;
		zIntArray gEdgeCOnnects;

		for (int i = 0; i < numTriangles ; i++)
		{
			if (i == 0)
			{
				//gEdgeCOnnects.push_back(gPositions.size());
				//gEdgeCOnnects.push_back(gPositions.size() + 1);

				//gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
				//gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);

				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);

				gPositions.push_back(topPts[top_indexInContainer[i * 3 + 2]]);
				gPositions.push_back(bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]]);

				
				
			}


			if (i == 1)
			{
				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);

				gPositions.push_back(topPts[top_indexInContainer[i * 3 + 1]]);
				gPositions.push_back(bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]]);

				
			}

			if (i == 2)
			{
				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);

				gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
				gPositions.push_back(bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]]);

				
			}

			if (i == 3)
			{
				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);

				gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
				gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);

				
			}

							


			/*gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);

			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);
			

			gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);

			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 2]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[ ((i + 1) * 3) + 0]]);*/
			
		}

		zObjGraph tempGraph;
		zFnGraph fnTempGraph(tempGraph);
		fnTempGraph.create(gPositions, gEdgeCOnnects);


		getScalarsAsEdgeDistance(scalars, tempGraph, printWidth, normalise);

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_topBottomTrim(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float offset, int type,  bool normalise)
	{

		scalars.clear();

		
		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zPointArray gPositions;
		zIntArray gEdgeCOnnects;

		if (type == 0 || type == 1)
		{
			gPositions.push_back(bottomHE[0].getStartVertex().getPosition());

			for (zItGraphHalfEdge& he : bottomHE)
			{
				gEdgeCOnnects.push_back(gPositions.size() - 1);
				gEdgeCOnnects.push_back(gPositions.size());

				gPositions.push_back(he.getVertex().getPosition());

			}
		}
		
		if (type == 0 || type == 2)
		{
			gPositions.push_back(topHE[0].getStartVertex().getPosition());

			for (zItGraphHalfEdge& he : topHE)
			{
				gEdgeCOnnects.push_back(gPositions.size() - 1);
				gEdgeCOnnects.push_back(gPositions.size());

				gPositions.push_back(he.getVertex().getPosition());

			}
		}
		

		zObjGraph tempGraph;
		zFnGraph fnTempGraph(tempGraph);
		fnTempGraph.create(gPositions, gEdgeCOnnects);

		
		getScalarsAsEdgeDistance(scalars, tempGraph,offset, false);

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_InfillTrim(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth,   bool normalise, zObjGraph& outGraph)
	{


		scalars.clear();

		zVector norm(0, 0, 1);
		zVector worldX(1, 0, 0);


		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		float topIncrement = topLength / (numTriangles);
		float topOverlapLength = printWidth * 1.0;


		int topNumPts = numTriangles * 3 + numTriangles * 2;
		zIntArray top_indexInContainer;
		top_indexInContainer.assign(topNumPts, -1);

		zPointArray topPts;

		int topID = 0;
		zItGraphHalfEdge tHe = topHE[topID];
		topPts.push_back(tHe.getStartVertex().getPosition());


		zPoint pOnCurve = topPts[topID];
		topID++;

		int counter = 0;
		for (int i = 0; i < numTriangles * 3; i++)
		{
			float distance_increment = topIncrement;
			if (i == 0) distance_increment = topIncrement / 2;

			if (i % 3 == 0) distance_increment -= (i == 0) ? topOverlapLength * 1.0 : topOverlapLength * 2.0;
			else distance_increment = topOverlapLength;

			

			zPoint eEndPoint = tHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tHe = topHE[topID];
				eEndPoint = tHe.getVertex().getPosition();
				topID++;
			}

			zVector he_vec = tHe.getVector();
			he_vec.normalize();

			//O
			pOnCurve = pOnCurve + he_vec * distance_increment;
					


			top_indexInContainer[counter] = topPts.size();
			topPts.push_back(pOnCurve);
			counter++;
		}


		/// BOTTOM B WITH GUIDE

		float bottomIncrement = bottomLength / (numTriangles);
		float d = (bottomIncrement - maxTriangleLength) * 0.5;
		float dTest = printWidth * 1.5;
		float bottomOverlapLength = (d < dTest) ? dTest : d;

		//printf("\n incr %1.4f %1.4f  %1.4f %1.4f", bottomLength, bottomIncrement, bottomOverlapLength, maxTriangleLength);

		int bottomNumPts = numTriangles * 3;
		zIntArray bottom_indexInContainer;
		bottom_indexInContainer.assign(bottomNumPts, -1);


		zPointArray bottomPts;

		int bottomID = 0;
		zItGraphHalfEdge bHe = bottomHE[bottomID];
		bottomPts.push_back(bHe.getStartVertex().getPosition());

		pOnCurve = bottomPts[bottomID];
		bottomID++;
		zColor red(1, 0, 0, 1);

		for (int i = 0; i < numTriangles; i++)
		{
			// find red vertex
			bool redVertex = false;

			while (!redVertex)
			{
				if (bHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					bHe = bottomHE[bottomID];
					bottomID++;
				}
			}

			// first point
			float distance_increment = bottomOverlapLength;

			zItGraphHalfEdge tempHe = bHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = bHe.getVertex().getPosition();

			bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// Third Point

			distance_increment = bottomOverlapLength;
			tempHe = bHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			//
			bHe = bottomHE[bottomID];
			bottomID++;
		}


		/// BOTTOM A NO GUID

		//float bottomIncrement = bottomLength / (numTriangles);
		//float bottomOverlapLength = printWidth * 1.0;

		//int bottomNumPts = numTriangles * 3 ;
		//zIntArray bottom_indexInContainer;
		//bottom_indexInContainer.assign(bottomNumPts, -1);


		//zPointArray bottomPts;

		//int bottomID = 0;
		//zItGraphHalfEdge bHe = bottomHE[bottomID];
		//bottomPts.push_back(bHe.getStartVertex().getPosition());

		//pOnCurve = bottomPts[bottomID];
		//bottomID++;

		//counter = 0;
		//for (int i = 0; i < numTriangles * 3; i++)
		//{
		//	float distance_increment = bottomIncrement;
		//	if (i == 0) distance_increment = bottomIncrement / 2;

		//	if (i % 3 == 0) distance_increment -= (i == 0) ? bottomOverlapLength * 1.0 : bottomOverlapLength * 2.0;
		//	else distance_increment = bottomOverlapLength;


		//	zPoint eEndPoint = bHe.getVertex().getPosition();

		//	while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
		//	{
		//		distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
		//		pOnCurve = eEndPoint;

		//		bottomPts.push_back(pOnCurve);

		//		bHe = bottomHE[bottomID];
		//		eEndPoint = bHe.getVertex().getPosition();
		//		bottomID++;
		//	}

		//	zVector he_vec = bHe.getVector();
		//	he_vec.normalize();

		//	//O
		//	pOnCurve = pOnCurve + he_vec * distance_increment;

		//	bottom_indexInContainer[counter] = bottomPts.size();
		//	bottomPts.push_back(pOnCurve);
		//	counter++;
		//	
		//	
		//}

		zPointArray gPositions;
		zIntArray gEdgeCOnnects;

		for (int i = 0; i < numTriangles; i++)
		{
			gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);
						
			zPoint p1 = (bottomPts[bottom_indexInContainer[i * 3 + 1]]);
			zPoint p0 = (topPts[top_indexInContainer[i * 3 + 1]]);
				
			
			zVector e = p0 - p1;
			float eLen = e.length();
			e.normalize();

			gPositions.push_back(p1 + e * 4 * printWidth);
			gPositions.push_back(p1 - e * 2 * printWidth);

		}

		zObjGraph tempGraph;
		zFnGraph fnTempGraph(tempGraph);
		fnTempGraph.create(gPositions, gEdgeCOnnects);


		getScalarsAsEdgeDistance(scalars, tempGraph, printWidth, normalise);



		/// ADD top  trim Lines
		topID = 0;
		tHe = topHE[topID];
		

		pOnCurve = topPts[topID];
		for (int i = 0; i < numTriangles -1 ; i++)
		{
			if (i == 0)
			{
				zVector he_vec = tHe.getVector();
				he_vec.normalize();

				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);


				zVector X = he_vec;

				zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
				Z.normalize();

				zVector Y = X ^ Z;
				Y.normalize();

				gPositions.push_back(pOnCurve + Y * 4.5 * printWidth);
				gPositions.push_back(pOnCurve - Y * 2.5 * printWidth);

			}
			
			
			float distance_increment = topIncrement;


			//length += distance_increment;

			zPoint eEndPoint = tHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				tHe = topHE[topID];
				eEndPoint = tHe.getVertex().getPosition();
				topID++;
			}

			zVector he_vec = tHe.getVector();
			he_vec.normalize();

			//O
			pOnCurve = pOnCurve + he_vec * distance_increment;



			gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);


			zVector X = he_vec;

			zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
			Z.normalize();

			zVector Y = X ^ Z;
			Y.normalize();

			gPositions.push_back(pOnCurve + Y * 4.5 * printWidth);
			gPositions.push_back(pOnCurve - Y * 2.5 * printWidth);
			
			
			if (i == numTriangles - 2)
			{
				tHe = topHE[topHE.size() - 1];

				pOnCurve = tHe.getVertex().getPosition();

				zVector he_vec = tHe.getVector();
				he_vec.normalize();

				gEdgeCOnnects.push_back(gPositions.size());
				gEdgeCOnnects.push_back(gPositions.size() + 1);


				zVector X = he_vec;

				zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
				Z.normalize();

				zVector Y = X ^ Z;
				Y.normalize();

				gPositions.push_back(pOnCurve + Y * 4.5 * printWidth);
				gPositions.push_back(pOnCurve - Y * 2.5 * printWidth);
			}
			

				
			
			
		}

		
		zFnGraph fnOutGraph(outGraph);
		fnOutGraph.create(gPositions, gEdgeCOnnects);

		///////////////////////////////////////
		//scalars.clear();


		//zFnGraph inFnGraph(inPolyObj);

		//zVector* meshPositions = fnMesh.getRawVertexPositions();
		//zVector* inPositions = inFnGraph.getRawVertexPositions();

		//zIntArray startVerts;

		//for (int i = 0; i < inFnGraph.numVertices(); i++)
		//{
		//	float d = coreUtils.minDist_Point_Plane(inPositions[i], startPlaneOrigin, startPlaneNorm);

		//	if (abs(d) < distanceTolerance)startVerts.push_back(i);
		//}
		//

		//// Flip if needed, to have lower point in 0 index
		//if (inPositions[startVerts[1]].y > inPositions[startVerts[0]].y)
		//{
		//	int temp = startVerts[0];

		//	startVerts[0] = startVerts[1];
		//	startVerts[1] = temp;
		//}


		//zItGraphHalfEdge topHe;
		//zItGraphHalfEdge bottomHe;

		//for (int i = 0; i < 2; i++)
		//{
		//	zItGraphVertex v(inPolyObj, startVerts[i]);

		//	zItGraphHalfEdgeArray cHEdges;
		//	v.getConnectedHalfEdges(cHEdges);

		//	if (i == 0)
		//	{
		//		bottomHe = (cHEdges[0].getVertex().getId() == startVerts[1]) ? cHEdges[1] : cHEdges[0];
		//	}

		//	if (i == 1)
		//	{
		//		topHe = (cHEdges[0].getVertex().getId() == startVerts[0]) ? cHEdges[1] : cHEdges[0];
		//	}

		//}

		//zBoolArray vertVisited;
		//vertVisited.assign(inFnGraph.numVertices(), false);

		//vertVisited[startVerts[0]] = true;
		//vertVisited[startVerts[1]] = true;

		//bool exit = false;
		//int counter = 0;
		//zPointArray gPositions;
		//zIntArray gEdgeCOnnects;

		//while (!exit)
		//{
		//	if (bottomHe.getLength() < 0.05)
		//	{
		//		vertVisited[bottomHe.getVertex().getId()] = true;
		//		bottomHe = bottomHe.getNext();
		//	}
		//	if (topHe.getLength() < 0.05)
		//	{
		//		vertVisited[topHe.getVertex().getId()] = true;
		//		topHe = topHe.getNext();
		//	}

		//	bottomHe.getEdge().setColor(zColor(0, 0, 1, 1));
		//	topHe.getEdge().setColor(zColor(0, 0, 1, 1));


		//	zPoint bottom_start = bottomHe.getStartVertex().getPosition();
		//	zVector bottom_vec = bottomHe.getVector();
		//	bottom_vec.normalize();
		//	float bottom_len = bottomHe.getLength();

		//	zPoint top_start = topHe.getStartVertex().getPosition();
		//	zVector top_vec = topHe.getVector();
		//	top_vec.normalize();
		//	float top_len = topHe.getLength();

		//	int numTrianglesperEdge = ceil(top_len / maxTriangleLength);

		//	float increment = 1.0 / (2 * numTrianglesperEdge);

		//	for (int i = 0; i < numTrianglesperEdge; i++)
		//	{			
		//		float fac = ((i + 1) * increment) + (i * increment);
		//		zPoint p0 = bottom_start + (bottom_vec * (fac * bottom_len));
		//		zPoint p1 = top_start + (top_vec * (fac * top_len));
		//		

		//		zVector e = p0 - p1;
		//		float eLen = e.length();
		//		e.normalize();

		//		p0 = p1;
		//		p1 = p0 + e * eLen * 0.5;

		//		gEdgeCOnnects.push_back(gPositions.size());
		//		gEdgeCOnnects.push_back(gPositions.size() + 1);

		//		gPositions.push_back(p0);
		//		gPositions.push_back(p1);

		//	}
		//			

		//	vertVisited[bottomHe.getVertex().getId()] = true;
		//	vertVisited[topHe.getVertex().getId()] = true;

		//	bottomHe = bottomHe.getNext();
		//	topHe = topHe.getNext();

		//	if (vertVisited[bottomHe.getVertex().getId()]) exit = true;
		//	if (vertVisited[topHe.getVertex().getId()]) exit = true;

		//	zItGraphVertex v = bottomHe.getVertex();
		//	if (topHe.getStartVertex() == v) exit = true;
		//	if (topHe.getVertex() == v) exit = true;

		//	counter++;
		//}

		//zObjGraph tempGraph;
		//zFnGraph fnTempGraph(tempGraph);
		//fnTempGraph.create(gPositions, gEdgeCOnnects);


		//getScalarsAsEdgeDistance(scalars, tempGraph, offset, normalise);

	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_InfillTrimBoundary(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth, bool stepTrim, bool normalise)
	{


		scalars.clear();

		zVector norm(0, 0, 1);
		zVector worldX(1, 0, 0);


		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		/// TOP B WITH GUIDE
		zColor red(1, 0, 0, 1);
		float dTest = printWidth * 1.0;
		float topOverlapLength = dTest;

		int topNumPts = numTriangles * 4;
		zIntArray top_indexInContainer;
		top_indexInContainer.assign(topNumPts, -1);

		zPointArray topPts;

		int topID = 0;
		zItGraphHalfEdge tHe = topHE[topID];
		topPts.push_back(tHe.getStartVertex().getPosition());


		zPoint pOnCurve = topPts[topID];
		topID++;

		int counter = 0;
		for (int i = 0; i < numTriangles; i++)
		{
			// find red vertex
			bool redVertex = false;

			while (!redVertex)
			{
				if (tHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					tHe = topHE[topID];
					topID++;
				}
			}

			// first point
			float distance_increment = (i == 2 || i == 3) ? topOverlapLength * 5.5 : topOverlapLength;
			//if (i == 3) distance_increment = topOverlapLength * 5.5;

			zItGraphHalfEdge tempHe = tHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			zVector X = he_vec;

			zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
			Z.normalize();

			zVector Y = X ^ Z; /*(1, 0, 0);*/
			Y.normalize();

			top_indexInContainer[(i * 4) + 0] = topPts.size();
			topPts.push_back(pOnCurve + Y * 2.5 * printWidth);

			top_indexInContainer[(i * 4) + 1] = topPts.size();
			topPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = tHe.getVertex().getPosition();						
			

			top_indexInContainer[(i * 4) + 2] = topPts.size();
			topPts.push_back(pOnCurve);

			// Third Point

			distance_increment = (i == 0) ? topOverlapLength * 4.5 : topOverlapLength * 2.5;

			tempHe = tHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			top_indexInContainer[(i * 4) + 3] = topPts.size();
			topPts.push_back(pOnCurve);

			//
			tHe = topHE[topID];
			topID++;
		}




		zPointArray gPositions;
		zIntArray gEdgeCOnnects;

		for (int i = 0; i < numTriangles; i++)
		{
			if (i != 0) continue;

			gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);

			zPoint p1 = (topPts[top_indexInContainer[i * 3 + 0]]);
			zPoint p0 = (topPts[top_indexInContainer[i * 3 + 1]]);


			zVector e = p0 - p1;
			float eLen = e.length();
			e.normalize();

			gPositions.push_back(p1 + e * 4 * printWidth);
			gPositions.push_back(p1 - e * 2 * printWidth);

		}

		zObjGraph tempGraph;
		zFnGraph fnTempGraph(tempGraph);
		fnTempGraph.create(gPositions, gEdgeCOnnects);


		getScalarsAsEdgeDistance(scalars, tempGraph, printWidth, normalise);

		

	}

	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_InfillInteriorTrimBoundary(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth, bool stepTrim, bool normalise, zObjGraph& outGraph)
	{

		scalars.clear();

		zVector norm(0, 0, 1);
		zVector worldX(1, 0, 0);

		zPointArray gOutPositions;
		zIntArray gOutEdgeConnects;


		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();


		/// TOP B WITH GUIDE
		zColor red(1, 0, 0, 1);
		float dTest = printWidth * 1.0;
		float topOverlapLength = dTest;

		int topNumPts = numTriangles * 3;
		zIntArray top_indexInContainer;
		top_indexInContainer.assign(topNumPts, -1);

		zPointArray topPts;

		int topID = 0;
		zItGraphHalfEdge tHe = topHE[topID];
		topPts.push_back(tHe.getStartVertex().getPosition());

		
		zPoint pOnCurve = topPts[topID];
		topID++;

		int counter = 0;
		
		for (int i = 0; i < numTriangles; i++)
		{
			// find red vertex
			bool redVertex = false;

			while (!redVertex)
			{
				if (tHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					tHe = topHE[topID];
					topID++;
				}
			}

			// first point
			float distance_increment = (i == 2 || i == 3) ? topOverlapLength * 5.5 : topOverlapLength;
			//if (i == 3) distance_increment = topOverlapLength * 5.5;

			zItGraphHalfEdge tempHe = tHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			// add pointsfor out graph
			if (i == 0)
			{
				zVector X = he_vec;

				zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
				Z.normalize();

				zVector Y = X ^ Z; /*(1, 0, 0);*/
				Y.normalize();

				gOutEdgeConnects.push_back(gOutPositions.size());
				gOutEdgeConnects.push_back(gOutPositions.size() + 1);

								
				gOutPositions.push_back(pOnCurve + Y * 5.5 * printWidth);
				gOutPositions.push_back(pOnCurve );


			}

			top_indexInContainer[(i * 3) + 0] = topPts.size();
			topPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = tHe.getVertex().getPosition();

			top_indexInContainer[(i * 3) + 1] = topPts.size();
			topPts.push_back(pOnCurve);

			// Third Point

			distance_increment = (i == 0) ? topOverlapLength * 4.5 : topOverlapLength * 2.5;
			

			tempHe = tHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				topPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			top_indexInContainer[(i * 3) + 2] = topPts.size();
			topPts.push_back(pOnCurve);

			//
			tHe = topHE[topID];
			topID++;
		}

		/// BOTTOM B WITH GUIDE

		dTest = printWidth * 2.5;
		float bottomOverlapLength = dTest;

		//printf("\n incr %1.4f %1.4f  %1.4f %1.4f", bottomLength, bottomIncrement, bottomOverlapLength, maxTriangleLength);

		int bottomNumPts = (numTriangles + 1) * 3;
		zIntArray bottom_indexInContainer;
		bottom_indexInContainer.assign(bottomNumPts, -1);


		zPointArray bottomPts;

		int bottomID = 0;
		zItGraphHalfEdge bHe = bottomHE[bottomID];
		bottomPts.push_back(bHe.getStartVertex().getPosition());

		pOnCurve = bottomPts[bottomID];
		bottomID++;


		for (int i = 0; i < numTriangles + 1; i++)
		{
			if (i == 0)
			{

				float distance_increment = dTest * 0.5;
				zItGraphHalfEdge tempHe = bottomHE[0];

				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

			}
			else if (i == numTriangles)
			{
				float distance_increment = dTest;

				zItGraphHalfEdge tempHe = bottomHE[bottomHE.size() - 1].getSym();
				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);
			}
			else
			{
				// find red vertex
				bool redVertex = false;

				while (!redVertex)
				{
					if (bHe.getVertex().getColor() == red) redVertex = true;
					else
					{
						bHe = bottomHE[bottomID];
						bottomID++;
					}
				}

				// first point
				float distance_increment = bottomOverlapLength;

				zItGraphHalfEdge tempHe = bHe.getSym();
				pOnCurve = tempHe.getStartVertex().getPosition();

				zPoint eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				zVector he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 0] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				// second Point 
				pOnCurve = bHe.getVertex().getPosition();

				bottom_indexInContainer[(i * 3) + 1] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				// Third Point

				distance_increment = (i == 3) ? bottomOverlapLength * 4.0 : bottomOverlapLength;
				tempHe = bHe.getNext();
				pOnCurve = tempHe.getStartVertex().getPosition();

				eEndPoint = tempHe.getVertex().getPosition();

				while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				{
					distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					bottomPts.push_back(pOnCurve);

					tempHe = tempHe.getNext();
					eEndPoint = tempHe.getVertex().getPosition();

				}

				he_vec = tempHe.getVector();
				he_vec.normalize();

				pOnCurve = pOnCurve + he_vec * distance_increment;

				bottom_indexInContainer[(i * 3) + 2] = bottomPts.size();
				bottomPts.push_back(pOnCurve);

				//
				bHe = bottomHE[bottomID];
				bottomID++;
			}


		}


		/// POLYGON SCALARS

		/*for (int i = 0; i < numTriangles ; i++)
		{
			zPointArray gPositions;

			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 0]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 1]]);
			gPositions.push_back(topPts[top_indexInContainer[i * 3 + 2]]);

			gPositions.push_back(bottomPts[bottom_indexInContainer[ ((i + 1) * 3) + 0]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 3 + 2]]);



			zIntArray gEdgeCOnnects;

			for (int i = 0; i < gPositions.size(); i++)
			{
				int next = (i + 1) % gPositions.size();

				gEdgeCOnnects.push_back(i);
				gEdgeCOnnects.push_back(next);
			}

			zObjGraph tempGraph;
			zFnGraph fnTempGraph(tempGraph);
			fnTempGraph.create(gPositions, gEdgeCOnnects);

			zScalarArray triField;
			getScalars_Polygon(triField, tempGraph, false);

			if (i == 0) scalars = triField;
			else
			{
				zScalarArray result;
				boolean_union(scalars, triField, result, false);

				scalars.clear();
				scalars = result;

			}
		}*/



		/// GRAPH SCALARS
		zPointArray gPositions;
		zIntArray gEdgeCOnnects;

	

		for (int i = 0; i < numTriangles; i++)
		{
			zPoint p0, p1;

			if (i == 0)
			{
				p0 = topPts[top_indexInContainer[i * 3 + 2]];
				p1 = bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]];	
				
				
			}


			if (i == 1)
			{
				p0 = (topPts[top_indexInContainer[i * 3 + 1]]);
				p1 = (bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]]);
				
				
			}

			if (i == 2)
			{
				p0 = (topPts[top_indexInContainer[i * 3 + 0]]);
				p1 = (bottomPts[bottom_indexInContainer[((i + 1) * 3) + 1]]);

				
			}

			if (i == 3)
			{
				p0 = (topPts[top_indexInContainer[i * 3 + 0]]);
				p1 = (bottomPts[bottom_indexInContainer[i * 3 + 2]]);

				
			}
			
		

			

			gOutEdgeConnects.push_back(gOutPositions.size());
			gOutEdgeConnects.push_back(gOutPositions.size() + 1);

			gEdgeCOnnects.push_back(gPositions.size());
			gEdgeCOnnects.push_back(gPositions.size() + 1);

			
			zVector e = p1 - p0;
			float eLen = e.length();
			e.normalize();

			zVector perp = norm ^ e;
			perp.normalize();

			float val = (stepTrim) ? 0.4 : 0.6;
			zPoint p = p0 + (e * eLen * val);


			gPositions.push_back(p + perp * printWidth);
			gPositions.push_back(p - perp * printWidth);

			gOutPositions.push_back(p0);
			gOutPositions.push_back(p1);

		}

		zObjGraph tempGraph;
		zFnGraph fnTempGraph(tempGraph);
		fnTempGraph.create(gPositions, gEdgeCOnnects);


		getScalarsAsEdgeDistance(scalars, tempGraph, printWidth * 0.75, normalise);

		/// add points at top of handrail
		
		zPoint p0 = topHE[topHE.size() - 1].getVertex().getPosition();
		zPoint p1 = bottomHE[bottomHE.size() - 1].getVertex().getPosition();

		pOnCurve = (p0 + p1) * 0.5;

		zVector X = p1 - p0;

		zVector Z = (X * worldX > 0) ? norm * 1 : norm * -1;
		Z.normalize();

		zVector Y = X ^ Z; /*(1, 0, 0);*/
		Y.normalize();

		gOutEdgeConnects.push_back(gOutPositions.size());
		gOutEdgeConnects.push_back(gOutPositions.size() + 1);


		gOutPositions.push_back(pOnCurve + Y * 3.5 * printWidth);
		gOutPositions.push_back(pOnCurve);	


		zFnGraph fnOutGraph(outGraph);
		fnOutGraph.create(gOutPositions, gOutEdgeConnects);

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_Pattern(zScalarArray& scalars, zObjGraph& inPolyObj, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, int numTriangles, float maxTriangleLength, float printWidth, bool normalise)
	{
		scalars.clear();

		zVector norm(0, 0, 1);
		zVector worldX(1, 0, 0);


		zFnGraph inFnGraph(inPolyObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();


		/// BOTTOM B WITH GUIDE

		float bottomIncrement = bottomLength / (numTriangles);
		float bottomOverlapLength = printWidth * 1.5;

		int bottomNumPts = numTriangles * 3 + numTriangles * 2;
		zIntArray bottom_indexInContainer;
		bottom_indexInContainer.assign(bottomNumPts, -1);

		float pokeMult = 3.5;

		zPointArray bottomPts;

		int bottomID = 0;
		zItGraphHalfEdge bHe = bottomHE[bottomID];
		bottomPts.push_back(bHe.getStartVertex().getPosition());

		zPoint pOnCurve = bottomPts[bottomID];
		bottomID++;

		zColor red(1, 0, 0, 1);

		for (int i = 0; i < numTriangles; i++)
		{
			// find red vertex
			bool redVertex = false;

			while (!redVertex)
			{
				if (bHe.getVertex().getColor() == red) redVertex = true;
				else
				{
					bHe = bottomHE[bottomID];
					bottomID++;
				}
			}

			// first point
			float distance_increment = bottomOverlapLength;

			zItGraphHalfEdge tempHe = bHe.getSym();
			pOnCurve = tempHe.getStartVertex().getPosition();

			zPoint eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			zVector he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			zVector X = he_vec;

			zVector Z = (X * worldX > 0) ? norm * -1 : norm * 1;
			Z.normalize();

			zVector Y = X ^ Z;
			Y.normalize();

			bottom_indexInContainer[(i * 5) + 0] = bottomPts.size();
			bottomPts.push_back(pOnCurve + Y * pokeMult * printWidth);

			bottom_indexInContainer[(i * 5) + 1] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// second Point 
			pOnCurve = bHe.getVertex().getPosition();

			bottom_indexInContainer[(i * 5) + 2] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			// Third Point

			distance_increment = bottomOverlapLength;
			tempHe = bHe.getNext();
			pOnCurve = tempHe.getStartVertex().getPosition();

			eEndPoint = tempHe.getVertex().getPosition();

			while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			{
				distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				pOnCurve = eEndPoint;

				bottomPts.push_back(pOnCurve);

				tempHe = tempHe.getNext();
				eEndPoint = tempHe.getVertex().getPosition();

			}

			he_vec = tempHe.getVector();
			he_vec.normalize();

			pOnCurve = pOnCurve + he_vec * distance_increment;

			X = he_vec;

			Z = (X * worldX > 0) ? norm * -1 : norm * 1;
			Z.normalize();

			Y = X ^ Z;
			Y.normalize();

			bottom_indexInContainer[(i * 5) + 3] = bottomPts.size();
			bottomPts.push_back(pOnCurve + Y * pokeMult * printWidth);

			bottom_indexInContainer[(i * 5) + 4] = bottomPts.size();
			bottomPts.push_back(pOnCurve);

			//
			bHe = bottomHE[bottomID];
			bottomID++;
		}

		
		/// BOTTOM A No GUIDE

		//float bottomIncrement = bottomLength / (numTriangles);
		//float bottomOverlapLength =  printWidth * 2.0 ;

		//int bottomNumPts = numTriangles * 3  + numTriangles * 2;
		//zIntArray bottom_indexInContainer;
		//bottom_indexInContainer.assign(bottomNumPts, -1);


		//zPointArray bottomPts;

		//int bottomID = 0;
		//zItGraphHalfEdge bHe = bottomHE[bottomID];
		//bottomPts.push_back(bHe.getStartVertex().getPosition());

		//zPoint pOnCurve = bottomPts[bottomID];
		//bottomID++;

		//int counter = 0;
		//for (int i = 0; i < numTriangles * 3; i++)
		//{
		//	float distance_increment = bottomIncrement;
		//	if (i == 0) distance_increment = bottomIncrement / 2;

		//	if (i % 3 == 0) distance_increment -= (i == 0) ? bottomOverlapLength * 1.0: bottomOverlapLength * 2.0;
		//	else distance_increment = bottomOverlapLength;


		//	zPoint eEndPoint = bHe.getVertex().getPosition();

		//	while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
		//	{
		//		distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
		//		pOnCurve = eEndPoint;

		//		bottomPts.push_back(pOnCurve);

		//		bHe = bottomHE[bottomID];
		//		eEndPoint = bHe.getVertex().getPosition();
		//		bottomID++;
		//	}

		//	zVector he_vec = bHe.getVector();
		//	he_vec.normalize();

		//	//O
		//	pOnCurve = pOnCurve + he_vec * distance_increment;

		//	if (i % 3 == 0 || i % 3 == 2)
		//	{
		//		zVector X = he_vec;

		//		zVector Z = (X * worldX > 0) ? norm * -1 : norm * 1;
		//		Z.normalize();

		//		zVector Y = X ^ Z;
		//		Y.normalize();

		//		bottom_indexInContainer[counter] = bottomPts.size();
		//		bottomPts.push_back(pOnCurve + Y * 2.5 * printWidth);

		//		counter++;
		//	}

		//	bottom_indexInContainer[counter] = bottomPts.size();
		//	bottomPts.push_back(pOnCurve);
		//	counter++;
		//}


		for (int i = 0; i < numTriangles; i++)
		{
			zPointArray gPositions;

			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 5 + 0]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 5 + 1]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 5 + 2]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 5 + 4]]);
			gPositions.push_back(bottomPts[bottom_indexInContainer[i * 5 + 3]]);			

			zIntArray gEdgeCOnnects;

			for (int i = 0; i < gPositions.size(); i++)
			{
				int next = (i + 1) % gPositions.size();

				gEdgeCOnnects.push_back(i);
				gEdgeCOnnects.push_back(next);
			}

			zObjGraph tempGraph;
			zFnGraph fnTempGraph(tempGraph);
			fnTempGraph.create(gPositions, gEdgeCOnnects);

			zScalarArray triField;
			getScalars_Polygon(triField, tempGraph, false);

			if (i == 0) scalars = triField;
			else
			{
				zScalarArray result;
				boolean_union(scalars, triField, result, false);

				scalars.clear();
				scalars = result;

			}
		}




		///////////////////////////////////
		//scalars.clear();


		//zFnGraph inFnGraph(inPolyObj);

		//zVector* meshPositions = fnMesh.getRawVertexPositions();
		//zVector* inPositions = inFnGraph.getRawVertexPositions();

		//zIntArray startVerts;

		//for (int i = 0; i < inFnGraph.numVertices(); i++)
		//{
		//	float d = coreUtils.minDist_Point_Plane(inPositions[i], startPlaneOrigin, startPlaneNorm);

		//	if (abs(d) < distanceTolerance)startVerts.push_back(i);
		//}


		//// Flip if needed, to have lower point in 0 index
		//if (inPositions[startVerts[1]].y > inPositions[startVerts[0]].y)
		//{
		//	int temp = startVerts[0];

		//	startVerts[0] = startVerts[1];
		//	startVerts[1] = temp;
		//}


		//zItGraphHalfEdge topHe;
		//zItGraphHalfEdge bottomHe;

		//for (int i = 0; i < 2; i++)
		//{
		//	zItGraphVertex v(inPolyObj, startVerts[i]);

		//	zItGraphHalfEdgeArray cHEdges;
		//	v.getConnectedHalfEdges(cHEdges);

		//	if (i == 0)
		//	{
		//		bottomHe = (cHEdges[0].getVertex().getId() == startVerts[1]) ? cHEdges[1] : cHEdges[0];
		//	}

		//	if (i == 1)
		//	{
		//		topHe = (cHEdges[0].getVertex().getId() == startVerts[0]) ? cHEdges[1] : cHEdges[0];
		//	}

		//}

		//zBoolArray vertVisited;
		//vertVisited.assign(inFnGraph.numVertices(), false);

		//vertVisited[startVerts[0]] = true;
		//vertVisited[startVerts[1]] = true;

		//bool exit = false;
		//int counter = 0;
		////zPointArray gPositions;
		////zIntArray gEdgeCOnnects;

		//while (!exit)
		//{
		//	if (bottomHe.getLength() < 0.05)
		//	{
		//		vertVisited[bottomHe.getVertex().getId()] = true;
		//		bottomHe = bottomHe.getNext();
		//	}
		//	if (topHe.getLength() < 0.05)
		//	{
		//		vertVisited[topHe.getVertex().getId()] = true;
		//		topHe = topHe.getNext();
		//	}

		//	bottomHe.getEdge().setColor(zColor(0, 0, 1, 1));
		//	topHe.getEdge().setColor(zColor(0, 0, 1, 1));


		//	zPoint bottom_start = bottomHe.getStartVertex().getPosition();
		//	zVector bottom_vec = bottomHe.getVector();
		//	bottom_vec.normalize();
		//	float bottom_len = bottomHe.getLength();

		//	zPoint top_start = topHe.getStartVertex().getPosition();
		//	zVector top_vec = topHe.getVector();
		//	top_vec.normalize();
		//	float top_len = topHe.getLength();

		//	int numTrianglesperEdge = ceil(top_len / maxTriangleLength);

		//	float increment = 1.0 / (2 * numTrianglesperEdge);
		//	float step = 0.25 * increment;

		//	for (int i = 0; i < numTrianglesperEdge; i++)
		//	{
		//		float fac = ((i + 1) * increment) + (i * increment);
		//		zPoint p0 = bottom_start + (bottom_vec * ((fac - (1.5 * step)) * bottom_len));

		//		zPoint p1 =  (flipTriangle) ? bottom_start + (bottom_vec * ((fac - step) * bottom_len)) : bottom_start + (bottom_vec * ((fac + step) * bottom_len));
		//		zPoint p2 = top_start + (top_vec * (fac * top_len));

		//		zPoint p3 = bottom_start + (bottom_vec * ((fac + (1.5 *step)) * bottom_len));

		//		/*zScalarArray triField;
		//		getScalars_Circle(triField, p0, offset, 0.0,false);

		//		if (counter == 0) scalars = triField;
		//		else
		//		{
		//			zScalarArray result;
		//			boolean_union(scalars, triField, result, false);

		//			scalars.clear();
		//			scalars = result;

		//		}

		//		counter++;*/


		//		zVector e = p1 - p2;
		//		float eLen = e.length();
		//		e.normalize();

		//		p0 = p0 - e * offset;

		//		p1 = p1 + e * offset;	

		//		p3 = p3 - e * offset;

		//		zPointArray gPositions;
		//		gPositions.push_back(p0);
		//		gPositions.push_back(p1);
		//		gPositions.push_back(p3);

		//		zIntArray gEdgeCOnnects = { 0,1,1,2,2,0 };

		//		zObjGraph tempGraph;
		//		zFnGraph fnTempGraph(tempGraph);
		//		fnTempGraph.create(gPositions, gEdgeCOnnects);

		//		zScalarArray triField;
		//		getScalars_Polygon(triField, tempGraph, false);				

		//		if (counter == 0) scalars = triField;
		//		else
		//		{
		//			zScalarArray result;
		//			boolean_union(scalars, triField, result, false);

		//			scalars.clear();
		//			scalars = result;

		//		}

		//		counter++;

		//	}


		//	vertVisited[bottomHe.getVertex().getId()] = true;
		//	vertVisited[topHe.getVertex().getId()] = true;

		//	bottomHe = bottomHe.getNext();
		//	topHe = topHe.getNext();

		//	if (vertVisited[bottomHe.getVertex().getId()]) exit = true;
		//	if (vertVisited[topHe.getVertex().getId()]) exit = true;

		//	zItGraphVertex v = bottomHe.getVertex();
		//	if (topHe.getStartVertex() == v) exit = true;
		//	if (topHe.getVertex() == v) exit = true;
		//				
		//}
		//
		///*zObjGraph tempGraph;
		//zFnGraph fnTempGraph(tempGraph);
		//fnTempGraph.create(gPositions, gEdgeCOnnects);


		//getScalarsAsEdgeDistance(scalars, tempGraph, offset, normalise);*/

	}


	template<>
	ZSPACE_INLINE 	void zFnMeshField<zScalar>::getScalars_3dp_Triangle(zScalarArray& scalars, zObjGraph& inGraphObj, int startVertexId, zFloatArray& offsets, zFloatArray& intervals, zVector planeNorm, bool normalise)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		zItGraphHalfEdgeArray orientedHalfEdges;

		zFloatArray vertex_distanceFromStart;
		vertex_distanceFromStart.assign(inFnGraph.numVertices(), 0);

		zItGraphVertex startV(inGraphObj, startVertexId);
		zItGraphHalfEdge walkHe = startV.getHalfEdge();

		float graphLen = 0;
		vertex_distanceFromStart[startV.getId()] = graphLen;

		planeNorm.normalize();
		zVector worldX(1, 0, 0);

		bool exit = false;
		do
		{
			graphLen += walkHe.getLength();
			vertex_distanceFromStart[walkHe.getVertex().getId()] = graphLen;

			orientedHalfEdges.push_back(walkHe);

			if (walkHe.getVertex().checkValency(1))
			{
				exit = true;
			}
			else walkHe = walkHe.getNext();

		} while (!exit);


		zPointArray positions;

		for (int i =0; i< intervals.size(); i++)
		{
			float dist = graphLen * intervals[i];

			walkHe = startV.getHalfEdge();

			bool exit = false;

			do
			{

				if (vertex_distanceFromStart[walkHe.getVertex().getId()] >= dist) exit = true;
				if (!exit) walkHe = walkHe.getNext();

			} while (!exit);

			float d = dist - vertex_distanceFromStart[walkHe.getStartVertex().getId()];

			zVector he_vec = walkHe.getVector();
			he_vec.normalize();			

			zVector  Y = he_vec ^ planeNorm;;	
			
			Y.normalize();

			

			zPoint p = walkHe.getStartVertex().getPosition() + (he_vec * d);

			p += Y * offsets[i];
			positions.push_back(p);

		}
	

		getScalars_Triangle(scalars, positions[0], positions[1], positions[2], 0.0, normalise);

	}


	//----  2D SD SCALAR FIELD METHODS

	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Polygon(zScalarArray& scalars, zObjGraph& inGraphObj, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			scalars[i] = getScalar_Polygon(inGraphObj, meshPositions[i]) ;
		}

		if (normalise)
		{
			normliseValues(scalars);
		}

	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Circle(zScalarArray &scalars, zVector &cen, float r, double annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		cen.z = 0;

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (annularVal == 0) scalars[i] = getScalar_Circle(cen, meshPositions[i], r);
			else scalars[i] = abs(getScalar_Circle(cen, meshPositions[i], r) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Ellipse(zScalarArray& scalars, zVector& cen, float a, float b, double annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		cen.z = 0;

		zVector* meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (annularVal == 0) scalars[i] = getScalar_Ellipse(cen, meshPositions[i], a,b);
			else scalars[i] = abs(getScalar_Ellipse(cen, meshPositions[i], a, b) - annularVal);
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Triangle(zScalarArray& scalars, zPoint& p0, zPoint& p1, zPoint& p2, double annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);


		p0.z = 0; p1.z = 0; p2.z = 0;

		zVector* meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			if (annularVal == 0.0) scalars[i] = getScalar_Triangle(meshPositions[i], p0, p1, p2);
			else scalars[i] = abs(getScalar_Triangle(meshPositions[i], p0, p1, p2) - annularVal);
		}

		if (normalise) normliseValues(scalars);
	}


	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_Square(zScalarArray &scalars, zVector& cen, zVector &dimensions, float annularVal, bool normalise)
	{
		scalars.clear();
		scalars.assign(fnMesh.numVertices(), 0.0);

		cen.z = 0;

		zVector *meshPositions = fnMesh.getRawVertexPositions();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			zVector p = meshPositions[i];
			if (annularVal == 0) scalars[i] = getScalar_Square(p, cen, dimensions);
			else scalars[i] = abs(getScalar_Square(p, cen, dimensions) - annularVal);
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


	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getScalars_SinBands(zScalarArray& scalars, zObjGraph& inGraphObj, zVector& pNorm, zPoint& pCen, float scale)
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector* meshPositions = fnMesh.getRawVertexPositions();
		zVector* inPositions = inFnGraph.getRawVertexPositions();

		
		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			// get graph closest point
			float closestDist = 100000;
			int closestId = -1;

			zPoint closestPt;


			for (zItGraphEdge e(inGraphObj); !e.end(); e++)
			{

				int e0 = e.getHalfEdge(0).getVertex().getId();
				int e1 = e.getHalfEdge(0).getStartVertex().getId();
				
				if (e.getLength() < EPS) continue;

				zPoint tempPt;

				double d = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], tempPt);
				
				if (d < closestDist)
				{
					closestDist = d;
					closestPt = tempPt;
				}
			}
						

			//Calculate distance to the input plane
			//float distToPlane = ( coreUtils.minDist_Point_Plane(closestPt, pCen, pNorm));

			float distToPlane = (coreUtils.minDist_Point_Plane(meshPositions[i], pCen, pNorm));

			//float distToPlane = closestDist;

			//Filter distance value with Sin function

			// regular sin

			//float s  = sin(distToPlane * scale);

			// triangular wave https://mathworld.wolfram.com/TriangleWave.html
			//float s =  2 * asin( sin(distToPlane * scale * PI));
			//s /= (scale * PI);

			//float val = distToPlane * scale * 0.5 + 0.25;
			//float fracVal = val - floor(val);
			//float s = 1 - (4 * abs(0.5 - fracVal));

			//float s = 1 - 2 * abs(round(distToPlane * scale * 0.5) - (distToPlane * scale * 0.5));

			// trapezoidal wave https://stackoverflow.com/questions/11041498/equation-for-trapezoidal-wave-equation 
			//a/pi(arcsin(sin((pi/m)x+l))+arccos(cos((pi/m)x+l)))-a/2+c

			/*float a = 2;
			float m = 0.1;
			float l = 0.2;
			float c = 0.5;
			float x = distToPlane * scale;

			float s = a / PI * (asin(sin((PI / m) * x + l)) + acos(cos((PI / m) * x + l))) - a / 2 + c;*/
			//printf("\n %1.2f ", s);

			// https://www.shadertoy.com/view/Md3yRH

			zPoint temp = meshPositions[i];
			zVector trans(0, 0.1, 0);

			temp += trans;

			float offset = 0.1;
			float a = temp.x / offset;
			float val = 2 * asin(sin(a * PI));
			val /= (PI);

			//float val = sin(a);
			
			zVector v(0.0, temp.y - 0.075 * val, 0);

			float s = v.length() - 0.075;
				
			scalars.push_back(s);

		}
	}



	//--- COMPUTE METHODS 
	
	template<typename T>
	ZSPACE_INLINE bool zFnMeshField<T>::checkPositionBounds(zPoint &pos, int &index)
	{
		bool out = true;

		int index_X = floor((pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
		int index_Y = floor((pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

		if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0) out = false;

		int id = index_X * fieldObj->field.n_X + index_Y;

		index = id;

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

		/*zDomainFloat outNeg(-1.0, 0.0);
		zDomainFloat outPos(0.0, 1.0);

		zDomainFloat inNeg(d.min, 0.0);
		zDomainFloat inPos(0.0, d.max);

		for (int i = 0; i < fieldValues.size(); i++)
		{
			if(fieldValues[i] < 0) fieldValues[i] = coreUtils.ofMap(fieldValues[i], inNeg, outNeg);
			else fieldValues[i] = coreUtils.ofMap(fieldValues[i], inPos, outPos);
		}*/
	}

	template<>
	ZSPACE_INLINE void zFnMeshField<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}

	//---- zScalar specilization for smoothField

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::smoothField(zScalarArray& scalars, int numSmooth, double diffuseDamp, zDiffusionType type)
	{
		for (int k = 0; k < numSmooth; k++)
		{
			zScalarArray tempValues;

			for( zItMeshScalarField s(*fieldObj); !s.end(); s++)			
			{
				float lapA = 0;

				vector<zItMeshScalarField> ringNeigbours;
				s.getNeighbour_Ring( 1, ringNeigbours);

				for (int j = 0; j < ringNeigbours.size(); j++)
				{
					int id = ringNeigbours[j].getId();
					zScalar val = scalars[id]/*ringNeigbours[j].getValue()*/;
					
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
					float val1 = scalars[s.getId()]/*s.getValue()*/;
					
					float newA = val1 + (lapA * diffuseDamp);
					tempValues.push_back(newA);
				}
				else if (type == zAverage)
				{
					if (lapA != 0) lapA /= (ringNeigbours.size());

					tempValues.push_back(lapA);
				}

			}

			//setFieldValues(tempValues);

			scalars = tempValues;

		}

		//updateColors();
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
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_clipwithPlane(zScalarArray& scalars, zPlane& clipPlane)
	{
		int i = 0;

		for (zItMeshVertex v(*fieldObj); !v.end(); v++, i++)
		{
			zVector O(clipPlane(3, 0), clipPlane(3, 1), clipPlane(3, 2));
			zVector Z(clipPlane(2, 0), clipPlane(2, 1), clipPlane(2, 2));

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

	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::boolean_clipwithPlane(zScalarArray& fieldValues_A, zScalarArray& fieldValues_Result, zPoint& O, zVector& Z)
	{
		int i = 0;
		
		zScalarArray temp;

		for (zItMeshVertex v(*fieldObj); !v.end(); v++, i++)
		{			
			zPoint p = v.getPosition();
			float minDist_Plane = coreUtils.minDist_Point_Plane(p, O, Z);
						
			temp.push_back(minDist_Plane);
			
		}
				
		boolean_subtract(fieldValues_A, temp, fieldValues_Result, false);				

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
				//cout << "\n update color : contourValueDomain " << contourValueDomain.min << " , " << contourValueDomain.max;
				//cout << "\n update color : fieldColorDomain min " << fieldColorDomain.min.r << " , " << fieldColorDomain.min.g << " , " << fieldColorDomain.min.b;
				//cout << "\n update color : fieldColorDomain max" << fieldColorDomain.max.r << " , " << fieldColorDomain.max.g << " , " << fieldColorDomain.max.b;


				fieldColorDomain.min.toHSV(); fieldColorDomain.max.toHSV();

				zColor* cols = fnMesh.getRawVertexColors();
				if (fnMesh.numPolygons() == scalars.size()) cols = fnMesh.getRawFaceColors();

				for (int i = 0; i < scalars.size(); i++)
				{
					// SLIME
					if (scalars[i] < contourValueDomain.min) cols[i] = fieldColorDomain.min;
					else if (scalars[i] > contourValueDomain.max) cols[i] = fieldColorDomain.max;
					else
					{
						cols[i] = coreUtils.blendColor(scalars[i], contourValueDomain, fieldColorDomain, zHSV);
					}					


					// SDFs
					//if (scalars[i] < -0.01)
					//{
					//	//temp = coreUtils.blendColor(scalars[i], dVal, dCol, zHSV);

					//	cols[i] = zColor(0, 0.550, 0.950, 1);
					//}
					//else if (scalars[i] > 0.01)
					//{

					//	cols[i] = zColor(0.25, 0.25, 0.25, 1) /*dCol.max*/;
					//}
					//else cols[i] = zColor(0.950, 0, 0.55, 1);;

					//OTHER
					//if (scalars[i] < -0.005)
					//{
					//	zDomainColor dCol(zColor(324, 0.0, 1), zColor(324, 0.0, 0.4));
					//	zDomainFloat dVal(contourValueDomain.min, -0.005);

					//	//cols[i] = zColor(1, 0, 0, 1);

					//	cols[i] = coreUtils.blendColor(scalars[i], dVal, dCol, zHSV);
					//}
					//else if (scalars[i] > 0.005)
					//{

					//	zDomainColor dCol(zColor(150, 0.0, 0.4), zColor(150, 0, 1));
					//	zDomainFloat dVal(0.005, contourValueDomain.max);

					//	//cols[i] = zColor(0, 1, 0, 1);

					//	cols[i] = coreUtils.blendColor(scalars[i], dVal, dCol, zHSV);
					//}
					//else cols[i] = zColor(324, 1, 1);
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
	
		//printf("\n working  colors  %i ", contourVertexValues.size());;

	}

	//---- CONTOUR METHODS
	
	template<>
	ZSPACE_INLINE void zFnMeshField<zScalar>::getIsocontour(zObjGraph &coutourGraphObj, float inThreshold)
	{
		//cout << "\n getIsocontour : contourValueDomain " << contourValueDomain.min << " , " << contourValueDomain.max;
		//cout << "\n getIsocontour : contourVertexValues " << coreUtils.zMin(contourVertexValues) << " , " << coreUtils.zMax(contourVertexValues);

		if (contourVertexValues.size() == 0)
		{		
			return;
		}
		if (contourVertexValues.size() != numFieldValues())
		{
			throw std::invalid_argument(" error: invalid contour condition. Call updateColors method. ");
			return;
		}

		float threshold = inThreshold /*coreUtils.ofMap(inThreshold, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max)*/;
		//printf("\n threshold %1.2f ", threshold);

		vector<zVector> pos;
		vector<int> edgeConnects;

		vector<int> edgetoIsoGraphVertexId;

		zVector *positions = fnMesh.getRawVertexPositions();

		zColorArray vColors;
		fnMesh.getIsoContour(contourVertexValues, inThreshold, pos, edgeConnects, vColors);

		//// compute positions
		//int i = 0;
		//for (zItMeshEdge e(*fieldObj); !e.end(); e++)
		//{
		//	edgetoIsoGraphVertexId.push_back(-1);
		//	edgetoIsoGraphVertexId.push_back(-1);


		//	int eV0 = e.getHalfEdge(0).getVertex().getId();
		//	int eV1 = e.getHalfEdge(0).getStartVertex().getId();



		//	float scalar_lower = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV0] : contourVertexValues[eV1];
		//	float scalar_higher = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV1] : contourVertexValues[eV0];;

		//	bool chkSplitEdge = (scalar_lower <= threshold && scalar_higher > threshold) ? true : false;

		//	if (chkSplitEdge)
		//	{
		//		// calculate split point

		//		int scalar_lower_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV0 : eV1;
		//		int scalar_higher_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV1 : eV0;

		//		zVector scalar_lower_vertPos = positions[scalar_lower_vertId];
		//		zVector scalar_higher_vertPos = positions[scalar_higher_vertId];

		//		float scaleVal = coreUtils.ofMap(threshold, scalar_lower, scalar_higher, 0.0f, 1.0f);

		//		zVector e = scalar_higher_vertPos - scalar_lower_vertPos;
		//		double eLen = e.length();

		//		e.normalize();

		//		zVector newPos = scalar_lower_vertPos + (e * eLen * scaleVal);
		//		int id;
		//		
		//		pos.push_back(newPos);

		//			// map edge to isographVertex
		//			edgetoIsoGraphVertexId[i] = pos.size() - 1;
		//			edgetoIsoGraphVertexId[i + 1] = pos.size() - 1;
		//		
		//		

		//	}

		//	i += 2;
		//}

		//// compute edgeConnects
		//for (zItMeshFace f(*fieldObj); !f.end(); f++)
		//{

		//	vector<int> fEdges;
		//	f.getHalfEdges(fEdges);
		//	vector<int> tempConnects;

		//	for (int j = 0; j < fEdges.size(); j++)
		//	{
		//		if (edgetoIsoGraphVertexId[fEdges[j]] != -1)
		//			tempConnects.push_back(edgetoIsoGraphVertexId[fEdges[j]]);
		//	}

		//	//printf("\n face %i | %i ", i, tempConnects.size());

		//	if (tempConnects.size() == 2)
		//	{
		//		edgeConnects.push_back(tempConnects[0]);
		//		edgeConnects.push_back(tempConnects[1]);
		//	}

		//}

		zFnGraph tempFn(coutourGraphObj);
		tempFn.clear(); // clear memory if the mobject exists.
				

		tempFn.create(pos, edgeConnects, false, PRECISION);
		//printf("\n %i %i ", tempFn.numVertices(), tempFn.numEdges());
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

		//fnMesh.getIsoMesh(contourVertexValues, inThreshold, false, coutourMeshObj);
		
		zFnMesh tempFn(coutourMeshObj);
		tempFn.clear(); // clear memory if the mobject exists.

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unordered_map <string, int> positionVertex;

		float threshold = inThreshold; /*coreUtils.ofMap(inThreshold, 0.0f, 1.0f, contourValueDomain.min, contourValueDomain.max)*/;


		for (zItMeshFace f(*fieldObj); !f.end(); f++)
		{
			//printf("\n %i ", f.getId());
			getIsolinePoly(f, positions, polyConnects, polyCounts, positionVertex, threshold, invertMesh);
		}


		tempFn.create(positions, polyCounts, polyConnects);;

		printf("\n isoMesh %i %i ", tempFn.numVertices(), tempFn.numPolygons());
		
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
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Polygon(zObjGraph& inGraphObj, zPoint& p)
	{
		zItGraphVertex v(inGraphObj, 0);
		

		zItGraphHalfEdge he = v.getHalfEdge();
		zItGraphHalfEdge start = he;
	
		zPoint v0 = he.getVertex().getPosition();

		float d = (p - v0) * (p - v0);

		float s = 1.0;

		do
		{			
			
			zPoint vj = he.getStartVertex().getPosition();
			zPoint vi = he.getVertex().getPosition();

			zVector e = vj - vi;

			zVector w = p - vi;

			zVector b = w - e * coreUtils.ofClamp<float>((w * e) / (e * e), 0.0, 1.0);

			d = coreUtils.zMin(d, (b * b));

			bool c1 = (p.y >= vi.y);
			bool c2 = (p.y < vj.y);
			bool c3 = (e.x * w.y > e.y * w.x);

			if (c1 && c2 && c3) s *= -1.0;
			if (!c1 && !c2 && !c3) s *= -1.0;

			he = he.getNext();

		} while (he != start);

				
		return s * sqrt(d);
	}

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Circle(zPoint &cen, zPoint &p, float r)
	{
		return ((p - cen).length() - r);
	}

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Ellipse(zPoint& cen, zPoint& p, float a, float b)
	{
		p.x = abs(p.x);  p.y = abs(p.y); p.z = abs(p.z);
		
		zVector ab(a, b, 0);
		
		if (p.x > p.y) 
		{ 
			zPoint temp = p;
			p.x = temp.y;
			p.y = temp.x;

			temp = ab;
			ab.x = temp.y;
			ab.y = temp.x;		
		}

		float l = ab.y * ab.y - ab.x * ab.x;
		float m = ab.x * p.x / l;      float m2 = m * m;
		float n = ab.y * p.y / l;      float n2 = n * n;
		float c = (m2 + n2 - 1.0) / 3.0; float c3 = c * c * c;
		float q = c3 + m2 * n2 * 2.0;
		float d = c3 + m2 * n2;
		float g = m + m * n2;
		float co;
		if (d < 0.0)
		{
			float h = acos(q / c3) / 3.0;
			float s = cos(h);
			float t = sin(h) * sqrt(3.0);
			float rx = sqrt(-c * (s + t + 2.0) + m2);
			float ry = sqrt(-c * (s - t + 2.0) + m2);
			co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0;
		}
		else
		{
			float h = 2.0 * m * n * sqrt(d);
			float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
			float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
			float rx = -s - u - c * 4.0 + 2.0 * m2;
			float ry = (s - u) * sqrt(3.0);
			float rm = sqrt(rx * rx + ry * ry);
			co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0;
		}

		zVector m1(co, sqrt(1.0 - co * co), 0);
		zVector  r;
		r.x = ab.x  * m1.x; 
		r.y = ab.y * m1.y; 
		r.z = ab.z * m1.z;

		return (r - (p-cen)).length() * coreUtils.zSign(p.y - r.y);

		
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
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Triangle(zPoint& p, zPoint& p0, zPoint& p1, zPoint& p2)
	{
		zVector e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
		zVector v0 = p - p0, v1 = p - p1, v2 = p - p2;
		
		
		zVector pq0 = v0 - e0 * coreUtils.ofClamp((v0 * e0) / (e0 * e0), 0.0f, 1.0f);
		zVector pq1 = v1 - e1 * coreUtils.ofClamp((v1 * e1) / (e1 * e1), 0.0f, 1.0f);
		zVector pq2 = v2 - e2 * coreUtils.ofClamp((v2 * e2) / (e2 * e2), 0.0f, 1.0f);
		
		float s = coreUtils.zSign(e0.x * e2.y - e0.y * e2.x);
		
		
		zVector d = coreUtils.zMin(coreUtils.zMin(zVector((pq0* pq0), s * (v0.x * e0.y - v0.y * e0.x),0),
			zVector((pq1* pq1), s * (v1.x * e1.y - v1.y * e1.x),0)),
			zVector((pq2* pq2), s * (v2.x * e2.y - v2.y * e2.x),0));
		

		return -sqrt(d.x) * coreUtils.zSign(d.y);
	}

	template<>
	ZSPACE_INLINE float zFnMeshField<zScalar>::getScalar_Square(zPoint &p, zVector& cen, zVector &dimensions)
	{
		zPoint transP = p - cen;
		transP.x = abs(transP.x); 
		transP.y = abs(transP.y); 
		transP.z = abs(transP.z);

		zVector d = transP - dimensions;
		d.x = coreUtils.zMax(d.x, 0.0f);
		d.y = coreUtils.zMax(d.y, 0.0f);
		d.z = coreUtils.zMax(d.z, 0.0f);	
		
		float r = d.length() + coreUtils.zMin (coreUtils.zMax(d.x,d.y), 0.0f);
	
		return(r);
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
			float s0 = contourVertexValues[fVerts[0].getId()];

			zVector v1 = fVerts[1].getPosition();
			float s1 = contourVertexValues[fVerts[1].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			newPositions.push_back(fVerts[2].getPosition());

			newPositions.push_back(fVerts[3].getPosition());

			v1 = fVerts[3].getPosition();
			s1 = contourVertexValues[fVerts[3].getId()];

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);


		}

		// CASE 2
		if (MS_case == 2)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = contourVertexValues[fVerts[0].getId()];

			zVector v1 = fVerts[1].getPosition();
			float s1 = contourVertexValues[fVerts[1].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[2].getPosition();
			s0 = contourVertexValues[fVerts[2].getId()];

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[2].getPosition());

			newPositions.push_back(fVerts[3].getPosition());

		}

		// CASE 3
		if (MS_case == 3)
		{
			zVector v0 = fVerts[3].getPosition();
			float s0 = contourVertexValues[fVerts[3].getId()];

			zVector v1 = fVerts[0].getPosition();
			float s1 = contourVertexValues[fVerts[0].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[2].getPosition();
			s0 = contourVertexValues[fVerts[2].getId()];

			v1 = fVerts[1].getPosition();
			s1 = contourVertexValues[fVerts[1].getId()];

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
			float s0 = contourVertexValues[fVerts[1].getId()];

			zVector v1 = fVerts[2].getPosition();
			float s1 = contourVertexValues[fVerts[2].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[3].getPosition();
			s0 = contourVertexValues[fVerts[3].getId()];

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
				float s0 = contourVertexValues[fVerts[1].getId()];

				zVector v1 = fVerts[0].getPosition();
				float s1 = contourVertexValues[fVerts[0].getId()];

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[1].getPosition());

				v1 = fVerts[2].getPosition();
				s1 = contourVertexValues[fVerts[2].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[3].getPosition();
				s0 = contourVertexValues[fVerts[3].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[3].getPosition());

				v1 = fVerts[0].getPosition();
				s1 = contourVertexValues[fVerts[0].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}
			if (SaddleCase == 1)
			{
				// tri 1
				zVector v0 = fVerts[1].getPosition();
				float s0 = contourVertexValues[fVerts[1].getId()];

				zVector v1 = fVerts[0].getPosition();
				float s1 = contourVertexValues[fVerts[0].getId()];

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[1].getPosition());


				v1 = fVerts[2].getPosition();
				s1 = contourVertexValues[fVerts[2].getId()];
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2
				v0 = fVerts[3].getPosition();
				s0 = contourVertexValues[fVerts[3].getId()];
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fVerts[3].getPosition());

				v1 = fVerts[0].getPosition();
				s1 = contourVertexValues[fVerts[0].getId()];
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 6
		if (MS_case == 6)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = contourVertexValues[fVerts[0].getId()];
			zVector v1 = fVerts[1].getPosition();
			float s1 = contourVertexValues[fVerts[1].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[3].getPosition();
			s0 = fVerts[3].getColor().r;
			s0 = contourVertexValues[fVerts[3].getId()];
			v1 = fVerts[2].getPosition();
			s1 = contourVertexValues[fVerts[2].getId()];

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[3].getPosition());

		}


		// CASE 7
		if (MS_case == 7)
		{
			zVector v0 = fVerts[3].getPosition();
			float s0 = contourVertexValues[fVerts[3].getId()];
			zVector v1 = fVerts[2].getPosition();
			float s1 = contourVertexValues[fVerts[2].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[3].getPosition());

			v1 = fVerts[0].getPosition();
			s1 = contourVertexValues[fVerts[0].getId()];

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
			float s0 = contourVertexValues[fVerts[2].getId()];
			zVector v1 = fVerts[3].getPosition();
			float s1 = contourVertexValues[fVerts[3].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[0].getPosition();
			s0 = contourVertexValues[fVerts[0].getId()];
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 9
		if (MS_case == 9)
		{
			zVector v0 = fVerts[1].getPosition();
			float s0 = contourVertexValues[fVerts[1].getId()];
			zVector v1 = fVerts[0].getPosition();
			float s1 = contourVertexValues[fVerts[0].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			newPositions.push_back(fVerts[2].getPosition());

			v0 = fVerts[2].getPosition();
			s0 = contourVertexValues[fVerts[2].getId()];
			v1 = fVerts[3].getPosition();
			s1 = contourVertexValues[fVerts[3].getId()];

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
				float s0 = contourVertexValues[fVerts[0].getId()];
				zVector v1 = fVerts[1].getPosition();
				float s1 = contourVertexValues[fVerts[1].getId()];

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[2].getPosition();
				s0 = contourVertexValues[fVerts[2].getId()];
				v1 = fVerts[1].getPosition();
				s1 = contourVertexValues[fVerts[1].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fVerts[2].getPosition());

				v1 = fVerts[3].getPosition();
				s1 = contourVertexValues[fVerts[3].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fVerts[0].getPosition();
				s0 = contourVertexValues[fVerts[0].getId()];
				v1 = fVerts[3].getPosition();
				s1 = contourVertexValues[fVerts[3].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);
			}

			if (SaddleCase == 1)
			{
				// tri 1

				newPositions.push_back(fVerts[0].getPosition());

				zVector v0 = fVerts[0].getPosition();
				float s0 = contourVertexValues[fVerts[0].getId()];
				zVector v1 = fVerts[1].getPosition();
				float s1 = contourVertexValues[fVerts[1].getId()];

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v1 = fVerts[3].getPosition();
				s1 = contourVertexValues[fVerts[1].getId()];
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2

				v0 = fVerts[2].getPosition();
				s0 = contourVertexValues[fVerts[2].getId()];
				v1 = fVerts[1].getPosition();
				s1 = contourVertexValues[fVerts[1].getId()];

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fVerts[2].getPosition());

				v1 = fVerts[3].getPosition();
				s1 = contourVertexValues[fVerts[3].getId()];
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 11
		if (MS_case == 11)
		{
			zVector v0 = fVerts[2].getPosition();
			float s0 = contourVertexValues[fVerts[2].getId()];
			zVector v1 = fVerts[1].getPosition();
			float s1 = contourVertexValues[fVerts[1].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[2].getPosition());

			v1 = fVerts[3].getPosition();
			s1 = contourVertexValues[fVerts[3].getId()];
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);



		}

		// CASE 12
		if (MS_case == 12)
		{
			newPositions.push_back(fVerts[0].getPosition());

			newPositions.push_back(fVerts[1].getPosition());

			zVector v0 = fVerts[1].getPosition();
			float s0 = contourVertexValues[fVerts[1].getId()];
			zVector v1 = fVerts[2].getPosition();
			float s1 = contourVertexValues[fVerts[2].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fVerts[0].getPosition();
			s0 = contourVertexValues[fVerts[0].getId()];
			v1 = fVerts[3].getPosition();
			s1 = contourVertexValues[fVerts[3].getId()];

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 13
		if (MS_case == 13)
		{
			zVector v0 = fVerts[1].getPosition();
			float s0 = contourVertexValues[fVerts[1].getId()];
			zVector v1 = fVerts[0].getPosition();
			float s1 = contourVertexValues[fVerts[0].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fVerts[1].getPosition());

			v1 = fVerts[2].getPosition();
			s1 = contourVertexValues[fVerts[2].getId()];

			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 14
		if (MS_case == 14)
		{
			newPositions.push_back(fVerts[0].getPosition());

			zVector v0 = fVerts[0].getPosition();
			float s0 = contourVertexValues[fVerts[0].getId()];
			zVector v1 = fVerts[1].getPosition();
			float s1 = contourVertexValues[fVerts[1].getId()];

			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v1 = fVerts[3].getPosition();
			s1 = contourVertexValues[fVerts[3].getId()];

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
			/*for (int i = 0; i < newPositions.size(); i++)
			{
				newPositions[i] = coreUtils.factoriseVector(newPositions[i], 6);

			}*/
						
			
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

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					/*string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;*/
					coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions.size());
		}


		// only if there are 2 tris Case : 5,10

		// Edge Length Check

		if (newPositions2.size() >= 3)
		{
			/*for (int i = 0; i < newPositions2.size(); i++)
			{
				newPositions2[i] = coreUtils.factoriseVector(newPositions2[i], 6);

			}*/

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

				bool vExists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					/*string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;*/
					coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
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