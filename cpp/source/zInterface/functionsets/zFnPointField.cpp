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


#include<headers/zInterface/functionsets/zFnPointField.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	template<typename T>
	ZSPACE_INLINE zFnPointField<T>::zFnPointField()
	{
		fnType = zFnType::zPointFieldFn;
		fieldObj = nullptr;
	}

	template<typename T>
	ZSPACE_INLINE zFnPointField<T>::zFnPointField(zObjPointField<T> &_fieldObj)
	{
		fieldObj = &_fieldObj;
		fnPoints = zFnPointCloud(_fieldObj);

		fnType = zFnType::zPointFieldFn;

	}

	//---- DESTRUCTOR

	template<typename T>
	ZSPACE_INLINE zFnPointField<T>::~zFnPointField() {}

	//--------------------------
	//---- FACTORY METHODS
	//--------------------------

	template<typename T>
	ZSPACE_INLINE zFnType zFnPointField<T>::getType()
	{
		return zFnType::zPointFieldFn;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		minBB = fieldObj->field.minBB;
		maxBB = fieldObj->field.maxBB;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::clear()
	{

		fieldObj->field.fieldValues.clear();

		fnPoints.clear();
	}

	//--------------------------
	//---- CREATE METHODS
	//--------------------------

	//---- zScalar &  zVector specilization for create

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR)
	{
		fieldObj->field = zField3D<zScalar>(_minBB, _maxBB, _n_X, _n_Y, _n_Z);

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		int i = 0;
		for (zItPointScalarField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		// create field points
		createPointCloud();
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR)
	{
		fieldObj->field = zField3D<zVector>(_minBB, _maxBB, _n_X, _n_Y, _n_Z);

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		int i = 0;
		for (zItPointVectorField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		// create field points
		createPointCloud();
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::create(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zPoint _minBB, int _NR)
	{
		fieldObj->field = zField3D<zScalar>(_unit_X, _unit_Y, _unit_Z, _n_X, _n_Y, _n_Z, _minBB);

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		int i = 0;
		for (zItPointScalarField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		// create field points
		createPointCloud();
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::create(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zPoint _minBB, int _NR)
	{
		fieldObj->field = zField3D<zVector>(_unit_X, _unit_Y, _unit_Z, _n_X, _n_Y, _n_Z, _minBB);

		// compute neighbours
		ringNeighbours.clear();
		adjacentNeighbours.clear();

		int i = 0;
		for (zItPointVectorField s(*fieldObj); !s.end(); s++, i++)
		{
			vector<int> temp_ringNeighbour;
			s.getNeighbour_Ring(_NR, temp_ringNeighbour);
			ringNeighbours[i] = (temp_ringNeighbour);

			vector<int> temp_adjacentNeighbour;
			s.getNeighbour_Adjacents(temp_adjacentNeighbour);
			adjacentNeighbours[i] = (temp_adjacentNeighbour);
		}

		// create field points
		createPointCloud();
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::createVectorFromScalarField(zFnPointField<zScalar> &inFnScalarField)
	{
		zVector minBB, maxBB;
		inFnScalarField.getBoundingBox(minBB, maxBB);

		int n_X, n_Y, n_Z;
		inFnScalarField.getResolution(n_X, n_Y, n_Z);

		vector<zVector> gradients = inFnScalarField.getGradients();

		create(minBB, maxBB, n_X, n_Y, n_Z);
		setFieldValues(gradients);
	}

	//--- FIELD QUERY METHODS 

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::getNeighbour_Contained(zPoint &pos, zIntArray &containedNeighbour)
	{

	}

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::getNeighbourPosition_Contained(zPoint &pos, zPointArray &containedNeighbour)
	{

	}

	//---- GET METHODS

	template<typename T>
	ZSPACE_INLINE int zFnPointField<T>::numFieldValues()
	{
		return fieldObj->field.fieldValues.size();
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getResolution(int &_n_X, int &_n_Y, int &_n_Z)
	{
		_n_X = fieldObj->field.n_X;
		_n_Y = fieldObj->field.n_Y;
		_n_Z = fieldObj->field.n_Z;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getUnitDistances(double &_unit_X, double &_unit_Y, double &_unit_Z)
	{
		_unit_X = fieldObj->field.unit_X;
		_unit_Y = fieldObj->field.unit_Y;
		_unit_Z = fieldObj->field.unit_Z;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getBoundingBox(zPoint &_minBB, zPoint &_maxBB)
	{
		_minBB = fieldObj->field.minBB;
		_maxBB = fieldObj->field.maxBB;
	}

	//---- zScalar and zVector specilization for getFieldValue

	template<>
	ZSPACE_INLINE bool zFnPointField<zScalar>::getFieldValue(zPoint &samplePos, zFieldValueType type, zScalar& fieldValue)
	{

		bool out = false;

		zItPointScalarField s(*fieldObj, samplePos);

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

			zItPointScalarField s(*fieldObj, index);

			vector<zItPointScalarField> ringNeighbours;
			s.getNeighbour_Ring(1, ringNeighbours);

			zPointArray positions;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				positions.push_back(ringNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w=0;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				zScalar val = ringNeighbours[i].getValue();
				fVal += val * weights[i];
				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldAdjacentWeighted)
		{
			zScalar fVal=0;

			zItPointScalarField s(*fieldObj, index);

			vector<zItPointScalarField> adjNeighbours;
			s.getNeighbour_Adjacents(adjNeighbours);

			zPointArray positions;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				positions.push_back(adjNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w=0;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				zScalar val = adjNeighbours[i].getValue();
				fVal += val * weights[i];

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else throw std::invalid_argument(" error: invalid zFieldValueType.");

		return true;
	}

	template<>
	ZSPACE_INLINE bool zFnPointField<zVector>::getFieldValue(zVector &samplePos, zFieldValueType type, zVector& fieldValue)
	{

		bool out = false;

		zItPointVectorField s(*fieldObj, samplePos);

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

			zItPointVectorField s(*fieldObj, index);

			vector<zItPointVectorField> ringNeighbours;
			s.getNeighbour_Ring(1, ringNeighbours);

			zPointArray positions;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				positions.push_back(ringNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w =0;
			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				zVector val = ringNeighbours[i].getValue();
				fVal += val * weights[i];
				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else if (type == zFieldAdjacentWeighted)
		{
			zVector fVal;

			zItPointVectorField s(*fieldObj, index);

			vector<zItPointVectorField> adjNeighbours;
			s.getNeighbour_Adjacents(adjNeighbours);

			zPointArray positions;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				positions.push_back(adjNeighbours[i].getPosition());
			}

			vector<double> weights;
			coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

			double w=0;
			for (int i = 0; i < adjNeighbours.size(); i++)
			{
				zVector val = adjNeighbours[i].getValue();
				fVal += val * weights[i];

				w += weights[i];
			}

			fVal /= w;

			fieldValue = fVal;
		}

		else throw std::invalid_argument(" error: invalid zFieldValueType.");

		return true;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getFieldValues(vector<T>& fieldValues)
	{
		fieldValues = fieldObj->field.fieldValues;
	}

	//---- zScalar specilization for getFieldValue
	
	template<>
	ZSPACE_INLINE zVector zFnPointField<zScalar>::getGradient(zItPointScalarField &s, float epsilon)
	{

		bool out = true;

		zVector samplePos = s.getPosition();

		int id_X, id_Y, id_Z;
		s.getIndices( id_X, id_Y, id_Z);

		if (id_X == 0 || id_Y == 0 || id_Z == 0 || id_X == fieldObj->field.n_X - 1 || id_Y == fieldObj->field.n_Y - 1 || id_Z == fieldObj->field.n_Z - 1)
		{
			return zVector();
		}


		zItPointScalarField s1(*fieldObj, id_X + 1, id_Y, id_Z);
		zVector samplePos1 = s1.getPosition();
		zScalar fieldVal1 = s1.getValue();

		zItPointScalarField s2(*fieldObj, id_X, id_Y + 1, id_Z);
		zVector samplePos2 = s2.getPosition();
		zScalar fieldVal2 = s2.getValue();

		zItPointScalarField s3(*fieldObj, id_X, id_Y, id_Z + 1);
		zVector samplePos3 = s3.getPosition();
		zScalar fieldVal3 = s3.getValue();

		zScalar fieldVal = s.getValue();

		zScalar gX = coreUtils.ofMap(samplePos.x + epsilon, samplePos.x, samplePos1.x, fieldVal, fieldVal1) - fieldVal;
		zScalar gY = coreUtils.ofMap(samplePos.y + epsilon, samplePos.y, samplePos2.y, fieldVal, fieldVal2) - fieldVal;
		zScalar gZ = coreUtils.ofMap(samplePos.z + epsilon, samplePos.z, samplePos2.z, fieldVal, fieldVal3) - fieldVal;

		zVector gradient = zVector(gX, gY, gZ);
		gradient /= (3.0 * epsilon);

		return gradient;
	}

	template<>
	ZSPACE_INLINE vector<zVector> zFnPointField<zScalar>::getGradients(float epsilon)
	{
		vector<zVector> out;

		for (zItPointScalarField s(*fieldObj); !s.end(); s++)
		{
			out.push_back(getGradient(s, epsilon));
		}

		return out;
	}

	//---- SET METHODS

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::setFieldColorDomain(zDomainColor &colDomain)
	{
		fieldColorDomain = colDomain;
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::setBoundingBox(zPoint &_minBB, zPoint &_maxBB)
	{
		fieldObj->field.minBB = _minBB;
		fieldObj->field.maxBB = _maxBB;
	}

	//---- zScalar and zVector specilization for setFieldValues

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::setFieldValues(zScalarArray &fValues)
	{
		if (fValues.size() == numFieldValues())
		{
			int i = 0;
			for (zItPointScalarField s(*fieldObj); !s.end(); s++, i++)
			{
				s.setValue(fValues[i]);
			}

			updateColors();
		}

	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::setFieldValues(vector<zVector>& fValues)
	{
		if (fValues.size() == numFieldValues())
		{
			int i = 0;
			for (zItPointVectorField s(*fieldObj); !s.end(); s++, i++)
			{
				s.setValue(fValues[i]);
			}

		}

		else throw std::invalid_argument("input fValues size not field vectors.");
	}

	//----  3D IDW FIELD METHODS

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjMesh &inMeshObj, T meshValue, double influence, double power, bool normalise)
	{
		fieldValues.clear();
		zFnMesh inFnMesh(inMeshObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnMesh.numVertices(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

				if (r < tempDist)
				{
					double w = pow(r, power);
					wSum += w;
					double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

					d = meshValue * val;

					tempDist = r;
				}

			}

			if (wSum > 0)  d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)
		{
			normliseValues(fieldValues);
		}

	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjGraph &inGraphObj, T graphValue, double influence, double power, bool normalise)
	{
		fieldValues.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnGraph.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnGraph.numVertices(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

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
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, T value, double influence, double power, bool normalise)
	{

		fieldValues.clear();
		zFnPointCloud inFnPoints(inPointsObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnPoints.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnPoints.numVertices(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

				d += (value * val);
			}


			if (wSum > 0)  d /= wSum;
			d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);



	}

	template <typename T>
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, vector<T> &values, vector<double>& influences, double power, bool normalise )
	{
		fieldValues.clear();
		zFnPointCloud inFnPoints(inPointsObj);


		if (inFnPoints.numVertices() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
		if (inFnPoints.numVertices() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnPoints.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnPoints.numVertices(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

				double w = pow(r, power);
				wSum += w;

				double val = (w > 0.0) ? ((r * influences[j]) / (w)) : 0.0;;

				d += (values[j] * val);
			}


			if (wSum > 0)  d /= wSum;
			else d = T();

			fieldValues.push_back(d);
		}

		if (normalise)	normliseValues(fieldValues);
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, T value, double influence, double power, bool normalise)
	{

		fieldValues.clear();

		zVector *positions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

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
	ZSPACE_INLINE void zFnPointField<T>::getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, vector<T> &values, vector<double>& influences, double power, bool normalise)
	{
		if (inPositions.size() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
		if (inPositions.size() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

		zVector *positions = fnPoints.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			T d;
			double wSum = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inPositions.size(); j++)
			{
				double r = positions[i].distanceTo(inPositions[j]);

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

	//----  3D SCALAR FIELD METHODS

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, bool normalise)
	{
		scalars.clear();

		zFnPointCloud inFnPoints(inPointsObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnPoints.getRawVertexPositions();

		vector<double> distVals;
		double dMin = 100000;
		double dMax = 0;;

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			distVals.push_back(10000);
		}

		for (int j = 0; j < fnPoints.numVertices(); j++)
		{

			for (int i = 0; i < inFnPoints.numVertices(); i++)
			{

				double dist = positions[i].squareDistanceTo(inPositions[j]);

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

		for (int j = 0; j < fnPoints.numVertices(); j++)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise)
	{
		scalars.clear();

		zFnMesh inFnMesh(inMeshObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnMesh.numVertices(); j++)
			{
				double r = positions[i].squareDistanceTo(inPositions[j]);

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
	ZSPACE_INLINE void zFnPointField<zScalar>::getScalarsAsVertexDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise)
	{
		scalars.clear();

		zFnGraph inFnGraph(inGraphObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnGraph.getRawVertexPositions();

		// update values from meta balls

		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inFnGraph.numVertices(); j++)
			{
				double r = positions[i].squareDistanceTo(inPositions[j]);

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
	ZSPACE_INLINE void zFnPointField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise)
	{
		scalars.clear();
		zFnMesh inFnMesh(inMeshObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnMesh.getRawVertexPositions();

		// update values from edge distance
		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (zItMeshEdge e(inMeshObj); !e.end(); e++)
			{

				int e0 = e.getHalfEdge(0).getVertex().getId();
				int e1 = e.getHalfEdge(0).getStartVertex().getId();

				zVector closestPt;

				double r = coreUtils.minDist_Edge_Point(positions[i], inPositions[e0], inPositions[e1], closestPt);


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
	ZSPACE_INLINE void zFnPointField<zScalar>::getScalarsAsEdgeDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise )
	{
		scalars.clear();
		zFnGraph inFnGraph(inGraphObj);

		zVector *positions = fnPoints.getRawVertexPositions();
		zVector *inPositions = inFnGraph.getRawVertexPositions();

		// update values from edge distance
		for (int i = 0; i < fnPoints.numVertices(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (zItGraphEdge e(inGraphObj); !e.end(); e++)
			{

				int e0 = e.getHalfEdge(0).getVertex().getId();
				int e1 = e.getHalfEdge(0).getStartVertex().getId();

				zVector closestPt;

				double r = coreUtils.minDist_Edge_Point(positions[i], inPositions[e0], inPositions[e1], closestPt);


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

	//--- COMPUTE METHODS 

	template<typename T>
	ZSPACE_INLINE bool zFnPointField<T>::checkBounds_X(int index_X)
	{
		return (index_X < fieldObj->field.n_X && index_X >= 0);
	}

	template<typename T>
	ZSPACE_INLINE bool zFnPointField<T>::checkBounds_Y(int index_Y)
	{
		return (index_Y < fieldObj->field.n_Y && index_Y >= 0);
	}

	template<typename T>
	ZSPACE_INLINE bool zFnPointField<T>::checkBounds_Z(int index_Z)
	{
		return (index_Z < fieldObj->field.n_Z && index_Z >= 0);
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::computeMinMaxOfScalars(vector<T> &values, T &dMin, T &dMax)
	{
		dMin = coreUtils.zMin(values);
		dMax = coreUtils.zMax(values);
	}

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::computeDomain(vector<T> &values, zDomain <T> &domain)
	{
		domain.min = coreUtils.zMin(values);
		domain.max = coreUtils.zMax(values);
	}

	//---- zScalar and zVector specilization for normliseValues

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::normliseValues(zScalarArray &fieldValues)
	{
		zDomainFloat d;
		computeDomain(fieldValues, d);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = d.max - fieldValues[i];

		computeDomain(fieldValues, d);

		zDomainFloat out(-1.0, 1.0);
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = coreUtils.ofMap(fieldValues[i], d, out);
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}

	//---- zScalar specilization for smoothField

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::smoothField(int numSmooth, double diffuseDamp , zDiffusionType type)
	{
		for (int k = 0; k < numSmooth; k++)
		{
			vector<float> tempValues;

			for (zItPointScalarField s(*fieldObj); !s.end(); s++)
			{
				double lapA = 0;

				vector<zItPointScalarField> aNeigbours;
				s.getNeighbour_Adjacents(aNeigbours);

				for (int j = 0; j < aNeigbours.size(); j++)
				{
					int id = aNeigbours[j].getId();
					zScalar val = aNeigbours[j].getValue();

					if (type == zLaplacian)
					{
						if (id != s.getId()) lapA += (val * 1);
						else lapA += (val * -6);
					}
					else if (type == zAverage)
					{
						lapA += (val * 1);
					}
				}



				if (type == zLaplacian)
				{
					double val1 = s.getValue();

					double newA = val1 + (lapA * diffuseDamp);
					tempValues.push_back(newA);
				}
				else if (type == zAverage)
				{
					if (lapA != 0) lapA /= (aNeigbours.size());

					tempValues.push_back(lapA);
				}

			}

			setFieldValues(tempValues);

		}

		updateColors();

	}

	//---- zScalar & zVector specilization for computePositionsInFieldIndex

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::computePositionsInFieldIndex(zPointArray &positions, vector<zPointArray> &fieldIndexPositions)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<zVector> temp;
			fieldIndexPositions.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItPointScalarField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositions[fieldIndex].push_back(positions[i]);
		}
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::computePositionsInFieldIndex(zPointArray &positions, vector<zPointArray> &fieldIndexPositions)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<zVector> temp;
			fieldIndexPositions.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItPointVectorField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositions[fieldIndex].push_back(positions[i]);
		}
	}

	//---- zScalar & zVector specilization for computePositionIndicesInFieldIndex

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::computePositionIndicesInFieldIndex(zPointArray &positions, vector<zIntArray> &fieldIndexPositionIndicies)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<int> temp;
			fieldIndexPositionIndicies.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItPointScalarField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositionIndicies[fieldIndex].push_back(i);
		}
	}

	template<>
	ZSPACE_INLINE void zFnPointField<zVector>::computePositionIndicesInFieldIndex(zPointArray &positions, vector<zIntArray> &fieldIndexPositionIndicies)
	{
		for (int i = 0; i < numFieldValues(); i++)
		{
			vector<int> temp;
			fieldIndexPositionIndicies.push_back(temp);
		}


		for (int i = 0; i < positions.size(); i++)
		{
			zItPointVectorField s(*fieldObj, positions[i]);
			int fieldIndex = s.getId();

			fieldIndexPositionIndicies[fieldIndex].push_back(i);
		}
	}

	template<typename T>
	ZSPACE_INLINE double zFnPointField<T>::F_of_r(double &r, double &a, double &b)
	{
		if (0 <= r && r <= b / 3.0)return (a * (1.0 - (3.0 * r * r) / (b*b)));
		if (b / 3.0 <= r && r <= b) return (3 * a / 2 * pow(1.0 - (r / b), 2.0));
		if (b <= r) return 0;
	}

	//----  SCALAR BOOLEAN METHODS

	template<>
	ZSPACE_INLINE void zFnPointField<zScalar>::boolean_union(zScalarArray& scalars0, zScalarArray& scalars1, zScalarArray& scalarsResult, bool normalise)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::boolean_subtract(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::boolean_intersect(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::boolean_difference(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::boolean_clipwithPlane(zScalarArray& scalars, zMatrix4& clipPlane)
	{
		int i = 0;
		for (zItPointCloudVertex v(*fieldObj); !v.end(); v++, i++)
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
	ZSPACE_INLINE void zFnPointField<zScalar>::updateColors()
	{

		vector<float> scalars;
		getFieldValues(scalars);

		computeDomain(scalars, contourValueDomain);

		//convert to HSV

		if (contourValueDomain.min == contourValueDomain.max)
		{
			zColor col(0.5, 0.5, 0.5);

			fnPoints.setVertexColor(col);

			return;
		}

		else
		{
			fieldColorDomain.min.toHSV(); fieldColorDomain.max.toHSV();

			zColor* cols = fnPoints.getRawVertexColors();
			
			for (int i = 0; i < scalars.size(); i++)
			{

				if (scalars[i] < contourValueDomain.min) cols[i] = fieldColorDomain.min;
				else if (scalars[i] > contourValueDomain.max) cols[i] = fieldColorDomain.max;
				else
				{
					cols[i] = coreUtils.blendColor(scalars[i], contourValueDomain, fieldColorDomain, zHSV);
				}

			}
			
		}

	}

	//----  PROTECTED METHODS

	template<typename T>
	ZSPACE_INLINE void zFnPointField<T>::createPointCloud()
	{
		vector<zVector>positions;

		zVector minBB, maxBB;
		double unit_X, unit_Y, unit_Z;
		int n_X, n_Y, n_Z;

		getUnitDistances(unit_X, unit_Y, unit_Z);
		getResolution(n_X, n_Y, n_Z);

		getBoundingBox(minBB, maxBB);

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


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zFnPointField<zVector>;

	template class zFnPointField<zScalar>;

#endif
}