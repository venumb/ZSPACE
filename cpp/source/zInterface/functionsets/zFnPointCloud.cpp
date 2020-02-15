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


#include<headers/zInterface/functionsets/zFnPointCloud.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnPointCloud::zFnPointCloud()
	{
		fnType = zFnType::zPointsFn;
		pointsObj = nullptr;

	}

	ZSPACE_INLINE zFnPointCloud::zFnPointCloud(zObjPointCloud &_pointsObj)
	{
		pointsObj = &_pointsObj;
		fnType = zFnType::zPointsFn;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnPointCloud::~zFnPointCloud() {}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE zFnType zFnPointCloud::getType()
	{
		return zFnType::zPointsFn;
	}

	ZSPACE_INLINE void zFnPointCloud::from(string path, zFileTpye type, bool staticGeom)
	{
		if (type == zCSV) fromCSV(path);
	}

	ZSPACE_INLINE void zFnPointCloud::to(string path, zFileTpye type)
	{
		if (type == zCSV) toCSV(path);
	}

	ZSPACE_INLINE void zFnPointCloud::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		pointsObj->getBounds(minBB, maxBB);
	}

	ZSPACE_INLINE void zFnPointCloud::clear()
	{
		pointsObj->pCloud.clear();
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnPointCloud::create(zPointArray(&_positions))
	{
		pointsObj->pCloud.create(_positions);

	}

	//---- APPEND METHODS

	ZSPACE_INLINE void zFnPointCloud::addPosition(zPoint &_position)
	{
		pointsObj->pCloud.addVertex(_position);
	}

	ZSPACE_INLINE void zFnPointCloud::addPositions(zPointArray &_positions)
	{
		for (int i = 0; i < _positions.size(); i++) addPosition(_positions[i]);
	}

	//---- QUERY METHODS

	ZSPACE_INLINE int zFnPointCloud::numVertices()
	{
		return  pointsObj->pCloud.n_v;
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zFnPointCloud::setVertexColor(zColor col)
	{

		for (int i = 0; i < pointsObj->pCloud.vertexColors.size(); i++)
		{
			pointsObj->pCloud.vertexColors[i] = col;
		}
	}

	ZSPACE_INLINE void zFnPointCloud::setVertexColors(zColorArray& col)
	{
		if (col.size() != pointsObj->pCloud.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

		for (int i = 0; i < pointsObj->pCloud.vertexColors.size(); i++)
		{
			pointsObj->pCloud.vertexColors[i] = col[i];
		}
	}

	//--- GET METHODS 

	ZSPACE_INLINE void zFnPointCloud::getVertexPositions(zPointArray& pos)
	{
		pos = pointsObj->pCloud.vertexPositions;
	}

	ZSPACE_INLINE zPoint* zFnPointCloud::getRawVertexPositions()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &pointsObj->pCloud.vertexPositions[0];
	}

	ZSPACE_INLINE void zFnPointCloud::getVertexColors(zColorArray& col)
	{
		col = pointsObj->pCloud.vertexColors;
	}

	ZSPACE_INLINE zColor* zFnPointCloud::getRawVertexColors()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &pointsObj->pCloud.vertexColors[0];
	}

	//---- TRANSFORM METHODS OVERRIDES

	ZSPACE_INLINE void zFnPointCloud::setTransform(zTransform &inTransform, bool decompose, bool updatePositions)
	{
		if (updatePositions)
		{
			zTransformationMatrix to;
			to.setTransform(inTransform, decompose);

			zTransform transMat = pointsObj->transformationMatrix.getToMatrix(to);
			transformObject(transMat);

			pointsObj->transformationMatrix.setTransform(inTransform);

			// update pivot values of object transformation matrix
			zVector p = pointsObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}
		else
		{
			pointsObj->transformationMatrix.setTransform(inTransform, decompose);

			zVector p = pointsObj->transformationMatrix.getO();
			setPivot(p);

		}

	}

	ZSPACE_INLINE void zFnPointCloud::setScale(zFloat4 &scale)
	{
		// get  inverse pivot translations
		zTransform invScalemat = pointsObj->transformationMatrix.asInverseScaleTransformMatrix();

		// set scale values of object transformation matrix
		pointsObj->transformationMatrix.setScale(scale);

		// get new scale transformation matrix
		zTransform scaleMat = pointsObj->transformationMatrix.asScaleTransformMatrix();

		// compute total transformation
		zTransform transMat = invScalemat * scaleMat;

		// transform object
		transformObject(transMat);
	}

	ZSPACE_INLINE void zFnPointCloud::setRotation(zFloat4 &rotation, bool appendRotations)
	{
		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = pointsObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = pointsObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = pointsObj->transformationMatrix;
		to.setRotation(rotation, appendRotations);
		zTransform toMat = pointsObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set rotation values of object transformation matrix
		pointsObj->transformationMatrix.setRotation(rotation, appendRotations);;
	}

	ZSPACE_INLINE void zFnPointCloud::setTranslation(zVector &translation, bool appendTranslations)
	{
		// get vector as zDouble3
		zFloat4 t;
		translation.getComponents(t);

		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = pointsObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = pointsObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = pointsObj->transformationMatrix;
		to.setTranslation(t, appendTranslations);
		zTransform toMat = pointsObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set translation values of object transformation matrix
		pointsObj->transformationMatrix.setTranslation(t, appendTranslations);;

		// update pivot values of object transformation matrix
		zVector p = pointsObj->transformationMatrix.getPivot();
		p = p * transMat;
		setPivot(p);

	}

	ZSPACE_INLINE void zFnPointCloud::setPivot(zVector &pivot)
	{
		// get vector as zDouble3
		zFloat4 p;
		pivot.getComponents(p);

		// set pivot values of object transformation matrix
		pointsObj->transformationMatrix.setPivot(p);
	}

	ZSPACE_INLINE void zFnPointCloud::getTransform(zTransform &transform)
	{
		transform = pointsObj->transformationMatrix.asMatrix();
	}

	//---- PROTECTED OVERRIDE METHODS

	ZSPACE_INLINE void zFnPointCloud::transformObject(zTransform &transform)
	{

		if (numVertices() == 0) return;


		zVector* pos = getRawVertexPositions();

		for (int i = 0; i < numVertices(); i++)
		{

			zVector newPos = pos[i] * transform;
			pos[i] = newPos;
		}

	}

	//---- PROTECTED FACTORY METHODS

	ZSPACE_INLINE void zFnPointCloud::fromCSV(string infilename)
	{
		pointsObj->pCloud.vertexPositions.clear();

		ifstream myfile;
		myfile.open(infilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << infilename.c_str() << endl;
			return;

		}

		while (!myfile.eof())
		{
			string str;
			getline(myfile, str);

			vector<string> perlineData = pointsObj->pCloud.coreUtils.splitString(str, " ");

			if (perlineData.size() > 0)
			{
				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						pointsObj->pCloud.vertexPositions.push_back(pos);
					}
					//printf("\n working vertex");
				}


			}
		}

		myfile.close();



		printf("\n inPositions: %i ", pointsObj->pCloud.vertexPositions.size());


	}

	ZSPACE_INLINE void zFnPointCloud::toCSV(string outfilename)
	{

		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		// vertex positions
		for (int i = 0; i < pointsObj->pCloud.vertexPositions.size(); i++)
		{

			myfile << "\n v " << pointsObj->pCloud.vertexPositions[i].x << " " << pointsObj->pCloud.vertexPositions[i].y << " " << pointsObj->pCloud.vertexPositions[i].z;

		}

		myfile.close();

		cout << endl << " TXT exported. File:   " << outfilename.c_str() << endl;
	}

}