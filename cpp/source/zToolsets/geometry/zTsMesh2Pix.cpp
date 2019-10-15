// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zToolsets/geometry/zTsMesh2Pix.h>


namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsMesh2Pix::zTsMesh2Pix() {}

	ZSPACE_INLINE zTsMesh2Pix::zTsMesh2Pix(zObjMesh &_meshObj, int _maxVerts, int _maxEdges, int _maxFaces)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);

		maxVertices = _maxVerts;
		maxEdges = _maxEdges;
		maxFaces = _maxFaces;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsMesh2Pix::~zTsMesh2Pix() {}

	//---- GENERATE DATA METHODS
	
	ZSPACE_INLINE void zTsMesh2Pix::printSupport2Pix(string directory, double angle_threshold, bool perturbPositions, zVector perturbVal)
	{
		vector<MatrixXd> outMat_A;
		vector<MatrixXd> outMat_B;

		if (!perturbPositions)
		{
			outMat_A.clear();

			// get Matrix from vertex normals
			zDomainDouble outDomain_A(0.0, 1.0);
			getMatrixFromNormals(zVertexVertex, outDomain_A, outMat_A);

			// get edge length data 
			zDoubleArray heLength;
			zIntPairArray hedgeVertexPair;
			zDomainDouble outDomain(0.0, 0.9);

			for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
			{
				heLength.push_back(he.getLength());

				zIntPair vertPair;
				vertPair.first = he.getStartVertex().getId();
				vertPair.second = he.getVertex().getId();

				hedgeVertexPair.push_back(vertPair);
			}

			getMatrixFromContainer(zVertexVertex, heLength, hedgeVertexPair, outDomain, outMat_A);

			string path1 = directory + "train/image_0_A.bmp";
			coreUtils.matrixBMP(outMat_A, path1);


			// support matrix 
			outMat_B.clear();

			zDomainDouble outDomain_B(0.0, 0.9);

			zBoolArray supports;
			getVertexSupport(angle_threshold, supports);

			getMatrixFromContainer(zVertexVertex, supports, outDomain_B, outMat_B);

			string path2 = directory + "train/image_0_B.bmp";
			coreUtils.matrixBMP(outMat_B, path2);
		}
		
		else
		{

		}

	}
	
	//---- PRIVATE GET METHODS

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromNormals(zConnectivityType type, zDomainDouble &outDomain, vector<MatrixXd> &normMat)
	{
		if (type == zVertexData)
		{		

			// get vertex normals diagonal matrix
			zVectorArray norms;
			fnMesh.getVertexNormals(norms);
			getMatrixFromContainer(type, norms, outDomain, normMat);
			
		}

		else if (type == zFaceData)
		{

			// get face normals diagonal matrix
			zVectorArray norms;
			fnMesh.getFaceNormals(norms);		

			getMatrixFromContainer(type, norms, outDomain, normMat);
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}
	
	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, zVectorArray &data, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexData)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			MatrixXd normsX(n_v, n_v);
			MatrixXd normsY(n_v, n_v);
			MatrixXd normsZ(n_v, n_v);

			for (int i = 0; i < data.size(); i++)
			{
				normsX(i, i) = coreUtils.ofMap(data[i].x, -1.0, 1.0, 0.0, 1.0);
				normsY(i, i) = coreUtils.ofMap(data[i].y, -1.0, 1.0, 0.0, 1.0);
				normsZ(i, i) = coreUtils.ofMap(data[i].z, -1.0, 1.0, 0.0, 1.0);
			}


			outMat.push_back(normsX);
			outMat.push_back(normsY);
			outMat.push_back(normsZ);
		}

		else if (type == zFaceData)
		{
			int n_f = (maxFaces != -1) ? maxFaces : fnMesh.numPolygons();

			MatrixXd normsX(n_f, n_f);
			MatrixXd normsY(n_f, n_f);
			MatrixXd normsZ(n_f, n_f);

			for (int i = 0; i < data.size(); i++)
			{
				normsX(i, i) = coreUtils.ofMap(data[i].x, -1.0, 1.0, outDomain.min, outDomain.max);
				normsY(i, i) = coreUtils.ofMap(data[i].y, -1.0, 1.0, outDomain.min, outDomain.max);
				normsZ(i, i) = coreUtils.ofMap(data[i].z, -1.0, 1.0, outDomain.min, outDomain.max);
			}


			outMat.push_back(normsX);
			outMat.push_back(normsY);
			outMat.push_back(normsZ);
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");

	}

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, zBoolArray &data, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexData)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			MatrixXd dataX(n_v, n_v);
			MatrixXd dataY(n_v, n_v);
			MatrixXd dataZ(n_v, n_v);

			dataZ.setOnes();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				if (data[i]) dataX(i, i) = 1.0;
				else  dataY(i, i) = 1.0;

				dataZ(i, i) = 0.0;
			}

		}

		else if (type == zFaceData)
		{
			int n_f = (maxFaces != -1) ? maxFaces : fnMesh.numPolygons();
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, zDoubleArray &data, zIntPairArray &dataPair, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexData)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			MatrixXd temp(n_v, n_v);

			zDomainDouble inDomain;

			inDomain.min = coreUtils.zMin(data);
			inDomain.max = coreUtils.zMax(data);

			for (int k = 0; k < dataPair.size(); k++)
			{
				int  i = dataPair[k].first;
				int  j = dataPair[k].second;

				temp(i, j) = coreUtils.ofMap(data[k], inDomain, outDomain);
			}		
			
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}
	
	//---- PRIVATE COMPUTE METHODS

	ZSPACE_INLINE void zTsMesh2Pix::getVertexSupport(double angle_threshold, zBoolArray &support)
	{
		int numVerts_lowPoly = fnMesh.numVertices();
		
		support.assign(numVerts_lowPoly, false);

		// get Duplicate
		zObjMesh smoothMesh;
		fnMesh.getDuplicate(smoothMesh);
		
		// get smooth mesh

		zFnMesh fnSmoothMesh(smoothMesh);
		fnSmoothMesh.smoothMesh(3);
		
		// get bounds
		zVector minBB, maxBB;
		fnSmoothMesh.getBounds(minBB, maxBB);

		// compute nearest lowest vertex
		zVector* positions = fnSmoothMesh.getRawVertexPositions();

		for (zItMeshVertex vIt(*meshObj); !vIt.end(); vIt++)
		{

			if (vIt.getId() >= numVerts_lowPoly) continue;

			zIntArray cVerts;
			vIt.getConnectedVertices(cVerts);

			int lowestId;
			double val = 10e10;
			for (auto vId : cVerts)
			{
				double zVal = positions[vId].z;

				if (zVal < val)
				{
					lowestId = vId;
					val = zVal;
				}
			}

			zVector lowestV = positions[lowestId];

			zVector vec = vIt.getPosition() - lowestV;
			zVector unitz = zVector(0, 0, 1);

			double ang = vec.angle(unitz);

			// compute support

			if (positions[vIt.getId()].z > maxBB.z)
				(ang > (angle_threshold)) ? support[vIt.getId()] = true : support[vIt.getId()] = false;
	
		}
	}

}