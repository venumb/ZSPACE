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

	ZSPACE_INLINE zTsMesh2Pix::zTsMesh2Pix(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsMesh2Pix::~zTsMesh2Pix() {}

	ZSPACE_INLINE void zTsMesh2Pix::toBMP(zConnectivityType connectivityType, string path)
	{
		if (connectivityType == zVertexVertex)
		{
			// get conectivity matrix
			int n_v = fnMesh.numVertices();
			MatrixXd vertToVertMat(n_v, n_v);

			zDoubleArray eLength;
			fnMesh.getEdgeLengths(eLength);
			double eLengthMin = coreUtils.zMin(eLength);
			double eLengthMax = coreUtils.zMax(eLength);

			for (zItMeshVertex v(*meshObj); !v.end(); v++)
			{
				vector<zItMeshHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				int j = v.getId();

				for (int k = 0; k < cEdges.size(); k++)
				{
					int i = cEdges[k].getVertex().getId();
					vertToVertMat(j, i) = coreUtils.ofMap<double>(cEdges[k].getLength(), eLengthMin, eLengthMax, 0.5, 1.0);
				}
			}
			
			// get vertex normals diagonal matrix
			zVectorArray norms;
			fnMesh.getVertexNormals(norms);
			zUtilsCore core;

			MatrixXd normsX(n_v, n_v);
			MatrixXd normsY(n_v, n_v);
			MatrixXd normsZ(n_v, n_v);

			for (int i = 0; i < norms.size(); i++)
			{
				normsX(i, i) = core.ofMap(norms[i].x, -1.0, 1.0, 0.0, 1.0);
				normsY(i, i) = core.ofMap(norms[i].y, -1.0, 1.0, 0.0, 1.0);
				normsZ(i, i) = core.ofMap(norms[i].z, -1.0, 1.0, 0.0, 1.0);
			}

			vector<MatrixXd> colOfMat;
			colOfMat.push_back(normsX);
			colOfMat.push_back(normsY);
			colOfMat.push_back(normsZ);
			colOfMat.push_back(vertToVertMat);

			coreUtils.matrixBMP(colOfMat, path);
		}

		else if (connectivityType == zVertexEdge)
		{
			throw std::invalid_argument(" error: zVertexEdge connectivity is not implemented yet");
		}

		else if (connectivityType == zFaceVertex)
		{
			throw std::invalid_argument(" error: zFaceVertex connectivity is not implemented yet");
		}

		else if (connectivityType == zFaceEdge)
		{
			throw std::invalid_argument(" error: zFaceEdge connectivity is not implemented yet");
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType type");
	}

	ZSPACE_INLINE void zTsMesh2Pix::toVertexDataBMP(vector<int> vertexData, string path)
	{
		if (vertexData.size() == fnMesh.numVertices())
		{
			// create sparseMatrix from support vector
			MatrixXd supportMatR(fnMesh.numVertices(), fnMesh.numVertices());
			MatrixXd supportMatG(fnMesh.numVertices(), fnMesh.numVertices());
			MatrixXd supportMatB(fnMesh.numVertices(), fnMesh.numVertices());
			supportMatB.setOnes();

			for (int i = 0; i < fnMesh.numVertices(); i++)		
				for (int j = 0; j < fnMesh.numVertices(); j++)			
					if (i == j)
					{
						if (vertexData[i] == 1)
							supportMatG(i, j) = 1;
						if (vertexData[i] == 2)
							supportMatR(i, j) = 1;
						supportMatB(i, j) = 0;
					}

			vector<MatrixXd> colOfMat;
			colOfMat.push_back(supportMatR);
			colOfMat.push_back(supportMatG);
			colOfMat.push_back(supportMatB);

			coreUtils.matrixBMP(colOfMat, path);
		}
		else
			throw std::invalid_argument( "error: invalid size of input matrix.");

		cout << "\nmeshToPix: success!";
	}

	ZSPACE_INLINE void zTsMesh2Pix::checkVertexSupport(zObjMesh &_objMesh, double angle_threshold, vector<int> &support)
	{
		support.assign(_objMesh.mesh.n_v, -1);

		zFnMesh fnMesh(_objMesh);

		zVector* positions = fnMesh.getRawVertexPositions();

		for (zItMeshVertex vIt(_objMesh); !vIt.end(); vIt++)
		{
			zIntArray cVerts;
			vIt.getConnectedVertices(cVerts);

			int lowestId;
			double val = 10e10;
			for (int i = 0; i < cVerts.size(); i++)
			{
				double zVal = positions[cVerts[i]].z;

				if (zVal < val)
				{
					lowestId = cVerts[i];
					val = zVal;
				}
			}

			zVector lowestV = positions[lowestId];

			zVector vec = vIt.getPosition() - lowestV;
			zVector unitz = zVector(0, 0, 1);

			double ang = vec.angle(unitz);

			if (vIt.getPosition().z > 0)
				(ang > (angle_threshold)) ? support[vIt.getId()] = 2 : support[vIt.getId()] = 1;
			else
				support[vIt.getId()] = 1;
		}
	}

}