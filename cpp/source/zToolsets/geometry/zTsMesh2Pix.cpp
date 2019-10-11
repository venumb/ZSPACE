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

	ZSPACE_INLINE void zTsMesh2Pix::toBMP(string path, zConnectivityType connectivityType)
	{
		if (connectivityType == zVertexVertex)
		{
			// get conectivity matrix
			int n_v = fnMesh.numVertices();
			zSparseMatrix vertToVert(n_v, n_v);

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
					vertToVert.insert(j, i) = coreUtils.ofMap<double>(cEdges[k].getLength(), eLengthMin, eLengthMax, 0.5, 1.0);
				}
			}
			

			// get vertex normals diagonal matrix
			zVectorArray vNormals;
			fnMesh.getVertexNormals(vNormals);
					
			vector<double> vNormalXs;
			for (int i = 0; i < vNormals.size(); i++) 
				vNormalXs.push_back(coreUtils.ofMap<double>(vNormals[i].x, -1, 1, 0, 1));
			zMatrix<double> vNormalXMat(n_v, n_v);
			vNormalXMat.setDiagonal(vNormalXs);

			vector<double> vNormalYs;
			for (int i = 0; i < vNormals.size(); i++) 
				vNormalYs.push_back(coreUtils.ofMap<double>(vNormals[i].y, -1, 1, 0, 1));
			zMatrix<double> vNormalYMat(n_v, n_v);
			vNormalYMat.setDiagonal(vNormalYs);

			vector<double> vNormalZs;
			for (int i = 0; i < vNormals.size(); i++) 
				vNormalZs.push_back(coreUtils.ofMap<double>(vNormals[i].z, -1, 1, 0, 1));
			zMatrix<double> vNormalZMat(n_v, n_v);
			vNormalZMat.setDiagonal(vNormalZs);


			// write BMP
			string fileName = "meshImage_zVertexVertex.bmp";
			string outPath = path + "/" + fileName;

			int resX = n_v;
			int resY = n_v;

			zUtilsBMP bmp(resX, resY);
			uint32_t channels = bmp.bmp_info_header.bit_count / 8;
			
			for (uint32_t x = 0; x < resX; ++x)
			{
				for (uint32_t y = 0; y < resY; ++y)
				{
					// blue
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = vNormalXMat(x, y) * 255;

					// green
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = vNormalYMat(x, y) * 255;

					// red
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = vNormalZMat(x, y) * 255;

					// alpha
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = vertToVert.coeff(x, y) * 255;
				}
			}

			bmp.write(outPath.c_str());
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

	ZSPACE_INLINE void zTsMesh2Pix::toBMP(string path, vector<int> vertexData)
	{
		if (vertexData.size() == fnMesh.numVertices())
		{
			// get data matrix
			int n_v = fnMesh.numVertices();
			zSparseMatrix vertToVertData(n_v, n_v);

			for (int i = 0; i < n_v; i++)			
					vertToVertData.insert(i, i) = vertexData[i];

			//cout << vertToVertData;

			// write BMP
			string fileName = "meshImage_zVertexVertexData.bmp";
			string outPath = path + "/" + fileName;

			int resX = n_v;
			int resY = n_v;

			zUtilsBMP bmp(resX, resY);
			uint32_t channels = bmp.bmp_info_header.bit_count / 8;

			for (uint32_t x = 0; x < resX; ++x)
			{
				for (uint32_t y = 0; y < resY; ++y)
				{
					// blue
					if(vertToVertData.coeff(x, y) == 0)
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = 255;
					else
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = 0;

					// green
					if (vertToVertData.coeff(x, y) == 1)
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = 255;
					else
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = 0;

					// red
					if (vertToVertData.coeff(x, y) == 2)
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = 255;
					else
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = 0;

					// alpha
					bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = 0;
				}
			}

			bmp.write(outPath.c_str());
		}

		else throw std::invalid_argument(" error: invalid size of input vertexData vector");

	}

}