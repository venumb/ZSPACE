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


#include<headers/zToolsets/geometry/zTsSpectral.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSpectral::zTsSpectral() {}

	ZSPACE_INLINE zTsSpectral::zTsSpectral(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);

		n_Eigens = fnMesh.numVertices() - 1;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSpectral::~zTsSpectral() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsSpectral::createMeshfromFile(string path, zFileTpye type)
	{
		fnMesh.from(path, type, false);

		n_Eigens = fnMesh.numVertices() - 1;
	}

	//---- COMPUTE METHODS

	ZSPACE_INLINE double zTsSpectral::computeEigenFunction(double &frequency, bool &computeEigenVectors)
	{
		int n_v = fnMesh.numVertices();

		if (meshLaplacian.cols() != n_v)
		{
			std::clock_t start;
			start = std::clock();

			meshLaplacian = fnMesh.getTopologicalLaplacian();

			printf("\n meshLaplacian: r %i  c %i ", meshLaplacian.rows(), meshLaplacian.cols());

			double t_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			std::cout << "\n mesh Laplacian compute: " << t_duration << " seconds";
		}

		if (computeEigenVectors)
		{
			std::clock_t start;
			start = std::clock();

			//using spectra
			SparseSymShiftSolve<double> op(meshLaplacian);
			SymEigsShiftSolver< double, LARGEST_MAGN, SparseSymShiftSolve<double> > eigs(&op, n_Eigens, n_Eigens + 1, 0.0);

			// Initialize and compute
			eigs.init();
			int nconv = eigs.compute();

			double t_duration2 = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			std::cout << "\n Eigen solve : " << t_duration2 << " seconds";

			// Retrieve results
			if (eigs.info() == SUCCESSFUL)
			{
				eigenVectors = eigs.eigenvectors(nconv);
				eigenValues = eigs.eigenvalues();
			}
			else
			{
				cout << "\n Eigen convergence unsuccessful ";
				return -1.0;
			}

			printf("\n Eigen num converge %i : eigenVectors r %i  c %i ", nconv, eigenVectors.rows(), eigenVectors.cols());

			computeEigenVectors = !computeEigenVectors;
		}


		if (eigenVectors.rows() != n_v) return -1.0;

		int val = (int)frequency;
		if (val >= n_Eigens) val = (int)frequency % n_Eigens;

		// compute derivatives
		zDomainFloat inDomain;
		computeMinMax_Eigen(val, inDomain);


		// compute eigen function operator
		if (eigenFunctionValues.size() == 0 || eigenFunctionValues.size() != n_v)
		{
			eigenFunctionValues.clear();
			eigenFunctionValues.assign(n_v, 0);
		}

		for (int i = 0; i < n_v; i++)
		{
			float matVal = eigenVectors.col(val).row(i).value();

			float EigenFunctionRegular = matVal;

			eigenFunctionValues[i] = coreUtils.ofMap(matVal, inDomain, eigenDomain);
		}

		setVertexColorFromEigen(true);

		return eigenValues[val];
	}

	ZSPACE_INLINE void zTsSpectral::computeVertexType()
	{
		int n_v = fnMesh.numVertices();

		// compute vertex neighbours if not computed already
		if (vertexRing.size() == 0 || vertexRing.size() != n_v)
		{
			vertexRing.clear();

			for (zItMeshVertex v(*meshObj); !v.end(); v++)
			{
				vector<int> cVertices;
				v.getConnectedVertices(cVertices);

				vertexRing.push_back(cVertices);
			}
		}

		// set size of vertex type to number of vertices
		if (vertexType.size() == 0 || vertexType.size() != n_v)
		{
			vertexType.clear();
			vertexType.assign(n_v, zSpectralVertexType::zRegular);
		}

		// compute vertex type
		for (int i = 0; i < n_v; i++)
		{
			int n_rv = vertexRing[i].size();

			int minCounter = 0;
			int maxCounter = 0;

			vector<int> minId;

			for (int j = 0; j < n_rv; j++)
			{
				int otherId = vertexRing[i][j];

				if (eigenFunctionValues[i] > eigenFunctionValues[otherId]) maxCounter++;
				else
				{
					minId.push_back(j);
					minCounter++;
				}
			}

			vertexType[i] = zSpectralVertexType::zRegular;
			if (maxCounter == n_rv) vertexType[i] = zSpectralVertexType::zMaxima;;
			if (minCounter == n_rv) vertexType[i] = zSpectralVertexType::zMinima;;

			// Check for  Saddle
			bool chkChain = true;

			if (maxCounter != n_rv && minCounter != n_rv)
			{
				int disContuinityCounter = 0;

				for (int j = 0; j < minCounter; j++)
				{

					int k = (j + 1) % minCounter;

					int chk = (minId[j] + 1) % n_rv;

					if (chk != minId[k]) disContuinityCounter++;

				}

				if (disContuinityCounter > 1)
				{
					chkChain = false;
				}

			}

			if (!chkChain) vertexType[i] = zSpectralVertexType::zSaddle;;;
		}
	}

	//---- SET METHODS

	ZSPACE_INLINE void zTsSpectral::setNumEigens(int _numEigen)
	{
		n_Eigens = _numEigen;

		if (_numEigen >= fnMesh.numVertices()) n_Eigens = fnMesh.numVertices() - 1;
	}

	ZSPACE_INLINE void zTsSpectral::setColorDomain(zDomainColor &colDomain, zColorType colType)
	{
		colorDomain = colDomain;
		colorType = colType;
	}

	ZSPACE_INLINE void zTsSpectral::setVertexColorFromEigen(bool setFaceColor)
	{
		zColor* cols = fnMesh.getRawVertexColors();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			cols[i] = coreUtils.blendColor(eigenFunctionValues[i], eigenDomain, colorDomain, colorType);
		}

		if (setFaceColor) fnMesh.computeFaceColorfromVertexColor();
	}

	//---- GET METHODS

	ZSPACE_INLINE int zTsSpectral::numEigens()
	{
		return n_Eigens;
	}

	ZSPACE_INLINE void zTsSpectral::getEigenFunctionValues(vector<double> &_eigenFunctionValues)
	{
		_eigenFunctionValues = eigenFunctionValues;
	}

	ZSPACE_INLINE double*zTsSpectral::getRawEigenFunctionValues()
	{
		return &eigenFunctionValues[0];
	}

	//---- PROTECTED UTILITY METHODS

	ZSPACE_INLINE void zTsSpectral::computeMinMax_Eigen(int &colIndex, zDomainFloat &inDomain)
	{
		inDomain.min = 10000;
		inDomain.max = -10000;


		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			double matVal = eigenVectors.col(colIndex).row(i).value();

			if (matVal < inDomain.min) inDomain.min = matVal;
			if (matVal > inDomain.max) inDomain.max = matVal;
		}

	}

}