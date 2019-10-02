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


#include<headers/zToolsets/data/zTsKMeans.h>

namespace zSpace
{

	
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsKMeans::zTsKMeans()
	{
		numClusters = 2;
		numIterations = 100;
	}

	ZSPACE_INLINE zTsKMeans::zTsKMeans(zMatrixd &_dataPoints)
	{
		dataPoints = _dataPoints;


		numClusters = 2;
		numIterations = 100;
	}

	ZSPACE_INLINE zTsKMeans::zTsKMeans(zMatrixd &_dataPoints, int &_numClusters, int &_numIterations)
	{
		dataPoints = _dataPoints;

		numClusters = _numClusters;
		numIterations = _numIterations;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsKMeans::~zTsKMeans() {}

	//----  SET METHODS

	ZSPACE_INLINE void zTsKMeans::setNumClusters(int &_numClusters)
	{
		numClusters = _numClusters;
	}

	ZSPACE_INLINE void zTsKMeans::setNumIterations(int &_numIterations)
	{
		numIterations = _numIterations;
	}

	//---- CLUSTERING METHODS

	ZSPACE_INLINE int zTsKMeans::getKMeansClusters(int &actualNumClusters)
	{
		int numRows = dataPoints.getNumRows();
		int numCols = dataPoints.getNumCols();

		// get min max value of the datapoints
		int minIndex;
		double minVal = coreUtils.zMin(dataPoints, minIndex);

		int maxIndex;
		double maxVal = coreUtils.zMax(dataPoints, maxIndex);

		// Initialise means
		zMatrixd tempMeans = intialiseMeans(minVal, maxVal);

		// Initialise container to store cluster index per data point
		vector<int> clusterIDS;
		for (int i = 0; i < numRows; i++)
		{
			clusterIDS.push_back(0);
		}


		// Initialise container to store item indicies per cluster
		vector<vector<int>> tempClusters;
		for (int i = 0; i < numClusters; i++)
		{
			vector<int> temp;
			tempClusters.push_back(temp);
		}


		// compute means

		int numIters = 0;
		bool exit = false;
		for (int i = 0; i < numIterations; i++)
		{
			numIters = i;

			exit = true;

			for (int j = 0; j < numRows; j++)
			{
				zMatrix<double> data = dataPoints.getRowMatrix(j);


				int clusterID = getClusterIndex(data, tempMeans);


				// check if data point changed cluster
				if (clusterID != clusterIDS[j]) exit = false;

				clusterIDS[j] = clusterID;



				// update clusters
				tempClusters[clusterID].push_back(j);


			}

			// update mean			
			for (int j = 0; j < numClusters; j++)
			{
				zMatrix<double> mean = tempMeans.getRowMatrix(j);

				for (int l = 0; l < tempClusters[j].size(); l++)
				{
					int dataId = tempClusters[j][l];
					zMatrix<double> data = dataPoints.getRowMatrix(dataId);

					updateMean(data, mean, l + 1);
				}

				tempMeans.setRow(j, mean);
			}



			if (exit) break;
			else
			{
				// clear clusters
				for (int j = 0; j < numClusters; j++)
				{
					tempClusters[j].clear();
				}
			}


		}


		// remove cluster with zero elements
		actualNumClusters = numClusters;
		clusters.clear();

		for (int i = 0; i < numClusters; i++)
		{
			if (tempClusters[i].size() != 0)
			{
				clusters.push_back(tempClusters[i]);
			}
			else actualNumClusters--;

		}

		means = zMatrixd(actualNumClusters, tempMeans.getNumCols());
		int id = 0;
		for (int i = 0; i < tempMeans.getNumRows(); i++)
		{
			if (tempClusters[i].size() != 0)
			{
				for (int j = 0; j < tempMeans.getNumCols(); j++)
				{
					means(id, j) = tempMeans(i, j);
				}

				id++;
			}
		}

		return numIters;

	}

	//---- PROTECTED METHODS

	ZSPACE_INLINE zMatrix<double> zTsKMeans::intialiseMeans(double &minVal, double &maxVal)
	{
		zMatrix<double> out(numClusters, dataPoints.getNumCols());

		// to generate different random number every time the program runs
		srand(time(NULL));

		vector<double> randNumbers;

		for (int i = 0; i < out.getNumRows() * out.getNumCols(); i++)
		{
			double v = coreUtils.randomNumber_double(minVal, maxVal);

			randNumbers.push_back(v);
		}

		int id = 0;

		for (int i = 0; i < out.getNumRows(); i++)
		{

			for (int j = 0; j < out.getNumCols(); j++)
			{
				out(i, j) = randNumbers[id];
				id++;
			}
		}

		return out;
	}

	ZSPACE_INLINE int zTsKMeans::getClusterIndex(zMatrixd &data, zMatrixd &means)
	{
		double minDist = 10000000;
		int out = -1;

		for (int i = 0; i < means.getNumRows(); i++)
		{
			zMatrix<double> mean = means.getRowMatrix(i);

			double dist = coreUtils.getEuclideanDistance(data, mean);

			if (dist < minDist)
			{
				minDist = dist;
				out = i;
			}
		}

		return out;
	}

	ZSPACE_INLINE void zTsKMeans::updateMean(zMatrixd &data, zMatrixd &mean, int clusterSize)
	{
		for (int i = 0; i < mean.getNumCols(); i++)
		{
			mean(0, i) = (mean(0, i) * (clusterSize - 1) + data(0, i)) / clusterSize;
		}
	}

}