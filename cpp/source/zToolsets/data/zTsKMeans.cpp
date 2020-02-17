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

	ZSPACE_INLINE zTsKMeans::zTsKMeans(MatrixXf &_dataPoints)
	{
		dataPoints = _dataPoints;


		numClusters = 2;
		numIterations = 100;
	}

	ZSPACE_INLINE zTsKMeans::zTsKMeans(MatrixXf &_dataPoints, int &_numClusters, int &_numIterations)
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
		int numRows = dataPoints.rows();
		int numCols = dataPoints.cols();

		// get min max value of the datapoints
		int minIndex;
		
		float minVal = dataPoints.minCoeff();

		int maxIndex;
		float maxVal = dataPoints.maxCoeff();

		// Initialise means
		MatrixXf tempMeans = intialiseMeans(minVal, maxVal);

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
				MatrixXf data = dataPoints.row(j);

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
				MatrixXf  mean = tempMeans.row(j);

				for (int l = 0; l < tempClusters[j].size(); l++)
				{
					int dataId = tempClusters[j][l];
					MatrixXf data = dataPoints.row(dataId);

					updateMean(data, mean, l + 1);
				}		

				tempMeans.row(j) = mean;
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

		means = MatrixXf(actualNumClusters, tempMeans.cols());
		int id = 0;
		for (int i = 0; i < tempMeans.rows(); i++)
		{
			if (tempClusters[i].size() != 0)
			{
				for (int j = 0; j < tempMeans.cols(); j++)
				{
					means(id, j) = tempMeans(i, j);
				}

				id++;
			}
		}

		return numIters;

	}

	//---- PROTECTED METHODS

	ZSPACE_INLINE MatrixXf zTsKMeans::intialiseMeans(float &minVal, float &maxVal)
	{
		MatrixXf out(numClusters, dataPoints.cols());

		// to generate different random number every time the program runs
		srand(time(NULL));

		vector<double> randNumbers;

		for (int i = 0; i < out.rows() * out.cols(); i++)
		{
			double v = coreUtils.randomNumber_double(minVal, maxVal);

			randNumbers.push_back(v);
		}

		int id = 0;

		for (int i = 0; i < out.rows(); i++)
		{

			for (int j = 0; j < out.cols(); j++)
			{
				out(i, j) = randNumbers[id];
				id++;
			}
		}

		return out;
	}

	ZSPACE_INLINE int zTsKMeans::getClusterIndex(MatrixXf &data, MatrixXf &means)
	{
		double minDist = 10000000;
		int out = -1;

		for (int i = 0; i < means.rows(); i++)
		{
			MatrixXf mean = means.row(i);

			double dist = coreUtils.getEuclideanDistance(data, mean);

			if (dist < minDist)
			{
				minDist = dist;
				out = i;
			}
		}

		return out;
	}

	ZSPACE_INLINE void zTsKMeans::updateMean(MatrixXf &data, MatrixXf &mean, int clusterSize)
	{
		for (int i = 0; i < mean.cols(); i++)
		{
			mean(0, i) = (mean(0, i) * (clusterSize - 1) + data(0, i)) / clusterSize;
		}
	}

}