#pragma once


#include <headers/api/functionsets/zFnMesh.h>

namespace zSpace
{
	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/*! \class zTsKMeans
	*	\brief A tool set for doing K-Means clustering.
	*	\details Based on https://www.geeksforgeeks.org/k-means-clustering-introduction/
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	class zTsKMeans
	{
	private:
	

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
		/*! \brief core utilities object */
		zUtilsCore coreUtils;

		/*!<\brief number of clusters.*/
		int numClusters;

		/*!<\brief number of maximum iterations.*/
		int numIterations;

		/*!<\brief minimum length of stream.*/
		double *minLength;

		/*!<\brief maximum length of stream.*/
		double *maxLength;

		/*!<\brief mean or average data.*/
		zMatrixd means;

		/*!	\brief matrix data*/
		zMatrixd dataPoints;
		
		/*!	\brief 2 dimensional container of cluster items*/
		vector<vector<int>> clusters;
		
	
		

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsKMeans()
		{
			numClusters = 2;
			numIterations = 100;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_data			- input matrix data.
		*	\since version 0.0.2
		*/
		zTsKMeans(zMatrixd &_dataPoints)
		{
			dataPoints = _dataPoints;
		

			numClusters = 2;
			numIterations = 100;
		}
		
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_dataPoints				- input matrix data.
		*	\param		[in]	_numClusters		- input snumber of clusters.
		*	\param		[in]	_numIterations		- input number of iterations.
		*	\since version 0.0.2
		*/
		zTsKMeans(zMatrixd &_dataPoints, int &_numClusters, int &_numIterations)
		{
			dataPoints = _dataPoints;

			numClusters = _numClusters;
			numIterations = _numIterations;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsKMeans() {}
		
		//--------------------------
		//----  SET METHODS
		//--------------------------

		/*! \brief This method sets the number of clusters.
		*
		*	\param		[in]	_numClusters		- input snumber of clusters.
		*	\since version 0.0.2
		*/
		void setNumClusters(int &_numClusters)
		{
			numClusters = _numClusters;
		}

		/*! \brief This method sets the number of iterations.
		*
		*	\param		[in]	_numIterations		- input number of iterations.
		*	\since version 0.0.2
		*/
		void setNumIterations(int &_numIterations)
		{
			numIterations = _numIterations;
		}

		//--------------------------
		//---- CLUSTERING METHODS
		//--------------------------
			   
		/*! \brief This method computes classify the input data into input number of clusters using the K-Means Algorithm.
		*
		*	\param	[out]	actualNumClusters		- actual number of clusters after removing clusters of size 0.
		*	\return			int						- number of interations the algorithm ran.
		*/
		inline int getKMeansClusters(int &actualNumClusters)
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

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------
	protected:
			
		/*! \brief This method initialises the means based on the minimum and maximum value in the data points.
		*
		*	\param	[out]	minVal			- input minimum value in the data.
		*	\param	[out]	maxVal			- input maximum value in the data.
		*	\return			int				- index of cluster.
		*/
		inline zMatrix<double> intialiseMeans(double &minVal, double &maxVal)
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

		/*! \brief This method computes the cluster index based on the least euclidean distance between input data point and mean values.
		*
		*	\param	[in]	data			- input row matrix of data.
		*	\return			int				- index of cluster.
		*/
		inline int getClusterIndex(zMatrixd &data, zMatrixd &means)
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

		/*! \brief This method updates the mean value of the cluster based on the input data point and cluster size.
		*
		*	\param	[in]	data			- input row matrix of datapoints.
		*	\param	[in]	mean			- input row matrix of means.
		*	\param	[in]	clusterSize		- current cluster size.
		*/
		inline void updateMean(zMatrixd &data, zMatrixd &mean, int clusterSize)
		{
			for (int i = 0; i < mean.getNumCols(); i++)
			{
				mean(0, i) = (mean(0, i) * (clusterSize - 1) + data(0, i)) / clusterSize;
			}
		}
			
		
	};


}