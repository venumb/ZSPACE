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

#ifndef ZSPACE_TS_DATA_KMEANS_H
#define ZSPACE_TS_DATA_KMEANS_H

#pragma once


#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zCore/base/zMatrix.h>

namespace zSpace
{
	
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsData
	*	\brief toolsets for data related utilities.
	*  @{
	*/

	/** \addtogroup zClustering
	*	\brief tool sets of clustering algorithms.
	*  @{
	*/

	/*! \class zTsKMeans
	*	\brief A tool set for doing K-Means clustering.
	*	\details Based on https://www.geeksforgeeks.org/k-means-clustering-introduction/
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	

	class ZSPACE_TOOLS zTsKMeans
	{
		

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
		MatrixXf means;

		/*!	\brief matrix data*/
		MatrixXf dataPoints;
		
		/*!	\brief 2 dimensional container of cluster items*/
		vector<vector<int>> clusters;
	
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsKMeans();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_data			- input matrix data.
		*	\since version 0.0.2
		*/
		zTsKMeans(MatrixXf &_dataPoints);
		
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_dataPoints				- input matrix data.
		*	\param		[in]	_numClusters		- input snumber of clusters.
		*	\param		[in]	_numIterations		- input number of iterations.
		*	\since version 0.0.2
		*/
		zTsKMeans(MatrixXf &_dataPoints, int &_numClusters, int &_numIterations);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsKMeans();
		
		//--------------------------
		//----  SET METHODS
		//--------------------------

		/*! \brief This method sets the number of clusters.
		*
		*	\param		[in]	_numClusters		- input snumber of clusters.
		*	\since version 0.0.2
		*/
		void setNumClusters(int &_numClusters);

		/*! \brief This method sets the number of iterations.
		*
		*	\param		[in]	_numIterations		- input number of iterations.
		*	\since version 0.0.2
		*/
		void setNumIterations(int &_numIterations);

		//--------------------------
		//---- CLUSTERING METHODS
		//--------------------------
			   
		/*! \brief This method computes classify the input data into input number of clusters using the K-Means Algorithm.
		*
		*	\param	[out]	actualNumClusters		- actual number of clusters after removing clusters of size 0.
		*	\return			int						- number of interations the algorithm ran.
		*/
		int getKMeansClusters(int &actualNumClusters);

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
		MatrixXf intialiseMeans(float &minVal, float &maxVal);

		/*! \brief This method computes the cluster index based on the least euclidean distance between input data point and mean values.
		*
		*	\param	[in]	data			- input row matrix of data.
		*	\return			int				- index of cluster.
		*/
		int getClusterIndex(MatrixXf &data, MatrixXf &means);

		/*! \brief This method updates the mean value of the cluster based on the input data point and cluster size.
		*
		*	\param	[in]	data			- input row matrix of datapoints.
		*	\param	[in]	mean			- input row matrix of means.
		*	\param	[in]	clusterSize		- current cluster size.
		*/
		void updateMean(MatrixXf &data, MatrixXf &mean, int clusterSize);
			
		
	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/data/zTsKMeans.cpp>
#endif

#endif