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

#ifndef ZSPACE_TS_GEOMETRY_MESH_IMAGE_H
#define ZSPACE_TS_GEOMETRY_MESH_IMAGE_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>

#include<headers/zCore/utilities/zUtilsBMP.h>
#include<headers/zToolsets/statics/zTsVault.h>

namespace zSpace
{

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsMesh2Pix
	*	\brief A function set to convert mesh data to images for machine learning using pix2pix.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsMesh2Pix
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;
			
		/*!	\brief pointer to form Object  */
		zObjMesh *predictedObj;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnMesh fnMesh;

		zFnMesh fnPredictedMesh;

		/*!	\brief maximum number of vertices for the data set  */
		int maxVertices;

		/*!	\brief maximum number of edges for the data set  */
		int maxEdges;

		/*!	\brief maximum number of faces for the data set  */
		int maxFaces;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zTsMesh2Pix();


		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_maxVerts			- input maximum number of vertices. If -1 , then no constraint.
		*	\since version 0.0.4
		*/
		zTsMesh2Pix(zObjMesh &_meshObj, int _maxVerts = -1 ,int _maxEdges = -1, int _maxFaces = -1);

		zTsMesh2Pix(zObjMesh &_meshObj, zObjMesh &_predictedObj, int _maxVerts = -1, int _maxEdges = -1, int _maxFaces = -1);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zTsMesh2Pix();


		//--------------------------
		//---- GENERATE DATA METHODS
		//--------------------------

		/*! \brief This method writes an image to the defined path representing the mesh connecticity.
		*
		*	\param		[in]	directory			- directory to write the image file to.
		*	\param		[in]	perturbPositions	- generates data by perturbing verrtice if true.
		*	\since version 0.0.4
		*/
		void generatePrintSupport2Pix(string directory, string filename, double angle_threshold, bool train = true, int numIters = 0, bool perturbPositions = false, zVector perturbVal = zVector(1,1,1));

		bool generateFDM2Pix(string directory, string filename, zIntArray &fixedConstrained, zFloatArray &forceDensities, zDomainFloat &densityDomain, bool train = true, int numIters = 0, bool perturbPositions = false, zDomainFloat maxDensityDomain = zDomainFloat(0.1, 1.0));

		//--------------------------
		//---- PREDICT DATA METHODS
		//--------------------------

		void predictPrintSupport2Pix(string directory, string filename, bool genPix);

		void predictFDM2Pix(string directory, string filename, zIntArray &fixedConstrained, zFloatArray &forceDensities, zDomainFloat &densityDomain, bool genPix);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		void scaleToBounds(double maxSide);

	private:		

		//--------------------------
		//---- PRIVATE GET METHODS
		//--------------------------

		/*! \brief This method gets a container of matrices from the normals of a mesh.
		*
		*	\param		[in]	type			- input  zConnectivityType
		*	\param		[out]	normMat			- output container of matrices.
		*	\since version 0.0.4
		*/
		void getMatrixFromNormals(zConnectivityType type, zDomainFloat &outDomain, vector<MatrixXf> &normMat);

		/*! \brief This method gets a container of matrices from the positions of a mesh.
		*
		*	\param		[in]	type			- input  zConnectivityType, 
		*	\param		[out]	normMat			- output container of matrices.
		*	\since version 0.0.4
		*/
		void getMatrixFromPositions(zConnectivityType type, zDomainFloat &outDomain, vector<MatrixXf> &posMat);

		/*! \brief This method gets a container of matrices from the normals of a mesh.
		*
		*	\param		[in]	type			- input  zConnectivityType, Works with zVertexData and zFaceData.
		*	\param		[out]	normMat			- output container of matrices.
		*	\since version 0.0.4
		*/
		void getMatrixFromContainer(zConnectivityType type, zVectorArray &data, zDomainFloat &outDomain, vector<MatrixXf> &normMat);


		/*! \brief This method gets a container of matrices from the normals of a mesh.
		*
		*	\param		[in]	type			- input  zConnectivityType, Works with zVertexData and zFaceData.
		*	\param		[out]	normMat			- output container of matrices.
		*	\since version 0.0.4
		*/
		void getMatrixFromContainer(zConnectivityType type, zBoolArray &data, zDomainFloat &outDomain, vector<MatrixXf> &outMat);

		/*! \brief This method gets a matrix from the input container and matix type.
		*
		*	\param		[in]	type			- input  zConnectivityType.
		*	\param		[in]	data			- input  data.
		*	\param		[in]	dataPair		- input data index pairs.
		*	\param		[in]	outDomain		- data output mapping domain.
		*	\param		[out]	outMat			- output matrix.
		*	\since version 0.0.4
		*/
		void getMatrixFromContainer(zConnectivityType type, int numVerts, zFloatArray &data, zIntPairArray &dataPair, zDomainFloat &outDomain, vector<MatrixXf> &outMat);

		void getVertexSupport(double angle_threshold, zBoolArray &support);

		void getCombinedMatrix(vector<MatrixXf> &mat1, vector<MatrixXf> &mat2, vector<MatrixXf> &out);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsMesh2Pix.cpp>
#endif

#endif