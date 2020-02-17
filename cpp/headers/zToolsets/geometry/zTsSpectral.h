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

#ifndef ZSPACE_TS_GEOMETRY_SPECTRAL_H
#define ZSPACE_TS_GEOMETRY_SPECTRAL_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>

#include <depends/spectra/include/Spectra/SymEigsShiftSolver.h>
#include <depends/spectra/include/Spectra/MatOp/SparseSymShiftSolve.h>
using namespace Spectra;

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

	/*! \class zTsSpectral
	*	\brief A mesh spectral processing tool set class on triangular meshes.
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsSpectral
	{
	protected:

		/*! \brief core utilities object */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;

		/*!	\brief container of area per vertex of the mesh object  */
		vector<double> vertexArea;

		/*!	\brief container storing vertex types.  */
		vector< zSpectralVertexType> vertexType;

		/*!	\brief container storing ring neighbour indices per vertex.  */
		vector< vector<int>> vertexRing;

		/*!	\brief matrix to store mesh laplacian weights  */
		zSparseMatrix meshLaplacian;

		/*!	\brief container storing eigen function values.  */
		vector<double> eigenFunctionValues;

		/*!	\brief color type - zHSV/ zRGB  */
		zColorType colorType;

		/*!	\brief color domain.  */
		zDomainColor colorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

		/*!	\brief eigen values domain.  */
		zDomainFloat eigenDomain = zDomainFloat(0.0, 1.0);

		

		/*!	\brief number of eigen vectors required.  */
		int n_Eigens;
		
	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief mesh function set  */
		zFnMesh fnMesh;	

		/*!	\brief Eigen vectors matrix  */		
		MatrixXd eigenVectors;

		/*!	\brief Eigen values vector  */
		VectorXd eigenValues;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsSpectral();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsSpectral(zObjMesh &_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsSpectral();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createMeshfromFile(string path, zFileTpye type);

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------		
		
		/*! \brief This method computes the Eigen function operator.
		*
		*	\details Based on http://mgarland.org/files/papers/ssq.pdf
		*	\param		[in]	frequency		- input frequency value.		
		*	\param		[in]	computeEigenVectors	- cmputes eigen vectors if true.
		*	\since version 0.0.2
		*/
		double computeEigenFunction(double &frequency, bool &computeEigenVectors);

		/*! \brief This method computes the vertex type.
		*
		*	\param		[in]	frequency		- input frequency value.
		*	\param		[in]	useVertexArea	- uses vertex area for laplcaian weight calculation if true.
		*	\since version 0.0.2
		*/
		void computeVertexType();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the number of eigens to be requested. The value is capped to number of vertices in the mesh , if the input value is higher.
		*
		*	\param		[in]	_numEigen		- input number of eigens. 
		*	\since version 0.0.2
		*/
		void setNumEigens(int _numEigen);

		/*! \brief This method sets the color domain.
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\param		[in]	colType			- input color type.
		*	\since version 0.0.2
		*/
		void setColorDomain(zDomainColor &colDomain, zColorType colType);

		/*! \brief This method sets vertex color of all the vertices based on the eigen function.
		*
		*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColorFromEigen(bool setFaceColor = false);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the number of eigens.
		*
		*	\return			int		-  number of eigens.
		*	\since version 0.0.2
		*/
		int numEigens();

		/*! \brief This method gets the current function values.
		*
		*	\param		[out]	_eigenFunctionValues	- contatiner of eigen function values per vertex.
		*	\since version 0.0.2
		*/
		void getEigenFunctionValues(vector<double> &_eigenFunctionValues);

		/*! \brief This method gets the pointer to the internal function value container.
		*
		*	\return		double*		- pointer to internal contatiner of eigen function values per vertex.
		*	\since version 0.0.2
		*/
		double* getRawEigenFunctionValues();

	protected:		
		//--------------------------
		//---- PROTECTED UTILITY METHODS
		//--------------------------
			   
		/*! \brief This method computes the min and max of the eigen function at the input column index.
		*
		*	\param		[in]	colIndex				- input column index.
		*	\param		[out]	EigenDomain				- output eigen value domain.
		*	\since version 0.0.2
		*/
		void computeMinMax_Eigen(int &colIndex, zDomainFloat &inDomain);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsSpectral.cpp>
#endif

#endif