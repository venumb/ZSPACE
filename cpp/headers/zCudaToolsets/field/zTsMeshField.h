// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>,
//

#ifndef ZSPACE_TS_MESHFIELD
#define ZSPACE_TS_MESHFIELD

#pragma once

#define MAX_SUNVECS_HOUR 366*24 * 3
#define MAX_SUNVECS_DAY 24*7 * 3
#define INVALID_VAL -10000.0

#define COMPASS_SUBD 12*2*3

#define RAD_ANGLE_MIN 30.0
#define RAD_ANGLE_MAX 120.0

#define RAD_MIN 300.0
#define RAD_MAX 1000.0


#include<headers/zCudaToolsets/base/zCudaDefinitions.h>
#include<headers/zCore/base/zInline.h>
#include<headers/zCore/base/zDomain.h>
#include<headers/zCore/utilities/zUtilsCore.h>

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

	/*! \class zTsMeshField
	*	\brief A tool set for mesh fields.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CUDA zTsMeshField
	{
	private:

		/*!	\brief pointer to container of positions	*/
		float* positions;

		/*!	\brief number of points in the container	*/
		int numVertices;

		/*!	\brief pointer to container of poly connects. Quads only	*/
		int* polyConnects;
		
		/*!	\brief number of points in the container	*/
		int numPolygons;

		/*!	\brief pointer to container of isovalues. */
		float* isoValues;

		/*!	\brief number of points in the container	*/
		int numEdges;

		/*!	\brief pointer to container of poly connects. Quads only	*/
		int* edgeConnects;

		/*!	\brief pointer to container of edge colors*/
		float* edgecolors;

		/*!	\brief pointer to container of contour positions, every pair of 2 positions make an graph edge*/
		float* contourPositions;

		/*!	\brief size of container for normals, cummulative raditation, colors*/
		int memSize;


	public:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE_HOST zTsMeshField();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE_HOST ~zTsMeshField();

		//--------------------------
		//---- SET METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE_HOST void setNormals(const float* _normals, int _numNormals, bool EPWread);

		ZSPACE_CUDA_CALLABLE_HOST bool setEPWData(string path);

		ZSPACE_CUDA_CALLABLE_HOST void setDomain_Dates(zDomainDate& _dDate);

		ZSPACE_CUDA_CALLABLE_HOST void setDomain_Colors(zDomainColor& _dColor);

		ZSPACE_CUDA_CALLABLE_HOST void setLocation(zLocation& _location);

		//--------------------------
		//---- GET METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE int numNormals();

		ZSPACE_CUDA_CALLABLE int numSunVecs();

		ZSPACE_CUDA_CALLABLE int numDataPoints();

		ZSPACE_CUDA_CALLABLE float* getRawNormals();

		ZSPACE_CUDA_CALLABLE float* getRawColors();

		ZSPACE_CUDA_CALLABLE float* getRawNormals_SunVectors();

		ZSPACE_CUDA_CALLABLE float* getRawSunVectors_hour();

		ZSPACE_CUDA_CALLABLE float* getRawSunVectors_day();

		ZSPACE_CUDA_CALLABLE float* getRawCompassPts();

		ZSPACE_CUDA_CALLABLE float* getRawEPWRadiation();

		ZSPACE_CUDA_CALLABLE float* getRawCummulativeRadiation();

		ZSPACE_CUDA_CALLABLE zVector getSunPosition(zDate& date);

		ZSPACE_CUDA_CALLABLE zDomainDate getSunRise_SunSet(zDate& date);

		ZSPACE_CUDA_CALLABLE zDomainDate getDomain_Dates();

		ZSPACE_CUDA_CALLABLE zDomainColor getDomain_Colors();

		ZSPACE_CUDA_CALLABLE zLocation getLocation();

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE void computeSunVectors_Year();

		ZSPACE_CUDA_CALLABLE void computeCompass();

		ZSPACE_CUDA_CALLABLE void computeCummulativeRadiation();


		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------	

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------	
	protected:

		ZSPACE_CUDA_CALLABLE_HOST void setMemory();

		ZSPACE_CUDA_CALLABLE_HOST void computeSunVectors_Hour();

		ZSPACE_CUDA_CALLABLE_HOST void computeSunVectors_Day();

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCudaToolsets/energy/zTsSolarAnalysis.cpp>
#endif

#endif