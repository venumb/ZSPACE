#pragma once
// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Federico Borello <federico.borello@zaha-hadid.com>
//

#ifndef ZSPACE_TS_VIZ_SCAN
#define ZSPACE_TS_VIZ_SCAN

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

	/*! \class zTsVizScan
	*	\brief A tool set to do solar analysis.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CUDA zTsVizScan
	{
	private:



		/*!	\brief pointer to container of positions for ground	*/
		float* positions_ground;

		/*!	\brief number of ground vertices in the container	*/
		int numVertices_ground;

		/*!	\brief pointer to container of  positions for collider	*/
		float* positions_collider;

		/*!	\brief number of collider vertices in the container	*/
		int numVertices_collider;
		
		/*!	\brief pointer to container of  face traingles vertex indices for collider	*/
		int* vertexList_collider;

		/*!	\brief number of collider positions in the container	*/
		int numPolygons_collider;

		/*!	\brief color domain	*/
		zDomainColor dColor;

		/*!	\brief pointer to container of colors*/
		float* colors;

		/*!	\brief pointer to container of normals + sunvectors*/
		float* norm_sunvecs;

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
		ZSPACE_CUDA_CALLABLE_HOST zTsVizScan();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE_HOST ~zTsVizScan();

		//--------------------------
		//---- SET METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE_HOST void setNormals(const float* _normals, int _numNormals, bool EPWread);

		ZSPACE_CUDA_CALLABLE_HOST void setDomain_Colors(zDomainColor& _dColor);

		ZSPACE_CUDA_CALLABLE_HOST void setObserver(zPoint& _position , zVector &_direction);

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