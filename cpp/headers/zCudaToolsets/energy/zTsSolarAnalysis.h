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

#ifndef ZSPACE_TS_SOLAR_ANALYSIS
#define ZSPACE_TS_SOLAR_ANALYSIS

#pragma once

#define MAX_SUNVECS_HOUR 366*24
#define MAX_SUNVECS_DAY 24*7
#define COMPASS_SUBD 12*2

#include<headers/zCudaToolsets/base/zCudaDefinitions.h>
#include<headers/zCore/base/zInline.h>
#include<headers/zCore/base/zDomain.h>
#include<headers/zCore/utilities/zUtilsCore.h>

namespace zSpace
{

	struct zNorm_SunVec
	{
		zVector norm;
		zVector sunVec;
	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsSolarAnalysis
	*	\brief A tool set to do solar analysis.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CUDA zTsSolarAnalysis
	{
	private:

		/*!	\brief pointer to container of normals	*/
		zVector *normals;

		/*!	\brief number of face normals in the container	*/
		int numNorms;

		/*!	\brief pointer to container of sun vectors	*/
		zVector *sunVecs_hour;

		/*!	\brief pointer to container of sun vectors	*/
		zVector *sunVecs_days;

		/*!	\brief pointer to container of compass display vectors	*/
		zVector *compassVecs;

		/*!	\brief pointer to container of epw data	*/
		zEPWData* epwData;		

		/*!	\brief number of epw data points in the container	*/
		int numData;			

		/*!	\brief date domain	*/
		zDomainDate dDate;	

		/*!	\brief location info	*/
		zLocation location;

		/*!	\brief pointer to container of cummulative raditation*/
		float *cummulativeRadiation;

		/*!	\brief size of container for normals and cummulative raditation*/
		int memSize;

			   			   
	public:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to container of normals and sun vectors. It should be same size as solar angles*/
		zNorm_SunVec *norm_sunVecs;			

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE_HOST zTsSolarAnalysis();
	
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.4
		*/
		ZSPACE_CUDA_CALLABLE_HOST ~zTsSolarAnalysis();

		//--------------------------
		//---- SET METHODS
		//--------------------------	
			   
		ZSPACE_CUDA_CALLABLE_HOST void setNormals(zVector *_normals, int _numNormals);

		ZSPACE_CUDA_CALLABLE_HOST bool setEPWData(string path);

		ZSPACE_CUDA_CALLABLE_HOST void setDates(zDomainDate & _dDate);

		ZSPACE_CUDA_CALLABLE_HOST void setLocation(zLocation &_location);

		ZSPACE_CUDA_CALLABLE_HOST void setNorm_SunVecs();

		//--------------------------
		//---- GET METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE int numNormals();

		ZSPACE_CUDA_CALLABLE int numSunVecs();

		ZSPACE_CUDA_CALLABLE int numDataPoints();

		ZSPACE_CUDA_CALLABLE zVector* getRawNormals();

		ZSPACE_CUDA_CALLABLE zVector* getRawSunVectors_hour();

		ZSPACE_CUDA_CALLABLE zVector* getRawSunVectors_day();

		ZSPACE_CUDA_CALLABLE zVector* getRawCompassVectors();

		ZSPACE_CUDA_CALLABLE zEPWData* getRawEPWData();

		ZSPACE_CUDA_CALLABLE float* getRawCummulativeRadiation();

		ZSPACE_CUDA_CALLABLE zVector getSunPosition(zDate &date);

		ZSPACE_CUDA_CALLABLE zDomainDate getSunRise_SunSet(zDate &date);

		ZSPACE_CUDA_CALLABLE zDomainDate getDates();

		ZSPACE_CUDA_CALLABLE zLocation getLocation();

		ZSPACE_CUDA_CALLABLE zNorm_SunVec* getNorm_SunVecs();

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------	
		
		ZSPACE_CUDA_CALLABLE void computeSunVectors_Year();

		ZSPACE_CUDA_CALLABLE void computeCompass();

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------	

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------	
	protected:

		ZSPACE_CUDA_CALLABLE_HOST void setMemory(int _newSize);

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