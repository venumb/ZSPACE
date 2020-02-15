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


#include<headers/zCudaToolsets/base/zCudaMacros.h>
#include<headers/zCore/base/zInline.h>
#include<headers/zCore/base/zDomain.h>
#include<headers/zCore/utilities/zUtilsCore.h>

#include <headers/zInterface/objects/zObjPointCloud.h>
#include <headers/zInterface/functionsets/zFnPointCloud.h>

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
		zVector *sunVecs;

		/*!	\brief number of sun vectors in the container	*/
		int numSunVec;

		/*!	\brief pointer to container of epw data	*/
		zEPWData* epwData;		

		/*!	\brief number of epw data points in the container	*/
		int numData;			

		/*!	\brief date domain	*/
		zDomainDate dDate;			
			   			   
	public:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to container of normals and sun vectors. It should be same size as solar angles*/
		zNorm_SunVec *norm_sunVecs;

		/*!	\brief location info	*/
		zLocation location;		

		/*!	\brief pointer to container of solar angles*/
		float *solarAngles;

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

		ZSPACE_CUDA_CALLABLE_HOST void setNorm_SunVecs();

		//--------------------------
		//---- GET METHODS
		//--------------------------	

		ZSPACE_CUDA_CALLABLE int numNormals();

		ZSPACE_CUDA_CALLABLE int numSunVecs();

		ZSPACE_CUDA_CALLABLE int numDataPoints();

		ZSPACE_CUDA_CALLABLE zVector* getNormals();

		ZSPACE_CUDA_CALLABLE zVector* getSunVectors();

		ZSPACE_CUDA_CALLABLE zEPWData* getEPWData();

		ZSPACE_CUDA_CALLABLE zVector getSunPosition(zDate &date, float radius);

		ZSPACE_CUDA_CALLABLE zDomainDate getSunRise_SunSet(zDate &date);

		ZSPACE_CUDA_CALLABLE zDomainDate getDates();

		ZSPACE_CUDA_CALLABLE zNorm_SunVec* getNorm_SunVecs();

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------	
		
		ZSPACE_CUDA_CALLABLE void computeSunVectors( float radius);		

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------	

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------	




	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCudaToolsets/energy/zTsSolarAnalysis.cpp>
#endif

#endif