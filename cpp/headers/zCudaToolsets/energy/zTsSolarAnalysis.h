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

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zCudaToolsets/energy/zCudaHost.h>

namespace zSpace
{
	struct zEPWData
	{
		int year, timeZone;
		string location;
		double dbTemperature, humidity, windSpeed, windDirection, radiation, pressure;
		double longitude, latitude;
	};

	struct zSunPosition
	{
		float altitude, azimuth;
	};

	/** \addtogroup zCudaToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup energy
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsSolarAnalysis
	*	\brief A function set for solar analysis.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class zTsSolarAnalysis
	{
	private:

		zCUDADate* cuda_Date;
		zCUDAVector* cuda_faceNormals;
		char* cuda_Path;

	protected:

		zObjMesh* objMesh;

		zUtilsCore coreUtils;

	public:

		zFnMesh fnMesh;
			
		zEPWData epwData;

		unordered_map<zDate, zEPWData> dataMap;

		double* angles_out;

		///////////////////////// Constructor - Overload - Destructor /////////////////////////

		zTsSolarAnalysis();
		 
		zTsSolarAnalysis(zObjMesh &_objMesh, string &ewp_path, string &mesh_path);

		~zTsSolarAnalysis();

		///////////////////////// Import /////////////////////////

		bool import_EPW(string &path);

		vector<zEPWData> createMap(zDomainDate _dDomain);
		
		///////////////////////// Sun Data /////////////////////////

		void sunData(zSunPosition&_sunposition, zDate date);

		void getTemperature(zDomainDate _dDomain, zDoubleArray&_temperatures);

		void getSunPosition(zVector&_sunPos, zDate &date, float radius = 100.f);

		///////////////////////// CUDA /////////////////////////

		void initializeCUDA(char* _path, zCUDADate *_date, zCUDAVector *_faceNormals, double *_anglesOut);

		void computeRadiation();

		///////////////////////// Utils /////////////////////////

		double GregorianToJulian(int year, int month, int day, int hour, int minute, int second);

		//https://social.msdn.microsoft.com/Forums/en-US/12bde6a1-3888-4857-bf71-a9fcadbe9c62/convert-julian-date-with-time-hms-to-date-time-in-c?forum=csharpgeneral
		void JulianToGregorian(double julian, zDate&_gDate);

		zVector SphericalToCartesian(double azimuth, double zenith, double radius);

		void CartesianToSpherical(zVector input, double &radius, double &zenith, double &azimuth);

		//////////////////////// Display /////////////////////////

		void getSunPath(zVectorArray&_sunPath, zDomainDate dDate, float Timezone, float radius, int interval);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCudaToolsets/energy/zTsSolarAnalysis.cpp>
#endif

#endif