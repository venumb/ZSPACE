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
#include <depends/spa/spa.h>




namespace zSpace
{
	struct zEPWDataPoint
	{
		int year, month, day, hour, minute;
		double db_temperature, radiation, pressure;
	};

	struct zEPWData
	{
		string location;
		double latitude, longitude, timezone, elevation;

		vector<zEPWDataPoint> dataPoints;
	};

	struct zYear
	{
		int id;
		vector<int> months;
		vector<int> days;
		vector<int> hours;
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
	*	\brief A function set to convert graph data to polyhedra.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class zTsSolarAnalysis
	{
	private:


	protected:

		zObjMesh* objMesh;

		zUtilsCore coreUtils;

	public:

		zFnMesh fnMesh;
			
		zEPWData epwData;
		spa_data SPA;

		multimap<MultiKey, double> radiationMap;

		unordered_map<int, int> yearsMap;
		vector<zYear> years;

		///////////////////////// Constructor - Overload - Destructor /////////////////////////

		zTsSolarAnalysis();
		 
		zTsSolarAnalysis(zObjMesh &_objMesh, string &path);

		~zTsSolarAnalysis();

		///////////////////////// Methods /////////////////////////

		bool import_EPW(string &path);

		void createMap();

		///// Methods for Sun Position Calculation 

		void computeSunPosition(int day, int hour, int minute, double longitude, double latitude, double timzone, double &azimuth, double &zenith);

		zVector computeSun(int year, int month, int day, int hour);

		void computeSPA_EPW(int idDataPoint);

		void computeSPA(int year, int month, int day, int hour);

		void computeSPA(int year, int month, int day, int hour, int minute, int second, double timezone, double longitude, double latitude, double elevation);

		///////////////////////// Utils /////////////////////////

		double GregorianToJulian(int year, int month, int day, int H, int M, int S);

		void JulianToGregorian(double julian, int &year, int &month, int&day);

		zVector SphericalToCartesian(double azimuth, double zenith, double radius);

		void CartesianToSpherical(zVector input, double &radius, double &zenith, double &azimuth);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCudaToolsets/energy/zTsSolarAnalysis.cpp>
#endif

#endif