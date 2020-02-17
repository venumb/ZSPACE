// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Federico Borello <federico.borello@zaha-hadid.com>, Cesar Fragachan <cesar.fragachan@zaha-hadid.com>
//

#ifndef ZSPACE_CD_SOLAR_ANALYSIS
#define ZSPACE_CD_SOLAR_ANALYSIS

#pragma once




#include<headers/zCudaToolsets/base/zCdUtilities.cuh>
#include<headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

using namespace zSpace;

//---- DEVICE VARIABLES



zVector *d_norms;

float *d_cummulativeRadiation;

int d_MemSize;


//---- CUDA HOST DEVICE METHODS 

ZSPACE_CUDA_CALLABLE zVector getSunPosition(zDate &date, zLocation &location)
{
	float LocalTime = date.tm_hour + (date.tm_min / 60.0);

	double JD = date.toJulian();

	double n = JD - 2451545.0;

	float LDeg = (float)fmod((280.460 + 0.9856474 * n), 360.0);
	float gDeg = (float)fmod((357.528 + 0.9856003 * n), 360.0);

	float LambdaDeg = LDeg + 1.915 * sin(gDeg * DEG_TO_RAD) + 0.01997 * sin(2 * gDeg * DEG_TO_RAD);

	float epsilonDeg = 23.439 - 0.0000004 * n;

	float alphaDeg;
	alphaDeg = atan(cos(epsilonDeg * DEG_TO_RAD) * tan(LambdaDeg * DEG_TO_RAD));
	alphaDeg *= RAD_TO_DEG;
	if (cos(LambdaDeg  * DEG_TO_RAD) < 0)	alphaDeg += (4 * (atan(1.0) * RAD_TO_DEG));

	float deltaDeg = asin(sin(epsilonDeg * DEG_TO_RAD) * sin(LambdaDeg  * DEG_TO_RAD)) * RAD_TO_DEG;

	zDate dZero(date.tm_year, date.tm_mon, date.tm_mday, 0, 0);
	double JDNull = dZero.toJulian();

	float TNull = ((JDNull - 2451545.0) / 36525);
	float T = LocalTime - location.timeZone;

	float thetaGh = 6.697376 + 2400.05134 * TNull + 1.002738 * T;

	float thetaG = (float)fmod(thetaGh * 15.0, 360.0);
	float theta = thetaG + location.longitude;

	float tauDeg = theta - alphaDeg;

	float denom = (cos(tauDeg  * DEG_TO_RAD)*sin(location.latitude  * DEG_TO_RAD) - tan(deltaDeg  * DEG_TO_RAD)*cos(location.latitude  * DEG_TO_RAD));
	float aDeg = atan(sin(tauDeg  * DEG_TO_RAD) / denom);
	aDeg *= RAD_TO_DEG;
	if (denom < 0) aDeg = aDeg + 180;
	aDeg += 180; //add 180 to azimuth to compute from the north.

	float hDeg = asin(cos(deltaDeg  * DEG_TO_RAD)*cos(tauDeg  * DEG_TO_RAD)*cos(location.latitude  * DEG_TO_RAD) + sin(deltaDeg  * DEG_TO_RAD)*sin(location.latitude  * DEG_TO_RAD));
	hDeg *= RAD_TO_DEG;

	float valDeg = hDeg + (10.3 / (hDeg + 5.11));
	float RDeg = 1.02 / (tan(valDeg * DEG_TO_RAD));

	float hRDeg = hDeg + (RDeg / 60);

	return zPoint(cos(aDeg * DEG_TO_RAD) * sin(hRDeg * DEG_TO_RAD), cos(aDeg * DEG_TO_RAD) * cos(hRDeg * DEG_TO_RAD), sin(aDeg * DEG_TO_RAD));
}

ZSPACE_CUDA_CALLABLE_HOST void cleanDeviceMemory()
{
	// Free memory.
	cudaFree(d_norms);
	cudaFree(d_cummulativeRadiation);
}

ZSPACE_CUDA_CALLABLE_HOST void setDeviceMemory(int _newSize)
{
	if (_newSize < d_MemSize) return;
	else
	{
		while (d_MemSize < _newSize) d_MemSize += d_MEMORYMULTIPLIER;

		cleanDeviceMemory();

		checkCudaErrors(cudaMalloc((void **)&d_norms, d_MemSize * zVectorSize));
		checkCudaErrors(cudaMalloc((void **)&d_cummulativeRadiation, d_MemSize * FloatSize));
	}
}



//---- CUDA KERNEL 

ZSPACE_CUDA_GLOBAL void computeCummulativeRadiation_kernel(zVector *norms, float *angles, int numAngles , zDomainDate dDate, zLocation location)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;	

	
	int  unixTime_s = (int) dDate.min.toUnix();
	int  unixTime_e = (int)dDate.max.toUnix();

	// get minute domain per day
	zDate minHour = dDate.min;
	zDate maxHour = dDate.min;

	maxHour.tm_hour = dDate.max.tm_hour;
	maxHour.tm_min = dDate.max.tm_min;

	int  unixTime_sh = (int) minHour.toUnix();
	int  unixTime_eh = (int) maxHour.toUnix();

	int count = 0;	
	zDate date;
	zVector sVec;

	angles[i] = 0;

	for (time_t day = unixTime_s; day <= unixTime_e; day += 86400)
	{
		for (time_t hour = unixTime_sh; hour <= unixTime_eh; hour += 3600)
		{			
			date.fromUnix(day + hour - unixTime_s);;

			sVec = getSunPosition(date, location);			
			
			angles[i] += norms[i].angle(sVec);

			count++; ;
		}

	}
		
	angles[i] /= count;

}

//---- launch KERNEL METHODS

ZSPACE_EXTERN bool cdpCummulativeRadiation(zTsSolarAnalysis &sAnalysis)
{
	int numSMs, numTB;
	cu_getAttributes(numSMs, numTB);

	// Allocate device memory
	int NUM_ANGLES = sAnalysis.numNormals();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	

	zDomainDate dDate = sAnalysis.getDates();
	zLocation location = sAnalysis.getLocation();

	cudaEventRecord(start);

	setDeviceMemory(NUM_ANGLES);

	// transfer memory to device

	checkCudaErrors(cudaMemcpy(d_norms, sAnalysis.getRawNormals(), d_MemSize * zVectorSize, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_solarAngles, sAnalysis.solarAngles, NUM_ANGLES * sizeof(float), cudaMemcpyHostToDevice));

	// Launch Kernel
	printf("\n Launching CDP kernel to compute solar radiation \n ");
	dim3 block(d_THREADSPERBLOCK);
	dim3 grid((uint)ceil(d_MemSize / (double)block.x));	
	computeCummulativeRadiation_kernel << < grid, block >> > (d_norms, d_cummulativeRadiation, d_MemSize, dDate, location);
	checkCudaErrors(cudaGetLastError());

	//system("Pause");

	// transfer memory to host
	checkCudaErrors(cudaMemcpy(sAnalysis.getRawCummulativeRadiation(), d_cummulativeRadiation, d_MemSize * FloatSize, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("\n gpu %1.8f ms \n", milliseconds);

	printf("\n gpu  d_MemSize %i \n", d_MemSize);

	return true;
}


#endif

// ZERO COPY METHOD
// build map example 
//cudaSetDeviceFlags(cudaDeviceMapHost);
//cudaHostAlloc(&sAnalysis.norm_sunVecs, NUM_ANGLES * sizeof(zNorm_SunVec), cudaHostAllocMapped);
//cudaHostGetDevicePointer(&d_norm_sunVecs, sAnalysis.norm_sunVecs, 0);

//cudaSetDeviceFlags(cudaDeviceMapHost);
//cudaHostAlloc(&sAnalysis.solarAngles, NUM_ANGLES * sizeof(float), cudaHostAllocMapped);
//cudaHostGetDevicePointer(&d_solarAngles, sAnalysis.solarAngles, 0);