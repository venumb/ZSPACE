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


#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

#include<headers/zCudaToolsets/base/zCdUtilities.cuh>
#include<headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

using namespace zSpace;

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

ZSPACE_CUDA_GLOBAL void computeSolarAngles_kernel(zNorm_SunVec *norm_sunVecs, float *angles, int numAngles)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;	
	angles[i] = norm_sunVecs[i].norm.angle(norm_sunVecs[i].sunVec);	
}

ZSPACE_CUDA_GLOBAL void computeCummulativeSolarAngles_kernel(zVector *norms, float *angles, int numAngles , zDomainDate dDate, zLocation location)
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
		for (time_t minute = unixTime_sh; minute <= unixTime_eh; minute += 60)
		{			
			date.fromUnix(day + minute - unixTime_s);;

			sVec = getSunPosition(date, location);			
			
			angles[i] += norms[i].angle(sVec);

			count++; ;
		}

	}

	angles[i] /= count;

}

ZSPACE_EXTERN bool cdpSolarAngles(zTsSolarAnalysis &sAnalysis)
{
	int numSMs, numTB;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cu_gpuGetMaxGflopsDeviceId());
	cudaDeviceGetAttribute(&numTB, cudaDevAttrMaxThreadsPerBlock, cu_gpuGetMaxGflopsDeviceId());

	printf("\n numSMs: %i  numTB :%i ", numSMs, numTB);

	// Allocate device memory
	int NUM_ANGLES = sAnalysis.numNormals() * sAnalysis.numSunVecs();
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	zNorm_SunVec *d_norm_sunVecs;
	float *d_solarAngles;


	cudaEventRecord(start);

	checkCudaErrors(cudaMalloc((void **)&d_norm_sunVecs, NUM_ANGLES * sizeof(zNorm_SunVec)));
	checkCudaErrors(cudaMalloc((void **)&d_solarAngles, NUM_ANGLES * sizeof(float)));

	// transfer memory to device

	checkCudaErrors(cudaMemcpy(d_norm_sunVecs, sAnalysis.norm_sunVecs, NUM_ANGLES * sizeof(zNorm_SunVec), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_solarAngles, sAnalysis.solarAngles, NUM_ANGLES * sizeof(float), cudaMemcpyHostToDevice));

	// Launch Kernel
	printf( "\n Launching CDP kernel to compute solar angles \n " );
	dim3 block(256);
	dim3 grid((uint)ceil(NUM_ANGLES / (double)block.x));	
	computeSolarAngles_kernel <<< grid, block >>> (d_norm_sunVecs, d_solarAngles, NUM_ANGLES);
	checkCudaErrors(cudaGetLastError());

	// transfer memory to host
	checkCudaErrors(cudaMemcpy(sAnalysis.solarAngles, d_solarAngles, NUM_ANGLES * sizeof(float), cudaMemcpyDeviceToHost));
	   
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("\n gpu %1.8f ms \n", milliseconds );

	// Free memory.
	cudaFree(d_norm_sunVecs);
	cudaFree(d_solarAngles);	

	return true;
}

ZSPACE_EXTERN bool cdpCummulativeSolarAngles(zTsSolarAnalysis &sAnalysis)
{
	int numSMs, numTB;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cu_gpuGetMaxGflopsDeviceId());
	cudaDeviceGetAttribute(&numTB, cudaDevAttrMaxThreadsPerBlock, cu_gpuGetMaxGflopsDeviceId());

	printf("\n numSMs: %i  numTB :%i ", numSMs, numTB);

	// Allocate device memory
	int NUM_ANGLES = sAnalysis.numNormals();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	zVector *d_norms;
	float *d_solarAngles;

	zDomainDate dDate = sAnalysis.getDates();
	zLocation location = sAnalysis.location;



	cudaEventRecord(start);

	checkCudaErrors(cudaMalloc((void **)&d_norms, NUM_ANGLES * sizeof(zVector)));
	checkCudaErrors(cudaMalloc((void **)&d_solarAngles, NUM_ANGLES * sizeof(float)));

	// transfer memory to device

	checkCudaErrors(cudaMemcpy(d_norms, sAnalysis.getNormals(), NUM_ANGLES * sizeof(zVector), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_solarAngles, sAnalysis.solarAngles, NUM_ANGLES * sizeof(float), cudaMemcpyHostToDevice));

	// Launch Kernel
	printf("\n Launching CDP kernel to compute solar angles \n ");
	dim3 block(256);
	dim3 grid((uint)ceil(NUM_ANGLES / (double)block.x));
	computeCummulativeSolarAngles_kernel << < grid, block >> > (d_norms, d_solarAngles, NUM_ANGLES, dDate, location);
	checkCudaErrors(cudaGetLastError());

	//system("Pause");

	// transfer memory to host
	checkCudaErrors(cudaMemcpy(sAnalysis.solarAngles, d_solarAngles, NUM_ANGLES * sizeof(float), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("\n gpu %1.8f ms \n", milliseconds);

	// Free memory.
	cudaFree(d_norms);
	cudaFree(d_solarAngles);

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