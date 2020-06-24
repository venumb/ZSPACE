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

float *d_norms_sunVecs;
float *d_colors;
float *d_cumulativeRadiation;


int d_MemSize;

//---- CUDA HOST DEVICE METHODS 

ZSPACE_CUDA_CALLABLE float ofMap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
{
	return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
}

ZSPACE_CUDA_CALLABLE zVector getSunPosition(zDate &date, zLocation &location)
{
	float LocalTime = date.tm_hour + (date.tm_min / 60.0);

	double JD = date.toJulian();

	double n = JD - 2451545.0;

	float LDeg = (float)fmod((280.460 + 0.9856474 * n), 360.0);
	float gDeg = (float)fmod((357.528 + 0.9856003 * n), 360.0);

	float LambdaDeg = LDeg + 1.915 * sin(gDeg * zDEG_TO_RAD) + 0.01997 * sin(2 * gDeg * zDEG_TO_RAD);

	float epsilonDeg = 23.439 - 0.0000004 * n;

	float alphaDeg;
	alphaDeg = atan(cos(epsilonDeg * zDEG_TO_RAD) * tan(LambdaDeg * zDEG_TO_RAD));
	alphaDeg *= zRAD_TO_DEG;
	if (cos(LambdaDeg  * zDEG_TO_RAD) < 0)	alphaDeg += (4 * (atan(1.0) * zRAD_TO_DEG));

	float deltaDeg = asin(sin(epsilonDeg * zDEG_TO_RAD) * sin(LambdaDeg  * zDEG_TO_RAD)) * zRAD_TO_DEG;

	zDate dZero(date.tm_year, date.tm_mon, date.tm_mday, 0, 0);
	double JDNull = dZero.toJulian();

	float TNull = ((JDNull - 2451545.0) / 36525);
	float T = LocalTime - location.timeZone;

	float thetaGh = 6.697376 + 2400.05134 * TNull + 1.002738 * T;

	float thetaG = (float)fmod(thetaGh * 15.0, 360.0);
	float theta = thetaG + location.longitude;

	float tauDeg = theta - alphaDeg;

	float denom = (cos(tauDeg  * zDEG_TO_RAD)*sin(location.latitude  * zDEG_TO_RAD) - tan(deltaDeg  * zDEG_TO_RAD)*cos(location.latitude  * zDEG_TO_RAD));
	float aDeg = atan(sin(tauDeg  * zDEG_TO_RAD) / denom);
	aDeg *= zRAD_TO_DEG;
	if (denom < 0) aDeg = aDeg + 180;
	aDeg += 180; //add 180 to azimuth to compute from the north.

	float hDeg = asin(cos(deltaDeg  * zDEG_TO_RAD)*cos(tauDeg  * zDEG_TO_RAD)*cos(location.latitude  * zDEG_TO_RAD) + sin(deltaDeg  * zDEG_TO_RAD)*sin(location.latitude  * zDEG_TO_RAD));
	hDeg *= zRAD_TO_DEG;

	float valDeg = hDeg + (10.3 / (hDeg + 5.11));
	float RDeg = 1.02 / (tan(valDeg * zDEG_TO_RAD));

	float hRDeg = hDeg + (RDeg / 60);

	return zPoint(cos(aDeg * zDEG_TO_RAD) * sin(hRDeg * zDEG_TO_RAD), cos(aDeg * zDEG_TO_RAD) * cos(hRDeg * zDEG_TO_RAD), sin(aDeg * zDEG_TO_RAD));
}

ZSPACE_EXTERN void cleanDeviceMemory()
{
	// Free memory.
	cudaFree(d_norms_sunVecs);
	cudaFree(d_colors);
	cudaFree(d_cumulativeRadiation);
}

ZSPACE_CUDA_CALLABLE_HOST void setDeviceMemory(int _numNormals, int _numSunVecs, bool EPW_Read)
{
	if (_numNormals + _numSunVecs < d_MemSize) return;
	else
	{
		while (d_MemSize < _numNormals + _numSunVecs) d_MemSize += d_MEMORYMULTIPLIER;

		cleanDeviceMemory();

		checkCudaErrors(cudaMalloc((void **)&d_norms_sunVecs, d_MemSize * FloatSize));

		checkCudaErrors(cudaMalloc((void **)&d_cumulativeRadiation, d_MemSize * FloatSize));

		// set size to num normals
		checkCudaErrors(cudaMalloc((void **)&d_colors, _numNormals * FloatSize));
	}
}

//---- CUDA KERNEL 

ZSPACE_CUDA_GLOBAL void computeCummulativeRadiation_kernel(float *norms_sunvecs,float *radiation, float *colors, int numNormals, int numSunvecs, zDomainColor domainColor)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;	

	if (i < numNormals && i %3 == 0)
	{
		zVector norm(norms_sunvecs[i + 0], norms_sunvecs[i + 1], norms_sunvecs[i + 2]);

		uint sunvecs_offset = numNormals - i;

		float angle = 0;
		float rad = 0;
		int count = 0;
		int radCount = 0;
		for (int o = i; o < i + numSunvecs; o+= 3)
		{
			int j = o + sunvecs_offset;
			

			if (norms_sunvecs[j + 0] != INVALID_VAL && norms_sunvecs[j + 1] != INVALID_VAL && norms_sunvecs[j + 2] != INVALID_VAL)
			{

				zVector sVec(norms_sunvecs[j + 0], norms_sunvecs[j + 1], norms_sunvecs[j + 2]);

				float a = norm.angle(sVec);	
				float weight = ofMap(a, 0.0, 180.0, 1.0, 0.0);		

				if (a > 0.001)
				{
					angle += a;
					count++;
				}
				
				if (weight * radiation[j + 2] > 0.001)
				{
					rad += (weight * radiation[j + 2]);
					radCount++;					
				}				
			}			
		}

		angle /= count;
		rad /= radCount;


		radiation[i + 0] = angle;
		radiation[i + 1] = rad;
		radiation[i + 2] = INVALID_VAL;

		if (rad < RAD_MIN)
		{
			colors[i + 0] = domainColor.min.h;
			colors[i + 1] = domainColor.min.s;
			colors[i + 2] = domainColor.min.v;
		}
		else if (rad >= RAD_MIN && rad <= RAD_MAX)
		{
			colors[i + 0] = ofMap(rad, RAD_MIN, RAD_MAX, domainColor.min.h, domainColor.max.h);
			colors[i + 1] = ofMap(rad, RAD_MIN, RAD_MAX, domainColor.min.s, domainColor.max.s);
			colors[i + 2] = ofMap(rad, RAD_MIN, RAD_MAX, domainColor.min.v, domainColor.max.v);
		}		
		else
		{
			colors[i + 0] = domainColor.max.h;
			colors[i + 1] = domainColor.max.s;
			colors[i + 2] = domainColor.max.v;
		}


	}
	

}

ZSPACE_CUDA_GLOBAL void computeCummulativeAngle_kernel(float *norms_sunvecs, float *colors, int numNormals, int numSunvecs, zDomainColor domainColor)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numNormals && i % 3 == 0)
	{
		zVector norm(norms_sunvecs[i + 0], norms_sunvecs[i + 1], norms_sunvecs[i + 2]);

		uint sunvecs_offset = numNormals - i;

		float angle = 0;
		int count = 0;
		for (int o = i; o < i + numSunvecs; o += 3)
		{
			int j = o + sunvecs_offset;
			zVector sVec(norms_sunvecs[j + 0], norms_sunvecs[j + 1], norms_sunvecs[j + 2]);

			if (sVec.x != INVALID_VAL && sVec.y != INVALID_VAL && sVec.z != INVALID_VAL)
			{
				float a = norm.angle(sVec);
				if (a > 0.001)
				{
					angle += a;
					count++;
				}				
			}
		}
		angle /= count;



		if (angle > RAD_ANGLE_MAX)
		{
			colors[i + 0] = domainColor.min.h;
			colors[i + 1] = domainColor.min.s;
			colors[i + 2] = domainColor.min.v;
		}
		else if (angle >= RAD_ANGLE_MIN && angle <= RAD_ANGLE_MAX)
		{
			colors[i + 0] = ofMap(angle, RAD_ANGLE_MAX, RAD_ANGLE_MIN, domainColor.min.h, domainColor.max.h);
			colors[i + 1] = ofMap(angle, RAD_ANGLE_MAX, RAD_ANGLE_MIN, domainColor.min.s, domainColor.max.s);
			colors[i + 2] = ofMap(angle, RAD_ANGLE_MAX, RAD_ANGLE_MIN, domainColor.min.v, domainColor.max.v);
		}
		else
		{
			colors[i + 0] = domainColor.max.h;
			colors[i + 1] = domainColor.max.s;
			colors[i + 2] = domainColor.max.v;
    }

		//printf("\n %1.2f %1.2f %1.2f  | %1.2f  | %1.2f %1.2f %1.2f", norm.x, norm.y, norm.z, angle, colors[i + 0], colors[i + 1], colors[i + 2]);



	}


}

//---- launch KERNEL METHODS

ZSPACE_EXTERN bool cdpCummulativeRadiation(zTsSolarAnalysis &sAnalysis, bool EPWRead)
{
	int numSMs, numTB;
	cdpGetAttributes(numSMs, numTB);

	// Allocate device memory
	int NUM_NORMALS = sAnalysis.numNormals();
	int NUM_SUNVECS = sAnalysis.numSunVecs();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	zDomainColor dColor = sAnalysis.getDomain_Colors();
  setDeviceMemory(NUM_NORMALS, NUM_SUNVECS, EPWRead);
  
	cudaEventRecord(start);
  
	// transfer memory to device

	checkCudaErrors(cudaMemcpy(d_norms_sunVecs, sAnalysis.getRawNormals_SunVectors(), d_MemSize * FloatSize, cudaMemcpyHostToDevice));
	if(EPWRead) checkCudaErrors(cudaMemcpy(d_cumulativeRadiation, sAnalysis.getRawCummulativeRadiation(), d_MemSize * FloatSize, cudaMemcpyHostToDevice));

	// Launch Kernel
	//printf("\n Launching CDP kernel to compute solar radiation \n ");
	dim3 block(d_THREADSPERBLOCK);
	dim3 grid((uint)ceil(d_MemSize / (double)block.x));	
	
	if(EPWRead) computeCummulativeRadiation_kernel << < grid, block >> > (d_norms_sunVecs, d_cumulativeRadiation, d_colors, NUM_NORMALS, NUM_SUNVECS, dColor);
	else computeCummulativeAngle_kernel << < grid, block >> > (d_norms_sunVecs, d_colors, NUM_NORMALS, NUM_SUNVECS, dColor);
  
	checkCudaErrors(cudaGetLastError());


	// transfer memory to host

	checkCudaErrors(cudaMemcpy(sAnalysis.getRawColors(), d_colors, NUM_NORMALS * FloatSize, cudaMemcpyDeviceToHost));
	if (EPWRead) checkCudaErrors(cudaMemcpy(sAnalysis.getRawCummulativeRadiation(), d_cumulativeRadiation, d_MemSize * FloatSize, cudaMemcpyDeviceToHost));

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