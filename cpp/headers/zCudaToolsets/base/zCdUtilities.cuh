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

#ifndef ZSPACE_CD_UTILITIES
#define ZSPACE_CD_UTILITIES

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

#include<headers/zCudaToolsets/base/zCudaDefinitions.h>
#include<headers/zCore/base/zInline.h>
#include<headers/zCore/base/zExtern.h>

#include <stdio.h>
using namespace std;

//---- UTILITIES

ZSPACE_CUDA_CALLABLE_HOST ZSPACE_INLINE int cu_ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct 
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
		{ 0x70,  64},
		{ 0x72,  64},
		{ 0x75,  64},
		{   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

ZSPACE_CUDA_CALLABLE_HOST ZSPACE_INLINE int cu_gpuGetMaxGflopsDeviceId()
{
	int current_device = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device = 0;
	int device_count = 0, best_SM_arch = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&device_count);

	// Find the best major SM Architecture GPU device
	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);
		if (deviceProp.major > 0 && deviceProp.major < 9999)
		{
			best_SM_arch = max(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;
	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{
			sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = cu_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

		if (compute_perf > max_compute_perf)
		{
			// If we find GPU with SM major > 2, search only these
			if (best_SM_arch > 2)
			{
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch)
				{
					max_compute_perf = compute_perf;
					max_perf_device = current_device;
				}
			}
			else
			{
				max_compute_perf = compute_perf;
				max_perf_device = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

ZSPACE_CUDA_CALLABLE_HOST ZSPACE_INLINE static void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) 
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));		
	}
}

ZSPACE_CUDA_CALLABLE_HOST ZSPACE_INLINE void cdpGetAttributes(int &numSMs, int &numTB)
{
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cu_gpuGetMaxGflopsDeviceId());
	cudaDeviceGetAttribute(&numTB, cudaDevAttrMaxThreadsPerBlock, cu_gpuGetMaxGflopsDeviceId());

	//printf("\n numSMs: %i  numTB :%i ", numSMs, numTB);
}

ZSPACE_EXTERN bool checkCudaExists(string &version)
{

	int runtime;
	cudaRuntimeGetVersion(&runtime);

	cout <<"\n CUDA: "<< runtime;

	if (runtime >= 10020)
	{
		version = "NVIDIA CUDA 10.2 or higher Installed.";
		return true;
	}
	else
	{
		version = "Install NVIDIA CUDA 10.2 or higher.";
		return false;
	}
	   
}

#endif