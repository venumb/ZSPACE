#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>

#include <headers/zCudaToolsets/energy/zCudaHost.h>

//////////////////////////////////////////////////////////////////////////  ------------------------------------------- DEVICE POINTERS

char* path_In;
zCUDADate* date_In;
zCUDAVector* faceNormals_In;

double* angles_Out;

int num_faces = 0;

//////////////////////////////////////////////////////////////////////////  ------------------------------------------- UTILS

__host__ inline int cu_ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
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
__host__ int cu_gpuGetMaxGflopsDeviceId()
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
__host__ static void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		//exit(EXIT_FAILURE); 
	}
}

//////////////////////////////////////////////////////////////////////////  ------------------------------------------- KERNELS

__global__ void computeAngles(char* _path_In, zCUDADate* _date_In, zCUDAVector* _faceNormals_In, int _num_faces, double* _angles_Out)
{
	///////// Grid-Stride Loop Method (Flexible Kernel)
	///////// It loops over the data array one grid - size at a time.

	//for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_nParticles; i += blockDim.x * gridDim.x) {}

	///////// Monolithic Kernel: it assumes a single large grid of threads to process the entire array in one pass  

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//_angles_Out[i] = _sunVector_In[0].angle(_faceNormals_In[i]);

	//printf("\n KERNEL: %1.3f", _angles_Out[i]);
	//printf("\n KERNEL: %1.3f,%1.3f,%1.3f", _faceNormals_In[i].v.x, _faceNormals_In[i].v.y, _faceNormals_In[i].v.z);
}

//////////////////////////////////////////////////////////////////////////------------------------------------------- DATA-TRANSFER

//////////////////////////////////// H-D

void initialise_DeviceMemory(int size)
{
	num_faces = size;

	if (path_In != NULL)
		cudaFree(path_In);
	cudaMalloc((void **)&path_In, num_faces * sizeof(char));

	if (faceNormals_In != NULL)
		cudaFree(faceNormals_In);
	cudaMalloc((void **) &faceNormals_In, num_faces * sizeof(zCUDAVector));

	if (date_In != NULL)
		cudaFree(date_In);
	cudaMalloc((void **)&date_In, num_faces * sizeof(zCUDADate));

	if (angles_Out != NULL)
		cudaFree(angles_Out);
	cudaMalloc((void **)&angles_Out, num_faces * sizeof(double));

	checkCUDAError(" cudaMalloc - device data storage ");
}

void copy_HostToDevice(double *host_angles_Out, zCUDAVector *host_faceNormals, zCUDADate *host_Date, char *host_Path, int size)
{
	initialise_DeviceMemory(size);

	cudaMemcpy(path_In, host_Path, size * sizeof(zCUDADate), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy In_A H-D");

	cudaMemcpy(faceNormals_In, host_faceNormals, size * sizeof(zCUDAVector), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy In_A H-D");

	cudaMemcpy(date_In, host_Date, size * sizeof(zCUDADate), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy In_B H-D");

	cudaMemcpy(angles_Out, host_angles_Out, size * sizeof(double), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy Out H-D");
}

//////////////////////////////////// D-H

double * copy_DeviceToHost()
{
	double *host_out = (double *) malloc(num_faces * sizeof(double));

	cudaMemcpy(host_out, angles_Out, num_faces *sizeof(double),cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy Out D-H");

	return host_out;
}

//////////////////////////////////////////////////////////////////////////------------------------------------------- SETUP DATA

void callKernel()
{

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cu_gpuGetMaxGflopsDeviceId());

	computeAngles << < num_faces, 1 >> > (path_In, date_In, faceNormals_In, num_faces, angles_Out);
	checkCUDAError("kernel_updateData");

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	printf("\nKernel Called \n");
}

//////////////////////////////////////////////////////////////////////////------------------------------------------- EXIT

void CUDA_Free()
{
	cudaFree(path_In);
	cudaFree(date_In);
	cudaFree(faceNormals_In);
	cudaFree(angles_Out);
	checkCUDAError("cudaFree: device data ");
}
