// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#ifdef ZSPACE_CUDA_CALLABLE
#undef ZSPACE_CUDA_CALLABLE
#endif

#ifdef ZSPACE_CUDA_CALLABLE_HOST
#undef ZSPACE_CUDA_CALLABLE_HOST
#endif

#ifdef ZSPACE_CUDA_CALLABLE_DEVICE
#undef ZSPACE_CUDA_CALLABLE_DEVICE
#endif

#ifdef __CUDACC__
#define ZSPACE_CUDA_CALLABLE __host__ __device__  
#define ZSPACE_CUDA_CALLABLE_HOST __host__
#define ZSPACE_CUDA_CALLABLE_DEVICE __device__ 
#define ZSPACE_CUDA_GLOBAL __global__ 
#else
#define ZSPACE_CUDA_CALLABLE 
#define ZSPACE_CUDA_CALLABLE_HOST 
#define ZSPACE_CUDA_CALLABLE_DEVICE 
#define ZSPACE_CUDA_GLOBAL
#endif


/*!
*	\brief Defines the size of d_THREADSPERBLOCK.
*/
#ifndef d_THREADSPERBLOCK
#define d_THREADSPERBLOCK 256
#endif

/*!
*	\brief Defines the size of d_MEMORYMULTIPLIER.
*/
#ifndef d_MEMORYMULTIPLIER
#define d_MEMORYMULTIPLIER (d_THREADSPERBLOCK * 2)
#endif

/*!
*	\brief Defines the size of FloatSize.
*/
#ifndef FloatSize
#define FloatSize sizeof(float)
#endif

/*!
*	\brief Defines the size of zVectorSize.
*/
#ifndef zVectorSize
#define zVectorSize sizeof(zVector)
#endif

/*!
*	\brief Defines the size of zColorSize.
*/
#ifndef zColorSize
#define zColorSize sizeof(zColor)
#endif
