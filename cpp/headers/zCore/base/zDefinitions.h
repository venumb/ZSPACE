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

#ifndef ZSPACE_DEFINITION
#define ZSPACE_DEFINITION

#pragma once
#include <vector>
#include <string>
using namespace std;

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zDefinitions
	*	\brief  The defintions of the library.
	*  @{
	*/

	//--------------------------
	//---- NUMERICAL DEFINITIONS
	//--------------------------

	/*!
	*	\brief Defines the value of Epsilon.
	*/
	#ifndef EPS
	#define EPS 0.0001 
	#endif

	/*!
	*	\brief Defines the default value of precision factor.
	*/
	#ifndef PRESFAC
	#define PRESFAC 4
	#endif

	/*!
	*	\brief Defines the value of zPI.
	*/
	#ifndef zPI
	#define zPI       3.14159265358979323846
	#endif

	/*!
	*	\brief Defines the value of 2 times zPI.
	*/
	#ifndef TWO_PI
	#define TWO_PI   6.28318530717958647693
	#endif

	/*!
	*	\brief Defines the value of 4 times zPI.
	*/
	#ifndef FOUR_PI
	#define FOUR_PI 12.56637061435917295385
	#endif

	/*!
	*	\brief Defines the value of 0.5 times zPI.
	*/
	#ifndef HALF_PI
	#define HALF_PI  1.57079632679489661923
	#endif

	/*!
	*	\brief Defines the conversion of degrees to radians.
	*/
	#ifndef zDEG_TO_RAD
	#define zDEG_TO_RAD (zPI/180.0)
	#endif

	/*!
	*	\brief Defines the conversion of radians to degrees.
	*/
	#ifndef zRAD_TO_DEG
	#define zRAD_TO_DEG (180.0/zPI)
	#endif

	/*!
	*	\brief Defines the distance calculation tolerance.
	*/
	#ifndef distanceTolerance
	#define distanceTolerance 0.00001
	#endif

	/*!
	*	\brief Defines the scale value for transformation when it is zero.
	*/
	#ifndef scaleZero
	#define scaleZero 0.00001
	#endif

	/*!
	*	\brief Defines the degrees of freedom.
	*/
	#ifndef DOF
	#define  DOF 6
	#endif

	//--------------------------
	//---- VBO DEFINITIONS
	//--------------------------

	/*!
	*	\brief Defines the vertex attribute stride. (position and normal)
	*/
	#ifndef vertexAttribStride
	#define vertexAttribStride 6
	#endif
	
	/*!
	*	\brief Defines the vertex color stride.
	*/
	#ifndef vertexColorStride
	#define vertexColorStride 4
	#endif

	/*!
	*	\brief Defines the edge index stride.
	*/
	#ifndef edgeIndexStride
	#define edgeIndexStride 1
	#endif

	/*!
	*	\brief Defines the face index stride.
	*/
	#ifndef faceIndexStride
	#define faceIndexStride 1
	#endif

	/*!
	*	\brief Defines the buffer offset in the VBO.
	*/
	#ifndef bufferOffset
	#define bufferOffset(i) ((void*)(i))
	#endif


	/*!
	*	\brief Defines the size of GLFloat.
	*/
	#ifndef GLFloatSize
	#define GLFloatSize sizeof(GLfloat)
	#endif

	/*!
	*	\brief Defines the size of GLInt.
	*/
	#ifndef GLIntSize
	#define GLIntSize sizeof(GLint)
	#endif

	/*! \typedef double3
	*	\brief An array of int of size 2.
	*
	*	\since version 0.0.2
	*/
	typedef int zInt2[2];

	/*! \typedef double3
	*	\brief An array of double of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef float zFloat3[3];
	
	/*! \typedef pDouble3
	*	\brief An pointer array of double of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef float* zPFloat3[3];

	/*! \typedef double4
	*	\brief An array of double of size 4.
	*
	*	\since version 0.0.2
	*/
	typedef float zFloat4[4];

	/*! \typedef pDouble4
	*	\brief An pointer array of double of size 4.
	*
	*	\since version 0.0.2
	*/
	typedef float* zPFloat4[4];

	/*! \typedef zScalar
	*	\brief A scalar definition used in scalar fields.
	*
	*	\since version 0.0.2
	*/
	typedef float zScalar;

	/** @}*/

	/** @}*/

	/** @}*/ 


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/** \addtogroup zHEHandles
	*	\brief The half edge geometry handle classes of the library.
	*  @{
	*/

	/*! \struct zVertexHandle
	*	\brief An vertex handle struct to  hold vertex information of a half-edge data structure as indicies. Used mostly internally for array resizing.
	*	\since version 0.0.3
	*/
	struct zVertexHandle { int id, he;  zVertexHandle() { id = he = -1; } };

	/*! \struct zHalfEdgeHandle
	*	\brief An half edge handle struct to  hold halfedge information of a half-edge data structure as indicies. Used mostly internally for array resizing.
	*	\since version 0.0.3
	*/
	struct zHalfEdgeHandle { int id, p, n, v, e, f; zHalfEdgeHandle() { id = p = n = v = e = f = -1; } };

	/*! \struct zEdgeHandle
	*	\brief An edge handle struct to  hold edge information of a half-edge data structure as indicies. Used mostly internally for array resizing.
	*	\since version 0.0.3
	*/
	struct zEdgeHandle { int id, he0, he1; zEdgeHandle() { id = he0 = he1 = -1; } };

	/*! \struct zFaceHandle
	*	\brief An face handle struct to  hold face information of a half-edge data structure as indicies. Used mostly internally for array resizing.
	*	\since version 0.0.3
	*/
	struct zFaceHandle { int id, he; zFaceHandle() { id = he = -1; } };

	/*! \struct zCurvature
	*	\brief A curvature struct defined to  hold defined to hold curvature information of a half-edge geometry.
	*	\since version 0.0.1
	*/
	struct zCurvature{double k1, k2; };

	/** @}*/
	/** @}*/	
	/** @}*/

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zSolar
	*	\brief The geometry classes of the library.
	*  @{
	*/
	

	/*! \struct zEPWData
	*	\brief A epw data struct defined to  hold energy plus weather information.
	*	\details https://energyplus.net/weather
	*	\since version 0.0.4
	*/
	struct zEPWData
	{
		float dbTemperature, humidity, windSpeed, windDirection, radiation, pressure;
	};

	/*! \struct zLocation
	*	\brief A location data struct defined to  hold timezone, latitude and longitude information.
	*	\since version 0.0.4
	*/
	struct zLocation
	{
		int timeZone;
		float longitude, latitude;
	};


	/** @}*/
	/** @}*/

}

#endif