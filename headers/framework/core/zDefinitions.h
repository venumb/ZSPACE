#pragma once


namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
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
	*	\brief Defines the value of PI.
	*/
	#ifndef PI
	#define PI       3.14159265358979323846
	#endif

	/*!
	*	\brief Defines the value of 2 times PI.
	*/
	#ifndef TWO_PI
	#define TWO_PI   6.28318530717958647693
	#endif

	/*!
	*	\brief Defines the value of 4 times PI.
	*/
	#ifndef FOUR_PI
	#define FOUR_PI 12.56637061435917295385
	#endif

	/*!
	*	\brief Defines the value of 0.5 times PI.
	*/
	#ifndef HALF_PI
	#define HALF_PI  1.57079632679489661923
	#endif

	/*!
	*	\brief Defines the conversion of degrees to radians.
	*/
	#ifndef DEG_TO_RAD
	#define DEG_TO_RAD (PI/180.0)
	#endif

	/*!
	*	\brief Defines the conversion of radians to degrees.
	*/
	#ifndef RAD_TO_DEG
	#define RAD_TO_DEG (180.0/PI)
	#endif

	/*!
	*	\brief Defines the distance calculation tolerance.
	*/
	#ifndef distanceTolerance
	#define distanceTolerance 0.00001
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

	

	/** @}*/

	/** @}*/

	/** @}*/ 
}
