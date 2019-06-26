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
	*	\brief An array of double of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef double double3[3];
	
	/*! \typedef pDouble3
	*	\brief An pointer array of double of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef double* pDouble3[3];	

	/** @}*/

	/** @}*/

	/** @}*/ 


		/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
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


	/** @}*/

	/** @}*/

	

}
