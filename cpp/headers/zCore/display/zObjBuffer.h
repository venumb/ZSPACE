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

#ifndef ZSPACE_OBJ_BUFFER_H
#define ZSPACE_OBJ_BUFFER_H

#pragma once

#define GL_GLEXT_PROTOTYPES


#include <depends/openGL/glew.h>
#include <depends/openGL/glext.h>

#include <GL/GL.h>			// Header File For The OpenGL32 Library
#include <gl/glu.h>			// Header File For The GLu32 Library



//#include <windows.h>		// Header File For Windows
//#include <stdio.h>

#include <headers/zCore/base/zDefinitions.h>
#include <headers/zCore/base/zVector.h>
#include <headers/zCore/base/zColor.h>
#include <headers/zCore/base/zTransformationMatrix.h>
#include <headers/zCore/base/zTypeDef.h>

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zDisplay
	*	\brief  The display classes of the library.
	*  @{
	*/

	/*! \class zObjBuffer
	*	\brief A vertex buffer object class used to append geometry to the buffer.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zObjBuffer
	{
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*!	\brief VBO for vertices - positions and normals(if available)  */
		GLuint VBO_vertices;

		/*!	\brief VBO for vertex color  */
		GLuint VBO_vertexColors;

		/*!	\brief VBO to store edge vertex connectivity indicies  */
		GLuint VBO_edgeIndices;

		/*!	\brief VBO to store face vertex connectivity indicies  */
		GLuint VBO_faceIndices;

		/*!	\brief stores number of max vertices possible in the buffer  */
		GLint max_nVertices;

		/*!	\brief stores number of vertices in the buffer  */
		GLint nVertices;

		/*!	\brief stores number of colors in the buffer  */
		GLint nColors;

		/*!	\brief stores number of edge vertex connectivity indicies in the buffer  */
		GLint nEdges;

		/*!	\brief stores number of edge vertex connectivity indicies in the buffer  */
		GLint nFaces;

		

		//--------------------------
		//----  CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/		
		zObjBuffer();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_Max_Num_Verts		- size of the buffer to be initialised.
		*	\since version 0.0.1
		*/
		zObjBuffer(GLint _Max_Num_Verts);

		//--------------------------
		//----  DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zObjBuffer();

		//--------------------------
		//----  APPEND METHODS
		//--------------------------


		/*! \brief This method appends the vertex attributes of position and normals(if available) to the buffer. 
		*
		*	\param		[in]	_positions			- zPoint pointer to vertex positions container.
		*	\param		[in]	_normals			- zVector pointer to vertex normals container.
		*	\param		[in]	size				- size of container.
		*	\since version 0.0.1
		*/
		int appendVertexAttributes(zPoint*_positions, zVector* _normals, int size);

		/*! \brief This method appends the vertex color attribute to the buffer.
		*
		*	\param		[in]	_colors				- zColor pointer to vertex colors container.
		*	\param		[in]	size				- size of container.
		*	\since version 0.0.1
		*/
		int appendVertexColors(zColor* _colors, int size);

		/*! \brief This method appends the edge vertex connectivity indicies to the buffer.
		*
		*	\param		[in]	_edgeIndicies			- container of edge vertex connectivity indicies.
		*	\since version 0.0.1
		*/
		int appendEdgeIndices(zIntArray &_edgeIndicies);

		/*! \brief This method appends the face vertex connectivity indicies to the buffer.
		*
		*	\param		[in]	_faceIndicies			- container of face vertex connectivity indicies.
		*	\since version 0.0.1
		*/
		int appendFaceIndices(zIntArray &_faceIndicies);
			
			   
		//--------------------------
		//----  UPDATE METHODS
		//--------------------------

		
		/*! \brief This method update the vertex positions stored in the buffer from the buffer offset given by the input start index.
		*
		*	\param		[in]	_positions			- zPoint pointer to vertex positions container.
		*	\param		[in]	size				- size of container.
		*	\param		[in]	startId				- buffer offset.
		*	\since version 0.0.1
		*/
		void updateVertexPositions(zPoint* _positions, int size, int startId);

		/*! \brief This method update the vertex normals stored in the buffer from the buffer offset given by the input start index.
		*
		*	\param		[in]	_normals			-  zVector pointer to vertex normals container.
		*	\param		[in]	size				- size of container.
		*	\param		[in]	startId				- buffer offset.
		*	\since version 0.0.1
		*/		
		void updateVertexNormals(zVector* _normals, int size, int &startId);

		/*! \brief This method update the vertex colors stored in the buffer from the buffer offset given by the input start index.
		*
		*	\param		[in]	_colors				- zColor pointer to vertex colors container.
		*	\param		[in]	size				- size of container.
		*	\param		[in]	startId				- buffer offset.
		*	\since version 0.0.1
		*/
		void updateVertexColors(zColor*_colors, int size, int &startId);


		//--------------------------
		//----  CLEAR METHODS
		//--------------------------
	
		/*! \brief This method clears all the buffer for rewriting.
		*
		*	\since version 0.0.1
		*/
		void clearBufferForRewrite();

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/display/zObjBuffer.cpp>
#endif

#endif