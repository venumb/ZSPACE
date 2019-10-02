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


#include<headers/zCore/display/zObjBuffer.h>

namespace zSpace
{
	
	//----  CONSTRUCTOR
	   
	ZSPACE_INLINE zObjBuffer::zObjBuffer()
	{

		nVertices = 0;
		nColors = 0;
		nEdges = 0;
		nFaces = 0;

		max_nVertices = 0;

	}

	ZSPACE_INLINE zObjBuffer::zObjBuffer(GLint _Max_Num_Verts)
	{
		nVertices = 0;
		nColors = 0;
		nEdges = 0;
		nFaces = 0;

		max_nVertices = _Max_Num_Verts;

		GLfloat *vertices = new GLfloat[max_nVertices * vertexAttribStride];
		GLfloat *colors = new GLfloat[max_nVertices * vertexColorStride];

		GLint *edgeIndicies = new GLint[max_nVertices * 2 * edgeIndexStride];
		GLint *faceIndicies = new GLint[max_nVertices * 2 * faceIndexStride];

		GLsizei max_size_Vertices = max_nVertices * vertexAttribStride * GLFloatSize;
		GLsizei max_size_Colors = max_nVertices * vertexColorStride *  GLFloatSize;
		GLsizei max_size_Edgeindicies = max_nVertices * 2 * edgeIndexStride *  GLIntSize;
		GLsizei max_size_Faceindicies = max_nVertices * 2 * faceIndexStride *  GLIntSize;

		glGenBuffers(1, &VBO_vertices);
		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertices);
		glBufferData(GL_ARRAY_BUFFER, max_size_Vertices, vertices, GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //  source a standard memory location(RAM)

		glGenBuffers(1, &VBO_vertexColors);
		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertexColors);
		glBufferData(GL_ARRAY_BUFFER, max_size_Colors, colors, GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &VBO_edgeIndices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO_edgeIndices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_size_Edgeindicies, edgeIndicies, GL_STREAM_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glGenBuffers(1, &VBO_faceIndices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO_faceIndices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_size_Faceindicies, faceIndicies, GL_STREAM_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		delete[] vertices;
		delete[] colors;

		delete[] edgeIndicies;
		delete[] faceIndicies;

	}

	//----  DESTRUCTOR

	ZSPACE_INLINE zObjBuffer::~zObjBuffer() {}

	//----  APPEND METHODS

	ZSPACE_INLINE int zObjBuffer::appendVertexAttributes(zPoint*_positions, zVector* _normals, int size)
	{

		int out = nVertices;

		GLfloat *vertices = new GLfloat[size * vertexAttribStride];

		for (int i = 0; i < size; i++)
		{
			vertices[(i * vertexAttribStride) + 0] = _positions[i].x;
			vertices[(i * vertexAttribStride) + 1] = _positions[i].y;
			vertices[(i * vertexAttribStride) + 2] = _positions[i].z;


			if (!_normals)
			{
				vertices[(i * vertexAttribStride) + 3] = _normals[i].x;
				vertices[(i * vertexAttribStride) + 4] = _normals[i].y;
				vertices[(i * vertexAttribStride) + 5] = _normals[i].z;
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertices);
		glBufferSubData(GL_ARRAY_BUFFER, nVertices * vertexAttribStride * GLFloatSize, size  * vertexAttribStride * GLFloatSize, vertices);

		nVertices += size;

		delete[] vertices;

		return out;
	}

	ZSPACE_INLINE int zObjBuffer::appendVertexColors(zColor* _colors, int size)
	{

		int out = nColors;

		GLfloat *colors = new GLfloat[size * vertexColorStride];

		for (int i = 0; i < size; i++)
		{
			colors[(i * vertexColorStride) + 0] = _colors[i].r;
			colors[(i * vertexColorStride) + 1] = _colors[i].g;
			colors[(i * vertexColorStride) + 2] = _colors[i].b;
			colors[(i * vertexColorStride) + 3] = _colors[i].a;
		}


		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertexColors);
		glBufferSubData(GL_ARRAY_BUFFER, nColors * vertexColorStride *GLFloatSize, size * vertexColorStride * GLFloatSize, colors);

		nColors += size;

		delete[] colors;

		return out;

	}

	ZSPACE_INLINE int zObjBuffer::appendEdgeIndices(zIntArray &_edgeIndicies)
	{
		int out = nEdges;

		GLint *edgeIndicies = new GLint[_edgeIndicies.size() * edgeIndexStride];

		for (int i = 0; i < _edgeIndicies.size(); i++)
		{
			edgeIndicies[i] = _edgeIndicies[i];
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO_edgeIndices);
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, nEdges * edgeIndexStride *GLIntSize, _edgeIndicies.size() * edgeIndexStride * GLIntSize, edgeIndicies);

		nEdges += _edgeIndicies.size();


		delete[] edgeIndicies;

		return out;
	}

	ZSPACE_INLINE int zObjBuffer::appendFaceIndices(zIntArray &_faceIndicies)
	{
		int out = nFaces;

		GLint *faceIndicies = new GLint[_faceIndicies.size() * faceIndexStride];

		for (int i = 0; i < _faceIndicies.size(); i++)
		{
			faceIndicies[i] = _faceIndicies[i];
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO_faceIndices);
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, nFaces * faceIndexStride *GLIntSize, _faceIndicies.size() * faceIndexStride * GLIntSize, faceIndicies);

		nFaces += _faceIndicies.size();


		delete[] faceIndicies;

		return out;
	}

	//----  UPDATE METHODS

	ZSPACE_INLINE void zObjBuffer::updateVertexPositions(zPoint* _positions, int size, int startId)
	{

		GLfloat *positions = new GLfloat[size * vertexAttribStride];

		for (int i = 0; i < size; i++)
		{
			positions[(i * vertexAttribStride) + 0] = _positions[i].x;
			positions[(i * vertexAttribStride) + 1] = _positions[i].y;
			positions[(i * vertexAttribStride) + 2] = _positions[i].z;
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertices);
		glBufferSubData(GL_ARRAY_BUFFER, startId * vertexAttribStride * GLFloatSize, size  * vertexAttribStride * GLFloatSize, positions);

		delete[] positions;

	}

	ZSPACE_INLINE void zObjBuffer::updateVertexNormals(zVector* _normals, int size, int &startId)
	{
		GLfloat *normals = new GLfloat[size * vertexAttribStride];

		for (int i = 0; i < size; i++)
		{
			normals[(i * vertexAttribStride) + 3] = _normals[i].x;
			normals[(i * vertexAttribStride) + 4] = _normals[i].y;
			normals[(i * vertexAttribStride) + 5] = _normals[i].z;
		}

		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertices);
		glBufferSubData(GL_ARRAY_BUFFER, startId * vertexAttribStride * GLFloatSize, size  * vertexAttribStride * GLFloatSize, normals);

		delete[] normals;

	}

	ZSPACE_INLINE void zObjBuffer::updateVertexColors(zColor*_colors, int size, int &startId)
	{
		GLfloat *colors = new GLfloat[size * vertexColorStride];

		for (int i = 0; i < size; i++)
		{
			colors[(i * vertexColorStride) + 0] = _colors[i].r;
			colors[(i * vertexColorStride) + 1] = _colors[i].g;
			colors[(i * vertexColorStride) + 2] = _colors[i].b;
			colors[(i * vertexColorStride) + 3] = _colors[i].a;
		}


		glBindBuffer(GL_ARRAY_BUFFER, VBO_vertexColors);
		glBufferSubData(GL_ARRAY_BUFFER, startId * vertexColorStride *GLFloatSize, size * vertexColorStride * GLFloatSize, colors);

		delete[] colors;

	}

	//----  CLEAR METHODS

	ZSPACE_INLINE void zObjBuffer::clearBufferForRewrite()
	{
		nVertices = nColors = nEdges = nFaces = 0;

	}

}