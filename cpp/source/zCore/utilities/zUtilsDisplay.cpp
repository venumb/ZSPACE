// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zCore/utilities/zUtilsDisplay.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zUtilsDisplay::zUtilsDisplay() {}

	ZSPACE_INLINE zUtilsDisplay::zUtilsDisplay(int _bufferSize)
	{
		bufferObj = zObjBuffer(_bufferSize);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zUtilsDisplay::~zUtilsDisplay() {}

	ZSPACE_INLINE void zUtilsDisplay::drawTextAtPoint(string s, zPoint &pt)
	{

#if defined(__CUDACC__)  || defined(ZSPACE_UNREAL_INTEROP) || defined(ZSPACE_MAYA_INTEROP)
		// do nothing
#else
		unsigned int i;
		glRasterPos3f(pt.x, pt.y, pt.z);

		//for (i = 0; i < s.length(); i++)
			//glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, s[i]);
#endif
	}

	//---- IMMEDIATE MODE DISPLAY

	ZSPACE_INLINE void zUtilsDisplay::drawPoint(zPoint &pos, const zColor &col, const double &wt)
	{
		glColor3f(col.r, col.g, col.b);
		glPointSize(wt);

		glBegin(GL_POINTS);
		glVertex3f(pos.x, pos.y, pos.z);
		glEnd();

		glPointSize(1.0);
		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawPoints(zPoint *pos, zColor *col, double *wt, int size)
	{
		glBegin(GL_POINTS);
		for (int i = 0; i < size; i++)
		{

			glPointSize(wt[i]);
			glColor3f(col[i].r, col[i].g, col[i].b);
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		}
		glEnd();

		glPointSize(1.0);

		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawLine(zPoint &p0, zPoint &p1, const zColor &col , const double &wt)
	{
		glColor3f(col.r, col.g, col.b);
		glLineWidth(wt);

		glBegin(GL_LINES);
		glVertex3f(p0.x, p0.y, p0.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glEnd();

		glLineWidth(1.0);
		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawPolygon(zPointArray &pos, const zColor &col)
	{
		glColor3f(col.r, col.g, col.b);

		glBegin(GL_POLYGON);
		for (int i = 0; i < pos.size(); i++)
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		glEnd();

		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawCircle(zVector &c0, zPointArray &circlePts, bool dispLines, const zColor &col, const double &wt)
	{
		for (int i = 0; i < circlePts.size(); i++)
		{
			if (dispLines) drawLine(circlePts[i], circlePts[(i + 1) % circlePts.size()], col, wt);
			else drawPoint(circlePts[i], col, wt);
		}
	}

	ZSPACE_INLINE void zUtilsDisplay::drawRectangle(zVector &minBB, zVector &maxBB, const zColor &col, const double &wt)
	{
		zVector p0(minBB.x, minBB.y, minBB.z);
		zVector p1(minBB.x, maxBB.y, minBB.z);
		zVector p2(maxBB.x, maxBB.y, minBB.z);
		zVector p3(maxBB.x, minBB.y, minBB.z);


		drawLine(p0, p1, col, wt);
		drawLine(p1, p2, col, wt);
		drawLine(p2, p3, col, wt);
		drawLine(p3, p0, col, wt);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawCube(zVector &minBB, zVector &maxBB, const zColor &col, const double &wt)
	{
		zVector p0(minBB.x, minBB.y, minBB.z);
		zVector p1(minBB.x, maxBB.y, minBB.z);
		zVector p2(maxBB.x, maxBB.y, minBB.z);
		zVector p3(maxBB.x, minBB.y, minBB.z);

		zVector p4(minBB.x, minBB.y, maxBB.z);
		zVector p5(minBB.x, maxBB.y, maxBB.z);
		zVector p6(maxBB.x, maxBB.y, maxBB.z);
		zVector p7(maxBB.x, minBB.y, maxBB.z);

		drawLine(p0, p1, col, wt);
		drawLine(p1, p2, col, wt);
		drawLine(p2, p3, col, wt);
		drawLine(p3, p0, col, wt);

		drawLine(p4, p5, col, wt);
		drawLine(p5, p6, col, wt);
		drawLine(p6, p7, col, wt);
		drawLine(p7, p4, col, wt);

		drawLine(p0, p4, col, wt);
		drawLine(p1, p5, col, wt);
		drawLine(p2, p6, col, wt);
		drawLine(p3, p7, col, wt);
	}

	//---- MESH / GRAPH

	ZSPACE_INLINE void zUtilsDisplay::drawVertices(vector<zVertexHandle> &vHandles, zVector *pos, zColor *col, double *wt)
	{
		glBegin(GL_POINTS);
		for (auto &v : vHandles)
		{
			if (v.id == -1) continue;

			int i = v.id;

			glPointSize(wt[i]);
			glColor3f(col[i].r, col[i].g, col[i].b);
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		}
		glEnd();

		glPointSize(1.0);
		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawVertexIds(int numVerts, zVector *pos, zColor &col)
	{
		glColor3f(col.r, col.g, col.b);
		for (int i = 0; i < numVerts; i++)
		{			
			drawTextAtPoint(to_string(i), pos[i]);
		}
	}

	ZSPACE_INLINE void zUtilsDisplay::drawEdges(vector<zEdgeHandle> &eHandles, vector<zIntArray> &edgeVerts, zVector *pos, zColor *col, double *wt)
	{
		for (auto &e : eHandles)
		{
			if (e.id == -1) continue;

			int i = e.id;

			glColor3f(col[i].r, col[i].g, col[i].b);
			glLineWidth(wt[i]);

			glBegin(GL_LINES);

			for (int j = 0; j < edgeVerts[i].size(); j++)
			{
				glVertex3f(pos[edgeVerts[i][j]].x, pos[edgeVerts[i][j]].y, pos[edgeVerts[i][j]].z);
			}
			glEnd();

		}

		glLineWidth(1.0);
		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawEdgeIds(int numEdges, zVector *pos, zColor &col)
	{
		glColor3f(col.r, col.g, col.b);
		for (int i = 0; i < numEdges; i++)
		{
			drawTextAtPoint(to_string(i), pos[i]);
		}
	}

	ZSPACE_INLINE void zUtilsDisplay::drawFaces(vector<zFaceHandle> &fHandles, vector<zIntArray> &faceVerts, zVector *pos, zColor *col)
	{
		for (auto &f : fHandles)
		{
			if (f.id == -1) continue;

			int i = f.id;

			glColor3f(col[i].r, col[i].g, col[i].b);


			glBegin(GL_POLYGON);
			for (int j = 0; j < faceVerts[i].size(); j++)
			{
				glVertex3f(pos[faceVerts[i][j]].x, pos[faceVerts[i][j]].y, pos[faceVerts[i][j]].z);
			}
			glEnd();
		}

		glColor3f(0, 0, 1);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawFaceIds(int numFaces, zVector *pos, zColor &col)
	{
		glColor3f(col.r, col.g, col.b);
		for (int i = 0; i < numFaces; i++)
		{
			drawTextAtPoint(to_string(i), pos[i]);
		}
	}

	//---- TRANSFORM
	
	ZSPACE_INLINE void zUtilsDisplay::drawTransform(zTransformationMatrix &transform, float scale)
	{
		float* val = transform.asRawMatrix();
		float* pivot = transform.getRawPivot();

		glLineWidth(2.0);

		glBegin(GL_LINES);

		//X
		glColor3f(1, 0, 0);
		glVertex3f(pivot[0], pivot[1], pivot[2]);
		glVertex3f(pivot[0] + val[0] * scale, pivot[1] + val[4] * scale, pivot[2] + val[8] * scale);

		//Y
		glColor3f(0, 1, 0);
		glVertex3f(pivot[0], pivot[1], pivot[2]);
		glVertex3f(pivot[0] + val[1] * scale, pivot[1] + val[5] * scale, pivot[2] + val[9] * scale);

		//Z
		glColor3f(0, 0, 1);
		glVertex3f(pivot[0], pivot[1], pivot[2]);
		glVertex3f(pivot[0] + val[2] * scale, pivot[1] + val[6] * scale, pivot[2] + val[10] * scale);

		glEnd();

		glLineWidth(1.0);
		glColor3f(0, 0, 0);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawTransform(zTransform &transform, float scale)
	{
#ifndef __CUDACC__
		float* val = transform.data();
#else
		float* val = transform.getRawMatrixValues();
#endif

		glLineWidth(2.0);

		glBegin(GL_LINES);

		//X
		glColor3f(1, 0, 0);
		glVertex3f(val[3], val[7], val[11]);
		glVertex3f(val[3] + val[0] * scale, val[7] + val[4] * scale, val[11] + val[8] * scale);

		//Y
		glColor3f(0, 1, 0);
		glVertex3f(val[3], val[7], val[11]);
		glVertex3f(val[3] + val[1] * scale, val[7] + val[5] * scale, val[11] + val[9] * scale);

		//Z
		glColor3f(0, 0, 1);
		glVertex3f(val[3], val[7], val[11]);
		glVertex3f(val[3] + val[2] * scale, val[7] + val[6] * scale, val[11] + val[10] * scale);

		glEnd();

		glLineWidth(1.0);
		glColor3f(0, 0, 0);
	}

	//---- VBO DISPLAY

	ZSPACE_INLINE void zUtilsDisplay::drawPointsFromBuffer(bool colors)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, bufferObj.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, bufferObj.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glDrawArrays(GL_POINTS, 0, bufferObj.nVertices);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawLinesFromBuffer(bool colors)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, bufferObj.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, bufferObj.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObj.VBO_edgeIndices);
		glDrawElements(GL_LINES, bufferObj.nEdges, GL_UNSIGNED_INT, bufferOffset(0));

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawTrianglesFromBuffer(bool colors)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, bufferObj.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, bufferObj.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObj.VBO_faceIndices);
		glDrawElements(GL_TRIANGLES, bufferObj.nFaces, GL_UNSIGNED_INT, bufferOffset(0));

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	ZSPACE_INLINE void zUtilsDisplay::drawQuadsFromBuffer(bool colors)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, bufferObj.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, bufferObj.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObj.VBO_faceIndices);
		glDrawElements(GL_QUADS, bufferObj.nFaces, GL_UNSIGNED_INT, bufferOffset(0));

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}
}