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

#ifndef ZSPACE_UTILS_DISPLAY_H
#define ZSPACE_UTILS_DISPLAY_H

#pragma once

#include<headers/zCore/display/zObjBuffer.h>

#if defined(__CUDACC__)  || defined(ZSPACE_UNREAL_INTEROP) || defined(ZSPACE_MAYA_INTEROP)
	// All defined OK so do nothing
#else
	//#include <depends/freeglut/freeglut_std.h>
#endif

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \class zUtilsDisplay
	*	\brief A display utility class for drawing points, lines , polygons using OPENGL.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zUtilsDisplay
	{
	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief Buffer object  */
		zObjBuffer bufferObj;
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zUtilsDisplay();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_bufferSize			- input buffer size.
		*	\since version 0.0.2
		*/
		zUtilsDisplay(int _bufferSize);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zUtilsDisplay();

		//--------------------------
		//---- IMMEDIATE MODE DISPLAY
		//--------------------------

		/*! \brief This method draws a point.
		*	\param		 [in]		s			- string to be displayed.
		*	\param		 [in]		pt			- location of text.
		*	\since version 0.0.4
		*/
		void drawTextAtPoint(string s, zPoint &pt);

		/*! \brief This method draws a point.
		*	\param		 [in]		pos			- position of the point to be drawn.
		*	\param		 [in]		col			- color of the point.
		*	\param		 [in]		wt			- weight of the point.
		*	\since version 0.0.1
		*/
		void drawPoint(zPoint &pos, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1);

		/*! \brief This method draws vertices of a graph or mesh.
		*	\param		 [in]		pos			- container of positions to be drawn.
		*	\param		 [in]		col			- container of colors.
		*	\param		 [in]		wt			- container of weights.
		*	\param		 [in]		size		- size of container.
		*	\since version 0.0.1
		*/
		void drawPoints(zPoint *pos, zColor *col, double *wt, int size);

		/*! \brief This method draws a line between the given two points.
		*	\param		[in]		p0			- start Point of the line.
		*	\param		[in]		p1			- end Point of the line.
		*	\param		[in]		col			- color of the line.
		*	\param		[in]		wt			- weight of the line.
		*	\since version 0.0.1
		*/
		void drawLine(zPoint &p0, zPoint &p1, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1);

		/*! \brief This method displays the a face of mesh.
		*
		*	\param		[in]	pos				- vector of type zVector storing the polygon points.
		*	\param		[in]	col				- color of the polygon.
		*	\since version 0.0.1
		*/
		void drawPolygon(zPointArray &pos, const zColor &col = zColor(1, 0, 0, 1));

		/*! \brief This method draws a poly-circle on the XY Plane given input center, radius and number of points.
		*	\param		[in]		c0			- center of circle.
		*	\param		[in]		circlePts	- points on the the circle. ( use getCircle method to compute the circle points).
		*	\param		[in]		dispLines	- draws lines if true, else displays points.
		*	\param		[in]		col			- color of the circle.
		*	\param		[in]		wt			- weight of the circle.
		*	\since version 0.0.1
		*/
		void drawCircle(zVector &c0, zPointArray &circlePts, bool dispLines = true, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1);

		/*! \brief This method draws a rectangle on the XY Plane given input bound vectors.
		*	\param		[in]		minBB		- min bounds of the rectangle.
		*	\param		[in]		maxBB		- max bounds of the rectangle.
		*	\param		[in]		col			- color of the rectangle.
		*	\param		[in]		wt			- weight of the rectangle.
		*	\since version 0.0.1
		*/
		void drawRectangle(zVector &minBB, zVector &maxBB, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1); 

		/*! \brief This method draws a cube given input bound vectors.
		*	\param		[in]		minBB		- min bounds of the rectangle.
		*	\param		[in]		maxBB		- max bounds of the rectangle.
		*	\param		[in]		col			- color of the rectangle.
		*	\param		[in]		wt			- weight of the rectangle.
		*	\since version 0.0.1
		*/
		void drawCube(zVector &minBB, zVector &maxBB, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1);		
		
		//--------------------------
		//---- MESH / GRAPH
		//--------------------------
		
		/*! \brief This method draws vertices of a graph or mesh.
		*	\param		 [in]		vHandles	- vertex handle container.
		*	\param		 [in]		pos			- container of positions to be drawn.
		*	\param		 [in]		col			- container of colors.
		*	\param		 [in]		wt			- container of weights.
		*	\since version 0.0.1
		*/
		void drawVertices(vector<zVertexHandle> &vHandles, zVector *pos, zColor *col, double *wt);

		/*! \brief This method draws vertexIds of a graph or mesh.
		*	\param		 [in]		numVerts		- number of vertices.
		*	\param		 [in]		*pos			- container of positions to be drawn.
		*	\param		 [in]		&col			- container of colors.
		*	\since version 0.0.4
		*/
		void drawVertexIds(int numVerts, zVector *pos, zColor &col);

		/*! \brief This method draws edge of a graph or mesh.
		*	\param		 [in]		eHandles	- edge handle container.
		*	\param		 [in]		edgeVerts	- container of vertices per edge to be drawn.
		*	\param		 [in]		pos			- container of positions.
		*	\param		 [in]		col			- container of colors.
		*	\param		 [in]		wt			- container of weights.
		*	\since version 0.0.1
		*/
		void drawEdges(vector<zEdgeHandle> &eHandles, vector<zIntArray> &edgeVerts, zVector *pos, zColor *col, double *wt);

		/*! \brief This method draws edgeIds of a graph or mesh.
		*	\param		 [in]		numEdges		- number of edges.
		*	\param		 [in]		*pos			- container of positions to be drawn.
		*	\param		 [in]		&col			- container of colors.
		*	\since version 0.0.4
		*/
		void drawEdgeIds(int numEdges, zPoint *pos, zColor &col);

		/*! \brief This method draws polygons of a mesh.
		*	\param		 [in]		eHandles	- edge handle container.
		*	\param		 [in]		faceVerts	- container of vertices per polygon to be drawn.
		*	\param		 [in]		pos			- container of positions.
		*	\param		 [in]		col			- container of colors.
		*	\since version 0.0.1
		*/
		void drawFaces(vector<zFaceHandle> &fHandles, vector<zIntArray> &faceVerts, zVector *pos, zColor *col);

		/*! \brief This method draws FaceIds of a graph or mesh.
		*	\param		 [in]		numFaces		- number of faces.
		*	\param		 [in]		*pos			- container of positions to be drawn.
		*	\param		 [in]		&col			- container of colors.
		*	\since version 0.0.4
		*/
		void drawFaceIds(int numFaces, zVector *pos, zColor &col);

		//--------------------------
		//---- TRANSFORM
		//--------------------------

		/*! \brief This method draws the X, Y Z axis of the transformation matrix.
		*	\param		 [in]		transform	- transform  to be drawn.
		*	\since version 0.0.2
		*/
		void drawTransform(zTransformationMatrix &transform);
		
		/*! \brief This method draws the X, Y Z axis of the transformation matrix.
		*	\param		 [in]		transform	- transform  to be drawn.
		*	\since version 0.0.2
		*/
		void drawTransform(zTransform &transform);

		//--------------------------
		//---- VBO DISPLAY
		//--------------------------

		/*! \brief This method draws points from the zBufferObject.
		*
 		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawPointsFromBuffer(bool colors = true);

		/*! \brief This method draws lines from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawLinesFromBuffer(bool colors = true);

		/*! \brief This method draws triangles from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawTrianglesFromBuffer(bool colors = true);

		/*! \brief This method draws quads from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawQuadsFromBuffer(bool colors = true);
	};	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/utilities/zUtilsDisplay.cpp>
#endif

#endif