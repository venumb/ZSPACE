#pragma once


#include<headers/framework/display/zObjBuffer.h>

namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
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

	class zUtilsDisplay
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
		zUtilsDisplay() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_bufferSize			- input buffer size.
		*	\since version 0.0.2
		*/
		zUtilsDisplay(int _bufferSize)
		{
			bufferObj = zObjBuffer(_bufferSize);
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zUtilsDisplay() {}

		//--------------------------
		//---- IMMEDIATE MODE DISPLAY
		//--------------------------

		/*! \brief This method draws a point.
		*	\param		 [in]		pos			- position of the point to be drawn.
		*	\param		 [in]		col			- color of the point.
		*	\param		 [in]		wt			- weight of the point.
		*	\since version 0.0.1
		*/
		void drawPoint(zVector &pos, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1)
		{

			glColor3f(col.r, col.g, col.b);
			glPointSize(wt);

			glBegin(GL_POINTS);
			glVertex3f(pos.x, pos.y, pos.z);
			glEnd();

			glPointSize(1.0);

			glColor3f(0, 0, 1);

		}

		/*! \brief This method draws points from a container.
		*	\param		 [in]		pos			- container of positions  to be drawn.
		*	\param		 [in]		col			- container of colors.
		*	\param		 [in]		wt			- container of weights.
		*	\since version 0.0.1
		*/
		void drawPoints(vector<zVector> &pos, vector<zColor> &col , vector<double> &wt)
		{

			glBegin(GL_POINTS);
			for (int i = 0; i < pos.size(); i++)
			{
				glPointSize(wt[i]);
				glColor3f(col[i].r, col[i].g, col[i].b);
				glVertex3f(pos[i].x, pos[i].y, pos[i].z);
			}			
			glEnd();

			glPointSize(1.0);

			glColor3f(0, 0, 1);

		}

		/*! \brief This method draws a line between the given two points.
		*	\param		[in]		p0			- start Point of the line.
		*	\param		[in]		p1			- end Point of the line.
		*	\param		[in]		col			- color of the line.
		*	\param		[in]		wt			- weight of the line.
		*	\since version 0.0.1
		*/
		void drawLine(zVector &p0, zVector &p1, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1)
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

		/*! \brief This method draws edge lines from a two dimensional container.
		*	\param		 [in]		pos			- container of positions  to be drawn.
		*	\param		 [in]		col			- container of colors.
		*	\param		 [in]		wt			- container of weights.
		*	\since version 0.0.1
		*/
		void drawEdges(vector<vector<zVector>> &pos, vector<zColor> &col, vector<double> wt)
		{
			glBegin(GL_LINES);
			for (int i = 0; i < pos.size(); i+= 2)
			{
				glColor3f(col[i].r, col[i].g, col[i].b);
				glLineWidth(wt[i]);

				for (int j = 0; j < pos[i].size(); j++)
				{
					glVertex3f(pos[i][j].x, pos[i][j].y,pos[i][j].z);
				}
			}

			glEnd();

			glLineWidth(1.0);
			glColor3f(0, 0, 1);

		}

		/*! \brief This method draws a poly-circle on the XY Plane given input center, radius and number of points.
		*	\param		[in]		c0			- center of circle.
		*	\param		[in]		circlePts	- points on the the circle. ( use getCircle method to compute the circle points).
		*	\param		[in]		dispLines	- draws lines if true, else displays points.
		*	\param		[in]		col			- color of the circle.
		*	\param		[in]		wt			- weight of the circle.
		*	\since version 0.0.1
		*/
		void drawCircle(zVector &c0, vector<zVector> &circlePts, bool dispLines = true, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1)
		{
			for (int i = 0; i < circlePts.size(); i++)
			{
				if (dispLines) drawLine(circlePts[i], circlePts[(i + 1) % circlePts.size()], col, wt);
				else drawPoint(circlePts[i], col, wt);
			}

		}

		/*! \brief This method draws a rectangle on the XY Plane given input bound vectors.
		*	\param		[in]		minBB		- min bounds of the rectangle.
		*	\param		[in]		maxBB		- max bounds of the rectangle.
		*	\param		[in]		col			- color of the rectangle.
		*	\param		[in]		wt			- weight of the rectangle.
		*	\since version 0.0.1
		*/
		void drawRectangle(zVector &minBB, zVector &maxBB, const zColor &col = zColor(1, 0, 0, 1), const double &wt = 1)
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

		/*! \brief This method displays the a face of zMesh.
		*
		*	\param		[in]	pos				- vector of type zVector storing the polygon points.
		*	\param		[in]	col				- color of the polygon.
		*	\since version 0.0.1
		*/
		void drawPolygon(vector<zVector> &pos, const zColor &col = zColor(1, 0, 0, 1))
		{
			glColor3f(col.r, col.g, col.b);

			glBegin(GL_POLYGON);
			for (int i = 0; i < pos.size(); i++)
				glVertex3f(pos[i].x, pos[i].y, pos[i].z);
			glEnd();

			glColor3f(0, 0, 1);
		}

		/*! \brief This method draws polygons from a two dimensional container.
		*	\param		 [in]		pos			- container of positions  to be drawn.
		*	\param		 [in]		col			- container of colors.
		*	\since version 0.0.1
		*/
		void drawPolygons(vector<vector<zVector>> &pos, vector<zColor> &col)
		{
			
			for (int i = 0; i < pos.size(); i += 1)
			{
				glColor3f(col[i].r, col[i].g, col[i].b);
				

				glBegin(GL_POLYGON);
				for (int j = 0; j < pos[i].size(); j++)
				{
					glVertex3f(pos[i][j].x, pos[i][j].y, pos[i][j].z);
				}
				glEnd();
			}
		
			glColor3f(0, 0, 1);

		}

		/*! \brief This method draws the X, Y Z axis of the transformation matrix.
		*	\param		 [in]		transform	- transform  to be drawn.
		*	\since version 0.0.2
		*/
		void drawTransform(zTransformationMatrix &transform)
		{

			double* val = transform.asRawMatrix();
			double* pivot = transform.getRawPivot();	


			glLineWidth(2.0);

			glBegin(GL_LINES);
			
			//X
			glColor3f(1, 0, 0);
			glVertex3f(pivot[0], pivot[1], pivot[2]);
			glVertex3f(pivot[0] + val[0], pivot[1] + val[4], pivot[2] + val[8]);

			//Y
			glColor3f(0, 1, 0);
			glVertex3f(pivot[0], pivot[1], pivot[2]);
			glVertex3f(pivot[0] + val[1], pivot[1] + val[5], pivot[2] + val[9]);

			//Z
			glColor3f(0, 0, 1);
			glVertex3f(pivot[0], pivot[1], pivot[2]);
			glVertex3f(pivot[0] + val[2], pivot[1] + val[6], pivot[2] + val[10]);
			
			glEnd();

			glLineWidth(1.0);
			glColor3f(0, 0, 0);

		}
		
		/*! \brief This method draws the X, Y Z axis of the transformation matrix.
		*	\param		 [in]		transform	- transform  to be drawn.
		*	\since version 0.0.2
		*/
		void drawTransform(zTransform &transform)
		{

			double* val = transform.data();

			glLineWidth(2.0);

			glBegin(GL_LINES);

			//X
			glColor3f(1, 0, 0);
			glVertex3f(val[3], val[7], val[11]);
			glVertex3f(val[3] + val[0], val[7] + val[4], val[11] + val[8]);

			//Y
			glColor3f(0, 1, 0);
			glVertex3f(val[3], val[7], val[11]);
			glVertex3f(val[3] + val[1], val[7] + val[5], val[11] + val[9]);

			//Z
			glColor3f(0, 0, 1);
			glVertex3f(val[3], val[7], val[11]);
			glVertex3f(val[3] + val[2], val[7] + val[6], val[11] + val[10]);

			glEnd();

			glLineWidth(1.0);
			glColor3f(0, 0, 0);

		}

		//--------------------------
		//---- VBO DISPLAY
		//--------------------------

		/*! \brief This method draws points from the zBufferObject.
		*
 		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawPointsFromBuffer( bool colors = true)
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


		/*! \brief This method draws lines from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawLinesFromBuffer(bool colors = true)
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

		/*! \brief This method draws triangles from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawTrianglesFromBuffer( bool colors = true)
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

		/*! \brief This method draws quads from the zBufferObject.
		*
		*	\param		[in]	colors					- true if color is to be displayed.
		*	\since version 0.0.1
		*/
		void drawQuadsFromBuffer(bool colors = true)
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

	};

	
}