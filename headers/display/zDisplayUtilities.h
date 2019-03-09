#pragma once

#include<headers/core/zVector.h>
#include<headers/core/zMatrix.h>
#include<headers/core/zColor.h>

#include<headers/geometry/zGraph.h>
#include<headers/geometry/zGraphUtilities.h>
#include<headers/geometry/zMesh.h>
#include<headers/geometry/zMeshUtilities.h>

#include<headers/display/zBufferObject.h>

namespace zSpace
{
	/** \addtogroup zDisplay
	*	\brief Collection of general display and console print methods along with bufferobject class. 
	*  @{
	*/

	/** \addtogroup zDisplayUtilities
	*	\brief Collection of methods for console printing.
	*	\details It uses OPENGL framework for the display operations.
	*  @{
	*/

	//--------------------------
	//---- IMMEDIATE MODE DISPLAY
	//--------------------------

	/*! \brief This method draws a point.
	*	\param		 [in]		pos			- position of the point to be drawn.
	*	\param		 [in]		col			- color of the point.
	*	\param		 [in]		wt			- weight of the point.
	*	\since version 0.0.1
	*/	
	inline void drawPoint(zVector &pos, zColor col = zColor(1,0,0,1), double wt = 1)
	{
		
			glColor3f(col.r, col.g, col.b);
			glPointSize(wt);

			glBegin(GL_POINTS);
			glVertex3f(pos.x, pos.y, pos.z);
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
	inline void drawLine(zVector &p0, zVector &p1, zColor col = zColor(0, 0, 0, 1), double wt = 1)
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

	/*! \brief This method draws a poly-circle on the XY Plane given input center, radius and number of points.
	*	\param		[in]		c0			- center of circle.
	*	\param		[in]		circlePts	- points on the the circle. ( use getCircle method to compute the circle points).
	*	\param		[in]		dispLines	- draws lines if true, else displays points.
	*	\param		[in]		col			- color of the circle.
	*	\param		[in]		wt			- weight of the circle.
	*	\since version 0.0.1
	*/
	inline void drawCircle(zVector &c0, vector<zVector> &circlePts, bool dispLines = true, zColor col = zColor(0, 0, 0, 1), double wt = 1)
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
	inline void drawRectangle(zVector &minBB, zVector &maxBB, zColor col = zColor(0, 0, 0, 1), double wt = 1)
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
	inline void drawPolygon(vector<zVector> &pos, zColor col = zColor(0.5, 0.5, 0.5, 1))
	{
		glColor3f(col.r, col.g, col.b);

		glBegin(GL_POLYGON);
		for (int i = 0; i < pos.size(); i++)
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		glEnd();

		glColor3f(0, 0, 1);
	}

	/*! \brief This method displays the zGraph.
	*
	*	\param		[in]	graph		- zGraph to be displayed.
	*	\param		[in]	showVerts	- boolean true if the vertices of zGraph are to be displayed.
	*	\param		[in]	showEdges	- boolean true if the edges of zGraph are to be displayed.
	*	\since version 0.0.1
	*/	
	inline void drawGraph(zGraph &graph, bool showVerts = false, bool showEdges = true)
	{

		//draw vertex
		if (showVerts)
		{
			for (int i = 0; i < graph.vertexActive.size(); i++)
			{
				if (graph.vertexActive[i])
				{
					zColor col;
					double wt = 1;

					if (graph.vertexColors.size() > i)  col = graph.vertexColors[i];
					if (graph.vertexWeights.size() > i) wt = graph.vertexWeights[i];

					drawPoint(graph.vertexPositions[i], col, wt);
				}
			}
		}

		//draw edges
		if (showEdges)
		{

			for (int i = 0; i < graph.edgeActive.size(); i += 2)
			{
				if (graph.edgeActive[i])
				{
					if (graph.edges[i].getVertex() && graph.edges[i + 1].getVertex())
					{
						zColor col;
						double wt = 1;

						if (graph.edgeColors.size() > i)  col = graph.edgeColors[i];
						if (graph.edgeWeights.size() > i) wt = graph.edgeWeights[i];

						int v1 = graph.edges[i].getVertex()->getVertexId();
						int v2 = graph.edges[i + 1].getVertex()->getVertexId();

						drawLine(graph.vertexPositions[v1], graph.vertexPositions[v2], col, wt);
					}
				}

			}

		}
	}


	/*! \brief This method displays the zMesh.
	*
	*	\param		[in]	mesh			- zMesh to be displayed.
	*	\param		[in]	showVerts		- boolean true if the vertices of zMesh are to be displayed.
	*	\param		[in]	showEdges		- boolean true if the edges of zMesh are to be displayed.
	*	\param		[in]	showFaces		- boolean true if the faces of zMesh are to be displayed.
	*	\since version 0.0.1
	*/	
	inline void drawMesh(zMesh &mesh, bool showVerts, bool showEdges, bool showFaces)
	{

		//draw vertex
		if (showVerts)
		{

			for (int i = 0; i < mesh.vertexActive.size(); i++)
			{
				if (mesh.vertexActive[i])
				{
					zColor col;
					double wt = 1;

					if (mesh.vertexColors.size() > i)  col = mesh.vertexColors[i];
					if (mesh.vertexWeights.size() > i) wt = mesh.vertexWeights[i];

					drawPoint(mesh.vertexPositions[i], col, wt);
				}

			}
		}
				

		//draw edges
		if (showEdges)
		{

			for (int i = 0; i < mesh.edgeActive.size(); i += 2)
			{

				if (mesh.edgeActive[i])
				{

					if (mesh.edges[i].getVertex() && mesh.edges[i + 1].getVertex())
					{
						zColor col;
						double wt = 1;

						if (mesh.edgeColors.size() > i)  col = mesh.edgeColors[i];
						if (mesh.edgeWeights.size() > i) wt = mesh.edgeWeights[i];

						int v1 = mesh.edges[i].getVertex()->getVertexId();
						int v2 = mesh.edges[i + 1].getVertex()->getVertexId();

						drawLine(mesh.vertexPositions[v1], mesh.vertexPositions[v2], col, wt);
					}
				}

			}
		}


		//draw polygon
		if (showFaces)
		{
			int polyConnectsIndex = 0;

			for (int i = 0; i < mesh.faceActive.size(); i++)
			{
				if (mesh.faceActive[i])
				{
					vector<int> faceVertsIds;
					mesh.getVertices(i, zFaceData, faceVertsIds);

					vector<zVector> faceVerts;
					for (int j = 0; j < faceVertsIds.size(); j++)
					{
						faceVerts.push_back(mesh.vertexPositions[faceVertsIds[j]]);
					}

					zColor col;
					if (mesh.faceColors.size() > i)  col = mesh.faceColors[i];
					drawPolygon(faceVerts, col);

				}

			}
		}
	}

	/*! \brief This method displays the dihedral edges of a mesh above the input angle threshold.
	*
	*	\param		[in]	mesh					- input mesh to be displayed.
	*	\param		[in]	edge_dihedralAngles		- container of dihedral angles of the edges. 
	*	\param		[in]	angleThreshold			- angle threshold.
	*	\since version 0.0.1
	*/
	inline void drawMesh_DihedralEdges(zMesh &inMesh, vector<double> & edge_dihedralAngles, double angleThreshold = 45)
	{
		for (int i = 0; i < inMesh.edgeActive.size(); i += 2)
		{

			if (inMesh.edgeActive[i])
			{

				if (inMesh.edges[i].getVertex() && inMesh.edges[i + 1].getVertex() && abs(edge_dihedralAngles[i]) > angleThreshold)
				{
					zColor col;
					double wt = 1;

					if (inMesh.edgeColors.size() > i)  col = inMesh.edgeColors[i];
					if (inMesh.edgeWeights.size() > i) wt = inMesh.edgeWeights[i];

					int v1 = inMesh.edges[i].getVertex()->getVertexId();
					int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();

					drawLine(inMesh.vertexPositions[v1], inMesh.vertexPositions[v2], col, wt);
				}
			}
		}
	}


	/*! \brief This method displays the vertex normals of a mesh.
	*
	*	\param		[in]	mesh					- input mesh to be displayed.
	*	\param		[in]	dispScale				- display scale of the normal.
	*	\since version 0.0.1
	*/
	inline void drawMesh_VertexNormals(zMesh &inMesh, double dispScale = 1)
	{
		
		if (inMesh.vertexNormals.size() == 0 || inMesh.vertexNormals.size() != inMesh.vertexActive.size()) throw std::invalid_argument(" error: mesh normals not computed.");

		for (int i = 0; i < inMesh.vertexActive.size(); i++)
		{
			if (inMesh.vertexActive[i])
			{
				zVector p1 = inMesh.vertexPositions[i];
				zVector p2 = p1 + (inMesh.faceNormals[i] * dispScale);

				drawLine(p1, p2, zColor(0, 1, 0, 1));
			}
			
		}
	
	}


	/*! \brief This method displays the face normals of a mesh.
	*
	*	\param		[in]	mesh					- input mesh to be displayed.
	*	\param		[in]	fCenters				- centers of the faces. 
	*	\param		[in]	dispScale				- display scale of the normal.
	*	\since version 0.0.1
	*/
	inline void drawMesh_FaceNormals(zMesh &inMesh, vector<zVector> &fCenters,  double dispScale = 1)
	{
		if (inMesh.faceNormals.size() == 0 || inMesh.faceNormals.size() != inMesh.faceActive.size()) throw std::invalid_argument(" error: mesh normals not computed.");

		if (inMesh.faceActive.size() != fCenters.size()) throw std::invalid_argument(" error: number of face centers not equal to number of faces .");

		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			if (inMesh.faceActive[i])
			{
				zVector p1 = fCenters[i];
				zVector p2 = p1 + (inMesh.faceNormals[i] * dispScale);

				drawLine(p1, p2, zColor(0, 1, 0, 1));
			}
			
		}
		
	
	}

	// display scalar field
	//void drawScalarField2D(zScalarField2D &scalarField2D, bool showPoints = true, bool showDirections = false);


	/*! \brief This method displays the zRobot.
	*/
	//void drawRobot(zRobot &inRobot, double factor = 0.1, bool drawJointFrames = true, bool  drawJointMesh = false, bool drawWireFrame = false, bool drawGCodePoints = false, bool drawTargetFrame = false);


	/*! \brief This method displays zSlime.
	*/
	//void drawSlime(zSlime &slime, bool drawAgentPos = true, bool drawAgentDir = true, bool drawAgentTrail = true, bool foodSource = true, bool drawEnvironment = false, bool drawEnvironmentBoundary = true);


	//--------------------------
	//---- VBO DISPLAY
	//--------------------------

	/*! \brief This method draws points from the zBufferObject.
	*
	*	\param		[in]	inBufferObject			- input buffer object.
	*	\param		[in]	colors					- true if color is to be displayed.
	*	\since version 0.0.1
	*/
	inline void drawPointsFromBuffer(zBufferObject &inBufferObject, bool colors = true)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, inBufferObject.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, inBufferObject.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glDrawArrays(GL_POINTS, 0, inBufferObject.nVertices);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);


	}


	/*! \brief This method draws lines from the zBufferObject.
	*
	*	\param		[in]	inBufferObject			- input buffer object.
	*	\param		[in]	colors					- true if color is to be displayed.
	*	\since version 0.0.1
	*/
	inline void drawLinesFromBuffer(zBufferObject &inBufferObject, bool colors = true)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, inBufferObject.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, inBufferObject.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, inBufferObject.VBO_edgeIndices);
		glDrawElements(GL_LINES, inBufferObject.nEdges, GL_UNSIGNED_INT, bufferOffset(0));

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	/*! \brief This method draws triangles from the zBufferObject.
	*
	*	\param		[in]	inBufferObject			- input buffer object.
	*	\param		[in]	colors					- true if color is to be displayed.
	*	\since version 0.0.1
	*/
	inline void drawTrianglesFromBuffer(zBufferObject &inBufferObject, bool colors = true)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, inBufferObject.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, inBufferObject.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, inBufferObject.VBO_faceIndices);
		glDrawElements(GL_TRIANGLES, inBufferObject.nFaces, GL_UNSIGNED_INT, bufferOffset(0));
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	/*! \brief This method draws quads from the zBufferObject.
	*
	*	\param		[in]	inBufferObject			- input buffer object.
	*	\param		[in]	colors					- true if color is to be displayed.
	*	\since version 0.0.1
	*/
	inline void drawQuadsFromBuffer(zBufferObject &inBufferObject, bool colors = true)
	{
		glEnableClientState(GL_VERTEX_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, inBufferObject.VBO_vertices);
		glVertexPointer(3, GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(0));

		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, (vertexAttribStride * GLFloatSize), bufferOffset(12));

		if (colors)
		{
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER_ARB, inBufferObject.VBO_vertexColors);
			glColorPointer(4, GL_FLOAT, (vertexColorStride * GLFloatSize), bufferOffset(0));
		}


		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, inBufferObject.VBO_faceIndices);
		glDrawElements(GL_QUADS, inBufferObject.nFaces, GL_UNSIGNED_INT, bufferOffset(0));

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		if (colors) glDisableClientState(GL_COLOR_ARRAY);
	}

	/** @}*/

	/** @}*/
}