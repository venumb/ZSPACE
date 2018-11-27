#pragma once

#include<headers/geometry/zGraph.h>
#include<headers/geometry/zMesh.h>
#include<headers/geometry/zScalarfield.h>



namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief  The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryUtilities
	*	\brief Collection of utility methods for graphs, meshes and scalarfields.
	*  @{
	*/

	/** \addtogroup zFieldDUtilities
	*	\brief Collection of general utility methods for fields.
	*  @{
	*/
	
	//--------------------------
	//----  FIELD UTILITIES
	//--------------------------	
	
	
	/*! \brief This method computes the minimum distance between a point and edge and the closest Point on the edge. 
	*
	*	\details based on http://paulbourke.net/geometry/pointlineplane/
	*	\param	[in]	pt			- point
	*	\param	[in]	e0			- start point of edge.
	*	\param	[in]	e1			- end point of edge.
	*	\param	[out]	closest_Pt	- closest point on edge to the input point.
	*	\return			minDist		- distance to closest point.
	*	\since version 0.0.1
	*/
	
	double minDist_Edge_Point_2D(zVector & pt, zVector & e0, zVector & e1, zVector & closest_Pt)
	{
		double out = 0.0;

		zVector n = (e1 - e0) ^ (zVector(0, 0, 1));
		n.normalize();
		closest_Pt = n * ((e0 - pt) * n);
		closest_Pt += pt;

		float len = e0.distanceTo(e1);

		zVector ed = (e0 - e1) / len;
		double param = (closest_Pt - e1) * ed;



		if (param > len)param = len;
		if (param < 0) param = 0;

		closest_Pt = e1 + ed * param;

		return closest_Pt.distanceTo(pt);
	}

	/*! \brief This method computes the distance function.
	*
	*	\param	[in]	r	- distance value.
	*	\param	[in]	a	- value of a.
	*	\param	[in]	b	- value of b.
	*	\since version 0.0.1
	*/
	
	double F_of_r(double &r, double &a, double &b)
	{
		if (0 <= r && r <= b / 3.0)return (a * (1.0 - (3.0 * r * r) / (b*b)));
		if (b / 3.0 <= r && r <= b) return (3 * a / 2 * pow(1.0 - (r / b), 2.0));
		if (b <= r) return 0;
	}

	/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
	*
	*	\param	[out]	dMin	- stores the minimum scalar value
	*	\param	[out]	dMax	- stores the maximum scalar value
	*	\param	[in]	buffer	- buffer of scalars.
	*	\since version 0.0.1
	*/
	
	void getMinMaxOfScalars(vector<double>& scalars, double &dMin, double &dMax)
	{
		dMin = zMin(scalars);
		dMax = zMax(scalars);		
	}

	/*! \brief This method normalises the scalar values at the given field buffer.
	*
	*	\param	[in]	buffer	- buffer of scalars.
	*	\since version 0.0.1
	*/
	
	void normaliseScalars(vector<double>& scalars)
	{
		double dMin, dMax;
		getMinMaxOfScalars(scalars, dMin, dMax);

		for (int i = 0; i < scalars.size(); i++)
			scalars[i] = ofMap(scalars[i], dMin, dMax, -1.0, 1.0);		
	}
	
	/** @}*/
	
	
	
	
	/** \addtogroup zScalarField2DUtilities
	*	\brief Collection of utility methods for scalar fields 2D.
	*  @{
	*/	

	//--------------------------
	//----  2D FIELD METHODS
	//--------------------------


	/*! \brief This method creates a mesh from the input scalar field.
	*
	*	\param	[in]	inField				- input zScalarfield2D
	*	\return			zMesh				- mesh of the scalar field.
	*	\since version 0.0.1
	*/
	
	zMesh fromScalarField2D(zScalarField2D inField)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		zVector unitVec = zVector(inField.unit_X, inField.unit_Y, 0);
		zVector startPt = inField.minBB;

		int resX = inField.n_X + 1;
		int resY = inField.n_Y + 1;

		for (int i = 0; i<resX; i++)
		{
			for (int j = 0; j <resY; j++)
			{
				zVector pos;
				pos.x = startPt.x + i * unitVec.x;
				pos.y = startPt.y + j * unitVec.y;

				positions.push_back(pos);
			}
		}



		for (int i = 0; i < resX - 1; i++)
		{
			for (int j = 0; j < resY - 1; j++)
			{
				int v0 = (i * resY) + j;
				int v1 = ((i + 1) * resY) + j;

				int v2 = v1 + 1;
				int v3 = v0 + 1;

				polyConnects.push_back(v0);
				polyConnects.push_back(v1);
				polyConnects.push_back(v2);
				polyConnects.push_back(v3);

				polyCounts.push_back(4);
			}
		}


		out = zMesh(positions, polyCounts, polyConnects);;
		
		printf("\n scalarfieldMesh: %i %i %i", out.numVertices(), out.numEdges(), out.numPolygons());

		return out;
	}

	/*! \brief This method creates a vertex distance Field from the input vector of zVector positions.
	*
	*	\param	[in]	fieldMesh			- zMesh of the field.
	*	\param	[in]	points				- container of positions.
	*	\param	[out]	scalars				- container for storing scalar values.
	*	\since version 0.0.1
	*/
	
	void assignScalarsAsVertexDistance(zMesh &fieldMesh, vector<zVector> &points, vector<double> &scalars)
	{
		vector<double> out;

		vector<double> distVals;
		double dMin = 100000;
		double dMax = 0;;

		for (int i = 0; i < fieldMesh.numVertices(); i++)
		{
			distVals.push_back(10000);
		}

		for (int i = 0; i < points.size(); i++)
		{
			for (int j = 0; j < fieldMesh.numVertices(); j++)
			{
				double dist = fieldMesh.vertexPositions[j].distanceTo(points[i]);

				if (dist < distVals[j])
				{
					distVals[j] = dist;
				}
			}
		}

		for (int i = 0; i < distVals.size(); i++)
		{
			dMin = zMin(dMin, distVals[i]);
			dMax = zMax(dMax, distVals[i]);
		}

		for (int j = 0; j < fieldMesh.vertexColors.size(); j++)
		{
			double val = ofMap(distVals[j], dMin, dMax, 0.0, 1.0);
			fieldMesh.vertexColors[j] = (zColor(val, 0, 0, 1));
		}

		fieldMesh.computeFaceColorfromVertexColor();

		for (int j = 0; j < fieldMesh.faceColors.size(); j++)
		{
			out.push_back(fieldMesh.faceColors[j].r);
		}

		scalars = out;
	}

	/*! \brief This method creates a vertex distance Field from the input mesh vertex positions.
	*
	*	\param	[in]	fieldMesh		- input field mesh.
	*	\param	[in]	inMesh			- input mesh for distance calculations.
	*	\param	[in]	a				- input variable for distance function.
	*	\param	[in]	b				- input variable for distance function.
	*	\param	[out]	scalars			- container for storing scalar values.
	*	\since version 0.0.1
	*/

	void assignScalarsAsVertexDistance(zMesh &fieldMesh, zMesh &inMesh, double a, double b, vector<double> &scalars)
	{
		vector<double> out;

		// update values from edge distance
		for (int i = 0; i < fieldMesh.vertexPositions.size(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inMesh.numEdges(); j++)
			{

				int e0 = inMesh.edges[j].getVertex()->getVertexId();
				int e1 = inMesh.edges[j].getSym()->getVertex()->getVertexId();

				zVector closestPt;

				double r = minDist_Edge_Point_2D(fieldMesh.vertexPositions[i], inMesh.vertexPositions[e0], inMesh.vertexPositions[e1], closestPt);

				if (r < tempDist)
				{
					d = F_of_r(r, a, b);
					tempDist = r;
				}
			}

			out.push_back(d);
		
		}

		double dMin, dMax;
		getMinMaxOfScalars(out, dMin, dMax);

		for (int i = 0; i < out.size(); i++)out[i] = dMax - out[i];

		normaliseScalars(out);

		scalars = out;
	}

	/*! \brief This method creates a vertex distance Field from the input graph vertex positions.
	*
	*	\param	[in]	fieldMesh		- input field mesh.
	*	\param	[in]	inGraph			- input graph for distance calculations.
	*	\param	[in]	a				- input variable for distance function.
	*	\param	[in]	b				- input variable for distance function.
	*	\param	[out]	scalars			- container for storing scalar values.
	*	\since version 0.0.1
	*/

	void assignScalarsAsVertexDistance(zMesh &fieldMesh, zGraph &inGraph, double a, double b, vector<double> &scalars)
	{
		vector<double> out;

		// update values from edge distance
		for (int i = 0; i < fieldMesh.vertexPositions.size(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inGraph.numEdges(); j++)
			{

				int e0 = inGraph.edges[j].getVertex()->getVertexId();
				int e1 = inGraph.edges[j].getSym()->getVertex()->getVertexId();

				zVector closestPt;

				double r = minDist_Edge_Point_2D(fieldMesh.vertexPositions[i], inGraph.vertexPositions[e0], inGraph.vertexPositions[e1], closestPt);



				//printf("\n v: %i e: %i r:  %1.2f ", i,j, r);

				//printf("\n pt: %1.2f %1.2f %1.2f closestEdgePt: %1.2f %1.2f %1.2f r: %1.2f", mesh.vertexPositions.getValue(i).x, mesh.vertexPositions.getValue(i).y, mesh.vertexPositions.getValue(i).z, closestPt.x, closestPt.y, closestPt.z,r);

				if (r < tempDist)
				{

					d = F_of_r(r, a, b);

					//printf("\n F_of_r:  %1.4f ", F_of_r(r, a, b));

					tempDist = r;
				}

				//d += F_of_r(r, a, b);
				//printf("\n F_of_r:  %1.4f ", F_of_r(r, a, b));
			}

			out.push_back(d);
			//printf("\n scalars[buffer][i]:  %1.4f ", scalars[buffer][i]);
		}

		double dMin, dMax;
		getMinMaxOfScalars(out, dMin, dMax);

		for (int i = 0; i < out.size(); i++)out[i] = dMax - out[i];

		normaliseScalars(out);

		scalars = out;
	}

	/*! \brief This method creates a edge distance Field from the input mesh.
	*
	*	\param	[in]	fieldMesh		- input field mesh.
	*	\param	[in]	inMesh			- input mesh for distance calculations.
	*	\param	[in]	a				- input variable for distance function.
	*	\param	[in]	b				- input variable for distance function.
	*	\param	[out]	scalars			- container for storing scalar values.
	*	\since version 0.0.1
	*/
	
	void assignScalarsAsEdgeDistance(zMesh &fieldMesh, zMesh &inMesh, double a, double b, vector<double> &scalars)
	{
		vector<double> out;

		// update values from edge distance
		for (int i = 0; i < fieldMesh.vertexPositions.size(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inMesh.numEdges(); j++)
			{

				int e0 = inMesh.edges[j].getVertex()->getVertexId();
				int e1 = inMesh.edges[j].getSym()->getVertex()->getVertexId();

				zVector closestPt;

				double r = minDist_Edge_Point_2D(fieldMesh.vertexPositions[i], inMesh.vertexPositions[e0], inMesh.vertexPositions[e1], closestPt);


				if (r < tempDist)
				{

					d = F_of_r(r, a, b);

					tempDist = r;
				}
			}

			out.push_back(d);
		
		}

		double dMin, dMax;
		getMinMaxOfScalars(out, dMin, dMax);

		for (int i = 0; i < out.size(); i++)out[i] = dMax - out[i];

		normaliseScalars(out);

		scalars = out;
	}


	/*! \brief This method creates a edge distance Field from the input graph.
	*
	*	\param	[in]	fieldMesh		- input field mesh.
	*	\param	[in]	inGraph			- input graph for distance calculations.
	*	\param	[in]	a				- input variable for distance function.
	*	\param	[in]	b				- input variable for distance function.
	*	\param	[out]	scalars			- container for storing scalar values.
	*	\since version 0.0.1
	*/

	void assignScalarsAsEdgeDistance(zMesh &fieldMesh, zGraph &inGraph, double a, double b, vector<double> &scalars)
	{
		vector<double> out;

		// update values from edge distance
		for (int i = 0; i < fieldMesh.vertexPositions.size(); i++)
		{
			double d = 0.0;
			double tempDist = 10000;

			for (int j = 0; j < inGraph.numEdges(); j++)
			{

				int e0 = inGraph.edges[j].getVertex()->getVertexId();
				int e1 = inGraph.edges[j].getSym()->getVertex()->getVertexId();

				zVector closestPt;

				double r = minDist_Edge_Point_2D(fieldMesh.vertexPositions[i], inGraph.vertexPositions[e0], inGraph.vertexPositions[e1], closestPt);


				if (r < tempDist)
				{

					d = F_of_r(r, a, b);

					tempDist = r;
				}
			}

			out.push_back(d);

		}

		double dMin, dMax;
		getMinMaxOfScalars(out, dMin, dMax);

		for (int i = 0; i < out.size(); i++)out[i] = dMax - out[i];

		normaliseScalars(out);

		scalars = out;
	}

	/*! \brief This method creates a union of the fields at the input buffers and stores them in the result buffer.
	*
	*	\param	[in]	scalars0				- value of buffer.
	*	\param	[in]	scalars1				- value of buffer.
	*	\param	[in]	scalarsResult			- value of buffer to store the results.
	*	\since version 0.0.1
	*/
	
	void union_fields(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult)
	{
		vector<double> out;

		for (int i = 0; i < scalars0.size(); i++)
		{
			out.push_back(zMin(scalars0[i], scalars1[i]));
		}

		normaliseScalars(out);

		scalarsResult = out;
	}

	/*! \brief This method creates a subtraction of the fields at the input buffers and stores them in the result buffer.
	*
	*	\param	[in]	scalars0				- value of buffer.
	*	\param	[in]	scalars1				- value of buffer.
	*	\param	[in]	scalarsResult			- value of buffer to store the results.
	*	\since version 0.0.1
	*/
	
	void subtract_fields(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult)
	{
		vector<double> out;
		
		for (int i = 0; i < scalars0.size(); i++)
		{
			out.push_back(zMax(scalars0[i], -1 * scalars1[i]));
		}

		normaliseScalars(out);

		scalarsResult = out;
	}

	/*! \brief This method creates a intersect of the fields at the input buffers and stores them in the result buffer.
	*
	*	\param	[in]	buffer0				- value of buffer.
	*	\param	[in]	buffer1				- value of buffer.
	*	\param	[in]	res_buffer			- value of buffer to store the results.
	*	\since version 0.0.1
	*/

	void intersect_fields(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult)
	{
		vector<double> out;

		for (int i = 0; i < scalars0.size(); i++)
		{
			out.push_back(zMax(scalars0[i], scalars1[i]));
		}

		normaliseScalars(out);

		scalarsResult = out;
	}
	
	/*! \brief This method creates a difference of the fields at the input buffers and stores them in the result buffer.
	*
	*	\param	[in]	scalars0				- value of buffer.
	*	\param	[in]	scalars1				- value of buffer.
	*	\param	[in]	scalarsResult			- value of buffer to store the results.
	*	\since version 0.0.1
	*/
	
	void difference_fields(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult)
	{
		vector<double> out;

		intersect_fields(scalars0, scalars1, out);

		for (int i = 0; i < out.size(); i++)
		{
			out[i] *= -1;
		}


		scalarsResult = out;
	}

	/*! \brief This method uses an input plane to clip an existing scalar field.
	*
	*	\param	[in]	fieldMesh			- input field mesh.
	*	\param	[in]	scalars				- vector of scalar values. Need to be equivalent to number of mesh vertices.
	*	\param	[in]	clipPlane			- input zPlane used for clipping.
	*	\since version 0.0.1
	*/
	
	void clipwithPlane(zMesh &fieldMesh, vector<double>& scalars, zMatrixd& clipPlane)
	{
		for (int i = 0; i < fieldMesh.vertexPositions.size(); i++)
		{
			zVector O = fromMatrixColumn(clipPlane, 3);
			zVector Z = fromMatrixColumn(clipPlane, 2);

			zVector A = fieldMesh.vertexPositions[i] - O;
			double minDist_Plane = A * Z;
			minDist_Plane /= Z.length();

			// printf("\n dist %1.2f ", minDist_Plane);

			if (minDist_Plane > 0)
			{
				scalars[i] = 1;
			}

		}

	}


	/*! \brief This method updates the color values at the given field buffer.
	*
	*	\param	[in]	fieldMesh	- input field mesh.
	*	\param	[in]	scalars		- container of  scalar values.
	*	\since version 0.0.1
	*/
	
	void updateColors(zMesh &fieldMesh, vector<double>& scalars)
	{
		if (fieldMesh.vertexPositions.size() != scalars.size()) return;

		double dMax, dMin;
		getMinMaxOfScalars(scalars, dMin, dMax);

		for (int i = 0; i < scalars.size(); i++)
		{
			zColor col;

			double outMin = 0.00;
			double outMax = 1.00;
			double val = ofMap(scalars[i], dMin, dMax, outMin, outMax);

			col.r = val;
			fieldMesh.vertexColors[i] = col;
		}
	}

	//--------------------------
	//---- MARCHING SQUARES METHODS
	//--------------------------

	/*! \brief This method gets the isoline case based on the input vertex binary values.
	*
	*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
	*	\param	[in]	vertexBinary	- vertex binary values.
	*	\return			int				- case type.
	*	\since version 0.0.1
	*/

	int getIsolineCase(bool vertexBinary[4])
	{
		int out = -1;

		if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 0;

		if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 1;

		if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 2;

		if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 3;

		if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 4;

		if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 5;

		if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 6;

		if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 7;

		if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 8;

		if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 9;

		if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 10;

		if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 11;

		if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 12;

		if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 13;

		if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 14;

		if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 15;

		return out;
	}

	/*! \brief This method gets the isoline case based on the input vertex ternary values.
	*
	*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
	*	\param	[in]	vertexTernary	- vertex ternary values.
	*	\return			int				- case type.
	*	\since version 0.0.1
	*/	
	
	int getIsobandCase(int vertexTernary[4])
	{
		int out = -1;

		// No Contour
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 0;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 1;


		// Single Triangle

		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 2;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 3;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 4;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 5;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 6;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 7;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 8;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 9;

		// Single Trapezoid

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 10;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 11;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 12;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 13;


		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 14;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 15;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 16;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 17;

		// Single Rectangle

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 18;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 19;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 20;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 21;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 22;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 23;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 24;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 25;


		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 26;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 27;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 28;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 29;

		// Single Square

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 30;

		// Single Pentagon

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 31;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 32;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 33;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 34;


		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 35;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 36;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 37;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 38;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 39;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 40;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 41;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 42;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 43;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 44;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 45;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 46;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 47;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 48;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 49;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 50;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 51;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 52;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 53;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 54;

		// Single Hexagon

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 55;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 56;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 57;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 58;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 59;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 60;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 61;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 62;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 63;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 64;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 65;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 66;


		// Saddles:  1 or 2 polygon

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 67;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 68;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 69;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 70;

		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 71;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 72;

		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 73;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 74;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 75;
		if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 76;

		if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 77;
		if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 78;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 79;
		if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 80;

		return out;
	}

	/*! \brief This method return the contour position  given 2 input positions at the input field threshold.
	*	
	*	\param	[in]	threshold		- field threshold.
	*	\param	[in]	vertex_lower	- lower threshold position.
	*	\param	[in]	vertex_higher	- higher threshold position.
	*	\param	[in]	thresholdLow	- field threshold domain minimum.
	*	\param	[in]	thresholdHigh	- field threshold domain maximum.
	*	\since version 0.0.1
	*/

	zVector getContourPosition(double &threshold, zVector& vertex_lower, zVector& vertex_higher, double& thresholdLow, double& thresholdHigh)
	{

		double scaleVal = ofMap(threshold, thresholdLow, thresholdHigh, 0.0, 1.0);

		zVector e = vertex_higher - vertex_lower;
		double edgeLen = e.length();
		e.normalize();

		return (vertex_lower + (e * edgeLen *scaleVal));
	}

	/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
	*
	*	\param	[in]	faceId			- input face index.
	*	\param	[in]	faceId			- input field mesh.
	*	\param	[in]	positions		- container of positions of the computed polygon.
	*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
	*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
	*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
	*	\param	[in]	threshold		- field threshold.
	*	\param	[in]	invertMesh	- true if inverted mesh is required.
	*	\since version 0.0.1
	*/
	
	void getIsolinePoly(int& faceId, zMesh &fieldMesh, vector<zVector> &positions, vector<int> &polyConnects, vector<int> &polyCounts, unordered_map <string, int> &positionVertex, double &threshold, bool invertMesh)
	{
		vector<int> fVerts;
		fieldMesh.getVertices(faceId, zFaceData, fVerts);

		if (fVerts.size() != 4) return;

		// chk if all the face vertices are below the threshold
		bool vertexBinary[4];
		double averageScalar = 0;

		for (int j = 0; j < fVerts.size(); j++)
		{
			if (fieldMesh.vertexColors[fVerts[j]].r < threshold)
			{
				vertexBinary[j] = (invertMesh) ? false : true;
			}
			else vertexBinary[j] = (invertMesh) ? true : false;

			averageScalar += fieldMesh.vertexColors[fVerts[j]].r;
		}

		averageScalar /= fVerts.size();

		int MS_case = getIsolineCase(vertexBinary);

		vector<zVector> newPositions;
		vector<zVector> newPositions2;


		// CASE 0
		if (MS_case == 0)
		{
			for (int j = 0; j < fVerts.size(); j++)
			{
				newPositions.push_back(fieldMesh.vertexPositions[fVerts[j]]);
			}

		}

		// CASE 1
		if (MS_case == 1)
		{
			zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
			double s0 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
			double s1 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

			v1 = fieldMesh.vertexPositions[fVerts[3]];
			s1 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);


		}

		// CASE 2
		if (MS_case == 2)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
			double s0 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
			double s1 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[2]];
			s0 = fieldMesh.vertexColors[fVerts[2]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

		}

		// CASE 3
		if (MS_case == 3)
		{
			zVector v0 = fieldMesh.vertexPositions[fVerts[3]];
			double s0 = fieldMesh.vertexColors[fVerts[3]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[0]];
			double s1 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[2]];
			s0 = fieldMesh.vertexColors[fVerts[2]].r;
			v1 = fieldMesh.vertexPositions[fVerts[1]];
			s1 = fieldMesh.vertexColors[fVerts[1]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

		}

		// CASE 4
		if (MS_case == 4)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
			double s0 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[2]];
			double s1 = fieldMesh.vertexColors[fVerts[2]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[3]];
			s0 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

		}

		// CASE 5
		if (MS_case == 5)
		{
			// SADDLE CASE 

			int SaddleCase = 1/*(averageScalar < threshold) ? 0 : 1*/;
			if (invertMesh) SaddleCase = 0/*(averageScalar < threshold) ? 1 : 0*/;

			if (SaddleCase == 0)
			{
				// hex
				zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
				double s0 = fieldMesh.vertexColors[fVerts[1]].r;
				zVector v1 = fieldMesh.vertexPositions[fVerts[0]];
				double s1 = fieldMesh.vertexColors[fVerts[0]].r;
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

				v1 = fieldMesh.vertexPositions[fVerts[2]];
				s1 = fieldMesh.vertexColors[fVerts[2]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);


				v0 = fieldMesh.vertexPositions[fVerts[3]];
				s0 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

				v1 = fieldMesh.vertexPositions[fVerts[0]];
				s1 = fieldMesh.vertexColors[fVerts[0]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}
			if (SaddleCase == 1)
			{
				// tri 1
				zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
				double s0 = fieldMesh.vertexColors[fVerts[1]].r;
				zVector v1 = fieldMesh.vertexPositions[fVerts[0]];
				double s1 = fieldMesh.vertexColors[fVerts[0]].r;
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

				v1 = fieldMesh.vertexPositions[fVerts[2]];
				s1 = fieldMesh.vertexColors[fVerts[2]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2
				v0 = fieldMesh.vertexPositions[fVerts[3]];
				s0 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fieldMesh.vertexPositions[fVerts[3]]);

				v1 = fieldMesh.vertexPositions[fVerts[0]];
				s1 = fieldMesh.vertexColors[fVerts[0]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 6
		if (MS_case == 6)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
			double s0 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
			double s1 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[3]];
			s0 = fieldMesh.vertexColors[fVerts[3]].r;
			v1 = fieldMesh.vertexPositions[fVerts[2]];
			s1 = fieldMesh.vertexColors[fVerts[2]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);

		}


		// CASE 7
		if (MS_case == 7)
		{

			zVector v0 = fieldMesh.vertexPositions[fVerts[3]];
			double s0 = fieldMesh.vertexColors[fVerts[3]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[2]];
			double s1 = fieldMesh.vertexColors[fVerts[2]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[3]]);


			v1 = fieldMesh.vertexPositions[fVerts[0]];
			s1 = fieldMesh.vertexColors[fVerts[0]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 8
		if (MS_case == 8)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[2]];
			double s0 = fieldMesh.vertexColors[fVerts[2]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[3]];
			double s1 = fieldMesh.vertexColors[fVerts[3]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[0]];
			s0 = fieldMesh.vertexColors[fVerts[0]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 9
		if (MS_case == 9)
		{
			zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
			double s0 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[0]];
			double s1 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			v0 = fieldMesh.vertexPositions[fVerts[2]];
			s0 = fieldMesh.vertexColors[fVerts[2]].r;
			v1 = fieldMesh.vertexPositions[fVerts[3]];
			s1 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 10
		if (MS_case == 10)
		{
			int SaddleCase = 1 /*(averageScalar < threshold) ? 0 : 1*/;
			if (invertMesh) SaddleCase = 0 /*(averageScalar < threshold) ? 1 : 0*/;

			if (SaddleCase == 0)
			{
				// hex
				newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

				zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
				double s0 = fieldMesh.vertexColors[fVerts[0]].r;
				zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
				double s1 = fieldMesh.vertexColors[fVerts[1]].r;
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);


				v0 = fieldMesh.vertexPositions[fVerts[2]];
				s0 = fieldMesh.vertexColors[fVerts[2]].r;
				v1 = fieldMesh.vertexPositions[fVerts[1]];
				s1 = fieldMesh.vertexColors[fVerts[1]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

				v1 = fieldMesh.vertexPositions[fVerts[3]];
				s1 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fieldMesh.vertexPositions[fVerts[0]];
				s0 = fieldMesh.vertexColors[fVerts[0]].r;
				v1 = fieldMesh.vertexPositions[fVerts[3]];
				s1 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);
			}

			if (SaddleCase == 1)
			{
				// tri 1

				newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

				zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
				double s0 = fieldMesh.vertexColors[fVerts[0]].r;
				zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
				double s1 = fieldMesh.vertexColors[fVerts[1]].r;
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v1 = fieldMesh.vertexPositions[fVerts[3]];
				s1 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				// tri 2
				v0 = fieldMesh.vertexPositions[fVerts[2]];
				s0 = fieldMesh.vertexColors[fVerts[2]].r;
				v1 = fieldMesh.vertexPositions[fVerts[1]];
				s1 = fieldMesh.vertexColors[fVerts[1]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);

				newPositions2.push_back(fieldMesh.vertexPositions[fVerts[2]]);

				v1 = fieldMesh.vertexPositions[fVerts[3]];
				s1 = fieldMesh.vertexColors[fVerts[3]].r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions2.push_back(pos);
			}


		}

		// CASE 11
		if (MS_case == 11)
		{

			zVector v0 = fieldMesh.vertexPositions[fVerts[2]];
			double s0 = fieldMesh.vertexColors[fVerts[2]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
			double s1 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			//printf("\n pos 1 : %1.4f %1.4f %1.4f ", pos.x, pos.y, pos.z);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[2]]);

			//pos = fieldMesh.vertexPositions[fVerts[2]);
			//printf("\n pos 2 : %1.4f %1.4f %1.4f ", pos.x, pos.y, pos.z);

			v1 = fieldMesh.vertexPositions[fVerts[3]];
			s1 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			//printf("\n pos 3 :%1.4f %1.4f %1.4f \n", pos.x, pos.y, pos.z);

		}

		// CASE 12
		if (MS_case == 12)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
			double s0 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[2]];
			double s1 = fieldMesh.vertexColors[fVerts[2]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v0 = fieldMesh.vertexPositions[fVerts[0]];
			s0 = fieldMesh.vertexColors[fVerts[0]].r;
			v1 = fieldMesh.vertexPositions[fVerts[3]];
			s1 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 13
		if (MS_case == 13)
		{

			zVector v0 = fieldMesh.vertexPositions[fVerts[1]];
			double s0 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[0]];
			double s1 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			newPositions.push_back(fieldMesh.vertexPositions[fVerts[1]]);

			v1 = fieldMesh.vertexPositions[fVerts[2]];
			s1 = fieldMesh.vertexColors[fVerts[2]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 14
		if (MS_case == 14)
		{
			newPositions.push_back(fieldMesh.vertexPositions[fVerts[0]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[0]];
			double s0 = fieldMesh.vertexColors[fVerts[0]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[1]];
			double s1 = fieldMesh.vertexColors[fVerts[1]].r;
			zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

			v1 = fieldMesh.vertexPositions[fVerts[3]];
			s1 = fieldMesh.vertexColors[fVerts[3]].r;
			pos = (getContourPosition(threshold, v0, v1, s0, s1));
			newPositions.push_back(pos);

		}

		// CASE 15
		if (MS_case == 15)
		{
			// No Veritices to be added 

		}







		// check for edge lengths


		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				int next = (i + 1) % newPositions.size();

				if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
				{
					newPositions.erase(newPositions.begin() + i);
				}
			}
		}


		// compute poly 
		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				zVector p0 = newPositions[i];
				int v0;

				bool vExists = vertexExists(positionVertex, p0, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions.size());
		}


		// only if there are 2 tris Case : 5,10

		// Edge Length Check

		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				int next = (i + 1) % newPositions2.size();


				if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
				{
					newPositions2.erase(newPositions.begin() + i);
				}

			}
		}


		// compute poly 
		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				zVector p0 = newPositions2[i];
				int v0;

				bool vExists = vertexExists(positionVertex, p0, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions2.size());
		}



	}

	/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
	*
	*	\param	[in]	faceId			- input face index.
	*	\param	[in]	faceId			- input field mesh.
	*	\param	[in]	positions		- container of positions of the computed polygon.
	*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
	*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
	*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
	*	\param	[in]	thresholdLow	- field threshold domain minimum.
	*	\param	[in]	thresholdHigh	- field threshold domain maximum.
	*	\since version 0.0.1
	*/
	
	void getIsobandPoly(int& faceId, zMesh &fieldMesh, vector<zVector> &positions, vector<int> &polyConnects, vector<int> &polyCounts, unordered_map <string, int> &positionVertex, double &thresholdLow, double &thresholdHigh)	
	{
		vector<int> fVerts;
		fieldMesh.getVertices(faceId, zFaceData, fVerts);

		//printf("\n fVs: %i ", fVerts.size());

		if (fVerts.size() != 4) return;

		// chk if all the face vertices are below the threshold
		int vertexTernary[4];
		double averageScalar = 0;

		for (int j = 0; j < fVerts.size(); j++)
		{
			if (fieldMesh.vertexColors[fVerts[j]].r <= thresholdLow)
			{
				vertexTernary[j] = 0;
			}

			else if (fieldMesh.vertexColors[fVerts[j]].r >= thresholdHigh)
			{
				vertexTernary[j] = 2;
			}
			else vertexTernary[j] = 1;

			averageScalar += fieldMesh.vertexColors[fVerts[j]].r;
		}

		averageScalar /= fVerts.size();

		int MS_case = getIsobandCase(vertexTernary);

		vector<zVector> newPositions;
		vector<zVector> newPositions2;

		//printf("\n MS_case : %i ", MS_case);
		//if (MS_case < 72 || MS_case > 80) return;




		// No Contours CASE 0 & 1
		if (MS_case == 0 || MS_case == 1)
		{
			// NO Vertices to be added
		}

		// Single Triangle CASE 2 to 9 
		if (MS_case >= 2 && MS_case <= 9)
		{
			int startID = -1;
			double threshold = thresholdHigh;

			if (MS_case == 2 || MS_case == 6)startID = 0;
			if (MS_case == 3 || MS_case == 7)startID = 1;
			if (MS_case == 4 || MS_case == 8)startID = 2;
			if (MS_case == 5 || MS_case == 9)startID = 3;

			if (MS_case > 5) threshold = thresholdLow;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();


			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;

			zVector v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			double s1 = fieldMesh.vertexColors[fVerts[prevID]].r;

			zVector pos0 = (getContourPosition(threshold, v0, v1, s0, s1));


			zVector pos1 = fieldMesh.vertexPositions[fVerts[startID]];


			v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			s1 = fieldMesh.vertexColors[fVerts[nextID]].r;

			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));


			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);

		}


		// Single Trapezoid CASE 10 to 17

		if (MS_case >= 10 && MS_case <= 17)
		{
			int startID = -1;

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case == 10 || MS_case == 14) startID = 0;
			if (MS_case == 11 || MS_case == 15) startID = 1;
			if (MS_case == 12 || MS_case == 16) startID = 2;
			if (MS_case == 13 || MS_case == 17) startID = 3;

			if (MS_case > 13)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;

			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;

			zVector pos0 = (getContourPosition(threshold0, v0, v1, s0, s1));

			zVector pos1 = (getContourPosition(threshold1, v0, v1, s0, s1));


			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;

			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

			zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));


			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);

		}


		// Single Rectangle CASE 18 to 29
		if (MS_case >= 18 && MS_case <= 25)
		{
			int startID = -1;
			double threshold = thresholdLow;
			if (MS_case > 21) threshold = thresholdHigh;

			if (MS_case == 18 || MS_case == 22) startID = 0;
			if (MS_case == 19 || MS_case == 23) startID = 1;
			if (MS_case == 20 || MS_case == 24) startID = 2;
			if (MS_case == 21 || MS_case == 25) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = (fieldMesh.vertexPositions[fVerts[startID]]);
			zVector pos1 = (fieldMesh.vertexPositions[fVerts[nextID]]);

			zVector v0 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s0 = fieldMesh.vertexColors[fVerts[nextID]].r;

			zVector v1 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[next_nextID]].r;

			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));



			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;

			zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);

		}

		// Single Rectangle CASE 26 to 29
		if (MS_case >= 26 && MS_case <= 29)
		{
			int startID = -1;

			if (MS_case == 26) startID = 0;
			if (MS_case == 27) startID = 1;
			if (MS_case == 28) startID = 2;
			if (MS_case == 29) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			double s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos0 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

			v0 = fieldMesh.vertexPositions[fVerts[nextID]];
			s0 = fieldMesh.vertexColors[fVerts[nextID]].r;
			v1 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s1 = fieldMesh.vertexColors[fVerts[next_nextID]].r;

			zVector pos1 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
			zVector pos2 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
		}

		// Single Square CASE 30
		if (MS_case == 30)
		{
			for (int j = 0; j< fVerts.size(); j++)
				newPositions.push_back(fieldMesh.vertexPositions[fVerts[j]]);
		}

		// Single Pentagon CASE 31 to 38
		if (MS_case >= 31 && MS_case <= 38)
		{
			int startID = -1;

			double threshold = thresholdHigh;
			if (MS_case > 34)threshold = thresholdLow;

			if (MS_case == 31 || MS_case == 35) startID = 0;
			if (MS_case == 32 || MS_case == 36) startID = 1;
			if (MS_case == 33 || MS_case == 37) startID = 2;
			if (MS_case == 34 || MS_case == 38) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];
			zVector pos1 = fieldMesh.vertexPositions[fVerts[nextID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s0 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = getContourPosition(threshold, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[prevID]];
			s0 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

			zVector pos4 = fieldMesh.vertexPositions[fVerts[prevID]];;

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Pentagon CASE 39 to 46
		if (MS_case >= 39 && MS_case <= 46)
		{
			int startID = -1;

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case > 42)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 39 || MS_case == 43) startID = 3;
			if (MS_case == 40 || MS_case == 44) startID = 2;
			if (MS_case == 41 || MS_case == 45) startID = 1;
			if (MS_case == 42 || MS_case == 46) startID = 0;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s0 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Pentagon CASE 47 to54
		if (MS_case >= 47 && MS_case <= 54)
		{
			int startID = -1;

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case > 50)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 47 || MS_case == 51) startID = 3;
			if (MS_case == 48 || MS_case == 52) startID = 2;
			if (MS_case == 49 || MS_case == 53) startID = 1;
			if (MS_case == 50 || MS_case == 54) startID = 0;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);



			v0 = fieldMesh.vertexPositions[fVerts[nextID]];
			s0 = fieldMesh.vertexColors[fVerts[nextID]].r;
			v1 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s1 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));

			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos4 = getContourPosition(threshold0, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
		}

		// Single Hexagon CASE 55 to 62
		if (MS_case >= 55 && MS_case <= 62)
		{
			int startID = -1;

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case > 58)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;

			}

			if (MS_case == 55 || MS_case == 59) startID = 0;
			if (MS_case == 56 || MS_case == 60) startID = 1;
			if (MS_case == 57 || MS_case == 61) startID = 2;
			if (MS_case == 58 || MS_case == 62) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];
			zVector pos1 = fieldMesh.vertexPositions[fVerts[nextID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s0 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = getContourPosition(threshold1, v0, v1, s0, s1);



			v0 = fieldMesh.vertexPositions[fVerts[prevID]];
			s0 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));

			v1 = fieldMesh.vertexPositions[fVerts[startID]];
			s1 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos5 = getContourPosition(threshold0, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
			newPositions.push_back(pos5);
		}

		// Single Hexagon CASE 63 to 66
		if (MS_case >= 63 && MS_case <= 66)
		{
			int startID = -1;


			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case % 2 == 0)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			if (MS_case == 63 || MS_case == 64) startID = 0;
			if (MS_case == 65 || MS_case == 66) startID = 1;



			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s0 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));

			zVector pos3 = fieldMesh.vertexPositions[fVerts[next_nextID]];

			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);

			newPositions.push_back(pos0);
			newPositions.push_back(pos1);
			newPositions.push_back(pos2);
			newPositions.push_back(pos3);
			newPositions.push_back(pos4);
			newPositions.push_back(pos5);
		}

		// SADDLE CASE 67 to 68 : 8 Sides
		if (MS_case >= 67 && MS_case <= 68)
		{

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case % 2 == 0)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int SaddleCase = -1;;

			if (averageScalar < thresholdLow) SaddleCase = 0;
			else if (averageScalar > thresholdHigh) SaddleCase = 2;
			else SaddleCase = 1;

			int startID = 0;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s0 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
			zVector pos3 = getContourPosition(threshold0, v0, v1, s0, s1);

			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));
			zVector pos5 = (getContourPosition(threshold1, v0, v1, s0, s1));

			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos6 = getContourPosition(threshold1, v0, v1, s0, s1);
			zVector pos7 = (getContourPosition(threshold0, v0, v1, s0, s1));


			if (SaddleCase == 0)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

				// quad 2
				newPositions2.push_back(pos4);
				newPositions2.push_back(pos5);
				newPositions2.push_back(pos6);
				newPositions2.push_back(pos7);
			}


			if (SaddleCase == 2)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos6);
				newPositions.push_back(pos7);

				// quad 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
				newPositions2.push_back(pos5);
			}


			if (SaddleCase == 1)
			{
				// octagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);
				newPositions.push_back(pos7);
			}

		}

		// SADDLE CASE 69 to 72 : 6 Sides
		if (MS_case >= 69 && MS_case <= 72)
		{

			double threshold = thresholdLow;

			if (MS_case > 70)
			{
				threshold = thresholdHigh;
			}

			int SaddleCase = -1;;

			if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
			else SaddleCase = 0;

			int startID = (MS_case % 2 == 0) ? 1 : 0;

			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();

			zVector pos0 = fieldMesh.vertexPositions[fVerts[startID]];

			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos1 = getContourPosition(threshold, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s0 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));

			zVector pos3 = fieldMesh.vertexPositions[fVerts[next_nextID]];

			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos4 = (getContourPosition(threshold, v0, v1, s0, s1));


			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos5 = getContourPosition(threshold, v0, v1, s0, s1);


			if (SaddleCase == 0)
			{
				// tri 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos5);

				// tri 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
			}


			if (SaddleCase == 1)
			{
				// hexagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);

			}

		}

		// SADDLE CASE 73 to 80 : 7 Sides
		if (MS_case >= 73 && MS_case <= 80)
		{

			double threshold0 = thresholdLow;
			double threshold1 = thresholdHigh;

			if (MS_case > 76)
			{
				threshold0 = thresholdHigh;
				threshold1 = thresholdLow;
			}

			int SaddleCase = -1;;

			if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
			else SaddleCase = 0;

			int startID = -1;

			if (MS_case == 73 || MS_case == 77) startID = 0;
			if (MS_case == 74 || MS_case == 78) startID = 2;
			if (MS_case == 75 || MS_case == 79) startID = 1;
			if (MS_case == 76 || MS_case == 80) startID = 3;


			int nextID = (startID + 1) % fVerts.size();
			int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
			int next_nextID = (nextID + 1) % fVerts.size();



			zVector v0 = fieldMesh.vertexPositions[fVerts[startID]];
			double s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector v1 = fieldMesh.vertexPositions[fVerts[nextID]];
			double s1 = fieldMesh.vertexColors[fVerts[nextID]].r;
			zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
			zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

			v0 = fieldMesh.vertexPositions[fVerts[next_nextID]];
			s0 = fieldMesh.vertexColors[fVerts[next_nextID]].r;
			zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

			zVector pos3 = fieldMesh.vertexPositions[fVerts[next_nextID]];

			v1 = fieldMesh.vertexPositions[fVerts[prevID]];
			s1 = fieldMesh.vertexColors[fVerts[prevID]].r;
			zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));


			v0 = fieldMesh.vertexPositions[fVerts[startID]];
			s0 = fieldMesh.vertexColors[fVerts[startID]].r;
			zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);
			zVector pos6 = getContourPosition(threshold0, v0, v1, s0, s1);

			if (SaddleCase == 0)
			{
				// quad 1
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);

				// tri 2
				newPositions2.push_back(pos2);
				newPositions2.push_back(pos3);
				newPositions2.push_back(pos4);
			}


			if (SaddleCase == 1)
			{
				// heptagon
				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
				newPositions.push_back(pos6);

			}

		}

		// check for edge lengths

		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				int next = (i + 1) % newPositions.size();

				if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
				{
					newPositions.erase(newPositions.begin() + i);

				}
			}
		}


		// compute poly 
		if (newPositions.size() >= 3)
		{
			for (int i = 0; i < newPositions.size(); i++)
			{
				zVector p0 = newPositions[i];
				int v0;

				bool vExists = vertexExists(positionVertex, p0, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions.size());
		}


		// only if there are 2 tris Case : 5,10

		// Edge Length Check


		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				int next = (i + 1) % newPositions2.size();


				if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
				{
					newPositions2.erase(newPositions.begin() + i);


				}

			}
		}


		// compute poly 
		if (newPositions2.size() >= 3)
		{
			for (int i = 0; i < newPositions2.size(); i++)
			{
				zVector p0 = newPositions2[i];
				int v0;

				bool vExists = vertexExists(positionVertex, p0, v0);

				if (!vExists)
				{
					v0 = positions.size();
					positions.push_back(p0);

					string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
					positionVertex[hashKey] = v0;
				}

				polyConnects.push_back(v0);
			}

			polyCounts.push_back(newPositions2.size());
		}

	}

	/*! \brief This method creates a isocontour graph from the input field mesh at the given field threshold.
	*
	*	\param	[in]	fieldMesh	- input field mesh.
	*	\param	[in]	threshold	- field threshold.
	*	\return			zGraph		- contour graph.
	*	\since version 0.0.1
	*/
	zGraph getIsocontour(zMesh &fieldMesh, double threshold = 0.5)
	{
		vector<double> scalarsValues;

		vector<zVector> pos;
		vector<int> edgeConnects;

		vector<int> edgetoIsoGraphVertexId;

		for (int i = 0; i < fieldMesh.numVertices(); i++)
		{
			scalarsValues.push_back(fieldMesh.vertexColors[i].r);
		}

		// compute positions
		for (int i = 0; i < fieldMesh.edgeActive.size(); i += 2)
		{
			edgetoIsoGraphVertexId.push_back(-1);
			edgetoIsoGraphVertexId.push_back(-1);

			if (fieldMesh.edgeActive[i])
			{
				int eV0 = fieldMesh.edges[i].getVertex()->getVertexId();
				int eV1 = fieldMesh.edges[i + 1].getVertex()->getVertexId();

				double scalar_lower = (scalarsValues[eV0] <= scalarsValues[eV1]) ? scalarsValues[eV0] : scalarsValues[eV1];
				double scalar_higher = (scalarsValues[eV0] <= scalarsValues[eV1]) ? scalarsValues[eV1] : scalarsValues[eV0];;

				bool chkSplitEdge = (scalar_lower <= threshold && scalar_higher > threshold) ? true : false;

				if (chkSplitEdge)
				{
					// calculate split point

					int scalar_lower_vertId = (scalarsValues[eV0] <= scalarsValues[eV1]) ? eV0 : eV1;
					int scalar_higher_vertId = (scalarsValues[eV0] <= scalarsValues[eV1]) ? eV1 : eV0;

					zVector scalar_lower_vertPos = fieldMesh.vertexPositions[scalar_lower_vertId];
					zVector scalar_higher_vertPos = fieldMesh.vertexPositions[scalar_higher_vertId];

					double scaleVal = ofMap(threshold, scalar_lower, scalar_higher, 0.0, 1.0);

					zVector e = scalar_higher_vertPos - scalar_lower_vertPos;
					double eLen = e.length();

					e.normalize();

					zVector newPos = scalar_lower_vertPos + (e * eLen * scaleVal);
					pos.push_back(newPos);

					// map edge to isographVertex
					edgetoIsoGraphVertexId[i] = pos.size() - 1;
					edgetoIsoGraphVertexId[i + 1] = pos.size() - 1;

				}
			}
		}

		// compute edgeConnects
		for (int i = 0; i < fieldMesh.faceActive.size(); i++)
		{
			if (fieldMesh.faceActive[i])
			{
				vector<int> fEdges;
				fieldMesh.getEdges(i, zFaceData, fEdges);
				vector<int> tempConnects;

				for (int j = 0; j < fEdges.size(); j++)
				{
					if (edgetoIsoGraphVertexId[fEdges[j]] != -1)
						tempConnects.push_back(edgetoIsoGraphVertexId[fEdges[j]]);
				}

				//printf("\n face %i | %i ", i, tempConnects.size());

				if (tempConnects.size() == 2)
				{
					edgeConnects.push_back(tempConnects[0]);
					edgeConnects.push_back(tempConnects[1]);
				}
			}
		}

		return zGraph(pos, edgeConnects);



	}

	/*! \brief This method creates a isoline mesh from the input field mesh at the given field threshold.
	*
	*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
	*	\param	[in]	fieldMesh	- input field mesh.
	*	\param	[in]	threshold	- field threshold.
	*	\param	[in]	invertMesh	- true if inverted mesh is required.
	*	\return			zMesh		- isoline mesh.
	*	\since version 0.0.1
	*/

	zMesh getIsolineMesh(zMesh &fieldMesh, double threshold = 0.5, bool invertMesh = false)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unordered_map <string, int> positionVertex;



		for (int i = 0; i < fieldMesh.numPolygons(); i++)
		{
			getIsolinePoly(i, fieldMesh, positions, polyConnects, polyCounts, positionVertex, threshold, invertMesh);
		}


		out = zMesh(positions, polyCounts, polyConnects);;

		return out;
	}


	/*! \brief This method creates a isoband mesh from the input field mesh at the given field threshold.
	*
	*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
	*	\param	[in]	fieldMesh		- input field mesh.
	*	\param	[in]	thresholdLow	- field threshold domain minimum.
	*	\param	[in]	thresholdHigh	- field threshold domain maximum.
	*	\param	[in]	invertMesh		- true if inverted mesh is required.
	*	\return			zMesh			- isoband mesh.
	*	\since version 0.0.1
	*/

	zMesh getIsobandMesh(zMesh &fieldMesh, double thresholdLow = 0.2, double thresholdHigh = 0.5, bool invertMesh = false)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		unordered_map <string, int> positionVertex;


		if (invertMesh)
		{
			zMesh m1 = getIsolineMesh(fieldMesh, (thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, false);
			zMesh m2 = getIsolineMesh(fieldMesh, (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow, true);

			/*if (m1.numVertices() > 0 && m2.numVertices() > 0)
			{
				out = combineDisjointMesh(m1, m2);

				return out;
			}

			else*/ if (m1.numVertices() > 0) return m1;

			else if (m2.numVertices() > 0) return m2;


		}

		if (!invertMesh)
		{
			for (int i = 0; i < fieldMesh.numPolygons(); i++)
			{
				getIsobandPoly(i, fieldMesh, positions, polyConnects, polyCounts, positionVertex, (thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow);
			}

			out = zMesh(positions, polyCounts, polyConnects);;		

			return out;
		}


	}

	



	/** @}*/

	/** @}*/

	/** @}*/
}