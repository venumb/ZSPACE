// This file is part of zspace, a simple C++ collection of geometr_y data-structures & algorithms, 
// data anal_ysis & visualization framework.
//
// Cop_yright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a cop_y of the MIT License was not distributed with this file, _you can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//


#include<headers/zArchGeom/zAgColumn.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgColumn::zAgColumn(){ }

	ZSPACE_INLINE zAgColumn::zAgColumn( zVector&_position, zVectorArray&_axis, zBoolArray&_axisAttributes, vector<zBoundary>&_boundaryArray, float _height)
	{
		position = _position;
		axis = _axis;
		axisAttributes = _axisAttributes;
		boundaryArray = _boundaryArray;

		z = zVector(0, 0, -1);

		height = 4.5f; //TODO update with variable passed
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgColumn::~zAgColumn() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zAgColumn::createRhwc()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		snapSlabpoints.assign(4, zVectorArray());

		//snapSlabpoints.assign(axis.size(), zVectorArray());
		int numPoints = 0;
		for (int i = 0; i < axis.size(); i++)
		{
			zVectorArray snap;

			if (boundaryArray[i] == zBoundary::zInterior)
			{
				///////top faces for slab snap
				snap.assign(8, zVector());

				x = axis[i];
				y = axis[(i + 1) % axis.size()];
				float xSpan = fabs(axis[i].length());
				float ySpan = fabs(axis[(i + 1) % axis.size()].length());

				x.normalize();
				y.normalize();
				////

				snap[0] = x * 0.1 * xSpan;
				snap[3] = y * 0.1 * ySpan;
				snap[1] = snap[0] + snap[3]; //center
				snap[2] = snap[0] + snap[3]; //center

				snap[0] += position + zVector(0, 0, -0.2);
				snap[2] += position + zVector(0, 0, -0.2);
				snap[1] += position + zVector(0, 0, -0.2);
				snap[3] += position + zVector(0, 0, -0.2);
				/////

				snap[4] = x * 0.3 * xSpan + zVector(0, 0, -0.2);
				snap[7] = y * 0.3 * ySpan + zVector(0, 0, -0.2);
				snap[5] = (x * 0.3 * xSpan) + (y * 0.3 * ySpan) + zVector(0, 0, -0.2); //center
				snap[6] = (x * 0.3 * xSpan) + (y * 0.3 * ySpan) + zVector(0, 0, -0.2); //center

				snap[6] += position;
				snap[5] += position;
				snap[4] += position;
				snap[7] += position;
				/////

				snapSlabpoints[i] = snap;

				for (auto v : snap)
				{
					pointArray.push_back(v);
				}

				//////////////////////////////////
				//bottom faces
				/////

				zPointArray bottomPoints;
				bottomPoints.assign(16, zVector());

				////breat detached edge repeat points
				bottomPoints[0] = snap[4];
				bottomPoints[3] = snap[7];
				bottomPoints[1] = snap[5]; //center
				bottomPoints[2] = snap[6]; //center
				/////

				bottomPoints[4] = x * 0.09 * xSpan + zVector(0, 0, -0.8) + position;
				bottomPoints[7] = y * 0.09 * ySpan + zVector(0, 0, -0.8) + position;
				bottomPoints[5] = (x * 0.09 * xSpan) + (y * 0.09 * ySpan) + zVector(0, 0, -0.8) + position; //center
				bottomPoints[6] = (x * 0.09 * xSpan) + (y * 0.09 * ySpan) + zVector(0, 0, -0.8) + position; //center
				/////

				bottomPoints[8] = x * 0.020 * xSpan + zVector(0, 0, -1.85) + position;
				bottomPoints[11] = y * 0.020 * ySpan + zVector(0, 0, -1.85) + position;
				bottomPoints[9] = (x * 0.020 * xSpan) + (y * 0.020 * ySpan) + zVector(0, 0, -1.85) + position; //center
				bottomPoints[10] = (x * 0.020 * xSpan) + (y * 0.020 * ySpan) + zVector(0, 0, -1.85) + position; //center
				/////

				bottomPoints[12] = x * 0.015 * xSpan + zVector(0, 0, -height) + position;
				bottomPoints[15] = y * 0.015 * ySpan + zVector(0, 0, -height) + position;
				bottomPoints[14] = (x * 0.015 * xSpan) + (y * 0.015 * ySpan) + zVector(0, 0, -height) + position; //center
				bottomPoints[13] = (x * 0.015 * xSpan) + (y * 0.015 * ySpan) + zVector(0, 0, -height) + position; //center
				/////////////////////////////////////////////////////

				for (auto v : bottomPoints)
				{
					pointArray.push_back(v);
				}

				/////////////////////
				//polyconnect and polycount quads
				int pInContour = 4; // number in each contour
				int faceCount = 0;
				int layerCount = 0;

				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{

					if (layerCount == 1)
					{
						layerCount++;
						continue;
					}

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
					{
						int pc_size = polyConnect.size();
						polyConnect.push_back((numPoints)+j + k);
						polyConnect.push_back((numPoints)+j + k + pInContour);
						polyConnect.push_back((numPoints)+j + k + pInContour + 1);
						polyConnect.push_back((numPoints)+j + k + 1);

						polyCount.push_back(4);
						//faceCount++;
					}
					layerCount++;
				}

				numPoints = pointArray.size();

			}

			if (boundaryArray[i] == zBoundary::zEdge)
			{
				///////top faces for slab snap
				snap.assign(20, zVector());

				if (boundaryArray[(i + 1) % axis.size()] == zBoundary::zInterior)
				{
					y = axis[i];
					x = axis[(i + 1) % axis.size()];
				}
				else
				{
					x = axis[i];
					y = axis[(i + 1) % axis.size()];
				}

				////

				snap[0] = x * 0.1 + position + zVector(0, 0, -0.2);
				snap[1] = x * 0.1  + y * 0.25 + position + zVector(0, 0, -0.2);
				snap[2] = snap[1];
				snap[3] = x * 0.07  + y * 0.25 + position + zVector(0, 0, -0.2);
				snap[4] = snap[3];
				snap[5] = x * 0.03  + y * 0.25 + position + zVector(0, 0, -0.2);
				snap[6] = snap[5];
				snap[7] = x * 0.01  + y * 0.2 + position + zVector(0, 0, -0.2);
				snap[8] = snap[7];
				snap[9] = x * 0.01  + position + zVector(0, 0, -0.2);

				snap[10] = x * 0.3  + position + zVector(0, 0, -0.2);
				snap[11] = x * 0.3  + y * 1.3 + position + zVector(0, 0, -0.2);
				snap[12] = snap[11];
				snap[13] = x * 0.15  + y * 1.3 + position + zVector(0, 0, -0.37);
				snap[14] = snap[13];
				snap[15] = x * 0.05  + y * 1.3 + position + zVector(0, 0, -0.5);
				snap[16] = snap[15];
				snap[17] = y + position + zVector(0, 0, -0.2);
				snap[18] = snap[17];
				snap[19] = position + zVector(0, 0, -0.2);

				for (auto v : snap)
				{
					pointArray.push_back(v);
				}

				/////
				snapSlabpoints[i] = snap;

				//////////////////////////////////
				//bottom faces
				/////
				zPointArray bottomPoints;
				bottomPoints.assign(40, zVector());

				int id = 0;
				////breat detached edge repeat points
				for (int i = 10; i < snap.size(); i++)
				{
					bottomPoints[id] = snap[i];
					id++;
				}

				bottomPoints[10] = x * 0.09  + position + zVector(0, 0, -0.8);
				bottomPoints[11] = x * 0.09  + y * 0.25 + position + zVector(0, 0, -0.8);
				bottomPoints[12] = bottomPoints[11];
				bottomPoints[13] = x * 0.09 + y * 0.5 + position + zVector(0, 0, -0.8);
				bottomPoints[14] = bottomPoints[13];
				bottomPoints[15] = x * 0.05  + y * 0.5 + position + zVector(0, 0, -0.8);
				bottomPoints[16] = bottomPoints[15];
				bottomPoints[17] = y * 0.35 + position + zVector(0, 0, -0.8);
				bottomPoints[18] = bottomPoints[17];
				bottomPoints[19] = position + zVector(0, 0, -0.8);

				bottomPoints[20] = x * 0.02 + position + zVector(0, 0, -1.85);
				bottomPoints[21] = x * 0.02 + y * 0.1 + position + zVector(0, 0, -1.85);
				bottomPoints[22] = bottomPoints[21];
				bottomPoints[23] = x * 0.02  + y * 0.2 + position + zVector(0, 0, -1.85);
				bottomPoints[24] = bottomPoints[23];
				bottomPoints[25] = x * 0.01  + y * 0.2 + position + zVector(0, 0, -1.85);
				bottomPoints[26] = bottomPoints[25];
				bottomPoints[27] = y * 0.2 + position + zVector(0, 0, -1.85);
				bottomPoints[28] = bottomPoints[27];
				bottomPoints[29] = position + zVector(0, 0, -1.85);

				bottomPoints[30] = x * 0.02  + position + zVector(0, 0, -height);
				bottomPoints[31] = x * 0.02 + y * 0.1  + position + zVector(0, 0, -height);
				bottomPoints[32] = bottomPoints[31];
				bottomPoints[33] = x * 0.02 + y * 0.2  + position + zVector(0, 0, -height);
				bottomPoints[34] = bottomPoints[33];
				bottomPoints[35] = x * 0.01 + y * 0.2 + position + zVector(0, 0, -height);
				bottomPoints[36] = bottomPoints[35];
				bottomPoints[37] = y * 0.2 + position + zVector(0, 0, -height);
				bottomPoints[38] = bottomPoints[37];
				bottomPoints[39] = position + zVector(0, 0, -height);

				for (auto v : bottomPoints)
				{
					pointArray.push_back(v);
				}

				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 10; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{
					if (layerCount == 1)
					{
						layerCount++;
						continue;
					}

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
					{
						int pc_size = polyConnect.size();
						polyConnect.push_back((numPoints)+j + k);
						polyConnect.push_back((numPoints)+j + k + pInContour);
						polyConnect.push_back((numPoints)+j + k + pInContour + 1);
						polyConnect.push_back((numPoints)+j + k + 1);

						polyCount.push_back(4);
					}
					layerCount++;
				}
				numPoints = pointArray.size();
			}
		}

		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			zFnMesh fnInMesh(columnMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);
			//printf("\n %i %i %i ", fnInMesh.numVertices(), fnInMesh.numEdges(), fnInMesh.numPolygons());
			fnInMesh.smoothMesh(2, false);
		}
	}

	ZSPACE_INLINE void zAgColumn::createTimber()
	{

		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		snapSlabpoints.assign(4, zVectorArray());

		//snapSlabpoints.assign(axis.size(), zVectorArray());
		int numPoints = 0;
		for (int i = 0; i < axis.size(); i++)
		{
			zVectorArray snap;

			if (boundaryArray[i] == zBoundary::zInterior)
			{
				///////top faces for slab snap
				snap.assign(8, zVector());

				x = axis[i];
				y = axis[(i + 1) % axis.size()];
				float xSpan = fabs(axis[i].length());
				float ySpan = fabs(axis[(i + 1) % axis.size()].length());

				x.normalize();
				y.normalize();
				////

				snap[0] = x * 0.1 * xSpan;
				snap[3] = y * 0.1 * ySpan;
				snap[1] = snap[0] + snap[3]; //center
				snap[2] = snap[0] + snap[3]; //center

				snap[0] += position + zVector(0,0,-0.2);
				snap[2] += position + zVector(0, 0, -0.2);
				snap[1] += position + zVector(0, 0, -0.2);
				snap[3] += position + zVector(0, 0, -0.2);
				/////

				snap[4] = x * 0.3 * xSpan + zVector(0,0,-0.3);
				snap[7] = y * 0.3 * ySpan + zVector(0, 0, -0.3);
				snap[5] = (x * 0.3 * xSpan) + (y * 0.3 * ySpan) + zVector(0, 0, -0.3); //center
				snap[6] = (x * 0.3 * xSpan) + (y * 0.3 * ySpan) + zVector(0, 0, -0.3); //center

				snap[6] += position + zVector(0, 0, -0.2);
				snap[5] += position + zVector(0, 0, -0.2);
				snap[4] += position + zVector(0, 0, -0.2);
				snap[7] += position + zVector(0, 0, -0.2);
				/////

				snapSlabpoints[i] = snap;

				for (auto v : snap)
				{
					pointArray.push_back(v);
				}

				//////////////////////////////////
				//bottom faces
				/////

				zPointArray bottomPoints;
				bottomPoints.assign(16, zVector());

				////breat detached edge repeat points
				bottomPoints[0] = snap[4];
				bottomPoints[3] = snap[7];
				bottomPoints[1] = snap[5]; //center
				bottomPoints[2] = snap[6]; //center
				/////

				bottomPoints[4] = x * 0.09 * xSpan + zVector(0, 0, -0.8) + position;
				bottomPoints[7] = y * 0.09 * ySpan + zVector(0, 0, -0.8) + position;
				bottomPoints[5] = (x * 0.09 * xSpan) + (y * 0.09 * ySpan) + zVector(0, 0, -0.8) + position; //center
				bottomPoints[6] = (x * 0.09 * xSpan) + (y * 0.09 * ySpan) + zVector(0, 0, -0.8) + position; //center
				/////

				bottomPoints[8] = x * 0.020 * xSpan + zVector(0, 0, -1.85) + position;
				bottomPoints[11] = y * 0.020 * ySpan + zVector(0, 0, -1.85) + position;
				bottomPoints[9] = (x * 0.020 * xSpan) + (y * 0.020 * ySpan) + zVector(0, 0, -1.85) + position; //center
				bottomPoints[10] = (x * 0.020 * xSpan) + (y * 0.020 * ySpan) + zVector(0, 0, -1.85) + position; //center
				/////

				bottomPoints[12] = x * 0.015 * xSpan + zVector(0, 0, -height) + position;
				bottomPoints[15] = y * 0.015 * ySpan + zVector(0, 0, -height) + position;
				bottomPoints[14] = (x * 0.015 * xSpan) + (y * 0.015 * ySpan) + zVector(0, 0, -height) + position; //center
				bottomPoints[13] = (x * 0.015 * xSpan) + (y * 0.015 * ySpan) + zVector(0, 0, -height) + position; //center
				/////////////////////////////////////////////////////

				for (auto v : bottomPoints)
				{
					pointArray.push_back(v);
				}

				/////////////////////
				//polyconnect and polycount quads
				int pInContour = 4; // number in each contour
				int faceCount = 0;
				int layerCount = 0;
				
				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{

					if (layerCount == 1)
					{
						layerCount++;
						continue;
					}

					for (int k = 0; k < pInContour - 1; k+=2) // loops though faces in a given j layer
					{
						int pc_size = polyConnect.size();
						polyConnect.push_back((numPoints) + j + k);
						polyConnect.push_back((numPoints) + j +k + pInContour);
						polyConnect.push_back((numPoints) + j  + k+ pInContour + 1);
						polyConnect.push_back((numPoints) + j + k + 1);

						polyCount.push_back(4);
						//faceCount++;
					}
					layerCount++;
				}

				numPoints = pointArray.size();
				
			}

			if (boundaryArray[i] == zBoundary::zEdge)
			{
				///////top faces for slab snap
				snap.assign(20, zVector());

				if (boundaryArray[(i + 1) % axis.size()] == zBoundary::zInterior)
				{
					y = axis[i];
					x = axis[(i + 1) % axis.size()];
				}
				else
				{
					x = axis[i];
					y = axis[(i + 1) % axis.size()];
				}


				////

				snap[0] = x * 0.1 + position + zVector(0, 0, -0.2);
				snap[1] = x * 0.1  + y * 0.25  + position + zVector(0, 0, -0.2);
				snap[2] = snap[1];
				snap[3] = x * 0.09  + y * 0.25 + position + zVector(0, 0, -0.2);
				snap[4] = snap[3];
				snap[5] = x * 0.08 + y * 0.25  + position + zVector(0, 0, -0.2);
				snap[6] = snap[5];
				snap[7] = x * 0.01  + y * 0.1  + position + zVector(0, 0, -0.2);
				snap[8] = snap[7];
				snap[9] = x * 0.01  + position + zVector(0, 0, -0.2);

				snap[10] = x * 0.4  + position + zVector(0, 0, -0.5);
				snap[11] = x * 0.4  + y * 1.7  + position + zVector(0, 0, -0.5);
				snap[12] = snap[11];
				snap[13] = x * 0.22 + y * 1.8 + position + zVector(0, 0, -0.2);
				snap[14] = snap[13];
				snap[15] = x * 0.09 + y * 1.2 + position + zVector(0, 0, -0.2);
				snap[16] = snap[15];
				snap[17] = y * 0.5 + position + zVector(0, 0, -0.2);
				snap[18] = snap[17];
				snap[19] = position + zVector(0, 0, -0.2);

				for (auto v : snap)
				{
					pointArray.push_back(v);
				}

				/////
				snapSlabpoints[i] = snap;

				//////////////////////////////////
				//bottom faces
				/////
				zPointArray bottomPoints;
				bottomPoints.assign(40, zVector());

				int id = 0;
				////breat detached edge repeat points
				for (int i = 10; i < snap.size(); i++)
				{
					bottomPoints[id] = snap[i];
					id++;
				}

				bottomPoints[10] = x * 0.09 + position + zVector(0, 0, -0.8);
				bottomPoints[11] = x * 0.09 + y * 0.35 + position + zVector(0, 0, -0.8);
				bottomPoints[12] = bottomPoints[11];
				bottomPoints[13] = x * 0.08 + y * 0.35 + position + zVector(0, 0, -0.8);
				bottomPoints[14] = bottomPoints[13];
				bottomPoints[15] = x * 0.07 + y * 0.35 + position + zVector(0, 0, -0.8);
				bottomPoints[16] = bottomPoints[15];
				bottomPoints[17] = y * 0.1 + position + zVector(0, 0, -0.8);
				bottomPoints[18] = bottomPoints[17];
				bottomPoints[19] = position + zVector(0, 0, -0.8);

				bottomPoints[20] = x * 0.02 + position + zVector(0, 0, -1.85);
				bottomPoints[21] = x * 0.02 + y * 0.1 + position + zVector(0, 0, -1.85);
				bottomPoints[22] = bottomPoints[21];
				bottomPoints[23] = x * 0.015 + y * 0.1 + position + zVector(0, 0, -1.85);
				bottomPoints[24] = bottomPoints[23];
				bottomPoints[25] = x * 0.01 + y * 0.1 + position + zVector(0, 0, -1.85);
				bottomPoints[26] = bottomPoints[25];
				bottomPoints[27] = y * 0.08 + position + zVector(0, 0, -1.85);
				bottomPoints[28] = bottomPoints[27];
				bottomPoints[29] = position + zVector(0, 0, -1.85);

				bottomPoints[30] = x * 0.02 + position + zVector(0, 0, -height);
				bottomPoints[31] = x * 0.02 + y * 0.1 + position + zVector(0, 0, -height);
				bottomPoints[32] = bottomPoints[31];
				bottomPoints[33] = x * 0.015 + y * 0.1 + position + zVector(0, 0, -height);
				bottomPoints[34] = bottomPoints[33];
				bottomPoints[35] = x * 0.01 + y * 0.1 + position + zVector(0, 0, -height);
				bottomPoints[36] = bottomPoints[35];
				bottomPoints[37] = y * 0.08 + position + zVector(0, 0, -height);
				bottomPoints[38] = bottomPoints[37];
				bottomPoints[39] = position + zVector(0, 0, -height);

				for (auto v : bottomPoints)
				{
					pointArray.push_back(v);
				}

				///////////////////////
				//polyconnect and polycount quads
				int pInContour = 10; // number in each contour
				int layerCount = 0;

				for (int j = 0; j < 5 * pInContour; j += pInContour) //jumps between layers (5 is total of layers:6 minus 1)
				{
					if (layerCount == 1)
					{
						layerCount++;
						continue;
					}

					for (int k = 0; k < pInContour - 1; k += 2) // loops though faces in a given j layer
					{
						int pc_size = polyConnect.size();
						polyConnect.push_back((numPoints)+j + k);
						polyConnect.push_back((numPoints)+j + k + pInContour);
						polyConnect.push_back((numPoints)+j + k + pInContour + 1);
						polyConnect.push_back((numPoints)+j + k + 1);

						polyCount.push_back(4);
					}
					layerCount++;
				}
				numPoints = pointArray.size();
			}
		}

		if (pointArray.size() != 0 && polyConnect.size() != 0 && polyCount.size() != 0)
		{
			zFnMesh fnInMesh(columnMeshObj);
			fnInMesh.create(pointArray, polyCount, polyConnect);

			fnInMesh.smoothMesh(2, false);
			createFrame();

		}
	}

	ZSPACE_INLINE void zAgColumn::createFrame()
	{
		zPointArray pointArray;
		zIntArray polyConnect;
		zIntArray polyCount;

		for (zItMeshHalfEdge he(columnMeshObj); !he.end(); he++)
		{
			if (he.onBoundary() || !he.getSym().onBoundary()) continue;

			zVector xd = he.getNext().getVector();
			xd.normalize();
			xd *= 0.05;

			zVector yd = he.getFace().getNormal();
			yd.normalize();
			yd *= 0.1;

			pointArray.push_back(he.getVertex().getPosition());
			pointArray.push_back(he.getVertex().getPosition() + yd);
			pointArray.push_back(he.getVertex().getPosition() + yd + xd);
			pointArray.push_back(he.getVertex().getPosition() + xd);

			pointArray.push_back(he.getPrev().getVertex().getPosition());
			pointArray.push_back(he.getPrev().getVertex().getPosition() + yd);
			pointArray.push_back(he.getPrev().getVertex().getPosition() + yd + xd);
			pointArray.push_back(he.getPrev().getVertex().getPosition() + xd);
		}

		for (int i = 0; i < pointArray.size() - 8; i += 8)
		{
			for (int j = 0; j < 4; j++)
			{
				int s = polyConnect.size();
				polyConnect.push_back(i + j);
				polyConnect.push_back(i + ((j + 1) % 4));
				polyConnect.push_back(i + ((j + 1) % 4) + 4);
				polyConnect.push_back(i + j + 4);

				polyCount.push_back(4);
			}
		}

		zFnMesh fnInMesh(columnMeshObj);
		fnInMesh.create(pointArray, polyCount, polyConnect);

	}

	//---- DISPLAY METHODS
	ZSPACE_INLINE void zAgColumn::displayColumn(bool showColumn)
	{
		columnMeshObj.setDisplayObject(showColumn);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zAgColumn::addObjsToModel()
	{
		model->addObject(columnMeshObj);
		columnMeshObj.setDisplayElements(false, true, true);
	}

#endif


}