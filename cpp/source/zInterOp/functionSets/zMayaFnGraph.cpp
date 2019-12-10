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


#include<headers/zInterOp/functionSets/zMayaFnGraph.h>

#if defined(ZSPACE_MAYA_INTEROP) 

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zMayaFnGraph::zMayaFnGraph() {}

	ZSPACE_INLINE zMayaFnGraph::zMayaFnGraph(zObjGraph &_zspace_graphObj)
	{
		graphObj = &_zspace_graphObj;;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zMayaFnGraph::~zMayaFnGraph() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE void zMayaFnGraph::fromMayaCurves(MObjectArray &maya_curves, bool onlyCorners)
	{
		zPointArray pos;
		zIntArray  edgeConnects;

		unordered_map <string, int> positionVertex;

		//printf("\n o_inCurves %i ", maya_curves.length());

		for (int i = 0; i < maya_curves.length(); i++)
		{
			MFnNurbsCurve  fn_inCurve(maya_curves[i]);

			MPointArray c_pts;
			fn_inCurve.getCVs(c_pts);

			//printf("\n n_CPts %i ", c_pts.length());

			if (c_pts.length() >= 2 && onlyCorners)
			{
				// vert 1
				MVector pos0 = c_pts[0];

				int VertexId0 = -1;
				zPoint p0(pos0.x, pos0.y, pos0.z);
				coreUtils.vertexExists(positionVertex, p0, PRESFAC, VertexId0);

				if (VertexId0 == -1)
				{
					pos.push_back(p0);
					VertexId0 = pos.size() - 1;

					coreUtils.addToPositionMap(positionVertex, p0, VertexId0, PRESFAC);					
				}

				// vert 2 
			
				MVector pos1 = c_pts[c_pts.length() - 1];

				int VertexId1 = -1;
				zPoint p1(pos1.x, pos1.y, pos1.z);
				coreUtils.vertexExists(positionVertex, p1, PRESFAC, VertexId1);

				if (VertexId1 == -1)
				{
					pos.push_back(p1);
					VertexId1 = pos.size() - 1;

					coreUtils.addToPositionMap(positionVertex, p1, VertexId1, PRESFAC);
				}


				edgeConnects.push_back(VertexId0);
				edgeConnects.push_back(VertexId1);
			}

			if (c_pts.length() >= 2 && !onlyCorners)
			{
				for (int j = 0; j < c_pts.length() - 1; j++)
				{
					// vert 1
					MVector pos0 = c_pts[j];

					int VertexId0 = -1;
					zPoint p0(pos0.x, pos0.y, pos0.z);
					coreUtils.vertexExists(positionVertex, p0, PRESFAC, VertexId0);

					if (VertexId0 == -1)
					{
						pos.push_back(p0);
						VertexId0 = pos.size() - 1;

						coreUtils.addToPositionMap(positionVertex, p0, VertexId0, PRESFAC);
					}

					// vert 2 

					MVector pos1 = c_pts[j + 1];

					int VertexId1 = -1;
					zPoint p1(pos1.x, pos1.y, pos1.z);
					coreUtils.vertexExists(positionVertex, p1, PRESFAC, VertexId1);

					if (VertexId1 == -1)
					{
						pos.push_back(p1);
						VertexId1 = pos.size() - 1;

						coreUtils.addToPositionMap(positionVertex, p1, VertexId1, PRESFAC);
					}


					edgeConnects.push_back(VertexId0);
					edgeConnects.push_back(VertexId1);
				}
			}


		}

		//printf("\n %i %i ", pos.size(), edgeConnects.size());	
		create(pos, edgeConnects);


	}

	ZSPACE_INLINE void zMayaFnGraph::fromMayaPoints(MPointArray &maya_Points)
	{
		if (maya_Points.length() == 0) return;
	

		// create new graph


		zPointArray pos;
		zIntArray  edgeConnects;

		unordered_map <string, int> positionVertex;

		for (int i = 0; i < maya_Points.length(); i += 2)
		{
			/// vert 1
			MVector pos0 = maya_Points[i];

			int VertexId0 = -1;
			zPoint p0(pos0.x, pos0.y, pos0.z);
			coreUtils.vertexExists(positionVertex, p0, PRESFAC, VertexId0);

			if (VertexId0 == -1)
			{
				pos.push_back(p0);
				VertexId0 = pos.size() - 1;

				coreUtils.addToPositionMap(positionVertex, p0, VertexId0, PRESFAC);
			}

			// vert 2 

			MVector pos1 = maya_Points[i + 1];

			int VertexId1 = -1;
			zPoint p1(pos1.x, pos1.y, pos1.z);
			coreUtils.vertexExists(positionVertex, p1, PRESFAC, VertexId1);

			if (VertexId1 == -1)
			{
				pos.push_back(p1);
				VertexId1 = pos.size() - 1;

				coreUtils.addToPositionMap(positionVertex, p1, VertexId1, PRESFAC);
			}

			edgeConnects.push_back(VertexId0);
			edgeConnects.push_back(VertexId1);

		}

		//	printf("\n %i %i ", pos.size(), edgeConnects.size());
		create(pos, edgeConnects);	
	}

	ZSPACE_INLINE void zMayaFnGraph::toMayaCurves(MObjectArray &maya_curves)
	{
		maya_curves.clear();

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			zPoint v1 = e.getHalfEdge(0).getStartVertex().getPosition();
			zPoint v2 = e.getHalfEdge(0).getVertex().getPosition();

			MPointArray pts;

			pts.append(MPoint(v1.x, v1.y, v1.z));
			pts.append(MPoint(v2.x, v2.y, v2.z));

			MFnNurbsCurveData  dataCreator_Curv;
			MObject o_fncurve = dataCreator_Curv.create();

			MFnNurbsCurve fn_operateCurve(o_fncurve);

			unsigned int deg = 1;
			MDoubleArray knots;
			unsigned int i;
			for (i = 0; i < pts.length(); i++)	knots.append((double)i);

			fn_operateCurve.create(pts, knots, 1, MFnNurbsCurve::kOpen, false, false);	
			fn_operateCurve.updateCurve();
			maya_curves.append(o_fncurve);
		}

	}

	ZSPACE_INLINE void zMayaFnGraph::toMayaCurves()
	{
		MStatus stat;
		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{			
			zPoint v1 = e.getHalfEdge(0).getStartVertex().getPosition();
			zPoint v2 = e.getHalfEdge(0).getVertex().getPosition();

			MPointArray pts;

			pts.append(MPoint(v1.x, v1.y, v1.z));
			pts.append(MPoint(v2.x, v2.y, v2.z));

			MFnNurbsCurveData  dataCreator_Curv;
			MObject o_fncurve = dataCreator_Curv.create();

			MFnNurbsCurve fn_operateCurve(o_fncurve, &stat);

			unsigned int deg = 1;

			fn_operateCurve.createWithEditPoints(pts, deg, MFnNurbsCurve::kOpen, false, true, true, MObject::kNullObj, NULL);
			
		}
		
	}

}

#endif