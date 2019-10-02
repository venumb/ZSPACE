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


#include<headers/zInterOp/functionSets/zRhinoFnGraph.h>

#if defined(ZSPACE_RHINO_INTEROP) 

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zRhinoFnGraph::zRhinoFnGraph() {}

	ZSPACE_INLINE zRhinoFnGraph::zRhinoFnGraph(zObjGraph &_zspace_graphObj)
	{
		graphObj = &_zspace_graphObj;;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zRhinoFnGraph::~zRhinoFnGraph() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE void zRhinoFnGraph::fromRhinoCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves, bool onlyCorners)
	{
		zPointArray pos;
		zIntArray  edgeConnects;

		unordered_map <string, int> positionVertex;

		//printf("\n o_inCurves %i ", o_inCurves.length());

		for (int i = 0; i < rhino_curves.Count(); i++)
		{
					
			//printf("\n n_CPts %i ", c_pts.length());

			if (rhino_curves[i].CVCount() >= 2 && onlyCorners)
			{				

				// vert 1
				ON_3dPoint pos0;
				rhino_curves[i].GetCV(0, pos0);

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
				ON_3dPoint pos1;
				rhino_curves[i].GetCV(rhino_curves[i].CVCount() -1, pos1);

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

			if (rhino_curves[i].CVCount() >= 2 && !onlyCorners)
			{
				for (int j = 0; j < rhino_curves[i].CVCount() - 1; j++)
				{
					// vert 1
					ON_3dPoint pos0;
					rhino_curves[i].GetCV(j, pos0);

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
					ON_3dPoint pos1;
					rhino_curves[i].GetCV(j+1, pos1);

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

	ZSPACE_INLINE void zRhinoFnGraph::toRhinoCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves)
	{
		rhino_curves.Empty();

		int crvCounter = 0;

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			zPoint v1 = e.getHalfEdge(0).getStartVertex().getPosition();
			zPoint v2 = e.getHalfEdge(0).getVertex().getPosition();

			ON_3dPointArray pts;

			pts.Append(ON_3dPoint(v1.x, v1.y, v1.z));
			pts.Append(ON_3dPoint(v2.x, v2.y, v2.z));

			rhino_curves.AppendNew();
			rhino_curves[crvCounter].CreateClampedUniformNurbs(3, 2, pts.Count(), pts);
		}

	}

}

#endif