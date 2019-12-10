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


#include<headers/zInterOp/interOp/zIOMeshSurface.h>

#if defined(ZSPACE_MAYA_INTEROP)  && defined(ZSPACE_RHINO_INTEROP)

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zIOMeshSurface::zIOMeshSurface() {}

	ZSPACE_INLINE zIOMeshSurface::zIOMeshSurface(MObject &_maya_meshObj, zObjMesh &_zspace_meshObj) 
	{
		// check for quad mesh
		inputQuadMesh = makeQuadMesh(_maya_meshObj);

		// interOp to zspace mesh
		zIOMesh temp;
		temp.toZSpaceMesh(_maya_meshObj, _zspace_meshObj);
		
		// set objects
		zspace_meshObj = &_zspace_meshObj;
		zspace_FnMesh = zFnMesh(_zspace_meshObj);

		maya_meshObj = &_maya_meshObj;		
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zIOMeshSurface::~zIOMeshSurface() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE void zIOMeshSurface::toRhinoSurface(int subdivs, ON_ClassArray<ON_NurbsCurve> &rhino_nurbsCurve, ON_ClassArray<ON_NurbsSurface>& rhino_surface)
	{
		int stride = 1 /*(subdivs == 0) ? 1 : pow(2, subdivs)*/;
		//if(!inputQuadMesh && subdivs == 4) stride = pow(2, 3);

		// mesh fn
		MFnMesh maya_FnMesh(*maya_meshObj);	
		MStatus stat;

		int nV_lowPoly = maya_FnMesh.numVertices();
		int nF_lowPoly = maya_FnMesh.numPolygons();

		//Smooth maya mesh 
		MFnMeshData dataCreator_mesh;
		MObject maya_SmoothMeshObj = dataCreator_mesh.create(&stat);
		maya_FnMesh.copy(*maya_meshObj, maya_SmoothMeshObj, &stat);

		MMeshSmoothOptions smoothOptions;
		smoothOptions.setKeepBorderEdge(false);
		(inputQuadMesh) ? smoothOptions.setDivisions(4) : smoothOptions.setDivisions(3);

		maya_FnMesh.generateSmoothMesh(maya_SmoothMeshObj, &smoothOptions, &stat);


		//Smooth zspace mesh
		zIOMesh meshIO;
		zObjMesh zspace_SmoothMeshObj;
		meshIO.toZSpaceMesh(maya_SmoothMeshObj, zspace_SmoothMeshObj);

		// low poly half edge -> nurbs curve
		rhino_nurbsCurve.SetCapacity(zspace_FnMesh.numHalfEdges());

		zBoolArray halfEdgeVisited;
		halfEdgeVisited.assign(zspace_FnMesh.numHalfEdges(), false);

		for (zItMeshVertex v(zspace_SmoothMeshObj); !v.end(); v++)
		{
			if (v.getId() >= nV_lowPoly) break;

			
			zItMeshHalfEdgeArray connectedHE;
			v.getConnectedHalfEdges(connectedHE);

			for (auto &he : connectedHE)
			{				

				zItMeshVertexArray cVs;
				getVertsforCurve(he, nV_lowPoly, cVs);

				// get corresponding low poly half edge
				int v0 = cVs[0].getId();
				int v1 = cVs[cVs.size() - 1].getId();
				
				int he_lowPoly = -1;
				bool chk  = zspace_FnMesh.halfEdgeExists(v0, v1, he_lowPoly);

				if (chk && !halfEdgeVisited[he_lowPoly])
				{
					// points
					ON_3dPointArray pts;

					for (int i = 0; i < cVs.size(); i += stride)
					{
						zPoint pos = cVs[i].getPosition();
						pts.Append(ON_3fPoint(pos.x, pos.y, pos.z));
					}

					int dimension = 3;			
					int order = (pts.Count() <= 4) ? pts.Count() : 4;	

					rhino_nurbsCurve[he_lowPoly].CreateClampedUniformNurbs(dimension, order, pts.Count(), pts);
					halfEdgeVisited[he_lowPoly] = true;	

					printf("\n %i %i  ", rhino_nurbsCurve[he_lowPoly].CVCount(), pts.Count());
				}

			}
		}

		//  Nurbs surfaces
		//rhino_surface.SetCapacity(nF_lowPoly);
		
		for (zItMeshFace f(*zspace_meshObj); !f.end(); f++)
		{
			zIntArray heIndicies;
			f.getHalfEdges(heIndicies);

			const ON_Curve* c[4];

			c[0] = &rhino_nurbsCurve[heIndicies[0]];
			c[1] = &rhino_nurbsCurve[heIndicies[1]];
			c[2] = &rhino_nurbsCurve[heIndicies[2]];
			c[3] = &rhino_nurbsCurve[heIndicies[3]];

			ON_Brep* brep = RhinoCreateEdgeSrf(4, c);

			if (nullptr != brep)
			{
				//std::cout << "Brep with " << brep->m_F.Count() << " faces created" << std::endl;
				for (int i = 0; i < brep->m_F.Count(); i++) rhino_surface.Append(*brep->m_F[0].NurbsSurface());
				delete brep; // Don't leak...
			}

			//delete[] c; // Don't leak...
		}

		printf("\n %i rhinoSurfaces created. ", rhino_surface.Count());
		
	}

	//---- PROTECTED METHODS
	
	ZSPACE_INLINE bool zIOMeshSurface::makeQuadMesh(MObject &inMeshObj)
	{
		bool quadMesh = true;

		for (MItMeshPolygon f(inMeshObj); !f.isDone(); f.next())
		{
			if (f.polygonVertexCount() != 4)
			{
				quadMesh = false;
				break;
			}
		}

		if (!quadMesh)
		{
			MFnMesh mayaFn(inMeshObj);
			
			MMeshSmoothOptions smoothOptions;
			smoothOptions.setKeepBorderEdge(false);
			smoothOptions.setDivisions(1);

			MObject o_SmoothMesh = mayaFn.generateSmoothMesh(inMeshObj, &smoothOptions);
		}

		return quadMesh;
	}

	ZSPACE_INLINE void zIOMeshSurface::getVertsforCurve(zItMeshHalfEdge &smooth_he, int nV_lowPoly, zItMeshVertexArray &crvVerts)
	{
		crvVerts.clear();

		if (smooth_he.onBoundary())
		{
			// first CV
			crvVerts.push_back(smooth_he.getStartVertex());

			zItMeshVertex v = smooth_he.getVertex();

			while (v.getId() >= nV_lowPoly)
			{
				crvVerts.push_back(v);

				smooth_he = smooth_he.getNext();

				v = smooth_he.getVertex();
			}

			// last CV
			crvVerts.push_back(v);

		}
		else
		{
			// first CV
			crvVerts.push_back(smooth_he.getStartVertex());

			zItMeshVertex v = smooth_he.getVertex();

			while (v.getId() >= nV_lowPoly)
			{
				crvVerts.push_back(v);

				smooth_he = smooth_he.getNext().getSym();
				smooth_he = smooth_he.getNext();

				v = smooth_he.getVertex();
			}

			// last CV
			crvVerts.push_back(v);
		}

	}

}

#endif