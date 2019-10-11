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


#include<headers/zInterOp/interOp/zIOMesh.h>

#if defined(ZSPACE_MAYA_INTEROP)  && defined(ZSPACE_RHINO_INTEROP)

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zIOMesh::zIOMesh() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zIOMesh::~zIOMesh() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE bool zIOMesh::toZSpaceMesh(MObject &maya_meshObj, zObjMesh &zspace_meshObj)
	{
		/*MFn::Type fn = MFn::kInvalid;
		maya_meshObj.hasFn(fn);

		if (fn != MFn::kMesh) return false;*/

		zMayaFnMesh tempFn(zspace_meshObj);
		tempFn.fromMayaMesh(maya_meshObj);

		return true;
	}

	ZSPACE_INLINE bool zIOMesh::toZSpaceMesh(ON_Mesh &rhino_meshObj, zObjMesh &zspace_meshObj)
	{
		zRhinoFnMesh tempFn(zspace_meshObj);
		tempFn.fromRhinoMesh(rhino_meshObj);

		return true;
	}

	ZSPACE_INLINE bool zIOMesh::toRhinoMesh(zObjMesh &zspace_meshObj, ON_Mesh &rhino_meshObj)
	{
		zRhinoFnMesh tempFn(zspace_meshObj);
		tempFn.toRhinoMesh(rhino_meshObj);

		return true;
	}

	ZSPACE_INLINE bool zIOMesh::toRhinoMesh(MObject &maya_meshObj, ON_Mesh &rhino_meshObj)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_meshObj.hasFn(fn);

		if (fn != MFn::kMesh) return false;

		MFnMesh maya_fnMesh(maya_meshObj);

		// vertices
		MPointArray pos;
		maya_fnMesh.getPoints(pos);

		for (int i = 0; i < maya_fnMesh.numVertices(); i++)
		{
			rhino_meshObj.m_V.Append(ON_3fPoint(pos[i].x, pos[i].y, pos[i].z));
		}

		// faces
		for (MItMeshPolygon f(maya_meshObj); !f.isDone(); f.next())
		{
			MIntArray fVerts;
			f.getVertices(fVerts);

			if (fVerts.length() == 3 || fVerts.length() == 4)
			{
				ON_MeshFace r_face;

				r_face.vi[0] = fVerts[0];
				r_face.vi[1] = fVerts[1];
				r_face.vi[2] = fVerts[2];

				if (fVerts.length() == 4) r_face.vi[3] = fVerts[3];
				else r_face.vi[3] = r_face.vi[2];

				rhino_meshObj.m_F.Append(r_face);
			}
			else
			{
				MPointArray triPts;
				MIntArray tris;
				f.getTriangles(triPts, tris);

				for (int i = 0; i < triPts.length(); i++)
				{
					ON_MeshFace r_face;

					r_face.vi[0] = tris[i * 3 + 0];
					r_face.vi[1] = tris[i * 3 + 1];
					r_face.vi[2] = tris[i * 3 + 2];

					rhino_meshObj.m_F.Append(r_face);
				}
			}
		}

		rhino_meshObj.ComputeFaceNormals();
		rhino_meshObj.ComputeVertexNormals();


		return true;
	}

	ZSPACE_INLINE bool zIOMesh::toMayaMesh(zObjMesh &zspace_meshObj, MObject &maya_meshObj)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_meshObj.hasFn(fn);

		if (fn != MFn::kMesh) return false;

		zMayaFnMesh tempFn(zspace_meshObj);
		tempFn.toMayaMesh(maya_meshObj);

		return true;
	}
	
	ZSPACE_INLINE bool zIOMesh::toMayaMesh(ON_Mesh &rhino_meshObj,MObject &maya_meshObj)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_meshObj.hasFn(fn);

		if (fn != MFn::kMesh) return false;

		int numVertices = rhino_meshObj.VertexCount();

		int numFaces = rhino_meshObj.FaceCount();

		int numPolyConnects = 0;


		MPointArray pos;
		MIntArray polyCounts;
		MIntArray polyConnects;

		pos.setLength(numVertices);

		int counter = 0;
		for (int i = 0; i < rhino_meshObj.m_V.Count(); i++)
		{
			pos[i] = MPoint(rhino_meshObj.m_V[i].x, rhino_meshObj.m_V[i].y, rhino_meshObj.m_V[i].z);
		}

		for (int i = 0; i < rhino_meshObj.m_F.Count(); i++)
		{
			const ON_MeshFace& face = rhino_meshObj.m_F[i];

			if (face.IsTriangle())
			{
				polyCounts.append(3);

				polyConnects.append(face.vi[0]);
				polyConnects.append(face.vi[1]);
				polyConnects.append(face.vi[2]);

			}
			else
			{
				polyCounts.append(4);

				polyConnects.append(face.vi[0]);
				polyConnects.append(face.vi[1]);
				polyConnects.append(face.vi[2]);
				polyConnects.append(face.vi[3]);
			}
		}

		// create maya mesh
		MFnMesh maya_fnMesh(maya_meshObj);
		maya_fnMesh.create(pos.length(), polyCounts.length(), pos, polyCounts, polyConnects, maya_meshObj);
		maya_fnMesh.updateSurface();

		return true;
	}

}

#endif