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


#include<headers/zInterOp/functionSets/zUnrealFnMesh.h>

#if defined(ZSPACE_UNREAL_INTEROP) 

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zUnrealFnMesh::zUnrealFnMesh() {}

	ZSPACE_INLINE zUnrealFnMesh::zUnrealFnMesh(zObjMesh &_zspace_meshObj)
	{
		meshObj = &_zspace_meshObj;;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zUnrealFnMesh::~zUnrealFnMesh() {}

	//---- INTEROP METHODS
	
	ZSPACE_INLINE void zUnrealFnMesh::toUnrealMesh(FDynamicMesh3 &unreal_mesh)
	{
		//int numVertices = unreal_mesh.VertexCount();

		//int numFaces = unreal_mesh.TriangleCount();

		//int numPolyConnects = 0;

		//zPointArray pos;
		//zIntArray polyCounts;
		//zIntArray polyConnects;

		//pos.assign(numVertices, zPoint());

		//check(unreal_mesh.IsCompactV());

		//for (int32 vid : unreal_mesh.VertexIndicesItr())
		//{
		//	FVector3d Pos = unreal_mesh.GetVertex(vid);
		//}

		//for (int32 tid : unreal_mesh.TriangleIndicesItr())
		//{
		//	FIndex3i Tri = unreal_mesh.GetTriangle(tid);

		//}

		//create(pos, polyCounts, polyConnects);
	}

	ZSPACE_INLINE void zUnrealFnMesh::toUnrealMesh(FDynamicMesh3 &unreal_mesh)
	{
		//// vertices
		//zVector* pos = getRawVertexPositions();
		//for (int i = 0; i < numVertices(); i++)
		//{
		//	rhino_mesh.m_V.Append(ON_3fPoint(pos[i].x, pos[i].y, pos[i].z));
		//}

		//// faces
		//for (zItMeshFace f(*meshObj); !f.end(); f++)
		//{
		//	zIntArray fVerts;
		//	f.getVertices(fVerts);			

		//	if (fVerts.size() == 3 || fVerts.size() == 4)
		//	{
		//		ON_MeshFace r_face;

		//		r_face.vi[0] = fVerts[0];
		//		r_face.vi[1] = fVerts[1];
		//		r_face.vi[2] = fVerts[2];

		//		if(fVerts.size() == 4) r_face.vi[3] = fVerts[3];
		//		else r_face.vi[3] = r_face.vi[2];

		//		rhino_mesh.m_F.Append(r_face);
		//	}
		//	else
		//	{
		//		int numT;
		//		zIntArray tris;
		//		f.getTriangles(numT, tris);

		//		for (int i = 0; i < numT; i++)
		//		{
		//			ON_MeshFace r_face;

		//			r_face.vi[0] = tris[i * 3 + 0];
		//			r_face.vi[1] = tris[i * 3 + 1];
		//			r_face.vi[2] = tris[i * 3 + 2];

		//			rhino_mesh.m_F.Append(r_face);
		//		}
		//	}
		//}

		//rhino_mesh.ComputeFaceNormals();
		//rhino_mesh.ComputeVertexNormals();

		//printf("\n RhinoMesh : v %i f%i ", rhino_mesh.VertexCount(), rhino_mesh.FaceCount());
	}
}

#endif