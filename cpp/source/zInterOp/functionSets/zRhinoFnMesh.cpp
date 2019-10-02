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


#include<headers/zInterOp/functionSets/zRhinoFnMesh.h>

#if defined(ZSPACE_RHINO_INTEROP) 

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zRhinoFnMesh::zRhinoFnMesh() {}

	ZSPACE_INLINE zRhinoFnMesh::zRhinoFnMesh(zObjMesh &_zspace_meshObj)
	{
		meshObj = &_zspace_meshObj;;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zRhinoFnMesh::~zRhinoFnMesh() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE void zRhinoFnMesh::fromRhinoMesh(ON_Mesh &rhino_mesh)
	{
		int numVertices = rhino_mesh.VertexCount();

		int numFaces = rhino_mesh.FaceCount();

		int numPolyConnects = 0;


		zPointArray pos;
		zIntArray polyCounts;
		zIntArray polyConnects;

		pos.assign(numVertices, zPoint());

		int counter = 0;
		for (int i = 0; i < rhino_mesh.m_V.Count(); i++)
		{
			pos[i] = zPoint(rhino_mesh.m_V[i].x, rhino_mesh.m_V[i].y, rhino_mesh.m_V[i].z);
		}

		for (int i = 0; i < rhino_mesh.m_F.Count(); i++)
		{
			const ON_MeshFace& face = rhino_mesh.m_F[i];

			if (face.IsTriangle())
			{
				polyCounts.push_back(3);

				polyConnects.push_back(face.vi[0]);
				polyConnects.push_back(face.vi[1]);
				polyConnects.push_back(face.vi[2]);
				
			}
			else
			{
				polyCounts.push_back(4);

				polyConnects.push_back(face.vi[0]);
				polyConnects.push_back(face.vi[1]);
				polyConnects.push_back(face.vi[2]);
				polyConnects.push_back(face.vi[3]);				
			}
		}

		create(pos, polyCounts, polyConnects);
	}

	ZSPACE_INLINE void zRhinoFnMesh::toRhinoMesh(ON_Mesh &rhino_mesh)
	{
		// vertices
		zVector* pos = getRawVertexPositions();
		for (int i = 0; i < numVertices(); i++)
		{
			rhino_mesh.m_V.Append(ON_3fPoint(pos[i].x, pos[i].y, pos[i].z));
		}

		// faces
		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			zIntArray fVerts;
			f.getVertices(fVerts);			

			if (fVerts.size() == 3 || fVerts.size() == 4)
			{
				ON_MeshFace r_face;

				r_face.vi[0] = fVerts[0];
				r_face.vi[1] = fVerts[1];
				r_face.vi[2] = fVerts[2];

				if(fVerts.size() == 4) r_face.vi[3] = fVerts[3];
				else r_face.vi[3] = r_face.vi[2];

				rhino_mesh.m_F.Append(r_face);
			}
			else
			{
				int numT;
				zIntArray tris;
				f.getTriangles(numT, tris);

				for (int i = 0; i < numT; i++)
				{
					ON_MeshFace r_face;

					r_face.vi[0] = tris[i * 3 + 0];
					r_face.vi[1] = tris[i * 3 + 1];
					r_face.vi[2] = tris[i * 3 + 2];

					rhino_mesh.m_F.Append(r_face);
				}
			}
		}

		rhino_mesh.ComputeFaceNormals();
		rhino_mesh.ComputeVertexNormals();

		printf("\n RhinoMesh : v %i f%i ", rhino_mesh.VertexCount(), rhino_mesh.FaceCount());
	}
}

#endif