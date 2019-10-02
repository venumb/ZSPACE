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


#include<headers/zInterOp/functionSets/zMayaFnMesh.h>

#if defined(ZSPACE_MAYA_INTEROP) 

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zMayaFnMesh::zMayaFnMesh() {}

	ZSPACE_INLINE zMayaFnMesh::zMayaFnMesh(zObjMesh &_zspace_meshObj)
	{
		meshObj = &_zspace_meshObj;;		
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zMayaFnMesh::~zMayaFnMesh() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE void zMayaFnMesh::fromMayaMesh(MObject &maya_meshObj)
	{
		MFnMesh maya_fnMesh(maya_meshObj);

		int numVertices = maya_fnMesh.numVertices();
		//printf("\n numVertices:%i", numVertices);

		int numEdges = maya_fnMesh.numEdges();
		//printf("\n numEdges:%i", numEdges);

		int numFaces = maya_fnMesh.numPolygons();
		//printf("\n numFaces:%i", numFaces);

		int numPolyConnects = 0;


		zPointArray pos;
		zIntArray  polyCounts;
		zIntArray  polyConnects;

		MPointArray inMesh_positions;
		maya_fnMesh.getPoints(inMesh_positions);

		pos.assign(maya_fnMesh.numVertices(), zPoint());

		int counter = 0;
		for (int i = 0; i < inMesh_positions.length(); i++)
		{
			pos[i] = zPoint(inMesh_positions[i].x, inMesh_positions[i].y, inMesh_positions[i].z);
		}

		MIntArray vCount, vList;
		maya_fnMesh.getVertices(vCount, vList);

		polyCounts.resize(vCount.length());
		vCount.get(&polyCounts[0]);

		polyConnects.resize(vList.length());
		vList.get(&polyConnects[0]);
			   		
		create(pos, polyCounts, polyConnects);
	}

	ZSPACE_INLINE void zMayaFnMesh::fromMayaMesh(MDagPath & maya_dagpath)
	{
		MFnMesh maya_fnMesh(maya_dagpath);

		int numVertices = maya_fnMesh.numVertices();

		int numEdges = maya_fnMesh.numEdges();

		int numFaces = maya_fnMesh.numPolygons();

		int numPolyConnects = 0;

		zPointArray pos;
		zIntArray  polyCounts;
		zIntArray  polyConnects;

		MPointArray inMesh_positions;
		maya_fnMesh.getPoints(inMesh_positions);

		for (int i = 0; i < inMesh_positions.length(); i++)
		{
			pos.push_back(zPoint(inMesh_positions[i].x, inMesh_positions[i].y, inMesh_positions[i].z));
		}

		for (MItMeshPolygon pIter(maya_dagpath); !pIter.isDone(); pIter.next())
		{
			MIntArray pVerts;
			pIter.getVertices(pVerts);

			polyCounts.push_back(pVerts.length());

			for (int i = 0; i < pVerts.length(); i++)
			{
				polyConnects.push_back(pVerts[i]);
				numPolyConnects++;
			}
		}
		
		create(pos, polyCounts, polyConnects);
	}

	ZSPACE_INLINE void zMayaFnMesh::toMayaMesh(MObject &maya_meshObj)
	{
		zIntArray  polyConnects;
		zIntArray  polyCount;

		getPolygonData(polyConnects, polyCount);

		MPointArray newMesh_verts;
		MIntArray  	newMesh_pCount(&polyCount[0], polyCount.size());
		MIntArray newMesh_pConnects(&polyConnects[0], polyConnects.size());

		// vertices
		newMesh_verts.setLength(numVertices());
		zPointArray pos;
		getVertexPositions(pos);
		for (int i = 0; i < numVertices(); i++)
		{
			newMesh_verts[i] = MPoint(pos[i].x, pos[i].y, pos[i].z);
		}

		//// create maya mesh

		MFnMesh maya_fnMesh(maya_meshObj);

		MStatus stat;
		maya_fnMesh.create(newMesh_verts.length(), newMesh_pCount.length(), newMesh_verts, newMesh_pCount, newMesh_pConnects, maya_meshObj, &stat);
		maya_fnMesh.updateSurface();
			
		printf("\n Maya Mesh: %i %i %i ", maya_fnMesh.numVertices(), maya_fnMesh.numEdges(), maya_fnMesh.numPolygons());
		
	}

	ZSPACE_INLINE void zMayaFnMesh::updateMayaOutmesh(MDataBlock & data, MObject & outMesh)
	{
		MStatus stat;

		MFnMeshData dataCreator_mesh;
		MObject o_outMesh = dataCreator_mesh.create(&stat);
		MFnMesh fn_operateMesh(o_outMesh, &stat);

		toMayaMesh( o_outMesh);

		MDataHandle h_outMesh = data.outputValue(outMesh, &stat);
		h_outMesh.set(o_outMesh);
		h_outMesh.setClean();
	}

}

#endif