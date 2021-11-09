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
	
	//---- OVERRIDE METHODS

	ZSPACE_INLINE zFnType zMayaFnMesh::getType()
	{
		return zMeshFn;
	}

	ZSPACE_INLINE void zMayaFnMesh::from(string path, zFileTpye type, bool staticGeom)
	{

		zFnMesh::from(path, type, staticGeom);

		// add crease data
		if (type == zJSON)
		{
			getCreaseDataJSON(path);
		}

	}

	ZSPACE_INLINE void zMayaFnMesh::to(string path, zFileTpye type)
	{
		zFnMesh::to(path, type);

		// add crease data
		if (type == zJSON)
		{
			setCreaseDataJSON(path);
		}

	}

	ZSPACE_INLINE void zMayaFnMesh::clear()
	{

		meshObj->mesh.clear();
	}

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

		zColorArray fColors;
		fColors.assign(polyCounts.size(), zColor(0.5, 0.5, 0.5, 1));

		for (MItMeshPolygon p(maya_meshObj); !p.isDone(); p.next())
		{
			if (p.hasColor())
			{
				MColor pCol;
				p.getColor(pCol);

				fColors[p.index()] = zColor(pCol.r, pCol.g, pCol.b, 1);
			}
			
		}

		setFaceColors(fColors);

		MUintArray  edgeIds;
		MDoubleArray creaseData;
		maya_fnMesh.getCreaseEdges(edgeIds, creaseData);

		creaseEdgeData.clear();
		creaseEdgeData.assign(creaseData.length(), 0.0);

		creaseEdgeIndex.clear();
		creaseEdgeIndex.assign(edgeIds.length(), -1);

		for (int i = 0; i < edgeIds.length(); i++)
		{
			creaseEdgeData[i] = creaseData[i];
			creaseEdgeIndex[i] = edgeIds[i];
		}

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

		MUintArray  edgeIds;
		MDoubleArray creaseData;
		maya_fnMesh.getCreaseEdges(edgeIds, creaseData);

		creaseEdgeData.clear();
		creaseEdgeData.assign(creaseData.length(), 0.0);

		creaseEdgeIndex.clear();
		creaseEdgeIndex.assign(edgeIds.length(), -1);

		for (int i = 0; i < edgeIds.length(); i++)
		{
			creaseEdgeData[i] = creaseData[i];
			creaseEdgeIndex[i] = edgeIds[i];
		}
	}

	ZSPACE_INLINE void zMayaFnMesh::toMayaMesh(MObject &maya_meshObj)
	{
		zIntArray  polyConnects;
		zIntArray  polyCount;
		zPointArray pos;

		getVertexPositions(pos);
		getPolygonData(polyConnects, polyCount);
				

		zPoint* p = pos.data();
		

		MPointArray newMesh_verts ;
		MIntArray  	newMesh_pCount(&polyCount[0], polyCount.size());
		MIntArray newMesh_pConnects(&polyConnects[0], polyConnects.size());

		// vertices
		newMesh_verts.setLength(numVertices());
		
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
		
		MUintArray  edgeIds;
		MDoubleArray creaseData;

		for (int i = 0; i < creaseEdgeIndex.size(); i++)
		{
			creaseData.append(creaseEdgeData[i]);
			edgeIds.append(creaseEdgeIndex[i]);
		}

		maya_fnMesh.setCreaseEdges(edgeIds, creaseData);

	}

	ZSPACE_INLINE void zMayaFnMesh::updateMayaOutmesh(MDataBlock & data, MObject & outMesh, bool updateVertexColor, bool updateFaceColor)
	{
		MStatus stat;

		MFnMeshData dataCreator_mesh;
		MObject o_outMesh = dataCreator_mesh.create(&stat);
		
		toMayaMesh( o_outMesh);

		MFnMesh fn_operateMesh(o_outMesh, &stat);

		if (updateVertexColor || updateFaceColor)
		{
			if (updateVertexColor)
			{
				zColor* vCols = getRawVertexColors();

				MColorArray cols;
				MIntArray vList;

				for (int i = 0; i < numVertices(); i++)
				{
					cols.append(MColor(vCols[i].r, vCols[i].g, vCols[i].b));
					vList.append(i);
				}

				fn_operateMesh.setVertexColors(cols, vList);				
			}

			if (updateFaceColor)
			{
				zColor* fCols = getRawFaceColors();

				MColorArray cols;
				MIntArray fList;

				for (int i = 0; i < numPolygons(); i++)
				{
					cols.append(MColor(fCols[i].r, fCols[i].g, fCols[i].b));
					fList.append(i);
				}

				fn_operateMesh.setFaceColors(cols, fList);
			}

			fn_operateMesh.updateSurface();
		}

		MDataHandle h_outMesh = data.outputValue(outMesh, &stat);
		h_outMesh.set(o_outMesh);
		h_outMesh.setClean();
	}

	//---- PRIVATE METHODS

	ZSPACE_INLINE void zMayaFnMesh::setCreaseDataJSON(string outfilename)
	{
		// remove inactive elements
		if (numVertices() != meshObj->mesh.vertices.size()) garbageCollection(zVertexData);
		if (numEdges() != meshObj->mesh.edges.size()) garbageCollection(zEdgeData);
		if (numPolygons() != meshObj->mesh.faces.size())garbageCollection(zFaceData);

		// read existing data in the json 
		json j;		

		ifstream in_myfile;
		in_myfile.open(outfilename.c_str());

		int lineCnt = 0;
    
		if (in_myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
		}

		in_myfile >> j;
		in_myfile.close();

		// CREATE JSON FILE
		zUtilsJsonHE meshJSON;		

		// Vertices
		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (v.getHalfEdge().isActive()) meshJSON.vertices.push_back(v.getHalfEdge().getId());
			else meshJSON.vertices.push_back(-1);

		}

		//Edge Crease	
		meshJSON.edgeCreaseData.assign(numEdges(), 0.0);

		for (int i =0; i< creaseEdgeIndex.size(); i++)
		{
			meshJSON.edgeCreaseData[creaseEdgeIndex[i]] = creaseEdgeData[i];	
		}

		// Json file 
		j["EdgeCreaseData"] = meshJSON.edgeCreaseData;		

		// EXPORT	
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;
		}		

		//myfile.precision(16);
		myfile << j.dump();
		myfile.close();

	}

	ZSPACE_INLINE void zMayaFnMesh::getCreaseDataJSON(string infilename)
	{

		json j;
		zUtilsJsonHE meshJSON;


		ifstream in_myfile;
		in_myfile.open(infilename.c_str());

		int lineCnt = 0;

		if (in_myfile.fail())
		{
			cout << " error in opening file  " << infilename.c_str() << endl;
			
		}

		in_myfile >> j;
		in_myfile.close();

		// READ Data from JSON

		// Vertices
		meshJSON.edgeCreaseData.clear();

		bool checkCreaseData = (j.find("EdgeCreaseData") != j.end()) ? true : false;
		creaseEdgeData.clear();
		creaseEdgeIndex.clear();

		if (checkCreaseData)
		{
			meshJSON.edgeCreaseData = (j["EdgeCreaseData"].get<vector<double>>());

			for (int i = 0; i < meshJSON.edgeCreaseData.size(); i++)
			{
				if (meshJSON.edgeCreaseData[i] != 0)
				{
					creaseEdgeIndex.push_back(i);
					creaseEdgeData.push_back(meshJSON.edgeCreaseData[i]);
				}
			}
		}
		
	}

}

#endif