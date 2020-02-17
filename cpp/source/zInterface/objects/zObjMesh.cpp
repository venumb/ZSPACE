// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zInterface/objects/zObjMesh.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjMesh::zObjMesh()
	{

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayVertices = false;
		displayEdges = true;
		displayFaces = true;

		displayDihedralEdges = false;
		displayVertexNormals = false;
		displayFaceNormals = false;

		dihedralAngleThreshold = 45;

		normalScale = 1.0;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjMesh::~zObjMesh() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjMesh::setDisplayElements(bool _displayVerts, bool _displayEdges, bool _displayFaces)
	{
		displayVertices = _displayVerts;
		displayEdges = _displayEdges;
		displayFaces = _displayFaces;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayElementIds(bool _displayVertIds, bool _displayEdgeIds, bool _displayFaceIds)
	{
		displayVertexIds = _displayVertIds;
		displayEdgeIds = _displayEdgeIds;
		displayFaceIds = _displayFaceIds;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayVertices(bool _displayVerts)
	{
		displayVertices = _displayVerts;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayEdges(bool _displayEdges)
	{
		displayEdges = _displayEdges;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayFaces(bool _displayFaces)
	{
		displayFaces = _displayFaces;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayDihedralEdges(bool _displayDihedralEdges, double _threshold)
	{
		displayDihedralEdges = _displayDihedralEdges;
		dihedralAngleThreshold = _threshold;

		if (_displayDihedralEdges) displayEdges = false;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayVertexNormals(bool _displayVertexNormal, double _normalScale)
	{
		displayVertexNormals = _displayVertexNormal;
		normalScale = _normalScale;
	}

	ZSPACE_INLINE void zObjMesh::setDisplayFaceNormals(bool _displayFaceNormal, double _normalScale)
	{
		displayFaceNormals = _displayFaceNormal;
		normalScale = _normalScale;
	}

	ZSPACE_INLINE void zObjMesh::setFaceCenters(zPointArray & _faceCenters)
	{
		faceCenters = _faceCenters;
	}

	ZSPACE_INLINE void zObjMesh::setEdgeCenters(zPointArray & _edgeCenters)
	{
		edgeCenters = _edgeCenters;
	}

	ZSPACE_INLINE void zObjMesh::setDihedralAngles(zDoubleArray & _edge_dihedralAngles)
	{
		edge_dihedralAngles = _edge_dihedralAngles;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zObjMesh::getVBO_VertexID()
	{
		return mesh.VBO_VertexId;
	}

	ZSPACE_INLINE int zObjMesh::getVBO_EdgeID()
	{
		return mesh.VBO_EdgeId;
	}

	ZSPACE_INLINE int zObjMesh::getVBO_FaceID()
	{
		return mesh.VBO_FaceId;
	}

	ZSPACE_INLINE int zObjMesh::getVBO_VertexColorID()
	{
		return mesh.VBO_VertexColorId;
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zObjMesh::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		coreUtils.getBounds(mesh.vertexPositions, minBB, maxBB);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else
	   
	ZSPACE_INLINE void zObjMesh::draw()
	{	
		if (displayObject)
		{
			drawMesh();		

			if (displayDihedralEdges) drawMesh_DihedralEdges();

			if (displayVertexNormals) drawMesh_VertexNormals();

			if (displayFaceNormals) drawMesh_FaceNormals();
		}

		if (displayObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}

	}

	//---- DISPLAY BUFFER METHODS

	ZSPACE_INLINE void zObjMesh::appendToBuffer(zDoubleArray edge_dihedralAngles, bool DihedralEdges, double angleThreshold)
	{
		displayObject = displayEdges = displayVertices = displayFaces = false;

		// Edge Indicies
		if (!DihedralEdges)
		{
			zIntArray _edgeIndicies;

			for (auto &e : mesh.edges)
			{
				if (e.isActive())
				{
					_edgeIndicies.push_back(e.getHalfEdge(0)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
					_edgeIndicies.push_back(e.getHalfEdge(1)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
				}
			}

			mesh.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);
		}
		else
		{


			zIntArray _edgeIndicies;

			for (auto &e : mesh.edges)
			{

				if (e.isActive())
				{
					if (abs(edge_dihedralAngles[e.getId()]) > angleThreshold || edge_dihedralAngles[e.getId()] == -1)
					{
						_edgeIndicies.push_back(e.getHalfEdge(0)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
						_edgeIndicies.push_back(e.getHalfEdge(1)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
					}
				}

			}

			mesh.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);
		}


		// Face Indicies

		zIntArray _faceIndicies;

		for (auto &f : mesh.faces)
		{
			if (f.isActive())
			{
				zIntArray faceVertsIds;
				mesh.getFaceVertices(f.getId(), faceVertsIds);

				for (int j = 0; j < faceVertsIds.size(); j++)
				{
					_faceIndicies.push_back(faceVertsIds[j] + displayUtils->bufferObj.nVertices);

				}
			}

		}

		mesh.VBO_FaceId = displayUtils->bufferObj.appendFaceIndices(_faceIndicies);

		// Vertex Attributes

		mesh.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(&mesh.vertexPositions[0], &mesh.vertexNormals[0], mesh.vertexPositions.size());
		mesh.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(&mesh.vertexColors[0], mesh.vertexColors.size());


	}

	//---- PROTECTED DISPLAY METHODS

	ZSPACE_INLINE void zObjMesh::drawMesh()
	{

		// draw vertex
		if (displayVertices)
		{
			displayUtils->drawVertices(mesh.vHandles, &mesh.vertexPositions[0], &mesh.vertexColors[0], &mesh.vertexWeights[0]);			
		}

		// draw vertex ID
		if (displayVertexIds)
		{
			zColor col(0.8, 0, 0, 1);
			displayUtils->drawVertexIds(mesh.n_v, &mesh.vertexPositions[0], col);
		}

		// draw edges
		if (displayEdges)
		{
			if (mesh.staticGeometry)
			{
				displayUtils->drawEdges(mesh.eHandles, mesh.edgeVertices, &mesh.vertexPositions[0], &mesh.edgeColors[0], &mesh.edgeWeights[0]);
			}

			else
			{
				vector<zIntArray> edgeVertices;
				edgeVertices.assign(mesh.edges.size(), zIntArray(2) = { -1,-1 });

				for (auto &e : mesh.edges)
				{

					if (mesh.eHandles[e.getId()].id != -1)
					{
						zIntArray eVerts;

						edgeVertices[e.getId()][0] = e.getHalfEdge(0)->getVertex()->getId();
						edgeVertices[e.getId()][1] = e.getHalfEdge(1)->getVertex()->getId();
					}
				}

				displayUtils->drawEdges(mesh.eHandles, edgeVertices, &mesh.vertexPositions[0], &mesh.edgeColors[0], &mesh.edgeWeights[0]);
			}
		}

		// draw edges ID
		if (displayEdgeIds)
		{
			if (edgeCenters.size() != mesh.n_e) throw std::invalid_argument(" error: edge centers are not computed.");

			zColor col(0, 0.8, 0, 1);
			displayUtils->drawEdgeIds(mesh.n_e, &edgeCenters[0], col);
		}

		// draw polygon
		if (displayFaces)
		{
			if (mesh.staticGeometry)
			{
				displayUtils->drawFaces(mesh.fHandles, mesh.faceVertices, &mesh.vertexPositions[0], &mesh.faceColors[0]);
			}
			else
			{
				vector<zIntArray> faceVertices;

				for (int i = 0; i < mesh.n_f; i++)
				{
					zIntArray faceVerts;
					if (mesh.fHandles[i].id != -1)
					{
						mesh.getFaceVertices(i, faceVerts);
					}

					faceVertices.push_back(faceVerts);

				}

				displayUtils->drawFaces(mesh.fHandles, faceVertices, &mesh.vertexPositions[0], &mesh.faceColors[0]);

			}
		}

		// draw polygon ID
		if (displayFaceIds)
		{
			if (faceCenters.size() != mesh.n_f) throw std::invalid_argument(" error: face centers are not computed.");

			zColor col(0, 0, 0.8, 1);
			displayUtils->drawFaceIds(mesh.n_f, &faceCenters[0], col);
		}
	}

	ZSPACE_INLINE void zObjMesh::drawMesh_DihedralEdges()
	{
		if(edge_dihedralAngles.size() != mesh.n_e) throw std::invalid_argument(" error: dihedral angles are not computed.");

		for (auto &e : mesh.edges)
		{
			int i = e.getId() * 2;

			if (e.isActive())
			{

				if (abs(edge_dihedralAngles[i]) > dihedralAngleThreshold)
				{
					zColor col;
					double wt = 1;

					if (mesh.edgeColors.size() > i)  col = mesh.edgeColors[i];
					if (mesh.edgeWeights.size() > i) wt = mesh.edgeWeights[i];

					int v1 = e.getHalfEdge(0)->getVertex()->getId();
					int v2 = e.getHalfEdge(1)->getVertex()->getId();

					displayUtils->drawLine(mesh.vertexPositions[v1], mesh.vertexPositions[v2], col, wt);
				}
			}
		}
	}

	ZSPACE_INLINE void zObjMesh::drawMesh_VertexNormals()
	{

		if (mesh.vertexNormals.size() == 0 || mesh.vertexNormals.size() != mesh.vertices.size()) throw std::invalid_argument(" error: mesh normals not computed.");

		for (auto &v : mesh.vertices)
		{
			int i = v.getId();
			if (v.isActive())
			{
				zVector p1 = mesh.vertexPositions[i];
				zVector p2 = p1 + (mesh.faceNormals[i] * normalScale);

				displayUtils->drawLine(p1, p2, zColor(0, 1, 0, 1));
			}

		}

	}

	ZSPACE_INLINE void zObjMesh::drawMesh_FaceNormals()
	{
		
		if (mesh.faceNormals.size() != mesh.faces.size()) throw std::invalid_argument(" error: mesh normals not computed.");

		if (mesh.faces.size() != faceCenters.size()) throw std::invalid_argument(" error: number of face centers not equal to number of faces .");

		for (auto &f : mesh.faces)
		{
			int i = f.getId();
			
			if (f.isActive())
			{
				zVector p1 = faceCenters[i];
				zVector p2 = p1 + (mesh.faceNormals[i] * normalScale);

				displayUtils->drawLine(p1, p2, zColor(0, 1, 0, 1));				
			}

		}


	}

#endif // !ZSPACE_UNREAL_INTEROP

}