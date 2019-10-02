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


#include<headers/zInterface/objects/zObjMesh.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjMesh::zObjMesh()
	{
		displayUtils = nullptr;

		showVertices = false;
		showEdges = true;
		showFaces = true;

		showDihedralEdges = false;
		showVertexNormals = false;
		showFaceNormals = false;

		dihedralAngleThreshold = 45;

		normalScale = 1.0;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjMesh::~zObjMesh() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjMesh::setShowElements(bool _showVerts, bool _showEdges, bool _showFaces)
	{
		showVertices = _showVerts;
		showEdges = _showEdges;
		showFaces = _showFaces;
	}

	ZSPACE_INLINE void zObjMesh::setShowVertices(bool _showVerts)
	{
		showVertices = _showVerts;
	}

	ZSPACE_INLINE void zObjMesh::setShowEdges(bool _showEdges)
	{
		showEdges = _showEdges;
	}

	ZSPACE_INLINE void zObjMesh::setShowFaces(bool _showFaces)
	{
		showFaces = _showFaces;
	}

	ZSPACE_INLINE void zObjMesh::setShowDihedralEdges(bool _showDihedralEdges, zDoubleArray &_angles, double _threshold)
	{
		edge_dihedralAngles = _angles;

		showDihedralEdges = _showDihedralEdges;

		dihedralAngleThreshold = _threshold;

		if (_showDihedralEdges) showEdges = false;
	}

	ZSPACE_INLINE void zObjMesh::setShowVertexNormals(bool _showVertexNormal, double _normalScale)
	{
		showVertexNormals = _showVertexNormal;

		normalScale = _normalScale;
	}

	ZSPACE_INLINE void zObjMesh::setShowFaceNormals(bool _showFaceNormal, vector<zVector> &_faceCenters, double _normalScale)
	{
		showFaceNormals = _showFaceNormal;

		normalScale = _normalScale;

		faceCenters = _faceCenters;
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

	ZSPACE_INLINE void zObjMesh::draw()
	{
		if (showObject)
		{
			drawMesh();

			if (showDihedralEdges) drawMesh_DihedralEdges();

			if (showVertexNormals) drawMesh_VertexNormals();

			if (showFaceNormals) drawMesh_FaceNormals();
		}

		if (showObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}

	}

	ZSPACE_INLINE void zObjMesh::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		coreUtils->getBounds(mesh.vertexPositions, minBB, maxBB);
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zObjMesh::drawMesh()
	{

		//draw vertex
		if (showVertices)
		{

			displayUtils->drawVertices(mesh.vHandles, &mesh.vertexPositions[0], &mesh.vertexColors[0], &mesh.vertexWeights[0]);

		}


		//draw edges
		if (showEdges)
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


		//draw polygon
		if (showFaces)
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
	}

	ZSPACE_INLINE void zObjMesh::drawMesh_DihedralEdges()
	{
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
		if (mesh.faceNormals.size() == 0 || mesh.faceNormals.size() != mesh.faces.size()) throw std::invalid_argument(" error: mesh normals not computed.");

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

	//---- DISPLAY BUFFER METHODS

	ZSPACE_INLINE void zObjMesh::appendToBuffer(zDoubleArray edge_dihedralAngles, bool DihedralEdges, double angleThreshold)
	{
		showObject = showEdges = showVertices = showFaces = false;

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
}