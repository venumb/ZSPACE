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


#include<headers/zInterface/iterators/zItMesh.h>

//---- ZIT_MESH_VERTEX ------------------------------------------------------------------------------

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshVertex::zItMeshVertex()
	{
		meshObj = nullptr;
	}

	ZSPACE_INLINE zItMeshVertex::zItMeshVertex(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.vertices.begin();
	}

	ZSPACE_INLINE zItMeshVertex::zItMeshVertex(zObjMesh &_meshObj, int _index)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.vertices.begin();

		if (_index < 0 && _index >= meshObj->mesh.vertices.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItMeshVertex::begin()
	{
		iter = meshObj->mesh.vertices.begin();
	}

	ZSPACE_INLINE void zItMeshVertex::operator++(int)
	{
		iter++;	
	}

	ZSPACE_INLINE void zItMeshVertex::operator--(int)
	{
		iter--;	
	}

	ZSPACE_INLINE bool zItMeshVertex::end()
	{
		return (iter == meshObj->mesh.vertices.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshVertex::reset()
	{
		iter = meshObj->mesh.vertices.begin();
	}

	ZSPACE_INLINE int zItMeshVertex::size()
	{

		return meshObj->mesh.vertices.size();
	}

	ZSPACE_INLINE void zItMeshVertex::deactivate()
	{
		meshObj->mesh.vHandles[iter->getId()] = zVertexHandle();
		iter->reset();
	}

	//---- TOPOLOGY QUERY METHODS

	ZSPACE_INLINE void zItMeshVertex::getConnectedHalfEdges(zItMeshHalfEdgeArray& halfedges)
	{
		if (!this->iter->getHalfEdge()) return;

		if (!getHalfEdge().isActive()) return;

		zItMeshHalfEdge start = getHalfEdge();
		zItMeshHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedges.push_back(e);
			e = e.getPrev().getSym();

		} while (e != start);
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedHalfEdges(zIntArray& halfedgeIndicies)
	{
		if (!this->iter->getHalfEdge()) return;

		if (!getHalfEdge().isActive()) return;


		zItMeshHalfEdge start = getHalfEdge();
		zItMeshHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedgeIndicies.push_back(e.getId());
			e = e.getPrev().getSym();

		} while (e != start);
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedEdges(zItMeshEdgeArray& edges)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edges.push_back(he.getEdge());
		}
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedEdges(zIntArray& edgeIndicies)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edgeIndicies.push_back(he.getEdge().getId());
		}
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedVertices(zItMeshVertexArray& verticies)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			verticies.push_back(he.getVertex());
		}
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedVertices(zIntArray& vertexIndicies)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			vertexIndicies.push_back(he.getVertex().getId());
		}
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedFaces(zItMeshFaceArray& faces)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			if (!he.onBoundary()) faces.push_back(he.getFace());
		}
	}

	ZSPACE_INLINE void zItMeshVertex::getConnectedFaces(zIntArray& faceIndicies)
	{
		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			if (!he.onBoundary()) faceIndicies.push_back(he.getFace().getId());
		}
	}

	ZSPACE_INLINE bool zItMeshVertex::onBoundary()
	{
		bool out = false;

		zItMeshHalfEdgeArray cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			if (he.onBoundary())
			{
				out = true;
				break;
			}

		}

		return out;
	}

	ZSPACE_INLINE int zItMeshVertex::getValence()
	{
		int out;

		zIntArray cHEdges;
		getConnectedHalfEdges(cHEdges);

		out = cHEdges.size();

		return out;
	}

	ZSPACE_INLINE bool zItMeshVertex::checkValency(int valence)
	{
		bool out = false;
		out = (getValence() == valence) ? true : false;

		return out;
	}

	ZSPACE_INLINE zCurvature zItMeshVertex::getPrincipalCurvature()
	{
		double angleSum = 0;
		double cotangentSum = 0;
		double areaSum = 0;
		double areaSumMixed = 0;
		double edgeLengthSquare = 0;
		float gaussianCurv = 0;
		float gaussianAngle = 0;

		zCurvature curv;
		curv.k1 = 0;
		curv.k2 = 0;

		zVector meanCurvNormal;


		if (!onBoundary())
		{
			zItMeshVertexArray cVerts;
			getConnectedVertices(cVerts);

			zVector pt = getPosition();

			float multFactor = 0.125;

			int i = 0;
			for (auto v : cVerts)
			{
				int next = (i + 1) % cVerts.size();
				int prev = (i + cVerts.size() - 1) % cVerts.size();

				zVector pt1 = v.getPosition();
				zVector pt2 = cVerts[next].getPosition();
				zVector pt3 = cVerts[prev].getPosition();

				zVector p01 = pt - pt1;
				zVector p02 = pt - pt2;
				zVector p10 = pt1 - pt;
				zVector p20 = pt2 - pt;
				zVector p12 = pt1 - pt2;
				zVector p21 = pt2 - pt1;
				zVector p31 = pt3 - pt1;

				zVector cr = (p10) ^ (p20);

				float ang = (p10).angle(p20);
				angleSum += ang;
				cotangentSum += (((p20)*(p10)) / cr.length());


				float e_Length = (pt1 - pt2).length();

				edgeLengthSquare += (e_Length * e_Length);

				zVector cr_alpha = (p01) ^ (p21);
				zVector cr_beta = (p01) ^ (p31);

				float coTan_alpha = (((p01)*(p21)) / cr_alpha.length());
				float coTan_beta = (((p01)*(p31)) / cr_beta.length());

				// check if triangle is obtuse
				if ((p10).angle(p20) <= 90 && (p01).angle(p21) <= 90 && (p12).angle(p02) <= 90)
				{
					areaSumMixed += (coTan_alpha + coTan_beta) * edgeLengthSquare * 0.125;
				}
				else
				{

					double triArea = (((p10) ^ (p20)).length()) / 2;

					if ((ang) <= 90) areaSumMixed += triArea * 0.25;
					else areaSumMixed += triArea * 0.5;

				}

				meanCurvNormal += ((pt - pt1)*(coTan_alpha + coTan_beta));

				i++;
			}

			meanCurvNormal /= (2 * areaSumMixed);

			gaussianCurv = (360 - angleSum) / ((0.5 * areaSum) - (multFactor * cotangentSum * edgeLengthSquare));
			//outGauss.push_back(gaussianCurv);

			////// Based on Discrete Differential-Geometry Operators for Triangulated 2-Manifolds

			//gaussianCurv = (360 - angleSum) / areaSumMixed;

			double meanCurv = (meanCurvNormal.length() / 2);
			//if (meanCurv <0.001) meanCurv = 0;

			double deltaX = (meanCurv*meanCurv) - gaussianCurv;
			if (deltaX < 0) deltaX = 0;


			curv.k1 = meanCurv + sqrt(deltaX);
			curv.k2 = meanCurv - sqrt(deltaX);


		}

		return curv;
	}

	ZSPACE_INLINE double zItMeshVertex::getArea()
	{
		vector<zVector> cFCenters, cECenters;

		if (isActive())
		{
			zItMeshHalfEdgeArray cEdges;
			getConnectedHalfEdges(cEdges);


			for (auto &he : cEdges)
			{
				cECenters.push_back(he.getCenter());
				cFCenters.push_back(he.getFace().getCenter());
			}


			double vArea = 0;

			for (int j = 0; j < cEdges.size(); j++)
			{
				int curId = j;
				int nextId = (j + 1) % cEdges.size();

				zItMeshHalfEdge cE = cEdges[j];
				zItMeshHalfEdge nE = cEdges[(j + 1) % cEdges.size()];

				if (cE.onBoundary() || nE.getSym().onBoundary()) continue;

				if (cE.getFace().getId() != nE.getSym().getFace().getId()) continue;

				zVector vPos = getPosition();
				zVector fCen = cFCenters[curId];
				zVector currentEdge_cen = cECenters[curId];
				zVector nextEdge_cen = cECenters[nextId];

				double Area1 = meshObj->mesh.coreUtils.getTriangleArea(vPos, currentEdge_cen, fCen);
				vArea += (Area1);

				double Area2 = meshObj->mesh.coreUtils.getTriangleArea(vPos, nextEdge_cen, fCen);
				vArea += (Area2);
			}

			return vArea;

		}

		else return 0;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshVertex::getId()
	{
		return iter->getId();
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshVertex::getHalfEdge()
	{
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge()->getId());
	}

	ZSPACE_INLINE zItVertex zItMeshVertex::getRawIter()
	{
		return iter;
	}

	ZSPACE_INLINE zVector zItMeshVertex::getPosition()
	{
		return meshObj->mesh.vertexPositions[getId()];
	}

	ZSPACE_INLINE zVector* zItMeshVertex::getRawPosition()
	{
		return &meshObj->mesh.vertexPositions[getId()];
	}

	ZSPACE_INLINE zVector zItMeshVertex::getNormal()
	{
		return meshObj->mesh.vertexNormals[getId()];
	}

	ZSPACE_INLINE zVector* zItMeshVertex::getRawNormal()
	{
		return &meshObj->mesh.vertexNormals[getId()];
	}

	ZSPACE_INLINE void zItMeshVertex::getNormals(vector<zVector> &vNormals)
	{
		vNormals.clear();
		zItMeshFaceArray cFaces;
		getConnectedFaces(cFaces);

		for (auto &f : cFaces)
		{
			vNormals.push_back(f.getNormal());
		}
	}

	ZSPACE_INLINE zColor zItMeshVertex::getColor()
	{
		return meshObj->mesh.vertexColors[getId()];
	}

	ZSPACE_INLINE zColor* zItMeshVertex::getRawColor()
	{
		return &meshObj->mesh.vertexColors[getId()];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshVertex::setId(int _id)
	{
		iter->setId(_id);
	}

	ZSPACE_INLINE void zItMeshVertex::setHalfEdge(zItMeshHalfEdge &he)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int heId = he.getId();

		meshObj->mesh.vHandles[id].he = heId;
	}

	ZSPACE_INLINE void zItMeshVertex::setPosition(zVector &pos)
	{
		meshObj->mesh.vertexPositions[getId()] = pos;
	}

	ZSPACE_INLINE void zItMeshVertex::setColor(zColor col)
	{
		meshObj->mesh.vertexColors[getId()] = col;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zItMeshVertex::isActive()
	{
		return iter->isActive();
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshVertex::operator==(zItMeshVertex &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshVertex::operator!=(zItMeshVertex &other)
	{
		return (getId() != other.getId());
	}

}

//---- ZIT_MESH_EDGE ------------------------------------------------------------------------------

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshEdge::zItMeshEdge()
	{
		meshObj = nullptr;
	}

	ZSPACE_INLINE zItMeshEdge::zItMeshEdge(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.edges.begin();
	}

	ZSPACE_INLINE zItMeshEdge::zItMeshEdge(zObjMesh &_meshObj, int _index)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.edges.begin();

		if (_index < 0 && _index >= meshObj->mesh.edges.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	//--------------------------
	//---- OVERRIDE METHODS
	//--------------------------

	ZSPACE_INLINE void zItMeshEdge::begin()
	{
		iter = meshObj->mesh.edges.begin();
	}

	ZSPACE_INLINE void zItMeshEdge::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItMeshEdge::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItMeshEdge::end()
	{
		return (iter == meshObj->mesh.edges.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshEdge::reset()
	{
		iter = meshObj->mesh.edges.begin();
	}

	ZSPACE_INLINE int zItMeshEdge::size()
	{

		return meshObj->mesh.edges.size();
	}

	ZSPACE_INLINE void zItMeshEdge::deactivate()
	{
		meshObj->mesh.eHandles[iter->getId()] = zEdgeHandle();
		iter->reset();


	}

	//--- TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItMeshEdge::getVertices(zItMeshVertexArray &verticies)
	{
		verticies.push_back(getHalfEdge(0).getVertex());
		verticies.push_back(getHalfEdge(1).getVertex());
	}

	ZSPACE_INLINE void zItMeshEdge::getVertices(zIntArray &vertexIndicies)
	{
		vertexIndicies.push_back(getHalfEdge(0).getVertex().getId());
		vertexIndicies.push_back(getHalfEdge(1).getVertex().getId());
	}

	ZSPACE_INLINE void zItMeshEdge::getVertexPositions(vector<zVector> &vertPositions)
	{
		zIntArray eVerts;

		getVertices(eVerts);

		for (int i = 0; i < eVerts.size(); i++)
		{
			vertPositions.push_back(meshObj->mesh.vertexPositions[eVerts[i]]);
		}
	}

	ZSPACE_INLINE void zItMeshEdge::getFaces(zItMeshFaceArray &faces)
	{
		faces.clear();

		if (!getHalfEdge(0).onBoundary()) faces.push_back(getHalfEdge(0).getFace());
		if (!getHalfEdge(1).onBoundary()) faces.push_back(getHalfEdge(1).getFace());
	}

	ZSPACE_INLINE void zItMeshEdge::getFaces(zIntArray &faceIndicies)
	{
		faceIndicies.clear();

		if (!getHalfEdge(0).onBoundary()) faceIndicies.push_back(getHalfEdge(0).getFace().getId());
		if (!getHalfEdge(1).onBoundary()) faceIndicies.push_back(getHalfEdge(1).getFace().getId());
	}

	ZSPACE_INLINE bool zItMeshEdge::onBoundary()
	{
		return (getHalfEdge(0).onBoundary() || getHalfEdge(1).onBoundary());
	}

	ZSPACE_INLINE zVector zItMeshEdge::getCenter()
	{
		zIntArray eVerts;
		getVertices(eVerts);

		return (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;
	}

	ZSPACE_INLINE zVector zItMeshEdge::getVector()
	{

		int v1 = getHalfEdge(0).getVertex().getId();
		int v2 = getHalfEdge(1).getVertex().getId();

		zVector out = meshObj->mesh.vertexPositions[v1] - (meshObj->mesh.vertexPositions[v2]);

		return out;
	}

	ZSPACE_INLINE double zItMeshEdge::getLength()
	{
		return getVector().length();
	}

	ZSPACE_INLINE double zItMeshEdge::getDihedralAngle()
	{
		if (isActive())
		{
			if (!onBoundary())
			{
				// get connected face to edge
				zIntArray cFaces;
				getFaces(cFaces);

				zVector n0 = meshObj->mesh.faceNormals[cFaces[0]];
				zVector n1 = meshObj->mesh.faceNormals[cFaces[1]];

				zVector eVec = getVector();

				double di_ang;
				di_ang = eVec.dihedralAngle(n0, n1);

				// per edge
				return (di_ang);
			}
			else
			{
				// per  edge
				return (-1);

			}
		}
		else return -2;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshEdge::getId()
	{
		return iter->getId();
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshEdge::getHalfEdge(int _index)
	{	
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge(_index)->getId());
	}

	ZSPACE_INLINE zItEdge  zItMeshEdge::getRawIter()
	{
		return iter;
	}

	ZSPACE_INLINE zColor zItMeshEdge::getColor()
	{
		return meshObj->mesh.edgeColors[getId()];
	}

	ZSPACE_INLINE zColor* zItMeshEdge::getRawColor()
	{
		return &meshObj->mesh.edgeColors[getId()];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshEdge::setId(int _id)
	{
		iter->setId(_id);
	}

	ZSPACE_INLINE void zItMeshEdge::setHalfEdge(zItMeshHalfEdge &he, int _index)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()], _index);

		int id = getId();
		int heId = he.getId();

		if (_index == 0) meshObj->mesh.eHandles[id].he0 = heId;
		if (_index == 1) meshObj->mesh.eHandles[id].he1 = heId;
	}

	ZSPACE_INLINE void zItMeshEdge::setColor(zColor col)
	{
		meshObj->mesh.edgeColors[getId()] = col;
	}

	ZSPACE_INLINE void zItMeshEdge::setWeight(double wt)
	{
		meshObj->mesh.edgeWeights[getId()] = wt;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zItMeshEdge::isActive()
	{
		return iter->isActive();
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshEdge::operator==(zItMeshEdge &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshEdge::operator!=(zItMeshEdge &other)
	{
		return (getId() != other.getId());
	}

}

//---- ZIT_MESH_FACE ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshFace::zItMeshFace()
	{
		meshObj = nullptr;
	}

	ZSPACE_INLINE zItMeshFace::zItMeshFace(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.faces.begin();
	}

	ZSPACE_INLINE zItMeshFace::zItMeshFace(zObjMesh &_meshObj, int _index)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.faces.begin();

		if (_index < 0 && _index >= meshObj->mesh.faces.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItMeshFace::begin()
	{
		iter = meshObj->mesh.faces.begin();
	}

	ZSPACE_INLINE void zItMeshFace::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItMeshFace::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItMeshFace::end()
	{
		return (iter == meshObj->mesh.faces.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshFace::reset()
	{
		iter = meshObj->mesh.faces.begin();
	}

	ZSPACE_INLINE int zItMeshFace::size()
	{

		return meshObj->mesh.faces.size();
	}

	ZSPACE_INLINE void zItMeshFace::deactivate()
	{
		if (iter != meshObj->mesh.faces.end())
		{		
			meshObj->mesh.fHandles[iter->getId()] = zFaceHandle();
			iter->reset();
		}

	}

	//--- TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE void zItMeshFace::getHalfEdges(zItMeshHalfEdgeArray &halfedges)
	{
		halfedges.clear();


		if (!getHalfEdge().onBoundary())
		{
			zItMeshHalfEdge start = getHalfEdge();
			zItMeshHalfEdge e = getHalfEdge();

			bool exit = false;

			do
			{
				halfedges.push_back(e);
				e = e.getNext();

			} while (e != start);
		}

	}

	ZSPACE_INLINE void zItMeshFace::getHalfEdges(zIntArray &halfedgeIndicies)
	{
		halfedgeIndicies.clear();


		if (!getHalfEdge().onBoundary())
		{
			zItMeshHalfEdge start = getHalfEdge();
			zItMeshHalfEdge e = getHalfEdge();

			bool exit = false;

			do
			{
				halfedgeIndicies.push_back(e.getId());
				e = e.getNext();

			} while (e != start);
		}
	}

	ZSPACE_INLINE void zItMeshFace::getVertices(zItMeshVertexArray &verticies)
	{
		zItMeshHalfEdgeArray faceHEdges;
		getHalfEdges(faceHEdges);

		for (auto &he : faceHEdges)
		{
			verticies.push_back(he.getSym().getVertex());
		}
	}

	ZSPACE_INLINE void zItMeshFace::getVertices(zIntArray &vertexIndicies)
	{
		zItMeshHalfEdgeArray faceHEdges;
		getHalfEdges(faceHEdges);

		for (auto &he : faceHEdges)
		{
			vertexIndicies.push_back(he.getSym().getVertex().getId());
		}
	}

	ZSPACE_INLINE void zItMeshFace::getVertexPositions(vector<zVector> &vertPositions)
	{
		zIntArray fVerts;

		getVertices(fVerts);

		for (int i = 0; i < fVerts.size(); i++)
		{
			vertPositions.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);
		}
	}

	ZSPACE_INLINE void zItMeshFace::getConnectedFaces(zItMeshFaceArray& faces)
	{
		zItMeshHalfEdgeArray cHEdges;
		getHalfEdges(cHEdges);		

		for (auto &he : cHEdges)
		{
			zItMeshFaceArray eFaces;
			he.getFaces(eFaces);

			printf("\n eFaces %i", eFaces.size());

			for (int k = 0; k < eFaces.size(); k++)
			{
				if (eFaces[k].getId() != getId()) faces.push_back(eFaces[k]);
			}
		}

		printf("\n %i  e %i %i", getId(), cHEdges.size(), faces.size());
	}

	ZSPACE_INLINE void zItMeshFace::getConnectedFaces(zIntArray& faceIndicies)
	{
		zItMeshHalfEdgeArray cHEdges;
		getHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			zIntArray eFaces;
			he.getFaces(eFaces);

			for (int k = 0; k < eFaces.size(); k++)
			{
				if (eFaces[k] != getId()) faceIndicies.push_back(eFaces[k]);
			}
		}
	}

	ZSPACE_INLINE bool zItMeshFace::onBoundary()
	{
		bool out = false;

		zItMeshHalfEdgeArray cHEdges;
		getHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			if (he.onBoundary())
			{
				out = true;
				break;
			}
		}

		return out;
	}

	ZSPACE_INLINE zVector zItMeshFace::getCenter()
	{
		zIntArray fVerts;
		getVertices(fVerts);
		zVector cen;

		for (int j = 0; j < fVerts.size(); j++) cen += meshObj->mesh.vertexPositions[fVerts[j]];
		cen /= fVerts.size();

		return cen;
	}

	ZSPACE_INLINE int zItMeshFace::getNumVertices()
	{
		zIntArray fEdges;
		getHalfEdges(fEdges);

		return fEdges.size();
	}

	ZSPACE_INLINE void zItMeshFace::getTriangles(int &numTris, zIntArray &tris)
	{
		double angle_Max = 90;
		bool noEars = true; // check for if there are no ears

		vector<bool> ears;
		vector<bool> reflexVerts;

		// get face vertices

		zIntArray fVerts;

		getVertices(fVerts);
		zIntArray vertexIndices = fVerts;

		int faceIndex = getId();

		vector<zVector> points;
		getVertexPositions(points);


		if (fVerts.size() < 3) throw std::invalid_argument(" error: invalid face, triangulation is not succesful.");

		// compute 			
		zVector norm = meshObj->mesh.faceNormals[faceIndex];

		// compute ears

		for (int i = 0; i < vertexIndices.size(); i++)
		{
			int nextId = (i + 1) % vertexIndices.size();
			int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

			// Triangle edges - e1 and e2 defined above
			zVector v1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];
			zVector v2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];

			zVector cross = v1 ^ v2;
			double ang = v1.angle(v2);

			if (cross * norm < 0) ang *= -1;

			if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
			else reflexVerts.push_back(false);

			// calculate ears
			if (!reflexVerts[i])
			{
				bool ear = true;

				zVector p0 = meshObj->mesh.vertexPositions[fVerts[i]];
				zVector p1 = meshObj->mesh.vertexPositions[fVerts[nextId]];
				zVector p2 = meshObj->mesh.vertexPositions[fVerts[prevId]];

				bool CheckPtTri = false;

				for (int j = 0; j < fVerts.size(); j++)
				{
					if (!CheckPtTri)
					{
						if (j != i && j != nextId && j != prevId)
						{
							// vector to point to be checked
							zVector pt = meshObj->mesh.vertexPositions[fVerts[j]];

							bool Chk = meshObj->mesh.coreUtils.pointInTriangle(pt, p0, p1, p2);
							CheckPtTri = Chk;

						}
					}

				}

				if (CheckPtTri) ear = false;
				ears.push_back(ear);

				if (noEars && ear) noEars = !noEars;
			}
			else ears.push_back(false);

			//printf("\n id: %i ang: %1.2f reflex: %s ear: %s", vertexIndices[i], ang, (reflexVerts[i] == true) ? "true" : "false",(ears[i] == true)?"true":"false");
		}

		if (noEars)
		{
			for (int i = 0; i < fVerts.size(); i++)
			{
				//printf("\n %1.2f %1.2f %1.2f ", points[i].x, points[i].y, points[i].z);
			}

			throw std::invalid_argument(" error: no ears found in the face, triangulation is not succesful.");
		}

		int maxTris = fVerts.size() - 2;

		// // triangulate 

		while (numTris < maxTris - 1)
		{
			printf("\n working!");

			int earId = -1;
			bool earFound = false;;

			for (int i = 0; i < ears.size(); i++)
			{
				if (!earFound)
				{
					if (ears[i])
					{
						earId = i;
						earFound = !earFound;
					}
				}

			}

			if (earFound)
			{


				for (int i = -1; i <= 1; i++)
				{
					int id = (earId + i + vertexIndices.size()) % vertexIndices.size();
					tris.push_back(vertexIndices[id]);
				}
				numTris++;

				// remove vertex earid 
				vertexIndices.erase(vertexIndices.begin() + earId);

				reflexVerts.clear();
				ears.clear();

				// check for ears
				for (int i = 0; i < vertexIndices.size(); i++)
				{

					int nextId = (i + 1) % vertexIndices.size();
					int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

					// Triangle edges - e1 and e2 defined above
					zVector v1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];
					zVector v2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];

					zVector cross = v1 ^ v2;
					double ang = v1.angle(v2);

					if (cross * norm < 0) ang *= -1;

					if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
					else reflexVerts.push_back(false);

					// calculate ears
					if (!reflexVerts[i])
					{
						bool ear = true;

						zVector p0 = meshObj->mesh.vertexPositions[vertexIndices[i]];
						zVector p1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]];
						zVector p2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]];

						bool CheckPtTri = false;

						for (int j = 0; j < vertexIndices.size(); j++)
						{
							if (!CheckPtTri)
							{
								if (j != i && j != nextId && j != prevId)
								{
									// vector to point to be checked
									zVector pt = meshObj->mesh.vertexPositions[vertexIndices[j]];

									bool Chk = meshObj->mesh.coreUtils.pointInTriangle(pt, p0, p1, p2);
									CheckPtTri = Chk;
								}
							}

						}

						if (CheckPtTri) ear = false;
						ears.push_back(ear);

					}
					else ears.push_back(false);


					//printf("\n earId %i id: %i ang: %1.2f reflex: %s ear: %s", earId, vertexIndices[i], ang, (reflexVerts[i] == true) ? "true" : "false", (ears[i] == true) ? "true" : "false");
				}



			}
			else
			{
				for (int i = 0; i < vertexIndices.size(); i++)
				{
					//printf("\n %1.2f %1.2f %1.2f ", meshObj->mesh.vertexPositions[vertexIndices[i]].x, meshObj->mesh.vertexPositions[vertexIndices[i]].y, meshObj->mesh.vertexPositions[vertexIndices[i]].z);
				}

				throw std::invalid_argument(" error: no ears found in the face, triangulation is not succesful.");
			}

		}

		// add the last remaining triangle
		tris.push_back(vertexIndices[0]);
		tris.push_back(vertexIndices[1]);
		tris.push_back(vertexIndices[2]);
		numTris++;

	}

	ZSPACE_INLINE double zItMeshFace::getVolume(zIntArray &faceTris, zVector &fCenter, bool absoluteVolume)
	{

		int faceNumTris;

		if (faceTris.size() == 0) 	getTriangles(faceNumTris, faceTris);
		if (fCenter == zVector(0, 0, 0)) fCenter = getCenter();

		double out = 0;

		int index = getId();

		// add volume of face tris			
		for (int j = 0; j < faceTris.size(); j += 3)
		{
			double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(meshObj->mesh.vertexPositions[faceTris[j + 0]], meshObj->mesh.vertexPositions[faceTris[j + 1]], meshObj->mesh.vertexPositions[faceTris[j + 2]]);

			out += vol;
		}

		// add volumes of tris formes by each pair of face edge vertices and face center

		vector<zVector> fVerts;
		getVertexPositions(fVerts);

		for (int j = 0; j < fVerts.size(); j += 1)
		{
			int prevId = (j - 1 + fVerts.size()) % fVerts.size();

			double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(fVerts[j], fVerts[prevId], fCenter);

			out += vol;
		}

		if (absoluteVolume) out = abs(out);

		return out;

	}

	ZSPACE_INLINE double zItMeshFace::getPlanarFaceArea()
	{
		double fArea = 0;

		if (isActive())
		{
			zVector fNorm = getNormal();

			vector<zVector> fVerts;
			getVertexPositions(fVerts);

			for (int j = 0; j < fVerts.size(); j++)
			{
				zVector v1 = fVerts[j];
				zVector v2 = fVerts[(j + 1) % fVerts.size()];


				fArea += fNorm * (v1 ^ v2);
			}

			fArea *= 0.5;

		}

		return fArea;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshFace::getId()
	{
		return iter->getId();
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshFace::getHalfEdge()
	{
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge()->getId());
	}

	ZSPACE_INLINE zItFace  zItMeshFace::getRawIter()
	{
		return iter;
	}

	ZSPACE_INLINE void zItMeshFace::getOffsetFacePositions(double offset, vector<zVector>& offsetPositions)
	{
		vector<zVector> out;

		zIntArray fVerts;
		getVertices(fVerts);

		for (int j = 0; j < fVerts.size(); j++)
		{
			int next = (j + 1) % fVerts.size();
			int prev = (j - 1 + fVerts.size()) % fVerts.size();


			zVector Ori = meshObj->mesh.vertexPositions[fVerts[j]];;
			zVector v1 = meshObj->mesh.vertexPositions[fVerts[prev]] - meshObj->mesh.vertexPositions[fVerts[j]];
			v1.normalize();

			zVector v2 = meshObj->mesh.vertexPositions[fVerts[next]] - meshObj->mesh.vertexPositions[fVerts[j]];
			v2.normalize();

			zVector v3 = v1;

			v1 = v1 ^ v2;
			v3 = v3 + v2;

			double cs = v3 * v2;

			zVector a1 = v2 * cs;
			zVector a2 = v3 - a1;

			double alpha = sqrt(a2.length() * a2.length());
			if (cs < 0) alpha *= -1;

			double length = offset / alpha;

			zVector mPos = meshObj->mesh.vertexPositions[fVerts[j]];
			zVector offPos = mPos + (v3 * length);

			out.push_back(offPos);

		}

		offsetPositions = out;

	}

	ZSPACE_INLINE void zItMeshFace::getOffsetFacePositions_Variable(vector<double>& offsets, zVector& faceCenter, zVector& faceNormal, vector<zVector>& intersectionPositions)
	{
		vector<zVector> offsetPoints;
		zIntArray fEdges;
		getHalfEdges(fEdges);

		for (int j = 0; j < fEdges.size(); j++)
		{
			zItMeshHalfEdge he(*meshObj, fEdges[j]);


			zVector p2 = meshObj->mesh.vertexPositions[he.getVertex().getId()];
			zVector p1 = meshObj->mesh.vertexPositions[he.getSym().getVertex().getId()];

			zVector norm1 = ((p1 - p2) ^ faceNormal);
			norm1.normalize();
			if ((faceCenter - p1) * norm1 < 0) norm1 *= -1;


			offsetPoints.push_back(p1 + norm1 * offsets[j]);
			offsetPoints.push_back(p2 + norm1 * offsets[j]);

		}


		for (int j = 0; j < fEdges.size(); j++)
		{
			int prevId = (j - 1 + fEdges.size()) % fEdges.size();

			zVector a0 = offsetPoints[j * 2];
			zVector a1 = offsetPoints[j * 2 + 1];

			zVector b0 = offsetPoints[prevId * 2];
			zVector b1 = offsetPoints[prevId * 2 + 1];



			double uA = -1;
			double uB = -1;
			bool intersect = meshObj->mesh.coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

			if (intersect)
			{
				//printf("\n %i working!! ", j);

				zVector closestPt;

				if (uA >= uB)
				{
					zVector dir = a1 - a0;
					double len = dir.length();
					dir.normalize();

					if (uA < 0) dir *= -1;
					closestPt = a0 + dir * len * uA;
				}
				else
				{
					zVector dir = b1 - b0;
					double len = dir.length();
					dir.normalize();

					if (uB < 0) dir *= -1;

					closestPt = b0 + dir * len * uB;
				}


				intersectionPositions.push_back(closestPt);
			}

		}
	}

	ZSPACE_INLINE zVector zItMeshFace::getNormal()
	{
		return meshObj->mesh.faceNormals[getId()];
	}

	ZSPACE_INLINE zVector* zItMeshFace::getRawNormal()
	{
		return &meshObj->mesh.faceNormals[getId()];

	}

	ZSPACE_INLINE zColor zItMeshFace::getColor()
	{
		return meshObj->mesh.faceColors[getId()];
	}

	ZSPACE_INLINE zColor* zItMeshFace::getRawColor()
	{
		return &meshObj->mesh.faceColors[getId()];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshFace::setId(int _id)
	{
		iter->setId(_id);
	}

	ZSPACE_INLINE void zItMeshFace::setHalfEdge(zItMeshHalfEdge &he)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int heId = he.getId();

		meshObj->mesh.fHandles[id].he = heId;
	}

	ZSPACE_INLINE void zItMeshFace::setColor(zColor col)
	{
		meshObj->mesh.faceColors[getId()] = col;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zItMeshFace::isActive()
	{
		return iter->isActive();
	}

	ZSPACE_INLINE bool zItMeshFace::checkPointInHalfSpace(zPoint & pt)
	{
		zVector n = getNormal();
		zPoint p = getHalfEdge().getVertex().getPosition();

		double D = n * p* -1;
		double Dis = (pt*n) + D;

		bool out = (Dis / sqrt(n*n) < 0) ? true : false;

		return out;
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshFace::operator==(zItMeshFace &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshFace::operator!=(zItMeshFace &other)
	{
		return (getId() != other.getId());
	}

}

//---- ZIT_MESH_HALFEDGE ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zItMeshHalfEdge::zItMeshHalfEdge()
	{
		meshObj = nullptr;
	}

	ZSPACE_INLINE zItMeshHalfEdge::zItMeshHalfEdge(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.halfEdges.begin();
	}

	ZSPACE_INLINE zItMeshHalfEdge::zItMeshHalfEdge(zObjMesh &_meshObj, int _index)
	{
		meshObj = &_meshObj;

		iter = meshObj->mesh.halfEdges.begin();

		if (_index < 0 && _index >= meshObj->mesh.halfEdges.size()) throw std::invalid_argument(" error: index out of bounds");
		advance(iter, _index);
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zItMeshHalfEdge::begin()
	{
		iter = meshObj->mesh.halfEdges.begin();
	}

	ZSPACE_INLINE void zItMeshHalfEdge::operator++(int)
	{
		iter++;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::operator--(int)
	{
		iter--;
	}

	ZSPACE_INLINE bool zItMeshHalfEdge::end()
	{
		return (iter == meshObj->mesh.halfEdges.end()) ? true : false;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::reset()
	{
		iter = meshObj->mesh.halfEdges.begin();
	}

	ZSPACE_INLINE int zItMeshHalfEdge::size()
	{

		return meshObj->mesh.halfEdges.size();
	}

	ZSPACE_INLINE void zItMeshHalfEdge::deactivate()
	{
		meshObj->mesh.heHandles[iter->getId()] = zHalfEdgeHandle();
		iter->reset();
	}

	//--- TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE zItMeshVertex zItMeshHalfEdge::getStartVertex()
	{
		if (!isActive()) throw std::invalid_argument(" error: out of bounds.");

		return getSym().getVertex();
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getVertices(zItMeshVertexArray &verticies)
	{
		verticies.push_back(getVertex());
		verticies.push_back(getSym().getVertex());
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getVertices(zIntArray &vertexIndicies)
	{
		vertexIndicies.push_back(getVertex().getId());
		vertexIndicies.push_back(getSym().getVertex().getId());
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getVertexPositions(vector<zVector> &vertPositions)
	{
		zIntArray eVerts;

		getVertices(eVerts);

		for (int i = 0; i < eVerts.size(); i++)
		{
			vertPositions.push_back(meshObj->mesh.vertexPositions[eVerts[i]]);
		}
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getConnectedHalfEdges(zItMeshHalfEdgeArray& edgeIndicies)
	{
		zItMeshVertex v1 = getVertex();
		zItMeshHalfEdgeArray connectedEdgestoVert0;
		v1.getConnectedHalfEdges(connectedEdgestoVert0);

		zItMeshVertex v2 = getSym().getVertex();
		zItMeshHalfEdgeArray connectedEdgestoVert1;
		v2.getConnectedHalfEdges(connectedEdgestoVert1);

		for (int i = 0; i < connectedEdgestoVert0.size(); i++)
		{
			if (connectedEdgestoVert0[i].getId() != getId()) edgeIndicies.push_back(connectedEdgestoVert0[i]);
		}


		for (int i = 0; i < connectedEdgestoVert1.size(); i++)
		{
			if (connectedEdgestoVert1[i].getId() != getId()) edgeIndicies.push_back(connectedEdgestoVert1[i]);
		}
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getConnectedHalfEdges(zIntArray& edgeIndicies)
	{
		zItMeshVertex v1 = getVertex();
		zIntArray connectedEdgestoVert0;
		v1.getConnectedHalfEdges(connectedEdgestoVert0);

		zItMeshVertex v2 = getSym().getVertex();
		zIntArray connectedEdgestoVert1;
		v2.getConnectedHalfEdges(connectedEdgestoVert1);

		for (int i = 0; i < connectedEdgestoVert0.size(); i++)
		{
			if (connectedEdgestoVert0[i] != getId()) edgeIndicies.push_back(connectedEdgestoVert0[i]);
		}


		for (int i = 0; i < connectedEdgestoVert1.size(); i++)
		{
			if (connectedEdgestoVert1[i] != getId()) edgeIndicies.push_back(connectedEdgestoVert1[i]);
		}
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getFaces(zItMeshFaceArray &faceIndicies)
	{
		this->getEdge().getFaces(faceIndicies);
	}

	ZSPACE_INLINE void zItMeshHalfEdge::getFaces(zIntArray &faceIndicies)
	{
		this->getEdge().getFaces(faceIndicies);
	}

	ZSPACE_INLINE bool zItMeshHalfEdge::onBoundary()
	{
		return !iter->getFace();
	}

	ZSPACE_INLINE zVector zItMeshHalfEdge::getCenter()
	{
		zIntArray eVerts;
		getVertices(eVerts);

		return (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;
	}

	ZSPACE_INLINE zVector zItMeshHalfEdge::getVector()
	{

		int v1 = getVertex().getId();
		int v2 = getSym().getVertex().getId();

		zVector out = meshObj->mesh.vertexPositions[v1] - (meshObj->mesh.vertexPositions[v2]);

		return out;
	}

	ZSPACE_INLINE double zItMeshHalfEdge::getLength()
	{
		return getVector().length();
	}

	//---- GET METHODS

	ZSPACE_INLINE int zItMeshHalfEdge::getId()
	{
		return iter->getId();
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshHalfEdge::getSym()
	{
		return zItMeshHalfEdge(*meshObj, iter->getSym()->getId());
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshHalfEdge::getNext()
	{
		return zItMeshHalfEdge(*meshObj, iter->getNext()->getId());
	}

	ZSPACE_INLINE zItMeshHalfEdge zItMeshHalfEdge::getPrev()
	{
		return zItMeshHalfEdge(*meshObj, iter->getPrev()->getId());
	}

	ZSPACE_INLINE zItMeshVertex zItMeshHalfEdge::getVertex()
	{
		return zItMeshVertex(*meshObj, iter->getVertex()->getId());
	}

	ZSPACE_INLINE zItMeshFace zItMeshHalfEdge::getFace()
	{
		return zItMeshFace(*meshObj, iter->getFace()->getId());
	}

	ZSPACE_INLINE zItMeshEdge zItMeshHalfEdge::getEdge()
	{
		return zItMeshEdge(*meshObj, iter->getEdge()->getId());
	}

	ZSPACE_INLINE zItHalfEdge  zItMeshHalfEdge::getRawIter()
	{
		return iter;
	}

	ZSPACE_INLINE zColor zItMeshHalfEdge::getColor()
	{
		return meshObj->mesh.edgeColors[iter->getEdge()->getId()];
	}

	ZSPACE_INLINE zColor* zItMeshHalfEdge::getRawColor()
	{
		return &meshObj->mesh.edgeColors[iter->getEdge()->getId()];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zItMeshHalfEdge::setId(int _id)
	{
		iter->setId(_id);
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setSym(zItMeshHalfEdge &he)
	{
		iter->setSym(&meshObj->mesh.halfEdges[he.getId()]);
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setNext(zItMeshHalfEdge &he)
	{
		iter->setNext(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int nextId = he.getId();

		meshObj->mesh.heHandles[id].n = nextId;
		meshObj->mesh.heHandles[nextId].p = id;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setPrev(zItMeshHalfEdge &he)
	{
		iter->setPrev(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int prevId = he.getId();

		meshObj->mesh.heHandles[id].p = prevId;
		meshObj->mesh.heHandles[prevId].n = id;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setVertex(zItMeshVertex &v)
	{
		iter->setVertex(&meshObj->mesh.vertices[v.getId()]);

		int id = getId();
		int vId = v.getId();

		meshObj->mesh.heHandles[id].v = vId;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setEdge(zItMeshEdge &e)
	{
		iter->setEdge(&meshObj->mesh.edges[e.getId()]);

		int id = getId();
		int eId = e.getId();

		meshObj->mesh.heHandles[id].e = eId;
	}

	ZSPACE_INLINE void zItMeshHalfEdge::setFace(zItMeshFace &f)
	{
		iter->setFace(&meshObj->mesh.faces[f.getId()]);

		int id = getId();
		int fId = f.getId();

		meshObj->mesh.heHandles[id].f = fId;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zItMeshHalfEdge::isActive()
	{
		return iter->isActive();
	}

	//---- OPERATOR METHODS

	ZSPACE_INLINE bool zItMeshHalfEdge::operator==(zItMeshHalfEdge &other)
	{
		return (getId() == other.getId());
	}

	ZSPACE_INLINE bool zItMeshHalfEdge::operator!=(zItMeshHalfEdge &other)
	{
		return (getId() != other.getId());
	}

}