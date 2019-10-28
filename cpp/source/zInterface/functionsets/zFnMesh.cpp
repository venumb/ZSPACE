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


#include<headers/zInterface/functionsets/zFnMesh.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnMesh::zFnMesh()
	{
		fnType = zFnType::zMeshFn; 
		meshObj = nullptr;
	}

	ZSPACE_INLINE zFnMesh::zFnMesh(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;

		fnType = zFnType::zMeshFn;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnMesh::~zFnMesh() {}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE zFnType zFnMesh::getType()
	{
		return zMeshFn;
	}

	ZSPACE_INLINE void zFnMesh::from(string path, zFileTpye type, bool staticGeom )
	{
		if (type == zOBJ)
		{
			bool chk = fromOBJ(path);
			if (chk && staticGeom) setStaticContainers();
		}

		else if (type == zJSON)
		{
			bool chk = fromJSON(path);
			if (chk && staticGeom) setStaticContainers();
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnMesh::to(string path, zFileTpye type)
	{
		if (type == zOBJ) toOBJ(path);
		else if (type == zJSON) toJSON(path);

		else throw std::invalid_argument(" error: invalid zFileType type");
	}

	ZSPACE_INLINE void zFnMesh::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		meshObj->getBounds(minBB, maxBB);
	}

	ZSPACE_INLINE void zFnMesh::clear()
	{

		meshObj->mesh.clear();
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnMesh::reserve(int _n_v, int  _n_e, int _n_f)
	{
		meshObj->mesh.clear();

		meshObj->mesh.vertices.reserve(_n_v);
		meshObj->mesh.faces.reserve(_n_f);
		meshObj->mesh.edges.reserve(_n_e);
		meshObj->mesh.halfEdges.reserve(_n_e * 2);

	}

	ZSPACE_INLINE void zFnMesh::create(zPointArray& _positions, zIntArray& polyCounts, zIntArray& polyConnects, bool staticMesh)
	{
		meshObj->mesh.create(_positions, polyCounts, polyConnects);

		// compute mesh normals
		computeMeshNormals();

		if (staticMesh) setStaticContainers();

	}

	ZSPACE_INLINE bool zFnMesh::addVertex(zPoint &_pos, bool checkDuplicates, zItMeshVertex &vertex)
	{
		if (checkDuplicates)
		{
			int id;
			bool chk = vertexExists(_pos, vertex);
			if (chk)	return false;

		}

		bool out = meshObj->mesh.addVertex(_pos);
		vertex = zItMeshVertex(*meshObj, numVertices() - 1);

		return out;
	}

	ZSPACE_INLINE bool zFnMesh::addEdges(int &v1, int &v2, bool checkDuplicates, zItMeshHalfEdge &halfEdge)
	{
		if (v1 < 0 && v1 >= numVertices()) throw std::invalid_argument(" error: index out of bounds");
		if (v2 < 0 && v2 >= numVertices()) throw std::invalid_argument(" error: index out of bounds");

		if (checkDuplicates)
		{
			int id;
			bool chk = halfEdgeExists(v1, v2, id);
			if (chk)
			{
				halfEdge = zItMeshHalfEdge(*meshObj, id);
				return false;
			}
		}

		bool out = meshObj->mesh.addEdges(v1, v2);

		halfEdge = zItMeshHalfEdge(*meshObj, numHalfEdges() - 2);

		return out;
	}

	ZSPACE_INLINE bool zFnMesh::addPolygon(zIntArray &fVertices, zItMeshFace &face)
	{
		for (auto &v : fVertices)
		{
			if (v < 0 && v >= numVertices()) throw std::invalid_argument(" error: index out of bounds");
		}

		bool out = meshObj->mesh.addPolygon(fVertices);

		face = zItMeshFace(*meshObj, numPolygons() - 1);

		return out;
	}

	ZSPACE_INLINE bool zFnMesh::addPolygon(zPointArray &fVertices, zItMeshFace &face)
	{
		vector<int> fVerts;

		for (auto &v : fVertices)
		{
			zItMeshVertex vId;
			addVertex(v, true, vId);
			fVerts.push_back(vId.getId());
		}

		return addPolygon(fVerts, face);
	}

	ZSPACE_INLINE bool zFnMesh::addPolygon(zItMeshFace &face)
	{
		bool out = meshObj->mesh.addPolygon();
		face = zItMeshFace(*meshObj, numPolygons() - 1);

		return out;
	}

	//--- TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE int zFnMesh::numVertices()
	{
		return meshObj->mesh.n_v;
	}

	ZSPACE_INLINE int zFnMesh::numEdges()
	{
		return meshObj->mesh.n_e;
	}

	ZSPACE_INLINE int zFnMesh::numHalfEdges()
	{
		return meshObj->mesh.n_he;
	}

	ZSPACE_INLINE int zFnMesh::numPolygons()
	{
		return meshObj->mesh.n_f;
	}

	ZSPACE_INLINE bool zFnMesh::vertexExists(zPoint pos, zItMeshVertex &outVertex, int precisionfactor)
	{

		int id;
		bool chk = meshObj->mesh.vertexExists(pos, id, precisionfactor);

		if (chk) outVertex = zItMeshVertex(*meshObj, id);

		return chk;
	}
	
	ZSPACE_INLINE bool zFnMesh::halfEdgeExists(int v1, int v2, int &outHalfEdgeId)
	{
		return meshObj->mesh.halfEdgeExists(v1, v2, outHalfEdgeId);
	}

	ZSPACE_INLINE bool zFnMesh::halfEdgeExists(int v1, int v2, zItMeshHalfEdge &outHalfEdge)
	{
		int id;
		bool chk = halfEdgeExists(v1, v2, id);

		if (chk) outHalfEdge = zItMeshHalfEdge(*meshObj, id);

		return chk;
	}

	ZSPACE_INLINE zSparseMatrix zFnMesh::getTopologicalLaplacian()
	{
		int n_v = numVertices();

		//MatrixXd meshLaplacian(n_v, n_v);
		//meshLaplacian.setZero();

		zSparseMatrix meshLaplacian(n_v, n_v);
		//meshLaplacian.setZero();


		// compute laplacian weights
		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			vector<zItMeshHalfEdge> cEdges;
			v.getConnectedHalfEdges(cEdges);

			int j = v.getId();

			double out = 0;

			for (int k = 0; k < cEdges.size(); k++)
			{

				double val = getEdgeCotangentWeight(cEdges[k]) * 0.5;
				out += val;

				int i = cEdges[k].getVertex().getId();

				meshLaplacian.insert(j, i) = val * -1;
			}

			meshLaplacian.insert(j, j) = out;

			/*for (int i = 0; i < n_v; i++)
			{
				if (i == j)
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					double out = 0;

					for (int k = 0; k < cEdges.size(); k++)
					{
						out += getEdgeCotangentWeight(cEdges[k]) * 0.5;
					}

					meshLaplacian(j, i) = out;
				}
				else
				{
					int e;
					bool chk = edgeExists(i, j, e);

					if (chk)  meshLaplacian(j, i) = getEdgeCotangentWeight(e) * 0.5 * -1;
					else meshLaplacian(j, i) = 0;
				}

			}*/
		}


		return meshLaplacian;
	}

	ZSPACE_INLINE double zFnMesh::getEdgeCotangentWeight(zItMeshHalfEdge &he)
	{

		zItMeshVertex i = he.getStartVertex();
		zItMeshVertex j = he.getVertex();

		zVector* pt = i.getRawPosition();

		zVector* pt1 = j.getRawPosition();

		zItMeshHalfEdge nextEdge = he.getSym().getNext();
		zItMeshVertex nextVert = nextEdge.getVertex();;
		zVector* pt2 = nextVert.getRawPosition();

		zItMeshHalfEdge prevEdge = he.getPrev().getSym();
		zItMeshVertex prevVert = prevEdge.getVertex();
		zVector* pt3 = prevVert.getRawPosition();

		zVector alpha1 = (*pt - *pt2);
		zVector alpha2 = (*pt1 - *pt2);
		double coTan_alpha = alpha1.cotan(alpha2);

		zVector beta1 = (*pt - *pt3);
		zVector beta2 = (*pt1 - *pt3);
		double coTan_beta = beta1.cotan(beta2);

		if (he.onBoundary())coTan_beta = 0;;
		if (he.getSym().onBoundary())coTan_alpha = 0;;

		double wt = coTan_alpha + coTan_beta;

		if (isnan(wt)) wt = 0.0;

		return wt;

	}

	//--- COMPUTE METHODS 

	ZSPACE_INLINE void zFnMesh::computeEdgeColorfromVertexColor()
	{
		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			if (e.isActive())
			{
				int v0 = e.getHalfEdge(0).getVertex().getId();
				int v1 = e.getHalfEdge(0).getVertex().getId();

				zColor col;
				col.r = (meshObj->mesh.vertexColors[v0].r + meshObj->mesh.vertexColors[v1].r) * 0.5;
				col.g = (meshObj->mesh.vertexColors[v0].g + meshObj->mesh.vertexColors[v1].g) * 0.5;
				col.b = (meshObj->mesh.vertexColors[v0].b + meshObj->mesh.vertexColors[v1].b) * 0.5;
				col.a = (meshObj->mesh.vertexColors[v0].a + meshObj->mesh.vertexColors[v1].a) * 0.5;

				if (meshObj->mesh.edgeColors.size() <= e.getId()) meshObj->mesh.edgeColors.push_back(col);
				else meshObj->mesh.edgeColors[e.getId()] = col;


			}


		}



	}

	ZSPACE_INLINE void zFnMesh::computeVertexColorfromEdgeColor()
	{

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (v.isActive())
			{
				vector<int> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zColor col;
				for (int j = 0; j < cEdges.size(); j++)
				{
					col.r += meshObj->mesh.edgeColors[cEdges[j]].r;
					col.g += meshObj->mesh.edgeColors[cEdges[j]].g;
					col.b += meshObj->mesh.edgeColors[cEdges[j]].b;
				}

				col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

				meshObj->mesh.vertexColors[v.getId()] = col;

			}
		}
	}

	ZSPACE_INLINE void zFnMesh::computeFaceColorfromVertexColor()
	{


		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			if (f.isActive())
			{
				vector<int> fVerts;
				f.getVertices(fVerts);

				zColor col;
				for (int j = 0; j < fVerts.size(); j++)
				{
					col.r += meshObj->mesh.vertexColors[fVerts[j]].r;
					col.g += meshObj->mesh.vertexColors[fVerts[j]].g;
					col.b += meshObj->mesh.vertexColors[fVerts[j]].b;
				}

				col.r /= fVerts.size(); col.g /= fVerts.size(); col.b /= fVerts.size();

				meshObj->mesh.faceColors[f.getId()] = col;
			}
		}

	}

	ZSPACE_INLINE void zFnMesh::computeVertexColorfromFaceColor()
	{

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (v.isActive())
			{
				vector<int> cFaces;
				v.getConnectedFaces(cFaces);

				zColor col;
				for (int j = 0; j < cFaces.size(); j++)
				{
					col.r += meshObj->mesh.faceColors[cFaces[j]].r;
					col.g += meshObj->mesh.faceColors[cFaces[j]].g;
					col.b += meshObj->mesh.faceColors[cFaces[j]].b;
				}

				col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

				meshObj->mesh.vertexColors[v.getId()] = col;
			}
		}

	}

	ZSPACE_INLINE void zFnMesh::smoothColors(int smoothVal, zHEData type)
	{
		for (int j = 0; j < smoothVal; j++)
		{
			if (type == zVertexData)
			{
				vector<zColor> tempColors;

				for (zItMeshVertex v(*meshObj); !v.end(); v++)
				{
					zColor col;
					if (v.isActive())
					{
						vector<int> cVerts;
						v.getConnectedVertices(cVerts);

						zColor currentCol = meshObj->mesh.vertexColors[v.getId()];


						for (int j = 0; j < cVerts.size(); j++)
						{
							col.r += meshObj->mesh.vertexColors[cVerts[j]].r;
							col.g += meshObj->mesh.vertexColors[cVerts[j]].g;
							col.b += meshObj->mesh.vertexColors[cVerts[j]].b;
						}

						col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

						col.r /= cVerts.size(); col.g /= cVerts.size(); col.b /= cVerts.size();


					}

					tempColors.push_back(col);

				}

				for (zItMeshVertex v(*meshObj); !v.end(); v++)
				{
					if (v.isActive())
					{
						meshObj->mesh.vertexColors[v.getId()] = (tempColors[v.isActive()]);
					}
				}
			}


			if (type == zFaceData)
			{
				vector<zColor> tempColors;

				for (zItMeshFace f(*meshObj); !f.end(); f++)
				{
					zColor col;
					if (f.isActive())
					{
						vector<int> cFaces;
						f.getConnectedFaces(cFaces);

						zColor currentCol = meshObj->mesh.faceColors[f.getId()];
						for (int j = 0; j < cFaces.size(); j++)
						{
							col.r += meshObj->mesh.faceColors[cFaces[j]].r;
							col.g += meshObj->mesh.faceColors[cFaces[j]].g;
							col.b += meshObj->mesh.faceColors[cFaces[j]].b;
						}

						col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

						col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

					}

					tempColors.push_back(col);

				}

				for (zItMeshFace f(*meshObj); !f.end(); f++)
				{
					if (f.isActive())
					{
						meshObj->mesh.faceColors[f.getId()] = (tempColors[f.getId()]);
					}
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}



	}

	ZSPACE_INLINE void zFnMesh::computeVertexNormalfromFaceNormal()
	{

		meshObj->mesh.vertexNormals.clear();

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (v.isActive())
			{
				vector<int> cFaces;
				v.getConnectedFaces(cFaces);

				zVector norm;

				for (int j = 0; j < cFaces.size(); j++)
				{
					norm += meshObj->mesh.faceNormals[cFaces[j]];
				}

				norm /= cFaces.size();
				norm.normalize();
				meshObj->mesh.vertexNormals.push_back(norm);
			}
			else meshObj->mesh.vertexNormals.push_back(zVector());
		}

	}

	ZSPACE_INLINE void zFnMesh::computeMeshNormals()
	{
		meshObj->mesh.faceNormals.clear();

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			if (f.isActive())
			{
				// get face vertices and correspondiing positions

				//printf("\n f %i :", i);
				vector<int> fVerts;
				f.getVertices(fVerts);

				zVector fCen; // face center

				vector<zVector> points;
				for (int i = 0; i < fVerts.size(); i++)
				{
					//printf(" %i ", fVerts[i]);
					points.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);

					fCen += meshObj->mesh.vertexPositions[fVerts[i]];
				}

				fCen /= fVerts.size();

				zVector fNorm; // face normal

				if (fVerts.size() != 3)
				{
					for (int j = 0; j < fVerts.size(); j++)
					{
						fNorm += (points[j] - fCen) ^ (points[(j + 1) % fVerts.size()] - fCen);


					}


					//  https://stackoverflow.com/questions/27326636/calculate-normal-vector-of-a-polygon-newells-method
					/*for (int j = 0; j < fVerts.size(); j++) 
					{
						int k = (j + 1) % (fVerts.size());
						fNorm.x += (points[j].y - points[k].y) * (points[j].z + points[k].z);
						fNorm.y += (points[j].z - points[k].z) * (points[j].x + points[k].x);
						fNorm.z += (points[j].x - points[k].x) * (points[j].y + points[k].y);						
					}*/
					


				}
				else
				{
					zVector cross = (points[1] - points[0]) ^ (points[fVerts.size() - 1] - points[0]);
					cross.normalize();

					fNorm = cross;

					//printf("\n working! %i ", i);

				}


				fNorm.normalize();
				meshObj->mesh.faceNormals.push_back(fNorm);
			}
			else meshObj->mesh.faceNormals.push_back(zVector());

		}


		// compute vertex normal
		computeVertexNormalfromFaceNormal();
	}

	ZSPACE_INLINE void zFnMesh::averageVertices(int numSteps)
	{
		for (int k = 0; k < numSteps; k++)
		{
			vector<zVector> tempVertPos;

			for (zItMeshVertex v(*meshObj); !v.end(); v++)
			{
				tempVertPos.push_back(meshObj->mesh.vertexPositions[v.getId()]);

				if (v.isActive())
				{
					if (!v.checkValency(1))
					{
						vector<int> cVerts;

						v.getConnectedVertices(cVerts);

						for (int j = 0; j < cVerts.size(); j++)
						{
							zVector p = meshObj->mesh.vertexPositions[cVerts[j]];
							tempVertPos[v.getId()] += p;
						}

						tempVertPos[v.getId()] /= (cVerts.size() + 1);
					}
				}

			}

			// update position
			for (int i = 0; i < tempVertPos.size(); i++) meshObj->mesh.vertexPositions[i] = tempVertPos[i];
		}

	}

	ZSPACE_INLINE void zFnMesh::garbageCollection(zHEData type)
	{
		removeInactive(type);
	}

	ZSPACE_INLINE void zFnMesh::makeStatic()
	{
		setStaticContainers();
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zFnMesh::setVertexPositions(zPointArray& pos)
	{
		if (pos.size() != meshObj->mesh.vertexPositions.size()) throw std::invalid_argument("size of position contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			meshObj->mesh.vertexPositions[i] = pos[i];
		}
	}

	ZSPACE_INLINE void zFnMesh::setVertexColor(zColor col, bool setFaceColor)
	{

		meshObj->mesh.vertexColors.clear();
		meshObj->mesh.vertexColors.assign(meshObj->mesh.n_v, col);


		if (setFaceColor) computeFaceColorfromVertexColor();
	}

	ZSPACE_INLINE void zFnMesh::setVertexColors(zColorArray& col, bool setFaceColor)
	{
		if (meshObj->mesh.vertexColors.size() != meshObj->mesh.vertices.size())
		{
			meshObj->mesh.vertexColors.clear();
			for (int i = 0; i < meshObj->mesh.vertices.size(); i++) meshObj->mesh.vertexColors.push_back(zColor(1, 0, 0, 1));
		}

		if (col.size() != meshObj->mesh.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

		for (int i = 0; i < meshObj->mesh.vertexColors.size(); i++)
		{
			meshObj->mesh.vertexColors[i] = col[i];
		}

		if (setFaceColor) computeFaceColorfromVertexColor();
	}

	ZSPACE_INLINE void zFnMesh::setFaceColor(zColor col, bool setVertexColor)
	{
		meshObj->mesh.faceColors.clear();
		meshObj->mesh.faceColors.assign(meshObj->mesh.n_f, col);

		if (setVertexColor) computeVertexColorfromFaceColor();
	}

	ZSPACE_INLINE void zFnMesh::setFaceColors(zColorArray& col, bool setVertexColor)
	{
		if (meshObj->mesh.faceColors.size() != meshObj->mesh.faces.size())
		{
			meshObj->mesh.faceColors.clear();
			for (int i = 0; i < meshObj->mesh.faces.size(); i++) meshObj->mesh.faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
		}

		if (col.size() != meshObj->mesh.faceColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

		for (int i = 0; i < meshObj->mesh.faceColors.size(); i++)
		{
			meshObj->mesh.faceColors[i] = col[i];
		}

		if (setVertexColor) computeVertexColorfromFaceColor();
	}

	ZSPACE_INLINE void zFnMesh::setFaceColorOcclusion(zVector &lightVec, bool setVertexColor)
	{
		zVector* norm = getRawFaceNormals();
		zColor* col = getRawFaceColors();


		for (int i = 0; i < numPolygons(); i++)
		{

			double ang = norm[i].angle(lightVec);

			zDomainDouble in(90.0, 180.0);
			zDomainDouble out(0.4, 1.0);

			double val;

			if (ang <= 90) 	val = /*ofMap(ang, 0, 90, 1, 0.4)*/ 0.4;
			else if (ang > 90) 	val = coreUtils.ofMap<double>(ang, in, out);


			col[i] = zColor(val, val, val, 1);

		}

		if (setVertexColor) computeVertexColorfromFaceColor();
	}

	ZSPACE_INLINE void zFnMesh::setFaceNormals(zVector &fNormal)
	{


		meshObj->mesh.faceNormals.clear();
		meshObj->mesh.faceNormals.assign(meshObj->mesh.n_f, fNormal);


		// compute normals per face based on vertex normals and store it in faceNormals
		computeVertexNormalfromFaceNormal();
	}

	ZSPACE_INLINE void zFnMesh::setFaceNormals(zVectorArray &fNormals)
	{

		if (meshObj->mesh.faces.size() != fNormals.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

		meshObj->mesh.faceNormals.clear();

		meshObj->mesh.faceNormals = fNormals;

		// compute normals per face based on vertex normals and store it in faceNormals
		computeVertexNormalfromFaceNormal();
	}

	ZSPACE_INLINE void zFnMesh::setEdgeColor(zColor col, bool setVertexColor)
	{

		meshObj->mesh.edgeColors.clear();
		meshObj->mesh.edgeColors.assign(meshObj->mesh.n_e, col);


		if (setVertexColor) computeVertexColorfromEdgeColor();

	}

	ZSPACE_INLINE void zFnMesh::setEdgeColors(zColorArray& col, bool setVertexColor)
	{
		if (col.size() != meshObj->mesh.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh half edges.");

		for (int i = 0; i < meshObj->mesh.edgeColors.size(); i++)
		{
			meshObj->mesh.edgeColors[i] = col[i];
		}

		if (setVertexColor) computeVertexColorfromEdgeColor();
	}

	ZSPACE_INLINE void zFnMesh::setEdgeWeight(int index, double wt)
	{

		if (meshObj->mesh.edgeWeights.size() != meshObj->mesh.edges.size())
		{
			meshObj->mesh.edgeWeights.clear();
			for (int i = 0; i < meshObj->mesh.edges.size(); i++) meshObj->mesh.edgeWeights.push_back(1);

		}

		meshObj->mesh.edgeWeights[index] = wt;

		int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

		meshObj->mesh.edgeWeights[symEdge] = wt;

	}

	ZSPACE_INLINE void zFnMesh::setEdgeWeights(zDoubleArray& wt)
	{
		if (wt.size() != meshObj->mesh.edgeColors.size()) throw std::invalid_argument("size of wt contatiner is not equal to number of mesh half edges.");

		for (int i = 0; i < meshObj->mesh.edgeWeights.size(); i++)
		{
			meshObj->mesh.edgeWeights[i] = wt[i];
		}
	}

	//--- GET METHODS 

	ZSPACE_INLINE void zFnMesh::getVertexPositions(zPointArray& pos)
	{
		pos = meshObj->mesh.vertexPositions;
	}

	ZSPACE_INLINE zPoint* zFnMesh::getRawVertexPositions()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.vertexPositions[0];
	}

	ZSPACE_INLINE void zFnMesh::getVertexNormals(zVectorArray& norm)
	{
		norm = meshObj->mesh.vertexNormals;
	}

	ZSPACE_INLINE zVector* zFnMesh::getRawVertexNormals()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.vertexNormals[0];
	}

	ZSPACE_INLINE void zFnMesh::getVertexColors(zColorArray& col)
	{
		col = meshObj->mesh.vertexColors;
	}

	ZSPACE_INLINE zColor* zFnMesh::getRawVertexColors()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.vertexColors[0];
	}

	ZSPACE_INLINE void zFnMesh::getEdgeColors(zColorArray& col)
	{
		col = meshObj->mesh.edgeColors;
	}

	ZSPACE_INLINE zColor* zFnMesh::getRawEdgeColors()
	{
		if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.edgeColors[0];
	}

	ZSPACE_INLINE void zFnMesh::getFaceNormals(zVectorArray& norm)
	{
		norm = meshObj->mesh.faceNormals;
	}

	ZSPACE_INLINE zVector* zFnMesh::getRawFaceNormals()
	{
		if (numPolygons() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.faceNormals[0];
	}

	ZSPACE_INLINE void zFnMesh::getFaceColors(zColorArray& col)
	{
		col = meshObj->mesh.faceColors;
	}

	ZSPACE_INLINE zColor* zFnMesh::getRawFaceColors()
	{
		if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

		return &meshObj->mesh.faceColors[0];
	}

	ZSPACE_INLINE zPoint zFnMesh::getCenter()
	{
		zPoint out;

		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			out += meshObj->mesh.vertexPositions[i];
		}

		out /= meshObj->mesh.vertexPositions.size();

		return out;

	}

	ZSPACE_INLINE void zFnMesh::getCenters(zHEData type, zPointArray &centers)
	{
		// Mesh Edge 
		if (type == zHalfEdgeData)
		{

			centers.clear();

			for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
			{
				if (he.isActive())
				{
					centers.push_back(he.getCenter());
				}
				else
				{
					centers.push_back(zVector());

				}
			}

		}
		else if (type == zEdgeData)
		{

			centers.clear();

			for (zItMeshEdge e(*meshObj); !e.end(); e++)
			{
				if (e.isActive())
				{
					centers.push_back(e.getCenter());
				}
				else
				{
					centers.push_back(zVector());

				}
			}

		}

		// Mesh Face 
		else if (type == zFaceData)
		{
			centers.clear();

			for (zItMeshFace f(*meshObj); !f.end(); f++)
			{
				if (f.isActive())
				{
					centers.push_back(f.getCenter());
				}
				else
				{
					centers.push_back(zVector());

				}
			}
		}
		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	ZSPACE_INLINE void zFnMesh::getDualMesh(zObjMesh &dualMeshObj, zIntArray &inEdge_dualEdge, zIntArray &dualEdge_inEdge, bool excludeBoundary, bool keepExistingBoundary, bool rotate90)
	{
		vector<zVector> positions;
		vector<int> polyConnects;
		vector<int> polyCounts;

		vector<zVector> fCenters;
		getCenters(zFaceData, fCenters);

		vector<zVector> eCenters;
		getCenters(zHalfEdgeData, eCenters);

		positions = fCenters;

		// store map for input mesh edge to new vertex Id
		vector<int> inEdge_dualVertex;

		//for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++)
		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{
			inEdge_dualVertex.push_back(-1);


			if (!he.isActive()) continue;

			if (he.onBoundary())
			{
				if (!excludeBoundary)
				{
					inEdge_dualVertex[he.getId()] = positions.size();
					positions.push_back(eCenters[he.getId()]);
				}
			}
			else
			{
				inEdge_dualVertex[he.getId()] = he.getFace().getId(); ;
			}
		}


		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (!v.isActive()) continue;

			if (v.onBoundary())
			{
				if (excludeBoundary) continue;

				zItMeshHalfEdge e = v.getHalfEdge(); ;

				zItMeshHalfEdge start = v.getHalfEdge();

				vector<int> tempPolyConnects;
				do
				{
					int eId = e.getId();
					int index = -1;
					bool checkRepeat = meshObj->mesh.coreUtils.checkRepeatElement(inEdge_dualVertex[eId], tempPolyConnects, index);
					if (!checkRepeat) tempPolyConnects.push_back(inEdge_dualVertex[eId]);

					if (keepExistingBoundary)
					{
						int vId = e.getVertex().getId();
						if (e.onBoundary() && vId == v.getId())
						{
							//printf("\n working!");
							//tempPolyConnects.push_back(vId);
						}
					}


					e = e.getPrev();
					eId = e.getId();

					if (keepExistingBoundary)
					{
						int vId = e.getVertex().getId();
						if (e.onBoundary() && vId == v.getId())
						{
							//printf("\n working2!");
							//tempPolyConnects.push_back(positions.size());
							//positions.push_back(meshObj->mesh.vertexPositions[vId]);

						}
					}

					index = -1;
					checkRepeat = meshObj->mesh.coreUtils.checkRepeatElement(inEdge_dualVertex[eId], tempPolyConnects, index);
					if (!checkRepeat) tempPolyConnects.push_back(inEdge_dualVertex[eId]);



					e = e.getSym();

				} while (e != start);


				for (int j = 0; j < tempPolyConnects.size(); j++)
				{
					polyConnects.push_back(tempPolyConnects[j]);
				}

				polyCounts.push_back(tempPolyConnects.size());
			}

			else
			{
				vector<int> cEdges;
				v.getConnectedHalfEdges(cEdges);

				for (int j = 0; j < cEdges.size(); j++)
				{
					polyConnects.push_back(inEdge_dualVertex[cEdges[j]]);
				}

				polyCounts.push_back(cEdges.size());
			}


		}


		dualMeshObj.mesh.create(positions, polyCounts, polyConnects);

		// rotate by 90

		if (rotate90)
		{
			zVector meshNorm = meshObj->mesh.vertexNormals[0];
			meshNorm.normalize();

			// bounding box
			zVector minBB, maxBB;
			meshObj->mesh.coreUtils.getBounds(dualMeshObj.mesh.vertexPositions, minBB, maxBB);

			zVector cen = (maxBB + minBB) * 0.5;

			for (int i = 0; i < dualMeshObj.mesh.vertexPositions.size(); i++)
			{
				dualMeshObj.mesh.vertexPositions[i] -= cen;
				dualMeshObj.mesh.vertexPositions[i] = dualMeshObj.mesh.vertexPositions[i].rotateAboutAxis(meshNorm, -90);
			}


		}

		// compute dualEdge_inEdge
		dualEdge_inEdge.clear();

		for (auto &he : dualMeshObj.mesh.halfEdges)
		{
			dualEdge_inEdge.push_back(-1);
		}

		// compute inEdge to dualEdge	
		inEdge_dualEdge.clear();

		for (int i = 0; i < numHalfEdges(); i++)
		{
			int v1 = inEdge_dualVertex[i];
			int v2 = (i % 2 == 0) ? inEdge_dualVertex[i + 1] : inEdge_dualVertex[i - 1];

			int eId;
			bool chk = dualMeshObj.mesh.halfEdgeExists(v1, v2, eId);
			if (chk) inEdge_dualEdge.push_back(eId);
			else inEdge_dualEdge.push_back(-1);

			if (inEdge_dualEdge[i] != -1)
			{
				dualEdge_inEdge[inEdge_dualEdge[i]] = i;
			}
		}



	}

	ZSPACE_INLINE void zFnMesh::getDualGraph(zObjGraph &dualGraphObj, zIntArray &inEdge_dualEdge, zIntArray &dualEdge_inEdge, bool excludeBoundary , bool PlanarMesh , bool rotate90)
	{
		vector<zVector> positions;
		vector<int> edgeConnects;

		vector<zVector> fCenters;
		getCenters(zFaceData, fCenters);

		vector<zVector> eCenters;
		getCenters(zHalfEdgeData, eCenters);

		positions = fCenters;

		// store map for input mesh edge to new vertex Id
		vector<int> inEdge_dualVertex;

		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{
			inEdge_dualVertex.push_back(-1);

			int i = he.getId();

			if (!he.isActive()) continue;

			if (he.onBoundary())
			{
				if (!excludeBoundary)
				{
					inEdge_dualVertex[i] = positions.size();
					positions.push_back(eCenters[i]);
				}
			}
			else
			{
				inEdge_dualVertex[i] = he.getFace().getId();
			}
		}

		for (int i = 0; i < meshObj->mesh.halfEdges.size(); i += 2)
		{
			int v_0 = inEdge_dualVertex[i];
			int v_1 = inEdge_dualVertex[i + 1];

			if (v_0 != -1 && v_1 != -1)
			{
				edgeConnects.push_back(v_0);
				edgeConnects.push_back(v_1);
			}

		}

		if (PlanarMesh)
		{
			zVector graphNorm = meshObj->mesh.vertexNormals[0];
			graphNorm.normalize();

			zVector x(1, 0, 0);
			zVector sortRef = graphNorm ^ x;

			dualGraphObj.graph.create(positions, edgeConnects, graphNorm, sortRef);
		}

		else dualGraphObj.graph.create(positions, edgeConnects);

		// rotate by 90

		if (rotate90 && PlanarMesh)
		{
			zVector graphNorm = meshObj->mesh.vertexNormals[0];
			graphNorm.normalize();

			// bounding box
			zVector minBB, maxBB;
			meshObj->mesh.coreUtils.getBounds(dualGraphObj.graph.vertexPositions, minBB, maxBB);

			zVector cen = (maxBB + minBB) * 0.5;

			for (int i = 0; i < dualGraphObj.graph.vertexPositions.size(); i++)
			{
				dualGraphObj.graph.vertexPositions[i] -= cen;
				dualGraphObj.graph.vertexPositions[i] = dualGraphObj.graph.vertexPositions[i].rotateAboutAxis(graphNorm, -90);
			}


		}

		// compute dualEdge_inEdge
		dualEdge_inEdge.clear();
		for (int i = 0; i < dualGraphObj.graph.halfEdges.size(); i++)
		{
			dualEdge_inEdge.push_back(-1);
		}

		// compute inEdge to dualEdge	
		inEdge_dualEdge.clear();

		for (int i = 0; i < numHalfEdges(); i++)
		{
			int v1 = inEdge_dualVertex[i];
			int v2 = (i % 2 == 0) ? inEdge_dualVertex[i + 1] : inEdge_dualVertex[i - 1];

			int eId;
			bool chk = dualGraphObj.graph.halfEdgeExists(v1, v2, eId);

			if (chk) inEdge_dualEdge.push_back(eId);
			else inEdge_dualEdge.push_back(-1);

			if (inEdge_dualEdge[i] != -1)
			{
				dualEdge_inEdge[inEdge_dualEdge[i]] = i;
			}
		}


	}

	ZSPACE_INLINE void zFnMesh::getRainflowGraph(zObjGraph &rainflowGraphObj, bool excludeBoundary)
	{
		vector<zVector> positions;
		vector<int> edgeConnects;

		zVector* pos = getRawVertexPositions();

		unordered_map <string, int> positionVertex;

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (excludeBoundary && v.onBoundary()) continue;

			vector<zItMeshFace> cFaces;
			v.getConnectedFaces(cFaces);


			vector<int> positionIndicies;
			for (auto &f : cFaces)
			{
				vector<int> fVerts;
				f.getVertices(fVerts);

				for (int k = 0; k < fVerts.size(); k++) positionIndicies.push_back(fVerts[k]);
			}


			// get lowest positions

			zVector lowPosition = pos[v.getId()];

			for (int j = 0; j < positionIndicies.size(); j++)
			{
				if (pos[positionIndicies[j]].z < lowPosition.z)
				{
					lowPosition = pos[positionIndicies[j]];

				}
			}

			vector<int> lowId;
			if (lowPosition.z != pos[v.getId()].z)
			{
				for (int j = 0; j < positionIndicies.size(); j++)
				{
					if (pos[positionIndicies[j]].z == lowPosition.z)
					{
						lowId.push_back(positionIndicies[j]);

					}
				}
			}


			if (lowId.size() > 0)
			{

				for (int j = 0; j < lowId.size(); j++)
				{
					zVector pos1 = pos[v.getId()];
					int v1;
					bool check1 = coreUtils.vertexExists(positionVertex, pos1, 3, v1);
					if (!check1)
					{
						v1 = positions.size();
						positions.push_back(pos1);
						coreUtils.addToPositionMap(positionVertex, pos1, v1, 3);
					}


					zVector pos2 = pos[lowId[j]];
					int v2;
					bool check2 = coreUtils.vertexExists(positionVertex, pos2, 3, v2);
					if (!check2)
					{
						v2 = positions.size();
						positions.push_back(pos2);
						coreUtils.addToPositionMap(positionVertex, pos2, v2, 3);
					}


					edgeConnects.push_back(v1);
					edgeConnects.push_back(v2);
				}


			}

		}

		rainflowGraphObj.graph.create(positions, edgeConnects);
	}

	ZSPACE_INLINE void zFnMesh::getMeshTriangles(vector<zIntArray> &faceTris)
	{
		if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();

		faceTris.clear();

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> Tri_connects;
			int i = f.getId();

			if (f.isActive())
			{

				vector<int> fVerts;
				f.getVertices(fVerts);

				// compute polygon Triangles


				int n_Tris = 0;
				if (fVerts.size() > 0) f.getTriangles(n_Tris, Tri_connects);
				else Tri_connects = fVerts;
			}


			faceTris.push_back(Tri_connects);
		}

	}

	ZSPACE_INLINE double zFnMesh::getMeshVolume()
	{
		double out = 0;

		vector<vector<int>> faceTris;
		getMeshTriangles(faceTris);

		for (int i = 0; i < faceTris.size(); i++)
		{
			for (int j = 0; j < faceTris[i].size(); j += 3)
			{
				double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(meshObj->mesh.vertexPositions[faceTris[i][j + 0]], meshObj->mesh.vertexPositions[faceTris[i][j + 1]], meshObj->mesh.vertexPositions[faceTris[i][j + 2]]);

				out += vol;
			}


		}

		return out;
	}

	ZSPACE_INLINE void zFnMesh::getMeshFaceVolumes(vector<zIntArray> &faceTris, zPointArray &fCenters, zDoubleArray &faceVolumes, bool absoluteVolumes)
	{
		if (faceTris.size() == 0) getMeshTriangles(faceTris);
		if (fCenters.size() == 0 || fCenters.size() != numPolygons()) getCenters(zFaceData, fCenters);

		faceVolumes.clear();

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			int i = f.getId();
			double vol = f.getVolume(faceTris[i], fCenters[i], absoluteVolumes);

			faceVolumes.push_back(vol);
		}
	}

	ZSPACE_INLINE void zFnMesh::getPrincipalCurvatures(zCurvatureArray &vertexCurvatures)
	{
		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			int j = v.getId();

			if (v.isActive())
			{
				vertexCurvatures.push_back(v.getPrincipalCurvature());
			}

			else
			{
				zCurvature curv;

				curv.k1 = -1;
				curv.k2 = -1;

				vertexCurvatures.push_back(curv);
			}
		}
	}

	ZSPACE_INLINE void zFnMesh::getGaussianCurvature(zDoubleArray &vertexCurvatures)
	{
		vector<zCurvature> pCurvature;

		getPrincipalCurvatures(pCurvature);


		vertexCurvatures.clear();
		vertexCurvatures.assign(numVertices(), -1);
		for (int i = 0; i < numVertices(); i++)
		{
			vertexCurvatures[i] = pCurvature[i].k1 * pCurvature[i].k2;
		}

	}

	ZSPACE_INLINE void zFnMesh::getEdgeDihedralAngles(zDoubleArray &dihedralAngles)
	{
		vector<double> out;

		if (meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();

		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			out.push_back(e.getDihedralAngle());
		}

		dihedralAngles = out;
	}

	ZSPACE_INLINE double zFnMesh::getHalfEdgeLengths(zDoubleArray &halfEdgeLengths)
	{
		double total = 0.0;


		halfEdgeLengths.clear();

		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			if (e.isActive())
			{
				double e_len = e.getLength();

				halfEdgeLengths.push_back(e_len);
				halfEdgeLengths.push_back(e_len);

				total += e_len;
			}
			else
			{
				halfEdgeLengths.push_back(0);
				halfEdgeLengths.push_back(0);
			}
		}

		return total;
	}

	ZSPACE_INLINE double zFnMesh::getEdgeLengths(zDoubleArray &edgeLengths)
	{
		double total = 0.0;


		edgeLengths.clear();

		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			if (e.isActive())
			{
				double e_len = e.getLength();
				edgeLengths.push_back(e_len);
				total += e_len;
			}
			else
			{
				edgeLengths.push_back(0);
			}
		}

		return total;
	}

	ZSPACE_INLINE double zFnMesh::getVertexAreas(zPointArray &faceCenters, zPointArray &edgeCenters, zDoubleArray &vertexAreas)
	{
		vector<double> out;

		double totalArea = 0;

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			double vArea = 0;

			int i = v.getId();

			if (v.isActive())
			{
				vector<zItMeshHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				for (int j = 0; j < cEdges.size(); j++)
				{


					zItMeshHalfEdge cE = cEdges[j];
					zItMeshHalfEdge nE = cEdges[(j + 1) % cEdges.size()];

					if (cE.onBoundary() || nE.getSym().onBoundary()) continue;

					if (cE.getFace().getId() != nE.getSym().getFace().getId()) continue;

					zVector vPos = meshObj->mesh.vertexPositions[i];
					zVector fCen = faceCenters[cE.getFace().getId()];
					zVector currentEdge_cen = edgeCenters[cE.getId()];
					zVector nextEdge_cen = edgeCenters[nE.getId()];

					double Area1 = meshObj->mesh.coreUtils.getTriangleArea(vPos, currentEdge_cen, fCen);
					vArea += (Area1);

					double Area2 = meshObj->mesh.coreUtils.getTriangleArea(vPos, nextEdge_cen, fCen);
					vArea += (Area2);

				}

			}


			out.push_back(vArea);

			totalArea += vArea;

		}

		//printf("\n totalArea : %1.4f ",  totalArea);

		vertexAreas = out;

		return totalArea;
	}

	ZSPACE_INLINE double zFnMesh::getPlanarFaceAreas(zDoubleArray &faceAreas)
	{

		if (meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();

		faceAreas.clear();

		double totalArea = 0;

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			double fArea = f.getPlanarFaceArea();
			faceAreas.push_back(fArea);

			totalArea += fArea;
		}


		return totalArea;
	}

	ZSPACE_INLINE void zFnMesh::getPolygonData(zIntArray(&polyConnects), zIntArray(&polyCounts))
	{
		polyConnects.clear();
		polyCounts.clear();

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			if (!f.isActive()) continue;


			vector<int> facevertices;
			f.getVertices(facevertices);

			polyCounts.push_back(facevertices.size());


			for (int j = 0; j < facevertices.size(); j++)
			{
				polyConnects.push_back(facevertices[j]);

			}
		}
	}

	ZSPACE_INLINE void zFnMesh::getEdgeData(zIntArray &edgeConnects)
	{
		edgeConnects.clear();

		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			edgeConnects.push_back(e.getHalfEdge(0).getVertex().getId());
			edgeConnects.push_back(e.getHalfEdge(1).getVertex().getId());
		}
	}

	ZSPACE_INLINE void zFnMesh::getDuplicate(zObjMesh &out)
	{

		vector<zVector> positions;
		vector<int> polyConnects;
		vector<int> polyCounts;

		positions = meshObj->mesh.vertexPositions;
		getPolygonData(polyConnects, polyCounts);

		out.mesh.create(positions, polyCounts, polyConnects);


		zFnMesh tempFn(out);

		tempFn.computeMeshNormals();
		tempFn.setVertexColors(meshObj->mesh.vertexColors, false);
		tempFn.setEdgeColors(meshObj->mesh.edgeColors, false);
		tempFn.setFaceColors(meshObj->mesh.faceColors, false);
	}

	ZSPACE_INLINE int zFnMesh::getVBOVertexIndex()
	{
		return meshObj->mesh.VBO_VertexId;
	}

	ZSPACE_INLINE int zFnMesh::getVBOEdgeIndex()
	{
		return meshObj->mesh.VBO_EdgeId;
	}

	ZSPACE_INLINE int zFnMesh::getVBOFaceIndex()
	{
		return meshObj->mesh.VBO_FaceId;
	}

	ZSPACE_INLINE int zFnMesh::getVBOVertexColorIndex()
	{
		return meshObj->mesh.VBO_VertexColorId;
	}

	//---- TRI-MESH MODIFIER METHODS

	ZSPACE_INLINE void zFnMesh::faceTriangulate(zItMeshFace &face)
	{

		if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();

		vector<int> fVerts;
		face.getVertices(fVerts);

		int numfaces_original = meshObj->mesh.faces.size();
		int numHalfEdges_original = meshObj->mesh.halfEdges.size();

		if (fVerts.size() != 3)
		{
			// compute polygon Triangles
			int n_Tris = 0;
			vector<int> Tri_connects;
			face.getTriangles(n_Tris, Tri_connects);

			//printf("\n %i numtris: %i %i ", faceIndex, n_Tris, Tri_connects.size());

			for (int j = 0; j < n_Tris; j++)
			{
				vector<int> triVerts;
				triVerts.push_back(Tri_connects[j * 3]);
				triVerts.push_back(Tri_connects[j * 3 + 1]);
				triVerts.push_back(Tri_connects[j * 3 + 2]);

				//printf("\n %i %i %i ", Tri_connects[j * 3], Tri_connects[j * 3 + 1], Tri_connects[j * 3 + 2]);

				// check if edges e01, e12 or e20					
				zItMeshHalfEdge e01, e12, e20;

				bool e01_Boundary = false;
				bool e12_Boundary = false;
				bool e20_Boundary = false;

				for (int k = 0; k < triVerts.size(); k++)
				{

					if (k == 0)
					{
						addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()], true, e01);

						if (e01.getId() < numHalfEdges_original)
						{
							if (e01.onBoundary())  e01_Boundary = true;
						}
					}

					if (k == 1)
					{
						addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()], true, e12);

						if (e12.getId() < numHalfEdges_original)
						{
							if (e12.onBoundary())  e12_Boundary = true;
						}
					}

					if (k == 2)
					{
						addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()], true, e20);

						if (e20.getId() < numHalfEdges_original)
						{
							if (e20.onBoundary())  e20_Boundary = true;
						}
					}

				}

				//printf("\n %i %i %i ", e01.getId(), e12.getId(), e20.getId());

				if (j > 0)
				{
					zItMeshFace newFace;
					bool check = addPolygon(newFace);

					newFace.setHalfEdge(e01);

					if (!e01_Boundary) e01.setFace(newFace);
					if (!e12_Boundary) e12.setFace(newFace);
					if (!e20_Boundary) e20.setFace(newFace);
				}
				else
				{

					if (!e01_Boundary) face.setHalfEdge(e01);
					else if (!e12_Boundary) face.setHalfEdge(e12);
					else if (!e20_Boundary) face.setHalfEdge(e20);


					if (!e01_Boundary) e01.setFace(face);
					if (!e12_Boundary) e12.setFace(face);
					if (!e20_Boundary) e20.setFace(face);
				}

				// update edge pointers
				e01.setNext(e12);
				e12.setPrev(e01);

				e01.setPrev(e20);
				e20.setNext(e01);

				e12.setNext(e20);
				e20.setPrev(e12);

			}
		}

	}

	ZSPACE_INLINE void zFnMesh::triangulate()
	{

		if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();



		// iterate through faces and triangulate faces with more than 3 vetices
		int numfaces_original = meshObj->mesh.faces.size();

		for (int i = 0; i < numfaces_original; i++)
		{

			zItMeshFace f(*meshObj, i);
			if (!f.isActive()) continue;

			faceTriangulate(f);

		}

		computeMeshNormals();

	}

	//---- DELETE MODIFIER METHODS

	ZSPACE_INLINE void zFnMesh::deleteVertex(int index, bool removeInactiveElems)
	{
		//if (index >= meshObj->mesh.vertices.size()) throw std::invalid_argument(" error: index out of bounds.");
		//if (!meshObj->mesh.indexToVertex[index]->isActive()) throw std::invalid_argument(" error: index out of bounds.");

		//// check if boundary vertex
		//bool boundaryVertex = (onBoundary(index, zVertexData));

		//// get connected faces
		//vector<int> cFaces;
		//getConnectedFaces(index, zVertexData, cFaces);

		//// get connected edges
		//vector<int> cEdges;
		//getConnectedEdges(index, zVertexData, cEdges);



		//// get vertices in cyclical orders without the  the vertex to be removed. remove duplicates if any
		//vector<int> outerVertices;

		//vector<int> deactivateVertices;
		//vector<int> deactivateEdges;

		//deactivateVertices.push_back(index);

		//// add to deactivate edges connected edges
		//for (int i = 0; i < cEdges.size(); i++) deactivateEdges.push_back(cEdges[i]);

		//// add to deactivate vertices with valence 2 and  connected edges of that vertex
		//for (int i = 0; i < cEdges.size(); i++)
		//{
		//	int v0 = meshObj->mesh.edges[cEdges[i]].getVertex()->getVertexId();

		//	if (!onBoundary(v0, zVertexData) && checkVertexValency(v0, 2))
		//	{
		//		deactivateVertices.push_back(v0);
		//		deactivateEdges.push_back(meshObj->mesh.edges[cEdges[i]].getNext()->getEdgeId());
		//	}
		//}

		//// compute new face vertices
		//for (int i = 0; i < cEdges.size(); i++)
		//{
		//	if (!meshObj->mesh.edges[cEdges[i]].getFace()) continue;

		//	zEdge *curEdge = &meshObj->mesh.edges[cEdges[i]];
		//	int v0 = curEdge->getVertex()->getVertexId();

		//	do
		//	{
		//		bool vertExists = false;

		//		for (int k = 0; k < outerVertices.size(); k++)
		//		{
		//			if (v0 == outerVertices[k])
		//			{
		//				vertExists = true;
		//				break;
		//			}
		//		}

		//		if (!vertExists)
		//		{
		//			for (int k = 0; k < deactivateVertices.size(); k++)
		//			{
		//				if (v0 == deactivateVertices[k])
		//				{
		//					vertExists = true;
		//					break;
		//				}
		//			}
		//		}

		//		if (!vertExists) outerVertices.push_back(v0);



		//		curEdge = curEdge->getNext();
		//		v0 = curEdge->getVertex()->getVertexId();


		//	} while (v0 != index);

		//}


		//// deactivate connected edges 
		//for (int i = 0; i < deactivateEdges.size(); i++)
		//{
		//	if (meshObj->mesh.edgeActive[deactivateEdges[i]])deactivateElement(deactivateEdges[i], zEdgeData);
		//}

		//// disable connected faces
		//for (int i = 0; i < cFaces.size(); i++)
		//{
		//	if (meshObj->mesh.faceActive[cFaces[i]]) deactivateElement(cFaces[i], zFaceData);
		//}

		//// deactivate vertex
		//for (int i = 0; i < deactivateVertices.size(); i++)
		//{
		//	if (meshObj->mesh.vertexActive[deactivateVertices[i]]) deactivateElement(deactivateVertices[i], zVertexData);
		//}



		//// add new face if outerVertices has more than 2 vertices

		//if (outerVertices.size() > 2)
		//{
		//	meshObj->mesh.addPolygon(outerVertices);

		//	if (boundaryVertex)  meshObj->mesh.update_BoundaryEdgePointers();
		//}

		//computeMeshNormals();

		//if (removeInactiveElems)
		//{
		//	removeInactiveElements(zVertexData);
		//	removeInactiveElements(zEdgeData);
		//	removeInactiveElements(zFaceData);
		//}
	}

	ZSPACE_INLINE void zFnMesh::deleteFace(int index, bool removeInactiveElems)
	{
		//if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		//if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		//// check if there is only 1 polygon. If true, cant perform collapse.
		//if (numPolygons() == 1)
		//{
		//	printf("\n Can't delete on single face mesh.");
		//	return;
		//}

		//// get faces vertices
		//vector<int> fVerts;
		//getVertices(index, zFaceData, fVerts);



		//// get face edges.
		//vector<int> fEdges;
		//getEdges(index, zFaceData, fEdges);

		//// connected edge for each face vertex
		//vector<int> fVertsValence;
		//for (int i = 0; i < fVerts.size(); i++)
		//{

		//	vector<int> cEdges;
		//	getConnectedEdges(fVerts[i], zVertexData, cEdges);
		//	fVertsValence.push_back(cEdges.size());

		//	// update vertex edge pointer if ther are pointing to face edges , as they will be disabled.

		//	for (int j = 0; j < cEdges.size(); j++)
		//	{
		//		bool chk = false;

		//		for (int k = 0; k < fEdges.size(); k++)
		//		{
		//			int sEdge = meshObj->mesh.edges[fEdges[k]].getSym()->getEdgeId();

		//			if (cEdges[j] == fEdges[k] || cEdges[j] == sEdge)
		//			{
		//				chk = true;
		//				break;
		//			}
		//		}

		//		if (!chk)
		//		{
		//			meshObj->mesh.vertices[fVerts[i]].setEdge(&meshObj->mesh.edges[cEdges[j]]);
		//			break;
		//		}
		//	}

		//}

		//// make face edges as  boundary edges, and disable them if both half edges have null face pointers.
		//for (int i = 0; i < fEdges.size(); i++)
		//{
		//	meshObj->mesh.edges[fEdges[i]].setFace(nullptr);

		//	int symEdge = meshObj->mesh.edges[fEdges[i]].getSym()->getEdgeId();

		//	if (onBoundary(fEdges[i], zEdgeData) && onBoundary(symEdge, zEdgeData))
		//	{
		//		deactivateElement(fEdges[i], zEdgeData);
		//	}
		//}

		//// get face vertices and deactivate them if all connected half edges are in active.
		//for (int i = 0; i < fVerts.size(); i++)
		//{
		//	bool removeVertex = true;
		//	if (fVertsValence[i] > 2) removeVertex = false;


		//	if (removeVertex)
		//	{
		//		deactivateElement(fVerts[i], zVertexData);
		//	}

		//}


		//// deactivate face
		//deactivateElement(index, zFaceData);


		//if (removeInactiveElems)
		//{
		//	removeInactiveElements(zVertexData);
		//	removeInactiveElements(zEdgeData);
		//	removeInactiveElements(zFaceData);
		//}

	}

	ZSPACE_INLINE void zFnMesh::deleteEdge(int index, bool removeInactiveElements) {}

	//---- TOPOLOGY MODIFIER METHODS

	ZSPACE_INLINE void zFnMesh::collapseEdge(int index, double edgeFactor, bool removeInactiveElems )
	{
		//if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		//if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		//int nFVerts = (onBoundary(index, zEdgeData)) ? 0 : getNumPolygonVertices( meshObj->mesh.edges[index].getFace()->getFaceId());

		//int sEdge = meshObj->mesh.edges[index].getSym()->getEdgeId();
		//int nFVerts_Sym = (onBoundary(sEdge, zEdgeData)) ? 0 : getNumPolygonVertices( meshObj->mesh.edges[sEdge].getFace()->getFaceId());

		//// check if there is only 1 polygon and its a triangle. If true, cant perform collapse.
		//if (numPolygons() == 1)
		//{
		//	if (nFVerts == 3 || nFVerts_Sym == 3)
		//	{
		//		printf("\n Can't perform collapse on single trianglular face.");
		//		return;
		//	}

		//}

		//// get edge faces
		//vector<int> eFaces;
		//getFaces(index, zEdgeData, eFaces);

		//if (numPolygons() == eFaces.size())
		//{
		//	if (nFVerts == nFVerts_Sym && nFVerts_Sym == 3)
		//	{
		//		printf("\n Can't perform collapse on common edge of 2 triangular face mesh.");
		//		return;
		//	}

		//}

		//int v1 = meshObj->mesh.edges[index].getVertex()->getVertexId();
		//int v2 = meshObj->mesh.edges[sEdge].getVertex()->getVertexId();

		//int vertexRemoveID = v1;
		//int vertexRetainID = v2;

		//if (getVertexValence(v1) > getVertexValence(v2))
		//{
		//	vertexRemoveID = v2;
		//	vertexRetainID = v1;

		//	edgeFactor = 1 - edgeFactor;

		//}

		//// set new position of retained vertex
		//zVector e = meshObj->mesh.vertexPositions[vertexRemoveID] - meshObj->mesh.vertexPositions[vertexRetainID];
		//double eLength = e.length();
		//e.normalize();

		//meshObj->mesh.vertexPositions[vertexRetainID] = meshObj->mesh.vertexPositions[vertexRetainID] + e * (edgeFactor * eLength);


		//// get connected edges of vertexRemoveID
		//vector<int> cEdges;
		//getConnectedEdges(vertexRemoveID, zVertexData, cEdges);


		//// get connected edges

		//int vNext = meshObj->mesh.edges[index].getNext()->getVertex()->getVertexId();
		//vector<int> cEdgesVNext;
		//getConnectedEdges(vNext, zVertexData, cEdgesVNext);

		//int vPrev = meshObj->mesh.edges[index].getPrev()->getVertex()->getVertexId();
		//vector<int> cEdgesVPrev;
		//getConnectedEdges(vPrev, zVertexData, cEdgesVPrev);

		//int vNext_sEdge = meshObj->mesh.edges[sEdge].getNext()->getVertex()->getVertexId();
		//vector<int> cEdgesVNext_sEdge;
		//getConnectedEdges(vNext_sEdge, zVertexData, cEdgesVNext_sEdge);

		//int vPrev_sEdge = meshObj->mesh.edges[sEdge].getPrev()->getVertex()->getVertexId();
		//vector<int> cEdgesVPrev_sEdge;
		//getConnectedEdges(vPrev_sEdge, zVertexData, cEdgesVPrev_sEdge);

		//// current edge 
		//if (nFVerts == 3)
		//{

		//	// update pointers
		//	meshObj->mesh.edges[index].getNext()->setNext(meshObj->mesh.edges[index].getPrev()->getSym()->getNext());
		//	meshObj->mesh.edges[index].getNext()->setPrev(meshObj->mesh.edges[index].getPrev()->getSym()->getPrev());

		//	meshObj->mesh.edges[index].getPrev()->setPrev(nullptr);
		//	meshObj->mesh.edges[index].getPrev()->setNext(nullptr);

		//	meshObj->mesh.edges[index].getPrev()->getSym()->setPrev(nullptr);
		//	meshObj->mesh.edges[index].getPrev()->getSym()->setNext(nullptr);

		//	meshObj->mesh.edges[index].getNext()->setFace(meshObj->mesh.edges[index].getPrev()->getSym()->getFace());

		//	if (meshObj->mesh.edges[index].getPrev()->getSym()->getFace())
		//	{
		//		meshObj->mesh.edges[index].getPrev()->getSym()->getFace()->setEdge(meshObj->mesh.edges[index].getNext());
		//		meshObj->mesh.edges[index].getPrev()->getSym()->setFace(nullptr);
		//	}

		//	// update vertex edge pointer if pointing to prev edge

		//	if (meshObj->mesh.vertices[vNext].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getPrev()->getEdgeId())
		//	{
		//		for (int i = 0; i < cEdgesVNext.size(); i++)
		//		{
		//			if (cEdgesVNext[i] != meshObj->mesh.edges[index].getPrev()->getEdgeId())
		//			{
		//				meshObj->mesh.vertices[vNext].setEdge(&meshObj->mesh.edges[cEdgesVNext[i]]);
		//			}
		//		}
		//	}

		//	// update vertex edge pointer if pointing to prev edge

		//	if (meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getPrev()->getSym()->getEdgeId() || meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == index)
		//	{
		//		for (int i = 0; i < cEdgesVPrev.size(); i++)
		//		{
		//			if (cEdgesVPrev[i] != meshObj->mesh.edges[index].getPrev()->getSym()->getEdgeId() && cEdgesVPrev[i] != index)
		//			{
		//				meshObj->mesh.vertices[vPrev].setEdge(&meshObj->mesh.edges[cEdgesVPrev[i]]);
		//			}
		//		}
		//	}

		//	// decativate prev edge
		//	deactivateElement(meshObj->mesh.edges[index].getPrev()->getEdgeId(), zEdgeData);

		//	// decativate next and sym pointer of the next edge are same, deactivate edge
		//	if (meshObj->mesh.edges[index].getNext()->getNext() == meshObj->mesh.edges[index].getNext()->getSym())
		//	{

		//		deactivateElement(meshObj->mesh.edges[index].getNext()->getEdgeId(), zEdgeData);
		//		deactivateElement(vNext, zVertexData);
		//	}

		//	// decativate prev and sym pointer of the next edge are same, deactivate edge
		//	else if (meshObj->mesh.edges[index].getNext()->getPrev() == meshObj->mesh.edges[index].getNext()->getSym())
		//	{
		//		deactivateElement(meshObj->mesh.edges[index].getNext()->getVertex()->getVertexId(), zVertexData);
		//		deactivateElement(vNext, zVertexData);
		//	}

		//	// deactivate face pointed by collapse edge
		//	deactivateElement(meshObj->mesh.edges[index].getFace()->getFaceId(), zFaceData);

		//	meshObj->mesh.edges[index].setFace(nullptr);

		//	meshObj->mesh.edges[index].setNext(nullptr);
		//	meshObj->mesh.edges[index].setPrev(nullptr);

		//}
		//else
		//{
		//	// update vertex edge pointer if pointing to current edge
		//	if (meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getEdgeId())
		//	{
		//		meshObj->mesh.vertices[vPrev].setEdge(meshObj->mesh.edges[index].getPrev()->getSym());
		//	}

		//	// update pointers
		//	meshObj->mesh.edges[index].getNext()->setPrev(meshObj->mesh.edges[index].getPrev());

		//	meshObj->mesh.edges[index].setNext(nullptr);
		//	meshObj->mesh.edges[index].setPrev(nullptr);
		//}

		//// symmetry edge 
		//if (nFVerts_Sym == 3)
		//{


		//	// update pointers
		//	meshObj->mesh.edges[sEdge].getNext()->setNext(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getNext());
		//	meshObj->mesh.edges[sEdge].getNext()->setPrev(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getPrev());

		//	meshObj->mesh.edges[sEdge].getPrev()->setPrev(nullptr);
		//	meshObj->mesh.edges[sEdge].getPrev()->setNext(nullptr);

		//	meshObj->mesh.edges[sEdge].getPrev()->getSym()->setPrev(nullptr);
		//	meshObj->mesh.edges[sEdge].getPrev()->getSym()->setNext(nullptr);

		//	meshObj->mesh.edges[sEdge].getNext()->setFace(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace());

		//	if (meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace())
		//	{
		//		meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace()->setEdge(meshObj->mesh.edges[sEdge].getNext());
		//		meshObj->mesh.edges[sEdge].getPrev()->getSym()->setFace(nullptr);
		//	}

		//	// update vertex edge pointer if pointing to prev edge

		//	if (meshObj->mesh.vertices[vNext_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getPrev()->getEdgeId())
		//	{
		//		for (int i = 0; i < cEdgesVNext_sEdge.size(); i++)
		//		{
		//			if (cEdgesVNext_sEdge[i] != meshObj->mesh.edges[sEdge].getPrev()->getEdgeId())
		//			{
		//				meshObj->mesh.vertices[vNext_sEdge].setEdge(&meshObj->mesh.edges[cEdgesVNext_sEdge[i]]);
		//			}
		//		}
		//	}

		//	// update vertex edge pointer if pointing to prev edge

		//	if (meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getPrev()->getSym()->getEdgeId() || meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == sEdge)
		//	{
		//		for (int i = 0; i < cEdgesVPrev_sEdge.size(); i++)
		//		{
		//			if (cEdgesVPrev_sEdge[i] != meshObj->mesh.edges[sEdge].getPrev()->getSym()->getEdgeId() && cEdgesVPrev_sEdge[i] != sEdge)
		//			{
		//				meshObj->mesh.vertices[vPrev_sEdge].setEdge(&meshObj->mesh.edges[cEdgesVPrev_sEdge[i]]);
		//			}
		//		}
		//	}

		//	// decativate prev edge
		//	deactivateElement(meshObj->mesh.edges[sEdge].getPrev()->getEdgeId(), zEdgeData);

		//	// decativate next and sym pointer of the next edge are same, deactivate edge
		//	if (meshObj->mesh.edges[sEdge].getNext()->getNext() == meshObj->mesh.edges[sEdge].getNext()->getSym())
		//	{
		//		deactivateElement(meshObj->mesh.edges[sEdge].getNext()->getEdgeId(), zEdgeData);
		//		deactivateElement(vNext_sEdge, zVertexData);
		//	}

		//	// decativate prev and sym pointer of the next edge are same, deactivate edge
		//	else if (meshObj->mesh.edges[sEdge].getNext()->getPrev() == meshObj->mesh.edges[sEdge].getNext()->getSym())
		//	{
		//		deactivateElement(meshObj->mesh.edges[sEdge].getNext()->getEdgeId(), zEdgeData);
		//		deactivateElement(vNext_sEdge, zVertexData);
		//	}

		//	// deactivate face pointed by collapse edge
		//	deactivateElement(meshObj->mesh.edges[sEdge].getFace()->getFaceId(), zFaceData);

		//	meshObj->mesh.edges[sEdge].setFace(nullptr);

		//	meshObj->mesh.edges[sEdge].setNext(nullptr);
		//	meshObj->mesh.edges[sEdge].setPrev(nullptr);

		//}
		//else
		//{
		//	// update vertex edge pointer if pointing to current edge
		//	if (meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getEdgeId())
		//	{
		//		meshObj->mesh.vertices[vPrev_sEdge].setEdge(meshObj->mesh.edges[sEdge].getPrev()->getSym());
		//	}

		//	// update pointers
		//	meshObj->mesh.edges[sEdge].getNext()->setPrev(meshObj->mesh.edges[sEdge].getPrev());

		//	meshObj->mesh.edges[sEdge].setNext(nullptr);
		//	meshObj->mesh.edges[sEdge].setPrev(nullptr);
		//}

		//// update connected edges verter pointer
		//for (int i = 0; i < cEdges.size(); i++)
		//{
		//	if (meshObj->mesh.edgeActive[cEdges[i]])
		//	{
		//		int v1 = meshObj->mesh.edges[cEdges[i]].getVertex()->getVertexId();
		//		int v2 = vertexRemoveID;
		//		meshObj->mesh.removeFromVerticesEdge(v1, v2);

		//		meshObj->mesh.edges[cEdges[i]].getSym()->setVertex(&meshObj->mesh.vertices[vertexRetainID]);

		//		meshObj->mesh.addToVerticesEdge(v1, vertexRetainID, cEdges[i]);
		//	}
		//}


		//// deactivate collapse edge
		//if (meshObj->mesh.edgeActive[index])
		//{
		//	deactivateElement(index, zEdgeData);
		//}

		//// deactivate vertexRemoveID
		//if (meshObj->mesh.vertexActive[vertexRemoveID])
		//{
		//	deactivateElement(vertexRemoveID, zVertexData);
		//}

		//// compute normals		
		//computeMeshNormals();


		//// remove inactive elements
		//if (removeInactiveElems)
		//{
		//	removeInactiveElements(zVertexData);
		//	removeInactiveElements(zEdgeData);
		//	removeInactiveElements(zFaceData);
		//}

	}

	ZSPACE_INLINE zItMeshVertex zFnMesh::splitEdge(zItMeshEdge &edge, double edgeFactor)
	{

		zItMeshHalfEdge he = edge.getHalfEdge(0);
		zItMeshHalfEdge heS = edge.getHalfEdge(1);

		zItMeshHalfEdge he_next = he.getNext();
		zItMeshHalfEdge he_prev = he.getPrev();

		zItMeshHalfEdge heS_next = heS.getNext();
		zItMeshHalfEdge heS_prev = heS.getPrev();


		zVector edgeDir = he.getVector();
		double  edgeLength = edgeDir.length();
		edgeDir.normalize();

		zVector newVertPos = he.getStartVertex().getPosition() + edgeDir * edgeFactor * edgeLength;

		int numOriginalVertices = numVertices();

		// check if vertex exists if not add new vertex
		zItMeshVertex newVertex;
		addVertex(newVertPos, true, newVertex);



		//printf("\n newVert: %1.2f %1.2f %1.2f   %s ", newVertPos.x, newVertPos.y, newVertPos.z, (vExists)?"true":"false");

		if (newVertex.getId() >= numOriginalVertices)
		{
			// remove from halfEdge vertices map
			removeFromHalfEdgesMap(he);

			// add new edges
			int v1 = newVertex.getId();
			int v2 = he.getVertex().getId();

			zItMeshHalfEdge newHe;
			bool edgesResize = addEdges(v1, v2, false, newHe);

			// recompute iterators if resize is true
			if (edgesResize)
			{
				he = edge.getHalfEdge(0);
				heS = edge.getHalfEdge(1);

				he_next = he.getNext();
				he_prev = he.getPrev();

				heS_next = heS.getNext();
				heS_prev = heS.getPrev();

				//printf("\n working!");
			}

			zItMeshHalfEdge newHeS = newHe.getSym();

			// update vertex pointers
			newVertex.setHalfEdge(newHe);
			he.getVertex().setHalfEdge(newHeS);

			//// update pointers
			he.setVertex(newVertex);		// current edge vertex pointer updated to new added vertex

			newHeS.setNext(heS); // new added edge next pointer to point to the next of current edge
			newHeS.setPrev(heS_prev);

			if (!heS.onBoundary())
			{
				zItMeshFace heS_f = heS.getFace();
				newHeS.setFace(heS_f);
			}

			newHe.setPrev(he);
			newHe.setNext(he_next);

			if (!he.onBoundary())
			{
				zItMeshFace he_f = he.getFace();
				newHe.setFace(he_f);
			}


			// update verticesEdge map
			addToHalfEdgesMap(he);

		}


		return newVertex;
	}

	ZSPACE_INLINE int zFnMesh::detachEdge(int index) { return 0; }

	ZSPACE_INLINE void zFnMesh::flipTriangleEdge(int &index)
	{
		//if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		//if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		//zEdge* edgetoFlip = &meshObj->mesh.edges[index];
		//zEdge* edgetoFlipSym = edgetoFlip->getSym();

		//if (!edgetoFlip->getFace() || !edgetoFlipSym->getFace())
		//{
		//	throw std::invalid_argument("\n Cannot flip boundary edge %i ");
		//	return;
		//}

		//vector<int> edgetoFlip_fVerts;
		//getVertices(edgetoFlip->getFace()->getFaceId(), zFaceData, edgetoFlip_fVerts);

		//vector<int> edgetoFlipSym_fVerts;
		//getVertices(edgetoFlipSym->getFace()->getFaceId(), zFaceData, edgetoFlipSym_fVerts);

		//if (edgetoFlip_fVerts.size() != 3 || edgetoFlipSym_fVerts.size() != 3)
		//{
		//	throw std::invalid_argument("\n Cannot flip edge not shared by two Triangles.");
		//	return;
		//}

		//zEdge* e_next = edgetoFlip->getNext();
		//zEdge* e_prev = edgetoFlip->getPrev();

		//zEdge* es_next = edgetoFlipSym->getNext();
		//zEdge* es_prev = edgetoFlipSym->getPrev();

		//// remove from verticesEdge map
		//string removeHashKey = (to_string(edgetoFlip->getVertex()->getVertexId()) + "," + to_string(edgetoFlipSym->getVertex()->getVertexId()));
		//meshObj->mesh.verticesEdge.erase(removeHashKey);

		//string removeHashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
		//meshObj->mesh.verticesEdge.erase(removeHashKey1);

		//// update pointers

		//if (edgetoFlip->getVertex()->getEdge() == edgetoFlipSym)edgetoFlip->getVertex()->setEdge(edgetoFlipSym->getPrev()->getSym());
		//if (edgetoFlipSym->getVertex()->getEdge() == edgetoFlip) edgetoFlipSym->getVertex()->setEdge(edgetoFlip->getPrev()->getSym());

		//edgetoFlip->setVertex(e_next->getVertex());
		//edgetoFlipSym->setVertex(es_next->getVertex());



		//edgetoFlip->setNext(e_prev);
		//edgetoFlip->setPrev(es_next);

		//edgetoFlipSym->setNext(es_prev);
		//edgetoFlipSym->setPrev(e_next);

		//e_prev->setNext(es_next);
		//es_prev->setNext(e_next);

		//edgetoFlip->getNext()->setFace(edgetoFlip->getFace());
		//edgetoFlip->getPrev()->setFace(edgetoFlip->getFace());

		//edgetoFlipSym->getNext()->setFace(edgetoFlipSym->getFace());
		//edgetoFlipSym->getPrev()->setFace(edgetoFlipSym->getFace());

		//edgetoFlip->getFace()->setEdge(edgetoFlip);
		//edgetoFlipSym->getFace()->setEdge(edgetoFlipSym);

		//// update verticesEdge map

		//string hashKey = (to_string(edgetoFlip->getVertex()->getVertexId()) + "," + to_string(edgetoFlipSym->getVertex()->getVertexId()));
		//meshObj->mesh.verticesEdge[hashKey] = edgetoFlipSym->getEdgeId();

		//string hashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
		//meshObj->mesh.verticesEdge[hashKey1] = edgetoFlip->getEdgeId();
	}

	ZSPACE_INLINE void zFnMesh::splitFaces(vector<int> &edgeList, vector<double> &edgeFactor)
	{
		//if (edgeFactor.size() > 0)
		//{
		//	if (edgeList.size() != edgeFactor.size()) throw std::invalid_argument(" error: size of edgelist and edge factor dont match.");
		//}

		//int numOriginalVertices = meshObj->mesh.vertexActive.size();
		//int numOriginalEdges = meshObj->mesh.edgeActive.size();
		//int numOriginalFaces = meshObj->mesh.faceActive.size();

		//for (int i = 0; i < edgeList.size(); i++)
		//{
		//	if (edgeFactor.size() > 0) splitEdge( edgeList[i], edgeFactor[i]);
		//	else splitEdge( edgeList[i]);
		//}

		//for (int j = 0; j < edgeList.size(); j++)
		//{
		//	for (int i = 0; i < 2; i++)
		//	{
		//		zEdge *start = (i == 0) ? &meshObj->mesh.edges[edgeList[j]] : meshObj->mesh.edges[edgeList[j]].getSym();

		//		zEdge *e = start;

		//		if (!start->getFace()) continue;

		//		bool exit = false;

		//		int v1 = start->getVertex()->getVertexId();
		//		int v2 = start->getVertex()->getVertexId();

		//		do
		//		{
		//			if (e->getNext())
		//			{
		//				e = e->getNext();
		//				if (e->getVertex()->getVertexId() > numOriginalVertices)
		//				{
		//					v2 = e->getVertex()->getVertexId();
		//					exit = true;
		//				}
		//			}
		//			else exit = true;

		//		} while (e != start && !exit);

		//		// add new edges and face
		//		if (v1 == v2) continue;

		//		// check if edge exists continue loop. 
		//		int outEdgeId;
		//		bool eExists = meshObj->mesh.edgeExists(v1, v2, outEdgeId);

		//		if (eExists) continue;

		//		int startEdgeId = start->getEdgeId();
		//		int e_EdgeId = e->getEdgeId();

		//		bool resizeEdges = meshObj->mesh.addEdges(v1, v2);

		//		if (resizeEdges)
		//		{
		//			start = &meshObj->mesh.edges[startEdgeId];
		//			e = &meshObj->mesh.edges[e_EdgeId];
		//		}

		//		meshObj->mesh.addPolygon(); // empty polygon

		//							 // update pointers
		//		zEdge *start_next = start->getNext();
		//		zEdge *e_next = e->getNext();

		//		start->setNext(&meshObj->mesh.edges[numEdges() - 2]);
		//		e_next->setPrev(&meshObj->mesh.edges[numEdges() - 2]);

		//		start_next->setPrev(&meshObj->mesh.edges[numEdges() - 1]);
		//		e->setNext(&meshObj->mesh.edges[numEdges() - 1]);

		//		meshObj->mesh.faces[numPolygons() - 1].setEdge(start_next);

		//		// edge face pointers to new face
		//		zEdge *newFace_E = start_next;

		//		do
		//		{
		//			newFace_E->setFace(&meshObj->mesh.faces[numPolygons() - 1]);

		//			if (newFace_E->getNext()) newFace_E = newFace_E->getNext();
		//			else exit = true;

		//		} while (newFace_E != start_next && !exit);

		//	}

		//}


	}

	ZSPACE_INLINE void zFnMesh::subdivideMesh(int numDivisions)
	{

		for (int j = 0; j < numDivisions; j++)
		{

			int numOriginalVertices = numVertices();

			// split edges at center
			int numOriginalHalfEdges = numHalfEdges();

			int numOrginalEdges = numEdges();
			

			for (int i = 0; i < numOrginalEdges; i++)
			{
				zItMeshEdge e(*meshObj,i);

				if (e.isActive()) splitEdge(e);
			}


			// get face centers
			vector<zVector> fCenters;
			getCenters(zFaceData, fCenters);

			

			// add faces
			int numOriginalfaces = numPolygons();
			
			for (int i = 0; i < numOriginalfaces; i++)
			{
				zItMeshFace f(*meshObj, i);

				if (!f.isActive()) continue;

				vector<zItMeshHalfEdge> fEdges;
				f.getHalfEdges(fEdges);


				// disable current face
				f.deactivate();

				// check if vertex exists if not add new vertex
				zItMeshVertex vertexCen;
				addVertex(fCenters[i], true, vertexCen);


				// add new faces				
				int startId = 0;
				if (fEdges[0].getVertex().getId() < numOriginalVertices) startId = 1;

				for (int k = startId; k < fEdges.size() + startId; k += 2)
				{
					vector<int> newFVerts;

					newFVerts.push_back(fEdges[k].getVertex().getId());

					newFVerts.push_back(vertexCen.getId());

					newFVerts.push_back(fEdges[k].getPrev().getStartVertex().getId());

					newFVerts.push_back(fEdges[k].getPrev().getVertex().getId());

					zItMeshFace newF;
				
					bool chk = addPolygon(newFVerts, newF);

					//printf("\n %s ", (chk) ? "true" : "false");
				}

			}


			// update half edge handles. 
			for (int i = 0; i < meshObj->mesh.heHandles.size(); i++)
			{
				if(meshObj->mesh.heHandles[i].f != -1) meshObj->mesh.heHandles[i].f -= numOriginalfaces;
			}
					

			// remove inactive faces
			garbageCollection(zFaceData);


			computeMeshNormals();

		}

	}

	ZSPACE_INLINE void zFnMesh::smoothMesh(int numDivisions, bool smoothCorner)
	{
		vector<zVector> fCenters;
		vector<zVector> tempECenters;
		vector<zVector> eCenters;

		for (int j = 0; j < numDivisions; j++)
		{


			// get face centers
			fCenters.clear();
			getCenters(zFaceData, fCenters);

			// get edge centers
		
			tempECenters.clear();

			eCenters.clear();
			getCenters(zEdgeData, eCenters);

			tempECenters = eCenters;

			zVector* vPositions = 	getRawVertexPositions();

			int numOriginalVertices = numVertices();
			int numOriginalEdges = numEdges();

			// compute new smooth positions of the edge centers
			for (int i =0; i< numOriginalEdges; i++)
			{

				zItMeshEdge e(*meshObj, i);

				if (e.onBoundary()) continue;

				zVector newPos;			

				vector<int> eVerts;
				e.getVertices(eVerts);
				for (auto &vId : eVerts) newPos += vPositions[vId];


				vector<int> eFaces;
				e.getFaces(eFaces);
				for (auto &fId : eFaces) newPos += fCenters[fId];

				newPos /= (eFaces.size() + eVerts.size());

				eCenters[i] = newPos;
			}

			// compute new smooth positions for the original vertices
			for (int i = 0; i < numOriginalVertices; i++)
			{

				zItMeshVertex v(*meshObj, i);

				if (v.onBoundary())
				{
					vector<zItMeshEdge> cEdges;
					v.getConnectedEdges(cEdges);

					if (!smoothCorner && cEdges.size() == 2) continue;

					zVector P = vPositions[i];
					//int n = 1; // rosetta , not matching with maya
					int n = 0;

					zVector R(0,0,0);
					for (auto &e : cEdges)
					{
						if (e.onBoundary())
						{
							R += tempECenters[e.getId()];
							n++;
						}
					}

					// rosetta , not matching with maya
					//vPositions[i] = (P + R) / n; 

					vPositions[i] = (P / n) + (R / (n*n));				

				}
				else
				{
					zVector R;

					vector<int> cEdges;
					v.getConnectedEdges(cEdges);

					for (auto &eId : cEdges) R += tempECenters[eId];
					R /= cEdges.size();

					zVector F;
					vector<int> cFaces;
					v.getConnectedFaces(cFaces);
					for (auto &fId : cFaces) F += fCenters[fId];
					F /= cFaces.size();

					zVector P = vPositions[i];
					int n = cFaces.size();

					vPositions[i] = (F + (R * 2) + (P * (n - 3))) / n;
				}


			}


			

			// split edges at center
			
			for (int i = 0; i < numOriginalEdges; i++)
			{
				zItMeshEdge e(*meshObj,i);
				if (e.isActive())
				{			
					zItMeshVertex newVert = splitEdge(e);
					newVert.setPosition(eCenters[i]);
				}
			}



			// add faces
			int numOriginalFaces = numPolygons();

			
			for (int i = 0; i < numOriginalFaces; i++)
			{
				zItMeshFace f(*meshObj, i);

				if (!f.isActive()) continue;

				vector<zItMeshHalfEdge> fEdges;
				f.getHalfEdges(fEdges);


				// disable current face
				f.deactivate();

				// check if vertex exists if not add new vertex
				zItMeshVertex vertexCen;
				addVertex(fCenters[i], true, vertexCen);


				// add new faces				
				int startId = 0;
				if (fEdges[0].getVertex().getId() < numOriginalVertices) startId = 1;

				for (int k = startId; k < fEdges.size() + startId; k += 2)
				{
					vector<int> newFVerts;

					newFVerts.push_back(fEdges[k].getVertex().getId());

					newFVerts.push_back(vertexCen.getId());

					newFVerts.push_back(fEdges[k].getPrev().getStartVertex().getId());

					newFVerts.push_back(fEdges[k].getPrev().getVertex().getId());

					zItMeshFace newF;
					addPolygon(newFVerts, newF);

				}


			}

			// update half edge handles. 
			for (int i = 0; i < meshObj->mesh.heHandles.size(); i++)
			{
				if (meshObj->mesh.heHandles[i].f != -1) meshObj->mesh.heHandles[i].f -= numOriginalFaces;
			}

			// remove inactive faces
			garbageCollection(zFaceData);

			computeMeshNormals();

		}	


	}

	ZSPACE_INLINE zObjMesh zFnMesh::extrudeMesh(float extrudeThickness, bool thicknessTris)
	{
		if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faces.size()) computeMeshNormals();

		zObjMesh out;

		vector<zVector> positions;
		vector<int> polyCounts;
		vector<int> polyConnects;



		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			positions.push_back(meshObj->mesh.vertexPositions[i]);
		}

		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			positions.push_back(meshObj->mesh.vertexPositions[i] + (meshObj->mesh.vertexNormals[i] * extrudeThickness));
		}

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> fVerts;
			f.getVertices(fVerts);

			for (int j = 0; j < fVerts.size(); j++)
			{
				polyConnects.push_back(fVerts[j]);
			}

			polyCounts.push_back(fVerts.size());

			for (int j = fVerts.size() - 1; j >= 0; j--)
			{
				polyConnects.push_back(fVerts[j] + meshObj->mesh.vertexPositions.size());
			}

			polyCounts.push_back(fVerts.size());

		}


		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{
			if (he.onBoundary())
			{
				vector<int> eVerts;
				he.getVertices(eVerts);

				if (thicknessTris)
				{
					polyConnects.push_back(eVerts[1]);
					polyConnects.push_back(eVerts[0]);
					polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());

					polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1] + meshObj->mesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1]);

					polyCounts.push_back(3);
					polyCounts.push_back(3);
				}
				else
				{
					polyConnects.push_back(eVerts[1]);
					polyConnects.push_back(eVerts[0]);
					polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());
					polyConnects.push_back(eVerts[1] + meshObj->mesh.vertexPositions.size());


					polyCounts.push_back(4);
				}


			}
		}

		out.mesh.create(positions, polyCounts, polyConnects);

		return out;
	}

	//---- TRANSFORM METHODS OVERRIDES

	ZSPACE_INLINE void zFnMesh::setTransform(zTransform &inTransform, bool decompose, bool updatePositions)
	{
		if (updatePositions)
		{
			zTransformationMatrix to;
			to.setTransform(inTransform, decompose);

			zTransform transMat = meshObj->transformationMatrix.getToMatrix(to);
			transformObject(transMat);

			meshObj->transformationMatrix.setTransform(inTransform);

			// update pivot values of object transformation matrix
			zVector p = meshObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}
		else
		{
			meshObj->transformationMatrix.setTransform(inTransform, decompose);

			zVector p = meshObj->transformationMatrix.getO();
			setPivot(p);

		}

	}

	ZSPACE_INLINE void zFnMesh::setScale(zDouble3 &scale)
	{
		// get  inverse pivot translations
		zTransform invScalemat = meshObj->transformationMatrix.asInverseScaleTransformMatrix();

		// set scale values of object transformation matrix
		meshObj->transformationMatrix.setScale(scale);

		// get new scale transformation matrix
		zTransform scaleMat = meshObj->transformationMatrix.asScaleTransformMatrix();

		// compute total transformation
		zTransform transMat = invScalemat * scaleMat;

		// transform object
		transformObject(transMat);
	}

	ZSPACE_INLINE void zFnMesh::setRotation(zDouble3 &rotation, bool appendRotations)
	{
		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = meshObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = meshObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = meshObj->transformationMatrix;
		to.setRotation(rotation, appendRotations);
		zTransform toMat = meshObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set rotation values of object transformation matrix
		meshObj->transformationMatrix.setRotation(rotation, appendRotations);;
	}

	ZSPACE_INLINE void zFnMesh::setTranslation(zVector &translation, bool appendTranslations)
	{
		// get vector as zDouble3
		zDouble3 t;
		translation.getComponents(t);

		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = meshObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = meshObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = meshObj->transformationMatrix;
		to.setTranslation(t, appendTranslations);
		zTransform toMat = meshObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set translation values of object transformation matrix
		meshObj->transformationMatrix.setTranslation(t, appendTranslations);;

		// update pivot values of object transformation matrix
		zVector p = meshObj->transformationMatrix.getPivot();
		p = p * transMat;
		setPivot(p);

	}

	ZSPACE_INLINE void zFnMesh::setPivot(zVector &pivot)
	{
		// get vector as zDouble3
		zDouble3 p;
		pivot.getComponents(p);

		// set pivot values of object transformation matrix
		meshObj->transformationMatrix.setPivot(p);
	}

	ZSPACE_INLINE void zFnMesh::getTransform(zTransform &transform)
	{
		transform = meshObj->transformationMatrix.asMatrix();
	}

	//---- PROTECTED TRANSFORM  METHODS

	ZSPACE_INLINE void zFnMesh::transformObject(zTransform &transform)
	{

		if (numVertices() == 0) return;


		zVector* pos = getRawVertexPositions();

		for (int i = 0; i < numVertices(); i++)
		{

			zVector newPos = pos[i] * transform;
			pos[i] = newPos;
		}

	}

	//---- FACTORY METHODS

	ZSPACE_INLINE void zFnMesh::toOBJ(string outfilename)
	{

		// remove inactive elements
		if (numVertices() != meshObj->mesh.vertices.size()) garbageCollection(zVertexData);
		if (numEdges() != meshObj->mesh.edges.size()) garbageCollection(zEdgeData);
		if (numPolygons() != meshObj->mesh.faces.size()) garbageCollection(zFaceData);

		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}



		// vertex positions
		for (auto &vPos : meshObj->mesh.vertexPositions)
		{
			myfile << "\n v " << vPos.x << " " << vPos.y << " " << vPos.z;

		}

		// vertex nornmals
		/*for (auto &vNorm : meshObj->mesh.vertexNormals)
		{
			myfile << "\n vn " << vNorm.x << " " << vNorm.y << " " << vNorm.z;

		}*/

		myfile << "\n";

		// face connectivity
		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> fVerts;
			f.getVertices(fVerts);

			myfile << "\n f ";

			for (int j = 0; j < fVerts.size(); j++)
			{
				myfile << fVerts[j] + 1;

				if (j != fVerts.size() - 1) myfile << " ";
			}

		}




		myfile.close();

		cout << endl << " OBJ exported. File:   " << outfilename.c_str() << endl;

	}

	ZSPACE_INLINE void zFnMesh::toJSON(string outfilename)
	{

		// remove inactive elements
		if (numVertices() != meshObj->mesh.vertices.size()) garbageCollection(zVertexData);
		if (numEdges() != meshObj->mesh.edges.size()) garbageCollection(zEdgeData);
		if (numPolygons() != meshObj->mesh.faces.size())garbageCollection(zFaceData);

		// CREATE JSON FILE
		zUtilsJsonHE meshJSON;
		json j;

		// Vertices
		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			if (v.getHalfEdge().isActive()) meshJSON.vertices.push_back(v.getHalfEdge().getId());
			else meshJSON.vertices.push_back(-1);

		}

		//Edges
		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{
			vector<int> HE_edges;

			if (he.getPrev().isActive()) HE_edges.push_back(he.getPrev().getId());
			else HE_edges.push_back(-1);

			if (he.getNext().isActive()) HE_edges.push_back(he.getNext().getId());
			else HE_edges.push_back(-1);

			if (he.getVertex().isActive()) HE_edges.push_back(he.getVertex().getId());
			else HE_edges.push_back(-1);

			if (!he.onBoundary()) HE_edges.push_back(he.getFace().getId());
			else HE_edges.push_back(-1);

			meshJSON.halfedges.push_back(HE_edges);
		}

		// Faces
		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			if (f.getHalfEdge().isActive()) meshJSON.faces.push_back(f.getHalfEdge().getId());
			else meshJSON.faces.push_back(-1);
		}

		// vertex Attributes
		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			vector<double> v_attrib;

			v_attrib.push_back(meshObj->mesh.vertexPositions[i].x);
			v_attrib.push_back(meshObj->mesh.vertexPositions[i].y);
			v_attrib.push_back(meshObj->mesh.vertexPositions[i].z);

			v_attrib.push_back(meshObj->mesh.vertexNormals[i].x);
			v_attrib.push_back(meshObj->mesh.vertexNormals[i].y);
			v_attrib.push_back(meshObj->mesh.vertexNormals[i].z);


			v_attrib.push_back(meshObj->mesh.vertexColors[i].r);
			v_attrib.push_back(meshObj->mesh.vertexColors[i].g);
			v_attrib.push_back(meshObj->mesh.vertexColors[i].b);


			meshJSON.vertexAttributes.push_back(v_attrib);
		}

		// face Attributes
		for (int i = 0; i < numPolygons(); i++)
		{
			vector<double> f_attrib;

			f_attrib.push_back(meshObj->mesh.faceNormals[i].x);
			f_attrib.push_back(meshObj->mesh.faceNormals[i].y);
			f_attrib.push_back(meshObj->mesh.faceNormals[i].z);

			f_attrib.push_back(meshObj->mesh.faceColors[i].r);
			f_attrib.push_back(meshObj->mesh.faceColors[i].g);
			f_attrib.push_back(meshObj->mesh.faceColors[i].b);

			meshJSON.faceAttributes.push_back(f_attrib);
		}



		// Json file 
		j["Vertices"] = meshJSON.vertices;
		j["Halfedges"] = meshJSON.halfedges;
		j["Faces"] = meshJSON.faces;
		j["VertexAttributes"] = meshJSON.vertexAttributes;
		j["HalfedgeAttributes"] = meshJSON.halfedgeAttributes;
		j["FaceAttributes"] = meshJSON.faceAttributes;

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

	ZSPACE_INLINE bool zFnMesh::fromOBJ(string infilename)
	{
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zVector>  vertexNormals;
		vector<zVector>  faceNormals;

		ifstream myfile;
		myfile.open(infilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << infilename.c_str() << endl;
			return false;

		}

		while (!myfile.eof())
		{
			string str;
			getline(myfile, str);

			vector<string> perlineData = meshObj->mesh.coreUtils.splitString(str, " ");

			if (perlineData.size() > 0)
			{
				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						positions.push_back(pos);
					}
					//printf("\n working vertex");
				}

				// vertex normal
				if (perlineData[0] == "vn")
				{
					//printf("\n working vertex normal ");
					if (perlineData.size() == 4)
					{
						zVector norm;
						norm.x = atof(perlineData[1].c_str());
						norm.y = atof(perlineData[2].c_str());
						norm.z = atof(perlineData[3].c_str());

						vertexNormals.push_back(norm);
					}
					//printf("\n working vertex");
				}

				// face
				if (perlineData[0] == "f")
				{

					zVector norm;



					for (int i = 1; i < perlineData.size(); i++)
					{
						vector<string> faceData = meshObj->mesh.coreUtils.splitString(perlineData[i], "/");

						//vector<string> cleanFaceData = splitString(faceData[0], "/\/");

						int id = atoi(faceData[0].c_str()) - 1;
						polyConnects.push_back(id);

						//printf(" %i ", id);

						int normId = atoi(faceData[faceData.size() - 1].c_str()) - 1;
						norm += vertexNormals[normId];

					}

					norm /= (perlineData.size() - 1);
					norm.normalize();
					faceNormals.push_back(norm);

					polyCounts.push_back(perlineData.size() - 1);
					//printf("\n working face ");
				}
			}
		}

		myfile.close();


		meshObj->mesh.create(positions, polyCounts, polyConnects);;
		printf("\n mesh: %i %i %i", numVertices(), numEdges(), numPolygons());


		setFaceNormals(faceNormals);

		return true;
	}

	ZSPACE_INLINE bool zFnMesh::fromJSON(string infilename)
	{
		json j;
		zUtilsJsonHE meshJSON;


		ifstream in_myfile;
		in_myfile.open(infilename.c_str());

		int lineCnt = 0;

		if (in_myfile.fail())
		{
			cout << " error in opening file  " << infilename.c_str() << endl;
			return false;
		}

		in_myfile >> j;
		in_myfile.close();

		// READ Data from JSON

		// Vertices
		meshJSON.vertices.clear();
		meshJSON.vertices = (j["Vertices"].get<vector<int>>());

		//Edges
		meshJSON.halfedges.clear();
		meshJSON.halfedges = (j["Halfedges"].get<vector<vector<int>>>());

		// Faces
		meshJSON.faces.clear();
		meshJSON.faces = (j["Faces"].get<vector<int>>());


		// update  mesh
		meshObj->mesh.clear();

		meshObj->mesh.vertices.assign(meshJSON.vertices.size(), zVertex());
		meshObj->mesh.halfEdges.assign(meshJSON.halfedges.size(), zHalfEdge());

		int numE = (int)floor(meshJSON.halfedges.size()*0.5);
		meshObj->mesh.edges.assign(numE, zEdge());
		meshObj->mesh.faces.assign(meshJSON.faces.size(), zFace());

		meshObj->mesh.vHandles.assign(meshJSON.vertices.size(), zVertexHandle());
		meshObj->mesh.eHandles.assign(numE, zEdgeHandle());
		meshObj->mesh.heHandles.assign(meshJSON.halfedges.size(), zHalfEdgeHandle());
		meshObj->mesh.fHandles.assign(meshJSON.faces.size(), zFaceHandle());

		// set IDs
		for (int i = 0; i < meshJSON.vertices.size(); i++) meshObj->mesh.vertices[i].setId(i);
		for (int i = 0; i < meshJSON.halfedges.size(); i++) meshObj->mesh.halfEdges[i].setId(i);
		for (int i = 0; i < meshJSON.faces.size(); i++) meshObj->mesh.faces[i].setId(i);

		// set Pointers
		int n_v = 0;
		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			v.setId(n_v);

			if (meshJSON.vertices[n_v] != -1)
			{
				zItMeshHalfEdge he(*meshObj, meshJSON.vertices[n_v]);;
				v.setHalfEdge(he);

				meshObj->mesh.vHandles[n_v].id = n_v;
				meshObj->mesh.vHandles[n_v].he = meshJSON.vertices[n_v];
			}



			n_v++;
		}
		meshObj->mesh.setNumVertices(n_v);

		int n_he = 0;
		int n_e = 0;

		for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
		{

			// Half Edge
			he.setId(n_he);
			meshObj->mesh.heHandles[n_he].id = n_he;

			if (meshJSON.halfedges[n_he][0] != -1)
			{
				zItMeshHalfEdge hePrev(*meshObj, meshJSON.halfedges[n_he][0]);
				he.setPrev(hePrev);

				meshObj->mesh.heHandles[n_he].p = meshJSON.halfedges[n_he][0];
			}

			if (meshJSON.halfedges[n_he][1] != -1)
			{
				zItMeshHalfEdge heNext(*meshObj, meshJSON.halfedges[n_he][1]);
				he.setNext(heNext);

				meshObj->mesh.heHandles[n_he].n = meshJSON.halfedges[n_he][1];
			}

			if (meshJSON.halfedges[n_he][2] != -1)
			{
				zItMeshVertex v(*meshObj, meshJSON.halfedges[n_he][2]);
				he.setVertex(v);


				meshObj->mesh.heHandles[n_he].v = meshJSON.halfedges[n_he][2];
			}

			if (meshJSON.halfedges[n_he][3] != -1)
			{
				zItMeshFace f(*meshObj, meshJSON.halfedges[n_he][3]);
				he.setFace(f);

				meshObj->mesh.heHandles[n_he].f = meshJSON.halfedges[n_he][3];
			}

			// symmetry half edges && Edge
			if (n_he % 2 == 1)
			{
				zItMeshHalfEdge heSym(*meshObj, n_he - 1);
				he.setSym(heSym);

				zItMeshEdge e(*meshObj, n_e);
				e.setId(n_e);

				e.setHalfEdge(heSym, 0);
				e.setHalfEdge(he, 1);

				he.setEdge(e);
				heSym.setEdge(e);

				meshObj->mesh.heHandles[n_he].e = n_e;
				meshObj->mesh.heHandles[n_he - 1].e = n_e;

				meshObj->mesh.eHandles[n_e].id = n_e;
				meshObj->mesh.eHandles[n_e].he0 = n_he - 1;
				meshObj->mesh.eHandles[n_e].he1 = n_he;

				n_e++;
			}

			n_he++;

		}


		meshObj->mesh.setNumEdges(n_e);

		int n_f = 0;

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			f.setId(n_f);

			if (meshJSON.vertices[n_f] != -1)
			{
				zItMeshHalfEdge he(*meshObj, meshJSON.faces[n_f]);
				f.setHalfEdge(he);

				meshObj->mesh.fHandles[n_f].id = n_f;
				meshObj->mesh.fHandles[n_f].he = meshJSON.faces[n_f];
			}



			n_f++;
		}
		meshObj->mesh.setNumPolygons(n_f);



		//// Vertex Attributes
		meshJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
		//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

		meshObj->mesh.vertexPositions.clear();
		meshObj->mesh.vertexNormals.clear();
		meshObj->mesh.vertexColors.clear();
		meshObj->mesh.vertexWeights.clear();
		for (int i = 0; i < meshJSON.vertexAttributes.size(); i++)
		{
			for (int k = 0; k < meshJSON.vertexAttributes[i].size(); k++)
			{

				// position and color

				if (meshJSON.vertexAttributes[i].size() == 9)
				{
					zVector pos(meshJSON.vertexAttributes[i][k], meshJSON.vertexAttributes[i][k + 1], meshJSON.vertexAttributes[i][k + 2]);
					meshObj->mesh.vertexPositions.push_back(pos);

					zVector normal(meshJSON.vertexAttributes[i][k + 3], meshJSON.vertexAttributes[i][k + 4], meshJSON.vertexAttributes[i][k + 5]);
					meshObj->mesh.vertexNormals.push_back(normal);

					zColor col(meshJSON.vertexAttributes[i][k + 6], meshJSON.vertexAttributes[i][k + 7], meshJSON.vertexAttributes[i][k + 8], 1);
					meshObj->mesh.vertexColors.push_back(col);

					meshObj->mesh.vertexWeights.push_back(2.0);

					k += 8;
				}
			}
		}


		// Edge Attributes
		meshJSON.halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();

		meshObj->mesh.edgeColors.clear();
		meshObj->mesh.edgeWeights.clear();
		if (meshJSON.halfedgeAttributes.size() == 0)
		{


			for (int i = 0; i < meshObj->mesh.n_e; i++)
			{
				meshObj->mesh.edgeColors.push_back(zColor());
				meshObj->mesh.edgeWeights.push_back(1.0);
			}
		}
		else
		{
			for (int i = 0; i < meshJSON.halfedgeAttributes.size(); i += 2)
			{
				// color
				if (meshJSON.halfedgeAttributes[i].size() == 3)
				{
					zColor col(meshJSON.halfedgeAttributes[i][0], meshJSON.halfedgeAttributes[i][1], meshJSON.halfedgeAttributes[i][2], 1);

					meshObj->mesh.edgeColors.push_back(col);
					meshObj->mesh.edgeWeights.push_back(1.0);

				}
			}
		}

		// face Attributes
		meshJSON.faceAttributes = j["FaceAttributes"].get<vector<vector<double>>>();

		meshObj->mesh.faceColors.clear();
		meshObj->mesh.faceNormals.clear();
		for (int i = 0; i < meshJSON.faceAttributes.size(); i++)
		{
			for (int k = 0; k < meshJSON.faceAttributes[i].size(); k++)
			{
				// normal and color
				if (meshJSON.faceAttributes[i].size() == 6)
				{
					zColor col(meshJSON.faceAttributes[i][k + 3], meshJSON.faceAttributes[i][k + 4], meshJSON.faceAttributes[i][k + 5], 1);
					meshObj->mesh.faceColors.push_back(col);


					zVector normal(meshJSON.faceAttributes[i][k], meshJSON.faceAttributes[i][k + 1], meshJSON.faceAttributes[i][k + 2]);
					meshObj->mesh.faceNormals.push_back(normal);

					k += 5;
				}

				if (meshJSON.faceAttributes[i].size() == 3)
				{
					zVector normal(meshJSON.faceAttributes[i][k], meshJSON.faceAttributes[i][k + 1], meshJSON.faceAttributes[i][k + 2]);
					meshObj->mesh.faceNormals.push_back(normal);


					meshObj->mesh.faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));

					k += 2;
				}
			}

		}

		if (meshJSON.faceAttributes.size() == 0)
		{
			computeMeshNormals();
			setFaceColor(zColor(0.5, 0.5, 0.5, 1));
		}


		// add to maps 
		for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
		{
			meshObj->mesh.addToPositionMap(meshObj->mesh.vertexPositions[i], i);
		}


		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			int v1 = e.getHalfEdge(0).getVertex().getId();
			int v2 = e.getHalfEdge(1).getVertex().getId();

			meshObj->mesh.addToHalfEdgesMap(v1, v2, e.getHalfEdge(0).getId());
		}



		printf("\n mesh: %i %i %i ", numVertices(), numEdges(), numPolygons());

		return true;

	}

	//---- PRIVATE METHODS

	ZSPACE_INLINE void zFnMesh::setStaticContainers()
	{
		meshObj->mesh.staticGeometry = true;

		vector<vector<int>> edgeVerts;

		for (zItMeshEdge e(*meshObj); !e.end(); e++)
		{
			vector<int> verts;
			e.getVertices(verts);

			edgeVerts.push_back(verts);
		}

		meshObj->mesh.setStaticEdgeVertices(edgeVerts);


		vector<vector<int>> faceVerts;

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> verts;
			f.getVertices(verts);

			faceVerts.push_back(verts);
		}

		meshObj->mesh.setStaticFaceVertices(faceVerts);


	}

	//---- PRIVATE DEACTIVATE AND REMOVE METHODS

	ZSPACE_INLINE void zFnMesh::addToHalfEdgesMap(zItMeshHalfEdge &he)
	{
		meshObj->mesh.addToHalfEdgesMap(he.getStartVertex().getId(), he.getVertex().getId(), he.getId());
	}

	ZSPACE_INLINE void zFnMesh::removeFromHalfEdgesMap(zItMeshHalfEdge &he)
	{
		meshObj->mesh.removeFromHalfEdgesMap(he.getStartVertex().getId(), he.getVertex().getId());
	}

	ZSPACE_INLINE void zFnMesh::removeInactive(zHEData type)
	{
		//  Vertex		
		if (type == zVertexData)
		{
			bool reindex = false;

			for (int i = 0; i < meshObj->mesh.vHandles.size(); i++)
			{
				bool active = (meshObj->mesh.vHandles[i].id == -1) ? false : true;

				if (!active)
				{
					reindex = true;
					while (meshObj->mesh.vHandles[i].id == -1)
					{
						meshObj->mesh.faceColors.erase(meshObj->mesh.faceColors.begin() + i);

						meshObj->mesh.faceNormals.erase(meshObj->mesh.faceNormals.begin() + i);

						meshObj->mesh.vHandles.erase(meshObj->mesh.vHandles.begin() + i);

						meshObj->mesh.n_v--;

					}
				}


			}

			for (int i = 0; i < meshObj->mesh.vHandles.size(); i++) meshObj->mesh.vHandles[i].id = i;

			meshObj->mesh.resizeArray(zVertexData, numVertices());

			printf("\n removed inactive vertices. ");


			/*vector<zVertexHandle>::iterator v = meshObj->mesh.vHandles.begin();

			while (v != meshObj->mesh.vHandles.end())
			{

				

				bool active = (v->id == -1) ? false : true;

				if (!active)
				{
					meshObj->mesh.vertexPositions.erase(meshObj->mesh.vertexPositions.begin() + v->id);

					meshObj->mesh.vertexColors.erase(meshObj->mesh.vertexColors.begin() + v->id);

					meshObj->mesh.vertexWeights.erase(meshObj->mesh.vertexWeights.begin() + v->id);

					meshObj->mesh.vertexNormals.erase(meshObj->mesh.vertexNormals.begin() + v->id);

					meshObj->mesh.vHandles.erase(v++);

					meshObj->mesh.n_v--;
				}
			}

			meshObj->mesh.resizeArray(zVertexData, numVertices());

			printf("\n removed inactive vertices. ");*/

		}

		//  Edge
		else if (type == zEdgeData || type == zHalfEdgeData)
		{
			vector<zHalfEdgeHandle>::iterator he = meshObj->mesh.heHandles.begin();

			while (he != meshObj->mesh.heHandles.end())
			{
				bool active = (he->id == -1) ? false : true;

				if (!active)
				{
					meshObj->mesh.heHandles.erase(he++);

					meshObj->mesh.n_he--;
				}
			}

			vector<zEdgeHandle>::iterator e = meshObj->mesh.eHandles.begin();


			while (e != meshObj->mesh.eHandles.end())
			{
				bool active = (e->id == -1) ? false : true;

				if (!active)
				{
					meshObj->mesh.edgeColors.erase(meshObj->mesh.edgeColors.begin() + e->id);

					meshObj->mesh.edgeWeights.erase(meshObj->mesh.edgeWeights.begin() + e->id);

					meshObj->mesh.eHandles.erase(e++);

					meshObj->mesh.n_e--;
				}
			}

			printf("\n removed inactive edges and had edges. ");

			meshObj->mesh.resizeArray(zHalfEdgeData, numHalfEdges());

			meshObj->mesh.resizeArray(zEdgeData, numHalfEdges());

		}

		// Mesh Face
		else if (type == zFaceData)
		{
			
			bool reindex = false;

			for (int i = 0; i < meshObj->mesh.fHandles.size(); i++)
			{
				bool active = (meshObj->mesh.fHandles[i].id == -1) ? false : true;

				if (!active)
				{
					reindex = true;
					while (meshObj->mesh.fHandles[i].id == -1)
					{
						meshObj->mesh.faceColors.erase(meshObj->mesh.faceColors.begin() + i);

						meshObj->mesh.faceNormals.erase(meshObj->mesh.faceNormals.begin() +i);

						meshObj->mesh.fHandles.erase(meshObj->mesh.fHandles.begin() + i);

						meshObj->mesh.n_f--;		
						
					}
				}
				

			}		

	
			for (int i = 0; i < meshObj->mesh.fHandles.size(); i++) meshObj->mesh.fHandles[i].id = i;
		

			meshObj->mesh.resizeArray(zFaceData, meshObj->mesh.fHandles.size());

			printf("\n removed inactive faces. ");
		}

		else throw std::invalid_argument(" error: invalid zHEData type");
	}

}