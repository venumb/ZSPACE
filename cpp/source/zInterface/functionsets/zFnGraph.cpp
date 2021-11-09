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


#include<headers/zInterface/functionsets/zFnGraph.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnGraph::zFnGraph()
	{
		fnType = zFnType::zGraphFn;
		graphObj = nullptr;
	}

	ZSPACE_INLINE zFnGraph::zFnGraph(zObjGraph &_graphObj, bool  _planarGraph, zVector _graphNormal)
	{
		fnType = zFnType::zGraphFn;

		graphObj = &_graphObj;

		planarGraph = _planarGraph;
		graphNormal = _graphNormal;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnGraph::~zFnGraph() {}

	//---- OVERRIDE METHODS
	
	ZSPACE_INLINE zFnType zFnGraph::getType()
	{
		return zGraphFn;
	}

	ZSPACE_INLINE void zFnGraph::from(string path, zFileTpye type, bool staticGeom)
	{
		if (type == zTXT)
		{
			fromTXT(path);
			setStaticContainers();
		}
		else if (type == zMAYATXT)
		{
			fromMAYATXT(path);
			setStaticContainers();
		}
		else if (type == zJSON)
		{
			fromJSON(path);
			setStaticContainers();
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnGraph::to(string path, zFileTpye type)
	{
		if (type == zTXT) toTXT(path);
		else if (type == zJSON) toJSON(path);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnGraph::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		graphObj->getBounds(minBB, maxBB);
	}

	ZSPACE_INLINE void zFnGraph::clear()
	{
		graphObj->graph.clear();
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnGraph::create(zPointArray(&_positions), zIntArray(&edgeConnects), bool staticGraph,int precision)
	{
		graphObj->graph.create(_positions, edgeConnects,staticGraph, precision);

		if (staticGraph) setStaticContainers();
	}

	ZSPACE_INLINE void zFnGraph::create(zPointArray(&_positions), zIntArray(&edgeConnects), zVector &graphNormal, bool staticGraph)
	{

		graphNormal.normalize();

		zVector x(1, 0, 0);
		zVector sortRef = graphNormal ^ x;

		graphObj->graph.create(_positions, edgeConnects, graphNormal, sortRef);

		if (staticGraph) setStaticContainers();
	}

	ZSPACE_INLINE void zFnGraph::createFromMesh(zObjMesh &meshObj, bool excludeBoundary, bool staticGraph)
	{
		zFnMesh fnMesh(meshObj);

		vector<int>edgeConnects;
		vector<zVector> vertexPositions;

		fnMesh.getVertexPositions(vertexPositions, excludeBoundary);
		fnMesh.getEdgeData(edgeConnects, excludeBoundary);

		create(vertexPositions, edgeConnects, staticGraph);

		if (staticGraph) setStaticContainers();
	}

	ZSPACE_INLINE bool zFnGraph::addVertex(zPoint &_pos, bool checkDuplicates, zItGraphVertex &vertex)
	{
		if (checkDuplicates)
		{
			int id;
			bool chk = vertexExists(_pos, vertex);
			if (chk)	return false;

		}

		bool out = graphObj->graph.addVertex(_pos);
		vertex = zItGraphVertex(*graphObj, numVertices() - 1);

		return out;
	}

	ZSPACE_INLINE bool zFnGraph::addEdges(int &v1, int &v2, bool checkDuplicates, zItGraphHalfEdge &halfEdge)
	{
		if (v1 < 0 && v1 >= numVertices()) throw std::invalid_argument(" error: index out of bounds");
		if (v2 < 0 && v2 >= numVertices()) throw std::invalid_argument(" error: index out of bounds");

		if (checkDuplicates)
		{
			int id;
			bool chk = halfEdgeExists(v1, v2, id);
			if (chk)
			{
				halfEdge = zItGraphHalfEdge(*graphObj, id);
				return false;
			}
		}

		bool out = graphObj->graph.addEdges(v1, v2);

		halfEdge = zItGraphHalfEdge(*graphObj, numHalfEdges() - 2);

		return out;
	}

	//--- TOPOLOGY QUERY METHODS 

	ZSPACE_INLINE int zFnGraph::numVertices()
	{
		return graphObj->graph.n_v;
	}

	ZSPACE_INLINE int zFnGraph::numEdges()
	{
		return graphObj->graph.n_e;
	}

	ZSPACE_INLINE int zFnGraph::numHalfEdges()
	{
		return graphObj->graph.n_he;
	}

	ZSPACE_INLINE bool zFnGraph::vertexExists(zPoint pos, zItGraphVertex &outVertex, int precisionfactor)
	{

		int id;
		bool chk = graphObj->graph.vertexExists(pos, id, precisionfactor);

		if (chk) outVertex = zItGraphVertex(*graphObj, id);

		return chk;
	}

	ZSPACE_INLINE bool zFnGraph::halfEdgeExists(int v1, int v2, int &outHalfEdgeId)
	{
		return graphObj->graph.halfEdgeExists(v1, v2, outHalfEdgeId);
	}

	ZSPACE_INLINE bool zFnGraph::halfEdgeExists(int v1, int v2, zItGraphHalfEdge &outHalfEdge)
	{
		int id;
		bool chk = halfEdgeExists(v1, v2, id);

		if (chk) outHalfEdge = zItGraphHalfEdge(*graphObj, id);

		return chk;
	}


	//--- COMPUTE METHODS 

	ZSPACE_INLINE void zFnGraph::computeEdgeColorfromVertexColor()
	{

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			if (e.isActive())
			{
				int v0 = e.getHalfEdge(0).getVertex().getId();
				int v1 = e.getHalfEdge(1).getVertex().getId();

				zColor col;
				col.r = (graphObj->graph.vertexColors[v0].r + graphObj->graph.vertexColors[v1].r) * 0.5;
				col.g = (graphObj->graph.vertexColors[v0].g + graphObj->graph.vertexColors[v1].g) * 0.5;
				col.b = (graphObj->graph.vertexColors[v0].b + graphObj->graph.vertexColors[v1].b) * 0.5;
				col.a = (graphObj->graph.vertexColors[v0].a + graphObj->graph.vertexColors[v1].a) * 0.5;

				if (graphObj->graph.edgeColors.size() <= e.getId()) graphObj->graph.edgeColors.push_back(col);
				else graphObj->graph.edgeColors[e.getId()] = col;


			}


		}

	}

	ZSPACE_INLINE void zFnGraph::computeVertexColorfromEdgeColor()
	{
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			if (v.isActive())
			{
				vector<int> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zColor col;
				for (int j = 0; j < cEdges.size(); j++)
				{
					col.r += graphObj->graph.edgeColors[cEdges[j]].r;
					col.g += graphObj->graph.edgeColors[cEdges[j]].g;
					col.b += graphObj->graph.edgeColors[cEdges[j]].b;
				}

				col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

				graphObj->graph.vertexColors[v.getId()] = col;

			}
		}
	}

	ZSPACE_INLINE void zFnGraph::averageVertices(int numSteps)
	{
		for (int k = 0; k < numSteps; k++)
		{
			vector<zVector> tempVertPos;

			for (zItGraphVertex v(*graphObj); !v.end(); v++)
			{
				tempVertPos.push_back(graphObj->graph.vertexPositions[v.getId()]);

				if (v.isActive())
				{
					if (!v.checkValency(1))
					{
						vector<int> cVerts;

						v.getConnectedVertices(cVerts);

						for (int j = 0; j < cVerts.size(); j++)
						{
							zVector p = graphObj->graph.vertexPositions[cVerts[j]];
							tempVertPos[v.getId()] += p;
						}

						tempVertPos[v.getId()] /= (cVerts.size() + 1);
					}
				}

			}

			// update position
			for (int i = 0; i < tempVertPos.size(); i++) graphObj->graph.vertexPositions[i] = tempVertPos[i];
		}

	}

	ZSPACE_INLINE void zFnGraph::removeInactiveElements(zHEData type)
	{
		if (type == zVertexData || type == zEdgeData || type == zHalfEdgeData) removeInactive(type);
		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	ZSPACE_INLINE void zFnGraph::makeStatic()
	{
		setStaticContainers();
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zFnGraph::setVertexPositions(zPointArray& pos)
	{
		if (pos.size() != graphObj->graph.vertexPositions.size()) throw std::invalid_argument("size of position contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
		{
			graphObj->graph.vertexPositions[i] = pos[i];
		}
	}

	ZSPACE_INLINE void zFnGraph::setVertexColor(zColor col, bool setEdgeColor)
	{
		graphObj->graph.vertexColors.clear();
		graphObj->graph.vertexColors.assign(graphObj->graph.n_v, col);

		if (setEdgeColor) computeEdgeColorfromVertexColor();
	}

	ZSPACE_INLINE void zFnGraph::setVertexColors(zColorArray& col, bool setEdgeColor)
	{
		if (graphObj->graph.vertexColors.size() != graphObj->graph.vertices.size())
		{
			graphObj->graph.vertexColors.clear();
			for (int i = 0; i < graphObj->graph.vertices.size(); i++) graphObj->graph.vertexColors.push_back(zColor(1, 0, 0, 1));
		}

		if (col.size() != graphObj->graph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < graphObj->graph.vertexColors.size(); i++)
		{
			graphObj->graph.vertexColors[i] = col[i];
		}

		if (setEdgeColor) computeEdgeColorfromVertexColor();
	}

	ZSPACE_INLINE void zFnGraph::setEdgeColor(zColor col, bool setVertexColor)
	{

		graphObj->graph.edgeColors.clear();
		graphObj->graph.edgeColors.assign(graphObj->graph.n_e, col);

		if (setVertexColor) computeVertexColorfromEdgeColor();

	}

	ZSPACE_INLINE void zFnGraph::setEdgeColors(zColorArray& col, bool setVertexColor)
	{
		if (col.size() != graphObj->graph.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph half edges.");

		for (int i = 0; i < graphObj->graph.edgeColors.size(); i++)
		{
			graphObj->graph.edgeColors[i] = col[i];
		}

		if (setVertexColor) computeVertexColorfromEdgeColor();
	}

	ZSPACE_INLINE void zFnGraph::setEdgeWeight(double wt)
	{
		graphObj->graph.edgeWeights.clear();
		graphObj->graph.edgeWeights.assign(graphObj->graph.n_e, wt);

	}

	ZSPACE_INLINE void zFnGraph::setEdgeWeights(zDoubleArray& wt)
	{
		if (wt.size() != graphObj->graph.edgeColors.size()) throw std::invalid_argument("size of wt contatiner is not equal to number of mesh half edges.");

		for (int i = 0; i < graphObj->graph.edgeWeights.size(); i++)
		{
			graphObj->graph.edgeWeights[i] = wt[i];
		}
	}

	//--- GET METHODS 

	ZSPACE_INLINE void zFnGraph::getVertexPositions(zPointArray& pos)
	{
		pos = graphObj->graph.vertexPositions;
	}

	ZSPACE_INLINE zPoint* zFnGraph::getRawVertexPositions()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &graphObj->graph.vertexPositions[0];
	}

	ZSPACE_INLINE void zFnGraph::getVertexColors(zColorArray& col)
	{
		col = graphObj->graph.vertexColors;
	}

	ZSPACE_INLINE zColor* zFnGraph::getRawVertexColors()
	{
		if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

		return &graphObj->graph.vertexColors[0];
	}

	ZSPACE_INLINE void zFnGraph::getEdgeColors(zColorArray& col)
	{
		col = graphObj->graph.edgeColors;
	}

	ZSPACE_INLINE zColor* zFnGraph::getRawEdgeColors()
	{
		if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

		return &graphObj->graph.edgeColors[0];
	}

	ZSPACE_INLINE zPoint zFnGraph::getCenter()
	{
		zPoint out;

		for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
		{
			out += graphObj->graph.vertexPositions[i];
		}

		out /= graphObj->graph.vertexPositions.size();

		return out;

	}

	ZSPACE_INLINE void zFnGraph::getCenters(zHEData type, zPointArray &centers)
	{
		// graph Edge 
		if (type == zHalfEdgeData)
		{

			centers.clear();

			for (zItGraphHalfEdge he(*graphObj); !he.end(); he++)
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

			for (zItGraphEdge e(*graphObj); !e.end(); e++)
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

		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	ZSPACE_INLINE double zFnGraph::getHalfEdgeLengths(zDoubleArray &halfEdgeLengths)
	{
		double total = 0.0;


		halfEdgeLengths.clear();

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
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

	ZSPACE_INLINE double zFnGraph::getEdgeLengths(zDoubleArray &edgeLengths)
	{
		double total = 0.0;


		edgeLengths.clear();

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
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

	ZSPACE_INLINE void zFnGraph::getEdgeData(zIntArray &edgeConnects)
	{
		edgeConnects.clear();

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			edgeConnects.push_back(e.getHalfEdge(0).getVertex().getId());
			edgeConnects.push_back(e.getHalfEdge(1).getVertex().getId());
		}
	}

	ZSPACE_INLINE zObjGraph zFnGraph::getDuplicate(bool planarGraph, zVector graphNormal)
	{
		zObjGraph out;

		if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
		if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);

		vector<zVector> positions;
		vector<int> edgeConnects;


		positions = graphObj->graph.vertexPositions;
		getEdgeData(edgeConnects);


		if (planarGraph)
		{
			graphNormal.normalize();

			zVector x(1, 0, 0);
			zVector sortRef = graphNormal ^ x;

			out.graph.create(positions, edgeConnects, graphNormal, sortRef);
		}
		else out.graph.create(positions, edgeConnects);

		out.graph.vertexColors = graphObj->graph.vertexColors;
		out.graph.edgeColors = graphObj->graph.edgeColors;

		return out;
	}

	ZSPACE_INLINE void zFnGraph::getGraphMesh(zObjMesh &out, double width, zVector graphNormal)
	{

		vector<zVector>positions;
		vector<int> polyConnects;
		vector<int> polyCounts;

		if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
		if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);

		positions = graphObj->graph.vertexPositions;

		vector<vector<int>> edgeVertices;
		for (zItGraphHalfEdge he(*graphObj); !he.end(); he++)
		{

			int v0 = he.getStartVertex().getId();
			int v1 = he.getVertex().getId();

			vector<int> temp;
			temp.push_back(v0);
			temp.push_back(v1);
			temp.push_back(-1);
			temp.push_back(-1);

			edgeVertices.push_back(temp);
		}

		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			vector<zItGraphHalfEdge> cEdges;
			v.getConnectedHalfEdges(cEdges);

			if (cEdges.size() == 1)
			{

				int currentId = cEdges[0].getId();
				int prevId = cEdges[0].getPrev().getId();

				zVector e_current = cEdges[0].getVector();
				e_current.normalize();

				zVector e_prev = cEdges[0].getPrev().getVector();
				e_prev.normalize();

				zVector n_current = graphNormal ^ e_current;
				n_current.normalize();

				zVector n_prev = graphNormal ^ e_prev;
				n_prev.normalize();


				double w = width * 0.5;

				edgeVertices[currentId][3] = positions.size();
				positions.push_back(v.getPosition() + (n_current * w));

				edgeVertices[prevId][2] = positions.size();
				positions.push_back(v.getPosition() + (n_prev * w));

			}

			else
			{
				for (int j = 0; j < cEdges.size(); j++)
				{
					int currentId = cEdges[j].getId();
					int prevId = cEdges[j].getPrev().getId();


					zVector e_current = cEdges[j].getVector();
					e_current.normalize();

					zVector e_prev = cEdges[j].getPrev().getVector();
					e_prev.normalize();

					zVector n_current = graphNormal ^ e_current;
					n_current.normalize();

					zVector n_prev = graphNormal ^ e_prev;
					n_prev.normalize();

					zVector norm = (n_current + n_prev) * 0.5;
					norm.normalize();


					double w = width * 0.5;

					zVector a0 = cEdges[j].getStartVertex().getPosition() + (n_current * w);
					zVector a1 = cEdges[j].getCenter() + (n_current * w);

					zVector b0 = cEdges[j].getStartVertex().getPosition() + (n_prev * w);
					zVector b1 = cEdges[j].getPrev().getCenter() + (n_prev * w);

					double uA, uB;
					bool intersect = graphObj->graph.coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);


					edgeVertices[currentId][3] = positions.size();
					edgeVertices[prevId][2] = positions.size();



					if (!intersect) positions.push_back(v.getPosition() + (norm * w));
					else
					{
						if (uA >= uB)
						{
							zVector dir = a1 - a0;
							double len = dir.length();
							dir.normalize();

							if (uA < 0) dir *= -1;
							positions.push_back(a0 + dir * len * uA);
						}
						else
						{
							zVector dir = b1 - b0;
							double len = dir.length();
							dir.normalize();

							if (uB < 0) dir *= -1;

							positions.push_back(b0 + dir * len * uB);
						}
					}
				}
			}

		}

		for (int i = 0; i < edgeVertices.size(); i++)
		{

			for (int j = 0; j < edgeVertices[i].size(); j++)
			{
				polyConnects.push_back(edgeVertices[i][j]);
			}

			polyCounts.push_back(edgeVertices[i].size());

		}




		// mesh
		if (positions.size() > 0)
		{
			out.mesh.create(positions, polyCounts, polyConnects);
		}



	}

	ZSPACE_INLINE void zFnGraph::getGraphEccentricityCenter(zItGraphVertexArray & outV)
	{
		const int N = numVertices();	// number of nodes in graph
		const int INF = 99999;
		MatrixXi d(N,N);				// distances between nodes
		VectorXi e(N);					// eccentricity of nodes
		set<int> c;						// center of graph
		int rad = INF;					// radius of graph
		int diam = 0;					// diamater of graph

		zIntArray boundaryVerts;
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
			if (v.checkValency(1)) boundaryVerts.push_back(v.getId());
		
		d.setConstant(INF);
		e.setZero();

		for (zItGraphHalfEdge he(*graphObj); !he.end(); he++)
		{
			d(he.getStartVertex().getId(), he.getVertex().getId()) = 1;
			d(he.getStartVertex().getId(), he.getStartVertex().getId()) = 0;
		}

		// Floyd-Warshall's algorithm
		for (int k = 0; k < N; k++)
			for (int j = 0; j < N; j++)
				for (int i = 0; i < N; i++)
					d(i,j) = coreUtils.zMin(d(i, j), d(i, k) + d(k, j));

		// Counting values of eccentricity
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				e(i) = coreUtils.zMax(e(i), d(i,j));

		for (int i = 0; i < N; i++) 
		{
			rad = coreUtils.zMin(rad, e(i));
			diam = coreUtils.zMax(diam, e(i));
		}


		for (int i = 0; i < N; i++) 
			if (e[i] == rad) 		
				c.insert(i);


		zItGraphVertex v(*graphObj);
		int maxScore = 0;

		//printf("\n cen:");
		for (auto id : c)
		{
			zItGraphVertex v1(*graphObj, id);
			outV.push_back(v1);

			double score = 0;

			for (int j = 0; j < boundaryVerts.size(); j++)
				score += d(v1.getId(), boundaryVerts[j]);

			if (score > maxScore)
			{
				maxScore = score;
				v = v1;
			}

			//printf("  %i ", v1.getId());
		}

		
		//outV.push_back(v);
	}

	//---- TOPOLOGY MODIFIER METHODS

	ZSPACE_INLINE zItGraphVertex zFnGraph::splitEdge(zItGraphEdge &edge, double edgeFactor)
	{
		int edgeId = edge.getId();

		zItGraphHalfEdge he = edge.getHalfEdge(0);
		zItGraphHalfEdge heS = edge.getHalfEdge(1);

		zItGraphHalfEdge he_next = he.getNext();
		zItGraphHalfEdge he_prev = he.getPrev();

		zItGraphHalfEdge heS_next = heS.getNext();
		zItGraphHalfEdge heS_prev = heS.getPrev();


		zVector edgeDir = he.getVector();
		double  edgeLength = edgeDir.length();
		edgeDir.normalize();

		zVector newVertPos = he.getStartVertex().getPosition() + edgeDir * edgeFactor * edgeLength;

		int numOriginalVertices = numVertices();

		// check if vertex exists if not add new vertex
		zItGraphVertex newVertex;
		addVertex(newVertPos, false, newVertex);

		if (newVertex.getId() >= numOriginalVertices)
		{
			// remove from halfEdge vertices map
			removeFromHalfEdgesMap(he);

			// add new edges
			int v1 = newVertex.getId();
			int v2 = he.getVertex().getId();

			bool v2_val1 = he.getVertex().checkValency(1);

			zItGraphHalfEdge newHe;
			bool edgesResize = addEdges(v1, v2, false, newHe);

			int newHeId = newHe.getId();

			// recompute iterators if resize is true
			if (edgesResize)
			{
				edge = zItGraphEdge(*graphObj, edgeId);

				he = edge.getHalfEdge(0);
				heS = edge.getHalfEdge(1);

				he_next = he.getNext();
				he_prev = he.getPrev();

				heS_next = heS.getNext();
				heS_prev = heS.getPrev();

				newHe = zItGraphHalfEdge(*graphObj, newHeId);
			}

			zItGraphHalfEdge newHeS = newHe.getSym();

			// update vertex pointers
			newVertex.setHalfEdge(newHe);
			he.getVertex().setHalfEdge(newHeS);

			//// update pointers
			he.setVertex(newVertex);		// current hedge vertex pointer updated to new added vertex

			newHeS.setNext(heS);			// new added symmetry hedge next pointer to point to the symmetry of current hedge
			
			if (!v2_val1) newHeS.setPrev(heS_prev);
			else newHeS.setPrev(newHe);
			
			newHe.setPrev(he);				// new added  hedge prev pointer to point to the current hedge
			if (!v2_val1) newHe.setNext(he_next);

			// update verticesEdge map
			addToHalfEdgesMap(he);
		}

		return newVertex;
	}

	//---- TRANSFORM METHODS OVERRIDES

	ZSPACE_INLINE void zFnGraph::setTransform(zTransform &inTransform, bool decompose, bool updatePositions)
	{
		if (updatePositions)
		{
			zTransformationMatrix to;
			to.setTransform(inTransform, decompose);

			zTransform transMat = graphObj->transformationMatrix.getToMatrix(to);
			transformObject(transMat);

			graphObj->transformationMatrix.setTransform(inTransform);

			// update pivot values of object transformation matrix
			zVector p = graphObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}
		else
		{
			graphObj->transformationMatrix.setTransform(inTransform, decompose);

			zVector p = graphObj->transformationMatrix.getO();
			setPivot(p);

		}

	}

	ZSPACE_INLINE void zFnGraph::setScale(zFloat4 &scale)
	{
		// get  inverse pivot translations
		zTransform invScalemat = graphObj->transformationMatrix.asInverseScaleTransformMatrix();

		// set scale values of object transformation matrix
		graphObj->transformationMatrix.setScale(scale);

		// get new scale transformation matrix
		zTransform scaleMat = graphObj->transformationMatrix.asScaleTransformMatrix();

		// compute total transformation
		zTransform transMat = invScalemat * scaleMat;

		// transform object
		transformObject(transMat);
	}

	ZSPACE_INLINE void zFnGraph::setRotation(zFloat4 &rotation, bool appendRotations)
	{
		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = graphObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = graphObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = graphObj->transformationMatrix;
		to.setRotation(rotation, appendRotations);
		zTransform toMat = graphObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set rotation values of object transformation matrix
		graphObj->transformationMatrix.setRotation(rotation, appendRotations);;
	}

	ZSPACE_INLINE void zFnGraph::setTranslation(zVector &translation, bool appendTranslations)
	{
		// get vector as zDouble3
		zFloat4 t;
		translation.getComponents(t);

		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = graphObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = graphObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = graphObj->transformationMatrix;
		to.setTranslation(t, appendTranslations);
		zTransform toMat = graphObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set translation values of object transformation matrix
		graphObj->transformationMatrix.setTranslation(t, appendTranslations);;

		// update pivot values of object transformation matrix
		zVector p = graphObj->transformationMatrix.getPivot();
		p = p * transMat;
		setPivot(p);

	}

	ZSPACE_INLINE void zFnGraph::setPivot(zVector &pivot)
	{
		// get vector as zDouble3
		zFloat4 p;
		pivot.getComponents(p);

		// set pivot values of object transformation matrix
		graphObj->transformationMatrix.setPivot(p);
	}

	ZSPACE_INLINE void zFnGraph::getTransform(zTransform &transform)
	{
		transform = graphObj->transformationMatrix.asMatrix();
	}

	//---- PROTECTED OVERRIDE METHODS

	ZSPACE_INLINE void zFnGraph::transformObject(zTransform &transform)
	{

		if (numVertices() == 0) return;


		zVector* pos = getRawVertexPositions();

		for (int i = 0; i < numVertices(); i++)
		{

			zVector newPos = pos[i] * transform;
			pos[i] = newPos;
		}

	}

	//---- PROTECTED REMOVE INACTIVE METHODS

	ZSPACE_INLINE void zFnGraph::removeInactive(zHEData type)
	{
		//  Vertex
		if (type == zVertexData)
		{
			zItVertex v = graphObj->graph.vertices.begin();

			while (v != graphObj->graph.vertices.end())
			{
				bool active = v->isActive();

				if (!active)
				{
					graphObj->graph.vertices.erase(v++);

					graphObj->graph.n_v--;
				}
			}

			graphObj->graph.indexElements(zVertexData);

			printf("\n removed inactive vertices. ");

		}

		//  Edge
		else if (type == zEdgeData || type == zHalfEdgeData)
		{

			zItHalfEdge he = graphObj->graph.halfEdges.begin();

			while (he != graphObj->graph.halfEdges.end())
			{
				bool active = he->isActive();

				if (!active)
				{
					graphObj->graph.halfEdges.erase(he++);

					graphObj->graph.n_he--;
				}
			}

			zItEdge e = graphObj->graph.edges.begin();

			while (e != graphObj->graph.edges.end())
			{
				bool active = e->isActive();

				if (!active)
				{
					graphObj->graph.edges.erase(e++);

					graphObj->graph.n_e--;
				}
			}

			printf("\n removed inactive edges. ");

			graphObj->graph.indexElements(zHalfEdgeData);
			graphObj->graph.indexElements(zEdgeData);

		}

		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	//---- PROTECTED FACTORY METHODS

	ZSPACE_INLINE bool zFnGraph::fromTXT(string infilename)
	{
		vector<zVector>positions;
		vector<int>edgeConnects;


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



			vector<string> perlineData = graphObj->graph.coreUtils.splitString(str, " ");

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


				// face
				if (perlineData[0] == "e")
				{

					for (int i = 1; i < perlineData.size(); i++)
					{
						int id = atoi(perlineData[i].c_str()) - 1;
						edgeConnects.push_back(id);
					}
				}
			}
		}

		myfile.close();


		if (!planarGraph) graphObj->graph.create(positions, edgeConnects);;
		if (planarGraph)
		{
			graphNormal.normalize();

			zVector x(1, 0, 0);
			zVector sortRef = graphNormal ^ x;

			graphObj->graph.create(positions, edgeConnects, graphNormal, sortRef);
		}
		printf("\n graphObj->graph: %i %i ", numVertices(), numEdges());

		return true;
	}

	ZSPACE_INLINE bool zFnGraph::fromMAYATXT(string infilename)
	{
		vector<zVector>positions;
		vector<int>edgeConnects;
		vector<int>positionIds;

		int hashCounter = 0;

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

			vector<string> perlineData = graphObj->graph.coreUtils.splitString(str, " ");

			if (perlineData.size() > 0)
			{
				if (perlineData[0] == "}")
				{

					for (int i = 1; i < positionIds.size(); i++)
					{

						edgeConnects.push_back(positionIds[i - 1]);
						edgeConnects.push_back(positionIds[i]);
					}

					positionIds.clear();
					//printf("\n working #");

				}


				if (perlineData[0] == "#")
				{
					if (hashCounter > 0)
					{
						if (positions.size() > 1)
						{
							if (!planarGraph) graphObj->graph.create(positions, edgeConnects);;
							if (planarGraph)
							{
								graphNormal.normalize();

								zVector x(1, 0, 0);
								zVector sortRef = graphNormal ^ x;

								graphObj->graph.create(positions, edgeConnects, graphNormal, sortRef);
							}
							printf("\n graph: %i %i", numVertices(), numEdges());
						}
					}

					hashCounter++;

					positions.clear();
					edgeConnects.clear();
				}

				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						int index = -1;
						bool chk = graphObj->graph.coreUtils.checkRepeatVector(pos, positions, index);

						if (!chk)
						{
							positionIds.push_back(positions.size());
							positions.push_back(pos);
						}
						else positionIds.push_back(index);
					}

				}
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zFnGraph::fromJSON(string infilename)
	{
		json j;
		zUtilsJsonHE graphJSON;
		// read data to json
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
		// read data to json graph
		// Vertices
		graphJSON.vertices.clear();
		graphJSON.vertices = (j["Vertices"].get<vector<int>>());
		// Edges
		graphJSON.halfedges.clear();
		graphJSON.halfedges = (j["Halfedges"].get<vector<vector<int>>>());
		graphObj->graph.edges.clear();
		// update graph
		graphObj->graph.clear();
		graphObj->graph.vertices.assign(graphJSON.vertices.size(), zVertex());
		graphObj->graph.halfEdges.assign(graphJSON.halfedges.size(), zHalfEdge());
		graphObj->graph.edges.assign(floor(graphJSON.halfedges.size() * 0.5), zEdge());
		graphObj->graph.vHandles.assign(graphJSON.vertices.size(), zVertexHandle());
		graphObj->graph.eHandles.assign(floor(graphJSON.halfedges.size() * 0.5), zEdgeHandle());
		graphObj->graph.heHandles.assign(graphJSON.halfedges.size(), zHalfEdgeHandle());
		// set IDs
		for (int i = 0; i < graphJSON.vertices.size(); i++) graphObj->graph.vertices[i].setId(i);
		for (int i = 0; i < graphJSON.halfedges.size(); i++)graphObj->graph.halfEdges[i].setId(i);
		// set Pointers
		int n_v = 0;
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			v.setId(n_v);
			if (graphJSON.vertices[n_v] != -1)
			{
				zItGraphHalfEdge he(*graphObj, graphJSON.vertices[n_v]);
				v.setHalfEdge(he);
				graphObj->graph.vHandles[n_v].id = n_v;
				graphObj->graph.vHandles[n_v].he = graphJSON.vertices[n_v];
			}
			n_v++;
		}
		graphObj->graph.setNumVertices(n_v);
		int n_he = 0;
		int n_e = 0;
		for (zItGraphHalfEdge he(*graphObj); !he.end(); he++)
		{
			// Half Edge
			he.setId(n_he);
			graphObj->graph.heHandles[n_he].id = n_he;
			if (graphJSON.halfedges[n_he][0] != -1)
			{
				zItGraphHalfEdge e(*graphObj, graphJSON.halfedges[n_he][0]);
				he.setPrev(e);
				graphObj->graph.heHandles[n_he].p = graphJSON.halfedges[n_he][0];
			}
			if (graphJSON.halfedges[n_he][1] != -1)
			{
				zItGraphHalfEdge e(*graphObj, graphJSON.halfedges[n_he][1]);
				he.setNext(e);
				graphObj->graph.heHandles[n_he].n = graphJSON.halfedges[n_he][1];
			}
			if (graphJSON.halfedges[n_he][2] != -1)
			{
				zItGraphVertex v(*graphObj, graphJSON.halfedges[n_he][2]);
				he.setVertex(v);
				graphObj->graph.heHandles[n_he].v = graphJSON.halfedges[n_he][2];
			}
			// symmetry half edges
			if (n_he % 2 == 1)
			{
				zItGraphHalfEdge heSym(*graphObj, n_he - 1);
				he.setSym(heSym);
				zItGraphEdge e(*graphObj, n_e);
				e.setId(n_e);
				e.setHalfEdge(heSym, 0);
				e.setHalfEdge(he, 1);
				he.setEdge(e);
				heSym.setEdge(e);
				graphObj->graph.heHandles[n_he].e = n_e;
				graphObj->graph.heHandles[n_he - 1].e = n_e;
				graphObj->graph.eHandles[n_e].id = n_e;
				graphObj->graph.eHandles[n_e].he0 = n_he - 1;
				graphObj->graph.eHandles[n_e].he1 = n_he;
				n_e++;
			}
			n_he++;
		}
		graphObj->graph.setNumEdges(n_e);

		// Vertex Attributes
		graphJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
		//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());
		graphObj->graph.vertexPositions.clear();
		graphObj->graph.vertexColors.clear();
		graphObj->graph.vertexWeights.clear();
		for (int i = 0; i < graphJSON.vertexAttributes.size(); i++)
		{
			for (int k = 0; k < graphJSON.vertexAttributes[i].size(); k++)
			{
				// position
				if (graphJSON.vertexAttributes[i].size() == 3)
				{
					zVector pos(graphJSON.vertexAttributes[i][k], graphJSON.vertexAttributes[i][k + 1], graphJSON.vertexAttributes[i][k + 2]);
					graphObj->graph.vertexPositions.push_back(pos);
					graphObj->graph.vertexColors.push_back(zColor(1,0,0,1));
					graphObj->graph.vertexWeights.push_back(2.0);
					k += 2;
				}
				
				// position and color
				if (graphJSON.vertexAttributes[i].size() == 6)
				{
					zVector pos(graphJSON.vertexAttributes[i][k], graphJSON.vertexAttributes[i][k + 1], graphJSON.vertexAttributes[i][k + 2]);
					graphObj->graph.vertexPositions.push_back(pos);
					zColor col(graphJSON.vertexAttributes[i][k + 3], graphJSON.vertexAttributes[i][k + 4], graphJSON.vertexAttributes[i][k + 5], 1);
					graphObj->graph.vertexColors.push_back(col);
					graphObj->graph.vertexWeights.push_back(2.0);
					k += 5;
				}
			}
		}
		// Edge Attributes
		graphJSON.halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();
		graphObj->graph.edgeColors.clear();
		graphObj->graph.edgeWeights.clear();
		if (graphJSON.halfedgeAttributes.size() == 0)
		{
			for (int i = 0; i < graphObj->graph.n_e; i++)
			{
				graphObj->graph.edgeColors.push_back(zColor());
				graphObj->graph.edgeWeights.push_back(1.0);
			}
		}
		else
		{
			for (int i = 0; i < graphJSON.halfedgeAttributes.size(); i += 2)
			{
				// color
				if (graphJSON.halfedgeAttributes[i].size() == 3)
				{
					zColor col(graphJSON.halfedgeAttributes[i][0], graphJSON.halfedgeAttributes[i][1], graphJSON.halfedgeAttributes[i][2], 1);
					graphObj->graph.edgeColors.push_back(col);
					graphObj->graph.edgeWeights.push_back(1.0);
				}
			}
		}
		printf("\n graph: %i %i ", numVertices(), numEdges());
		// add to maps
		for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
		{
			graphObj->graph.addToPositionMap(graphObj->graph.vertexPositions[i], i);
		}
		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			int v1 = e.getHalfEdge(0).getVertex().getId();
			int v2 = e.getHalfEdge(1).getVertex().getId();
			graphObj->graph.addToHalfEdgesMap(v1, v2, e.getHalfEdge(0).getId());
		}
		return true;

	}

	ZSPACE_INLINE void zFnGraph::toTXT(string outfilename)
	{
		// remove inactive elements
		if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
		if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);


		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		myfile << "\n ";

		// vertex positions
		for (auto &vPos : graphObj->graph.vertexPositions)
		{

			myfile << "\n v " << vPos.x << " " << vPos.y << " " << vPos.z;

		}

		myfile << "\n ";

		// edge connectivity
		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			int v1 = e.getHalfEdge(0).getVertex().getId();
			int v2 = e.getHalfEdge(1).getVertex().getId();

			myfile << "\n e ";

			myfile << v1 << " ";
			myfile << v2;
		}

		myfile << "\n ";

		myfile.close();

		cout << endl << " TXT exported. File:   " << outfilename.c_str() << endl;
	}

	ZSPACE_INLINE void zFnGraph::toJSON(string outfilename)
	{
		// remove inactive elements
		if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
		if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);


		// output file
		zUtilsJsonHE graphJSON;
		json j;

		// create json

		// Vertices
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			if (v.getHalfEdge().isActive()) graphJSON.vertices.push_back(v.getHalfEdge().getId());
			else graphJSON.vertices.push_back(-1);

		}

		//Edges
		for (zItGraphHalfEdge he(*graphObj); !he.end(); he++)
		{
			vector<int> HE_edges;

			if (he.getPrev().isActive()) HE_edges.push_back(he.getPrev().getId());
			else HE_edges.push_back(-1);

			if (he.getNext().isActive()) HE_edges.push_back(he.getNext().getId());
			else HE_edges.push_back(-1);

			if (he.getVertex().isActive()) HE_edges.push_back(he.getVertex().getId());
			else HE_edges.push_back(-1);

			graphJSON.halfedges.push_back(HE_edges);
		}


		// vertex Attributes
		for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
		{
			vector<double> v_attrib;

			v_attrib.push_back(graphObj->graph.vertexPositions[i].x);
			v_attrib.push_back(graphObj->graph.vertexPositions[i].y);
			v_attrib.push_back(graphObj->graph.vertexPositions[i].z);


			v_attrib.push_back(graphObj->graph.vertexColors[i].r);
			v_attrib.push_back(graphObj->graph.vertexColors[i].g);
			v_attrib.push_back(graphObj->graph.vertexColors[i].b);



			graphJSON.vertexAttributes.push_back(v_attrib);
		}


		// Json file 
		j["Vertices"] = graphJSON.vertices;
		j["Halfedges"] = graphJSON.halfedges;
		j["VertexAttributes"] = graphJSON.vertexAttributes;
		j["HalfedgeAttributes"] = graphJSON.halfedgeAttributes;


		// export json

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

	//---- PRIVATE METHODS

	ZSPACE_INLINE void zFnGraph::setStaticContainers()
	{
		graphObj->graph.staticGeometry = true;

		vector<vector<int>> edgeVerts;

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			vector<int> verts;
			e.getVertices(verts);

			edgeVerts.push_back(verts);
		}

		graphObj->graph.setStaticEdgeVertices(edgeVerts);
	}

	//---- PRIVATE DEACTIVATE AND REMOVE METHODS

	ZSPACE_INLINE void zFnGraph::addToHalfEdgesMap(zItGraphHalfEdge &he)
	{
		graphObj->graph.addToHalfEdgesMap(he.getStartVertex().getId(), he.getVertex().getId(), he.getId());
	}

	ZSPACE_INLINE void zFnGraph::removeFromHalfEdgesMap(zItGraphHalfEdge &he)
	{
		graphObj->graph.removeFromHalfEdgesMap(he.getStartVertex().getId(), he.getVertex().getId());
	}
}