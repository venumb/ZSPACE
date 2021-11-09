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


#include<headers/zCore/geometry/zGraph.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zGraph::zGraph()
	{
		n_v = n_he = n_e = 0;
	}
	
	//---- DESTRUCTOR

	ZSPACE_INLINE zGraph::~zGraph(){}

	//---- CREATE METHODS

	ZSPACE_INLINE void zGraph::create(zPointArray(&_positions), zIntArray(&edgeConnects), bool staticGraph, int precision)
	{
		// clear containers
		clear();

		vertices.reserve(_positions.size());
		edges.reserve(floor(edgeConnects.size() * 0.5));
		halfEdges.reserve(edgeConnects.size());


		//// temp containers
		vector<connectedEdgesPerVerts> cEdgesperVert;
		cEdgesperVert.assign(_positions.size(), connectedEdgesPerVerts());

		// create vertices
		for (int i = 0; i < _positions.size(); i++)
		{
			addVertex(_positions[i], precision);
			cEdgesperVert[i].vertId = i;

		}

		// position attribute
		vertexPositions = (_positions);

		// create edges and update connections
		for (int i = 0; i < edgeConnects.size(); i += 2)
		{

			addEdges(edgeConnects[i], edgeConnects[i + 1]);

			cEdgesperVert[edgeConnects[i]].temp_connectedEdges.push_back(n_he - 2);
			cEdgesperVert[edgeConnects[i + 1]].temp_connectedEdges.push_back(n_he - 1);

			vertices[edgeConnects[i]].setHalfEdge(&halfEdges[n_he - 2]);
			vertices[edgeConnects[i + 1]].setHalfEdge(&halfEdges[n_he - 1]);


			vHandles[edgeConnects[i]].he = n_he - 2;
			vHandles[edgeConnects[i + 1]].he = n_he - 1;
		}

		// update pointers
		zVector sortReference(1, 0, 0);
		zVector graphNormal(0, 0, 1);

		for (int i = 0; i < _positions.size(); i++)
		{
			if (cEdgesperVert[i].temp_connectedEdges.size() > 0)
			{

				zVector cen = vertexPositions[i];
				vector<int> sorted_cEdges;				

				//cyclic_sortEdges(cEdgesperVert[i].temp_connectedEdges, cen, cEdgesperVert[i].temp_connectedEdges[0], sorted_cEdges);
				
				cyclic_sortEdges(cEdgesperVert[i].temp_connectedEdges, cen, sortReference, graphNormal, sorted_cEdges);

				if (sorted_cEdges.size() > 0)
				{


					for (int j = 0; j < sorted_cEdges.size(); j++)
					{
						zHalfEdge* e1 = &halfEdges[sorted_cEdges[j]];
						zHalfEdge* e2 = &halfEdges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];

						e1->setPrev(e2->getSym());

						heHandles[e1->getId()].p = e2->getSym()->getId();
						heHandles[e2->getSym()->getId()].n = e1->getId();
					}


				}
			}

		}

	}

	ZSPACE_INLINE void zGraph::create(zPointArray(&_positions), zIntArray(&edgeConnects), zVector &graphNormal, zVector &sortReference, int precision)
	{

		// clear containers
		clear();

		vertices.reserve(_positions.size() + 1);
		edges.reserve(floor(edgeConnects.size() * 0.5) + 1);
		halfEdges.reserve(edgeConnects.size() + 2);

		// temp containers
		connectedEdgesPerVerts *cEdgesperVert = new connectedEdgesPerVerts[_positions.size()];

		// create vertices
		for (int i = 0; i < _positions.size(); i++)
		{
			addVertex(_positions[i], precision);
			cEdgesperVert[i].vertId = i;

		}

		// position attribute
		vertexPositions = (_positions);

		// create edges and update connections

		for (int i = 0; i < edgeConnects.size(); i += 2)
		{

			addEdges(edgeConnects[i], edgeConnects[i + 1]);

			cEdgesperVert[edgeConnects[i]].temp_connectedEdges.push_back(n_he - 2);
			cEdgesperVert[edgeConnects[i + 1]].temp_connectedEdges.push_back(n_he - 1);


			vertices[edgeConnects[i]].setHalfEdge(&halfEdges[n_he - 2]);
			vertices[edgeConnects[i + 1]].setHalfEdge(&halfEdges[n_he - 1]);


			vHandles[edgeConnects[i]].he = n_he - 2;
			vHandles[edgeConnects[i + 1]].he = n_he - 1;
		}

		// update pointers
		for (int i = 0; i < n_v; i++)
		{
			if (cEdgesperVert[i].temp_connectedEdges.size() > 0)
			{
				zVector cen = vertexPositions[i];
				vector<int> sorted_cEdges;
				cyclic_sortEdges(cEdgesperVert[i].temp_connectedEdges, cen, sortReference, graphNormal, sorted_cEdges);

				if (sorted_cEdges.size() > 0)
				{

					for (int j = 0; j < sorted_cEdges.size(); j++)
					{
						zHalfEdge* e1 = &halfEdges[sorted_cEdges[j]];
						zHalfEdge* e2 = &halfEdges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];

						e1->setPrev(e2->getSym());

						heHandles[e1->getId()].p = e2->getSym()->getId();
						heHandles[e2->getSym()->getId()].n = e1->getId();
					}


				}
			}

		}


		delete[] cEdgesperVert;
		cEdgesperVert = NULL;

	}

	ZSPACE_INLINE void zGraph::clear()
	{
		vertices.clear();
		vertexPositions.clear();
		vertexColors.clear();
		vertexWeights.clear();
		positionVertex.clear();

		edges.clear();
		edgeColors.clear();
		edgeWeights.clear();


		halfEdges.clear();
		existingHalfEdges.clear();

		vHandles.clear();
		eHandles.clear();
		heHandles.clear();

		n_v = n_e = n_he = 0;
	}

	//---- VERTEX METHODS

	ZSPACE_INLINE bool zGraph::addVertex(zPoint &pos, int precision)
	{
		bool out = false;

		if (n_v == vertices.capacity())
		{
			if (n_v > 0) resizeArray(zVertexData, n_v * 4);
			else resizeArray(zVertexData, 100);
			out = true;
		}


		addToPositionMap(pos, n_v);

		zItVertex newV = vertices.insert(vertices.end(), zVertex());
		newV->setId(n_v);


		vertexPositions.push_back(pos);
		vHandles.push_back(zVertexHandle());
		vHandles[n_v].id = n_v;

		n_v++;



		// default Attibute values			
		vertexColors.push_back(zColor(1, 0, 0, 1));
		vertexWeights.push_back(2.0);

		return out;
	}

	ZSPACE_INLINE bool zGraph::vertexExists(zPoint pos, int &outVertexId, int precisionfactor)
	{
		bool out = false;;
		outVertexId = -1;

		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


		if (got != positionVertex.end())
		{
			out = true;
			outVertexId = got->second;
		}


		return out;
	}

	ZSPACE_INLINE void zGraph::setNumVertices(int _n_v, bool setMax)
	{
		n_v = _n_v;
	}

	//---- MAP METHODS

	ZSPACE_INLINE void zGraph::addToPositionMap(zPoint &pos, int index, int precisionfactor)
	{
		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		positionVertex[hashKey] = index;
	}

	ZSPACE_INLINE void zGraph::removeFromPositionMap(zPoint &pos, int precisionfactor)
	{
		double factor = pow(10, precisionfactor);
		double x = std::round(pos.x *factor) / factor;
		double y = std::round(pos.y *factor) / factor;
		double z = std::round(pos.z *factor) / factor;

		string removeHashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
		positionVertex.erase(removeHashKey);
	}

	ZSPACE_INLINE void zGraph::addToHalfEdgesMap(int v1, int v2, int index)
	{

		string e1 = (to_string(v1) + "," + to_string(v2));
		existingHalfEdges[e1] = index;


		string e2 = (to_string(v2) + "," + to_string(v1));
		existingHalfEdges[e2] = index + 1;

	}

	ZSPACE_INLINE void zGraph::removeFromHalfEdgesMap(int v1, int v2)
	{

		string e1 = (to_string(v1) + "," + to_string(v2));
		existingHalfEdges.erase(e1);

		string e2 = (to_string(v2) + "," + to_string(v1));
		existingHalfEdges.erase(e2);
	}

	ZSPACE_INLINE bool zGraph::halfEdgeExists(int v1, int v2, int &outEdgeId)
	{

		bool out = false;

		string e1 = (to_string(v1) + "," + to_string(v2));
		std::unordered_map<std::string, int>::const_iterator got = existingHalfEdges.find(e1);


		if (got != existingHalfEdges.end())
		{
			out = true;
			outEdgeId = got->second;
		}
		else outEdgeId = -1;

		return out;
	}

	//---- EDGE METHODS

	ZSPACE_INLINE bool zGraph::addEdges(int &v1, int &v2)
	{

		bool out = false;

		if (n_e == edges.capacity())
		{
			if (n_e > 0) resizeArray(zEdgeData, n_e * 4);
			else	resizeArray(zEdgeData, 100);

			out = true;
		}

		if (n_he >= halfEdges.capacity())
		{
			if (n_he > 0)resizeArray(zHalfEdgeData, n_he * 4);
			else resizeArray(zHalfEdgeData, 200);
			out = true;
		}


		// Handles
		heHandles.push_back(zHalfEdgeHandle());
		heHandles.push_back(zHalfEdgeHandle());
		eHandles.push_back(zEdgeHandle());

		// HALF edge	
		zItHalfEdge newHE1 = halfEdges.insert(halfEdges.end(), zHalfEdge());
		newHE1->setId(n_he);
		newHE1->setVertex(&vertices[v2]);

		heHandles[n_he].id = n_he;
		heHandles[n_he].v = v2;
		heHandles[n_he].e = n_e;

		n_he++;

		// SYMMETRY edge			
		zItHalfEdge newHE2 = halfEdges.insert(halfEdges.end(), zHalfEdge());
		newHE2->setId(n_he);
		newHE2->setVertex(&vertices[v1]);

		heHandles[n_he].id = n_he;
		heHandles[n_he].v = v1;
		heHandles[n_he].e = n_e;

		n_he++;

		// set symmetry pointers 
		newHE2->setSym(&halfEdges[n_he - 2]);
		addToHalfEdgesMap(v1, v2, newHE1->getId());

		//EDGE
		zItEdge newE = edges.insert(edges.end(), zEdge());
		newE->setId(n_e);
		newE->setHalfEdge(&halfEdges[n_he - 2], 0);
		newE->setHalfEdge(&halfEdges[n_he - 1], 1);;


		newHE1->setEdge(&edges[n_e]);
		newHE2->setEdge(&edges[n_e]);

		eHandles[n_e].id = n_e;
		eHandles[n_e].he0 = n_he - 2;
		eHandles[n_e].he1 = n_he - 1;

		n_e++;


		// default color and weights
		edgeColors.push_back(zColor(0, 0, 0, 0));
		edgeWeights.push_back(1.0);

		return out;
	}

	ZSPACE_INLINE void zGraph::setNumEdges(int _n_e, bool setMax)
	{
		n_e = _n_e;
		n_he = _n_e * 2;
	}

	ZSPACE_INLINE void zGraph::setStaticEdgeVertices(vector<vector<int>> &_edgeVertices)
	{
		if (!staticGeometry) 	throw std::invalid_argument(" error: mesh not static");
		edgeVertices = _edgeVertices;
	}

	ZSPACE_INLINE void zGraph::cyclic_sortEdges(zIntArray &unSortedEdges, zVector &center, int sortReferenceIndex, zIntArray &sortedEdges)
	{

		vector<int> out;

		vector<double> angles;
		map< double, int > angle_e_Map;

		// find best fit plane
		vector<zVector> points;

		for (int i = 0; i < unSortedEdges.size(); i++)
		{

			zHalfEdge *e = &halfEdges[unSortedEdges[i]];
			points.push_back(vertexPositions[e->getVertex()->getId()]);
		}



		zTransform bestPlane = coreUtils.getBestFitPlane(points);
		zVector norm(bestPlane(0, 2), bestPlane(1, 2), bestPlane(2, 2));

		// iterate through edges in list, get angle to horz, sort;

		zVector horz(bestPlane(0, 0), bestPlane(1, 0), bestPlane(2, 0));// = coreUtils.fromMatrixColumn(bestPlane, 0);;

    zVector upVec(bestPlane(0, 2), bestPlane(1, 2), bestPlane(2, 2));// = coreUtils.fromMatrixColumn(bestPlane, 2);; 

		zVector cross = upVec ^ horz;

		if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size())
		{
			zHalfEdge *e = &halfEdges[unSortedEdges[sortReferenceIndex]];

			horz = zVector(vertexPositions[e->getVertex()->getId()] - center);
		}


		angles.clear();

		for (int i = 0; i < unSortedEdges.size(); i++)
		{
			float angle = 0;

			if (i != sortReferenceIndex)
			{
				zHalfEdge *e = &halfEdges[unSortedEdges[i]];

				zVector vec1(vertexPositions[e->getVertex()->getId()] - center);


				double ang = cross.angle(vec1);

				angle = horz.angle(vec1);
				if (ang > 90) angle = 360 - angle;

				//printf("\n cen: %i vert1 : %i vert2: %i andle: %1.2f angle360: %1.2f ", center.vertexId, unSortedEdges[sortReferenceIndex].v->vertexId, unSortedEdges[i].v->vertexId, angle, horz.Angle360(vec1));

			}

			//if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size()) printf("\n sortReferenceIndex : %i currentedge: %i angle : %1.2f id: %i ", unSortedEdges[sortReferenceIndex].edgeId , unSortedEdges[i].edgeId, angle, i);

			// check if same key exists
			for (int k = 0; k < angles.size(); k++)
			{
				if (angles[k] == angle) angle += 0.01;
			}

			angle_e_Map[angle] = i;
			angles.push_back(angle);
		}

		sort(angles.begin(), angles.end());

		for (int i = 0; i < angles.size(); i++)
		{
			int id = angle_e_Map.find(angles[i])->second;
			if (id > unSortedEdges.size())
			{
				id = 0;
			}

			out.push_back((unSortedEdges[id]));

		}



		sortedEdges = out;

	}

	ZSPACE_INLINE void zGraph::cyclic_sortEdges(zIntArray &unSortedEdges, zVector &center, zVector& referenceDir, zVector& norm, zIntArray &sortedEdges)
	{


		vector<int> out;

		vector<double> angles;
		map< double, int > angle_e_Map;

		// find best fit plane
		vector<zVector> points;

		for (int i = 0; i < unSortedEdges.size(); i++)
		{

			zHalfEdge *e = &halfEdges[unSortedEdges[i]];

			points.push_back(vertexPositions[e->getVertex()->getId()]);
		}




		// iterate through edges in list, get angle to horz, sort;

		zVector horz = referenceDir;;
		zVector upVec = norm;;

		zVector cross = upVec ^ horz;



		angles.clear();

		for (int i = 0; i < unSortedEdges.size(); i++)
		{
			float angle = 0;

			zHalfEdge *e = &halfEdges[unSortedEdges[i]];

			zVector vec1(vertexPositions[e->getVertex()->getId()] - center);
			angle = horz.angle360(vec1, upVec);



			// check if same key exists
			for (int k = 0; k < angles.size(); k++)
			{
				if (angles[k] == angle) angle += 0.01;
			}

			angle_e_Map[angle] = i;
			angles.push_back(angle);
		}

		sort(angles.begin(), angles.end());

		for (int i = 0; i < angles.size(); i++)
		{


			int id = angle_e_Map.find(angles[i])->second;
			if (id > unSortedEdges.size())
			{
				id = 0;
			}
			out.push_back((unSortedEdges[id]));


		}



		sortedEdges = out;

	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zGraph::indexElements(zHEData type)
	{
		if (type == zVertexData)
		{
			int n_v = 0;
			for (auto &v : vertices)
			{
				v.setId(n_v);
				n_v++;
			}
		}

		else if (type == zEdgeData)
		{
			int n_e = 0;
			for (auto &e : edges)
			{
				e.setId(n_e);
				n_e++;
			}
		}

		else if (type == zHalfEdgeData)
		{
			int n_he = 0;
			for (auto &he : halfEdges)
			{
				he.setId(n_he);
				n_he++;
			}
		}
		else throw std::invalid_argument(" error: invalid zHEData type");
	}

	ZSPACE_INLINE void zGraph::resizeArray(zHEData type, int newSize)
	{
		//  Vertex
		if (type == zVertexData)
		{
			vertices.clear();
			vertices.reserve(newSize);

			// reassign pointers
			int n_v = 0;
			for (auto &v : vHandles)
			{
				zItVertex newV = vertices.insert(vertices.end(), zVertex());
				newV->setId(n_v);
				if (v.he != -1)newV->setHalfEdge(&halfEdges[v.he]);

				n_v++;
			}

			for (int i = 0; i < heHandles.size(); i++)
			{
				if (heHandles[i].v != -1) halfEdges[i].setVertex(&vertices[heHandles[i].v]);
			}

			printf("\n graph vertices resized. ");

		}

		//  Edge
		else if (type == zHalfEdgeData)
		{

			halfEdges.clear();
			halfEdges.reserve(newSize);

			halfEdges.assign(heHandles.size(), zHalfEdge());

			// reassign pointers
			int n_he = 0;
			for (auto &he : heHandles)
			{
				halfEdges[n_he].setId(n_he);

				int sym = (n_he % 2 == 0) ? n_he + 1 : n_he - 1;

				halfEdges[n_he].setSym(&halfEdges[sym]);
				if (he.n != -1) halfEdges[n_he].setNext(&halfEdges[he.n]);
				if (he.p != -1) halfEdges[n_he].setPrev(&halfEdges[he.p]);

				if (he.v != -1) halfEdges[n_he].setVertex(&vertices[he.v]);
				if (he.e != -1) halfEdges[n_he].setEdge(&edges[he.e]);


				n_he++;
			}

			for (int i = 0; i < vHandles.size(); i++)
			{
				if (vHandles[i].he != -1) vertices[i].setHalfEdge(&halfEdges[vHandles[i].he]);
			}

			for (int i = 0; i < eHandles.size(); i++)
			{
				if (eHandles[i].he0 != -1) edges[i].setHalfEdge(&halfEdges[eHandles[i].he0], 0);
				if (eHandles[i].he1 != -1) edges[i].setHalfEdge(&halfEdges[eHandles[i].he1], 1);
			}



			printf("\n graph half edges resized. ");

		}

		else if (type == zEdgeData)
		{

			edges.clear();
			edges.reserve(newSize);

			// reassign pointers
			int n_e = 0;
			for (auto &e : eHandles)
			{
				zItEdge newE = edges.insert(edges.end(), zEdge());
				newE->setId(n_e);

				if (e.he0 != -1)newE->setHalfEdge(&halfEdges[e.he0], 0);
				if (e.he1 != -1)newE->setHalfEdge(&halfEdges[e.he1], 1);

				n_e++;

			}

			for (int i = 0; i < heHandles.size(); i++)
			{
				if (heHandles[i].e != -1) halfEdges[i].setEdge(&edges[heHandles[i].e]);
			}



			printf("\n graph edges resized. ");

		}



		else throw std::invalid_argument(" error: invalid zHEData type");
	}

}