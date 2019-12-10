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


#include<headers/zToolsets/geometry/zTsGraphPolyhedra.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsGraphPolyhedra::zTsGraphPolyhedra() {}

	ZSPACE_INLINE zTsGraphPolyhedra::zTsGraphPolyhedra(zObjGraph &_graphObj, zModel &_model)
	{
		graphObj = &_graphObj;
		fnGraph = zFnGraph(_graphObj);

		model = &_model;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsGraphPolyhedra::~zTsGraphPolyhedra() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::createGraphFromFile(string &_path, zFileTpye _type, bool _staticGeom)
	{
		// create graph
		fnGraph.from(_path, _type, _staticGeom);

		// allocate memory
		conHullCol.assign(fnGraph.numVertices(), zObjMesh());
		dualMeshCol.assign(fnGraph.numVertices(), zObjMesh());
		c_graphEdge_dualCellFace.assign(fnGraph.numEdges(), zIntPairArray());

		// color vertices
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
			if (v.getValence() > 1) v.setColor(zColor(0.0, 1.0, 0.0, 1.0));

		// sort vertices
		sortGraphVertices(sortedGraphVertices);

		// add to model
		model->addObject(*graphObj);
		graphObj->setDisplayElements(true, true);

		for (auto &obj : conHullCol) model->addObject(obj);
		for (auto &obj : dualMeshCol) model->addObject(obj);
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::createGraphFromMesh(zObjMesh &_inMeshObj, zVector &_verticalForce)
	{
		zFnMesh fnMesh(_inMeshObj);

		// allocate memory
		conHullCol.assign(fnMesh.numVertices(), zObjMesh());
		dualMeshCol.assign(fnMesh.numVertices(), zObjMesh());
		c_graphEdge_dualCellFace.assign(fnMesh.numEdges(), zIntPairArray());

		vector<int>edgeConnects;
		vector<zVector> vertexPositions;

		fnMesh.getEdgeData(edgeConnects);
		fnMesh.getVertexPositions(vertexPositions);

		for (zItMeshVertex v(_inMeshObj); !v.end(); v++)
		{
			zPoint newPos = v.getPosition() + (_verticalForce * -1);
			
			edgeConnects.push_back(v.getId());
			edgeConnects.push_back(vertexPositions.size());

			vertexPositions.push_back(newPos);
		}

		fnGraph.create(vertexPositions, edgeConnects, true);

		// color vertices
		for (zItGraphVertex v(*graphObj); !v.end(); v++)
			if (v.getValence() > 1) v.setColor(zColor(0.0, 1.0, 0.0, 1.0));

		// sort vertices
		sortGraphVertices(sortedGraphVertices);

		// add graph to model
		model->addObject(*graphObj);
		graphObj->setDisplayElements(true, true);

	}
	
	ZSPACE_INLINE void zTsGraphPolyhedra::create()
	{
		for (auto g_v : sortedGraphVertices)
		{		
			if (!g_v.checkValency(1))
			{
				nodeId.push_back(g_v.getId());

				//cout << "\n\t\t\tNODE_ID: " << g_v.getId();

				// make clean convex hull
				zPointArray hullPts;
				cleanConvexHull(g_v, 4, hullPts);

				// dual mesh from convex hull
				createDualMesh(g_v);
								
				//cout << "\n////////////////////////////////////////////////////////////////////";
			}
		}
		// set edge and face centers
		for (auto &m : conHullCol)
		{
			zPointArray edgeCenters, faceCenters;

			for (zItMeshEdge e(m); !e.end(); e++)
				edgeCenters.push_back(e.getCenter());

			for (zItMeshFace f(m); !f.end(); f++)
				faceCenters.push_back(f.getCenter());

			m.setEdgeCenters(edgeCenters);
			m.setFaceCenters(faceCenters);
		}

		for (auto &m : dualMeshCol)
		{
			zPointArray edgeCenters, faceCenters;

			for (zItMeshEdge e(m); !e.end(); e++)
				edgeCenters.push_back(e.getCenter());

			for (zItMeshFace f(m); !f.end(); f++)
				faceCenters.push_back(f.getCenter());

			m.setEdgeCenters(edgeCenters);
			m.setFaceCenters(faceCenters);
		}

		zPointArray graphEdgeCenters;
		for (zItGraphEdge e(*graphObj); !e.end(); e++) graphEdgeCenters.push_back(e.getCenter());
		graphObj->setEdgeCenters(graphEdgeCenters);

		// color connected faces
		colorDualFaceConnectivity();
		
		// snap and merge dual cells
		zItGraphVertexArray(graphCenters);
		fnGraph.getGraphEccentricityCenter(graphCenters);
		snapDualCells(sortedGraphVertices, graphCenters);
	}

	//---- DRAW METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::setDisplayGraphElements(bool _drawGraph, bool _drawVertIds, bool _drawEdgeIds)
	{
		graphObj->setDisplayObject(_drawGraph);
		graphObj->setDisplayElementIds(_drawVertIds, _drawEdgeIds);
	}
	
	ZSPACE_INLINE void zTsGraphPolyhedra::setDisplayHullElements(bool _drawConvexHulls, bool _drawFaces, bool _drawVertexIds, bool _drawEdgeIds, bool _drawFaceIds)
	{
		for (auto &m : conHullCol)
		{
			m.setDisplayFaces(_drawFaces);
			m.setDisplayElementIds(_drawVertexIds, _drawEdgeIds, _drawFaceIds);
			m.setDisplayObject(_drawConvexHulls);
		}		
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::setDisplayPolyhedraElements(bool _drawDualMesh, bool _drawFaces, bool _drawVertexIds, bool _drawEdgeIds, bool _drawFaceIds)
	{	
		for (auto &m : dualMeshCol)
		{
			m.setDisplayFaces(_drawFaces);
			m.setDisplayElementIds(_drawVertexIds, _drawEdgeIds, _drawFaceIds);
			m.setDisplayObject(_drawDualMesh);
		}
	}

	//---- PRIVATE CREATE METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::createDualMesh(zItGraphVertex &_graphVertex)
	{
		zFnMesh fnMesh(conHullCol[_graphVertex.getId()]);
		zIntArray inEdge_dualEdge;
		zIntArray dualEdge_inEdge;
		fnMesh.getDualMesh(dualMeshCol[_graphVertex.getId()], inEdge_dualEdge, dualEdge_inEdge, true);

		// get connected edges
		zItGraphHalfEdgeArray connectedHalfeEdges;
		_graphVertex.getConnectedHalfEdges(connectedHalfeEdges);

		zFnMesh fnConHull(conHullCol[_graphVertex.getId()]);

		// map positionId -> graphVertexId
		zPoint* hullPos = fnConHull.getRawVertexPositions();

		for (auto he : connectedHalfeEdges)
		{
			zPoint hullP = (he.getVertex().checkValency(1)) ? he.getVertex().getPosition() : he.getCenter();

			for (int i = 0; i < fnConHull.numVertices(); i++)
			{
				if (hullP.distanceTo(hullPos[i]) < 0.001)
				{
					c_graphEdge_dualCellFace[he.getEdge().getId()].push_back(make_pair(_graphVertex.getId(), i));
				}
			}
		}
	}
	
	//---- PRIVATE UTILITY METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::sortGraphVertices(zItGraphVertexArray &_graphVertices)
	{
		// get eccentricity center
		zItGraphVertexArray outV;
		fnGraph.getGraphEccentricityCenter(outV);

		zItGraphVertex bsf(*graphObj);
		if (outV.size() > 0) bsf = outV[0];

		// breadth search first sorting
		bsf.getBSF(_graphVertices);
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::cleanConvexHull(zItGraphVertex &_vIt, int _maxPolygons, zPointArray &_hullPts)
	{
		// get connected edges
		zItGraphHalfEdgeArray connectedHalfeEdges;
		_vIt.getConnectedHalfEdges(connectedHalfeEdges);

		for (auto he : connectedHalfeEdges)
		{
			if (he.getVertex().checkValency(1))	_hullPts.push_back(he.getVertex().getPosition());
			else _hullPts.push_back((_vIt.getPosition() + he.getVertex().getPosition()) / 2);
		}

		zFnMesh fnMesh(conHullCol[_vIt.getId()]);
		fnMesh.makeConvexHull(_hullPts);

		if (fnMesh.numPolygons() > _maxPolygons) // FIX has to work for ngons
		{
			zDoubleArray hedralAngles;
			fnMesh.getEdgeDihedralAngles(hedralAngles);

			vector<std::pair<double, int>> tmpPair;

			for (int i = 0; i < hedralAngles.size(); i++)
				tmpPair.push_back(make_pair(hedralAngles[i], i));

			std::sort(tmpPair.begin(), tmpPair.end());

			zItMeshEdge e(conHullCol[_vIt.getId()], tmpPair[0].second);

			fnMesh.deleteEdge(e);
		}
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::colorDualFaceConnectivity()
	{
		for (int i = 0; i < c_graphEdge_dualCellFace.size(); i++)
		{
			
			for (int j = 0; j < c_graphEdge_dualCellFace[i].size(); j++)
			{
				int cellId = c_graphEdge_dualCellFace[i][j].first;
				int faceId = c_graphEdge_dualCellFace[i][j].second;

				if (c_graphEdge_dualCellFace[i].size() == 2)
				{
					zItMeshFace f(dualMeshCol[cellId], faceId);
					f.setColor(zColor(1, 0, 0, 1));
				}
			}
		}
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::snapDualCells(zItGraphVertexArray &_bsf, zItGraphVertexArray &_gCenters)
	{
		
		map<zIntPair, int> volVert_globalId;
		int globalId_counter = 0;

		map<zIntPair, zIntPair> volEdge_mappedVolEdge;

		vector<zIntArray> mapped_volEdges;

		zBoolArray edgeVisited;
		edgeVisited.assign(fnGraph.numEdges(), false);


		for (auto g_v : _bsf)
		{			
			// get vertices > valence 1
			if (g_v.getValence() > 1)
			{
				zItGraphHalfEdgeArray conHEdges;
				g_v.getConnectedHalfEdges(conHEdges);

				//get conneted edges
				for (auto he : conHEdges)
				{
					int graphEdgeId = he.getEdge().getId();

					if (edgeVisited[graphEdgeId]) continue;

					if (c_graphEdge_dualCellFace[graphEdgeId].size() != 2) continue;

					printf("\n vol face: ");
					for (int j = 0; j < c_graphEdge_dualCellFace[graphEdgeId].size(); j++)
					{
						printf(" %i ", c_graphEdge_dualCellFace[graphEdgeId][j]);
					}

					zItMeshFace f1(dualMeshCol[c_graphEdge_dualCellFace[graphEdgeId][0].first], c_graphEdge_dualCellFace[graphEdgeId][0].second);

					zPointArray f1VertPos;
					f1.getVertexPositions(f1VertPos);

					zIntArray f1Verts;
					f1.getVertices(f1Verts);

					//zVector ref = f1VertPos[0] - f1.getCenter();

					//printf("\nvolume: %i face: %i", c_graphEdge_dualCellFace[graphEdgeId][0], f1.getId());
					//for (auto fV : f1Verts) printf(" %i ", fV);

					zItMeshFace f2(dualMeshCol[c_graphEdge_dualCellFace[graphEdgeId][1].first], c_graphEdge_dualCellFace[graphEdgeId][1].second);

					zPointArray f2VertPos;
					f2.getVertexPositions(f2VertPos);

					zIntArray f2Verts;
					f2.getVertices(f2Verts);

					//printf("\nvolume: %i face: %i", c_graphEdge_dualCellFace[graphEdgeId][2], f2.getId());
					//for (auto fV : f2Verts) printf(" %i ", fV);

					// best fit plane
					zItMeshHalfEdge f1Edge = f1.getHalfEdge();
					zItMeshHalfEdge f2Edge = f2.getHalfEdge();

					zItMeshHalfEdge start = f2Edge;
					bool exit = false;

					do
					{
						zVectorArray pts;
						pts.push_back(f1Edge.getStartVertex().getPosition());
						pts.push_back(f1Edge.getVertex().getPosition());

						pts.push_back(f2Edge.getStartVertex().getPosition());
						pts.push_back(f2Edge.getVertex().getPosition());

						// get projected points in best fit plane 
						zPointArray projectedPts;
						coreUtils.getProjectedPoints_BestFitPlane(pts, projectedPts);

						// check angle
						zIntArray verts = { 0, 1, 3, 2 };
						double totalAngle = 0.0;

						for (int i = 0; i < verts.size(); i++)
						{
							zVector a = projectedPts[(i + 1) % verts.size()] - projectedPts[i];
							zVector b = projectedPts[(i - 1 + verts.size()) % verts.size()] - projectedPts[i];

							totalAngle += a.angle(b);
						}
						//cout<< "\n totalAngle =" <<  totalAngle;


						// check condition
						if (abs(totalAngle - 360) < EPS) exit = true;
						if (!exit) f2Edge = f2Edge.getNext();
						if (exit)
						{
							//printf("\n%i maps to %i - %i maps to %i\n", f1Edge.getStartVertex().getId(), f2Edge.getVertex().getId(), f1Edge.getVertex().getId(), f2Edge.getStartVertex().getId());

							zIntArray tmpArray = { c_graphEdge_dualCellFace[graphEdgeId][0].first, f1Edge.getId(), c_graphEdge_dualCellFace[graphEdgeId][1].first, f2Edge.getId() };

							mapped_volEdges.push_back(tmpArray);
							
							zIntPair tmpp1(c_graphEdge_dualCellFace[graphEdgeId][0].first, f1Edge.getId());
							zIntPair tmpp2(c_graphEdge_dualCellFace[graphEdgeId][1].first, f2Edge.getId());
							
							volEdge_mappedVolEdge[tmpp1] = tmpp2;
						}

					} while (f2Edge != start && !exit);

					edgeVisited[graphEdgeId] = true;
				}
			}
		}

		vector<zIntArray> nodeIdVertId_1;
		vector<zIntArray> nodeIdVertId_2;

		zBoolArray eVisited;
		eVisited.assign(fnGraph.numEdges(), false);

		//for (auto g : gCenters) vVisited[g.getId()] = true;

		//for (zItGraphVertex v(*graphObj); !v.end(); v++) if (v.checkValency(1)) vVisited[v.getId()] = true;
		
			

		for (int j = 0; j < _gCenters.size(); j++)
		{
			int vol1 = _gCenters[j].getId();

			zItGraphHalfEdgeArray conHEdges;
			_gCenters[j].getConnectedHalfEdges(conHEdges);

			queue<zItGraphHalfEdge> q;
			for (auto &he : conHEdges) q.push(he);
						

			while (!q.empty())
			{
				auto he = q.front();
				q.pop();
				
				if (eVisited[he.getEdge().getId()]) continue;
				
				zItGraphHalfEdgeArray tConHEdges;
				he.getVertex().getConnectedHalfEdges(tConHEdges);
				for (auto &tHe : tConHEdges) q.push(tHe);
				
					
				eVisited[he.getEdge().getId()] = true;

				printf("\n he: %i %i", he.getStartVertex().getId(), he.getVertex().getId());

				int graphEdgeId = he.getEdge().getId();
				
				if (c_graphEdge_dualCellFace[graphEdgeId].size() != 2)	continue;

				int vol1 = c_graphEdge_dualCellFace[graphEdgeId][0].first;
				int vol2 = c_graphEdge_dualCellFace[graphEdgeId][1].first;

				zItMeshFace f1(dualMeshCol[vol1], c_graphEdge_dualCellFace[graphEdgeId][0].second);
				zIntArray f1Verts;
				f1.getVertices(f1Verts);

					
				for (auto id : f1Verts)
				{
					bool chk = false;

					map<zIntPair, int>::const_iterator it = volVert_globalId.find(make_pair(vol1, id));
					if (it != volVert_globalId.end()) chk = true;

					if (!chk)
					{
						zIntPair tmp(vol1, id);
						volVert_globalId[tmp] = globalId_counter;;
						printf("\n %i %i added : %i ", vol1, id, globalId_counter);

						globalId_counter++;

						
					}
				}
					
				

				zItMeshFace f2(dualMeshCol[vol2], c_graphEdge_dualCellFace[graphEdgeId][1].second);
				zIntArray f2Verts;
				f2.getVertices(f2Verts);				

				for (zItMeshVertex v(dualMeshCol[vol2]); !v.end(); v++)
				{
					bool chk = false;
					for (auto id : f2Verts)
					{
						if (id == v.getId()) chk = true;
					}

					if (!chk)
					{
						map<zIntPair, int>::const_iterator it = volVert_globalId.find(make_pair(vol2, v.getId()));
						if (it != volVert_globalId.end()) chk = true;
					}

					if (!chk)
					{
						zIntPair tmp(vol2, v.getId());
						volVert_globalId[tmp] = globalId_counter;;
						printf("\n %i %i added : %i ", vol2, v.getId(), globalId_counter);

						globalId_counter++;
					}
				}

				zItMeshHalfEdge f1Edge = f1.getHalfEdge();

				map<zIntPair, zIntPair>::const_iterator it_pair = volEdge_mappedVolEdge.find(make_pair(vol1, f1Edge.getId()));

				zItMeshHalfEdge f2Edge(dualMeshCol[vol2], it_pair->second.second);

				for (int i = 0; i < f1.getNumVertices(); i++)
				{
					map<zIntPair, int>::const_iterator it = volVert_globalId.find(make_pair(vol1, f1Edge.getVertex().getId()));

					if (it != volVert_globalId.end())
					{
						zIntPair tmpPair2(vol2, f2Edge.getStartVertex().getId());
						volVert_globalId[tmpPair2] = it->second;

						printf("\n %i %i added : %i ", vol2, f2Edge.getStartVertex().getId(), it->second);
					}

					f1Edge = f1Edge.getNext();
					f2Edge = f2Edge.getPrev();					
				}		
			}
		}


		// snap vertices
		for (int j = 0; j < mapped_volEdges.size(); j++)
		{
			int vol1 = mapped_volEdges[j][0];
			int vol2 = mapped_volEdges[j][2];


			zItMeshHalfEdge f1Edge(dualMeshCol[vol1], mapped_volEdges[j][1]);
			zItMeshHalfEdge f2Edge(dualMeshCol[vol2], mapped_volEdges[j][3]);
			
			zItMeshFace f1 = f1Edge.getFace();
			zItMeshFace f2 = f2Edge.getFace();
			zIntArray f2Vert;
			f2.getVertices(f2Vert);
			
			zFnMesh tempFn_Vol2(dualMeshCol[vol2]);
			zPoint* pos2 = tempFn_Vol2.getRawVertexPositions();

			zFnMesh tempFn_Vol1(dualMeshCol[vol1]);
			zPoint* pos1 = tempFn_Vol1.getRawVertexPositions();

			/*if (j == 0)
			{
				for (zItMeshVertex v(dualMeshCol[vol1]); !v.end(); v++)
				{
					zIntPair tmp(vol1, v.getId());
					volVert_globalId[tmp] = globalId_counter;;
					globalId_counter++;
				}
			}
			
			for (zItMeshVertex v(dualMeshCol[vol2]); !v.end(); v++)
			{
				bool chk = false;
				for (auto id : f2Vert)
				{
					if (id == v.getId()) chk = true;
				}

				if (!chk)
				{
					map<zIntPair, int>::const_iterator it = volVert_globalId.find(make_pair(vol2, v.getId()));
					if (it != volVert_globalId.end()) chk = true;
				}

				if (!chk)
				{
					zIntPair tmp(vol2, v.getId());
					volVert_globalId[tmp] = globalId_counter;;
					globalId_counter++;
				}
			}*/
			

			// snap corresponding vertices
			for (int i = 0; i <f1.getNumVertices(); i++)
			{
				if (vol1 < snapSteps) pos2[f2Edge.getVertex().getId()] = pos1[f1Edge.getStartVertex().getId()];

				f1Edge = f1Edge.getNext();
				f2Edge = f2Edge.getPrev();


				// connectivity lines for display				
				dualConnectivityLines.push_back(make_pair(f1Edge.getStartVertex().getPosition(), f2Edge.getVertex().getPosition()));


				//printf("\nvol: %i vId: %i  ->  vol: %i vId: %i", vol1, f1Edge.getVertex().getId(), vol2, f2Edge.getStartVertex().getId());

				//zIntPair tmpPair1(vol1, f1Edge.getVertex().getId());
				/*map<zIntPair, int>::const_iterator it = volVert_globalId.find(make_pair(vol1, f1Edge.getVertex().getId()));

				if (it != volVert_globalId.end())
				{
					zIntPair tmpPair2(vol2, f2Edge.getStartVertex().getId());
					volVert_globalId[tmpPair2] = it->second;
				}*/

				zIntArray tmp1 = { vol1, f1Edge.getVertex().getId() };
				nodeIdVertId_1.push_back(tmp1);


				zIntArray tmp2 = { vol2, f2Edge.getStartVertex().getId() };
				nodeIdVertId_2.push_back(tmp2);
			}
		}

		cout << "\n dualConnectivityLines size: " << dualConnectivityLines.size();

		vector<zIntPairArray> globalId_volVert;
		globalId_volVert.assign(globalId_counter, zIntPairArray());
		for (auto p : volVert_globalId)
		{
			globalId_volVert[p.second].push_back(p.first);
		}

		int tmpcounter = 0;
		for (auto globalId : globalId_volVert)
		{
			printf("\n %i: ", tmpcounter);

			for (auto pair : globalId) printf(" %i %i | ", pair.first, pair.second);

			tmpcounter++;
		}

		// merge vertices to center

		for (int i = 0; i < nodeIdVertId_1.size(); i++)
		{
			//nodeVert_1[i][0]; // vol1
			//nodeVert_1[i][1]; // vId
			for (int j = 0; j < nodeIdVertId_1.size(); j++)
			{
				if (nodeIdVertId_1[i][0] == nodeIdVertId_1[j][0] && nodeIdVertId_1[i][1] == nodeIdVertId_1[j][1] && i != j)
				{
					/*cout << "\n\n i " << i;
					cout << "\n j " << j;
					
					cout << "\n " << nodeIdVertId_1[i][0] << " " << nodeIdVertId_1[i][1] << "  ->  " << nodeIdVertId_2[i][0] << " " << nodeIdVertId_2[i][1];

					cout << "\n " << nodeIdVertId_1[j][0] << " " << nodeIdVertId_1[j][1] << "  ->  " << nodeIdVertId_2[j][0] << " " << nodeIdVertId_2[j][1];
*/




					/*zItMeshVertex v1_1(dualMeshCol[nodeIdVertId_1[i][0]], nodeIdVertId_1[i][1]);
					zItMeshVertex v2_1(dualMeshCol[nodeIdVertId_2[i][0]], nodeIdVertId_2[i][1]);

					zItMeshVertex v1_2(dualMeshCol[nodeIdVertId_1[j][0]], nodeIdVertId_1[j][1]);
					zItMeshVertex v2_2(dualMeshCol[nodeIdVertId_2[j][0]], nodeIdVertId_2[j][1]);	*/

					/*zVector newPos = (v1_1.getPosition() + v2_1.getPosition() + v1_2.getPosition() + v2_2.getPosition()) / 4;

					v1_1.setPosition(newPos);
					v1_2.setPosition(newPos);
					v2_1.setPosition(newPos);
					v2_2.setPosition(newPos);*/


				}
			}



		//for (auto i : nodeVert_1)
		//{
		//	cout << "\n";
		//	for (auto j : i)
		//	{
		//		cout << " " << j;
		//	}
		//}

		//for (int i = 0; i < nodeIdVertId_1.size(); i++)
		//{
		//	//nodeVert_1[i][0]; // vol1
		//	//nodeVert_1[i][1]; // vId
		//	for (int j = 0; j < nodeIdVertId_1.size(); j++)
		//	{
		//		if (nodeIdVertId_1[i][0] == nodeIdVertId_1[j][0] && nodeIdVertId_1[i][1] == nodeIdVertId_1[j][1] && i != j)
		//		{
		//			cout << "\n\n i " << i;
		//			cout << "\n j " << j;
		//			cout << "\n " << nodeIdVertId_1[i][0] << " " << nodeIdVertId_1[i][1] << " -> " << nodeIdVertId_2[i][0] << " " << nodeIdVertId_2[i][1];

		//			zItMeshVertex v1_1(dualMeshCol[nodeIdVertId_1[i][0]], nodeIdVertId_1[i][1]);
		//			zItMeshVertex v2_1(dualMeshCol[nodeIdVertId_2[i][0]], nodeIdVertId_2[i][1]);

		//			zItMeshVertex v1_2(dualMeshCol[nodeIdVertId_1[j][0]], nodeIdVertId_1[j][1]);
		//			zItMeshVertex v2_2(dualMeshCol[nodeIdVertId_2[j][0]], nodeIdVertId_2[j][1]);	

		//			zVector newPos = (v1_1.getPosition() + v2_1.getPosition() + v1_2.getPosition() + v2_2.getPosition()) / 4;

		//			v1_1.setPosition(newPos);
		//			v1_2.setPosition(newPos);
		//			v2_1.setPosition(newPos);
		//			v2_2.setPosition(newPos);


		//		}
		//	}



		}
	}
}