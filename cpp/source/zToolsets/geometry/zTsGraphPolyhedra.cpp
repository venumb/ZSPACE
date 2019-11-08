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

	ZSPACE_INLINE zTsGraphPolyhedra::zTsGraphPolyhedra(zObjGraph &_graphObj)
	{
		graphObj = &_graphObj;
		fnGraph = zFnGraph(_graphObj);

		n_nodes = 0;
		getInternalVertex();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsGraphPolyhedra::~zTsGraphPolyhedra() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::create()
	{
		graphEdge_DualCellFace.assign(fnGraph.numEdges(), zIntArray());

		dualMeshCol.assign(fnGraph.numVertices(), zObjMesh());

		dualGraphCol.assign(fnGraph.numVertices(), zObjGraph());

		dualFaceCenter.assign(fnGraph.numVertices(), zPointArray());

		conHullsCol.assign(fnGraph.numVertices(), zObjMesh());
		zFnMesh fnConHull;

		graphMeshCol.assign(fnGraph.numVertices(), zObjMesh());
		zFnMesh fnGraphMesh;
		
		// get crate eccentricity center
		zItGraphVertexArray outV;
		fnGraph.getGraphEccentricityCenter(outV);

		zItGraphVertex bsf(*graphObj);
		if(outV.size() >0) bsf = outV[0];

		// breadth search first sorting
		zItGraphVertexArray g_v_bsf;
		bsf.getBSF(g_v_bsf);

		for (auto g_v : g_v_bsf)
		{		
			fnConHull = zFnMesh(conHullsCol[g_v.getId()]);
			fnGraphMesh = zFnMesh(graphMeshCol[g_v.getId()]);

			vector<zIntArray> primalFace_graphEdge;
			zIntArray graphMeshVertex_primalVertexMap;

			// get vertices > valence 1
			if (g_v.getValence() > 1)
			{		
				zItGraphHalfEdgeArray cHEdges;
				zPointArray v_p;
				zIntArray vpi_graphV; // hull vertex id to v_p vertex id
				zIntArray hvi_vpi; // hull vertex id to v_p vertex id

				// color vertices
				g_v.setColor(zColor(0.0, 1.0, 0.0, 1.0));

				// get connected edges
				g_v.getConnectedHalfEdges(cHEdges);

				// get vertex positions
				//for (int i = 0; i < connected_v_i.size(); i++)
				for (auto he: cHEdges)
				{
					vpi_graphV.push_back(he.getVertex().getId());
					
					if (he.getVertex().checkValency(1))	v_p.push_back(he.getVertex().getPosition());
					else v_p.push_back((g_v.getPosition() + he.getVertex().getPosition()) / 2);
				}

				// make convex hull
				fnConHull.makeConvexHull(v_p);

				zPoint* hullPos = fnConHull.getRawVertexPositions();
				for( int i =0; i< fnConHull.numVertices(); i++)				
					for (int j = 0; j < v_p.size(); j++)
						if (v_p[j].distanceTo(hullPos[i]) < 0.001)						
							hvi_vpi.push_back(vpi_graphV[j]);


				//Primal_EdgeFaceMatrix	
				zIntPairArray face_Cells;

				zSparseMatrix C_ef(cHEdges.size(), fnConHull.numEdges());
				C_ef.setZero();
				vector<zTriplet> coefs_ef;
				
				for (zItMeshEdge hullEdge(conHullsCol[g_v.getId()]); !hullEdge.end(); hullEdge++)
				{
					zIntArray eFaces;
					hullEdge.getFaces(eFaces);
					
					zItMeshVertexArray eVerts;
					hullEdge.getVertices(eVerts);

					zIntArray fVerts;

					// vert 1 -> hvi
					zItMeshVertex v0;
					fnGraphMesh.addVertex(eVerts[0].getPosition(), false, v0);					
					fVerts.push_back(v0.getId());
					// vert 3 -> hvj
					zItMeshVertex v1;
					fnGraphMesh.addVertex(eVerts[1].getPosition(), false, v1);
					fVerts.push_back(v1.getId());
					// vert 2 -> node center
					zItMeshVertex v2;
					fnGraphMesh.addVertex(g_v.getPosition(), false, v2);
					fVerts.push_back(v2.getId());
								
					//build face fi
					zItMeshFace fi;
					fnGraphMesh.addPolygon(fVerts, fi);

					// face cell pairs
					zIntPair tmpPair = std::make_pair(eFaces[0], eFaces[1]);
					face_Cells.push_back(tmpPair);

					// face edge coefficients
					int edge0 = eVerts[0].getId() ;					
					coefs_ef.push_back(zTriplet(edge0, hullEdge.getId(), 1.0));

					int edge1 = eVerts[1].getId();					
					coefs_ef.push_back(zTriplet(edge1, hullEdge.getId(), -1.0));

					// map primal face to connected graph edges

					primalFace_graphEdge.push_back(zIntArray());

					int graph_v0 = hvi_vpi[eVerts[0].getId()];
					int graph_v1 = g_v.getId();

					zItGraphHalfEdge he0;
					bool chk0 = fnGraph.halfEdgeExists(graph_v0, graph_v1, he0);
					if (chk0) primalFace_graphEdge[hullEdge.getId()].push_back(he0.getEdge().getId());

					graph_v0 = hvi_vpi[eVerts[1].getId()];

					zItGraphHalfEdge he1;
					bool chk1 = fnGraph.halfEdgeExists(graph_v0, graph_v1, he1);
					if (chk1) primalFace_graphEdge[hullEdge.getId()].push_back(he1.getEdge().getId());
				}

				C_ef.setFromTriplets(coefs_ef.begin(), coefs_ef.end());			
				
				fnGraphMesh.setVertexColor(zColor(0, 0, 1, 1));
		
				// get cell centers			
				zPointArray cellCenters;
				for (zItMeshFace hullFace(conHullsCol[g_v.getId()]); !hullFace.end(); hullFace++)
				{
					zPoint cen;
					zPointArray fVerts;
					hullFace.getVertexPositions(fVerts);

					for (auto p : fVerts) cen += p;
					cen += g_v.getPosition();
					cen /= (fVerts.size() + 1);

					cellCenters.push_back(cen);
				}

				// Primal_FaceCellMatrix
				zSparseMatrix C_fc(fnGraphMesh.numPolygons(), cellCenters.size());
				C_fc.setZero();

				vector<zTriplet> coefs_fc;
			
				for (zItMeshFace f(graphMeshCol[g_v.getId()]); !f.end(); f++)
				{
					zPointArray fVerts;
					f.getVertexPositions(fVerts);

					int fId= f.getId();
					zVector norm = (fVerts[2] - fVerts[0]) ^ (fVerts[1] - fVerts[0]);
					norm.normalize();

					int c0 = face_Cells[fId].first;
					zVector dir0 = cellCenters[c0] - f.getCenter();
					dir0.normalize();

					( norm * dir0 > 0) ? coefs_fc.push_back(zTriplet(fId, c0, 1.0)) : coefs_fc.push_back(zTriplet(fId, c0, -1.0));

					int c1 = face_Cells[fId].second;
					zVector dir1 = cellCenters[c1] - f.getCenter();
					dir1.normalize();

					(norm * dir1 > 0) ? coefs_fc.push_back(zTriplet(fId, c1, 1.0)) : coefs_fc.push_back(zTriplet(fId, c1, -1.0));
				}

				C_fc.setFromTriplets(coefs_fc.begin(), coefs_fc.end());
				
				// create dual mesh
				createDualMesh(cellCenters, fnGraphMesh.numPolygons(), C_ef, C_fc, dualMeshCol[g_v.getId()]);
				
				zFnMesh fnDual(dualMeshCol[g_v.getId()]);
				zPointArray fCenters;
				fnDual.getCenters(zFaceData, fCenters);

				//dualMeshCol[n_nodes].setShowFaceNormals(true, fCenters, 0.1);

				// map dual face cell to graph half edge
				for (int i = 0; i < C_ef.rows(); i++)
				{
					zIntArray primalFaces;

					for (int j = 0; j < C_ef.cols(); j++)					
						if (C_ef.coeff(i, j) == 1 || C_ef.coeff(i, j) == -1)					
							primalFaces.push_back(j);
												
					int commongraphEdge = -1;
					int testEdge = primalFace_graphEdge[primalFaces[0]][0];
					bool commonEdge = true;

					for (int j = 1; j < primalFaces.size(); j++)
					{
						if (primalFace_graphEdge[primalFaces[j]][0] == testEdge || primalFace_graphEdge[primalFaces[j]][1] == testEdge)
						{
							continue;
						}
						else
						{
							commonEdge = false;
							break;
						}
					}

					if (commonEdge) commongraphEdge = testEdge;
					else commongraphEdge = primalFace_graphEdge[primalFaces[0]][1];;
					
					graphEdge_DualCellFace[commongraphEdge].push_back(g_v.getId());
					graphEdge_DualCellFace[commongraphEdge].push_back(i);
				}



				n_nodes++;
				cout << "\n///////////////////////";
			}		
		}

		/*for (int i = 0; i < graphEdge_DualCellFace.size(); i++)
		{
			printf("\n");
			for (int j = 0; j < graphEdge_DualCellFace[i].size(); j++)
			{
				printf(" %i ", graphEdge_DualCellFace[i][j]);
			}


		}*/
		// snap and merge dual cells
		snapDualCells(g_v_bsf, outV);

		drawDualFaceConnectivity();

		//// tmp export dual meshes
		//int num = 0;
		//for (auto m : dualMeshCol)
		//{
		//	zFnMesh fnDual(m);
		//	fnDual.to("C:/Users/Leo.b/Desktop/graph/export/mesh_" + to_string(num) + ".obj", zOBJ);

		//	num++;
		//}
	}

	//---- DRAW METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::drawGraph(bool drawIds)
	{
		glPointSize(6.0);
		graphObj->setShowElements(true, true);
		graphObj->draw();
		glPointSize(1.0);

		//if (drawIds)
		//{
		//	glColor3f(1, 0, 0.4);
		//	for (zItGraphVertex v(*graphObj); !v.end(); v++)
		//		display.drawTextAtPoint(to_string(v.getId()), v.getPosition());
		//}
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::drawConvexHulls()
	{
		for (auto m : conHullsCol)
			m.drawMesh();		
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::drawGraphMeshes()
	{
		zUtilsDisplay disp;
		for (auto m : graphMeshCol)
		{
			m.setShowVertices(false);
			m.setShowFaces(true);
			m.drawMesh();			
		}	
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::drawDual(bool drawDualMeshFaces, bool drawIds)
	{	
		for (int i = 0; i < dualMeshCol.size(); i++)
		{			
			dualMeshCol[i].setShowFaces(drawDualMeshFaces);
			dualMeshCol[i].draw();

			if (drawIds)
			{
				glColor3f(1, 0, 0.4);
				for (zItMeshVertex v(dualMeshCol[i]); !v.end(); v++)
					display.drawTextAtPoint(to_string(v.getId()), v.getPosition());

				zFnMesh tmp(dualMeshCol[i]);
				glColor3f(0, 1, 0.4);	
				display.drawTextAtPoint(to_string(i), tmp.getCenter());
			}
		}
	}

	//---- PRIVATE CREATE METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::createDualMesh(zPointArray &positions, int numEdges, zSparseMatrix &C_dual_fe, zSparseMatrix &C_fc, zObjMesh &dualMesh)
	{
		zObjGraph dualGraph;
		createDualGraph(positions, numEdges, C_fc, dualGraph);

		zIntArray polyConnects, polyCounts;

		zIntPairArray existingHalfEdges;

		for (int i = 0; i < C_dual_fe.rows(); i++)
		{
			// get num eges per face
			int n_ef = 0;

			zIntArray tempPolyConnect;
			for (int j = 0; j < C_dual_fe.cols(); j++)
			{
				if (C_dual_fe.coeff(i, j) == 1)
				{
					zItGraphEdge gE(dualGraph, j);

					tempPolyConnect.push_back(gE.getHalfEdge(0).getVertex().getId());
					n_ef++;
				}

				if (C_dual_fe.coeff(i, j) == -1)
				{
					zItGraphEdge gE(dualGraph, j);

					tempPolyConnect.push_back(gE.getHalfEdge(1).getVertex().getId());
					n_ef++;
				}
			}

			// check if winding is correct
			bool windingCorrect = true;
			for (int i = 0; i < tempPolyConnect.size(); i++)
			{
				int next = tempPolyConnect[(i + 1) % tempPolyConnect.size()];
				int cur = tempPolyConnect[i];
				auto p = std::make_pair(cur, next);

				if (std::find(existingHalfEdges.begin(), existingHalfEdges.end(), p) != existingHalfEdges.end())
					windingCorrect = false;
			}

			// reverse winding if not correct
			if (!windingCorrect) reverse(tempPolyConnect.begin(), tempPolyConnect.end());

			// added to existing edges
			for (int i = 0; i < tempPolyConnect.size(); i++)
			{
				int next = tempPolyConnect[(i + 1) % tempPolyConnect.size()];
				int cur = tempPolyConnect[i];
				auto p = std::make_pair(cur, next);

				existingHalfEdges.push_back(p);

				polyConnects.push_back(cur);
			}


			polyCounts.push_back(n_ef);
		}

		zFnMesh fnMesh(dualMesh);
		fnMesh.create(positions, polyCounts, polyConnects);
		fnMesh.computeMeshNormals();
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::createDualGraph(zPointArray &positions, int numEdges,  zSparseMatrix &C_fc, zObjGraph &dualGraph)
	{
		zFnGraph fnDual(dualGraph);
		zIntArray edgeConnects;
		edgeConnects.assign(numEdges * 2, -1);
		
		zPointArray pos;
		pos.push_back(positions[0]);

		for (int j = 0; j < C_fc.rows(); j++)
		{
			int edgeId = j;

			for (int i = 0; i < C_fc.cols(); i++)
			{
				int vertexId = i;

				// head of the edge
				if (C_fc.coeff(edgeId, vertexId) == 1)
				{
					edgeConnects[edgeId * 2 + 1] = vertexId;
				}

				// tail of the edge
				if (C_fc.coeff(edgeId, vertexId) == -1)
				{
					edgeConnects[edgeId * 2] = vertexId;
				}
			}
		}


		fnDual.create(positions, edgeConnects);
		fnDual.setEdgeColor(zColor(0, 1, 0, 1));
	}
	
	ZSPACE_INLINE void zTsGraphPolyhedra::getCellCenter(zItGraphVertex &graphVertIt)
	{

	}

	ZSPACE_INLINE void zTsGraphPolyhedra::getContainedFace()
	{

	}

	//---- PRIVATE UTILITY METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::getInternalVertex()
	{
		for (zItGraphVertex g_v(*graphObj); !g_v.end(); g_v++)
			if (g_v.getValence() > 1)		
				internalVertexIds.push_back(g_v.getId());		
				
		firstInternalVertexId = internalVertexIds[0];
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::cyclicSort(zIntArray &unsorted, zPoint &cen,zPointArray &pts, zVector refDir, zVector normal)
	{
		vector<double> angles;
		map< double, int > angle_e_Map;

		zVector horz = refDir;;
		zVector upVec = normal;;

		zVector cross = upVec ^ horz;

		zIntArray sorted;

		angles.clear();

		for (auto &p: pts)
		{
			float angle = 0;			

			zVector vec1(p - cen);
			angle = horz.angle360(vec1, normal);



			// check if same key exists
			for (int k = 0; k < angles.size(); k++)
			{
				if (angles[k] == angle) angle += 0.01;
			}

			angle_e_Map[angle] = (&p - &pts[0]);
			angles.push_back(angle);
		}

		sort(angles.begin(), angles.end());

		for (int i = 0; i < angles.size(); i++)
		{


			int id = angle_e_Map.find(angles[i])->second;
			if (id > pts.size())
			{
				id = 0;
			}
			sorted.push_back((unsorted[id]));


		}

		unsorted.clear();
		unsorted = sorted;
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::snapDualCells(zItGraphVertexArray &bsf, zItGraphVertexArray &gCenters)
	{
		if (n_nodes == 0) return;
		
		map<zIntPair, int> volVert_globalId;
		int globalId_counter = 0;

		map<zIntPair, zIntPair> volEdge_mappedVolEdge;

		vector<zIntArray> mapped_volEdges;

		zBoolArray edgeVisited;
		edgeVisited.assign(fnGraph.numEdges(), false);

		for (auto g_v : bsf)
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

					if (graphEdge_DualCellFace[graphEdgeId].size() != 4)
						continue;

					//printf("\n vol face: ");
					//for (int j = 0; j < graphEdge_DualCellFace[graphEdgeId].size(); j++)
					//{
					//	printf(" %i ", graphEdge_DualCellFace[graphEdgeId][j]);
					//}

					zItMeshFace f1(dualMeshCol[graphEdge_DualCellFace[graphEdgeId][0]], graphEdge_DualCellFace[graphEdgeId][1]);

					zPointArray f1VertPos;
					f1.getVertexPositions(f1VertPos);

					zIntArray f1Verts;
					f1.getVertices(f1Verts);

					//zVector ref = f1VertPos[0] - f1.getCenter();

					//printf("\nvolume: %i face: %i", graphEdge_DualCellFace[graphEdgeId][0], f1.getId());
					//for (auto fV : f1Verts) printf(" %i ", fV);

					zItMeshFace f2(dualMeshCol[graphEdge_DualCellFace[graphEdgeId][2]], graphEdge_DualCellFace[graphEdgeId][3]);

					zPointArray f2VertPos;
					f2.getVertexPositions(f2VertPos);

					zIntArray f2Verts;
					f2.getVertices(f2Verts);

					//printf("\nvolume: %i face: %i", graphEdge_DualCellFace[graphEdgeId][2], f2.getId());
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

							zIntArray tmpArray = { graphEdge_DualCellFace[graphEdgeId][0], f1Edge.getId(), graphEdge_DualCellFace[graphEdgeId][2], f2Edge.getId() };

							mapped_volEdges.push_back(tmpArray);
							
							zIntPair tmpp1(graphEdge_DualCellFace[graphEdgeId][0], f1Edge.getId());
							zIntPair tmpp2(graphEdge_DualCellFace[graphEdgeId][2], f2Edge.getId());
							
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
		
			

		for (int j = 0; j < gCenters.size(); j++)
		{
			int vol1 = gCenters[j].getId();

			zItGraphHalfEdgeArray conHEdges;
			gCenters[j].getConnectedHalfEdges(conHEdges);

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
				
				if (graphEdge_DualCellFace[graphEdgeId].size() != 4)	continue;

				int vol1 = graphEdge_DualCellFace[graphEdgeId][0];
				int vol2 = graphEdge_DualCellFace[graphEdgeId][2];

				zItMeshFace f1(dualMeshCol[vol1], graphEdge_DualCellFace[graphEdgeId][1]);
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
					
				

				zItMeshFace f2(dualMeshCol[vol2], graphEdge_DualCellFace[graphEdgeId][3]);
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
				if (vol1 < snap) pos2[f2Edge.getVertex().getId()] = pos1[f1Edge.getStartVertex().getId()];

				f1Edge = f1Edge.getNext();
				f2Edge = f2Edge.getPrev();



				// tmp to visualise lines
				
				tmp1.push_back(f1Edge.getStartVertex().getPosition());
				tmp2.push_back(f2Edge.getVertex().getPosition());

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

	ZSPACE_INLINE void zTsGraphPolyhedra::drawDualFaceConnectivity()
	{
		for (int i = 0; i < graphEdge_DualCellFace.size(); i++)
		{
			for (int j = 0; j < graphEdge_DualCellFace[i].size(); j += 2)
			{
				int cellId = graphEdge_DualCellFace[i][j];
				int faceId = graphEdge_DualCellFace[i][j + 1];

				if (graphEdge_DualCellFace[i].size() == 4)
				{
					zItMeshFace f(dualMeshCol[cellId], faceId);
					f.setColor(zColor(1, 0, 0, 1));
				}
			}
		}
	}
}