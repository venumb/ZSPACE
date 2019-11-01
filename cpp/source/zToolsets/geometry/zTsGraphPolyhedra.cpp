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
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsGraphPolyhedra::~zTsGraphPolyhedra() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsGraphPolyhedra::createGraphMesh()
	{
		graphEdge_DualCellFace.assign(fnGraph.numEdges(), zIntArray());

		dualMeshCol.assign(fnGraph.numVertices(), zObjMesh());

		dualGraphCol.assign(fnGraph.numVertices(), zObjGraph());


		conHullsCol.assign(fnGraph.numVertices(), zObjMesh());
		zFnMesh fnConHull;

		graphMeshCol.assign(fnGraph.numVertices(), zObjMesh());
		zFnMesh fnGraphMesh;

		graphObj->setShowElements(true, true);
		
		// get vertices > valence 1
		for (zItGraphVertex g_v(*graphObj); !g_v.end(); g_v++)
		{
			fnConHull = zFnMesh(conHullsCol[n_nodes]);
			fnGraphMesh = zFnMesh(graphMeshCol[n_nodes]);

			vector<zIntArray> primalface_graphEdge;
			zIntArray graphMeshVertex_primalVertexMap;

			if (g_v.getValence() > 1)
			{
				zItGraphHalfEdgeArray cHEdges;
				
				zPointArray v_p;
				v_p.clear();

				zIntArray vpi_graphV; // hull vertex id to v_p vertex id

				zIntArray hvi_vpi; // hull vertex id to v_p vertex id
				hvi_vpi.clear();

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
				{
					for (int j = 0; j < v_p.size(); j++)
					{
						if (v_p[j].distanceTo(hullPos[i]) < 0.001)
						{
							hvi_vpi.push_back(vpi_graphV[j]);
						}
					}
				}

				printf("\n hvi_vpi ");

				for (int i = 0; i < hvi_vpi.size(); i++)
				{
					printf("\n %i %i ", i, hvi_vpi[i]);
				}

				zIntPairArray existingHalfEdges;
				zIntPairArray face_Cells;
				zIntPairArray face_Edges;


				//Primal_EdgeVertexMatrix	
				zSparseMatrix C_ev(cHEdges.size(), cHEdges.size() + 1);
				C_ev.setZero();

				vector<zTriplet> coefs_ev;
				for (int i = 1; i <= cHEdges.size(); i++)
				{
					coefs_ev.push_back(zTriplet(i - 1, 0, 1.0));
					coefs_ev.push_back(zTriplet(i - 1, i, -1.0));
				}

				C_ev.setFromTriplets(coefs_ev.begin(), coefs_ev.end());

				//Primal_EdgeFaceMatrix	
				zSparseMatrix C_ef(cHEdges.size(), fnConHull.numEdges());
				C_ef.setZero();
				vector<zTriplet> coefs_ef;
				
				for (zItMeshEdge hullEdge(conHullsCol[n_nodes]); !hullEdge.end(); hullEdge++)
				{
					zIntArray eFaces;
					hullEdge.getFaces(eFaces);
					
					zItMeshVertexArray eVerts;
					hullEdge.getVertices(eVerts);

					zIntArray fVerts;

					//tmpP.clear();
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

					primalface_graphEdge.push_back(zIntArray());

					int graph_v0 = hvi_vpi[eVerts[0].getId()];
					int graph_v1 = g_v.getId();

					zItGraphHalfEdge he0;
					bool chk0 = fnGraph.halfEdgeExists(graph_v0, graph_v1, he0);
					if (chk0) primalface_graphEdge[hullEdge.getId()].push_back(he0.getEdge().getId());
					printf("\n %i : %i %i : %i ", hullEdge.getId(), graph_v0, graph_v1, he0.getEdge().getId());


					graph_v0 = hvi_vpi[eVerts[1].getId()];

					zItGraphHalfEdge he1;
					bool chk1 = fnGraph.halfEdgeExists(graph_v0, graph_v1, he1);
					if (chk1) primalface_graphEdge[hullEdge.getId()].push_back(he1.getEdge().getId());

					printf("\n %i : %i %i : %i ", hullEdge.getId(), graph_v0, graph_v1, he1.getEdge().getId());
				}

				C_ef.setFromTriplets(coefs_ef.begin(), coefs_ef.end());			
				
				fnGraphMesh.setVertexColor(zColor(0, 0, 1, 1));


				
				//get cell centers			
				zPointArray cellCenters;
				for (zItMeshFace hullFace(conHullsCol[n_nodes]); !hullFace.end(); hullFace++)
				{
					zPoint cen;

					zPointArray fVerts;
					hullFace.getVertexPositions(fVerts);

					for (auto p : fVerts) cen += p;
					cen += g_v.getPosition();

					cen /= (fVerts.size() + 1);

					cellCenters.push_back(cen);
				}

								
				//Primal_FaceCellMatrix
				zSparseMatrix C_fc(fnGraphMesh.numPolygons(), cellCenters.size());
				C_fc.setZero();

				vector<zTriplet> coefs_fc;

				
				for (zItMeshFace f(graphMeshCol[n_nodes]); !f.end(); f++)
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
				

				// create dual
				createDualGraph(cellCenters, fnGraphMesh.numPolygons(), C_fc, dualGraphCol[n_nodes]);
				createDualMesh(cellCenters, fnGraphMesh.numPolygons(), C_ef, C_fc, dualMeshCol[n_nodes]);
				
				// map dual face cell to graph half edge
				for (int i = 0; i < C_ef.rows(); i++)
				{
					zIntArray primalfaces;
					for (int j = 0; j < C_ef.cols(); j++)
					{
						if (C_ef.coeff(i, j) == 1 || C_ef.coeff(i, j) == -1)
						{
							primalfaces.push_back(j);
						}						
					}

					int commongraphEdge = -1;
					int testEdge = primalface_graphEdge[primalfaces[0]][0];
					bool commonEdge = true;
					for (int j = 1; j < primalfaces.size(); j++)
					{
						if (primalface_graphEdge[primalfaces[j]][0] == testEdge || primalface_graphEdge[primalfaces[j]][1] == testEdge)
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
					else commongraphEdge = primalface_graphEdge[primalfaces[0]][1];;
					
					graphEdge_DualCellFace[commongraphEdge].push_back(n_nodes);
					graphEdge_DualCellFace[commongraphEdge].push_back(i);
				}

				

				// track visited edges
				zBoolArray eVisited;
				eVisited.clear();

				for (int i = 0; i < fnConHull.numEdges(); i++)
					eVisited.push_back(false);

				//vector<zIntArray> f_edges;

				//for (zItMeshVertex hvi(conHullsCol[n_nodes]); !hvi.end(); hvi++)
				//{
				//	// track visited edges
				//	zItMeshHalfEdgeArray connectedHE;
				//	connectedHE.clear();
				//	hvi.getConnectedHalfEdges(connectedHE);

				//	// get connected vertices ID vj
				//	zIntArray hvj;
				//	hvj.clear();


				//	hvi.getConnectedVertices(hvj);

				//	//build face fi
				//	zItMeshFace fi;
				//	
				//	for (int i = 0; i < connectedHE.size(); i++)
				//	{

				//		if (eVisited[connectedE[i]] == false)
				//		{
				//			tmpP.clear();
				//			// vert 1 -> hvi
				//			tmpP.push_back(v_p[hvi_vpi[hvi.getId()]]);
				//			// vert 2 -> node center
				//			tmpP.push_back(g_v.getPosition());
				//			// vert 3 -> hvj
				//			tmpP.push_back(v_p[hvi_vpi[hvj[i]]]);

				//			cout << "\nADD POLYGON!";
				//			for (auto p : tmpP)						
				//				printf("\nx: %1.2f \t y: %1.2f \t z: %1.2f", p.x, p.y, p.z);
				//			

				//			//fnGraphMesh.addPolygon(tmpP, fi);
				//			
				//			//cout << "\nfnGraphMesh size: " << fnGraphMesh.numVertices();
				//		}
				//	}

				//	// mark edge as visited
				//	for (auto _conE : connectedE)
				//		eVisited[_conE] = true;

				//	//for (int i = 0; i < eVisited.size(); i++)
				//	//	printf("\nedgeId %i visited: %s", i, (eVisited[i]) ? ("true") : ("false"));
				//	//cout << "\n";



				//	// get face to garphEdgeId connection f_edges (two graphEdges per face)
				//}

				n_nodes++;

				cout << "\n/////////////////////////////////////////////\n";
			}

			
		}


		for (int i = 0; i < graphEdge_DualCellFace.size(); i++)
		{
			printf("\n e %i :", i);


			for (int j = 0; j < graphEdge_DualCellFace[i].size(); j+= 2)
			{
				int cellId = graphEdge_DualCellFace[i][j];
				int faceId = graphEdge_DualCellFace[i][j +1];

				printf(" %i %i | ", cellId, faceId);

				if (graphEdge_DualCellFace[i].size() == 4)
				{
					zItMeshFace f(dualMeshCol[cellId], faceId);
					f.setColor(zColor(1, 0, 0, 1));
				}
				

			}
		}

		//cout << "\nn_nodes: " << n_nodes;

		//for (auto id : nodeCenterIdOfGraph)		
		//	cout << "\nnodeCenterIdOfGraph: " << id;	

		//fnGraphMesh.to("C:/Users/Leo.b/Desktop/graph/graphMesh.obj", zOBJ);

	}


	//---- UTILITY METHODS

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
			m.setShowVertices(true);
			m.setShowFaces(false);
			m.drawMesh();			
		}	

		/*for (zItMeshFace f(graphMeshCol[0]); !f.end(); f++)
		{
			disp.drawTextAtPoint(to_string(f.getId()), f.getCenter());
		}*/
	}

	ZSPACE_INLINE void zTsGraphPolyhedra::drawDual()
	{	

		for (int i = 0; i < dualMeshCol.size(); i++)
		{
			dualMeshCol[i].setShowFaces(true);
			dualMeshCol[i].drawMesh();

			//dualGraphCol[i].drawGraph();
		}

		/*zUtilsDisplay disp;
		glColor3f(1, 0, 0);
		for (zItGraphEdge e(dualGraphCol[0]); !e.end(); e++)
		{
			disp.drawTextAtPoint(to_string(e.getId()), e.getCenter());
		}*/
	}

	//---- PRIVATE GET METHODS

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

	}

	ZSPACE_INLINE void zTsGraphPolyhedra::createDualGraph(zPointArray &positions, int numEdges,  zSparseMatrix &C_fc, zObjGraph &dualGraph)
	{
		zFnGraph fnDual(dualGraph);
		zIntArray edgeConnects;
		edgeConnects.assign(numEdges * 2, -1);

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

		//printf("\n dual %i %i", fnDual.numVertices(), fnDual.numEdges());
	}

	ZSPACE_INLINE bool zTsGraphPolyhedra::getPrimal_EdgeVertexMatrix(zObjMesh inMesh, zSparseMatrix &out)
	{

	}

}