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


#include<headers/zToolsets/geometry/zTsSDFBridge.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSDFBridge::zTsSDFBridge() {}


	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSDFBridge::~zTsSDFBridge() {}

	//---- CREATE METHODS
	
	ZSPACE_INLINE void zTsSDFBridge::createCutPlaneMesh(double width)
	{

		zFnMesh fnGuideMesh(*o_guideMesh);

		zPoint minBB, maxBB;
		fnGuideMesh.getBounds(minBB, maxBB);
		zPoint top_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, maxBB.z);
		zPoint bottom_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, minBB.z);
		zVector up(0, 0, 1);

		globalVertices.clear();
		guideEdge_globalVertex.clear();
		guideFace_globalVertex.clear();

		zPointArray positions;
		zIntArray pCounts, pConnects;
					

		// add face centers to global vertices

		int n_gV = 0;
	
		for (zItMeshFace f(*o_guideMesh); !f.end(); f++)
		{
			//if (!excludeFacesBoolean[f.getId()])
			//{
				zPoint p = f.getCenter();
				zVector n = f.getNormal();

				zIntArray vIds = { n_gV,n_gV + 1 };
				guideFace_globalVertex.push_back(vIds);

				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p + n * width * 0.5;;
				n_gV++;

				//// project p to top plane
				//double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
				//zPoint projected_p = (p - (up * minDist));
				//globalVertices.push_back(zGlobalVertex());
				//globalVertices[n_gV].pos = projected_p;;
				//n_gV++;

				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p - n * width * 0.5;;
				n_gV++;

				if (excludeFacesBoolean[f.getId()])
				{
					zItMeshHalfEdge he = f.getHalfEdge();

					while (!he.getEdge().onBoundary())
					{
						he = he.getNext();
					}
					
					he = he.getSym();
					zVector he_vec = he.getVector();
					zVector perp = n ^ he_vec;
					perp.normalize();

					p = he.getCenter();

					globalVertices[n_gV - 2].pos = p + n * width * 0.5;;
					globalVertices[n_gV - 2].pos += perp * width * 0.25;

					globalVertices[n_gV - 1].pos = p - n * width * 0.5;;
					globalVertices[n_gV - 1].pos += perp * width * 0.25;
				}
			//}
			//else
			//{
				//zIntArray vIds = { -1,-1 };
				//guideFace_globalVertex.push_back(vIds);
			//}		
									
		}

		//// add boundary edge centers to global vertices
		//for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		//{
		//	if (e.onBoundary())
		//	{				

		//		zIntArray eVerts;
		//		e.getVertices(eVerts);

		//		/*if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]])
		//		{
		//			zIntArray vIds = { -1,-1 };
		//			guideEdge_globalVertex.push_back(vIds);					
		//			
		//		}
		//		else
		//		{*/
		//			zPoint p = e.getCenter();

		//			zItMeshFaceArray efaces;
		//			e.getFaces(efaces);
		//			zVector n = efaces[0].getNormal();

		//			zItMeshHalfEdge he = e.getHalfEdge(0).onBoundary() ? e.getHalfEdge(0) : e.getHalfEdge(1);
		//			zVector he_vec = he.getVector();
		//			zVector perp = n ^ he_vec;
		//			perp.normalize();

		//			zIntArray vIds = { n_gV,n_gV + 1 };
		//			guideEdge_globalVertex.push_back(vIds);

		//			globalVertices.push_back(zGlobalVertex());
		//			globalVertices[n_gV].pos = p + n * width * 0.5;;
		//			globalVertices[n_gV].pos += perp * width * 0.25;
		//			n_gV++;

		//			//// project p to top plane
		//			//double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
		//			//zPoint projected_p = (p - (up * minDist));
		//			//globalVertices.push_back(zGlobalVertex());
		//			//globalVertices[n_gV].pos = projected_p;;
		//			//n_gV++;

		//			globalVertices.push_back(zGlobalVertex());
		//			globalVertices[n_gV].pos = p - n * width * 0.5;;
		//			globalVertices[n_gV].pos += perp * width * 0.25;
		//			n_gV++;
		//		//}
		//		
		//	}
		//	else
		//	{
		//		zIntArray vIds = { -1,-1 };
		//		guideEdge_globalVertex.push_back(vIds);
		//	}

		//}

		printf("\n globalVertices %i ", globalVertices.size());

		//create plane mesh
		planeVertex_globalVertex.clear();
		guideVertex_planeFace.clear();
		guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		guideHalfEdge_planeFace.clear();
		guideHalfEdge_planeFace.assign(fnGuideMesh.numHalfEdges(), zIntArray());

		zColorArray faceColors;

		// add face per non constrained guide vertex 
		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{		
			//if (!fixedVerticesBoolean[v.getId()])
			//{				
				if (v.onBoundary())
				{
					zItMeshHalfEdgeArray cHEdges;
					v.getConnectedHalfEdges(cHEdges);

					//if (cHEdges.size() == 3)
					//{												
						// TOP face
						/*for (auto &cHE: cHEdges)
						{
							if (cHE.onBoundary())
							{

								zItMeshHalfEdge tempHE;
								tempHE = cHE;

								bool exit = false;
								do
								{
									int gV = (tempHE.onBoundary()) ? guideEdge_globalVertex[tempHE.getEdge().getId()][0] : guideFace_globalVertex[tempHE.getFace().getId()][0];

									globalVertices[gV].coincidentVertices.push_back(positions.size());
									planeVertex_globalVertex.push_back(gV);

									pConnects.push_back(positions.size());
									positions.push_back(globalVertices[gV].pos);

									tempHE = tempHE.getPrev();
									if (tempHE.onBoundary())
									{
										gV = guideEdge_globalVertex[tempHE.getEdge().getId()][0];

										globalVertices[gV].coincidentVertices.push_back(positions.size());
										planeVertex_globalVertex.push_back(gV);

										pConnects.push_back(positions.size());
										positions.push_back(globalVertices[gV].pos);
									}

									tempHE = tempHE.getSym();
									if (tempHE == cHE) exit = true;


								} while (!exit);

								guideVertex_planeFace[v.getId()] = pCounts.size();
								pCounts.push_back(4);
								faceColors.push_back(zColor(1, 0, 0, 1));

								guideVertex_planeFace[v.getId()] = pCounts.size();

							}
						}*/

					//// EDGE faces
					//for (auto& cHE : cHEdges)
					//{
					//	//0
					//	int gV = (cHE.onBoundary()) ? guideEdge_globalVertex[cHE.getEdge().getId()][1] : guideFace_globalVertex[cHE.getFace().getId()][1];
					//	globalVertices[gV].coincidentVertices.push_back(positions.size());
					//	planeVertex_globalVertex.push_back(gV);
					//	pConnects.push_back(positions.size());
					//	positions.push_back(globalVertices[gV].pos);

					//	//1
					//	gV = (cHE.onBoundary()) ? guideEdge_globalVertex[cHE.getEdge().getId()][0] : guideFace_globalVertex[cHE.getFace().getId()][0];
					//	globalVertices[gV].coincidentVertices.push_back(positions.size());
					//	planeVertex_globalVertex.push_back(gV);
					//	pConnects.push_back(positions.size());
					//	positions.push_back(globalVertices[gV].pos);

					//	//2
					//	gV = (cHE.getSym().onBoundary()) ? guideEdge_globalVertex[cHE.getSym().getEdge().getId()][0] : guideFace_globalVertex[cHE.getSym().getFace().getId()][0];
					//	globalVertices[gV].coincidentVertices.push_back(positions.size());
					//	planeVertex_globalVertex.push_back(gV);
					//	pConnects.push_back(positions.size());
					//	positions.push_back(globalVertices[gV].pos);

					//	//3
					//	gV = (cHE.getSym().onBoundary()) ? guideEdge_globalVertex[cHE.getSym().getEdge().getId()][1] : guideFace_globalVertex[cHE.getSym().getFace().getId()][1];
					//	globalVertices[gV].coincidentVertices.push_back(positions.size());
					//	planeVertex_globalVertex.push_back(gV);
					//	pConnects.push_back(positions.size());
					//	positions.push_back(globalVertices[gV].pos);

					//	guideHalfEdge_planeFace[cHE.getId()] = pCounts.size();
					//	pCounts.push_back(4);

					//	faceColors.push_back(zColor(0, 1, 0, 1));

					//}

					//}
						
					
					
				}
				else
				{
					
						zItMeshHalfEdgeArray cHEdges;
						v.getConnectedHalfEdges(cHEdges);

						// TOP face
						/*for (auto& cHE : cHEdges)
						{
							int gV = guideFace_globalVertex[cHE.getFace().getId()][0];

							globalVertices[gV].coincidentVertices.push_back(positions.size());
							planeVertex_globalVertex.push_back(gV);

							pConnects.push_back(positions.size());
							positions.push_back(globalVertices[gV].pos);

						}
						guideVertex_planeFace[v.getId()] = pCounts.size();
						pCounts.push_back(cHEdges.size());
						faceColors.push_back(zColor(1, 0, 0, 1));*/

						// EDGE faces
						for (auto& cHE : cHEdges)
						{

							int numExcludefaces = 0; 
							if (excludeFacesBoolean[cHE.getFace().getId()]) numExcludefaces++;
							if (excludeFacesBoolean[cHE.getSym().getFace().getId()]) numExcludefaces++;

							//printf("\n v %i : ", v.getId());

							if (numExcludefaces < 2)
							{
								//0
								int gV = guideFace_globalVertex[cHE.getFace().getId()][1];
								printf(" %i ", gV);
								globalVertices[gV].coincidentVertices.push_back(positions.size());
								planeVertex_globalVertex.push_back(gV);
								pConnects.push_back(positions.size());
								positions.push_back(globalVertices[gV].pos);

								//1
								gV = guideFace_globalVertex[cHE.getFace().getId()][0];
								//printf(" %i ", gV);
								globalVertices[gV].coincidentVertices.push_back(positions.size());
								planeVertex_globalVertex.push_back(gV);
								pConnects.push_back(positions.size());
								positions.push_back(globalVertices[gV].pos);

								//2
								gV = guideFace_globalVertex[cHE.getSym().getFace().getId()][0];
								//printf(" %i ", gV);
								globalVertices[gV].coincidentVertices.push_back(positions.size());
								planeVertex_globalVertex.push_back(gV);
								pConnects.push_back(positions.size());
								positions.push_back(globalVertices[gV].pos);

								//3
								gV = guideFace_globalVertex[cHE.getSym().getFace().getId()][1];
								//printf(" %i ", gV);
								globalVertices[gV].coincidentVertices.push_back(positions.size());
								planeVertex_globalVertex.push_back(gV);
								pConnects.push_back(positions.size());
								positions.push_back(globalVertices[gV].pos);

								guideHalfEdge_planeFace[cHE.getId()].push_back(pCounts.size());
								pCounts.push_back(4);

								faceColors.push_back(zColor(0, 1, 0, 1));
							}
							
						}
				
					

				}
			//}
			
		}


		// add face per non constrained guide edge 
		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		//fnCutPlanes.setFaceColors(faceColors);

	}

	ZSPACE_INLINE void zTsSDFBridge::createSplitMesh(double width)
	{

		zFnMesh fnGuideMesh(*o_guideMesh);
		zFnMesh fnGuideSmoothMesh(o_guideSmoothMesh);

		zPoint minBB, maxBB;
		fnGuideMesh.getBounds(minBB, maxBB);
		zPoint top_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, maxBB.z);
		zPoint bottom_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, minBB.z);
		zVector up(0, 0, 1);

		globalVertices.clear();
		guideVertex_globalVertex.clear();
		guideEdge_globalVertex.clear();
		guideFace_globalVertex.clear();
		

		zPointArray positions;
		zIntArray pCounts, pConnects;


		// add vertex not on medial edge to  global vertices

		zPoint* smoothPositions = fnGuideSmoothMesh.getRawVertexPositions();

		int n_gV = 0;
		int internalEdgecounter = 0;
		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{
			zIntArray cEdges;
			v.getConnectedEdges(cEdges);

			bool medialEdgeVertex = false;
			for (auto cE : cEdges)
			{
				if (guide_MedialEdgesBoolean[cE])
				{
					medialEdgeVertex = true;
					break;
				}
			}

			if (!medialEdgeVertex)
			{
				zPoint p = v.getPosition(); /*smoothPositions[v.getId()];*/
				zVector n = v.getNormal();
				zIntArray vIds = { n_gV,n_gV + 1 };
				guideVertex_globalVertex.push_back (vIds);


				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p + n * width * 0.5;;
				n_gV++;


				globalVertices.push_back(zGlobalVertex());
				globalVertices[n_gV].pos = p - n * width * 0.5;;
				n_gV++;

				if (v.onBoundary())
				{
					zItMeshHalfEdgeArray cHEdges;
					v.getConnectedHalfEdges(cHEdges);

					zItMeshHalfEdge he;
					bool internalEdge = false;

					for (auto &cHE : cHEdges)
					{
						if (!cHE.getEdge().onBoundary())
						{
							he = cHE;
							internalEdge = true;
						}
					}

					if (!internalEdge)
					{
						internalEdgecounter++;
						continue;
					}

					he = he.getSym();
					zVector he_vec = he.getVector();
					
					he_vec.normalize();

					

					globalVertices[n_gV - 2].pos = p + n * width * 0.5;;
					globalVertices[n_gV - 2].pos += he_vec * width * 0.75;

					globalVertices[n_gV - 1].pos = p - n * width * 0.5;;
					globalVertices[n_gV - 1].pos += he_vec * width * 0.75;
				}
			}
			else
			{
				zIntArray vIds = { -1,-1 };
				guideVertex_globalVertex.push_back(vIds);
			}
		}

		printf("\n globalVertices %i ", globalVertices.size());

		printf("\n internalEdgecounter %i ", internalEdgecounter);

		printf("\n nE %i  eVCount %i ", fnGuideMesh.numEdges(), guideEdge_globalVertex.size());

		//create plane mesh
		planeVertex_globalVertex.clear();
		guideVertex_planeFace.clear();
		guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		guideHalfEdge_planeFace.clear();
		guideHalfEdge_planeFace.assign(fnGuideMesh.numHalfEdges(), zIntArray());

		zColorArray faceColors;

		// add faces
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (guide_MedialEdgesBoolean[e.getId()])
			{
				zItMeshHalfEdge he = e.getHalfEdge(0);

				bool print = false;
				if (he.getVertex().getId() == 223 && he.getStartVertex().getId() == 194) print = true;
				if (he.getVertex().getId() == 194 && he.getStartVertex().getId() == 223) print = true;

				if (he.getVertex().getId() == 213 && he.getStartVertex().getId() == 164) print = true;
				if (he.getVertex().getId() == 164 && he.getStartVertex().getId() == 213) print = true;

				if (!he.getVertex().onBoundary())
				{
					//if(print) printf("\n working 1");

					//0
					int gV = guideVertex_globalVertex[he.getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//1
					 gV = guideVertex_globalVertex[he.getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//2
					gV = guideVertex_globalVertex[he.getSym().getPrev().getStartVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//3
					gV = guideVertex_globalVertex[he.getSym().getPrev().getStartVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);


					guideHalfEdge_planeFace[he.getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getNext().getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getSym().getPrev().getId()].push_back(pCounts.size());

					pCounts.push_back(4);

					faceColors.push_back(zColor(0, 1, 0, 1));
					
				}

				if (!he.getNext().getVertex().onBoundary())
				{
					//if (print) printf("\n working 2");

					//0
					int gV = guideVertex_globalVertex[he.getNext().getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//1
					gV = guideVertex_globalVertex[he.getNext().getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//2
					gV = guideVertex_globalVertex[he.getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//3
					gV = guideVertex_globalVertex[he.getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);


					guideHalfEdge_planeFace[he.getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getNext().getNext().getId()].push_back(pCounts.size());
					

					pCounts.push_back(4);

					faceColors.push_back(zColor(0, 1, 0, 1));
				}


				if (!he.getStartVertex().onBoundary())
				{
					//if (print) printf("\n working 3");

					//0
					int gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//1
					gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//2
					gV = guideVertex_globalVertex[he.getPrev().getStartVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//3
					gV = guideVertex_globalVertex[he.getPrev().getStartVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);


					guideHalfEdge_planeFace[he.getSym().getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getSym().getNext().getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getPrev().getId()].push_back(pCounts.size());
					pCounts.push_back(4);

					faceColors.push_back(zColor(0, 1, 0, 1));

				}

				if (!he.getSym().getPrev().getStartVertex().onBoundary())
				{
					//if (print) printf("\n working 4");

					//0
					int gV = guideVertex_globalVertex[he.getSym().getNext().getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//1
					gV = guideVertex_globalVertex[he.getSym().getNext().getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//2
					gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][0];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);

					//3
					gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][1];
					globalVertices[gV].coincidentVertices.push_back(positions.size());
					planeVertex_globalVertex.push_back(gV);
					pConnects.push_back(positions.size());
					positions.push_back(globalVertices[gV].pos);


					guideHalfEdge_planeFace[he.getSym().getId()].push_back(pCounts.size());
					guideHalfEdge_planeFace[he.getSym().getNext().getNext().getId()].push_back(pCounts.size());

					pCounts.push_back(4);

					faceColors.push_back(zColor(0, 1, 0, 1));

				}

				if (print)
				{
					if (!he.getSym().getNext().getStartVertex().onBoundary())
					{
						if (print) printf("\n working print 4");

						//0
						int gV = guideVertex_globalVertex[he.getSym().getNext().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//1
						gV = guideVertex_globalVertex[he.getSym().getNext().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//2
						gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//3
						gV = guideVertex_globalVertex[he.getSym().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);


						guideHalfEdge_planeFace[he.getSym().getId()].push_back(pCounts.size());
						guideHalfEdge_planeFace[he.getSym().getNext().getNext().getId()].push_back(pCounts.size());
						pCounts.push_back(4);

						faceColors.push_back(zColor(0, 1, 0, 1));

					}
				}


			}

		}

	
		// add face per non constrained guide edge 
		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		// compute face pairs which need to have common targets
		planeFace_targetPair.clear();
		planeFace_targetPair.assign(fnCutPlanes.numPolygons(), -1);

		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			//if (!guide_MedialEdgesBoolean[e.getId()]) continue;

			//
			//
			//zItMeshHalfEdge cHe = e.getHalfEdge(0);
			//zColor currentFaceColor = cHe.getFace().getColor();

			//if (cHe.getNext().getVertex().getValence() > 4) continue;
			//if (cHe.getSym().getNext().getVertex().getValence() > 4) continue;

			//// left
			//if (!cHe.getNext().getNext().getEdge().onBoundary())
			//{
			//	zItMeshHalfEdge temp = cHe;
			//	temp = temp.getNext().getNext().getSym();
			//	temp = temp.getNext().getNext();

			//	zColor tempFaceColor = temp.getFace().getColor();

			//	if (currentFaceColor == tempFaceColor)
			//	{
			//		if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

			//		int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
			//		int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

			//		planeFace_targetPair[f1_id] = f2_id;
			//		planeFace_targetPair[f2_id] = f1_id;

			//		int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
			//		int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

			//		planeFace_targetPair[f3_id] = f4_id;
			//		planeFace_targetPair[f4_id] = f3_id;

			//		//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

			//	}

			//}

			//////// SYM Edge

			//cHe = e.getHalfEdge(1);
			//currentFaceColor = cHe.getFace().getColor();

			//// left
			//if (!cHe.getNext().getNext().getEdge().onBoundary())
			//{
			//	zItMeshHalfEdge temp = cHe;
			//	temp = temp.getNext().getNext().getSym();
			//	temp = temp.getNext().getNext();

			//	zColor tempFaceColor = temp.getFace().getColor();

			//	if (currentFaceColor == tempFaceColor)
			//	{

			//		if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

			//		int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
			//		int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

			//		planeFace_targetPair[f1_id] = f2_id;
			//		planeFace_targetPair[f2_id] = f1_id;

			//		int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
			//		int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

			//		planeFace_targetPair[f3_id] = f4_id;
			//		planeFace_targetPair[f4_id] = f3_id;

			//		//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);
			//	}
			//}

			
		}

		int counter = 0;
		for (auto &p: planeFace_targetPair)
		{
			if (p != -1) printf("\n %i %i ", counter, p);
			counter++;

		}

		//fnCutPlanes.setFaceColors(faceColors);

	}

	ZSPACE_INLINE void zTsSDFBridge::createSmoothGuideMesh(int subdiv)
	{
		zFnMesh fnGuideMesh(*o_guideMesh);
		fnGuideMesh.getDuplicate(o_guideSmoothMesh);

		zFnMesh fnSmoothGuideMesh(o_guideSmoothMesh);
		fnSmoothGuideMesh.smoothMesh(subdiv);

		fnSmoothGuideMesh.setVertexColor(zColor(1, 1, 1, 1));

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			zItMeshVertex v(o_guideSmoothMesh, fixedVertices[i]);
			zColor col(0, 0, 0, 1);
			v.setColor(col);			
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::createDualPlaneMesh(double width)
	{

		//zFnMesh fnGuideMesh(*o_guideMesh);

		//
		//zPoint minBB, maxBB;
		//fnGuideMesh.getBounds(minBB, maxBB);
		//zPoint top_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, maxBB.z) ;
		//zPoint bottom_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, minBB.z);
		//zVector up(0, 0, 1);

		//globalVertices.clear();
		//guideEdge_globalVertex.clear();
		//guideFace_globalVertex.clear();

		//planeVWeights.clear();

		//zPointArray positions;
		//zIntArray pCounts, pConnects;

		//

		//int n_gV = 0;

		//// add bottom center to global vertices
		//globalVertices.push_back(zGlobalVertex());
		//globalVertices[n_gV].pos = bottom_center;;
		//n_gV++;
		//			

		//// add face centers to global vertices
		//for (zItMeshFace f(*o_guideMesh); !f.end(); f++)
		//{
		//	zPoint p = f.getCenter();
		//	zVector n = f.getNormal();
		//	
		//	zIntArray vIds = { n_gV, 0 };
		//	guideFace_globalVertex.push_back(vIds);


		//	// project p to top plane
		//	double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
		//	zPoint projected_p = (p - (up * minDist));

		//	globalVertices.push_back(zGlobalVertex());
		//	globalVertices[n_gV].pos = projected_p;;
		//	n_gV++;
		//		

		//}

		//// add boundary edge centers to global vertices
		//for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		//{
		//	zIntArray vIds = { -1,-1 };
		//	guideEdge_globalVertex.push_back(vIds);

		//}

		//for (zItMeshEdge guide_e(*o_guideMesh); !guide_e.end(); guide_e++)
		//{
		//	if (guide_e.onBoundary())
		//	{

		//		zIntArray eVerts;
		//		guide_e.getVertices(eVerts);

		//		// if both vertices of edge constrained continue
		//		if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;
		//	
		//		// if edge already visited continue
		//		if (guideEdge_globalVertex[guide_e.getId()][0] != -1) continue;
		//		
		//		zItMeshHalfEdge he = (guide_e.getHalfEdge(0).onBoundary()) ? guide_e.getHalfEdge(0) : guide_e.getHalfEdge(1);
		//		
		//		zItMeshHalfEdgeArray  boundaryHalfEdges;
		//		zItMeshVertexArray  boundaryVertices;

		//		zItMeshHalfEdge start = he;
		//		zItMeshHalfEdge e = he;
		//		
		//		bool exit = false;

		//		// walk prev
		//		do
		//		{

		//			if (fixedVerticesBoolean[e.getSym().getVertex().getId()])
		//			{

		//				exit = true;
		//				boundaryVertices.push_back(e.getSym().getVertex());
		//			}

		//			boundaryHalfEdges.push_back(e);
		//			boundaryVertices.push_back(e.getVertex());

		//			if (e.getPrev().isActive())
		//			{
		//				e = e.getPrev();
		//			}
		//			else exit = true;

		//		} while (e != start && !exit);

		//		// walk next 
		//		// checking if the prev walk has completed the full edge loop
		//		if (e != start)
		//		{
		//			bool exit = false;
		//			e = start;
		//			do
		//			{
		//				if (fixedVerticesBoolean[e.getVertex().getId()])
		//				{
		//					exit = true;

		//				}

		//				if (exit) continue;

		//				if (e.getNext().isActive())
		//				{
		//					e = e.getNext();
		//					boundaryVertices.push_back(e.getVertex());
		//					boundaryHalfEdges.push_back(e);

		//					if (fixedVerticesBoolean[e.getVertex().getId()])
		//					{
		//						exit = true;
		//					}

		//				}
		//				else exit = true;


		//			} while (e != start && !exit);
		//		}

		//		if (boundaryHalfEdges.size() > 1)
		//		{

		//			
		//			zVector p;
		//			for (auto &he : boundaryHalfEdges)
		//			{
		//				guideEdge_globalVertex[he.getEdge().getId()][0] = n_gV;
		//				guideEdge_globalVertex[he.getEdge().getId()][1] = 0;
		//			}



		//			for (auto &v : boundaryVertices)
		//			{						
		//				zVector pos = v.getPosition();
		//				p += pos;
		//			}



		//			p /= boundaryVertices.size();
		//			
		//			// project p to top plane
		//			double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
		//			zPoint projected_p = (p - (up * minDist));

		//			globalVertices.push_back(zGlobalVertex());
		//			globalVertices[n_gV].pos = projected_p;;
		//			n_gV++;

		//		}
		//	}
		//

		//}

		//printf("\n globalVertices %i ", globalVertices.size());

		////create plane mesh
		//planeVertex_globalVertex.clear();
		//guideVertex_planeFace.clear();
		//guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		//guideEdge_planeFace.clear();
		//guideEdge_planeFace.assign(fnGuideMesh.numEdges(), -1);

		//zColorArray faceColors;

		//// add face per non constrained guide vertex 
		//for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		//{
		//	if (!fixedVerticesBoolean[v.getId()])
		//	{
		//		if (v.onBoundary())
		//		{
		//			zItMeshEdgeArray cEdges;
		//			v.getConnectedEdges(cEdges);

		//			if (cEdges.size() == 3)
		//			{
		//				int count = 0;
		//				for (auto& cE : cEdges)
		//				{
		//					if (cE.onBoundary())
		//					{
		//						// add face vertex 
		//						if (count == 1)
		//						{
		//							zIntArray eFaces;
		//							cE.getFaces(eFaces);

		//							int cF = eFaces[0];
		//							int gV = guideFace_globalVertex[cF][0];

		//							globalVertices[gV].coincidentVertices.push_back(positions.size());
		//							planeVertex_globalVertex.push_back(gV);

		//							pConnects.push_back(positions.size());
		//							positions.push_back(globalVertices[gV].pos);

		//							planeVWeights.push_back(1.0); // top point

		//						}

		//						// add edge vertex

		//						int gV = guideEdge_globalVertex[cE.getId()][0];

		//						globalVertices[gV].coincidentVertices.push_back(positions.size());
		//						planeVertex_globalVertex.push_back(gV);

		//						pConnects.push_back(positions.size());								
		//						positions.push_back(globalVertices[gV].pos);

		//						planeVWeights.push_back(1.0); // top point

		//						// add face vertex 
		//						if (count == 0)
		//						{
		//							zIntArray eFaces;
		//							cE.getFaces(eFaces);

		//							int cF = eFaces[0];
		//							int gV = guideFace_globalVertex[cF][0];

		//							globalVertices[gV].coincidentVertices.push_back(positions.size());
		//							planeVertex_globalVertex.push_back(gV);

		//							pConnects.push_back(positions.size());									
		//							positions.push_back(globalVertices[gV].pos);

		//							planeVWeights.push_back(1.0); // top point

		//							count++;

		//						}

		//					}
		//				}

		//				guideVertex_planeFace[v.getId()] = pCounts.size();
		//				pCounts.push_back(4);

		//				faceColors.push_back(zColor(1, 0, 0, 1));
		//			}
		//		}
		//		else
		//		{
		//			zIntArray cFaces;
		//			v.getConnectedFaces(cFaces);

		//			for (auto& cF : cFaces)
		//			{
		//				int gV = guideFace_globalVertex[cF][0];

		//				globalVertices[gV].coincidentVertices.push_back(positions.size());
		//				planeVertex_globalVertex.push_back(gV);

		//				pConnects.push_back(positions.size());
		//				positions.push_back(globalVertices[gV].pos);

		//				planeVWeights.push_back(1.0); // top point

		//			}

		//			guideVertex_planeFace[v.getId()] = pCounts.size();
		//			pCounts.push_back(cFaces.size());
		//			faceColors.push_back(zColor(1, 0, 0, 1));
		//		}
		//	}

		//}


		//// add face per non constrained guide edge 
		//for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		//{
		//	if (e.onBoundary())
		//	{
		//		zIntArray eVerts;
		//		e.getVertices(eVerts);

		//		if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;

		//		zIntArray eFaces;
		//		e.getFaces(eFaces);

		//		//0
		//		int gV = guideFace_globalVertex[eFaces[0]][1];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);
		//						
		//		planeVWeights.push_back(1.0); // bottom point

		//		//1
		//		gV = guideFace_globalVertex[eFaces[0]][0];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);

		//		planeVWeights.push_back(1.0); // top point

		//		//2
		//		gV = guideEdge_globalVertex[e.getId()][0];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);

		//		planeVWeights.push_back(1.0); // top point

		//		//3
		//		//gV = guideEdge_globalVertex[e.getId()][1];
		//		//globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		//planeVertex_globalVertex.push_back(gV);
		//		//pConnects.push_back(positions.size());
		//		//positions.push_back(globalVertices[gV].pos);

		//		//planeVWeights.push_back(1.0); // bottom point

		//		guideEdge_planeFace[e.getId()] = pCounts.size();
		//		pCounts.push_back(3);

		//		faceColors.push_back(zColor(0, 1, 0, 1));

		//	}

		//	else
		//	{
		//		zIntArray eFaces;
		//		e.getFaces(eFaces);

		//		//0
		//		int gV = guideFace_globalVertex[eFaces[0]][1];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);

		//		planeVWeights.push_back(1.0); // bottom point

		//		//1
		//		gV = guideFace_globalVertex[eFaces[0]][0];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);

		//		planeVWeights.push_back(1.0); // top point

		//		//2
		//		gV = guideFace_globalVertex[eFaces[1]][0];
		//		globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		planeVertex_globalVertex.push_back(gV);
		//		pConnects.push_back(positions.size());
		//		positions.push_back(globalVertices[gV].pos);

		//		planeVWeights.push_back(1.0); // top point

		//		//3
		//		//gV = guideFace_globalVertex[eFaces[1]][1];
		//		//globalVertices[gV].coincidentVertices.push_back(positions.size());
		//		//planeVertex_globalVertex.push_back(gV);
		//		//pConnects.push_back(positions.size());
		//		//positions.push_back(globalVertices[gV].pos);

		//		//planeVWeights.push_back(1.0); // bottom point


		//		guideEdge_planeFace[e.getId()] = pCounts.size();
		//		pCounts.push_back(3);

		//		faceColors.push_back(zColor(0, 1, 0, 1));
		//	}

		//}

		//zFnMesh fnCutPlanes(o_planeMesh);
		//fnCutPlanes.create(positions, pCounts, pConnects);

		////fnCutPlanes.setFaceColors(faceColors);

	}

	ZSPACE_INLINE void zTsSDFBridge::createBlockSectionGraphsfromFiles(string fileDir, zFileTpye type)
	{
		zStringArray graphFiles;
		coreUtils.getFilesFromDirectory(graphFiles, fileDir, zJSON);

		for (auto& fileName : graphFiles)
		{

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::createFieldMesh(zDomain<zPoint>& bb, int res)
	{
		zFnMeshScalarField fnField(o_field);

		fnField.create(bb.min, bb.max, res, res, 1, true, false);
		
		
	}

	//---- COMPUTE METHODS

	ZSPACE_INLINE bool zTsSDFBridge::equilibrium(bool& computeTargets, float guideWeight, double dT, zIntergrationType type, int numIterations, double angleTolerance, bool colorEdges, bool printInfo)
	{
		// compute  equilibrium targets
		if (computeTargets)
		{
			computeEquilibriumTargets(guideWeight);			

			computeTargets = !computeTargets;
		}

		// update meshes

		if (guideWeight != 1.0)
		{
			updateGuideMesh(0.1, dT, type, numIterations);
		}

		if (guideWeight != 0.0)
		{
			updatePlaneMesh( dT, type, numIterations);
		}

		// check deviations
		zDomainFloat dev;
		bool out = checkDeviation(dev, angleTolerance, colorEdges, printInfo);

		return out;
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zTsSDFBridge::setGuideMesh(zObjMesh& _o_guideMesh)
	{
		o_guideMesh = &_o_guideMesh;

		// set boundary constraints
		setConstraints();
	}

	ZSPACE_INLINE void zTsSDFBridge::setConstraints(const vector<int>& _fixedVertices)
	{
		if (_fixedVertices.size() == 0)
		{
			fixedVertices.clear();

			for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
			{
				if (v.onBoundary())
				{
					fixedVertices.push_back(v.getId());
				}
			}
		}
		else
		{
			fixedVertices.clear();
			fixedVertices = _fixedVertices;
		}


		fixedVerticesBoolean.clear();
		guideVWeights.clear();

		zFnMesh fnGuideMesh(*o_guideMesh);

		fixedVerticesBoolean.assign(fnGuideMesh.numVertices(), false);
		guideVWeights.assign(fnGuideMesh.numVertices(), 1.0);

		fnGuideMesh.setVertexColor(zColor(1, 1, 1, 1));

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			zItMeshVertex v(*o_guideMesh, fixedVertices[i]);
			zColor col(0,0,0,1);
			v.setColor(col);
			fixedVerticesBoolean[fixedVertices[i]] = true;		

			guideVWeights[fixedVertices[i]] = 0.0;
		}

		//---

		excludeFacesBoolean.clear();
		excludeFacesBoolean.assign(fnGuideMesh.numPolygons(), false);

		//for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		//{
		//	if (e.onBoundary())
		//	{

		//		int numConstraintVerts = 0; 

		//		//if (fixedVerticesBoolean[e.getHalfEdge(0).getVertex().getId()]) numConstraintVerts++;
		//		//if (fixedVerticesBoolean[e.getHalfEdge(1).getVertex().getId()]) numConstraintVerts++;

		//		//if (numConstraintVerts == 0 || numConstraintVerts == 1)
		//		//{
		//			zIntArray eFaces;
		//			e.getFaces(eFaces);

		//			for (auto& f : eFaces) excludeFacesBoolean[f] = true;
		//		//}

		//	}
		//	
		//}

		for (zItMeshFace f(*o_guideMesh); !f.end(); f++)
		{
			if (f.onBoundary())
			{
				 excludeFacesBoolean[f.getId()] = true;

			}

		}

	}


	//---- GET METHODS

	ZSPACE_INLINE zObjMesh* zTsSDFBridge::getRawCutPlaneMesh()
	{
		return &o_planeMesh;
	}	

	ZSPACE_INLINE zObjMesh* zTsSDFBridge::getRawSmoothGuideMesh()
	{
		return &o_guideSmoothMesh;
	}

	ZSPACE_INLINE int zTsSDFBridge::getCorrespondingPlaneFace(int guideEdgeId)
	{
		int fId_plane = -1;
		//int id = guideEdgeId * 2;
		//if (guideHalfEdge_planeFace[id] != -1)
		//{
		//	// plane face
		//	fId_plane = guideHalfEdge_planeFace[id];

		//	zItMeshHalfEdge he_guide(*o_guideMesh, guideEdgeId);

		//	zVector he_guide_vec = he_guide.getVector();
		//	he_guide_vec.normalize();

		//	zVector fTarget = targetNormals_cutplane[fId_plane];

		//	cout << " \n e " << guideEdgeId << " : " << he_guide_vec;
		//	cout << " \n f " << fId_plane << " : " << fTarget;
		//	cout << " \n ";
		//}

		return  fId_plane;
	}

	ZSPACE_INLINE zPointArray zTsSDFBridge::getBlockIntersectionPoints(int blockId)
	{
		if (blockId >= blocks.size() ) return zPointArray();

		return blocks[blockId].intersectionPoints;

	}

	ZSPACE_INLINE vector<zTransform> zTsSDFBridge::getBlockFrames(int blockId)
	{

		if (blockId >= blocks.size()) return vector<zTransform>();

		return blocks[blockId].sectionFrames;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= blocks.size()) return out;

		numGraphs = blocks[blockId].o_sectionGraphs.size();

		if(numGraphs == 0 )return out;
		
		for (auto& graph : blocks[blockId].o_sectionGraphs)
		{
			out.push_back(&graph);
		}
				
		return out;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockContourGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= blocks.size()) return out;

		numGraphs = blocks[blockId].o_contourGraphs.size();

		if (numGraphs == 0)return out;

		for (auto& graph : blocks[blockId].o_contourGraphs)
		{			
			out.push_back(&graph);
		}

		return out;
	}

	ZSPACE_INLINE zObjMeshScalarField* zTsSDFBridge::getRawFieldMesh()
	{
		
		return &o_field;
	}

	ZSPACE_INLINE	zObjMesh* zTsSDFBridge::getRawIsocontour()
	{
		//zFnMeshScalarField fn(o_field);
		//fn.updateColors();
		
		//fn.getIsocontour(o_isoContour, 0.5);

		return &o_isoMesh;
	}

	//---- COMPUTE METHODS

	ZSPACE_INLINE void zTsSDFBridge::computeEquilibriumTargets(float guideWeight)
	{
		//targetHalfEdges_guide.clear();
		//targetNormals_cutplane.clear();

		//zFnMesh fnPlaneMesh(o_planeMesh);
		//zFnMesh fnGuideMesh(*o_guideMesh);

		//targetNormals_cutplane.assign(fnPlaneMesh.numPolygons(), zVector());
		//targetHalfEdges_guide.assign(fnGuideMesh.numHalfEdges(), zVector());

		//// target per guide edge
		//for (zItMeshHalfEdge he_guide(*o_guideMesh); !he_guide.end(); he_guide++)
		//{
		//	int i = he_guide.getId();

		//	//guide edge
		//	int heId_guide = he_guide.getId();
		//	zVector he_guide_vec = he_guide.getVector();
		//	he_guide_vec.normalize();

		//	if (guideHalfEdge_planeFace[i] != -1)
		//	{
		//		// plane face
		//		int fId_plane = guideHalfEdge_planeFace[i];
		//		zItMeshFace f_plane(o_planeMesh, fId_plane);

		//		zVector f_plane_norm = f_plane.getNormal();
		//		f_plane_norm.normalize();

		//		// target edge 
		//		zVector he_target = (he_guide_vec * guideWeight) + (f_plane_norm * (1 - guideWeight));
		//		he_target.normalize();

		//		targetHalfEdges_guide[i] = he_target;
		//		targetNormals_cutplane[fId_plane] = he_target;

		//	}
		//	else
		//	{
		//		// target edge 
		//		zVector he_target = (he_guide_vec * 1);
		//		targetHalfEdges_guide[i] = (he_target);
		//	}
		//}

		//// target per guide vertex
		//for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
		//{
		//	int i = v_guide.getId();

		//	if (guideVertex_planeFace[i] != -1)
		//	{
		//		// plane face
		//		int fId_plane = guideVertex_planeFace[i];
		//		zItMeshFace f_plane(o_planeMesh, fId_plane);

		//		zVector f_plane_norm = f_plane.getNormal();
		//		f_plane_norm.normalize();

		//		// target normal 
		//		zVector he_target(0, 0, 1);
		//		targetNormals_cutplane[fId_plane] = he_target;
		//	}

		//}

	}

	ZSPACE_INLINE void zTsSDFBridge::computeMedialEdgesfromConstraints()
	{

		guide_MedialEdges.clear();
		guide_MedialEdgesBoolean.clear();

		zFnMesh fn_guideMesh(*o_guideMesh);
		guide_MedialEdgesBoolean.assign(fn_guideMesh.numEdges(), false);

		// strart from constraint and walk till its hits another constraint
		//discard boundary edge walks

		for (auto fV : fixedVertices)
		{
			zItMeshVertex v(*o_guideMesh, fV);

			zItMeshHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			zItMeshHalfEdge start, he;

			// get halfedge of internal edge
			bool internalEdge = false;
			for (auto& cHe : cHEdges)
			{
				//if (fixedVerticesBoolean[cHe.getVertex().getId()] && fixedVerticesBoolean[cHe.getStartVertex().getId()])
				//	continue;
				if (!cHe.getEdge().onBoundary())
				{
					start = cHe;
					he = cHe;
					internalEdge = true;
					
					break;
				}
			}
				
			if (!internalEdge) continue;
						
			bool exit = false;
			
			// walk to find medial edges
			do
			{
				if (fixedVerticesBoolean[he.getVertex().getId()]) exit = true;				

				if (!guide_MedialEdgesBoolean[he.getEdge().getId()])
				{
					guide_MedialEdgesBoolean[he.getEdge().getId()] = true;

					guide_MedialEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;								

			} while (!exit);
		}

		// Add boundary edge to list
		/*zItMeshHalfEdge boundaryStart;

		for(zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{
			if (he.onBoundary())
			{
				boundaryStart = he;
				break;
			}
		}

		zItMeshHalfEdge he = boundaryStart;
		do
		{
			if (!guide_MedialEdgesBoolean[he.getEdge().getId()])
			{
				if (he.getVertex().getPosition().z - 0.083 < 0.01 && he.getStartVertex().getPosition().z - 0.083 < 0.01)
				{

				}
				else
				{
					guide_MedialEdgesBoolean[he.getEdge().getId()] = true;

					guide_MedialEdges.push_back(he.getEdge().getId());
				}
				
			}

			he = he.getNext();

		} while (he != boundaryStart);*/


		fn_guideMesh.setEdgeColor(zColor());

		for (auto mEdge : guide_MedialEdges)
		{
			zItMeshEdge e(*o_guideMesh, mEdge.first);
			zColor col(1, 0.5, 0, 1);
			e.setColor(col);
		}



		// SMOOTH MESH 

		guideSmooth_MedialEdges.clear();
		guideSmooth_MedialEdgesBoolean.clear();

		zFnMesh fn_guideSmoothMesh(o_guideSmoothMesh);
		guideSmooth_MedialEdgesBoolean.assign(fn_guideSmoothMesh.numEdges(), false);

		// start from constraint and walk till its hits another constraint
		//discard boundary edge walks
		for (auto fV : fixedVertices)
		{
			zItMeshVertex v(o_guideSmoothMesh, fV);

			zItMeshHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			zItMeshHalfEdge start, he;

			// get halfedge of internal edge
			bool internalEdge = false;
			for (auto& cHe : cHEdges)
			{
				

				if (!cHe.getEdge().onBoundary())
				{
					start = cHe;
					he = cHe;

					internalEdge = true;

					break;
				}
			}

			if (!internalEdge) continue;

			bool exit = false;

			// walk to find medial edges
			do
			{
				if (he.getVertex().getId() < fn_guideMesh.numVertices())
				{
					if (fixedVerticesBoolean[he.getVertex().getId()]) exit = true;
				}
				

				if (!guideSmooth_MedialEdgesBoolean[he.getEdge().getId()])
				{

					guideSmooth_MedialEdgesBoolean[he.getEdge().getId()] = true;

					guideSmooth_MedialEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}

		// Add boundary edge to list
		 
		/*for (zItMeshHalfEdge he(o_guideSmoothMesh); !he.end(); he++)
		{
			if (he.onBoundary())
			{
				boundaryStart = he;
				break;
			}
		}

		he = boundaryStart;
		do
		{
			if (!guideSmooth_MedialEdgesBoolean[he.getEdge().getId()])
			{
				if (he.getVertex().getPosition().z - 0.083 < 0.01 && he.getStartVertex().getPosition().z - 0.083 < 0.01)
				{

				}
				else
				{
					guideSmooth_MedialEdgesBoolean[he.getEdge().getId()] = true;
					guideSmooth_MedialEdges.push_back(he.getEdge().getId());
				}
				
			}

			he = he.getNext();

		} while (he != boundaryStart);*/

		fn_guideSmoothMesh.setEdgeColor(zColor());

		for (auto mEdge : guideSmooth_MedialEdges)
		{
			zItMeshEdge e(o_guideSmoothMesh, mEdge.first);
			zColor col(1, 0.5, 0, 1);
			e.setColor(col);
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlocks(float printLayerDepth)
	{
		zFnMesh  fn_guideMesh(*o_guideMesh);
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);
		zFnMesh fn_planeMesh(o_planeMesh);

		blocks.clear();
		blocks.assign(fn_guideMesh.numEdges(), zBlock());

		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{

			if (!guide_MedialEdgesBoolean[e.getId()]) continue;

			// set index
			int blockId = e.getId();

			blocks[blockId].id = blockId;
		}

		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			
			if (!guide_MedialEdgesBoolean[e.getId()]) continue;		

			// index
			int blockId = e.getId();

			// set faces
			computeBlockFaces_fromEdge(blocks[blockId], e);

			//bool computeMacro = true;

			//if (e.getHalfEdge(0).getStartVertex().getId() == 237 && e.getHalfEdge(0).getVertex().getId() == 202) computeMacro = true;
			//if (e.getHalfEdge(0).getStartVertex().getId() == 202 && e.getHalfEdge(0).getVertex().getId() == 237) computeMacro = true;

			//if (computeMacro)
			//{
				computeMacroBlockIndices(blocks[blockId], e);

				//printf(" |  e%i : m %i ", e.getId(), blocks[blockId].macroBlocks.size());

				//for (auto mB : blocks[blockId].macroBlocks)
				//{
				//	zItMeshEdge tE(*o_guideMesh, mB);
				//	//printf("\n te% i : v %i %i ", tE.getId(), tE.getHalfEdge(0).getStartVertex().getId(), tE.getHalfEdge(0).getVertex().getId());
				//}
			//}
			
		}

	
		computeMacroBlock();
		
	}

	ZSPACE_INLINE void zTsSDFBridge::computeSDF()
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;


			if (b.id != 398) continue;

			b.o_contourGraphs.clear();
			b.o_contourGraphs.assign(b.o_sectionGraphs.size(), zObjGraph());

			for (int i = 1; i <4 /*b.o_contourGraphs.size()*/; i++)
			{
				computeBalustradeSDF(b, i);
			}
			
		}
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zTsSDFBridge::updateGuideMesh( float minmax_Edge, float dT, zIntergrationType type, int numIterations)
	{
		zFnMesh fnGuideMesh(*o_guideMesh);

		if (fnGuideParticles.size() != fnGuideMesh.numVertices())
		{
			fnGuideParticles.clear();
			o_guideParticles.clear();


			for (int i = 0; i < fnGuideMesh.numVertices(); i++)
			{
				bool fixed = false;

				zObjParticle p;
				p.particle = zParticle(o_guideMesh->mesh.vertexPositions[i], fixed);
				o_guideParticles.push_back(p);

			}

			for (int i = 0; i < o_guideParticles.size(); i++)
			{
				fnGuideParticles.push_back(zFnParticle(o_guideParticles[i]));
			}

		}

		vector<zVector> v_residual;
		v_residual.assign(fnGuideMesh.numVertices(), zVector());

		vector<double> edgelengths;
		fnGuideMesh.getEdgeLengths(edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = coreUtils.zMin(edgelengths);
		maxEdgeLength = coreUtils.zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;

		zVector* positions = fnGuideMesh.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
			{
				int i = v.getId();
				zItMeshHalfEdgeArray cHEdges;
				v.getConnectedHalfEdges(cHEdges);

				zVector v_i = positions[v.getId()];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (auto& he : cHEdges)
				{

					zVector v_j = positions[he.getVertex().getId()];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetHalfEdges_guide[he.getId()];
					t_ij.normalize();

					t_ij = (t_ij.angle(e_ij) > 90) ? t_ij * -1 : t_ij;

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cHEdges.size();

				// compute residue force				
				v_residual[i] = (b_i - v_i);

				zVector forceV = v_residual[i] * guideVWeights[i];
				fnGuideParticles[i].addForce(forceV);
			}

			// update positions
			for (int i = 0; i < fnGuideParticles.size(); i++)
			{
				fnGuideParticles[i].integrateForces(dT, type);
				fnGuideParticles[i].updateParticle(true);
			}
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::updatePlaneMesh(float dT, zIntergrationType type, int numIterations)
	{
		zFnMesh fnPlaneMesh(o_planeMesh);
		
		// create particles if it doesnt exist
		if (fnPlaneParticles.size() != fnPlaneMesh.numVertices())
		{
			fnPlaneParticles.clear();
			o_planeParticles.clear();


			for (int i = 0; i < fnPlaneMesh.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(o_planeMesh.mesh.vertexPositions[i], fixed);
				o_planeParticles.push_back(p);

			}

			for (int i = 0; i < o_planeParticles.size(); i++)
			{
				fnPlaneParticles.push_back(zFnParticle(o_planeParticles[i]));
			}
		}

		// update
		vector<zVector> v_residual;
		v_residual.assign(fnPlaneMesh.numVertices(), zVector());

		zVector* positions = fnPlaneMesh.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			
			for (auto& gV : globalVertices)
			{		

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);

				for (auto& cV : gV.coincidentVertices)
				{
					int i = cV;
					zVector v_i = positions[i];

					zItMeshVertex v(o_planeMesh, i);
					
					zItMeshFaceArray cFaces;
					v.getConnectedFaces(cFaces);

					int j = cFaces[0].getId();

					zVector p_norm = targetNormals_cutplane[j];
					zPoint p_ori = cFaces[0].getCenter();

					// based on https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
					double minDist = coreUtils.minDist_Point_Plane(v_i, p_ori, p_norm);
					b_i += (v_i - (p_norm * minDist));

				}

				b_i /= gV.coincidentVertices.size();

				for (auto& cV : gV.coincidentVertices)
				{
					int i = cV;
					zVector v_i = positions[i];

					// compute residue force
					v_residual[i] = b_i - v_i;

					zVector forceV = v_residual[i] * planeVWeights[i];
					fnPlaneParticles[i].addForce(forceV);
				}
			}

			// update positions
			for (int i = 0; i < fnPlaneParticles.size(); i++)
			{
				fnPlaneParticles[i].integrateForces(dT, type);
				fnPlaneParticles[i].updateParticle(true);
			}
			
		}

		fnPlaneMesh.computeMeshNormals();
	}

	ZSPACE_INLINE bool zTsSDFBridge::checkDeviation(zDomainFloat& deviation, float angleTolerance, bool colorEdges, bool printInfo)
	{
		//bool out = true;

		//vector<double> deviations;
		//deviation.min = 10000;
		//deviation.max = -10000;

		//zFnMesh fnPlaneMesh(o_planeMesh);
		//deviations.assign(fnPlaneMesh.numPolygons(), -1);

		//for (zItMeshHalfEdge he_guide(*o_guideMesh); !he_guide.end(); he_guide++)
		//{
		//	int i = he_guide.getId(); 

		//	//guide edge
		//	int heId_guide = he_guide.getId();
		//	zVector he_guide_vec = he_guide.getVector();
		//	he_guide_vec.normalize();

		//	if (guideHalfEdge_planeFace[i] != -1)
		//	{
		//		// plane face
		//		int fId_plane = guideHalfEdge_planeFace[i];
		//		zItMeshFace f_plane(o_planeMesh, fId_plane);

		//		zVector f_plane_norm = f_plane.getNormal();
		//		f_plane_norm.normalize();

		//		// angle
		//		double a_i =  he_guide_vec.angle(f_plane_norm) ;
		//		a_i = coreUtils.zMin(a_i, 180 -a_i);

		//		deviations[fId_plane] = (a_i);

		//		if (a_i > angleTolerance)
		//		{
		//			out = false;
		//		}

		//		if (a_i < deviation.min) deviation.min = a_i;
		//		if (a_i > deviation.max) deviation.max = a_i;
		//	}			
		//}

		//for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
		//{
		//	int i = v_guide.getId();

		//	if (guideVertex_planeFace[i] != -1)
		//	{
		//		// plane face
		//		int fId_plane = guideVertex_planeFace[i];
		//		zItMeshFace f_plane(o_planeMesh, fId_plane);

		//		// target normal 
		//		zVector e_target(0, 0, 1);

		//		zVector f_plane_norm = f_plane.getNormal();
		//		f_plane_norm.normalize();

		//		// angle
		//		double a_i = e_target.angle(f_plane_norm);
		//		a_i = coreUtils.zMin(a_i, 180 - a_i);
		//		
		//		deviations[fId_plane] = (a_i);

		//		if (a_i > angleTolerance)
		//		{
		//			out = false;
		//		}

		//		if (a_i < deviation.min) deviation.min = a_i;
		//		if (a_i > deviation.max) deviation.max = a_i;
		//	}

		//}

		//if (printInfo)
		//{
		//	printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
		//}

		//if (colorEdges)
		//{
		//	zDomainColor colDomain(zColor(180, 1, 1), zColor(360, 1, 1));

		//	for (zItMeshEdge e_guide(*o_guideMesh); !e_guide.end(); e_guide++)
		//	{
		//		int i = e_guide.getId() * 2;

		//		if (guideHalfEdge_planeFace[i] != -1)
		//		{
		//			zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

		//			// plane face
		//			int fId_plane = guideHalfEdge_planeFace[i];
		//			if (deviations[fId_plane] < angleTolerance) col = zColor();

		//			e_guide.setColor(col);
		//			
		//			zItMeshFace f_plane(o_planeMesh, fId_plane);
		//			f_plane.setColor(col);
		//		}
		//	}

		//	for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
		//	{
		//		int i = v_guide.getId();

		//		if (guideVertex_planeFace[i] != -1)
		//		{
		//			zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);
		//			
		//			// plane face
		//			int fId_plane = guideVertex_planeFace[i];

		//			if (deviations[fId_plane] < angleTolerance) col = zColor();

		//			zItMeshFace f_plane(o_planeMesh, fId_plane);
		//			f_plane.setColor(col);
		//		}
		//	}

		//}

		//return out;
	}

	ZSPACE_INLINE bool zTsSDFBridge::planarisePlaneMesh(zDomainFloat& deviation,float dT, zIntergrationType type, float tolerance, int numIterations)
	{
		zFnMesh fnPlaneMesh(o_planeMesh);
		

		// create particles if it doesnt exist
		if (fnPlaneParticles.size() != fnPlaneMesh.numVertices())
		{
			fnPlaneParticles.clear();
			o_planeParticles.clear();


			for (int i = 0; i < fnPlaneMesh.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(o_planeMesh.mesh.vertexPositions[i], fixed);
				o_planeParticles.push_back(p);

			}

			for (int i = 0; i < o_planeParticles.size(); i++)
			{
				fnPlaneParticles.push_back(zFnParticle(o_planeParticles[i]));
			}
		}

		// update

		vector<zIntArray> fTris;
		zPointArray fCenters; 
		zDoubleArray fVolumes;
		zVectorArray fNormalTargets;
		zVectorArray fNormals;
		
		fnPlaneMesh.getMeshTriangles(fTris);

		vector<zVector> v_residual;
		v_residual.assign(fnPlaneMesh.numVertices(), zVector());

		zVector* positions = fnPlaneMesh.getRawVertexPositions();
		//zVector* fNormals = fnPlaneMesh.getRawFaceNormals();

		for (int k = 0; k < numIterations; k++)
		{
			fCenters.clear();
			fnPlaneMesh.getMeshFaceVolumes(fTris, fCenters, fVolumes, true);

			fNormalTargets.clear();
			fNormals.clear();
			fnPlaneMesh.getFaceNormals(fNormals);
			fNormalTargets = fNormals;
			
			int fCounter = 0;
			for (auto fPair : planeFace_targetPair)
			{
				if (fPair != -1)
				{
					fNormalTargets[fCounter] = (fNormals[fCounter] + fNormals[fPair]) * 0.5;
					fNormalTargets[fCounter].normalize();
				}
				fCounter++;
			
			}

			for (zItMeshFace f(o_planeMesh); !f.end(); f++)
			{
				int i = f.getId();

				if (fVolumes[i] > tolerance)
				{
					vector<int> fVerts;
					f.getVertices(fVerts);										

					for (int j = 0; j < fVerts.size(); j++)
					{
						double dist = coreUtils.minDist_Point_Plane(positions[fVerts[j]], fCenters[i], fNormalTargets[i] /*fNormals[i]*/);
						zVector pForce = /*fNormals[i]*/ fNormalTargets[i] * dist * -1.0;
						
						int gV = planeVertex_globalVertex[fVerts[j]];

						for (auto coincidentV : globalVertices[gV].coincidentVertices)
						{
							fnPlaneParticles[coincidentV].addForce(pForce);
						}						

					}
				}
				
			}
					

			// update positions
			for (int i = 0; i < fnPlaneParticles.size(); i++)
			{
				fnPlaneParticles[i].integrateForces(dT, type);
				fnPlaneParticles[i].updateParticle(true);
			}

			fnPlaneMesh.computeMeshNormals();
		}


		bool out = true;

		zFloatArray deviations;
		
		fCenters.clear();
		fnPlaneMesh.getMeshFaceVolumes(fTris, fCenters, fVolumes, true);

		deviation.min = coreUtils.zMin(fVolumes);
		deviation.max = coreUtils.zMax(fVolumes);

		if (deviation.max > tolerance) out = false;

		zDomainColor colDomain(zColor(180, 1, 1), zColor(360, 1, 1));

		for (zItMeshFace f(o_planeMesh); !f.end(); f++)
		{
			int i = f.getId();

			
			zColor col = coreUtils.blendColor(fVolumes[i], deviation, colDomain, zHSV);

			if (fVolumes[i] < tolerance) col = zColor(120,1,1);
			f.setColor(col);
			
		}

		printf("\n  tolerance : %1.4f minDeviation : %1.6f , maxDeviation: %1.6f ", tolerance, deviation.min, deviation.max);

		return out;
		
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockFaces_fromVertex(zBlock& _block, zItMeshVertex& guideMesh_vertex)
	{
		//if (fixedVerticesBoolean[guideMesh_vertex.getId()]) return;
		
		if (guideVertex_planeFace[guideMesh_vertex.getId()] != -1) _block.faces.push_back(guideVertex_planeFace[guideMesh_vertex.getId()]);

		zItMeshHalfEdgeArray cHEdges;
		guideMesh_vertex.getConnectedHalfEdges(cHEdges);

		for (auto& cHe : cHEdges)
		{
			if (guideHalfEdge_planeFace[cHe.getId()].size() != 0)
			{
				for(auto f: guideHalfEdge_planeFace[cHe.getId()])	_block.faces.push_back(f);
			}

			//if (guide_MedialEdgesBoolean[cHe.getEdge().getId()]) _block.sectionFaces.push_back(guideHalfEdge_planeFace[cHe.getId()]);
			//(guide_MedialEdgesBoolean[cHe.getEdge().getId()]) ?	_block.sectionFace.push_back(true) : _block.sectionFace.push_back(false);
		
		}
				
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockFaces_fromEdge(zBlock& _block, zItMeshEdge& guideMesh_edge)
	{
		zItMeshHalfEdge cHe = guideMesh_edge.getHalfEdge(0);
		if (guideHalfEdge_planeFace[cHe.getId()].size() != 0)
		{
			for (auto f : guideHalfEdge_planeFace[cHe.getId()])	_block.faces.push_back(f);
		}

		cHe = guideMesh_edge.getHalfEdge(1);
		if (guideHalfEdge_planeFace[cHe.getId()].size() != 0)
		{
			for (auto f : guideHalfEdge_planeFace[cHe.getId()])	_block.faces.push_back(f);
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::computeMacroBlockIndices(zBlock& _block, zItMeshEdge& guideMesh_edge)
	{
		zItMeshHalfEdge cHe = guideMesh_edge.getHalfEdge(0);
		zColor currentFaceColor = cHe.getFace().getColor();

		// left
		if (!cHe.getNext().getNext().getEdge().onBoundary())
		{
			zItMeshHalfEdge temp = cHe;
			temp = temp.getNext().getNext().getSym();
			temp = temp.getNext().getNext();

			zColor tempFaceColor = temp.getFace().getColor();

			if (currentFaceColor == tempFaceColor)
				addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);				
			
		}
		
		if (!cHe.getVertex().onBoundary())
		{
			// top
			zItMeshHalfEdge temp = cHe;
			temp = temp.getNext().getSym().getNext();
			zColor tempFaceColor = temp.getFace().getColor();

			if (currentFaceColor == tempFaceColor)
				_block.macroBlocks.push_back(temp.getEdge().getId());
			

			// top left

			if (!temp.getNext().getNext().getEdge().onBoundary())
			{
				temp = temp.getNext().getNext().getSym();
				temp = temp.getNext().getNext();
				tempFaceColor = temp.getFace().getColor();

				if (currentFaceColor == tempFaceColor)			
					addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);
			}
		}
		

		// bottom
		if (!cHe.getStartVertex().onBoundary())
		{
			zItMeshHalfEdge temp = cHe;
			temp = temp.getPrev().getSym().getPrev();

			zColor tempFaceColor = temp.getFace().getColor();

			if (currentFaceColor == tempFaceColor)
				addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);

			// bottom left

			if (!temp.getNext().getNext().getEdge().onBoundary())
			{
				temp = temp.getNext().getNext().getSym();
				temp = temp.getNext().getNext();
				tempFaceColor = temp.getFace().getColor();

				if (currentFaceColor == tempFaceColor)
					addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);
			}
		}


		////// SYM Edge

		cHe = guideMesh_edge.getHalfEdge(1);
		currentFaceColor = cHe.getFace().getColor();

		// left
		if (!cHe.getNext().getNext().getEdge().onBoundary())
		{
			zItMeshHalfEdge temp = cHe;
			temp = temp.getNext().getNext().getSym();
			temp = temp.getNext().getNext();

			zColor tempFaceColor = temp.getFace().getColor();

			if (currentFaceColor == tempFaceColor)
				addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);
		}

		if (!cHe.getVertex().onBoundary())
		{
			// top
			zItMeshHalfEdge temp = cHe;
			temp = temp.getNext().getSym().getNext();
			zColor tempFaceColor = temp.getFace().getColor();			


			// top left

			if (!temp.getNext().getNext().getEdge().onBoundary())
			{
				temp = temp.getNext().getNext().getSym();
				temp = temp.getNext().getNext();
				tempFaceColor = temp.getFace().getColor();

				if (currentFaceColor == tempFaceColor)
					addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);
			}
		}


		// bottom
		if (!cHe.getStartVertex().onBoundary())
		{
			zItMeshHalfEdge temp = cHe;
			temp = temp.getPrev().getSym().getPrev();
			zColor tempFaceColor = temp.getFace().getColor();		

			// bottom left

			if (!temp.getNext().getNext().getEdge().onBoundary())
			{
				temp = temp.getNext().getNext().getSym();
				temp = temp.getNext().getNext();
				tempFaceColor = temp.getFace().getColor();

				if (currentFaceColor == tempFaceColor)
					addBlocktoMacro(_block, blocks[temp.getEdge().getId()]);
			}
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computeMacroBlock()
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			

			zIntArray mBoundaryFaces;

			blockVisited[b.id] = true;

			zItMeshEdge guideMesh_edge(*o_guideMesh, b.id);

			zItMeshFaceArray eFaces;
			guideMesh_edge.getFaces(eFaces);

			for (auto& eF : eFaces)
			{
				zItMeshHalfEdgeArray fHEdges;
				eF.getHalfEdges(fHEdges);

				zColor fColor = eF.getColor();

				for (auto& he : fHEdges)
				{
					if (he.getEdge().onBoundary()) continue;

					zColor symFColor = he.getSym().getFace().getColor();

					if (fColor == symFColor) continue;

					if (he.getId() == guideMesh_edge.getHalfEdge(0).getNext().getId()
						|| he.getId() == guideMesh_edge.getHalfEdge(0).getPrev().getId()
						|| he.getId() == guideMesh_edge.getHalfEdge(1).getPrev().getId()
						|| he.getId() == guideMesh_edge.getHalfEdge(1).getNext().getId()
						)
					{
						bool chekRepeat = false;
						for (auto bF : b.macroBlock_BoundaryFaces)
						{
							if (bF == guideHalfEdge_planeFace[he.getId()][0]) chekRepeat = true;
						}

						if (!chekRepeat) b.macroBlock_BoundaryFaces.push_back(guideHalfEdge_planeFace[he.getId()][0]);
					}
					else
					{
						bool chekRepeat = false;
						for (auto bF : b.macroBlock_sideFaces)
						{
							if (bF == guideHalfEdge_planeFace[he.getId()][0]) chekRepeat = true;
						}

						if (!chekRepeat) b.macroBlock_sideFaces.push_back(guideHalfEdge_planeFace[he.getId()][0]);
					}
					

					
				}
			}

			// for other blocks 
			for (int mB : b.macroBlocks)
			{
				blockVisited[mB] = true;

				zItMeshEdge guideMesh_edge1(*o_guideMesh, blocks[mB].id);

				zItMeshFaceArray eFaces;
				guideMesh_edge1.getFaces(eFaces);

				for (auto& eF : eFaces)
				{
					zItMeshHalfEdgeArray fHEdges;
					eF.getHalfEdges(fHEdges);

					zColor fColor = eF.getColor();

					for (auto &he : fHEdges)
					{
						if (he.getEdge().onBoundary()) continue;

						zColor symFColor = he.getSym().getFace().getColor();

						if (fColor == symFColor) continue;
						if (he.getId() == guideMesh_edge1.getHalfEdge(0).getNext().getId()
							|| he.getId() == guideMesh_edge1.getHalfEdge(0).getPrev().getId()
							|| he.getId() == guideMesh_edge1.getHalfEdge(1).getPrev().getId()
							|| he.getId() == guideMesh_edge1.getHalfEdge(1).getNext().getId()
							)
						{
							bool chekRepeat = false;
							for (auto bF : b.macroBlock_BoundaryFaces)
							{
								if (bF == guideHalfEdge_planeFace[he.getId()][0]) chekRepeat = true;
							}

							if (!chekRepeat) b.macroBlock_BoundaryFaces.push_back(guideHalfEdge_planeFace[he.getId()][0]);
						}
					
						else
						{
							bool chekRepeat = false;
							for (auto bF : b.macroBlock_sideFaces)
							{
								if (bF == guideHalfEdge_planeFace[he.getId()][0]) chekRepeat = true;
							}

							if (!chekRepeat) b.macroBlock_sideFaces.push_back(guideHalfEdge_planeFace[he.getId()][0]);
						}
						
					}
				}

			}

			//printf("\n %i | %i %i | %i ", b.id, guideMesh_edge.getHalfEdge(0).getStartVertex().getId(), guideMesh_edge.getHalfEdge(0).getVertex().getId(), b.macroBlock_BoundaryFaces.size());

			computeBlockIntersections(b, guideMesh_edge);
			///if (b.id == 253)
			//{
				//printf("\n ");
				//for (auto bF : b.macroBlock_BoundaryFaces) printf(" %i ", bF);
				computeBlockFrames(b, 0.03);
			//}
			//for (int mB : b.macroBlocks) blockVisited[mB] = true;
			

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockIntersections(zBlock& _block, zItMeshVertex& guideMesh_V)
	{
		//if (fixedVerticesBoolean[guideMesh_V.getId()]) return;

		zFnMesh  fn_guideMesh(*o_guideMesh);
		int blockId = _block.id;

		// start vertex
		

		zItMeshVertex guideSmoothMesh_V(o_guideSmoothMesh, guideMesh_V.getId());

		zItMeshHalfEdgeArray cHEdges;
		guideSmoothMesh_V.getConnectedHalfEdges(cHEdges);
		for (auto& cHe : cHEdges)
		{
			zItMeshHalfEdge walkHe = cHe;

			// walk along medial edges till it intersects a block face
			if (guideSmooth_MedialEdgesBoolean[cHe.getEdge().getId()])
			{
				bool exit = false;

				do
				{
					if (walkHe.getVertex().getId() < fn_guideMesh.numVertices())
					{
						if (fixedVerticesBoolean[walkHe.getVertex().getId()]) exit = true;
					}

					zPoint eE = walkHe.getVertex().getPosition();
					zPoint eS = walkHe.getStartVertex().getPosition();

					bool planeIntersection = false;
					zPoint intersectionPoint;

					for (auto fId : _block.faces)
					{
						zItMeshFace f(o_planeMesh, fId);

						zVector fNorm = f.getNormal();
						zVector fCenter = f.getCenter();

						planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, fNorm, fCenter, intersectionPoint);

						if (planeIntersection)
						{
							zIntPair p(fId, walkHe.getId());
							_block.sectionPlaneFace_GuideSmoothEdge.push_back(p);
							_block.intersectionPoints.push_back(intersectionPoint);

							exit = true;
							break;
						}
					}

					walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();


				} while (!exit);

			}


		}	
		
		
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockIntersections(zBlock& _block, zItMeshEdge& guideMesh_edge)
	{
		zFnMesh  fn_guideMesh(*o_guideMesh);
		int blockId = _block.id;

		zItMeshHalfEdge he = guideMesh_edge.getHalfEdge(0);

		zItMeshVertex startVertex = he.getStartVertex();
		zItMeshVertex endVertex = he.getVertex();
		
		if (!he.getNext().getVertex().onBoundary())
		{
			zColor f1 = he.getFace().getColor();
			zColor f2 = he.getNext().getNext().getSym().getFace().getColor();

			if (f1 == f2)
			{
				startVertex = he.getNext().getNext().getVertex();
				endVertex = he.getNext().getNext().getStartVertex();
			}
		}

		if (!he.getSym().getPrev().getStartVertex().onBoundary())
		{
			zColor f1 = he.getSym().getFace().getColor();
			zColor f2 = he.getSym().getPrev().getPrev().getSym().getFace().getColor();

			if (f1 == f2)
			{
				startVertex = he.getSym().getPrev().getPrev().getStartVertex();
				endVertex = he.getSym().getPrev().getPrev().getVertex();
			}
		}

		zVector guideEdge = endVertex.getPosition() - startVertex.getPosition();
		
		//printf(" sV : %i , eV %i ", startVertex.getId(), endVertex.getId());

		zItMeshVertex guideSmoothMesh_V(o_guideSmoothMesh, startVertex.getId());

		zItMeshHalfEdgeArray cHEdges;
		zItMeshHalfEdge walkHe;
		float angle = 179;

		guideSmoothMesh_V.getConnectedHalfEdges(cHEdges);
		for (auto& cHe : cHEdges)
		{
			if (cHe.getVector().angle(guideEdge) < angle)
			{
				walkHe = cHe;
				angle = cHe.getVector().angle(guideEdge);
			}
		}

		//printf(" | smooth_sV : %i , smmoth_eV %i ", walkHe.getStartVertex().getId(), walkHe.getVertex().getId());
		zItMeshHalfEdge walkHe_sym = walkHe.getSym();;
		
		walkHe = walkHe.getNext().getSym().getNext();
		// walk along medial edges till it intersects a block face
		bool exit = false;

		do
		{
			if (walkHe.getVertex().onBoundary())
			{
				if (fixedVerticesBoolean[walkHe.getVertex().getId()]) exit = true;
			}

			//if (_block.intersectionPoints.size() >= 2)exit = true;


			zPoint eE = walkHe.getVertex().getPosition();
			zPoint eS = walkHe.getStartVertex().getPosition();

			bool planeIntersection = false;
			zPoint intersectionPoint;

			for (auto fId : _block.macroBlock_BoundaryFaces)
			{
				if (exit) continue;;
				zItMeshFace f(o_planeMesh, fId);

				zVector fNorm = f.getNormal();
				zVector fCenter = f.getCenter();

				planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, fNorm, fCenter, intersectionPoint);

				if (planeIntersection)
				{
					zIntPair p(fId, walkHe.getId());
					if (_block.intersectionPoints.size() == 1)  p = zIntPair(fId, walkHe.getSym().getId());
									
					int rId;
					bool chkRepeat = coreUtils.checkRepeatVector(intersectionPoint, _block.intersectionPoints, rId,2);
					
					if (!chkRepeat)
					{
						_block.sectionPlaneFace_GuideSmoothEdge.push_back(p);
						_block.intersectionPoints.push_back(intersectionPoint);
						exit = true;
						//printf("\n fid1 : %i ", fId);
					}
					

					//exit = true;
					//break;
				}
			}

			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();


		} while (!exit);

		// walk in reverse direction
		walkHe = walkHe_sym;
		//walkHe =  walkHe.getNext().getSym().getNext();

		exit = false;

		do
		{
			if (walkHe.getVertex().onBoundary())
			{
				if (fixedVerticesBoolean[walkHe.getVertex().getId()]) exit = true;
			}

			//if (_block.intersectionPoints.size() >= 2)exit = true;

			zPoint eE = walkHe.getVertex().getPosition();
			zPoint eS = walkHe.getStartVertex().getPosition();

			bool planeIntersection = false;
			zPoint intersectionPoint;

			for (auto fId : _block.macroBlock_BoundaryFaces)
			{
				if (exit) continue;;

				zItMeshFace f(o_planeMesh, fId);

				zVector fNorm = f.getNormal();
				zVector fCenter = f.getCenter();

				planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, fNorm, fCenter, intersectionPoint);

				if (planeIntersection)
				{
					zIntPair p(fId, walkHe.getId());					

					int rId;
					bool chkRepeat = coreUtils.checkRepeatVector(intersectionPoint, _block.intersectionPoints, rId,2);

					if (!chkRepeat)
					{
						_block.sectionPlaneFace_GuideSmoothEdge.push_back(p);
						_block.intersectionPoints.push_back(intersectionPoint);
						exit = true;

						//printf("\n fid2 : %i ", fId);
					}
					
					//
					//break;
				}
			}

			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();


		} while (!exit);

			
		

		//printf(" | intersections : %i  ", _block.intersectionPoints.size());

	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockFrames(zBlock& _block, float printLayerDepth)
	{
		if (_block.id == -1) return;

		// quad or triangle block
		if (_block.intersectionPoints.size() == 2)
		{
			zItMeshHalfEdge startHe (o_guideSmoothMesh,  _block.sectionPlaneFace_GuideSmoothEdge[0].second);
			zItMeshHalfEdge endHe(o_guideSmoothMesh, _block.sectionPlaneFace_GuideSmoothEdge[1].second);

			// compute total edge length
			float length = 0;

			length += startHe.getStartVertex().getPosition().distanceTo(_block.intersectionPoints[0]);
			length += endHe.getStartVertex().getPosition().distanceTo(_block.intersectionPoints[1]);

			zItMeshHalfEdge walkHe = startHe;
			walkHe = walkHe.getSym();
			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

			bool exit = false;
			while(walkHe != endHe)
			{
				length += walkHe.getLength();

				walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
			} 

			_block.mg_Length = length;
			
			int numLayers = floor(length / printLayerDepth);

			zItMeshFace startF(o_planeMesh, _block.sectionPlaneFace_GuideSmoothEdge[0].first);
			zItMeshFace endF(o_planeMesh, _block.sectionPlaneFace_GuideSmoothEdge[1].first);

			zVector startFNorm = startF.getNormal() * -1;
			zVector endFNorm = endF.getNormal();

			float angleStep = endFNorm.angle(startFNorm) / numLayers;
						
			

			float xStep = (endFNorm.x - startFNorm.x) / numLayers;
			float yStep = (endFNorm.y - startFNorm.y) / numLayers;
			float zStep = (endFNorm.z - startFNorm.z) / numLayers;

			//cout << "\n start : " << startFNorm;
			//cout << "\n end : " << endFNorm;

			//printf("\n step %1.6f  %1.6f  %1.6f \n ", xStep, yStep, zStep);

			// compute frames
			walkHe = startHe;
			walkHe = walkHe.getSym();

			// Start point
			float tLen = 0.1;
			zPoint O = _block.intersectionPoints[0];
			zVector Z = startFNorm;

			zVector tempZ = Z;
			tempZ.normalize();
			tempZ *= tLen;
					
			
			zVector X, Y;

			zItMeshFaceArray eFaces;
			walkHe.getEdge().getFaces(eFaces);
			
			Y = zVector();
			for (auto& eF : eFaces)	Y += eF.getNormal();
			Y /= eFaces.size();
			Y.normalize();
			Y *= -1;
			Y *= tLen;


			X = Z ^ Y;
			X.normalize();
			X *= tLen;

			zTransform sFrame = setTransformFromVectors(O, X, Y, tempZ);
			_block.sectionFrames.push_back(sFrame);

			
			// in between points
			zPoint pOnCurve = O;
			for (int j = 0; j < numLayers ; j++)
			{
				zPoint eEndPoint = walkHe.getVertex().getPosition();
				float distance_increment = printLayerDepth;

				if (pOnCurve.distanceTo(eEndPoint) < printLayerDepth)
				{
					distance_increment = printLayerDepth - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
				}

				zVector he_vec = walkHe.getVector();
				he_vec.normalize();
				
				//O
				O = pOnCurve + he_vec * distance_increment;

				//Y
				Y = zVector();
				eFaces.clear();
				walkHe.getEdge().getFaces(eFaces);

				for (auto& eF : eFaces)	Y += eF.getNormal();
				Y /= eFaces.size();
				Y.normalize();
				Y *= -1;
				Y *= tLen;

				// Z 
				
				//Z = Z.rotateAboutAxis(Y, angleStep);
				Z.x += xStep;
				Z.y += yStep;
				Z.z += zStep;
				
				zVector tempZ = Z;
				tempZ.normalize();
				tempZ *= tLen;

				//X
				X = Z ^ Y;
				X.normalize();
				X *= tLen;

				// add frame
				sFrame = setTransformFromVectors(O, X, Y, tempZ);
				_block.sectionFrames.push_back(sFrame);

				pOnCurve = O;

				//cout << "\n current : " << Z;

			}

			// end point
			//cout << "\n dist " << _block.intersectionPoints[1].distanceTo(O);
			O = _block.intersectionPoints[1];
			Z = endFNorm;
			
			tempZ = Z;
			tempZ.normalize();
			tempZ *= tLen;

			Y = zVector();
			eFaces.clear();
			walkHe.getEdge().getFaces(eFaces);

			for (auto& eF : eFaces)	Y += eF.getNormal();
			Y /= eFaces.size();
			Y.normalize();
			Y *= -1;
			Y *= tLen;

			X = Z ^ Y;
			X.normalize();
			X *= tLen;

			// add frame
			sFrame = setTransformFromVectors(O, X, Y, tempZ);
			_block.sectionFrames.push_back(sFrame);

			//for (auto& sf : _block.sectionFrames) cout << "\n " << sf << "\n";

			printf("\n block interior %i : l %1.2f  : numF %i ", _block.id, _block.mg_Length, _block.sectionFrames.size());
		}


		// ground block
		if (_block.intersectionPoints.size() == 1)
		{
			zItMeshHalfEdge startHe(o_guideSmoothMesh, _block.sectionPlaneFace_GuideSmoothEdge[0].second);
			//zItMeshHalfEdge endHe(o_guideSmoothMesh, _block.sectionPlaneFace_GuideSmoothEdge[1].second);
								

			// compute total edge length
			float length = 0;

			length += startHe.getStartVertex().getPosition().distanceTo(_block.intersectionPoints[0]);
			

			zItMeshHalfEdge walkHe = startHe;
			walkHe = walkHe.getSym();
			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

			bool exit = false;
			do
			{
				if (walkHe.getVertex().onBoundary()) exit = true;

				length += walkHe.getLength();

				if(!exit) walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
			} while (!exit);
			//length += walkHe.getLength();

			zItMeshHalfEdge endHe = walkHe;

			_block.mg_Length = length;

			int numLayers = floor(length / printLayerDepth);

			zItMeshFace startF(o_planeMesh, _block.sectionPlaneFace_GuideSmoothEdge[0].first);
			
			zVector startFNorm = startF.getNormal() * -1;
			zVector endFNorm (0,0,-1);

			float angleStep = endFNorm.angle(startFNorm) / numLayers;



			float xStep = (endFNorm.x - startFNorm.x) / numLayers;
			float yStep = (endFNorm.y - startFNorm.y) / numLayers;
			float zStep = (endFNorm.z - startFNorm.z) / numLayers;

			//cout << "\n start : " << startFNorm;
			//cout << "\n end : " << endFNorm;

			//printf("\n step %1.6f  %1.6f  %1.6f \n ", xStep, yStep, zStep);

			// compute frames
			walkHe = startHe;
			walkHe = walkHe.getSym();

			// Start point
			float tLen = 0.1;
			zPoint O = _block.intersectionPoints[0];
			zVector Z = startFNorm;

			zVector tempZ = Z;
			tempZ.normalize();
			tempZ *= tLen;


			zVector X, Y;

			zItMeshFaceArray eFaces;
			walkHe.getEdge().getFaces(eFaces);

			Y = zVector();
			for (auto& eF : eFaces)	Y += eF.getNormal();
			Y /= eFaces.size();
			Y.normalize();
			Y *= -1;
			Y *= tLen;


			X = Z ^ Y;
			X.normalize();
			X *= tLen;

			zTransform sFrame = setTransformFromVectors(O, X, Y, tempZ);
			_block.sectionFrames.push_back(sFrame);


			// in between points
			zPoint pOnCurve = O;
			for (int j = 0; j < numLayers; j++)
			{
				zPoint eEndPoint = walkHe.getVertex().getPosition();
				float distance_increment = printLayerDepth;

				if (pOnCurve.distanceTo(eEndPoint) < printLayerDepth)
				{
					distance_increment = printLayerDepth - pOnCurve.distanceTo(eEndPoint);
					pOnCurve = eEndPoint;

					walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
				}

				zVector he_vec = walkHe.getVector();
				he_vec.normalize();

				//O
				O = pOnCurve + he_vec * distance_increment;

				//Y
				Y = zVector();
				eFaces.clear();
				walkHe.getEdge().getFaces(eFaces);

				for (auto& eF : eFaces)	Y += eF.getNormal();
				Y /= eFaces.size();
				Y.normalize();
				Y *= -1;
				Y *= tLen;

				// Z 

				//Z = Z.rotateAboutAxis(Y, angleStep);
				Z.x += xStep;
				Z.y += yStep;
				Z.z += zStep;

				zVector tempZ = Z;
				tempZ.normalize();
				tempZ *= tLen;

				//X
				X = Z ^ Y;
				X.normalize();
				X *= tLen;

				// add frame
				sFrame = setTransformFromVectors(O, X, Y, tempZ);
				_block.sectionFrames.push_back(sFrame);

				pOnCurve = O;

				//cout << "\n current : " << Z;

			}

			// end point
			//cout << "\n dist " << _block.intersectionPoints[1].distanceTo(O);
			O = endHe.getVertex().getPosition();
			Z = endFNorm;

			tempZ = Z;
			tempZ.normalize();
			tempZ *= tLen;

			Y = zVector();
			eFaces.clear();
			walkHe.getEdge().getFaces(eFaces);

			for (auto& eF : eFaces)	Y += eF.getNormal();
			Y /= eFaces.size();
			Y.normalize();
			Y *= -1;
			Y *= tLen;

			X = Z ^ Y;
			X.normalize();
			X *= tLen;

			// add frame
			sFrame = setTransformFromVectors(O, X, Y, tempZ);
			_block.sectionFrames.push_back(sFrame);

			//for (auto& sf : _block.sectionFrames) cout << "\n " << sf << "\n";

			printf("\n block ground %i : l %1.2f  : numF %i ", _block.id, _block.mg_Length, _block.sectionFrames.size());
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computeBalustradeSDF(zBlock& _block, int graphId)
	{
		if (_block.id == -1) return;
		if (graphId >= _block.o_sectionGraphs.size())return;

		zFnGraph fnGraph(_block.o_sectionGraphs[graphId]);
		zPoint* positions = fnGraph.getRawVertexPositions();

		zTransform t = _block.sectionFrames[graphId];
		fnGraph.setTransform(t, true, false);

		zPoint o(t(3, 0), t(3, 1), t(3, 2));
		zVector n(t(2, 0), t(2, 1), t(2, 2));

		/*for (int i = 0; i < fnGraph.numVertices(); i++)
		{			

			double d = coreUtils.minDist_Point_Plane(positions[i], o, n);
			printf("\n %i %1.2f ", i, d);
		}*/

		

		// clip planes

		zPoint v_startPos = positions[0];
		int faceId = -1;
		float dist = 10000;
		for (auto sF : _block.macroBlock_sideFaces)
		{
			zItMeshFace f(o_planeMesh, sF);

			

			double d = coreUtils.minDist_Point_Plane(v_startPos, o, n);

			if (d < dist)
			{
				dist = d; 
				faceId = sF;
			}
		}

		zVector fNorm;
		zPoint fCen;
		if (faceId != -1)
		{
			zItMeshFace f(o_planeMesh, faceId);
			f.setColor(zColor(0, 1, 0, 1));			

			zTransformationMatrix from;
			from.setTransform(_block.sectionFrames[graphId], true);

			zTransform transform = from.getLocalMatrix();

			zPointArray fVertPos;
			f.getVertexPositions(fVertPos);

			zItMeshVertexArray fVerts;
			f.getVertices(fVerts);

			for (auto& v : fVerts)
			{
				zPoint vPos = v.getPosition();

				zPoint newPos = vPos * transform;		

				v.setPosition(newPos);
			}		

			f.updateNormal();
			fNorm = f.getNormal();
			//fNorm *= -1;

			fCen = f.getCenter();		

			// revert to original position

			int counter = 0;
			for (auto& v : fVerts)
			{	
				v.setPosition(fVertPos[counter]);
				counter++;
			}

			f.updateNormal();
		}


		// Transform
			
		
		zTransform tLocal;
		tLocal.setIdentity();
		fnGraph.setTransform(tLocal, true, true);
		

		// vertex Iterators
		zItGraphVertex v_start(_block.o_sectionGraphs[graphId], 0);
		zItGraphVertex v_end(_block.o_sectionGraphs[graphId], fnGraph.numVertices() - 1);
				

		zItGraphHalfEdge he_start = v_start.getHalfEdge().getSym();;
		zVector vec = he_start.getSym().getVector();
		vec.normalize();

		zItGraphHalfEdge he_end = v_end.getHalfEdge().getSym();

		zPoint p0 = v_start.getPosition() + vec * 0.25;

		zItGraphHalfEdge he = he_start;
		he = he.getNext();
		he = he.getNext();
		he = he.getNext();
		he = he.getNext();
		he = he.getNext();
		

		zVector avgVec = (he_end.getVector() + he_start.getVector()) * 0.5;
		avgVec.normalize();

		zPoint p1 = he.getVertex().getPosition() + (avgVec * 0.2);
		zPoint p2 = he.getVertex().getPosition() + (avgVec * 0.05 * -1);		
		zPoint p3 = v_end.getPosition();

		he = he.getNext().getNext();
		zPoint p4 = he.getVertex().getPosition() + (avgVec * 0.05 * -1);

		cout << "\n p0: " << p0;
		cout << "\n p1: " << p1;
		cout << "\n p2: " << p2;
		cout << "\n p3: " << p3;
				
		
		// field
		zFnMeshScalarField fnField(o_field);

		// circle 
		zScalarArray circleField_top;
		fnField.getScalars_Circle(circleField_top, p3, 0.05, 0.0, false);

		zScalarArray circleField_side;
		fnField.getScalars_Circle(circleField_side, p1, 0.1, 0.0, false);

		// line 
		zScalarArray circleField_side1;
		fnField.getScalars_Circle(circleField_side1, p4, 0.05, 0.0, false);

		// triangle
	
		zScalarArray triangleField_0;
		fnField.getScalars_Triangle(triangleField_0, p0, p1, p2, 0.0, false);

		zScalarArray triangleField_1;
		fnField.getScalars_Triangle(triangleField_1, p3, p1, p2, 0.0, false);
		

		//cout << "\n " << coreUtils.zMin(triangleField) << " , " << coreUtils.zMax(triangleField);

		// BOOLEANS

		zScalarArray boolean_field1;
		fnField.boolean_union(triangleField_0, triangleField_1, boolean_field1, false);

		zScalarArray boolean_field2;
		fnField.boolean_union(boolean_field1, circleField_top, boolean_field2, false);	

		//zScalarArray boolean_field3;
		//fnField.boolean_union(boolean_field2, circleField_side1, boolean_field3, false);

		zScalarArray boolean_field4;
		fnField.boolean_subtract(boolean_field2, circleField_side, boolean_field4, false);
			
		// CLIP PLANES
		
		zScalarArray clipPlaneField;
		fnField.boolean_clipwithPlane(boolean_field4, clipPlaneField, fCen, fNorm);
				

		fnField.smoothField(clipPlaneField,2, 0.0, zSpace::zDiffusionType::zAverage);
				

		// CONTOURS
		fnField.setFieldValues(clipPlaneField);
		
		fnField.getIsocontour(_block.o_contourGraphs[graphId], 0.0);

		fnField.getIsolineMesh(o_isoMesh, 0.0);
		
		// transform back 
		fnGraph.setTransform(t, true, true);
		
		zFnGraph fnIsoGraph(_block.o_contourGraphs[graphId]);
		fnIsoGraph.setEdgeColor(zColor(0, 1, 0, 1));

		fnIsoGraph.setTransform(t, true, true);

	}

	ZSPACE_INLINE zTransform zTsSDFBridge::setTransformFromVectors(zPoint& O, zVector& X, zVector& Y, zVector& Z)
	{
		zTransform out;

		out(0, 0) = X.x; out(0, 1) = X.y; out(0, 2) = X.z; out(0, 3) = 1;
		out(1, 0) = Y.x; out(1, 1) = Y.y; out(1, 2) = Y.z; out(1, 3) = 1;
		out(2, 0) = Z.x; out(2, 1) = Z.y; out(2, 2) = Z.z; out(2, 3) = 1;
		out(3, 0) = O.x; out(3, 1) = O.y; out(3, 2) = O.z; out(3, 3) = 1;


		return out;
	}

	ZSPACE_INLINE void zTsSDFBridge::blockPlanesToTXT(string dir, string filename)
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			printf("\n %i ", b.id);
			// output file

			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(b.id);
			outfilename += ".txt";

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				break;
			}

			myfile << b.faces.size();
			for (int mB : b.macroBlocks)
			{
				myfile << "," << blocks[mB].faces.size();
			}

			myfile << "\n";

			int counter = 0;
			for (auto fId : b.faces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPoint O = f.getCenter();
				zVector N = f.getNormal();
				
				myfile << O.x << "," << O.y << "," << O.z << "," << N.x << "," << N.y << "," << N.z;

				if (counter != b.faces.size() - 1) myfile << "\n ";
				counter++;

			}

			blockVisited[b.id] = true;

			for (int mB : b.macroBlocks)
			{
				myfile << "\n ";
				int count = 0;
				for (auto fId :blocks[mB].faces)
				{
					zItMeshFace f(o_planeMesh, fId);

					zPoint O = f.getCenter();
					zVector N = f.getNormal();

					myfile << O.x << "," << O.y << "," << O.z << "," << N.x << "," << N.y << "," << N.z;

					if (count != blocks[mB].faces.size() - 1) myfile << "\n ";
					count++;

				}

				blockVisited[mB] = true;
			}

			myfile.close();

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockSidePlanesToTXT(string dir, string filename)
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			printf("\n %i ", b.id);
			// output file

			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(b.id);
			outfilename += ".txt";

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				break;
			}

			myfile << b.faces.size();
			for (int mB : b.macroBlocks)
			{
				myfile << "," << blocks[mB].faces.size();
			}

			myfile << "\n";

			int counter = 0;
			printf(" %i ", b.macroBlock_sideFaces.size());
			for (auto fId : b.macroBlock_sideFaces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPoint O = f.getCenter();
				zVector N = f.getNormal();

				myfile << O.x << "," << O.y << "," << O.z << "," << N.x << "," << N.y << "," << N.z;

				if (counter != b.macroBlock_sideFaces.size() - 1) myfile << "\n ";
				counter++;

			}

			blockVisited[b.id] = true;

			for (int mB : b.macroBlocks)
			{			
				blockVisited[mB] = true;
			}

			myfile.close();

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockSectionPlanesToTXT(string dir, string filename)
	{

		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;


			// output file

			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(b.id);
			outfilename += ".txt";

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				break;
			}

			int counter = 0;
			for (auto t : b.sectionFrames)
			{
				myfile << t(0, 0) << "," << t(0, 1) << "," << t(0, 2) << "," << t(1, 0) << "," << t(1, 1) << "," << t(1, 2) << "," << t(2,0) << "," << t(2, 1) << "," << t(2, 2) << "," << t(3, 0) << "," << t(3, 1) << "," << t(3, 2);

				if (counter != b.sectionFrames.size() -1 ) myfile<< "\n ";
				counter++;
			}

			blockVisited[b.id] = true;
			for (int mB : b.macroBlocks) blockVisited[mB] = true;

			myfile.close();

		}


	}

	ZSPACE_INLINE void zTsSDFBridge::blockSectionsFromJSON(string dir, string filename)
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			if (b.id != 398) continue;

			// input file

			string infilename = dir;
			infilename += "/";		
			infilename += to_string(b.id);
			infilename += "_";
			infilename += filename;
			infilename += "_";

			b.o_sectionGraphs.clear();
			b.o_sectionGraphs.assign(b.sectionFrames.size(), zObjGraph());
			
			for (int j = 0; j < b.sectionFrames.size(); j++)
			{
				string temp = infilename;
				temp += to_string(j);
				temp += ".json";

				zFnGraph fnGraph(b.o_sectionGraphs[j]);
				fnGraph.from(temp, zJSON, true);
			}

			blockVisited[b.id] = true;
			for (int mB : b.macroBlocks) blockVisited[mB] = true;
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToJSON(string dir, string filename)
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			if (b.id != 398) continue;


			// output file

			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(b.id);
			outfilename += "_";

			for (int j = 0; j < b.o_contourGraphs.size(); j++)
			{
				string temp = outfilename;
				temp += to_string(j);
				temp += ".json";

				zFnGraph fnGraph(b.o_contourGraphs[j]);
				fnGraph.to(temp, zJSON);
			}

			blockVisited[b.id] = true;
			for (int mB : b.macroBlocks) blockVisited[mB] = true;

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToIncr3D(string dir, string filename, float layerWidth)
	{
		zBoolArray blockVisited;
		blockVisited.assign(blocks.size(), false);

		for (auto& b : blocks)
		{
			if (b.id == -1) continue;
			if (blockVisited[b.id]) continue;

			if (b.id != 398) continue;

			// output file

			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(b.id);
			outfilename += ".h";

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				break;
			}

			float maxLayerHeight = 0;
			float minLayerHeight = 10000;

									
			int startVertexId = -1;
			zItGraphHalfEdge he;
			zItGraphVertex v;
			float totalLength = 0;
			for (int j = 0; j < b.o_contourGraphs.size() - 1; j++)
			{		
				
				myfile << "Layer " << j << "\n";
				myfile << "/* " << "\n";

				zFnGraph fnG(b.o_contourGraphs[j]);
				if (fnG.numVertices() == 0)
				{
					myfile << "*/ " << "\n";
					continue;
				}

				bool flip = false;

				if (startVertexId == -1) startVertexId = 0;
				else
				{
					zPoint startP_Prev = v.getPosition();
					float dist = 10000;

					zPoint* positions = fnG.getRawVertexPositions();
					for (int i =0; i< fnG.numVertices(); i++)
					{
						float d = positions[i].distanceTo(startP_Prev);
						
						if (d < dist)
						{
							dist = d;
							startVertexId = i;
						}
					}

					zItGraphVertex temp(b.o_contourGraphs[j], startVertexId);

					flip = (temp.getHalfEdge().getVector() * he.getVector() < 0) ? true : false;
				}

				v = zItGraphVertex(b.o_contourGraphs[j], startVertexId);
				v.setColor(zColor(0, 1, 0, 1));
				
				he = v.getHalfEdge();
				if (flip) he = he.getSym();

				zItGraphHalfEdge start = he;
				start.getVertex().setColor(zColor(0, 0, 1, 1));

				zVector norm(b.sectionFrames[j](2, 0), b.sectionFrames[j](2, 1), b.sectionFrames[j](2, 2));
				norm.normalize();

				zVector nextNorm(b.sectionFrames[j + 1](2, 0), b.sectionFrames[j + 1](2, 1), b.sectionFrames[j + 1](2, 2));
				zVector nextOrigin(b.sectionFrames[j + 1](3, 0), b.sectionFrames[j + 1](3, 1), b.sectionFrames[j + 1](3, 2));

				do
				{
					totalLength += he.getLength();
					zPoint p = he.getVertex().getPosition();

					zPoint p1 = p + norm * 1.0;

					zPoint intPt;
					bool check = coreUtils.line_PlaneIntersection(p, p1, nextNorm, nextOrigin, intPt);

					if (!check) printf("\n %i %i no Intersection ",j, he.getVertex().getId());

					float layerHeight = intPt.distanceTo(p);
					maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
					minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

					myfile << p.x << "," << p.y << "," << p.z << "," ;
					myfile << norm.x << "," << norm.y << "," << norm.z << ",";
					myfile << layerWidth << ",";
					myfile << layerHeight << "\n";
					

					he = he.getNext();

				} while (he != start);
				

				myfile << "*/ " << "\n";
			}

			myfile << "Attributes " << "\n";
			myfile << "/* " << "\n";
			myfile << "Units: meters " << "\n";
			myfile << "Block ID:  " << b.id << "\n";
			myfile << "max layer height: " << maxLayerHeight << "\n";
			myfile << "min layer height: " << minLayerHeight << "\n";
			myfile << "total print length: " << totalLength << "\n";
			myfile << "*/ " << "\n";

			myfile.close();

			blockVisited[b.id] = true;
			for (int mB : b.macroBlocks) blockVisited[mB] = true;
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::addBlocktoMacro(zBlock& _blockA, zBlock& _blockB)
	{
		// check duplicate
		bool repeat = false;
		for (auto mB : _blockA.macroBlocks)
		{
			if (mB == _blockB.id)
			{
				repeat = true;
				break;
			}
		}
		if (!repeat && _blockB.id != _blockA.id) _blockA.macroBlocks.push_back(_blockB.id);

		//// add to all existing macro list 

		//for (auto mB : _blockA.macroBlocks)
		//{

		//	bool chkRepeat = false;
		//	for (auto mB : blocks[mB].macroBlocks)
		//	{
		//		if (mB == blocks[mB].id)
		//		{
		//			chkRepeat = true;
		//			break;
		//		}
		//	}
		//	if (!chkRepeat && _blockB.id != blocks[mB].id) blocks[mB].macroBlocks.push_back(_blockB.id);
		//}

		//// add to current block t o block with tempedge id
		//repeat = false;
		//for (auto mB : _blockB.macroBlocks)
		//{
		//	if (mB == _blockA.id)
		//	{
		//		repeat = true;
		//		break;
		//	}
		//}
		//if (!repeat && _blockB.id != _blockA.id) _blockB.macroBlocks.push_back(_blockA.id);
	}



}