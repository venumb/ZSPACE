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
						
		}

		// add boundary edge centers to global vertices
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (e.onBoundary())
			{				

				zIntArray eVerts;
				e.getVertices(eVerts);

				if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]])
				{
					zIntArray vIds = { -1,-1 };
					guideEdge_globalVertex.push_back(vIds);					
					
				}
				else
				{
					zPoint p = e.getCenter();

					zItMeshFaceArray efaces;
					e.getFaces(efaces);
					zVector n = efaces[0].getNormal();

					zIntArray vIds = { n_gV,n_gV + 1 };
					guideEdge_globalVertex.push_back(vIds);

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
				}
				
			}
			else
			{
				zIntArray vIds = { -1,-1 };
				guideEdge_globalVertex.push_back(vIds);
			}

		}

		printf("\n globalVertices %i ", globalVertices.size());

		//create plane mesh
		planeVertex_globalVertex.clear();
		guideVertex_planeFace.clear();
		guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		guideEdge_planeFace.clear();
		guideEdge_planeFace.assign(fnGuideMesh.numEdges(), -1);

		zColorArray faceColors;

		// add face per non constrained guide vertex 
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
		//				for (auto &cE: cEdges)
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

		//						}

		//						// add edge vertex
		//														
		//						int gV = guideEdge_globalVertex[cE.getId()][0];

		//						globalVertices[gV].coincidentVertices.push_back(positions.size());
		//						planeVertex_globalVertex.push_back(gV);

		//						pConnects.push_back(positions.size());
		//						positions.push_back(globalVertices[gV].pos);

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

		//			for (auto &cF : cFaces)
		//			{
		//				int gV = guideFace_globalVertex[cF][0];

		//				globalVertices[gV].coincidentVertices.push_back(positions.size());						
		//				planeVertex_globalVertex.push_back(gV);

		//				pConnects.push_back(positions.size());
		//				positions.push_back(globalVertices[gV].pos);

		//			}

		//			guideVertex_planeFace[v.getId()] = pCounts.size();
		//			pCounts.push_back(cFaces.size());
		//			faceColors.push_back(zColor(1, 0, 0, 1));
		//		}
		//	}
		//	
		//}


		// add face per non constrained guide edge 
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (e.onBoundary())
			{
				zIntArray eVerts;
				e.getVertices(eVerts);

				if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;

				zIntArray eFaces;
				e.getFaces(eFaces);

				//0
				int gV = guideFace_globalVertex[eFaces[0]][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//1
				gV = guideFace_globalVertex[eFaces[0]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//2
				gV = guideEdge_globalVertex[e.getId()][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//3
				gV = guideEdge_globalVertex[e.getId()][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);


				guideEdge_planeFace[e.getId()] = pCounts.size();
				pCounts.push_back(4);

				faceColors.push_back(zColor(0, 1, 0, 1));

			}

			else
			{
				zIntArray eFaces;
				e.getFaces(eFaces);

				//0
				int gV = guideFace_globalVertex[eFaces[0]][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//1
				gV = guideFace_globalVertex[eFaces[0]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//2
				gV = guideFace_globalVertex[eFaces[1]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				//3
				gV = guideFace_globalVertex[eFaces[1]][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);
				

				guideEdge_planeFace[e.getId()] = pCounts.size();
				pCounts.push_back(4);

				faceColors.push_back(zColor(0, 1, 0, 1));
			}

		}

		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		//fnCutPlanes.setFaceColors(faceColors);

	}

	ZSPACE_INLINE void zTsSDFBridge::createDualPlaneMesh(double width)
	{

		zFnMesh fnGuideMesh(*o_guideMesh);

		
		zPoint minBB, maxBB;
		fnGuideMesh.getBounds(minBB, maxBB);
		zPoint top_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, maxBB.z) ;
		zPoint bottom_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, minBB.z);
		zVector up(0, 0, 1);

		globalVertices.clear();
		guideEdge_globalVertex.clear();
		guideFace_globalVertex.clear();

		planeVWeights.clear();

		zPointArray positions;
		zIntArray pCounts, pConnects;

		

		int n_gV = 0;

		// add bottom center to global vertices
		globalVertices.push_back(zGlobalVertex());
		globalVertices[n_gV].pos = bottom_center;;
		n_gV++;
					

		// add face centers to global vertices
		for (zItMeshFace f(*o_guideMesh); !f.end(); f++)
		{
			zPoint p = f.getCenter();
			zVector n = f.getNormal();
			
			zIntArray vIds = { n_gV, 0 };
			guideFace_globalVertex.push_back(vIds);


			// project p to top plane
			double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
			zPoint projected_p = (p - (up * minDist));

			globalVertices.push_back(zGlobalVertex());
			globalVertices[n_gV].pos = projected_p;;
			n_gV++;
				

		}

		// add boundary edge centers to global vertices
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			zIntArray vIds = { -1,-1 };
			guideEdge_globalVertex.push_back(vIds);

		}

		for (zItMeshEdge guide_e(*o_guideMesh); !guide_e.end(); guide_e++)
		{
			if (guide_e.onBoundary())
			{

				zIntArray eVerts;
				guide_e.getVertices(eVerts);

				// if both vertices of edge constrained continue
				if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;
			
				// if edge already visited continue
				if (guideEdge_globalVertex[guide_e.getId()][0] != -1) continue;
				
				zItMeshHalfEdge he = (guide_e.getHalfEdge(0).onBoundary()) ? guide_e.getHalfEdge(0) : guide_e.getHalfEdge(1);
				
				zItMeshHalfEdgeArray  boundaryHalfEdges;
				zItMeshVertexArray  boundaryVertices;

				zItMeshHalfEdge start = he;
				zItMeshHalfEdge e = he;
				
				bool exit = false;

				// walk prev
				do
				{

					if (fixedVerticesBoolean[e.getSym().getVertex().getId()])
					{

						exit = true;
						boundaryVertices.push_back(e.getSym().getVertex());
					}

					boundaryHalfEdges.push_back(e);
					boundaryVertices.push_back(e.getVertex());

					if (e.getPrev().isActive())
					{
						e = e.getPrev();
					}
					else exit = true;

				} while (e != start && !exit);

				// walk next 
				// checking if the prev walk has completed the full edge loop
				if (e != start)
				{
					bool exit = false;
					e = start;
					do
					{
						if (fixedVerticesBoolean[e.getVertex().getId()])
						{
							exit = true;

						}

						if (exit) continue;

						if (e.getNext().isActive())
						{
							e = e.getNext();
							boundaryVertices.push_back(e.getVertex());
							boundaryHalfEdges.push_back(e);

							if (fixedVerticesBoolean[e.getVertex().getId()])
							{
								exit = true;
							}

						}
						else exit = true;


					} while (e != start && !exit);
				}

				if (boundaryHalfEdges.size() > 1)
				{

					
					zVector p;
					for (auto &he : boundaryHalfEdges)
					{
						guideEdge_globalVertex[he.getEdge().getId()][0] = n_gV;
						guideEdge_globalVertex[he.getEdge().getId()][1] = 0;
					}



					for (auto &v : boundaryVertices)
					{						
						zVector pos = v.getPosition();
						p += pos;
					}



					p /= boundaryVertices.size();
					
					// project p to top plane
					double minDist = coreUtils.minDist_Point_Plane(p, top_center, up);
					zPoint projected_p = (p - (up * minDist));

					globalVertices.push_back(zGlobalVertex());
					globalVertices[n_gV].pos = projected_p;;
					n_gV++;

				}
			}
		

		}

		printf("\n globalVertices %i ", globalVertices.size());

		//create plane mesh
		planeVertex_globalVertex.clear();
		guideVertex_planeFace.clear();
		guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		guideEdge_planeFace.clear();
		guideEdge_planeFace.assign(fnGuideMesh.numEdges(), -1);

		zColorArray faceColors;

		// add face per non constrained guide vertex 
		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{
			if (!fixedVerticesBoolean[v.getId()])
			{
				if (v.onBoundary())
				{
					zItMeshEdgeArray cEdges;
					v.getConnectedEdges(cEdges);

					if (cEdges.size() == 3)
					{
						int count = 0;
						for (auto& cE : cEdges)
						{
							if (cE.onBoundary())
							{
								// add face vertex 
								if (count == 1)
								{
									zIntArray eFaces;
									cE.getFaces(eFaces);

									int cF = eFaces[0];
									int gV = guideFace_globalVertex[cF][0];

									globalVertices[gV].coincidentVertices.push_back(positions.size());
									planeVertex_globalVertex.push_back(gV);

									pConnects.push_back(positions.size());
									positions.push_back(globalVertices[gV].pos);

									planeVWeights.push_back(1.0); // top point

								}

								// add edge vertex

								int gV = guideEdge_globalVertex[cE.getId()][0];

								globalVertices[gV].coincidentVertices.push_back(positions.size());
								planeVertex_globalVertex.push_back(gV);

								pConnects.push_back(positions.size());								
								positions.push_back(globalVertices[gV].pos);

								planeVWeights.push_back(1.0); // top point

								// add face vertex 
								if (count == 0)
								{
									zIntArray eFaces;
									cE.getFaces(eFaces);

									int cF = eFaces[0];
									int gV = guideFace_globalVertex[cF][0];

									globalVertices[gV].coincidentVertices.push_back(positions.size());
									planeVertex_globalVertex.push_back(gV);

									pConnects.push_back(positions.size());									
									positions.push_back(globalVertices[gV].pos);

									planeVWeights.push_back(1.0); // top point

									count++;

								}

							}
						}

						guideVertex_planeFace[v.getId()] = pCounts.size();
						pCounts.push_back(4);

						faceColors.push_back(zColor(1, 0, 0, 1));
					}
				}
				else
				{
					zIntArray cFaces;
					v.getConnectedFaces(cFaces);

					for (auto& cF : cFaces)
					{
						int gV = guideFace_globalVertex[cF][0];

						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);

						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						planeVWeights.push_back(1.0); // top point

					}

					guideVertex_planeFace[v.getId()] = pCounts.size();
					pCounts.push_back(cFaces.size());
					faceColors.push_back(zColor(1, 0, 0, 1));
				}
			}

		}


		// add face per non constrained guide edge 
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (e.onBoundary())
			{
				zIntArray eVerts;
				e.getVertices(eVerts);

				if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;

				zIntArray eFaces;
				e.getFaces(eFaces);

				//0
				int gV = guideFace_globalVertex[eFaces[0]][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);
								
				planeVWeights.push_back(1.0); // bottom point

				//1
				gV = guideFace_globalVertex[eFaces[0]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				planeVWeights.push_back(1.0); // top point

				//2
				gV = guideEdge_globalVertex[e.getId()][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				planeVWeights.push_back(1.0); // top point

				//3
				//gV = guideEdge_globalVertex[e.getId()][1];
				//globalVertices[gV].coincidentVertices.push_back(positions.size());
				//planeVertex_globalVertex.push_back(gV);
				//pConnects.push_back(positions.size());
				//positions.push_back(globalVertices[gV].pos);

				//planeVWeights.push_back(1.0); // bottom point

				guideEdge_planeFace[e.getId()] = pCounts.size();
				pCounts.push_back(3);

				faceColors.push_back(zColor(0, 1, 0, 1));

			}

			else
			{
				zIntArray eFaces;
				e.getFaces(eFaces);

				//0
				int gV = guideFace_globalVertex[eFaces[0]][1];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				planeVWeights.push_back(1.0); // bottom point

				//1
				gV = guideFace_globalVertex[eFaces[0]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				planeVWeights.push_back(1.0); // top point

				//2
				gV = guideFace_globalVertex[eFaces[1]][0];
				globalVertices[gV].coincidentVertices.push_back(positions.size());
				planeVertex_globalVertex.push_back(gV);
				pConnects.push_back(positions.size());
				positions.push_back(globalVertices[gV].pos);

				planeVWeights.push_back(1.0); // top point

				//3
				//gV = guideFace_globalVertex[eFaces[1]][1];
				//globalVertices[gV].coincidentVertices.push_back(positions.size());
				//planeVertex_globalVertex.push_back(gV);
				//pConnects.push_back(positions.size());
				//positions.push_back(globalVertices[gV].pos);

				//planeVWeights.push_back(1.0); // bottom point


				guideEdge_planeFace[e.getId()] = pCounts.size();
				pCounts.push_back(3);

				faceColors.push_back(zColor(0, 1, 0, 1));
			}

		}

		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		//fnCutPlanes.setFaceColors(faceColors);

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

	}


	//---- GET METHODS

	ZSPACE_INLINE zObjMesh* zTsSDFBridge::getRawCutPlanelMesh()
	{
		return &o_planeMesh;
	}	

	ZSPACE_INLINE int zTsSDFBridge::getCorrespondingPlaneFace(int guideEdgeId)
	{
		int fId_plane = -1;
		if (guideEdge_planeFace[guideEdgeId] != -1)
		{
			// plane face
			fId_plane = guideEdge_planeFace[guideEdgeId];

			zItMeshEdge e_guide(*o_guideMesh, guideEdgeId);

			zVector e_guide_vec = e_guide.getVector();
			e_guide_vec.normalize();

			zVector fTarget = targetNormals_cutplane[fId_plane];

			cout << " \n e " << guideEdgeId << " : " << e_guide_vec;
			cout << " \n f " << fId_plane << " : " << fTarget;
			cout << " \n ";
		}

		return  fId_plane;
	}

	//---- COMPUTE METHODS

	ZSPACE_INLINE void zTsSDFBridge::computeEquilibriumTargets(float guideWeight)
	{
		targetEdges_guide.clear();
		targetNormals_cutplane.clear();

		zFnMesh fnPlaneMesh(o_planeMesh);
		zFnMesh fnGuideMesh(*o_guideMesh);

		targetNormals_cutplane.assign(fnPlaneMesh.numPolygons(), zVector());
		targetEdges_guide.assign(fnGuideMesh.numEdges(), zVector());

		// target per guide edge
		for (zItMeshEdge e_guide(*o_guideMesh); !e_guide.end(); e_guide++)
		{
			int i = e_guide.getId();

			//guide edge
			int eId_guide = e_guide.getId();
			zVector e_guide_vec = e_guide.getVector();
			e_guide_vec.normalize();

			if (guideEdge_planeFace[i] != -1)
			{
				// plane face
				int fId_plane = guideEdge_planeFace[i];
				zItMeshFace f_plane(o_planeMesh, fId_plane);

				zVector f_plane_norm = f_plane.getNormal();
				f_plane_norm.normalize();

				// target edge 
				zVector e_target = (e_guide_vec * guideWeight) + (f_plane_norm * (1 - guideWeight));
				e_target.normalize();

				targetEdges_guide[i] = (e_target.angle(e_guide_vec) < 90) ? e_target : e_target * -1;
				targetNormals_cutplane[fId_plane] = (e_target.angle(f_plane_norm) < 90) ? e_target : e_target * -1;

			}
			else
			{
				// target edge 
				zVector e_target = (e_guide_vec * 1);
				targetEdges_guide[i] = (e_target);
			}
		}

		// target per guide vertex
		for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
		{
			int i = v_guide.getId();

			if (guideVertex_planeFace[i] != -1)
			{
				// plane face
				int fId_plane = guideVertex_planeFace[i];
				zItMeshFace f_plane(o_planeMesh, fId_plane);

				zVector f_plane_norm = f_plane.getNormal();
				f_plane_norm.normalize();

				// target normal 
				zVector e_target(0, 0, 1);
				targetNormals_cutplane[fId_plane] = (e_target.angle(f_plane_norm) < 90) ? e_target : e_target * -1;
			}

		}

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

					guide_MedialEdges.push_back(he.getEdge().getId());
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;								

			} while (!exit);
		}

		fn_guideMesh.setEdgeColor(zColor());

		for (auto mEdge : guide_MedialEdges)
		{
			zItMeshEdge e(*o_guideMesh, mEdge);
			zColor col(1, 0.5, 0, 1);
			e.setColor(col);
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

					zVector t_ij = targetEdges_guide[he.getEdge().getId()];
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
		bool out = true;

		vector<double> deviations;
		deviation.min = 10000;
		deviation.max = -10000;

		zFnMesh fnPlaneMesh(o_planeMesh);
		deviations.assign(fnPlaneMesh.numPolygons(), -1);

		for (zItMeshEdge e_guide(*o_guideMesh); !e_guide.end(); e_guide++)
		{
			int i = e_guide.getId(); 

			//guide edge
			int eId_guide = e_guide.getId();
			zVector e_guide_vec = e_guide.getVector();
			e_guide_vec.normalize();

			if (guideEdge_planeFace[i] != -1)
			{
				// plane face
				int fId_plane = guideEdge_planeFace[i];
				zItMeshFace f_plane(o_planeMesh, fId_plane);

				zVector f_plane_norm = f_plane.getNormal();
				f_plane_norm.normalize();

				// angle
				double a_i =  e_guide_vec.angle(f_plane_norm) ;
				a_i = coreUtils.zMin(a_i, 180 -a_i);

				deviations[fId_plane] = (a_i);

				if (a_i > angleTolerance)
				{
					out = false;
				}

				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;
			}			
		}

		for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
		{
			int i = v_guide.getId();

			if (guideVertex_planeFace[i] != -1)
			{
				// plane face
				int fId_plane = guideVertex_planeFace[i];
				zItMeshFace f_plane(o_planeMesh, fId_plane);

				// target normal 
				zVector e_target(0, 0, 1);

				zVector f_plane_norm = f_plane.getNormal();
				f_plane_norm.normalize();

				// angle
				double a_i = e_target.angle(f_plane_norm);
				a_i = coreUtils.zMin(a_i, 180 - a_i);
				
				deviations[fId_plane] = (a_i);

				if (a_i > angleTolerance)
				{
					out = false;
				}

				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;
			}

		}

		if (printInfo)
		{
			printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
		}

		if (colorEdges)
		{
			zDomainColor colDomain(zColor(180, 1, 1), zColor(360, 1, 1));

			for (zItMeshEdge e_guide(*o_guideMesh); !e_guide.end(); e_guide++)
			{
				int i = e_guide.getId();

				if (guideEdge_planeFace[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

					// plane face
					int fId_plane = guideEdge_planeFace[i];
					if (deviations[fId_plane] < angleTolerance) col = zColor();

					e_guide.setColor(col);
					
					zItMeshFace f_plane(o_planeMesh, fId_plane);
					f_plane.setColor(col);
				}
			}

			for (zItMeshVertex v_guide(*o_guideMesh); !v_guide.end(); v_guide++)
			{
				int i = v_guide.getId();

				if (guideVertex_planeFace[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);
					
					// plane face
					int fId_plane = guideVertex_planeFace[i];

					if (deviations[fId_plane] < angleTolerance) col = zColor();

					zItMeshFace f_plane(o_planeMesh, fId_plane);
					f_plane.setColor(col);
				}
			}

		}

		return out;
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
		
		fnPlaneMesh.getMeshTriangles(fTris);

		vector<zVector> v_residual;
		v_residual.assign(fnPlaneMesh.numVertices(), zVector());

		zVector* positions = fnPlaneMesh.getRawVertexPositions();
		zVector* fNormals = fnPlaneMesh.getRawFaceNormals();

		for (int k = 0; k < numIterations; k++)
		{
			fCenters.clear();
			fnPlaneMesh.getMeshFaceVolumes(fTris, fCenters, fVolumes, true);

			for (zItMeshFace f(o_planeMesh); !f.end(); f++)
			{
				int i = f.getId();

				if (fVolumes[i] > tolerance)
				{
					vector<int> fVerts;
					f.getVertices(fVerts);										

					for (int j = 0; j < fVerts.size(); j++)
					{
						double dist = coreUtils.minDist_Point_Plane(positions[fVerts[j]], fCenters[i], fNormals[i]);
						zVector pForce = fNormals[i] * dist * -1.0;
						
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



}