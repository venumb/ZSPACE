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

	ZSPACE_INLINE void zTsSDFBridge::createEdgeBeamMesh(double width, const vector<int>& _constraintVertices)
	{
		zFnMesh fnGuideMesh(*o_guideMesh);

		globalVertices.clear();
		guideVertex_globalVertex.clear();
		guideEdge_globalVertex.clear();
		guideFace_globalVertex.clear();
		globalFixedVertices.clear();

		zPoint minBB, maxBB;
		fnGuideMesh.getBounds(minBB, maxBB);
		zPoint top_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, maxBB.z);
		zPoint bottom_center((minBB.x + maxBB.x) * 0.5, (minBB.y + maxBB.y) * 0.5, minBB.z);
		zVector up(0, 0, 1);


		zPointArray positions;
		zIntArray pCounts, pConnects;

		zBoolArray _constraintVerticesBoolean;
		_constraintVerticesBoolean.assign(fnGuideMesh.numVertices(), false);
		for (int i = 0; i < _constraintVertices.size(); i++)
		{
			_constraintVerticesBoolean[_constraintVertices[i]] = true;
		}


		int n_gV = 0;

		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{
			zPoint p = v.getPosition(); /*smoothPositions[v.getId()];*/
			zVector n = v.getNormal();
			zIntArray vIds = { n_gV,n_gV + 1 };
			guideVertex_globalVertex.push_back(vIds);


			globalVertices.push_back(zGlobalVertex());
			globalVertices[n_gV].pos = p + n * width * 0.5;;
			globalFixedVertices.push_back(false);
			n_gV++;


			globalVertices.push_back(zGlobalVertex());
			globalVertices[n_gV].pos = p - n * width * 0.5;;
			globalFixedVertices.push_back(false);
			n_gV++;

			if (_constraintVerticesBoolean[v.getId()])
			{
				double minDist = coreUtils.minDist_Point_Plane(globalVertices[n_gV - 2].pos, bottom_center, up);
				zPoint projected_p = (globalVertices[n_gV -2].pos - (up * minDist));
				globalVertices[n_gV - 2].pos = projected_p;;
				globalFixedVertices[n_gV - 2] = (true);
				
				minDist = coreUtils.minDist_Point_Plane(globalVertices[n_gV - 1].pos, bottom_center, up);
				zPoint projected_p1 = (globalVertices[n_gV - 1].pos - (up * minDist));
				globalVertices[n_gV - 1].pos = projected_p1;;
				globalFixedVertices[n_gV - 1] = (true);
			}
		}

		printf("\n globalVertices %i ", globalVertices.size());

		planeVertex_globalVertex.clear();
		guideVertex_planeFace.clear();
		guideVertex_planeFace.assign(fnGuideMesh.numVertices(), -1);

		guideHalfEdge_planeFace.clear();
		guideHalfEdge_planeFace.assign(fnGuideMesh.numHalfEdges(), zIntArray());

		zColorArray faceColors;

		for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{

			//0
			int gV = guideVertex_globalVertex[he.getVertex().getId()][1];
			globalVertices[gV].coincidentVertices.push_back(positions.size());
			planeVertex_globalVertex.push_back(gV);
			pConnects.push_back(positions.size());
			positions.push_back(globalVertices[gV].pos);

			//1
			gV = guideVertex_globalVertex[he.getVertex().getId()][0];
			globalVertices[gV].coincidentVertices.push_back(positions.size());
			planeVertex_globalVertex.push_back(gV);
			pConnects.push_back(positions.size());
			positions.push_back(globalVertices[gV].pos);

			//2
			gV = guideVertex_globalVertex[he.getSym().getVertex().getId()][0];
			globalVertices[gV].coincidentVertices.push_back(positions.size());
			planeVertex_globalVertex.push_back(gV);
			pConnects.push_back(positions.size());
			positions.push_back(globalVertices[gV].pos);

			//3
			gV = guideVertex_globalVertex[he.getSym().getVertex().getId()][1];
			globalVertices[gV].coincidentVertices.push_back(positions.size());
			planeVertex_globalVertex.push_back(gV);
			pConnects.push_back(positions.size());
			positions.push_back(globalVertices[gV].pos);


			guideHalfEdge_planeFace[he.getId()].push_back(pCounts.size());
			guideHalfEdge_planeFace[he.getNext().getId()].push_back(pCounts.size());
			guideHalfEdge_planeFace[he.getSym().getPrev().getId()].push_back(pCounts.size());

			pCounts.push_back(4);

			//zColor f1 = he.getFace().getColor();
			//zColor f2 = he.getNext().getSym().getFace().getColor();

			bool v1 = _constraintVerticesBoolean[he.getVertex().getId()];
			bool v2 = _constraintVerticesBoolean[he.getSym().getVertex().getId()];

			if (v1 && v2 )  faceColors.push_back(zColor(0, 0, 1, 1)); 
			else faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
		}

		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		planeFace_colors.clear();
		planeFace_colors = faceColors;

		fnCutPlanes.setFaceColors(planeFace_colors);
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
		globalFixedVertices.clear();

		zPointArray positions;
		zIntArray pCounts, pConnects;




		// add vertex not on medial edge to  global vertices

		//zPoint* smoothPositions = fnGuideSmoothMesh.getRawVertexPositions();

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
				guideVertex_globalVertex.push_back(vIds);


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

					for (auto& cHE : cHEdges)
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


				// NEED BETTER
				if (!v.onBoundary())
				{
					// check  vertex  on both balustrade and central blocks
					bool check = false;
					zItMeshHalfEdge he;

					zItMeshHalfEdgeArray cHEdges;
					v.getConnectedHalfEdges(cHEdges);

					for (auto& cHE : cHEdges)
					{
						if (!cHE.getNext().getSym().getNext().onBoundary() && !cHE.getNext().getSym().getNext().getVertex().onBoundary())
						{
							zItMeshFaceArray cFaces;
							cHE.getEdge().getFaces(cFaces);

							int boundaryColors = 0;
							for (auto& cF : cFaces)
							{
								zColor f = cF.getColor();
								f.toHSV();

								//cout << "\n " << floor(f.h);

								if ((int)floor(f.h) == 336 || (int)floor(f.h) == 212 || (int)floor(f.h) == 211) boundaryColors++;
							}

							if (boundaryColors == cFaces.size())
							{
								he = cHE;
								check = true;
							}

						}
					}

					if (check)
					{
						zVector he_vec = he.getVector();

						he_vec.normalize();

						globalVertices[n_gV - 2].pos = p + n * width * 0.5;;
						globalVertices[n_gV - 2].pos += he_vec * width * 0.1;
					}

				}
			}
			else
			{
				zIntArray vIds = { -1,-1 };
				guideVertex_globalVertex.push_back(vIds);
			}
		}

		globalFixedVertices.assign(globalVertices.size(), false);

		printf("\n globalVertices %i ", globalVertices.size());

		//printf("\n internalEdgecounter %i ", internalEdgecounter);

		//printf("\n nE %i  eVCount %i ", fnGuideMesh.numEdges(), guideEdge_globalVertex.size());

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
				
				if (he.getVertex().onBoundary() && he.getSym().getPrev().getStartVertex().onBoundary()) {}
				else
				{
					////if(print) printf("\n working 1");

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

					zColor f1 = he.getFace().getColor();
					zColor f2 = he.getNext().getSym().getFace().getColor();

					if (f2 == f1) faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
					else
					{
						faceColors.push_back(zColor(1, 0, 0, 1));

						he.getNext().getEdge().setWeight(4);
					}

				}


				if (he.getNext().getNext().getVertex().onBoundary() && he.getNext().getVertex().onBoundary()) {}
				else
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
					faceColors.push_back(zColor(0, 0, 1, 1));

					he.getNext().getNext().getEdge().setWeight(4);
				}


				if (he.getSym().getNext().getVertex().onBoundary() && he.getPrev().getStartVertex().onBoundary()) {}
				else
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


					zColor f1 = he.getSym().getFace().getColor();
					zColor f2 = he.getSym().getNext().getSym().getFace().getColor();

					if (f2 == f1) faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
					else
					{
						faceColors.push_back(zColor(1, 0, 0, 1));

						he.getSym().getNext().getEdge().setWeight(4);
					}

				}

				if (he.getSym().getNext().getNext().getVertex().onBoundary() && he.getSym().getNext().getVertex().onBoundary()) {}
				else
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

					faceColors.push_back(zColor(0, 0, 1, 1));

					he.getSym().getNext().getNext().getEdge().setWeight(4);

				}
							


			}

		}


		// add face per non constrained guide edge 
		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		printf("\n split mesh nP : %i", fnCutPlanes.numPolygons());

		// compute face pairs which need to have common targets
		planeFace_targetPair.clear();
		planeFace_targetPair.assign(fnCutPlanes.numPolygons(), -1);

		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (!guide_PrintMedialEdgesBoolean[e.getId()]) continue;



			zItMeshHalfEdge cHe = e.getHalfEdge(0);

			if (cHe.getVertex().getValence() > 4) continue;
			if (cHe.getSym().getVertex().getValence() > 4) continue;

			// left
			if (!cHe.getNext().getEdge().onBoundary())
			{
				zItMeshHalfEdge temp = cHe.getSym();
				

				if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] ;

				zColor red(1, 0, 0, 1);

				if (currentFaceColor == red)
				{
					if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

					int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
					int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

					planeFace_targetPair[f1_id] = f2_id;
					planeFace_targetPair[f2_id] = f1_id;

					int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
					int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

					planeFace_targetPair[f3_id] = f4_id;
					planeFace_targetPair[f4_id] = f3_id;

					//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

					zColor green(0, 1, 0, 1);

					faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = green;


					faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = green;
				}

			}

			////// SYM Edge

			cHe = e.getHalfEdge(1);

			// left
			if (!cHe.getNext().onBoundary())
			{
				zItMeshHalfEdge temp = cHe.getSym();
				
				if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]];
							
				zColor red(1, 0, 0, 1);

				if (currentFaceColor == red)
				{

					if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

					int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
					int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

					planeFace_targetPair[f1_id] = f2_id;
					planeFace_targetPair[f2_id] = f1_id;

					int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
					int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

					planeFace_targetPair[f3_id] = f4_id;
					planeFace_targetPair[f4_id] = f3_id;

					//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

					zColor green(0, 1, 0, 1);

					faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = green;


					faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = green;
				}
			}


		}

		/*int counter = 0;
		for (auto& p : planeFace_targetPair)
		{
			if (p != -1) printf("\n %i %i ", counter, p);
			counter++;

		}*/


		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{

			if (e.onBoundary()) continue;
			if (guide_MedialEdgesBoolean[e.getId()]) continue;


			zColor blue(0, 0, 1, 1);
			if (guideHalfEdge_planeFace[e.getHalfEdge(0).getId()].size() == 0) continue;
			zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[e.getHalfEdge(0).getId()][0]];


			zColor f1 = e.getHalfEdge(0).getFace().getColor();
			zColor f2 = e.getHalfEdge(1).getFace().getColor();

			if (f1 == f2) {}
			else
			{
				if (currentFaceColor == blue)
				{
					zColor yellow(1, 1, 0, 1);

					faceColors[guideHalfEdge_planeFace[e.getHalfEdge(0).getId()][0]] = yellow;
					faceColors[guideHalfEdge_planeFace[e.getHalfEdge(1).getId()][0]] = yellow;
				}

			}




		}

		planeFace_colors.clear();
		planeFace_colors = faceColors;

		fnCutPlanes.setFaceColors(planeFace_colors);

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

	ZSPACE_INLINE void zTsSDFBridge::createSmoothGuideMeshfromFile(string path, zFileTpye fileType)
	{
		zFnMesh fnGuideSmoothMesh(o_guideSmoothMesh);
		fnGuideSmoothMesh.from(path, fileType);
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

	//--- SET METHODS 

	ZSPACE_INLINE void zTsSDFBridge::setGuideMesh(zObjMesh& _o_guideMesh)
	{
		o_guideMesh = &_o_guideMesh;		
	}

	ZSPACE_INLINE void zTsSDFBridge::setConvexMedials(const vector<int>& _fixedVertices)
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
			zColor col(0, 0, 0, 1);
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

	ZSPACE_INLINE void zTsSDFBridge::setPrintMedials(const vector<int>& _fixedVertices)
	{

		printMedialVertices.clear();
		printMedialVertices = _fixedVertices;



		printMedialVerticesBoolean.clear();

		zFnMesh fnGuideMesh(*o_guideMesh);

		printMedialVerticesBoolean.assign(fnGuideMesh.numVertices(), false);

		for (int i = 0; i < printMedialVertices.size(); i++)
		{
			zItMeshVertex v(*o_guideMesh, printMedialVertices[i]);
			zColor col(0, 1, 0, 1);
			v.setColor(col);
			printMedialVerticesBoolean[printMedialVertices[i]] = true;

		}
	}


	//---- GET METHODS

	ZSPACE_INLINE int zTsSDFBridge::numPrintBlocks()
	{
		return printBlocks.size();
	}
	
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

	ZSPACE_INLINE zPoint zTsSDFBridge::getBlockInteriorPoint(int blockId)
	{
		if (blockId >= printBlocks.size()) return zPoint();

		return printBlocks[blockId].guideMesh_interiorVertexPosition;;

	}

	ZSPACE_INLINE zPointArray zTsSDFBridge::getBlockIntersectionPoints(int blockId)
	{
		if (blockId >= printBlocks.size()) return zPointArray();

		return printBlocks[blockId].intersectionPoints;;

	}

	ZSPACE_INLINE vector<zTransform> zTsSDFBridge::getBlockFrames(int blockId)
	{

		if (blockId >= printBlocks.size()) return vector<zTransform>();

		return printBlocks[blockId].sectionFrames;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= printBlocks.size()) return out;

		numGraphs = printBlocks[blockId].o_sectionGraphs.size();

		if (numGraphs == 0)return out;

		for (auto& graph : printBlocks[blockId].o_sectionGraphs)
		{
			out.push_back(&graph);
		}

		return out;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockContourGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= printBlocks.size()) return out;

		numGraphs = printBlocks[blockId].o_contourGraphs.size();

		if (numGraphs == 0)return out;

		for (auto& graph : printBlocks[blockId].o_contourGraphs)
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

	ZSPACE_INLINE void zTsSDFBridge::computePrintMedialEdgesfromMedialVertices()
	{

		guide_PrintMedialEdges.clear();
		guide_PrintMedialEdgesBoolean.clear();

		zFnMesh fn_guideMesh(*o_guideMesh);
		guide_PrintMedialEdgesBoolean.assign(fn_guideMesh.numEdges(), false);

		// strart from constraint and walk till its hits another constraint
		//discard boundary edge walks

		for (auto fV : printMedialVertices)
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
				if (printMedialVerticesBoolean[he.getVertex().getId()]) exit = true;

				if (!guide_PrintMedialEdgesBoolean[he.getEdge().getId()])
				{
					guide_PrintMedialEdgesBoolean[he.getEdge().getId()] = true;

					guide_PrintMedialEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}




		///fn_guideMesh.setEdgeColor(zColor());

		for (auto mEdge : guide_PrintMedialEdges)
		{
			zItMeshEdge e(*o_guideMesh, mEdge.first);
			zColor col(1, 0, 0.5, 1);
			e.setColor(col);
		}



		// SMOOTH MESH 

		guideSmooth_PrintMedialEdges.clear();
		guideSmooth_PrintMedialEdgesBoolean.clear();

		zFnMesh fn_guideSmoothMesh(o_guideSmoothMesh);
		guideSmooth_PrintMedialEdgesBoolean.assign(fn_guideSmoothMesh.numEdges(), false);

		// start from constraint and walk till its hits another constraint
		//discard boundary edge walks
		for (auto fV : printMedialVertices)
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
					if (printMedialVerticesBoolean[he.getVertex().getId()]) exit = true;
				}


				if (!guideSmooth_PrintMedialEdgesBoolean[he.getEdge().getId()])
				{

					guideSmooth_PrintMedialEdgesBoolean[he.getEdge().getId()] = true;

					guideSmooth_PrintMedialEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}


		for (auto mEdge : guideSmooth_PrintMedialEdges)
		{
			zItMeshEdge e(o_guideSmoothMesh, mEdge.first);
			zColor col(1, 0, 0.5, 1);
			e.setColor(col);
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlocks(float printLayerDepth)
	{
		zFnMesh  fn_guideMesh(*o_guideMesh);
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);
		zFnMesh fn_planeMesh(o_planeMesh);

		printBlocks.clear();

		int macroBlockCOunter = 0;
		zBoolArray edgeVisited;
		edgeVisited.assign(fn_guideMesh.numEdges(), false);

		// start from constraint and walk till its hits another constraint
		//discard boundary edge walks
		for (auto fV : printMedialVertices)
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
				if (printMedialVerticesBoolean[he.getVertex().getId()]) exit = true;

				zColor black(0, 0, 0, 1);
				if (he.getFace().getColor() == black)  he = he.getNext().getSym().getNext();

				zItMeshHalfEdge temp = he;
				zColor current = temp.getFace().getColor();

				bool incrCounter = false;

				if (!edgeVisited[temp.getEdge().getId()])
				{
					printBlocks.push_back(zPrintBlock());
					printBlocks[macroBlockCOunter].id = macroBlockCOunter;

					printBlocks[macroBlockCOunter].guideMesh_interiorVertex = temp.getVertex().getId();
					printBlocks[macroBlockCOunter].guideMesh_interiorVertexPosition = temp.getVertex().getPosition();

					incrCounter = true;
				}

				
				bool tempExit = false;
				do
				{
					if (printMedialVerticesBoolean[temp.getVertex().getId()])
					{
						tempExit = true;
						exit = true;
						he = temp;					

						if (temp.getFace().getColor() == black) {}
						else
						{
							if (!edgeVisited[temp.getEdge().getId()])
							{
								edgeVisited[temp.getEdge().getId()] = true;
								addConvexBlocks(printBlocks[macroBlockCOunter], temp);
							}
						}
						


					}

					if (!tempExit)
					{
						if (temp.getFace().getColor() == current) {}
						else
						{
							tempExit = true;
							he = temp;
						}
					}

					if (!tempExit)
					{											
						if (!edgeVisited[temp.getEdge().getId()])
						{
							edgeVisited[temp.getEdge().getId()] = true;
							addConvexBlocks(printBlocks[macroBlockCOunter], temp);
							
						}

					}

					temp = temp.getNext().getSym();
					temp = temp.getNext();

					//if (temp.getFace().getColor() == black) tempExit = true;



				} while (!tempExit);

				if (incrCounter)
				{
					macroBlockCOunter++;
				}



				//he = he.getNext().getSym();
				//he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}


		for (auto& mB : printBlocks)
		{
			mB.right_BoundaryFaces.clear();
			mB.left_BoundaryFaces.clear();

			mB.right_sideFaces.clear();
			mB.left_sideFaces.clear();

			addPrintBlockBoundary(mB);	

			computePrintBlockIntersections(mB);
			computePrintBlockFrames(mB, printLayerDepth);

			printf("\n %i | %i %i | %i %i ", mB.id, mB.right_BoundaryFaces.size(), mB.right_sideFaces.size(), mB.left_BoundaryFaces.size(), mB.left_sideFaces.size());
			printf("\n %i | %i %i %i ", mB.id, mB.intersectionPoints.size(), mB.right_sectionPlaneFace_GuideSmoothEdge.size(), mB.left_sectionPlaneFace_GuideSmoothEdge.size());
			printf("\n %i | %1.2f %1.2f  \n", mB.id, mB.right_planeAngle, mB.left_planeAngle);

			if (mB.id == 3)
			{
				computePrintBlockSections(mB);
			}
		
		}


	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockMesh(int blockId)
	{
		if (blockId >= printBlocks.size()) return;

		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);

		zObjMeshArray splitMeshes;
		splitMeshes.assign(printBlocks[blockId].rightBlocks.size() + printBlocks[blockId].leftBlocks.size(), zObjMesh());

		int counter = 0;
		//for (auto& b : macroblocks[blockId].rightBlocks)
		//{
			zPointArray splitPlanes_origins;
			zVectorArray splitPlanes_normals;
						
			for (auto fId : printBlocks[blockId].rightBlocks[1].faces)
			{
				zItMeshFace f(o_planeMesh, fId);

				splitPlanes_origins.push_back( f.getCenter());
				splitPlanes_normals.push_back( f.getNormal());

				cout << "\n " << f.getCenter();
				cout << "\n " << f.getNormal() << "\n";
			}

			fn_guideSmoothMesh.splitMesh_Mixed(splitPlanes_origins, splitPlanes_normals, splitMeshes[counter]);
			counter++;
		//}

		
		for (auto& b : printBlocks[blockId].leftBlocks)
		{
			zPointArray splitPlanes_origins;
			zVectorArray splitPlanes_normals;

			printf("\n faces %i ", b.faces.size());
			for (int i = 0; i< b.faces.size(); i++)
			{
				int id = i /*(i + 1) % macroblocks[blockId].leftBlocks[2].faces.size()*/;
				int fId = b.faces[id];
				zItMeshFace f(o_planeMesh, fId);

				splitPlanes_origins.push_back(f.getCenter());
				splitPlanes_normals.push_back(f.getNormal());

				cout << "\n " << f.getCenter();
				cout << "\n " << f.getNormal() << "\n";
				
			}

			fn_guideSmoothMesh.splitMesh_Mixed(splitPlanes_origins, splitPlanes_normals, splitMeshes[counter]);
			counter++;
		}

		for (auto& m : splitMeshes)
		{
			zFnMesh fn(m);
			printf("\n %i %i %i ", fn.numVertices(), fn.numEdges(), fn.numPolygons());
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::computeSDF()
	{
		//zBoolArray blockVisited;
		//blockVisited.assign(macroblocks.size(), false);

		//for (auto& b : macroblocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;


		//	if (b.id != 398) continue;

		//	b.o_contourGraphs.clear();
		//	b.o_contourGraphs.assign(b.o_sectionGraphs.size(), zObjGraph());

		//	for (int i = 1; i <4 /*b.o_contourGraphs.size()*/; i++)
		//	{
		//		computeBalustradeSDF(b, i);
		//	}
		//	
		//}
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE bool zTsSDFBridge::planarisePlaneMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations, bool printInfo, bool minEdgeConstraint, float minEdgeLen)
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

				int gV = planeVertex_globalVertex[i];
				//if (globalVertices[gV].coincidentVertices.size() == 12) fixed = true;

				if(globalFixedVertices[gV]) fixed = true;
				
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

		zColor yellow(1, 1, 0, 1);

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

			// edge length constraint
			if (minEdgeConstraint)
			{
				
				for (zItMeshEdge e(o_planeMesh); !e.end(); e++)
				{
					int i = e.getId();

					if (e.getLength() < minEdgeLen)
					{

						zItMeshHalfEdge he0 = e.getHalfEdge(0);
						zItMeshHalfEdge he1 = e.getHalfEdge(1);

						zVector  he0_vec = he0.getVector();
						zVector  he1_vec = he1.getVector();
						he0_vec.normalize();
						he1_vec.normalize();

						zVector pForce1 = (he1.getStartVertex().getPosition() + he1_vec * minEdgeLen) - he1.getVertex().getPosition();
						int gV1 = planeVertex_globalVertex[he1.getVertex().getId()];

						for (auto coincidentV : globalVertices[gV1].coincidentVertices)
						{
							fnPlaneParticles[coincidentV].addForce(pForce1);
						}


						zVector pForce0 = (he0.getStartVertex().getPosition() + he0_vec * minEdgeLen) - he0.getVertex().getPosition();
						int gV0 = planeVertex_globalVertex[he0.getVertex().getId()];

						for (auto coincidentV : globalVertices[gV0].coincidentVertices)
						{
							fnPlaneParticles[coincidentV].addForce(pForce0);
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

			if (fVolumes[i] < tolerance) col = zColor(120, 1, 1);
			f.setColor(col);

		}

		if(printInfo) printf("\n planarity tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);

		if (out)
		{
			printf("\n planarity tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);
			fnPlaneMesh.setFaceColors(planeFace_colors);
		}

		return out;

	}

	ZSPACE_INLINE bool zTsSDFBridge::alignToBRGTargets(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations, bool printInfo)
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

				int gV = planeVertex_globalVertex[i];
				//if (globalVertices[gV].coincidentVertices.size() == 12) fixed = true;

				if (globalFixedVertices[gV]) fixed = true;

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

		zColor yellow(1, 1, 0, 1);

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
						
						

			for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
			{
				int i = he.getId();

				if (targetNormals[i].length2() > 0)
				{
					
					
					if (guideHalfEdge_planeFace[he.getId()].size() == 1)
					{
						int fId = guideHalfEdge_planeFace[he.getId()][0];

						zItMeshFace f(o_planeMesh, fId);
						zVector n = f.getNormal();
						zPoint p = f.getCenter();
						

						vector<int> fVerts;
						f.getVertices(fVerts);

						for (int j = 0; j < fVerts.size(); j++)
						{
							double dist = coreUtils.minDist_Point_Plane(positions[fVerts[j]], fCenters[fId], targetNormals[i]);
							zVector pForce = targetNormals[i] * dist * -1.0;

							int gV = planeVertex_globalVertex[fVerts[j]];

							for (auto coincidentV : globalVertices[gV].coincidentVertices)
							{
								fnPlaneParticles[coincidentV].addForce(pForce);
							}

						}

					}
					
				}
				else
				{
					if (guideHalfEdge_planeFace[he.getId()].size() == 1)
					{
						int fId = guideHalfEdge_planeFace[he.getId()][0];

						zItMeshFace f(o_planeMesh, fId);
						zVector n = f.getNormal();
						zPoint p = f.getCenter();



						if (fVolumes[fId] > tolerance)
						{
							vector<int> fVerts;
							f.getVertices(fVerts);

							for (int j = 0; j < fVerts.size(); j++)
							{
								double dist = coreUtils.minDist_Point_Plane(positions[fVerts[j]], fCenters[fId], fNormalTargets[fId] /*fNormals[i]*/);
								zVector pForce = /*fNormals[i]*/ fNormalTargets[fId] * dist * -1.0;

								int gV = planeVertex_globalVertex[fVerts[j]];

								for (auto coincidentV : globalVertices[gV].coincidentVertices)
								{
									fnPlaneParticles[coincidentV].addForce(pForce);
								}

							}
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

			if (fVolumes[i] < tolerance) col = zColor(120, 1, 1);
			f.setColor(col);

		}

		
		//if (out) fnPlaneMesh.setFaceColors(planeFace_colors);

		fNormals.clear();
		fnPlaneMesh.getFaceNormals(fNormals);

		zDomainFloat devBRGDomain(10, 0);
		zFloatArray deviationsBRG;

		for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{
			int i = he.getId();

			float d = 0.5;

			if (targetNormals[i].length2() > 0)
			{
				int fId = guideHalfEdge_planeFace[he.getId()][0];

				if (guideHalfEdge_planeFace[he.getId()].size() == 1)
				{
					zItMeshFace f(o_planeMesh, fId);
					zVector n = f.getNormal();
					zPoint p = f.getCenter();

					d = n * targetNormals[i];

					if (d < devBRGDomain.min) devBRGDomain.min = d;
					if (d > devBRGDomain.max) devBRGDomain.max = d;

				}
			}

			deviationsBRG.push_back(d);
		}

		if (printInfo)
		{
			printf("\n \n  planarity tolerance : %1.7f,  dev | %1.7f %1.7f ", tolerance, deviation.min, deviation.max);

			printf("\n dotProduct tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, devBRGDomain.min, devBRGDomain.max);
		}

		if (out)
		{
			printf("\n \n  planarity tolerance : %1.7f,  dev | %1.7f %1.7f ", tolerance, deviation.min, deviation.max);

			printf("\n dotProduct tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, devBRGDomain.min, devBRGDomain.max);
		}

		return out;
	}

	ZSPACE_INLINE bool zTsSDFBridge::updateSmoothMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations)
	{
		zFnMesh fnGuideSmoothMesh(o_guideSmoothMesh);


		// create particles if it doesnt exist
		if (fnSmoothMeshParticles.size() != fnGuideSmoothMesh.numVertices())
		{
			fnSmoothMeshParticles.clear();
			o_smoothMeshParticles.clear();


			for (int i = 0; i < fnGuideSmoothMesh.numVertices(); i++)
			{
				bool fixed = false;								

				zObjParticle p;
				p.particle = zParticle(o_guideSmoothMesh.mesh.vertexPositions[i], fixed);
				o_smoothMeshParticles.push_back(p);
			}

			for (int i = 0; i < o_smoothMeshParticles.size(); i++)
			{
				fnSmoothMeshParticles.push_back(zFnParticle(o_smoothMeshParticles[i]));
			}
		}

		zVector* positions = fnGuideSmoothMesh.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
			{
				if (v.getValence() != 6) continue;
				
				zIntArray cHEdges;
				v.getConnectedHalfEdges(cHEdges);

				int smoothMeshVertexId = v.getId();

				for (auto cHE : cHEdges)
				{
					int fId = guideHalfEdge_planeFace[cHE][0];
					zItMeshFace f(o_planeMesh, fId);

					zVector n = f.getNormal();
					zPoint p = f.getCenter();

					double dist = coreUtils.minDist_Point_Plane(positions[smoothMeshVertexId], p, n);
					zVector pForce = n * dist * -1.0;

					fnSmoothMeshParticles[smoothMeshVertexId].addForce(pForce);
				}
				
			}

			// update positions
			for (int i = 0; i < fnSmoothMeshParticles.size(); i++)
			{
				fnSmoothMeshParticles[i].integrateForces(dT, type);
				fnSmoothMeshParticles[i].updateParticle(true);
			}
		}

		bool out = true;

		deviation.min = 1000;
		deviation.max = 0;
		
		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{
			if (v.getValence() != 6) continue;

			zIntArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			int smoothMeshVertexId = v.getId();

			for (auto cHE : cHEdges)
			{
				int fId = guideHalfEdge_planeFace[cHE][0];
				zItMeshFace f(o_planeMesh, fId);

				zVector n = f.getNormal();
				zPoint p = f.getCenter();

				double dist = coreUtils.minDist_Point_Plane(positions[smoothMeshVertexId], p, n);

				if (abs(dist) < deviation.min) deviation.min = abs(dist);
				if (abs(dist) > deviation.max) deviation.max = abs(dist);
			}
		}
			

		if (deviation.max > tolerance) out = false;

		printf("\n move star | tolerance : %1.4f minDeviation : %1.6f , maxDeviation: %1.6f ", tolerance, deviation.min, deviation.max);

		return out;
	}

	ZSPACE_INLINE void zTsSDFBridge::addPrintBlockBoundary(zPrintBlock& mB)
	{
		zColor red(1, 0, 0, 1);
		zColor green(0, 1, 0, 1);

		zColor yellow(1, 1, 0, 1);
		zColor blue(0, 0, 1, 1);

		// right blocks
		if (mB.rightBlocks.size() > 0)
		{
			for (auto& b : mB.rightBlocks)
			{
				for (auto& fId : b.faces)
				{
					zItMeshFace f(o_planeMesh, fId);
					
					if (f.getColor() == red || f.getColor() == green)
					{
						mB.right_BoundaryFaces.push_back(fId);
					}

					if (f.getColor() == yellow || f.getColor() == blue)
					{
						mB.right_sideFaces.push_back(fId);
					}
				}
			}
		}

		// left blocks
		if (mB.leftBlocks.size() > 0)
		{
			for (auto& b : mB.leftBlocks)
			{
				for (auto& fId : b.faces)
				{
					zItMeshFace f(o_planeMesh, fId);

					if (f.getColor() == red || f.getColor() == green)
					{
						mB.left_BoundaryFaces.push_back(fId);
					}


					if (f.getColor() == yellow || f.getColor() == blue)
					{
						mB.left_sideFaces.push_back(fId);
					}
				}
			}
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::addConvexBlocks(zPrintBlock& mBlock, zItMeshHalfEdge& guideMesh_halfedge)
	{
		//left block
		if (!guideMesh_halfedge.getNext().getNext().getEdge().onBoundary())
		{
			int l_id = mBlock.leftBlocks.size();
			mBlock.leftBlocks.push_back(zConvexBlock());



			if (!guideMesh_halfedge.getSym().getNext().getNext().getEdge().onBoundary())
			{
				zItMeshEdge l_e = guideMesh_halfedge.getNext().getNext().getEdge();
				mBlock.leftBlocks[l_id].id = l_e.getId();
				addConvexBlockFaces_fromEdge(mBlock.leftBlocks[l_id], l_e);
			}
			else
			{
				zItMeshEdge l_e = guideMesh_halfedge.getEdge();
				mBlock.leftBlocks[l_id].id = l_e.getId();
				addConvexBlockFaces_fromEdge(mBlock.leftBlocks[l_id], l_e);
			}
		}


		//right block
		if (!guideMesh_halfedge.getSym().getNext().getNext().getEdge().onBoundary())
		{
			int r_id = mBlock.rightBlocks.size();
			mBlock.rightBlocks.push_back(zConvexBlock());

			if (!guideMesh_halfedge.getNext().getNext().getEdge().onBoundary())
			{
				zItMeshEdge r_e = guideMesh_halfedge.getSym().getNext().getNext().getEdge();
				mBlock.rightBlocks[r_id].id = r_e.getId();
				addConvexBlockFaces_fromEdge(mBlock.rightBlocks[r_id], r_e);
			}
			else
			{
				zItMeshEdge r_e = guideMesh_halfedge.getSym().getEdge();
				mBlock.rightBlocks[r_id].id = r_e.getId();
				addConvexBlockFaces_fromEdge(mBlock.rightBlocks[r_id], r_e);
			}
		}

	}

	ZSPACE_INLINE void zTsSDFBridge::addConvexBlockFaces_fromEdge(zConvexBlock& _block, zItMeshEdge& guideMesh_edge)
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

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockIntersections(zPrintBlock& _block)
	{
		//if (fixedVerticesBoolean[guideMesh_V.getId()]) return;

		zFnMesh  fn_guideMesh(*o_guideMesh);
		int blockId = _block.id;


		zItMeshVertex guideMesh_V(*o_guideMesh, _block.guideMesh_interiorVertex);
		// start vertex
		

		zItMeshVertex guideSmoothMesh_V(o_guideSmoothMesh, guideMesh_V.getId());

		zItMeshHalfEdgeArray cHEdges;
		guideSmoothMesh_V.getConnectedHalfEdges(cHEdges);
		
		// Block Intersection
		for (auto& cHe : cHEdges)
		{
			zItMeshHalfEdge walkHe = cHe;

			// walk along medial edges till it intersects a block face
			if (guideSmooth_PrintMedialEdgesBoolean[cHe.getEdge().getId()])
			{
				bool exit = false;

				do
				{
					bool planeIntersection = false;
					bool computeBlockIntersectionFace = false;
					zPoint intersectionPoint;

					if (walkHe.getVertex().getId() < fn_guideMesh.numVertices())
					{
						if (printMedialVerticesBoolean[walkHe.getVertex().getId()])
						{	
							intersectionPoint = walkHe.getVertex().getPosition();
							_block.intersectionPoints.push_back(intersectionPoint);

							zIntPair p(-1, walkHe.getId());
							if (_block.right_BoundaryFaces.size() > 0) _block.right_sectionPlaneFace_GuideSmoothEdge.push_back(p);
							if (_block.left_BoundaryFaces.size() > 0) _block.left_sectionPlaneFace_GuideSmoothEdge.push_back(p);

							exit = true;
						}
					}

					zPoint eE = walkHe.getVertex().getPosition();
					zPoint eS = walkHe.getStartVertex().getPosition();


					if (!exit)
					{
						if (walkHe.getVertex().getValence() == 6)
						{
							intersectionPoint = walkHe.getVertex().getPosition();
							_block.intersectionPoints.push_back(intersectionPoint);

							exit = true;
							computeBlockIntersectionFace = true;
						}
					}


					if (!exit)
					{
						if (_block.right_BoundaryFaces.size() > 0)
						{
							for (auto fId : _block.right_BoundaryFaces)
							{
								zItMeshFace f(o_planeMesh, fId);

								zVector fNorm = f.getNormal();
								zVector fCenter = f.getCenter();

								planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, fNorm, fCenter, intersectionPoint);

								if (planeIntersection)
								{									
									_block.intersectionPoints.push_back(intersectionPoint);

									exit = true;
									computeBlockIntersectionFace = true;
									break;
								}
							}
						}


					}


					if (!exit)
					{
						if (_block.left_BoundaryFaces.size() > 0)
						{
							
							for (auto fId : _block.left_BoundaryFaces)
							{
								zItMeshFace f(o_planeMesh, fId);

								zVector fNorm = f.getNormal();
								zVector fCenter = f.getCenter();

								planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, fNorm, fCenter, intersectionPoint);

								if (planeIntersection)
								{								
									_block.intersectionPoints.push_back(intersectionPoint);

									exit = true;
									computeBlockIntersectionFace = true;
									break;
								}
							}

						}

					}

					///////////////////
					// compute block intesection face plane 
					if (exit && computeBlockIntersectionFace)
					{
						
						if (_block.right_BoundaryFaces.size() > 0)
						{
							zIntPair p(-1, -1);
							float dist = 100000;

							for (auto fId : _block.right_BoundaryFaces)
							{
								zItMeshFace f(o_planeMesh, fId);

								zVector fNorm = f.getNormal();
								zVector fCenter = f.getCenter();

								float d = coreUtils.minDist_Point_Plane(intersectionPoint, fCenter, fNorm);

								if (abs(d) < dist)
								{
									dist = abs(d);
									p = zIntPair(fId, walkHe.getId());								
																	
								}
								
							}

							_block.right_sectionPlaneFace_GuideSmoothEdge.push_back(p);
						}

						if (_block.left_BoundaryFaces.size() > 0)
						{
							zIntPair p(-1, -1);
							float dist = 100000;

							for (auto fId : _block.left_BoundaryFaces)
							{
								zItMeshFace f(o_planeMesh, fId);

								zVector fNorm = f.getNormal();
								zVector fCenter = f.getCenter();

								float d = coreUtils.minDist_Point_Plane(intersectionPoint, fCenter, fNorm);

								if (abs(d) < dist)
								{
									dist = abs(d);
									p = zIntPair(fId, walkHe.getId());

								}

							}

							_block.left_sectionPlaneFace_GuideSmoothEdge.push_back(p);
						}

					}


					walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();


				} while (!exit);

			}
		}
		

	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockFrames(zPrintBlock& _block, float printLayerDepth)
	{
		//if (_block.id == -1) return;
		
		(_block.rightBlocks.size() > 0)? computePrintBlockLength(_block, false) : computePrintBlockLength(_block, true);
	
		 //// ANGLE COMPUTE
		 // compute angle for right blocks if they exist
		 if (_block.rightBlocks.size() > 0)
		 {
			 zVector startFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 startFNorm = startFNorm * -1;
			 _block.right_planeAngle = endFNorm.angle(startFNorm);
		 }

		 if (_block.leftBlocks.size() > 0)
		 {
			 zVector startFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 startFNorm = startFNorm * -1;

			 _block.left_planeAngle = endFNorm.angle(startFNorm);
		 }

		 //// FRAME COMPUTE

		 // compute frames for right blocks
		 if (_block.rightBlocks.size() > 0)
		 {
			 int numLayers = floor(_block.mg_Length / printLayerDepth);

			 zVector startFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0,0,-1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			startFNorm = startFNorm * -1;
		 
			float angle = _block.right_planeAngle;
			float angleStep = _block.right_planeAngle / numLayers;

			 float xStep = (endFNorm.x - startFNorm.x) / numLayers;
			 float yStep = (endFNorm.y - startFNorm.y) / numLayers;
			 float zStep = (endFNorm.z - startFNorm.z) / numLayers;
			 			 
			 //printf("\n step %1.6f  %1.6f  %1.6f \n ", xStep, yStep, zStep);

			// compute frames
			 zItMeshHalfEdge startHe(o_guideSmoothMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].second);
			 zItMeshHalfEdge endHe(o_guideSmoothMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].second);
			 zItMeshHalfEdge walkHe = startHe;
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

			 zTransform pFrame = setTransformFromVectors(O, X, Y, tempZ);
			 _block.sectionFrames.push_back(pFrame);
			 

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
				 pFrame = setTransformFromVectors(O, X, Y, tempZ);
				 _block.sectionFrames.push_back(pFrame);

				 pOnCurve = O;

				 //cout << "\n current : " << Z;

			 }

			 // end point
			////cout << "\n dist " << _block.intersectionPoints[1].distanceTo(O);
			// O = _block.intersectionPoints[1];
			// Z = endFNorm;

			// tempZ = Z;
			// tempZ.normalize();
			// tempZ *= tLen;

			// Y = zVector();
			// eFaces.clear();
			// walkHe.getEdge().getFaces(eFaces);

			// for (auto& eF : eFaces)	Y += eF.getNormal();
			// Y /= eFaces.size();
			// Y.normalize();
			// Y *= -1;
			// Y *= tLen;

			// X = Z ^ Y;
			// X.normalize();
			// X *= tLen;

			// // add frame
			// pFrame = setTransformFromVectors(O, X, Y, tempZ);
			// _block.sectionFrames.push_back(pFrame);

		 }

		 // compute frames for left blocks
		 if (_block.leftBlocks.size() > 0)
		 {
			 int numLayers = floor(_block.mg_Length / printLayerDepth);

			 zVector startFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 startFNorm = startFNorm * -1;

			 float angle = _block.left_planeAngle;
			 float angleStep = _block.left_planeAngle / numLayers;
			 			 float xStep = (endFNorm.x - startFNorm.x) / numLayers;
			 float yStep = (endFNorm.y - startFNorm.y) / numLayers;
			 float zStep = (endFNorm.z - startFNorm.z) / numLayers;

			 //printf("\n step %1.6f  %1.6f  %1.6f \n ", xStep, yStep, zStep);

			// compute frames
			 zItMeshHalfEdge startHe(o_guideSmoothMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].second);
			 zItMeshHalfEdge endHe(o_guideSmoothMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].second);
			 zItMeshHalfEdge walkHe = startHe;
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

			 zTransform pFrame = setTransformFromVectors(O, X, Y, tempZ);
			 _block.sectionFrames.push_back(pFrame);


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
				 pFrame = setTransformFromVectors(O, X, Y, tempZ);
				 _block.sectionFrames.push_back(pFrame);

				 pOnCurve = O;

				 //cout << "\n current : " << Z;

			 }

			 // end point
			////cout << "\n dist " << _block.intersectionPoints[1].distanceTo(O);
			// O = _block.intersectionPoints[1];
			// Z = endFNorm;

			// tempZ = Z;
			// tempZ.normalize();
			// tempZ *= tLen;

			// Y = zVector();
			// eFaces.clear();
			// walkHe.getEdge().getFaces(eFaces);

			// for (auto& eF : eFaces)	Y += eF.getNormal();
			// Y /= eFaces.size();
			// Y.normalize();
			// Y *= -1;
			// Y *= tLen;

			// X = Z ^ Y;
			// X.normalize();
			// X *= tLen;

			// // add frame
			// pFrame = setTransformFromVectors(O, X, Y, tempZ);
			// _block.sectionFrames.push_back(pFrame);

		 }

		
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockSections(zPrintBlock& _block)
	{
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);

		_block.o_sectionGraphs.clear();
		_block.o_sectionGraphs.assign(_block.sectionFrames.size(), zObjGraph());

		zScalarArray scalars;

		//Right Blocks
		if (_block.rightBlocks.size() > 0 && _block.sectionFrames.size() > 0)
		{
			int start = 0;
			int end =(_block.leftBlocks.size() == 0) ? _block.sectionFrames.size() : floor(_block.sectionFrames.size() * 0.5);
						
			
			for (int i = start; i < end; i++)
			{
				
				scalars.clear();
				zPoint O(_block.sectionFrames[i](3, 0), _block.sectionFrames[i](3, 1), _block.sectionFrames[i](3, 2));
				zVector N(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));


				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				//fn_guideSmoothMesh.setVertexColorsfromScalars(scalars);

				zPointArray positions;
				zIntArray edgeConnects;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects);

				// compute inside which convex block the graph exits
				zConvexBlock tempConvexB;
				zPoint pTemp = positions[0];
				for (auto& rB : _block.rightBlocks)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zItMeshFace f0(o_planeMesh, rB.faces[0]);
					zPoint O0 = f0.getCenter();
					zVector N0 = f0.getNormal();

					float minDist_Plane0 = coreUtils.minDist_Point_Plane(pTemp, O0, N0);
					if (minDist_Plane0 > EPS) internal_v0 = false;

					zItMeshFace f1(o_planeMesh, rB.faces[2]);
					zPoint O1 = f1.getCenter();
					zVector N1 = f1.getNormal();

					float minDist_Plane1 = coreUtils.minDist_Point_Plane(pTemp, O1, N1);
					if (minDist_Plane1 > EPS) internal_v1 = false;

					if (internal_v0 && internal_v1)
					{
						tempConvexB = rB;
						break;
					}

				}

				// trim with side plane
				zPointArray t_positions;
				zIntArray t_edgeConnects;
				unordered_map <string, int> positionVertex;
				zIntArray tVertex_vertexmap;

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 1; j < tempConvexB.faces.size(); j += 2)
					{

						zItMeshFace f(o_planeMesh, tempConvexB.faces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						if (minDist_Plane0 > 0) internal_v0 = false;

						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);
						if (minDist_Plane1 > 0) internal_v1 = false;

						// intersection
						if (internal_v0 && !internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								p1 = iPt;
								internal_v1 = true;
							}
						}

						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								p0 = iPt;
								internal_v0 = true;
							}
						}
					}

					if (internal_v0 && internal_v1)
					{

						int v0;

						bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

						if (!v0Exists)
						{
							v0 = t_positions.size();
							t_positions.push_back(p0);
							coreUtils.addToPositionMap(positionVertex, p0, v0, 3);
						}

						int v1;
						bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 3, v1);
						if (!v1Exists)
						{
							v1 = t_positions.size();
							t_positions.push_back(p1);
							coreUtils.addToPositionMap(positionVertex, p1, v1, 3);
						}

						t_edgeConnects.push_back(v0);
						t_edgeConnects.push_back(v1);
					}

				}


				zFnGraph tempFn(_block.o_sectionGraphs[i]);
				tempFn.create(t_positions, t_edgeConnects);;

				//printf("\n graph %i %i ", tempFn.numVertices(), tempFn.numEdges());

				tempFn.setEdgeColor(zColor(1, 0, 1, 1));
				tempFn.setEdgeWeight(3);
			}
			
		}

		//Left blocks
		if (_block.leftBlocks.size() > 0 && _block.sectionFrames.size() > 0)
		{
			int start = (_block.rightBlocks.size() == 0) ? 0 : floor(_block.sectionFrames.size() * 0.5);
			int end = _block.sectionFrames.size() ;
			

			for (int i = start; i < end; i++)
			{

				scalars.clear();
				zPoint O(_block.sectionFrames[i](3, 0), _block.sectionFrames[i](3, 1), _block.sectionFrames[i](3, 2));
				zVector N(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));


				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				//fn_guideSmoothMesh.setVertexColorsfromScalars(scalars);

				zPointArray positions;
				zIntArray edgeConnects;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects);

				// compute inside which convex block the graph exits
				zConvexBlock tempConvexB;
				zPoint pTemp = positions[0];
				for (auto& lB : _block.leftBlocks)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zItMeshFace f0(o_planeMesh, lB.faces[0]);
					zPoint O0 = f0.getCenter();
					zVector N0 = f0.getNormal();

					float minDist_Plane0 = coreUtils.minDist_Point_Plane(pTemp, O0, N0);
					if (minDist_Plane0 > EPS) internal_v0 = false;

					zItMeshFace f1(o_planeMesh, lB.faces[2]);
					zPoint O1 = f1.getCenter();
					zVector N1 = f1.getNormal();

					float minDist_Plane1 = coreUtils.minDist_Point_Plane(pTemp, O1, N1);
					if (minDist_Plane1 > EPS) internal_v1 = false;

					if (internal_v0 && internal_v1)
					{
						tempConvexB = lB;
						break;
					}

				}

				// trim with side plane
				zPointArray t_positions;
				zIntArray t_edgeConnects;
				unordered_map <string, int> positionVertex;
				zIntArray tVertex_vertexmap;

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 1; j < tempConvexB.faces.size(); j += 2)
					{

						zItMeshFace f(o_planeMesh, tempConvexB.faces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						if (minDist_Plane0 > 0) internal_v0 = false;

						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);
						if (minDist_Plane1 > 0) internal_v1 = false;

						// intersection
						if (internal_v0 && !internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								p1 = iPt;
								internal_v1 = true;
							}
						}

						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								p0 = iPt;
								internal_v0 = true;
							}
						}
					}

					if (internal_v0 && internal_v1)
					{

						int v0;

						bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

						if (!v0Exists)
						{
							v0 = t_positions.size();
							t_positions.push_back(p0);
							coreUtils.addToPositionMap(positionVertex, p0, v0, 3);
						}

						int v1;
						bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 3, v1);
						if (!v1Exists)
						{
							v1 = t_positions.size();
							t_positions.push_back(p1);
							coreUtils.addToPositionMap(positionVertex, p1, v1, 3);
						}

						t_edgeConnects.push_back(v0);
						t_edgeConnects.push_back(v1);
					}

				}


				zFnGraph tempFn(_block.o_sectionGraphs[i]);
				tempFn.create(t_positions, t_edgeConnects);;

				//printf("\n graph %i %i ", tempFn.numVertices(), tempFn.numEdges());

				tempFn.setEdgeColor(zColor(1, 0, 1, 1));
				tempFn.setEdgeWeight(3);
			}

		}

	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockLength(zPrintBlock& _block, bool leftBlock)
	{
		zItMeshHalfEdge startHe(o_guideSmoothMesh, (leftBlock)? _block.left_sectionPlaneFace_GuideSmoothEdge[0].second : _block.right_sectionPlaneFace_GuideSmoothEdge[0].second);
		zItMeshHalfEdge endHe(o_guideSmoothMesh, (leftBlock) ? _block.left_sectionPlaneFace_GuideSmoothEdge[1].second : _block.right_sectionPlaneFace_GuideSmoothEdge[1].second);

		// compute total edge length
		float length = 0;

		length += startHe.getStartVertex().getPosition().distanceTo(_block.intersectionPoints[0]);
		length += endHe.getStartVertex().getPosition().distanceTo(_block.intersectionPoints[1]);

		zItMeshHalfEdge walkHe = startHe;
		walkHe = walkHe.getSym();
		walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

		bool exit = false;
		while (walkHe != endHe)
		{
			length += walkHe.getLength();

			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
		}

		_block.mg_Length = length;				

	}

	ZSPACE_INLINE void zTsSDFBridge::computeBalustradeSDF(zConvexBlock& _block, int graphId)
	{
		//if (_block.id == -1) return;
		//if (graphId >= _block.o_sectionGraphs.size())return;

		//zFnGraph fnGraph(_block.o_sectionGraphs[graphId]);
		//zPoint* positions = fnGraph.getRawVertexPositions();

		//zTransform t = _block.sectionFrames[graphId];
		//fnGraph.setTransform(t, true, false);

		//zPoint o(t(3, 0), t(3, 1), t(3, 2));
		//zVector n(t(2, 0), t(2, 1), t(2, 2));

		///*for (int i = 0; i < fnGraph.numVertices(); i++)
		//{			

		//	double d = coreUtils.minDist_Point_Plane(positions[i], o, n);
		//	printf("\n %i %1.2f ", i, d);
		//}*/

		//

		//// clip planes

		//zPoint v_startPos = positions[0];
		//int faceId = -1;
		//float dist = 10000;
		//for (auto sF : _block.macroBlock_sideFaces)
		//{
		//	zItMeshFace f(o_planeMesh, sF);

		//	

		//	double d = coreUtils.minDist_Point_Plane(v_startPos, o, n);

		//	if (d < dist)
		//	{
		//		dist = d; 
		//		faceId = sF;
		//	}
		//}

		//zVector fNorm;
		//zPoint fCen;
		//if (faceId != -1)
		//{
		//	zItMeshFace f(o_planeMesh, faceId);
		//	f.setColor(zColor(0, 1, 0, 1));			

		//	zTransformationMatrix from;
		//	from.setTransform(_block.sectionFrames[graphId], true);

		//	zTransform transform = from.getLocalMatrix();

		//	zPointArray fVertPos;
		//	f.getVertexPositions(fVertPos);

		//	zItMeshVertexArray fVerts;
		//	f.getVertices(fVerts);

		//	for (auto& v : fVerts)
		//	{
		//		zPoint vPos = v.getPosition();

		//		zPoint newPos = vPos * transform;		

		//		v.setPosition(newPos);
		//	}		

		//	f.updateNormal();
		//	fNorm = f.getNormal();
		//	//fNorm *= -1;

		//	fCen = f.getCenter();		

		//	// revert to original position

		//	int counter = 0;
		//	for (auto& v : fVerts)
		//	{	
		//		v.setPosition(fVertPos[counter]);
		//		counter++;
		//	}

		//	f.updateNormal();
		//}


		//// Transform
		//	
		//
		//zTransform tLocal;
		//tLocal.setIdentity();
		//fnGraph.setTransform(tLocal, true, true);
		//

		//// vertex Iterators
		//zItGraphVertex v_start(_block.o_sectionGraphs[graphId], 0);
		//zItGraphVertex v_end(_block.o_sectionGraphs[graphId], fnGraph.numVertices() - 1);
		//		

		//zItGraphHalfEdge he_start = v_start.getHalfEdge().getSym();;
		//zVector vec = he_start.getSym().getVector();
		//vec.normalize();

		//zItGraphHalfEdge he_end = v_end.getHalfEdge().getSym();

		//zPoint p0 = v_start.getPosition() + vec * 0.25;

		//zItGraphHalfEdge he = he_start;
		//he = he.getNext();
		//he = he.getNext();
		//he = he.getNext();
		//he = he.getNext();
		//he = he.getNext();
		//

		//zVector avgVec = (he_end.getVector() + he_start.getVector()) * 0.5;
		//avgVec.normalize();

		//zPoint p1 = he.getVertex().getPosition() + (avgVec * 0.2);
		//zPoint p2 = he.getVertex().getPosition() + (avgVec * 0.05 * -1);		
		//zPoint p3 = v_end.getPosition();

		//he = he.getNext().getNext();
		//zPoint p4 = he.getVertex().getPosition() + (avgVec * 0.05 * -1);

		//cout << "\n p0: " << p0;
		//cout << "\n p1: " << p1;
		//cout << "\n p2: " << p2;
		//cout << "\n p3: " << p3;
		//		
		//
		//// field
		//zFnMeshScalarField fnField(o_field);

		//// circle 
		//zScalarArray circleField_top;
		//fnField.getScalars_Circle(circleField_top, p3, 0.05, 0.0, false);

		//zScalarArray circleField_side;
		//fnField.getScalars_Circle(circleField_side, p1, 0.1, 0.0, false);

		//// line 
		//zScalarArray circleField_side1;
		//fnField.getScalars_Circle(circleField_side1, p4, 0.05, 0.0, false);

		//// triangle

		//zScalarArray triangleField_0;
		//fnField.getScalars_Triangle(triangleField_0, p0, p1, p2, 0.0, false);

		//zScalarArray triangleField_1;
		//fnField.getScalars_Triangle(triangleField_1, p3, p1, p2, 0.0, false);
		//

		////cout << "\n " << coreUtils.zMin(triangleField) << " , " << coreUtils.zMax(triangleField);

		//// BOOLEANS

		//zScalarArray boolean_field1;
		//fnField.boolean_union(triangleField_0, triangleField_1, boolean_field1, false);

		//zScalarArray boolean_field2;
		//fnField.boolean_union(boolean_field1, circleField_top, boolean_field2, false);	

		////zScalarArray boolean_field3;
		////fnField.boolean_union(boolean_field2, circleField_side1, boolean_field3, false);

		//zScalarArray boolean_field4;
		//fnField.boolean_subtract(boolean_field2, circleField_side, boolean_field4, false);
		//	
		//// CLIP PLANES
		//
		//zScalarArray clipPlaneField;
		//fnField.boolean_clipwithPlane(boolean_field4, clipPlaneField, fCen, fNorm);
		//		

		//fnField.smoothField(clipPlaneField,2, 0.0, zSpace::zDiffusionType::zAverage);
		//		

		//// CONTOURS
		//fnField.setFieldValues(clipPlaneField);
		//
		//fnField.getIsocontour(_block.o_contourGraphs[graphId], 0.0);

		//fnField.getIsolineMesh(o_isoMesh, 0.0);
		//
		//// transform back 
		//fnGraph.setTransform(t, true, true);
		//
		//zFnGraph fnIsoGraph(_block.o_contourGraphs[graphId]);
		//fnIsoGraph.setEdgeColor(zColor(0, 1, 0, 1));

		//fnIsoGraph.setTransform(t, true, true);

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
		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		for (auto& mB : printBlocks)
		{
			if (mB.id == -1) continue;


			string outfilename = dir;
			outfilename += "/";
			outfilename += filename;
			outfilename += "_";
			outfilename += to_string(mB.id);

			ofstream myfile;

			// RIGHT SIDE

			if (mB.rightBlocks.size() > 0)
			{
				string temp_right = outfilename;
				temp_right += "_right.txt";

				
				myfile.open(temp_right.c_str());

				if (myfile.fail())
				{
					cout << " error in opening file  " << temp_right.c_str() << endl;
					break;
				}

				int bCounter = 0;
				for (auto& b : mB.rightBlocks)
				{
					myfile << b.faces.size();

					if (bCounter < mB.rightBlocks.size() - 1) myfile << ",";
					bCounter++;
				}

				myfile << "\n";

				bCounter = 0;
				for (auto& b : mB.rightBlocks)
				{
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

					if (bCounter < mB.rightBlocks.size() - 1) myfile << "\n ";
					bCounter++;
				}



				myfile.close();
			}
			

			// LEFT SIDE
			if (mB.leftBlocks.size() > 0)
			{
				string temp_left = outfilename;
				temp_left += "_left.txt";


				myfile.open(temp_left.c_str());

				if (myfile.fail())
				{
					cout << " error in opening file  " << temp_left.c_str() << endl;
					break;
				}

				int bCounter = 0;
				for (auto& b : mB.leftBlocks)
				{
					myfile << b.faces.size();

					if (bCounter < mB.leftBlocks.size() - 1) myfile << ",";
					bCounter++;
				}

				myfile << "\n";

				bCounter = 0;
				for (auto& b : mB.leftBlocks)
				{
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

					if (bCounter < mB.leftBlocks.size() - 1) myfile << "\n ";
					bCounter++;
				}



				myfile.close();
			}
			

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockSidePlanesToTXT(string dir, string filename)
	{
		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		//for (auto& b : blocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;

		//	printf("\n %i ", b.id);
		//	// output file

		//	string outfilename = dir;
		//	outfilename += "/";
		//	outfilename += filename;
		//	outfilename += "_";
		//	outfilename += to_string(b.id);
		//	outfilename += ".txt";

		//	ofstream myfile;
		//	myfile.open(outfilename.c_str());

		//	if (myfile.fail())
		//	{
		//		cout << " error in opening file  " << outfilename.c_str() << endl;
		//		break;
		//	}

		//	myfile << b.faces.size();
		//	for (int mB : b.macroBlocks)
		//	{
		//		myfile << "," << blocks[mB].faces.size();
		//	}

		//	myfile << "\n";

		//	int counter = 0;
		//	printf(" %i ", b.macroBlock_sideFaces.size());
		//	for (auto fId : b.macroBlock_sideFaces)
		//	{
		//		zItMeshFace f(o_planeMesh, fId);

		//		zPoint O = f.getCenter();
		//		zVector N = f.getNormal();

		//		myfile << O.x << "," << O.y << "," << O.z << "," << N.x << "," << N.y << "," << N.z;

		//		if (counter != b.macroBlock_sideFaces.size() - 1) myfile << "\n ";
		//		counter++;

		//	}

		//	blockVisited[b.id] = true;

		//	for (int mB : b.macroBlocks)
		//	{			
		//		blockVisited[mB] = true;
		//	}

		//	myfile.close();

		//}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockSectionPlanesToTXT(string dir, string filename)
	{

		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		//for (auto& b : blocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;


		//	// output file

		//	string outfilename = dir;
		//	outfilename += "/";
		//	outfilename += filename;
		//	outfilename += "_";
		//	outfilename += to_string(b.id);
		//	outfilename += ".txt";

		//	ofstream myfile;
		//	myfile.open(outfilename.c_str());

		//	if (myfile.fail())
		//	{
		//		cout << " error in opening file  " << outfilename.c_str() << endl;
		//		break;
		//	}

		//	int counter = 0;
		//	for (auto t : b.sectionFrames)
		//	{
		//		myfile << t(0, 0) << "," << t(0, 1) << "," << t(0, 2) << "," << t(1, 0) << "," << t(1, 1) << "," << t(1, 2) << "," << t(2,0) << "," << t(2, 1) << "," << t(2, 2) << "," << t(3, 0) << "," << t(3, 1) << "," << t(3, 2);

		//		if (counter != b.sectionFrames.size() -1 ) myfile<< "\n ";
		//		counter++;
		//	}

		//	blockVisited[b.id] = true;
		//	for (int mB : b.macroBlocks) blockVisited[mB] = true;

		//	myfile.close();

		//}


	}

	ZSPACE_INLINE void zTsSDFBridge::blockSectionsFromJSON(string dir, string filename)
	{
		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		//for (auto& b : blocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;

		//	if (b.id != 398) continue;

		//	// input file

		//	string infilename = dir;
		//	infilename += "/";		
		//	infilename += to_string(b.id);
		//	infilename += "_";
		//	infilename += filename;
		//	infilename += "_";

		//	b.o_sectionGraphs.clear();
		//	b.o_sectionGraphs.assign(b.sectionFrames.size(), zObjGraph());
		//	
		//	for (int j = 0; j < b.sectionFrames.size(); j++)
		//	{
		//		string temp = infilename;
		//		temp += to_string(j);
		//		temp += ".json";

		//		zFnGraph fnGraph(b.o_sectionGraphs[j]);
		//		fnGraph.from(temp, zJSON, true);
		//	}

		//	blockVisited[b.id] = true;
		//	for (int mB : b.macroBlocks) blockVisited[mB] = true;
		//}

	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToJSON(string dir, string filename)
	{
		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		//for (auto& b : blocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;

		//	if (b.id != 398) continue;


		//	// output file

		//	string outfilename = dir;
		//	outfilename += "/";
		//	outfilename += filename;
		//	outfilename += "_";
		//	outfilename += to_string(b.id);
		//	outfilename += "_";

		//	for (int j = 0; j < b.o_contourGraphs.size(); j++)
		//	{
		//		string temp = outfilename;
		//		temp += to_string(j);
		//		temp += ".json";

		//		zFnGraph fnGraph(b.o_contourGraphs[j]);
		//		fnGraph.to(temp, zJSON);
		//	}

		//	blockVisited[b.id] = true;
		//	for (int mB : b.macroBlocks) blockVisited[mB] = true;

		//}
	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToIncr3D(string dir, string filename, float layerWidth)
	{
		//zBoolArray blockVisited;
		//blockVisited.assign(blocks.size(), false);

		//for (auto& b : blocks)
		//{
		//	if (b.id == -1) continue;
		//	if (blockVisited[b.id]) continue;

		//	if (b.id != 398) continue;

		//	// output file

		//	string outfilename = dir;
		//	outfilename += "/";
		//	outfilename += filename;
		//	outfilename += "_";
		//	outfilename += to_string(b.id);
		//	outfilename += ".h";

		//	ofstream myfile;
		//	myfile.open(outfilename.c_str());

		//	if (myfile.fail())
		//	{
		//		cout << " error in opening file  " << outfilename.c_str() << endl;
		//		break;
		//	}

		//	float maxLayerHeight = 0;
		//	float minLayerHeight = 10000;

		//							
		//	int startVertexId = -1;
		//	zItGraphHalfEdge he;
		//	zItGraphVertex v;
		//	float totalLength = 0;
		//	for (int j = 0; j < b.o_contourGraphs.size() - 1; j++)
		//	{		
		//		
		//		myfile << "Layer " << j << "\n";
		//		myfile << "/* " << "\n";

		//		zFnGraph fnG(b.o_contourGraphs[j]);
		//		if (fnG.numVertices() == 0)
		//		{
		//			myfile << "*/ " << "\n";
		//			continue;
		//		}

		//		bool flip = false;

		//		if (startVertexId == -1) startVertexId = 0;
		//		else
		//		{
		//			zPoint startP_Prev = v.getPosition();
		//			float dist = 10000;

		//			zPoint* positions = fnG.getRawVertexPositions();
		//			for (int i =0; i< fnG.numVertices(); i++)
		//			{
		//				float d = positions[i].distanceTo(startP_Prev);
		//				
		//				if (d < dist)
		//				{
		//					dist = d;
		//					startVertexId = i;
		//				}
		//			}

		//			zItGraphVertex temp(b.o_contourGraphs[j], startVertexId);

		//			flip = (temp.getHalfEdge().getVector() * he.getVector() < 0) ? true : false;
		//		}

		//		v = zItGraphVertex(b.o_contourGraphs[j], startVertexId);
		//		v.setColor(zColor(0, 1, 0, 1));
		//		
		//		he = v.getHalfEdge();
		//		if (flip) he = he.getSym();

		//		zItGraphHalfEdge start = he;
		//		start.getVertex().setColor(zColor(0, 0, 1, 1));

		//		zVector norm(b.sectionFrames[j](2, 0), b.sectionFrames[j](2, 1), b.sectionFrames[j](2, 2));
		//		norm.normalize();

		//		zVector nextNorm(b.sectionFrames[j + 1](2, 0), b.sectionFrames[j + 1](2, 1), b.sectionFrames[j + 1](2, 2));
		//		zVector nextOrigin(b.sectionFrames[j + 1](3, 0), b.sectionFrames[j + 1](3, 1), b.sectionFrames[j + 1](3, 2));

		//		do
		//		{
		//			totalLength += he.getLength();
		//			zPoint p = he.getVertex().getPosition();

		//			zPoint p1 = p + norm * 1.0;

		//			zPoint intPt;
		//			bool check = coreUtils.line_PlaneIntersection(p, p1, nextNorm, nextOrigin, intPt);

		//			if (!check) printf("\n %i %i no Intersection ",j, he.getVertex().getId());

		//			float layerHeight = intPt.distanceTo(p);
		//			maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
		//			minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

		//			myfile << p.x << "," << p.y << "," << p.z << "," ;
		//			myfile << norm.x << "," << norm.y << "," << norm.z << ",";
		//			myfile << layerWidth << ",";
		//			myfile << layerHeight << "\n";
		//			

		//			he = he.getNext();

		//		} while (he != start);
		//		

		//		myfile << "*/ " << "\n";
		//	}

		//	myfile << "Attributes " << "\n";
		//	myfile << "/* " << "\n";
		//	myfile << "Units: meters " << "\n";
		//	myfile << "Block ID:  " << b.id << "\n";
		//	myfile << "max layer height: " << maxLayerHeight << "\n";
		//	myfile << "min layer height: " << minLayerHeight << "\n";
		//	myfile << "total print length: " << totalLength << "\n";
		//	myfile << "*/ " << "\n";

		//	myfile.close();

		//	blockVisited[b.id] = true;
		//	for (int mB : b.macroBlocks) blockVisited[mB] = true;
		//}
	}

	ZSPACE_INLINE void zTsSDFBridge::toBRGJSON(string path, zPointArray &points,  zVectorArray &normals)
	{
		zFnMesh fnGuidemesh(*o_guideMesh);

		fnGuidemesh.to(path, zJSON);
		fnGuidemesh.setEdgeColor(zColor());

		points.clear();
		normals.clear();

		// read existing data in the json 
		json j;

		ifstream in_myfile;
		in_myfile.open(path.c_str());

		int lineCnt = 0;

		if (in_myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
		}

		in_myfile >> j;
		in_myfile.close();

		// CREATE JSON FILE
		zUtilsJsonHE meshJSON;

	
		//Halfedge planes
		zColor red(1, 0, 0, 1);
		zColor green(0, 1, 0, 1);
		zColor yellow(1, 1, 0, 1);
		for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{
			vector<double> he_attrib;
			if (guide_MedialEdgesBoolean[he.getEdge().getId()])
			{
				he_attrib.push_back(0);
				he_attrib.push_back(0);
				he_attrib.push_back(0);
				he_attrib.push_back(0);
				he_attrib.push_back(0);
				he_attrib.push_back(0);
			}
			
			else
			{
				if (guideHalfEdge_planeFace[he.getId()].size() == 1)
				{
					int fId = guideHalfEdge_planeFace[he.getId()][0];

					zItMeshFace f(o_planeMesh, fId);
					zVector n = f.getNormal();
					zPoint p = f.getCenter();
					zColor col = f.getColor();

					if (col == red || col == green || col == yellow)
					{
						he_attrib.push_back(p.x);
						he_attrib.push_back(p.y);
						he_attrib.push_back(p.z);

						he_attrib.push_back(n.x);
						he_attrib.push_back(n.y);
						he_attrib.push_back(n.z);

						he.getEdge().setColor(zColor(1, 0, 0, 1));
						he.getEdge().setWeight(2);

						points.push_back(p);
						normals.push_back(n);
										

					}
					else
					{
						he_attrib.push_back(0);
						he_attrib.push_back(0);
						he_attrib.push_back(0);

						he_attrib.push_back(0);
						he_attrib.push_back(0);
						he_attrib.push_back(0);
					}


				}
				else
				{
					he_attrib.push_back(0);
					he_attrib.push_back(0);
					he_attrib.push_back(0);
					he_attrib.push_back(0);
					he_attrib.push_back(0);
					he_attrib.push_back(0);
				}
			}

			

			


			meshJSON.halfedgeAttributes.push_back(he_attrib);
		}


		//printf("\n %i %i ", meshJSON.halfedgeAttributes.size(), fnGuidemesh.numHalfEdges());

		// Json file 
		j["HalfedgeAttributes"] = meshJSON.halfedgeAttributes;

		// EXPORT	
		ofstream myfile;
		myfile.open(path.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
			return;
		}

		//myfile.precision(16);
		myfile << j.dump();
		myfile.close();

	}

	ZSPACE_INLINE bool zTsSDFBridge::fromBRGJSON(string path, zPointArray& points, zVectorArray& normals, zPointArray& vThickness)
	{
		json j;
		zUtilsJsonHE meshJSON;

		points.clear();
		normals.clear();
		vThickness.clear();

		zFloatArray deviations;
		zDomainFloat devDomain (10,0);


		ifstream in_myfile;
		in_myfile.open(path.c_str());

		int lineCnt = 0;

		if (in_myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
			return false;
		}

		in_myfile >> j;
		in_myfile.close();

		// vertex attributes
		meshJSON.vertexAttributes.clear();
		meshJSON.vertexAttributes = (j["VertexAttributes"].get<vector<vector<double>>>());

		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{			
			vector<double> v_attrib = meshJSON.vertexAttributes[v.getId()];

			//if (v_attrib.size() == 10)
			//{
			zPoint p1(v_attrib[9], v_attrib[10], v_attrib[11]);
			zVector p2(v_attrib[12], v_attrib[13], v_attrib[14]);

			vThickness.push_back(p1);
			vThickness.push_back(p2);
			//}

		}


		//halfEdge attributes
		meshJSON.halfedgeAttributes.clear();
		meshJSON.halfedgeAttributes = (j["HalfedgeAttributes"].get<vector<vector<double>>>());


		for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{

			//if (he.getId() != 8)continue;
			vector<double> he_attrib = meshJSON.halfedgeAttributes[he.getId()];

			zPoint p (he_attrib[0], he_attrib[1], he_attrib[2]) ;
			zVector n(he_attrib[3], he_attrib[4], he_attrib[5]);

		
			
			
			float d = 0;

			he.getEdge().setColor(zColor(0, 0, 0, 1));

			if (n.length() > 0 && guideHalfEdge_planeFace[he.getId()].size() >0)
			{
				he.getEdge().setColor(zColor(0, 1, 0, 1));

				int fId = guideHalfEdge_planeFace[he.getId()][0];

				zItMeshFace f(o_planeMesh, fId);
				zVector nf = f.getNormal();
				zPoint pf = f.getCenter();

				if (n * nf < 0) n *= -1;

				d = n * nf;
				
				if (d < devDomain.min) devDomain.min = d;
				if (d > devDomain.max) devDomain.max = d;

				p = pf;
			}

			normals.push_back(n);
			deviations.push_back(d);

			points.push_back(p);

		}

		targetNormals.clear();
		targetNormals = normals;

		targetCenters.clear();
		targetCenters = points;

		

		zFnMesh fnPlaneMesh(o_planeMesh);
		fnPlaneMesh.setFaceColor(zColor(0.95, 0.95, 0.95, 1));

		printf("\n nf %i nT %i ", fnPlaneMesh.numPolygons(), targetNormals.size());

		zDomainColor colDomain(zColor(0, 1, 1), zColor(120, 1, 1));
		zDomainFloat inDomain(-1, 1);
		for (int i = 0; i < deviations.size(); i++)
		{
			if (deviations[i] != 0)
			{
				int fId = guideHalfEdge_planeFace[i][0];
				zItMeshFace f(o_planeMesh, fId);

				zColor fCol = coreUtils.blendColor(deviations[i], inDomain, colDomain, zHSV);
				f.setColor(fCol);
			}
		}

		printf("\n BRG dev | %1.6f  %1.6f ", devDomain.min, devDomain.max);

	}



}