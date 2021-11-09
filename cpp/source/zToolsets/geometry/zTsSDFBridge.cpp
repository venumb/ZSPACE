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

	ZSPACE_INLINE zTsSDFBridge::zTsSDFBridge() 
	{
		red = zColor(1, 0, 0, 1);
		yellow = zColor(1, 1, 0, 1); 
		green = zColor(0, 1, 0, 1);
		cyan = zColor(0, 1, 1, 1); 
		blue = zColor(0, 0, 1, 1); 
		magenta = zColor(1, 0, 1, 1);

		grey = zColor(0.5, 0.5, 0.5, 1);

		orange = zColor(1, 0.5, 0, 1);
	
	}


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

			if (v1 && v2 )  faceColors.push_back(blue); 
			else faceColors.push_back(grey);
		}

		zFnMesh fnCutPlanes(o_planeMesh);
		fnCutPlanes.create(positions, pCounts, pConnects);

		planeFace_colors.clear();
		planeFace_colors = faceColors;

		fnCutPlanes.setFaceColors(planeFace_colors);
	}

	ZSPACE_INLINE void zTsSDFBridge::createSplitMesh(double width, bool useThickMeshPoints)
	{

		zFnMesh fnGuideMesh(*o_guideMesh);
		zFnMesh fnGuideSmoothMesh(o_guideSmoothMesh);

		zFnMesh fnGuideThickMesh(*o_guideThickMesh);
		zPointArray thickPositions;
		fnGuideThickMesh.getVertexPositions(thickPositions);

		int guideNumV = fnGuideMesh.numVertices();

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

				if (useThickMeshPoints)
				{
					globalVertices.push_back(zGlobalVertex());
					globalVertices[n_gV].pos = thickPositions[v.getId() + guideNumV];;
					n_gV++;


					globalVertices.push_back(zGlobalVertex());
					globalVertices[n_gV].pos = thickPositions[v.getId() ];;
					n_gV++;
				}
				else
				{
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

				int step; 
				if (e.getColor() == red) step = 3;
				if (e.getColor() == orange) step = 1;

				bool print = false;
				
				if (he.getVertex().onBoundary() && he.getSym().getPrev().getStartVertex().onBoundary()) {}
				else
				{
					////if(print) printf("\n working 1");

					zColor f1 = he.getFace().getColor();
					zColor f2 = (he.getNext().getSym().getEdge().onBoundary()) ? f1 : he.getNext().getSym().getFace().getColor();

					if (f2 == f1){}
					else
					{						
						int gV;									

						//0
						zItMeshHalfEdge tempHE1 = he;
						for (int k = 1; k < step; k++) tempHE1 = tempHE1.getNext().getNext().getSym();
						
						gV = guideVertex_globalVertex[tempHE1.getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//1
						gV = guideVertex_globalVertex[tempHE1.getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//2
						zItMeshHalfEdge tempHE2 = he;
						for (int k = 1; k < step; k++) tempHE2 = tempHE2.getSym().getPrev().getPrev();

						gV = guideVertex_globalVertex[tempHE2.getSym().getPrev().getStartVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//3
						gV = guideVertex_globalVertex[tempHE2.getSym().getPrev().getStartVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);


						guideHalfEdge_planeFace[he.getId()].push_back(pCounts.size());
						//guideHalfEdge_planeFace[he.getNext().getId()].push_back(pCounts.size());
						//guideHalfEdge_planeFace[he.getSym().getPrev().getId()].push_back(pCounts.size());

						zItMeshHalfEdge tempHE3 = he;
						for (int k = 0; k < step; k++)
						{
							guideHalfEdge_planeFace[tempHE3.getNext().getId()].push_back(pCounts.size());
							
							tempHE3.getNext().getEdge().setWeight(4);
							
							tempHE3 = tempHE3.getNext().getNext().getSym();
						}

						zItMeshHalfEdge tempHE4 = he;
						for (int k = 0; k < step; k++)
						{
							guideHalfEdge_planeFace[tempHE4.getSym().getPrev().getId()].push_back(pCounts.size());
							tempHE4 = tempHE4.getSym().getPrev().getPrev();
						}

						pCounts.push_back(4);

						// color
						
						/*if (he.getFace().getColor() == blockColors[0] || he.getFace().getColor() == blockColors[1]) faceColors.push_back(grey);
						else*/ faceColors.push_back(red);
						//he.getNext().getEdge().setWeight(4);
					}
					
					/*if (f2 == f1) faceColors.push_back(grey);					
					else
					{
						faceColors.push_back(red);

						he.getNext().getEdge().setWeight(4);
					}*/

				}


				if (he.getNext().getNext().getVertex().onBoundary() && he.getNext().getVertex().onBoundary()) {}
				else
				{
					//if (print) printf("\n working 2");

					int gV;

					//0
					zItMeshHalfEdge tempHE1 = he;
					for (int k = 1; k < step; k++) tempHE1 = tempHE1.getNext().getNext().getSym();

					if (!tempHE1.getNext().getVertex().onBoundary())
					{
						gV = guideVertex_globalVertex[tempHE1.getNext().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//1
						gV = guideVertex_globalVertex[tempHE1.getNext().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//2
						gV = guideVertex_globalVertex[tempHE1.getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//3
						gV = guideVertex_globalVertex[tempHE1.getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);


						guideHalfEdge_planeFace[he.getId()].push_back(pCounts.size());
						guideHalfEdge_planeFace[tempHE1.getNext().getNext().getId()].push_back(pCounts.size());


						pCounts.push_back(4);
						faceColors.push_back(blue);

						//he.getNext().getNext().getEdge().setWeight(4);

						tempHE1.getNext().getNext().getEdge().setWeight(4);
					}

					
				}


				if (he.getSym().getNext().getVertex().onBoundary() && he.getPrev().getStartVertex().onBoundary()) {}
				else
				{
					//if (print) printf("\n working 3");

					zColor f1 = he.getSym().getFace().getColor();
					zColor f2 = (he.getSym().getNext().getSym().getEdge().onBoundary()) ? f1 : he.getSym().getNext().getSym().getFace().getColor();
					
					if (f2 == f1) {}
					else
					{
						int gV;

						//0
						zItMeshHalfEdge tempHE1 = he;
						for (int k = 1; k < step; k++) tempHE1 = tempHE1.getSym().getNext().getNext();

						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//1
						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//2
						zItMeshHalfEdge tempHE2 = he;
						for (int k = 1; k < step; k++) tempHE2 = tempHE2.getPrev().getPrev().getSym();

						gV = guideVertex_globalVertex[tempHE2.getPrev().getStartVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//3
						gV = guideVertex_globalVertex[tempHE2.getPrev().getStartVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);


						guideHalfEdge_planeFace[he.getSym().getId()].push_back(pCounts.size());
						//guideHalfEdge_planeFace[he.getSym().getNext().getId()].push_back(pCounts.size());
						//guideHalfEdge_planeFace[he.getPrev().getId()].push_back(pCounts.size());
						

						zItMeshHalfEdge tempHE3 = he;
						for (int k = 0; k < step; k++)
						{
							guideHalfEdge_planeFace[tempHE3.getSym().getNext().getId()].push_back(pCounts.size());
							
							tempHE3.getSym().getNext().getEdge().setWeight(4);
							tempHE3 = tempHE3.getSym().getNext().getNext();
						}

						zItMeshHalfEdge tempHE4 = he;
						for (int k = 0; k < step; k++)
						{
							guideHalfEdge_planeFace[tempHE4.getPrev().getId()].push_back(pCounts.size());
							tempHE4 = tempHE4.getPrev().getPrev().getSym();
						}
						
						pCounts.push_back(4);

						// color
						/*if (he.getFace().getColor() == blockColors[0] || he.getFace().getColor() == blockColors[1]) faceColors.push_back(grey);
						else*/ faceColors.push_back(red);
						//he.getSym().getNext().getEdge().setWeight(4);
					}

					//if (f2 == f1) faceColors.push_back(grey);
					//else
					//{
					//	
					//}

				}

				if (he.getSym().getNext().getNext().getVertex().onBoundary() && he.getSym().getNext().getVertex().onBoundary()) {}
				else
				{
					//if (print) printf("\n working 4");

					int gV;

					zItMeshHalfEdge tempHE1 = he;
					for (int k = 1; k < step; k++) tempHE1 = tempHE1.getSym().getNext().getNext();

					if (!tempHE1.getSym().getNext().getVertex().onBoundary())
					{
						//0
						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//1
						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//2
						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getVertex().getId()][0];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);

						//3
						gV = guideVertex_globalVertex[tempHE1.getSym().getNext().getVertex().getId()][1];
						globalVertices[gV].coincidentVertices.push_back(positions.size());
						planeVertex_globalVertex.push_back(gV);
						pConnects.push_back(positions.size());
						positions.push_back(globalVertices[gV].pos);


						guideHalfEdge_planeFace[he.getSym().getId()].push_back(pCounts.size());
						guideHalfEdge_planeFace[tempHE1.getSym().getNext().getNext().getId()].push_back(pCounts.size());

						pCounts.push_back(4);

						faceColors.push_back(blue);

						//he.getSym().getNext().getNext().getEdge().setWeight(4);
						tempHE1.getSym().getNext().getNext().getEdge().setWeight(4);
					}

					

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
			
		int mCounter = 0;
		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{
			if (!guide_PrintMedialEdgesBoolean[e.getId()]) continue;

			

			zItMeshHalfEdge cHe = e.getHalfEdge(0);

			if (cHe.getVertex().getValence() > 4) continue;
			if (cHe.getSym().getVertex().getValence() > 4) continue;

			mCounter++;

			// left
			if (!cHe.getNext().getEdge().onBoundary())
			{
				zItMeshHalfEdge temp = cHe.getSym();
				
				if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] ;

				

				if (currentFaceColor == red)
				{
					//printf("\n working p");
					if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0 || guideHalfEdge_planeFace[temp.getPrev().getId()].size() == 0) continue;

					int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
					int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

					planeFace_targetPair[f1_id] = f2_id;
					planeFace_targetPair[f2_id] = f1_id;

					int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
					int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

					planeFace_targetPair[f3_id] = f4_id;
					planeFace_targetPair[f4_id] = f3_id;

					//printf(" %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);
									

					faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = green;


					faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = green;
				}

				//if (currentFaceColor == grey)
				//{
				//	if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

				//	int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
				//	int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

				//	planeFace_targetPair[f1_id] = f2_id;
				//	planeFace_targetPair[f2_id] = f1_id;

				//	int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
				//	int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

				//	planeFace_targetPair[f3_id] = f4_id;
				//	planeFace_targetPair[f4_id] = f3_id;

				//	//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

				//	

				//	faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = cyan;
				//	faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = cyan;


				//	faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = cyan;
				//	faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = cyan;
				//}

			}

			////// SYM Edge

			cHe = e.getHalfEdge(1);

			// left
			if (!cHe.getNext().onBoundary())
			{
				zItMeshHalfEdge temp = cHe.getSym();
				
				
				if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]];
						
				
				if (currentFaceColor == red)
				{
					//printf("\n working p");
					if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0 || guideHalfEdge_planeFace[cHe.getNext().getSym().getId()].size() == 0) continue;

					int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
					int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

					planeFace_targetPair[f1_id] = f2_id;
					planeFace_targetPair[f2_id] = f1_id;

					int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
					int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

					planeFace_targetPair[f3_id] = f4_id;
					planeFace_targetPair[f4_id] = f3_id;

					//printf(" %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

					faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = green;


					faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = green;
					faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = green;
				}

				//if (currentFaceColor == grey)
				//{

				//	if (guideHalfEdge_planeFace[cHe.getNext().getId()].size() == 0) continue;

				//	int f1_id = guideHalfEdge_planeFace[cHe.getNext().getId()][0];
				//	int f2_id = guideHalfEdge_planeFace[temp.getPrev().getId()][0];

				//	planeFace_targetPair[f1_id] = f2_id;
				//	planeFace_targetPair[f2_id] = f1_id;

				//	int f3_id = guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0];
				//	int f4_id = guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0];

				//	planeFace_targetPair[f3_id] = f4_id;
				//	planeFace_targetPair[f4_id] = f3_id;

				//	//printf("\n %i %i | %i %i ", f1_id, f2_id, f3_id, f4_id);

				//	zColor green(0, 1, 0, 1);

				//	faceColors[guideHalfEdge_planeFace[cHe.getNext().getId()][0]] = cyan;
				//	faceColors[guideHalfEdge_planeFace[temp.getPrev().getId()][0]] = cyan;


				//	faceColors[guideHalfEdge_planeFace[cHe.getNext().getSym().getId()][0]] = cyan;
				//	faceColors[guideHalfEdge_planeFace[temp.getPrev().getSym().getId()][0]] = cyan;
				//}
			}


		}

		printf("\n num P Medials %i ", mCounter);

		/*int counter = 0;
		for (auto& p : planeFace_targetPair)
		{
			if (p != -1) printf("\n %i %i ", counter, p);
			counter++;

		}*/


		for (zItMeshEdge e(*o_guideMesh); !e.end(); e++)
		{

			if (!e.onBoundary())
			{
				if (guide_MedialEdgesBoolean[e.getId()]) continue;



				if (guideHalfEdge_planeFace[e.getHalfEdge(0).getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[e.getHalfEdge(0).getId()][0]];


				zColor f1 = e.getHalfEdge(0).getFace().getColor();
				zColor f2 = e.getHalfEdge(1).getFace().getColor();

				if (f1 == f2) {}
				else
				{
					if (currentFaceColor == blue)
					{
						faceColors[guideHalfEdge_planeFace[e.getHalfEdge(0).getId()][0]] = yellow;
						faceColors[guideHalfEdge_planeFace[e.getHalfEdge(1).getId()][0]] = yellow;
					}

				}
			}
			/*else
			{
				zItMeshHalfEdge he = (e.getHalfEdge(0).onBoundary()) ? e.getHalfEdge(1) : e.getHalfEdge(0);

				if (guideHalfEdge_planeFace[he.getId()].size() == 0) continue;
				zColor currentFaceColor = faceColors[guideHalfEdge_planeFace[he.getId()][0]];

				if (currentFaceColor == blue)
					faceColors[guideHalfEdge_planeFace[he.getId()][0]] = yellow;
			}*/




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

	ZSPACE_INLINE void zTsSDFBridge::createFieldMesh(zDomain<zPoint>& bb, int resX, int resY)
	{
		zFnMeshScalarField fnField(o_field);

		fnField.create(bb.min, bb.max, resX, resY, 1, true, false);


		zDomainColor dCol(red, green);
		fnField.setFieldColorDomain(dCol);
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zTsSDFBridge::setGuideMesh(zObjMesh& _o_guideMesh)
	{
		o_guideMesh = &_o_guideMesh;		
	}

	ZSPACE_INLINE void zTsSDFBridge::setThickGuideMesh(zObjMesh& _o_guideThickMesh)
	{
		o_guideThickMesh = &_o_guideThickMesh;
	}

	ZSPACE_INLINE void zTsSDFBridge::setSmoothThickGuideMesh(zObjMesh& _o_guideSmoothThickMesh)
	{
		o_guideSmoothThickMesh = &_o_guideSmoothThickMesh;
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
			
			v.setColor(green);
			printMedialVerticesBoolean[printMedialVertices[i]] = true;

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::setPrintRails(const vector<int>& _fixedVertices)
	{

		printRailVertices.clear();
		printRailVertices = _fixedVertices;



		printRailVerticesBoolean.clear();

		zFnMesh fnGuideMesh(*o_guideMesh);

		printRailVerticesBoolean.assign(fnGuideMesh.numVertices(), false);

		for (int i = 0; i < printRailVertices.size(); i++)
		{
			zItMeshVertex v(*o_guideMesh, printRailVertices[i]);

			v.setColor(green);
			printRailVerticesBoolean[printRailVertices[i]] = true;

		}
	}

	ZSPACE_INLINE void zTsSDFBridge::resetPlaneColors()
	{
		zFnMesh fnPlaneMesh(o_planeMesh);
		fnPlaneMesh.setFaceColors(planeFace_colors);
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

	ZSPACE_INLINE zPointArray zTsSDFBridge::getBlockPrintStartEndPoints(int blockId)
	{
		if (blockId >= printBlocks.size()) return zPointArray();

		return printBlocks[blockId].printStartEndPoints;;

	}

	ZSPACE_INLINE vector<zTransform> zTsSDFBridge::getBlockFrames(int blockId)
	{

		if (blockId >= printBlocks.size()) return vector<zTransform>();

		return printBlocks[blockId].sectionFrames;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockGraphs(int blockId, int& numGraphs, zPointArray& startPoints, zPointArray& endPoints)
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

		startPoints = printBlocks[blockId].startPos;
		endPoints = printBlocks[blockId].endPos;

		return out;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockRaftGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= printBlocks.size()) return out;

		numGraphs = printBlocks[blockId].o_raftGraphs.size();

		if (numGraphs == 0)return out;

		for (auto& graph : printBlocks[blockId].o_raftGraphs)
		{
			out.push_back(&graph);
		}

		return out;
	}

	ZSPACE_INLINE zObjGraphPointerArray zTsSDFBridge::getBlockGuideGraphs(int blockId, int& numGraphs)
	{
		zObjGraphPointerArray out;
		numGraphs = 0;

		if (blockId >= printBlocks.size()) return out;

		numGraphs = printBlocks[blockId].o_sectionGuideGraphs.size();

		if (numGraphs == 0)return out;

		for (auto& graph : printBlocks[blockId].o_sectionGuideGraphs)
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


	ZSPACE_INLINE bool zTsSDFBridge::onBoundaryBlock(int blockId)
	{
		if (blockId >= printBlocks.size()) return false;			
	
		return (printBlocks[blockId].right_sideFaces.size() != printBlocks[blockId].left_sideFaces.size()) ;
	}

	ZSPACE_INLINE void zTsSDFBridge::computeMedialEdgesfromConstraints(const vector<int>& pattern)
	{
		blockColors.clear();

		guide_MedialEdges.clear();
		guide_MedialEdgesBoolean.clear();

		zFnMesh fn_guideMesh(*o_guideMesh);
		guide_MedialEdgesBoolean.assign(fn_guideMesh.numEdges(), false);

		fn_guideMesh.setEdgeColor(zColor());

		// strart from constraint and walk till its hits another constraint
		//discard boundary edge walks

		for (int j =0; j< fixedVertices.size(); j++)
		{
			zItMeshVertex v(*o_guideMesh, fixedVertices[j]);

			int stepPattern = pattern[j];

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

					zColor col;
					if (stepPattern != 1) col = red;
					if (stepPattern == 1) col = orange;

					he.getEdge().setColor(col);

					guide_MedialEdges.push_back(zIntPair(he.getEdge().getId(), fixedVertices[j]));

					// add color to block
					zColor bCol = he.getFace().getColor();

					bool checkColRepeat = false;
					for (zColor &blockCol : blockColors)
					{
						if (bCol == blockCol)
						{
							checkColRepeat = true;
							break;
						}
					}

					if (!checkColRepeat) blockColors.push_back(bCol);
				}

				he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}

		/*for (zColor& blockCol : blockColors)
		{
			printf("\n bCol %1.2f %1.2f %1.2f ", blockCol.r, blockCol.g, blockCol.b);
		}*/

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


		

		//for (auto mEdge : guide_MedialEdges)
		//{
		//	zItMeshEdge e(*o_guideMesh, mEdge.first);
		//	zColor col(1, 0.5, 0, 1);
		//	e.setColor(col);
		//}



		// SMOOTH MESH 

		

		guideSmooth_MedialEdges.clear();
		guideSmooth_MedialEdgesBoolean.clear();

		zFnMesh fn_guideSmoothMesh(o_guideSmoothMesh);
		guideSmooth_MedialEdgesBoolean.assign(fn_guideSmoothMesh.numEdges(), false);

		fn_guideSmoothMesh.setEdgeColor(zColor());

		// start from constraint and walk till its hits another constraint
		//discard boundary edge walks
		for (int j = 0; j < fixedVertices.size(); j++)
		{
			zItMeshVertex v(o_guideSmoothMesh, fixedVertices[j]);
			int stepPattern = pattern[j];

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

					zColor col;
					if (stepPattern != 1) col = red;
					if (stepPattern == 1) col = orange;

					he.getEdge().setColor(col);

					guideSmooth_MedialEdges.push_back(zIntPair(he.getEdge().getId(), fixedVertices[j]));
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

		

	/*	for (auto mEdge : guideSmooth_MedialEdges)
		{
			zItMeshEdge e(o_guideSmoothMesh, mEdge.first);
			zColor col(1, 0.5, 0, 1);
			e.setColor(col);
		}*/
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

	ZSPACE_INLINE void zTsSDFBridge::computePrintRailEdgesfromRailVertices()
	{

		guide_PrintRailEdges.clear();
		guide_PrintRailEdgesBoolean.clear();

		zFnMesh fn_guideMesh(*o_guideMesh);
		guide_PrintRailEdgesBoolean.assign(fn_guideMesh.numEdges(), false);

		// strart from constraint and walk till its hits another constraint
		//discard boundary edge walks

		for (auto fV : printRailVertices)
		{
			zItMeshVertex v(*o_guideMesh, fV);

			zItMeshHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			zItMeshHalfEdge start, he;

			// get halfedge of internal edge
			bool internalEdge = false;

			if (v.checkValency(2))
			{
				start = (cHEdges[0].getVertex().getPosition().z > cHEdges[1].getVertex().getPosition().z) ? cHEdges[0] : cHEdges[1];
				he = start;
				internalEdge = true;

			}
			else
			{
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
			}
			

			if (!internalEdge) continue;

			bool exit = false;

			// walk to find Rail edges
			do
			{
				if (printRailVerticesBoolean[he.getVertex().getId()]) exit = true;

				if (!guide_PrintRailEdgesBoolean[he.getEdge().getId()])
				{
					guide_PrintRailEdgesBoolean[he.getEdge().getId()] = true;

					guide_PrintRailEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				if(!he.onBoundary()) he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}




		///fn_guideMesh.setEdgeColor(zColor());

		for (auto mEdge : guide_PrintRailEdges)
		{
			zItMeshEdge e(*o_guideMesh, mEdge.first);
			e.setColor(green);
		}



		// SMOOTH MESH 

		guideSmooth_PrintRailEdges.clear();
		guideSmooth_PrintRailEdgesBoolean.clear();

		zFnMesh fn_guideSmoothMesh(o_guideSmoothMesh);
		guideSmooth_PrintRailEdgesBoolean.assign(fn_guideSmoothMesh.numEdges(), false);

		// start from constraint and walk till its hits another constraint
		//discard boundary edge walks
		for (auto fV : printRailVertices)
		{
			zItMeshVertex v(o_guideSmoothMesh, fV);

			zItMeshHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			zItMeshHalfEdge start, he;

			// get halfedge of internal edge
			bool internalEdge = false;

			if (v.checkValency(2))
			{
				start = (cHEdges[0].getVertex().getPosition().z > cHEdges[1].getVertex().getPosition().z) ? cHEdges[0] : cHEdges[1];
				he = start;
				internalEdge = true;

			}
			else
			{
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
			}
			

			if (!internalEdge) continue;

			bool exit = false;

			// walk to find Rail edges
			do
			{
				if (he.getVertex().getId() < fn_guideMesh.numVertices())
				{
					if (printRailVerticesBoolean[he.getVertex().getId()]) exit = true;
				}


				if (!guideSmooth_PrintRailEdgesBoolean[he.getEdge().getId()])
				{

					guideSmooth_PrintRailEdgesBoolean[he.getEdge().getId()] = true;

					guideSmooth_PrintRailEdges.push_back(zIntPair(he.getEdge().getId(), fV));
				}

				if (!he.onBoundary()) he = he.getNext().getSym();
				he = he.getNext();

				if (he == start) exit = true;

			} while (!exit);
		}


		for (auto mEdge : guideSmooth_PrintRailEdges)
		{
			zItMeshEdge e(o_guideSmoothMesh, mEdge.first);
			e.setColor(green);
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlocks(int blockId,float printLayerDepth, float printLayerWidth, zDomainFloat neopreneOffset, float raftLayerWidth, const zIntArray& flipBlockIds, bool compBLOCKS, bool compFRAMES,  bool compSDF)
	{
		zFnMesh  fn_guideMesh(*o_guideMesh);
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);
		zFnMesh fn_planeMesh(o_planeMesh);

		if (compBLOCKS)
		{
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

					bool footingBlock = false;
					zColor black(0, 0, 0, 1);
					if (he.getFace().getColor() == black)
					{
						he = he.getNext().getSym().getNext();
						footingBlock = true;
					}

					bool boundaryBlock = true;
					zColor f1 = he.getSym().getFace().getColor();
					zColor f0 = he.getFace().getColor();
					if (f1 == f0) boundaryBlock = false;


					zItMeshHalfEdge temp = he;
					zColor current = temp.getFace().getColor();

					bool checkSymmetryColor = false;
					if (boundaryBlock)
					{
						if (f1 == blockColors[1] || f1 == blockColors[2])
						{
							checkSymmetryColor = true;
							current = f1;
						}
					}

					bool incrCounter = false;

					if (!edgeVisited[temp.getEdge().getId()])
					{
						printBlocks.push_back(zPrintBlock());
						printBlocks[macroBlockCOunter].id = macroBlockCOunter;

						printBlocks[macroBlockCOunter].guideMesh_interiorVertex = temp.getVertex().getId();
						printBlocks[macroBlockCOunter].guideMesh_interiorVertexPosition = temp.getVertex().getPosition();

						printBlocks[macroBlockCOunter].footingBlock = footingBlock;

						printBlocks[macroBlockCOunter].color = current;

						computePrintBlockRailInteriorVertex(printBlocks[macroBlockCOunter], temp, boundaryBlock);

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

							if (temp.getFace().getColor() == black) { printBlocks[macroBlockCOunter].footingBlock = true; }
							else
							{
								if (!edgeVisited[temp.getEdge().getId()])
								{
									edgeVisited[temp.getEdge().getId()] = true;
									addConvexBlocks(printBlocks[macroBlockCOunter], temp, boundaryBlock);
								}
							}



						}

						if (!tempExit)
						{
							if (checkSymmetryColor)
							{
								if (temp.getSym().getFace().getColor() == current) {}
								else
								{
									tempExit = true;
									he = temp;
								}
							}
							else
							{
								if (temp.getFace().getColor() == current) {}
								else
								{
									tempExit = true;
									he = temp;
								}
							}

						}

						if (!tempExit)
						{
							if (!edgeVisited[temp.getEdge().getId()])
							{
								edgeVisited[temp.getEdge().getId()] = true;
								addConvexBlocks(printBlocks[macroBlockCOunter], temp, boundaryBlock);

							}

						}

						temp = temp.getNext().getSym();
						temp = temp.getNext();

						//if (temp.getFace().getColor() == black)
						//{
						//	printBlocks[macroBlockCOunter].footingBlock = true;
						//	//tempExit = true;
						//}



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

			printf("\n numBlocks %i ", macroBlockCOunter);
		}
		

		if (compFRAMES)
		{
			int footingBlock = 0;



			for (auto& mB : printBlocks)
			{
				if (mB.id != blockId) continue;

				bool flipBlock = false;
				for (auto& flipId : flipBlockIds)
				{
					if (mB.id == flipId) flipBlock = true;
				}

				mB.right_BoundaryFaces.clear();
				mB.left_BoundaryFaces.clear();

				mB.right_sideFaces.clear();
				mB.left_sideFaces.clear();

				addPrintBlockBoundary(mB);
				computePrintBlockIntersections(mB, flipBlockIds);

				bool useRails = true;

				if (flipBlock) computePrintBlockFrames(mB, printLayerDepth, neopreneOffset.min, (mB.footingBlock) ? neopreneOffset.max : neopreneOffset.min, flipBlockIds, useRails);
				else 	computePrintBlockFrames(mB, printLayerDepth, (mB.footingBlock) ? neopreneOffset.max : neopreneOffset.min, neopreneOffset.min, flipBlockIds, useRails);

				printf("\n %i | %i %i %i | %i %i %i ", mB.id, mB.rightBlocks.size(), mB.right_BoundaryFaces.size(), mB.right_sideFaces.size(), mB.leftBlocks.size(), mB.left_BoundaryFaces.size(), mB.left_sideFaces.size());
				printf("\n %i | %i %i %i ", mB.id, mB.intersectionPoints.size(), mB.right_sectionPlaneFace_GuideSmoothEdge.size(), mB.left_sectionPlaneFace_GuideSmoothEdge.size());
				printf("\n %i | %1.2f %1.2f  ", mB.id, mB.right_planeAngle, mB.left_planeAngle);
				printf("\n %i | %i  %s ", mB.id, mB.sectionFrames.size(), (mB.footingBlock) ? "T" : "F");
				printf("\n %i | %s \n", mB.id, (flipBlock) ? "flip T" : "flip F");
				//printf("\n %i , ", mB.id );

				//printf("\n %i , %1.3f, %1.3f, %1.3f ", mB.id, mB.guideMesh_interiorVertexPosition.x, mB.guideMesh_interiorVertexPosition.y, mB.guideMesh_interiorVertexPosition.z);

				if (mB.footingBlock) footingBlock++;



				//if (mB.id == blockId)
				//{



				if (mB.right_sideFaces.size() == mB.left_sideFaces.size())
				{
					computePrintBlockSections_Internal(mB);
				}
				else
				{
					computePrintBlockSections_Boundary(mB);

				}

				checkPrintLayerHeights(mB);

				//computeSDF(mB, printLayerWidth, neopreneOffset.min, raftLayerWidth);
			}
		}

		
		if (compSDF)
		{
			for (auto& mB : printBlocks)
			{
				if (mB.id != blockId) continue;

				computeSDF(mB, printLayerWidth, neopreneOffset.min, raftLayerWidth);
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

	ZSPACE_INLINE void zTsSDFBridge::computeSDF(zPrintBlock& _block, float printWidth, float neopreneOffset, float raftWidth)
	{
		_block.o_contourGraphs.clear();
		_block.o_contourGraphs.assign(_block.o_sectionGraphs.size(), zObjGraph());

		_block.o_trimGraphs.clear();
		_block.o_trimGraphs.assign(_block.o_sectionGraphs.size(), zObjGraph());

		_block.o_raftGraphs.clear();
		_block.o_raftGraphs.assign(4, zObjGraph());

		printf("\n num frames : %i ", _block.o_sectionGraphs.size());

		int raftId = 0; 
		int r0 = 0;
		int r1 = floor(_block.o_sectionGraphs.size() * 0.5) -1;
		int r2 = floor(_block.o_sectionGraphs.size() * 0.5);
		int r3 = (_block.o_sectionGraphs.size()) -1;

		printf("\n r: %i  %i %i %i ", r0, r1, r2, r3);
		int end = (_block.right_sideFaces.size() != _block.left_sideFaces.size()) ? _block.o_sectionGraphs.size() : floor(_block.o_sectionGraphs.size() * 0.5);
		
		bool exportBMP = true;

		for (int j = 1; j <2; j++)
		{
			int kStart = ((onBoundaryBlock(_block.id)) && _block.right_BoundaryFaces.size() == 0) ? 1 : 0;
			int kEnd = ((onBoundaryBlock(_block.id)) && _block.left_BoundaryFaces.size() == 0) ? 1 : 2;



			for (int k = kStart; k < kEnd; k++)
			{
				int i = (k == 0) ? j : j + end;

				if (onBoundaryBlock(_block.id)) i = j;

				//if (i == r0 || i == r2) continue;

				if (_block.right_sideFaces.size() != _block.left_sideFaces.size())
				{

					if (i == r0)computeBlockSDF_Boundary(_block, i, printWidth, neopreneOffset, true, 0);
					//else if (i == r2 ) computeBlockSDF_Boundary(_block, i, printWidth, neopreneOffset, true, 1);
					//else if (i == r1) computeBlockSDF_Boundary(_block, i, printWidth, neopreneOffset, true, 2);
					//else if (i == r3) computeBlockSDF_Boundary(_block, i, printWidth, neopreneOffset, true, 3);

					else computeBlockSDF_Boundary(_block, i, printWidth, neopreneOffset, false, 0);
				}
				else
				{
					if (i == r0 )computeBlockSDF_Internal(_block, i,j, (k == 0), printWidth, neopreneOffset, true, 0, raftWidth, false);
					else if (i == r2) computeBlockSDF_Internal(_block, i, j, (k == 0), printWidth, neopreneOffset, true, 1, raftWidth, false);
					//else if (i == r1) computeBlockSDF_Internal(_block, i, printWidth, neopreneOffset, true, 2, raftWidth);
					//else if (i == r3) computeBlockSDF_Internal(_block, i, printWidth, neopreneOffset, true, 3, raftWidth);

					else computeBlockSDF_Internal(_block, i, j, (k == 0), printWidth, neopreneOffset, false, 0, raftWidth, exportBMP);
				}
			}
			
		}
				
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

				//int gV = planeVertex_globalVertex[i];
				//if (globalVertices[gV].coincidentVertices.size() == 12) fixed = true;

				//if(globalFixedVertices[gV]) fixed = true;
				
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

		//zDomainColor colDomain(zColor(180, 1, 1), zColor(360, 1, 1));
		zDomainColor colDomain(zColor(0, 0.55,0.95, 1), zColor(0.95, 0, 0.55,1));

		for (zItMeshFace f(o_planeMesh); !f.end(); f++)
		{
			int i = f.getId();


			//zColor col = coreUtils.blendColor(fVolumes[i], deviation, colDomain, zHSV);
			//if (fVolumes[i] < tolerance) col = zColor(120, 1, 1);
			//f.setColor(col);

			if (fVolumes[i] < tolerance) f.setColor(colDomain.min);
			else f.setColor(colDomain.max);

		}

		if(printInfo) printf("\n planarity tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);

		if (out)
		{
			printf("\n planarity tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);
			fnPlaneMesh.setFaceColors(planeFace_colors);
		}

		return out;

	}

	ZSPACE_INLINE bool zTsSDFBridge::alignFacePairs(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations, bool printInfo)
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
				if (globalVertices[gV].coincidentVertices.size() == 12) fixed = true;

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

				if (planeFace_targetPair[i] != -1)
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

		

		fNormals.clear();
		fnPlaneMesh.getFaceNormals(fNormals);

		zDomainFloat devDomain(10, 0);
		zFloatArray deviations;

		int fCounter = 0;
		for (auto fPair : planeFace_targetPair)
		{
			float d = 1;
			if (fPair != -1)
			{
				fNormals[fCounter] = (fNormals[fCounter] + fNormals[fPair]) * 0.5;
				fNormalTargets[fCounter].normalize();

				d = fNormals[fCounter] * fNormals[fPair];

				if (d < devDomain.min) devDomain.min = d;
				if (d > devDomain.max) devDomain.max = d;
			}

			deviations.push_back(d);

			fCounter++;

		}

		deviation = devDomain;

		// FOR ANIMATION
		zDomainColor colDomain(zColor(180, 1, 1), zColor(360, 1, 1));


		fCounter = 0;
		for (auto fPair : planeFace_targetPair)
		{
			zColor col = (deviations[fCounter] > 0.999999) ? colDomain.min : colDomain.max;

			zItMeshFace f(o_planeMesh, fCounter);
			f.setColor(col);

			//if (fCounter == 0) printf("\n working");

			fCounter++;
		}
		
		///////////////////////////////////////////////////////////

		if (printInfo) printf("\n align face pairs tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);

		if (out && printInfo)
		{
			printf("\n align face pairs tolerance : %1.7f minDeviation : %1.7f , maxDeviation: %1.7f ", tolerance, deviation.min, deviation.max);
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

				//int gV = planeVertex_globalVertex[i];
				//if (globalVertices[gV].coincidentVertices.size() == 12) fixed = true;

				//if (globalFixedVertices[gV]) fixed = true;

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
			
		for (int k = 0; k < numIterations; k++)
		{
			fCenters.clear();
			fnPlaneMesh.getMeshFaceVolumes(fTris, fCenters, fVolumes, true);

			fNormalTargets.clear();
			fNormals.clear();
			fnPlaneMesh.getFaceNormals(fNormals);
			fNormalTargets = fNormals;

			int fCounter = 0;
			/*for (auto fPair : planeFace_targetPair)
			{
				if (fPair != -1)
				{
					fNormalTargets[fCounter] = (fNormals[fCounter] + fNormals[fPair]) * 0.5;
					fNormalTargets[fCounter].normalize();
				}			
				fCounter++;

			}*/
						
						

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
						zPoint p = targetCenters[i] /*f.getCenter()*/;
						

						vector<int> fVerts;
						f.getVertices(fVerts);

						for (int j = 0; j < fVerts.size(); j++)
						{
							double dist = coreUtils.minDist_Point_Plane(positions[fVerts[j]], p /*fCenters[fId]*/, targetNormals[i]);
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

		zDomainFloat dev_distDomain(10, 0);
		zFloatArray deviationsDist;

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

					float dist = abs(coreUtils.minDist_Point_Plane(targetCenters[i], p, n));

					if (dist < dev_distDomain.min) dev_distDomain.min = dist;
					if (dist > dev_distDomain.max) dev_distDomain.max = dist;
				}
			}

			deviationsBRG.push_back(d);
		}

		if (printInfo)
		{
			printf("\n \n  planarity tolerance : %1.7f,  dev | %1.7f %1.7f ", tolerance, deviation.min, deviation.max);

			printf("\n dotProduct tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, devBRGDomain.min, devBRGDomain.max);

			printf("\n dist tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, dev_distDomain.min, dev_distDomain.max);
		}

		if (out)
		{
			printf("\n \n  planarity tolerance : %1.7f,  dev | %1.7f %1.7f ", tolerance, deviation.min, deviation.max);

			printf("\n dotProduct tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, devBRGDomain.min, devBRGDomain.max);

			printf("\n dist tolerance : %1.7f,  dev | %1.7f  %1.7f ", tolerance, dev_distDomain.min, dev_distDomain.max);
		}

		return out;
	}

	ZSPACE_INLINE bool zTsSDFBridge::updateSmoothMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations, bool printInfo)
	{
		zFnMesh fnGuideSmoothMesh(o_guideSmoothMesh);

		zFnMesh fnPlaneMesh(o_planeMesh);
		fnPlaneMesh.setFaceColors(planeFace_colors);


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

		if( printInfo || out)	printf("\n move star | tolerance : %1.4f minDeviation : %1.6f , maxDeviation: %1.6f ", tolerance, deviation.min, deviation.max);

		return out;
	}

	ZSPACE_INLINE bool zTsSDFBridge::updateGuideMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance, int numIterations)
	{
		//zFnMesh fnGuideMesh(*o_guideMesh);
		//
		//// create particles if it doesnt exist
		//if (fnGuideParticles.size() != fnGuideMesh.numVertices())
		//{
		//	fnGuideParticles.clear();
		//	o_guideParticles.clear();


		//	for (int i = 0; i < fnGuideMesh.numVertices(); i++)
		//	{
		//		bool fixed = false;

		//		zObjParticle p;
		//		p.particle = zParticle(o_guideMesh->mesh.vertexPositions[i], fixed);
		//		o_guideParticles.push_back(p);
		//	}

		//	for (int i = 0; i < o_guideParticles.size(); i++)
		//	{
		//		fnGuideParticles.push_back(zFnParticle(o_guideParticles[i]));
		//	}

		//	fnGuideMesh.getVertexPositions(orig_GuideMeshPositions);
		//}


		//
		//

		//zVector* positions = fnGuideMesh.getRawVertexPositions();

		//for (int k = 0; k < numIterations; k++)
		//{
		//	for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		//	{
		//		//if (v.getValence() != 6) continue;

		//		zItMeshHalfEdgeArray cHEdges;
		//		v.getConnectedHalfEdges(cHEdges);

		//		int vertexId = v.getId();

		//		for (auto cHE : cHEdges)
		//		{
		//			if (guide_MedialEdgesBoolean[cHE.getEdge().getId()]) continue;
		//			if (guideHalfEdge_planeFace[cHE.getId()].size() == 0) continue;


		//			int fId = guideHalfEdge_planeFace[cHE.getId()][0];
		//			if (fId == -1) continue;

		//			zItMeshFace f(o_planeMesh, fId);

		//			if (f.getColor() == blue)
		//			{
		//				zVector n = f.getNormal();
		//				zPoint p = f.getCenter();

		//				double dist = coreUtils.minDist_Point_Plane(positions[vertexId], p, n);
		//				zVector pForce = n * dist * -1.0;

		//				fnGuideParticles[vertexId].addForce(pForce);
		//			}

		//			
		//		}

		//		// be close to original
		//		zVector pForce1 = orig_GuideMeshPositions[vertexId] - positions[vertexId];
		//		pForce1.normalize();
		//		float dist = orig_GuideMeshPositions[vertexId].distanceTo(positions[vertexId]);

		//		//pForce1 *= dist * 0.5;

		//		//if(dist > distanceTolerance) fnGuideParticles[vertexId].addForce(pForce1);

		//	}

		//	// update positions
		//	for (int i = 0; i < fnGuideParticles.size(); i++)
		//	{
		//		fnGuideParticles[i].integrateForces(dT, type);
		//		fnGuideParticles[i].updateParticle(true);
		//	}
		//}

		bool out = true;

		//deviation.min = 1000;
		//deviation.max = 0;

		//for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		//{
		//	//if (v.getValence() != 6) continue;

		//	zItMeshHalfEdgeArray cHEdges;
		//	v.getConnectedHalfEdges(cHEdges);

		//	int smoothMeshVertexId = v.getId();

		//	for (auto cHE : cHEdges)
		//	{
		//		if (guide_MedialEdgesBoolean[cHE.getEdge().getId()]) continue;
		//		if (guideHalfEdge_planeFace[cHE.getId()].size() == 0) continue;

		//		int fId = guideHalfEdge_planeFace[cHE.getId()][0];
		//		if (fId == -1) continue;

		//		

		//		zItMeshFace f(o_planeMesh, fId);

		//		if (f.getColor() == blue)
		//		{
		//			zVector n = f.getNormal();
		//			zPoint p = f.getCenter();

		//			double dist = coreUtils.minDist_Point_Plane(positions[smoothMeshVertexId], p, n);

		//			if (abs(dist) < deviation.min) deviation.min = abs(dist);
		//			if (abs(dist) > deviation.max) deviation.max = abs(dist);
		//		}
		//		
		//	}
		//}


		//if (deviation.max > tolerance) out = false;

		//printf("\n move guide mesh | tolerance : %1.4f minDeviation : %1.6f , maxDeviation: %1.6f ", tolerance, deviation.min, deviation.max);

		//if (out)
		//{
		//	createSmoothGuideMesh(1);

		//	computeMedialEdgesfromConstraints();

		//	computePrintMedialEdgesfromMedialVertices();
		//}

		return out;
	}

	ZSPACE_INLINE void zTsSDFBridge::addPrintBlockBoundary(zPrintBlock& mB)
	{
		
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

	ZSPACE_INLINE void zTsSDFBridge::addConvexBlocks(zPrintBlock& mBlock, zItMeshHalfEdge& guideMesh_halfedge, bool boundaryBlock)
	{
		//left block
		if (!guideMesh_halfedge.getNext().getNext().getEdge().onBoundary())
		{
		
			bool addBlock = true; 

			if (boundaryBlock)
			{
				if (guideMesh_halfedge.getFace().getColor() == blockColors[1] || guideMesh_halfedge.getFace().getColor() == blockColors[2])
					addBlock = true;
				else addBlock = false;
			}


			if (addBlock)
			{
				int l_id = mBlock.leftBlocks.size();
				mBlock.leftBlocks.push_back(zConvexBlock());

				if (!guideMesh_halfedge.getSym().getNext().getNext().getEdge().onBoundary())
				{
					zItMeshHalfEdge tempHE = guideMesh_halfedge;

					while (!guide_MedialEdgesBoolean[tempHE.getNext().getNext().getEdge().getId()])
					{
						tempHE = tempHE.getNext().getNext().getSym();
					}

					zItMeshEdge l_e = tempHE.getNext().getNext().getEdge();
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

			
		}


		//right block
		if (!guideMesh_halfedge.getSym().getNext().getNext().getEdge().onBoundary())
		{
			bool addBlock = true;

			if (boundaryBlock)
			{
				if (guideMesh_halfedge.getSym().getFace().getColor() == blockColors[1] || guideMesh_halfedge.getSym().getFace().getColor() == blockColors[2])
					addBlock = true;
				else addBlock = false;				
			}

			if (addBlock)
			{
				int r_id = mBlock.rightBlocks.size();
				mBlock.rightBlocks.push_back(zConvexBlock());

				if (!guideMesh_halfedge.getNext().getNext().getEdge().onBoundary())
				{
					zItMeshHalfEdge tempHE = guideMesh_halfedge;

					while (!guide_MedialEdgesBoolean[tempHE.getSym().getNext().getNext().getEdge().getId()])
					{
						tempHE = tempHE.getSym().getNext().getNext();
					}

					zItMeshEdge r_e = tempHE.getSym().getNext().getNext().getEdge();
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

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockIntersections(zPrintBlock& _block, const zIntArray& FlipBlockIds)
	{
		//if (fixedVerticesBoolean[guideMesh_V.getId()]) return;

		zFnMesh  fn_guideMesh(*o_guideMesh);
		int blockId = _block.id;


		zItMeshVertex guideMesh_V(*o_guideMesh, _block.guideMesh_interiorVertex);
		// start vertex
		

		zItMeshVertex guideSmoothMesh_V(o_guideSmoothMesh, guideMesh_V.getId());

		zItMeshHalfEdgeArray cHEdges;
		guideSmoothMesh_V.getConnectedHalfEdges(cHEdges);

		int numPlaneIntersections = 0;
		
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

					/*if (walkHe.getVertex().getId() < fn_guideMesh.numVertices())
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
					}*/

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
									numPlaneIntersections++;
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
									numPlaneIntersections++;
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
		
		//printf("\n  %i  intersections %i  %i ", _block.id, _block.intersectionPoints.size(), numPlaneIntersections);

		if (_block.intersectionPoints.size() == 2)
		{
			zPoint p0 = _block.intersectionPoints[0];
			zPoint p1 = _block.intersectionPoints[1];

			if (p1.z < p0.z)
			{
				// flip start end planes
				_block.intersectionPoints[0]= p1;
				_block.intersectionPoints[1] = p0;

				if (_block.right_BoundaryFaces.size() > 0)
				{
					zIntPair R0 = _block.right_sectionPlaneFace_GuideSmoothEdge[0];
					zIntPair R1 = _block.right_sectionPlaneFace_GuideSmoothEdge[1];

					_block.right_sectionPlaneFace_GuideSmoothEdge[0] = R1;
					_block.right_sectionPlaneFace_GuideSmoothEdge[1] = R0;
				}
				
				if (_block.left_BoundaryFaces.size() > 0)
				{
					zIntPair L0 = _block.left_sectionPlaneFace_GuideSmoothEdge[0];
					zIntPair L1 = _block.left_sectionPlaneFace_GuideSmoothEdge[1];

					_block.left_sectionPlaneFace_GuideSmoothEdge[0] = L1;
					_block.left_sectionPlaneFace_GuideSmoothEdge[1] = L0;
				}
				

			}
		}
				

		for (auto flipId : FlipBlockIds)
		{
			if (_block.intersectionPoints.size() == 2 && _block.id == flipId)
			{
				zPoint p0 = _block.intersectionPoints[0];
				zPoint p1 = _block.intersectionPoints[1];

				//if (p1.z < p0.z)
				//{
					// flip start end planes
				_block.intersectionPoints[0] = p1;
				_block.intersectionPoints[1] = p0;

				zIntPair R0 = _block.right_sectionPlaneFace_GuideSmoothEdge[0];
				zIntPair R1 = _block.right_sectionPlaneFace_GuideSmoothEdge[1];

				_block.right_sectionPlaneFace_GuideSmoothEdge[0] = R1;
				_block.right_sectionPlaneFace_GuideSmoothEdge[1] = R0;

				zIntPair L0 = _block.left_sectionPlaneFace_GuideSmoothEdge[0];
				zIntPair L1 = _block.left_sectionPlaneFace_GuideSmoothEdge[1];

				_block.left_sectionPlaneFace_GuideSmoothEdge[0] = L1;
				_block.left_sectionPlaneFace_GuideSmoothEdge[1] = L0;

				//}
			}
		}
		

	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockRailIntersections(zPrintBlock& _block, int& railInteriorVertex, zPoint& startPlaneOrigin, zVector& startPlaneNorm, zPoint& endPlaneOrigin, zVector& endPlaneNorm, const zIntArray& FlipBlockIds, zItMeshHalfEdgeArray& railIntersectionHE, zPointArray& railIntersectionPoints)
	{
		//if (fixedVerticesBoolean[guideMesh_V.getId()]) return;

		zFnMesh  fn_guideMesh(*o_guideMesh);
		int blockId = _block.id;
			


		zItMeshVertex guideMesh_V (*o_guideMesh, railInteriorVertex);
		// start vertex


		zItMeshVertex guideSmoothMesh_V(o_guideSmoothMesh, guideMesh_V.getId());

		zItMeshHalfEdgeArray cHEdges;
		guideSmoothMesh_V.getConnectedHalfEdges(cHEdges);

		int numPlaneIntersections = 0;

		// Block Intersection
		for (auto& cHe : cHEdges)
		{
			zItMeshHalfEdge walkHe = cHe;

			// walk along medial edges till it intersects a block face
			if (guideSmooth_PrintRailEdgesBoolean[cHe.getEdge().getId()])
			{
				bool exit = false;

				do
				{
					bool planeIntersection = false;
					bool computeBlockIntersectionFace = false;
					zPoint intersectionPoint;
										

					zPoint eE = walkHe.getVertex().getPosition();
					zPoint eS = walkHe.getStartVertex().getPosition();


					

					if (!exit)
					{
						
						planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, startPlaneNorm, startPlaneOrigin, intersectionPoint);

						if (planeIntersection)
						{
							numPlaneIntersections++;
							railIntersectionPoints.push_back(intersectionPoint);
							railIntersectionHE.push_back(walkHe);

							exit = true;							
							break;
						}

					}


					if (!exit)
					{
						planeIntersection = coreUtils.line_PlaneIntersection(eE, eS, endPlaneNorm, endPlaneOrigin, intersectionPoint);

						if (planeIntersection)
						{
							numPlaneIntersections++;
							railIntersectionPoints.push_back(intersectionPoint);
							railIntersectionHE.push_back(walkHe);

							exit = true;							
							break;
						}

					}

					///////////////////				

					walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

					if (!guideSmooth_PrintRailEdgesBoolean[walkHe.getEdge().getId()]) exit = true;

				} while (!exit);

			}
		}

		//printf("\n  %i  intersections %i  %i ", _block.id, railIntersectionPoints.size(), numPlaneIntersections);

		if (numPlaneIntersections == 2)
		{
			zPoint p0 = railIntersectionPoints[0];
			zPoint p1 = railIntersectionPoints[1];

			if (p1.z < p0.z)
			{
				
				railIntersectionPoints[0] = p1;
				railIntersectionPoints[1] = p0;

				zItMeshHalfEdge he0 = railIntersectionHE[0];
				zItMeshHalfEdge he1 = railIntersectionHE[1];

				railIntersectionHE[0] = he1;
				railIntersectionHE[1] = he0;

			}
		}


		for (auto flipId : FlipBlockIds)
		{
			if (railIntersectionPoints.size() == 2 && _block.id == flipId)
			{
				zPoint p0 = railIntersectionPoints[0];
				zPoint p1 = railIntersectionPoints[1];

				railIntersectionPoints[0] = p1;
				railIntersectionPoints[1] = p0;

				zItMeshHalfEdge he0 = railIntersectionHE[0];
				zItMeshHalfEdge he1 = railIntersectionHE[1];

				railIntersectionHE[0] = he1;
				railIntersectionHE[1] = he0;
			}
		}


	}


	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockFrames(zPrintBlock& _block, float printLayerDepth, float neopreneOffset_start, float neopreneOffset_end, const zIntArray& FlipBlockIds, bool useRail)
	{
		//if (_block.id == -1) return;
		
		(_block.rightBlocks.size() > 0)? computePrintBlockLength(_block, false) : computePrintBlockLength(_block, true);
	
		zItMeshHalfEdge startHe, endHe;

		 // ANGLE COMPUTE
		
		 // compute angle for right blocks if they exist
		 if (_block.rightBlocks.size() > 0)
		 {
			 zVector startFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 startFNorm = startFNorm * -1;
			 _block.right_planeAngle = endFNorm.angle(startFNorm);

			  startHe = zItMeshHalfEdge(o_guideSmoothMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].second);
			  endHe = zItMeshHalfEdge(o_guideSmoothMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].second);
		 }

		 if (_block.leftBlocks.size() > 0)
		 {
			 zVector startFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 startFNorm = startFNorm * -1;

			 _block.left_planeAngle = endFNorm.angle(startFNorm);

			 startHe = zItMeshHalfEdge(o_guideSmoothMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].second);
			 endHe = zItMeshHalfEdge(o_guideSmoothMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].second);
		 }

		 // START
		 zPoint start, end;
		 float dIncrement = 0.0001;
		 
		 zItMeshHalfEdge walkHe = startHe;
		 walkHe = walkHe.getSym();
		 		
		 zPoint pOnCurve = _block.intersectionPoints[0];;

		 bool exit = false;

		 bool left = (_block.leftBlocks.size() > 0) ? false : true;
		 bool right = (_block.rightBlocks.size() > 0) ? false : true;

		 float dStart = 0;
		 float dEnd =0;

		 while (!exit)
		 {
			 zPoint eEndPoint = walkHe.getVertex().getPosition();
			 dStart += dIncrement;
			 float distance_increment = dIncrement;
			 while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			 {
				 distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				 pOnCurve = eEndPoint;

				 walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
				 eEndPoint = walkHe.getVertex().getPosition();
			 }

			 zVector he_vec = walkHe.getVector();
			 he_vec.normalize();

			 start = pOnCurve + he_vec * distance_increment;
			 pOnCurve = start;
			 // check 

			

			 if (!right)
			 {
				 zPoint startPlanePoint = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getCenter();
				 zPoint startPlaneNormal = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();

				 float dStart = coreUtils.minDist_Point_Plane(pOnCurve, startPlanePoint, startPlaneNormal);
				 				
				 if (abs(dStart) >= neopreneOffset_start)  right = true;

				
			 }

			 if (!left)
			 {
				 zPoint startPlanePoint = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getCenter();
				 zPoint startPlaneNormal = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();

				 float dStart = coreUtils.minDist_Point_Plane(pOnCurve, startPlanePoint, startPlaneNormal);
				 				
				 if (abs(dStart) >= neopreneOffset_start)  left = true;

				
			 }

			 if (right && left) exit = true;
			
		 }
			 

		 _block.printStartEndPoints.push_back(start);

		 // END
		
		 walkHe = endHe;
		 walkHe = walkHe.getSym();
		 pOnCurve = _block.intersectionPoints[1];;

		 exit = false;
		 left = (_block.leftBlocks.size() > 0) ? false : true;
		 right = (_block.rightBlocks.size() > 0) ? false : true;

		 while (!exit)
		 {
			 zPoint eEndPoint = walkHe.getVertex().getPosition();
			 dEnd += dIncrement;
			 float distance_increment = dIncrement;
			 while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
			 {
				 distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
				 pOnCurve = eEndPoint;

				 walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
				 eEndPoint = walkHe.getVertex().getPosition();
			 }

			 zVector he_vec = walkHe.getVector();
			 he_vec.normalize();

			 end = pOnCurve + he_vec * distance_increment;
			 pOnCurve = end;
			 // check 

			

			 if (!right)
			 {
				 zPoint endPlanePoint = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getCenter();
				 zPoint endPlaneNormal = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

				 float dEnd = coreUtils.minDist_Point_Plane(pOnCurve, endPlanePoint, endPlaneNormal);

				 if (abs(dEnd) >= neopreneOffset_end)  right = true;
				 				 
			 }

			 if (!left)
			 {
				 zPoint endPlanePoint = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getCenter();
				 zPoint endPlaneNormal = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

				 float dEnd = coreUtils.minDist_Point_Plane(pOnCurve, endPlanePoint, endPlaneNormal);

				 if (abs(dEnd) >= neopreneOffset_end)  left = true;
				
			 }

			 if (right && left) exit = true;

		 }
		
		 	

		 int currentId = 46;
		 //if(_block.id == currentId) cout<< "\n dist : " << dStart  << "," << dEnd;
		 _block.printStartEndPoints.push_back(end);

		 float midWeight = 0.5;
		 float midMultValue = 0.5;

		 zFloatArray weights = { 0.0, 0.4, 1.0 };
		 zFloatArray multVals = { 0.0, 0.6,  1.0 };

		  //// FRAME COMPUTE
		
		  // compute frames for right blocks
		 if (_block.rightBlocks.size() > 0)
		 {
			 float len = _block.mg_Length - (dStart + dEnd);
			 int numLayers = floor(len / printLayerDepth);

			 float equalisedLayerDepth = len / numLayers;

			 zVector startFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0,0,-1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.right_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();
			 			 
			 
			 zPoint startRail, endRail;
			 float equalisedRailLayerDepth;
			 zItMeshHalfEdge railWalkHe;

			 if (useRail)
			 {
				 zPoint startOrig = _block.printStartEndPoints[0];
				 zPoint endOrig = _block.printStartEndPoints[1];

				 zItMeshHalfEdgeArray railIntersectionHE;
				 zPointArray railIntersectionPoints;

				 float railLength = computePrintBlockRailLength(_block, startOrig, startFNorm, endOrig, endFNorm, _block.color, false, FlipBlockIds, railIntersectionHE, railIntersectionPoints);
			
				 equalisedRailLayerDepth = railLength / numLayers;

				 railWalkHe = railIntersectionHE[0];
				 railWalkHe = railWalkHe.getSym();

				 startRail = railIntersectionPoints[0];
				 endRail = railIntersectionPoints[1];


				 zVector dir = railWalkHe.getVertex().getPosition() - startRail;
				 dir.normalize();

				 zVector vec = railWalkHe.getVector();
				 vec.normalize();

				 if (vec * dir < 0) railWalkHe = (railWalkHe.onBoundary()) ? railWalkHe.getNext() : railWalkHe.getNext().getSym().getNext();

				 //printf("\n %i : %1.2f  %1.2f ", _block.id, len, railLength);
			 }


			startFNorm = startFNorm * -1;
		 
			float angle = _block.right_planeAngle;
			float angleStep = _block.right_planeAngle / (numLayers );

			 float xStep = (endFNorm.x - startFNorm.x) / (numLayers );
			 float yStep = (endFNorm.y - startFNorm.y) / (numLayers );
			 float zStep = (endFNorm.z - startFNorm.z) / (numLayers );
			 			 
			 //printf("\n step %1.6f  %1.6f  %1.6f \n ", xStep, yStep, zStep);

			// compute frames
			 walkHe = startHe;
			 walkHe = walkHe.getSym();

			 zVector dir = walkHe.getVertex().getPosition() - start;
			 dir.normalize();

			 zVector vec = walkHe.getVector();
			 vec.normalize();		

			 if (vec * dir < 0) walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

			 // Start point
			 float tLen = 0.1;
			 zPoint O = start;
			 zVector Z = startFNorm;

			 zVector tempZ = Z;
			 tempZ.normalize();

			 zVector X, Y;

			 zPoint ORail = startRail;
			 zItMeshFaceArray eFaces;

			 if (useRail)
			 {
				 //Y
				 Y = zVector();
				 eFaces.clear();
				 walkHe.getEdge().getFaces(eFaces);

				 for (auto& eF : eFaces)	Y += eF.getNormal();
				 Y /= eFaces.size();
				 Y.normalize();

				 float weight = 0;

				 float mult;

				 for (int l = 0; l < weights.size() -1; l++)
				 {
					 if(weight >= weights[l] && weight <= weights[l+1])   mult = coreUtils.ofMap(weight, weights[l], weights[l + 1], multVals[l], multVals[l+ 1]);
				 }


				// if (weight <= midWeight) mult = coreUtils.ofMap(weight, 0.0f, midWeight, 0.0f, midMultValue);
				// else mult = coreUtils.ofMap(weight, midWeight, 1.0f, midMultValue, 1.0f);

				 tempZ.x = (startFNorm.x * (1 - mult)) + (endFNorm.x * mult);
				 tempZ.y = (startFNorm.y * (1 - mult)) + (endFNorm.y * mult);
				 tempZ.z = (startFNorm.z * (1 - mult)) + (endFNorm.z * mult);
				 tempZ.normalize();

				 X = Y ^ tempZ;
				 X.normalize();

				 Y = tempZ ^ X;
				 Y.normalize();

				 
			 }
			 else
			 {
				
				 walkHe.getEdge().getFaces(eFaces);

				 Y = zVector();
				 for (auto& eF : eFaces)	Y += eF.getNormal();
				 Y /= eFaces.size();

				 Y.normalize();
				 Y *= 1;

				 X = Y ^ tempZ;
				 X.normalize();

				 Y = tempZ ^ X;
				 Y.normalize();

			 }

			
			 zTransform pFrame = setTransformFromVectors(O, X, Y, tempZ);
			 _block.sectionFrames.push_back(pFrame);
			 
			 zPoint pOnCurve = O;
			 zPoint pOnRailCurve = ORail;

			 zPoint startPlanePoint = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[0].first).getCenter();
			 zPoint startPlaneNormal = startFNorm;

			 float dStart = coreUtils.minDist_Point_Plane(pOnCurve, startPlanePoint, startPlaneNormal);
			
			 

			 // in between points
			 int size = numLayers;
			 for (int j = 0; j < size; j++)
			 {
				 zPoint prevPoint = pOnCurve;				

				 zPoint eEndPoint = walkHe.getVertex().getPosition();				 
				 

				 float distance_increment = equalisedLayerDepth;

				 //if (_block.id == currentId && j == 0) cout << endl << pOnCurve;

				 while (pOnCurve.distanceTo(eEndPoint) < distance_increment)
				 {
					 distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					 pOnCurve = eEndPoint;
					
					 walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
					 eEndPoint = walkHe.getVertex().getPosition();
				 }

				 zVector he_vec = walkHe.getVector();
				 he_vec.normalize();

				 if (useRail)
				 {
					 zPoint eRailEndPoint = railWalkHe.getVertex().getPosition();


					 float distance_Railincrement = equalisedRailLayerDepth;

					 //if (_block.id == currentId && j == 0) cout << endl << pOnCurve;

					 while (pOnRailCurve.distanceTo(eRailEndPoint) < distance_Railincrement)
					 {
						 distance_Railincrement = distance_Railincrement - pOnRailCurve.distanceTo(eRailEndPoint);
						 pOnRailCurve = eRailEndPoint;

						 railWalkHe = (railWalkHe.onBoundary()) ? railWalkHe.getNext() : railWalkHe.getNext().getSym().getNext();
						 eRailEndPoint = railWalkHe.getVertex().getPosition();
					 }

					 zVector he_RailVec = railWalkHe.getVector();
					 he_RailVec.normalize();

					 //O Rail
					 ORail = pOnRailCurve + he_RailVec * distance_Railincrement;

					 if (j == size - 1) ORail = endRail;

				 }

				// if (_block.id == currentId && j == 0) cout << "hevec " << he_vec << distance_increment;

				 //O
				 O = pOnCurve + he_vec * distance_increment;
				
				 //if (_block.id == currentId && j == 0) cout << endl << O << pOnCurve;

				 if (j == size -1)
				 {
					O = end;				
				 }
				 				
				 
				 if (j >= 0)
				 {
					 Z.x += xStep;
					 Z.y += yStep;
					 Z.z += zStep;
				 }

				 if (j == size - 1)
				 {
					// if (_block.id == currentId) cout << Z << endFNorm;
					 Z = endFNorm;
				 }

				
				 zVector tempZ = Z;
				 tempZ.normalize();

				 if (useRail)
				 {
					 //Y
					 Y = zVector();
					 eFaces.clear();
					 walkHe.getEdge().getFaces(eFaces);

					 for (auto& eF : eFaces)	Y += eF.getNormal();
					 Y /= eFaces.size();
					 Y.normalize();

					 float weight = (float)(j + 1) / numLayers;

					 float mult;

					 for (int l = 0; l < weights.size() - 1; l++)
					 {
						 if (weight >= weights[l] && weight <= weights[l + 1])   mult = coreUtils.ofMap(weight, weights[l], weights[l + 1], multVals[l], multVals[l + 1]);
					 }

					// if (weight <= midWeight) mult = coreUtils.ofMap(weight, 0.0f, midWeight, 0.0f, midMultValue);
					// else mult = coreUtils.ofMap(weight, midWeight, 1.0f, midMultValue, 1.0f);

					 tempZ.x = (startFNorm.x * (1 - mult)) + (endFNorm.x * mult);
					 tempZ.y = (startFNorm.y * (1 - mult)) + (endFNorm.y * mult);
					 tempZ.z = (startFNorm.z * (1 - mult)) + (endFNorm.z * mult);
					 tempZ.normalize();

					 X = Y ^ tempZ;
					 X.normalize();

					 Y = tempZ ^ X;
					 Y.normalize();

					 //cout << endl << "right " << startFNorm << endFNorm << tempZ << mult << "," << weight;

					 
					 pOnRailCurve = ORail;

				 }
				 else
				 {
					 //Y
					 Y = zVector();
					 eFaces.clear();
					 walkHe.getEdge().getFaces(eFaces);

					 for (auto& eF : eFaces)	Y += eF.getNormal();
					 Y /= eFaces.size();


					 Y.normalize();
					 Y *= 1;

					 //X
					 X = Y ^ tempZ;
					 X.normalize();

					 Y = tempZ ^ X;
					 Y.normalize();					

				 }

				

				 // add frame
				 pFrame = setTransformFromVectors(O, X, Y, tempZ);
				 _block.sectionFrames.push_back(pFrame);

				 //if (_block.id == currentId) printf("\n right %i %1.4f", _block.sectionFrames.size() - 1, O.distanceTo(prevPoint));

				 pOnCurve = O;
				
				 if (j == size - 1)
				 {
					 zPoint endPlanePoint = zItMeshFace(o_planeMesh, _block.right_sectionPlaneFace_GuideSmoothEdge[1].first).getCenter();
					 zPoint endPlaneNormal = endFNorm;

					 float dEnd = coreUtils.minDist_Point_Plane(pOnCurve, endPlanePoint, endPlaneNormal);

					 printf(" \n right sD %1.4f eD %1.4f ", dStart, dEnd);
				 }

			 }


		 }

		 // compute frames for left blocks
		 if (_block.leftBlocks.size() > 0)
		 {
			 float len = _block.mg_Length - (dStart + dEnd);;
			 int numLayers = floor(len / printLayerDepth);

			 float equalisedLayerDepth = len / numLayers;

			 zVector startFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[0].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getNormal();
			 zVector endFNorm = (_block.left_sectionPlaneFace_GuideSmoothEdge[1].first == -1) ? zVector(0, 0, -1) : zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getNormal();

			 zPoint startRail, endRail;
			 float equalisedRailLayerDepth;
			 zItMeshHalfEdge railWalkHe;

			 if (useRail)
			 {
				 zPoint startOrig = _block.printStartEndPoints[0];
				 zPoint endOrig = _block.printStartEndPoints[1];

				 zItMeshHalfEdgeArray railIntersectionHE;
				 zPointArray railIntersectionPoints;

				 float railLength = computePrintBlockRailLength(_block, startOrig, startFNorm, endOrig, endFNorm, _block.color, true, FlipBlockIds, railIntersectionHE, railIntersectionPoints);

				 equalisedRailLayerDepth = railLength / numLayers;

				 railWalkHe = railIntersectionHE[0];
				 railWalkHe = railWalkHe.getSym();

				 startRail = railIntersectionPoints[0];
				 endRail = railIntersectionPoints[1];


				 zVector dir = railWalkHe.getVertex().getPosition() - startRail;
				 dir.normalize();

				 zVector vec = railWalkHe.getVector();
				 vec.normalize();

				 if (vec * dir < 0) railWalkHe = (railWalkHe.onBoundary()) ? railWalkHe.getNext() : railWalkHe.getNext().getSym().getNext();

				 //printf("\n %i : %1.2f  %1.2f ", _block.id, len, railLength);
			 }

			 startFNorm = startFNorm * -1;

			 float angle = _block.left_planeAngle;
			 float angleStep = _block.left_planeAngle / (numLayers);
			 float xStep = (endFNorm.x - startFNorm.x) / (numLayers);
			 float yStep = (endFNorm.y - startFNorm.y) / (numLayers);
			 float zStep = (endFNorm.z - startFNorm.z) / (numLayers);
			 

			// compute frames
			 walkHe = startHe;
			 walkHe = walkHe.getSym();

			 zVector dir = walkHe.getVertex().getPosition() - start;
			 dir.normalize();

			 zVector vec = walkHe.getVector();
			 vec.normalize();

			 if (vec * dir < 0) walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

			 // Start point
			 float tLen = 0.1;
			 zPoint O = start;
			 zVector Z = startFNorm;

			 zVector tempZ = Z;
			 tempZ.normalize();

			 zVector X, Y;

			 zPoint ORail = startRail;
			 zItMeshFaceArray eFaces;

			 if (useRail)
			 {
				 //Y
				 Y = zVector();
				 eFaces.clear();
				 walkHe.getEdge().getFaces(eFaces);

				 for (auto& eF : eFaces)	Y += eF.getNormal();
				 Y /= eFaces.size();
				 Y.normalize();

				 float weight = 0;

				 float mult;
				 for (int l = 0; l < weights.size() - 1; l++)
				 {
					 if (weight >= weights[l] && weight <= weights[l + 1])   mult = coreUtils.ofMap(weight, weights[l], weights[l + 1], multVals[l], multVals[l + 1]);
				 }
				 
				 // if (weight <= midWeight) mult = coreUtils.ofMap(weight, 0.0f, midWeight, 0.0f, midMultValue);
				// else mult = coreUtils.ofMap(weight, midWeight, 1.0f, midMultValue, 1.0f);

				 tempZ.x = (startFNorm.x * (1 - mult)) + (endFNorm.x * mult);
				 tempZ.y = (startFNorm.y * (1 - mult)) + (endFNorm.y * mult);
				 tempZ.z = (startFNorm.z * (1 - mult)) + (endFNorm.z * mult);
				 tempZ.normalize();
				 			

				 X = Y ^ tempZ;
				 X.normalize();

				 Y = tempZ ^ X;
				 Y.normalize();

				
			 }
			 else
			 {

				 walkHe.getEdge().getFaces(eFaces);

				 Y = zVector();
				 for (auto& eF : eFaces)	Y += eF.getNormal();
				 Y /= eFaces.size();

				 Y.normalize();
				 Y *= 1;

				 X = Y ^ tempZ;
				 X.normalize();

				 Y = tempZ ^ X;
				 Y.normalize();

			 }

			 zTransform pFrame = setTransformFromVectors(O, X, Y, tempZ);
			 _block.sectionFrames.push_back(pFrame);

			 zPoint pOnCurve = O;
			 zPoint pOnRailCurve = ORail;

			 zPoint startPlanePoint = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[0].first).getCenter();
			 zPoint startPlaneNormal = startFNorm;

			 float dStart = coreUtils.minDist_Point_Plane(pOnCurve, startPlanePoint, startPlaneNormal);

			 // in between points
			 int size = numLayers ;
			 for (int j = 0; j < size; j++)
			 {
				 zPoint prevPoint = pOnCurve;

				  zPoint eEndPoint = walkHe.getVertex().getPosition();
				 float distance_increment = equalisedLayerDepth;

				 while(pOnCurve.distanceTo(eEndPoint) < distance_increment)
				 {
					 distance_increment = distance_increment - pOnCurve.distanceTo(eEndPoint);
					 pOnCurve = eEndPoint;
					 
					 walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
					 eEndPoint = walkHe.getVertex().getPosition();
				 }

				 zVector he_vec = walkHe.getVector();
				 he_vec.normalize();

				 if (useRail)
				 {
					 zPoint eRailEndPoint = railWalkHe.getVertex().getPosition();


					 float distance_Railincrement = equalisedRailLayerDepth;

					 //if (_block.id == currentId && j == 0) cout << endl << pOnCurve;

					 while (pOnRailCurve.distanceTo(eRailEndPoint) < distance_Railincrement)
					 {
						 distance_Railincrement = distance_Railincrement - pOnRailCurve.distanceTo(eRailEndPoint);
						 pOnRailCurve = eRailEndPoint;

						 railWalkHe = (railWalkHe.onBoundary()) ? railWalkHe.getNext() : railWalkHe.getNext().getSym().getNext();
						 eRailEndPoint = railWalkHe.getVertex().getPosition();
					 }

					 zVector he_RailVec = railWalkHe.getVector();
					 he_RailVec.normalize();

					 //O Rail
					 ORail = pOnRailCurve + he_RailVec * distance_Railincrement;

					 if (j == size - 1) ORail = endRail;

					 //printf("\n %1.3f, %1.3f, %1.3f ", ORail.x, ORail.y, ORail.z);
				 }

				 //O
				 O = pOnCurve + he_vec * distance_increment;
				
				 if (j == size -1)
				 {
					 O = end;

					
				 }
				 				 

				 // Z 
				 if (j >= 0)
				 {
					 Z.x += xStep;
					 Z.y += yStep;
					 Z.z += zStep;
				 }

				 if (j == size - 1)
				 {
					// if (_block.id == currentId) cout << Z << endFNorm;

					 Z = endFNorm;
				 }

				 zVector tempZ = Z;
				 tempZ.normalize();
				// tempZ *= tLen;

				 if (useRail)
				 {
					 //Y
					 Y = zVector();
					 eFaces.clear();
					 walkHe.getEdge().getFaces(eFaces);

					 for (auto& eF : eFaces)	Y += eF.getNormal();
					 Y /= eFaces.size();
					 Y.normalize();					 

					 float weight =(float) (j + 1) / numLayers;

					 float mult;
					 for (int l = 0; l < weights.size() - 1; l++)
					 {
						 if (weight >= weights[l] && weight <= weights[l + 1])   mult = coreUtils.ofMap(weight, weights[l], weights[l + 1], multVals[l], multVals[l + 1]);
					 }
					 
					 //if (weight <= midWeight) mult = coreUtils.ofMap(weight, 0.0f, midWeight, 0.0f, midMultValue);
					// else mult = coreUtils.ofMap(weight, midWeight, 1.0f, midMultValue, 1.0f);

					 tempZ.x = (startFNorm.x * (1 - mult)) + (endFNorm.x * mult);
					 tempZ.y = (startFNorm.y * (1 - mult)) + (endFNorm.y * mult);
					 tempZ.z = (startFNorm.z * (1 - mult)) + (endFNorm.z * mult);
					 tempZ.normalize();

					 X = Y ^ tempZ;
					 X.normalize();

					 Y = tempZ ^ X;
					 Y.normalize();

					// cout << endl << "left "<< startFNorm << endFNorm << tempZ << mult << "," << weight;

					 pOnRailCurve = ORail;
				 }
				 else
				 {
					 //Y
					 Y = zVector();
					 eFaces.clear();
					 walkHe.getEdge().getFaces(eFaces);

					 for (auto& eF : eFaces)	Y += eF.getNormal();
					 Y /= eFaces.size();


					 Y.normalize();
					 Y *= 1;

					 //X
					 X = Y ^ tempZ;
					 X.normalize();

					 Y = tempZ ^ X;
					 Y.normalize();

					

				 }
				 

				 // add frame
				 pFrame = setTransformFromVectors(O, X, Y, tempZ);
				 _block.sectionFrames.push_back(pFrame);

				// if (_block.id == currentId) printf("\n  left %i %1.4f", _block.sectionFrames.size() - 1, O.distanceTo(prevPoint));

				 pOnCurve = O;

				 if (j == size - 1)
				 {
					 zPoint endPlanePoint = zItMeshFace(o_planeMesh, _block.left_sectionPlaneFace_GuideSmoothEdge[1].first).getCenter();
					 zPoint endPlaneNormal = endFNorm;

					 float dEnd = coreUtils.minDist_Point_Plane(pOnCurve, endPlanePoint, endPlaneNormal);

					 printf(" \n left sD %1.4f eD %1.4f ", dStart, dEnd);
				 }

			 }

			

		 }

		
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockSections_Internal(zPrintBlock& _block)
	{
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);
		zFnMesh  fn_guideMesh(*o_guideMesh);

		zFnMesh  fn_guideThickMesh(*o_guideSmoothThickMesh);

		_block.sectionTrimFaces.clear();
		_block.sectionTrimFaces.assign(_block.sectionFrames.size(), zIntPair(-1,-1));

		_block.o_sectionGraphs.clear();
		_block.o_sectionGraphs.assign(_block.sectionFrames.size(), zObjGraph());

		_block.sectionGraphs_startEndVertex.clear();
		_block.sectionGraphs_startEndVertex.assign(_block.sectionFrames.size(), zIntPair(-1,-1));

		_block.startPos.clear();
		_block.startPos.assign(_block.sectionFrames.size(), zPoint());

		_block.endPos.clear();
		_block.endPos.assign(_block.sectionFrames.size(), zPoint());

		_block.o_sectionGuideGraphs.clear();
		_block.o_sectionGuideGraphs.assign(_block.sectionFrames.size(), zObjGraph());

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

				zVector X(_block.sectionFrames[i](0, 0), _block.sectionFrames[i](0, 1), _block.sectionFrames[i](0, 2));
				X.normalize();

				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				//fn_guideSmoothMesh.setVertexColorsfromScalars(scalars);

				zPointArray positions;
				zIntArray edgeConnects;
				zColorArray vColors;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);
						
				//printf("\n %i %i ", positions.size(), edgeConnects.size());

				
				

				// compute inside which convex block the graph exits
				/*zConvexBlock tempConvexB;
				zPoint pTemp = positions[0];
				int id = 0; 
				for (auto& rB : _block.rightBlocks)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zItMeshFace f0(o_planeMesh, rB.faces[0]);
					zPoint O0 = f0.getCenter();
					zVector N0 = f0.getNormal();

					float minDist_Plane0 = coreUtils.minDist_Point_Plane(pTemp, O0, N0);
					if (minDist_Plane0 > distanceTolerance) internal_v0 = false;

					zItMeshFace f1(o_planeMesh, rB.faces[2]);
					zPoint O1 = f1.getCenter();
					zVector N1 = f1.getNormal();

					float minDist_Plane1 = coreUtils.minDist_Point_Plane(pTemp, O1, N1);
					if (minDist_Plane1 > distanceTolerance) internal_v1 = false;

					if (internal_v0 && internal_v1)
					{
						tempConvexB = rB;
						zIntPair p(1, id);

						_block.trimConvexBlockType_Index[i] = p;
						break;
					}

					id++;
				}*/

				// compute trim planes
				zIntPair trimFaces(-1, -1);
				zPoint pointOnEndPlane;
				zPoint pointOnStartPlane;
				
				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 0; j < _block.right_sideFaces.size(); j += 1)
					{

						zItMeshFace f(o_planeMesh, _block.right_sideFaces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);

						// intersection
						if (minDist_Plane0 >= 0 && minDist_Plane1 < 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);

							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.right_sideFaces[j];
										pointOnStartPlane = iPt;
									}
									if (col == blue)
									{
										trimFaces.second = _block.right_sideFaces[j];

										pointOnEndPlane = iPt;
									}

								}

							}
						}


						if (minDist_Plane0 < 0 && minDist_Plane1 >= 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.right_sideFaces[j];
										pointOnStartPlane = iPt;
									}
									if (col == blue)
									{
										trimFaces.second = _block.right_sideFaces[j];
										pointOnEndPlane = iPt;
									}
								}

							}
						}
					}
				}
						

				//printf("\n right s %i  e %i ", trimFaces.first, trimFaces.second);
								

				//if (trimFaces.first == -1 || trimFaces.second == -1) tempFn.setEdgeColor(magenta);

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

					
					for (int j = 0; j < 2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();									

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
								internal_v1 = true;
								p1 = iPt;				
							}
						}

						

						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{		
								internal_v0 = true;
								p0 = iPt;	
							}
						}
					}
					

					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{						

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}
							

						
					}

				}

				// create temp graph
				zObjGraph tempGraph;					
				//zFnGraph tempFn(_block.o_sectionGraphs[i]);
				zFnGraph tempFn(tempGraph);
				tempFn.create(t_positions, t_edgeConnects);;

				//printf("\n %i %i ", t_positions.size(), t_edgeConnects.size());

				// Remove discontinuous lines
				float dist = 10000;
				int tempStartId;
				for (zItGraphVertex v(tempGraph); !v.end(); v++)
				{
					zItMeshFace f(o_planeMesh, trimFaces.second);
					zPoint O1 = f.getCenter();
					zVector N1 = f.getNormal();

					zPoint p0 = v.getPosition();
					float minDist_Plane = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

					if (minDist_Plane < dist)
					{
						dist = minDist_Plane;
						tempStartId = v.getId();
					}

				}

				zItGraphVertex startV(tempGraph, tempStartId);
				zItGraphHalfEdge walkHe = startV.getHalfEdge();

				t_positions.clear();
				t_edgeConnects.clear();

				t_positions.push_back(walkHe.getStartVertex().getPosition());
				bool exit = false;
				
				do
				{	
					t_edgeConnects.push_back(t_positions.size() - 1);
					t_edgeConnects.push_back(t_positions.size());
					t_positions.push_back(walkHe.getVertex().getPosition());

					if (walkHe.getVertex().checkValency(1))
					{
						exit = true;
					}
					else walkHe = walkHe.getNext();

				} while (!exit);
							

				zFnGraph sectionFn(_block.o_sectionGraphs[i]);
				
				sectionFn.create(t_positions, t_edgeConnects);;
				sectionFn.setEdgeColor(magenta);
				sectionFn.setEdgeWeight(3);

				//printf("\n temp %i  section %i ", tempFn.numVertices(), sectionFn.numVertices());

				// compute start- end vertex 
				float distYellow = 10000;
				float distBlue  = 10000;
				for (zItGraphVertex v(_block.o_sectionGraphs[i]); !v.end(); v++)
				{
					for (int j = 0; j <2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();

						zColor col = f.getColor();
						zPoint p0 = v.getPosition();
						float minDist_Plane = abs (coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (col == yellow && minDist_Plane < distYellow)
						{
							distYellow = minDist_Plane;

							_block.sectionGraphs_startEndVertex[i].first = v.getId();
							_block.startPos[i] = v.getPosition();

						}

						if (col == blue && minDist_Plane < distBlue)
						{
							distBlue = minDist_Plane;

							_block.sectionGraphs_startEndVertex[i].second = v.getId();
							_block.endPos[i] = v.getPosition();

						}
					}

					
				}

				_block.sectionTrimFaces[i] = trimFaces;
				

				// THICKENED mesh

				scalars.clear();
				for (zItMeshVertex v(*o_guideSmoothThickMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				positions.clear();
				edgeConnects.clear();
				vColors.clear();
				
				zColorArray t_Colors;

				//printf("\n graph right %i ", i);
				fn_guideThickMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);

				
				// trim with side plane
				t_positions.clear();
				t_edgeConnects.clear();
				positionVertex.clear();
				tVertex_vertexmap.clear();

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					zColor c0 = vColors[edgeConnects[k]];
					zColor c1 = vColors[edgeConnects[k + 1]];

					for (int j = 0; j < 2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);

								t_Colors.push_back(c0);
									;
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);

								t_Colors.push_back(c1);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}


				zObjGraph o_tempThk;
				zFnGraph tempThkFn(o_tempThk);
				tempThkFn.create(t_positions, t_edgeConnects);;
			
				zIntArray start, end;

				zItMeshFace f1(o_planeMesh, trimFaces.second);
				zPoint O1 = f1.getCenter();
				zVector N1 = f1.getNormal();

				zItMeshFace f0(o_planeMesh, trimFaces.first);
				zPoint O0 = f0.getCenter();
				zVector N0 = f0.getNormal();

				for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				{


					if (v.getValence() == 1)
					{

						zPoint p0 = v.getPosition();
						float d = abs(coreUtils.minDist_Point_Plane(p0, O0, N0));

						if (d < distanceTolerance)
						{
							float distToSP = p0.distanceTo(pointOnStartPlane);
							if (distToSP < 0.5) start.push_back(v.getId());

						}

						d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (d < distanceTolerance)
						{
							float distToEP = p0.distanceTo(pointOnEndPlane);
							if (distToEP < 0.5) end.push_back(v.getId());

						}
					}
				}

				/*zIntArray vertexMap;
				vertexMap.assign(tempThkFn.numVertices(), -1);
				t_positions.clear();
				t_edgeConnects.clear();

				zIntArray start, end;
				zItMeshFace f(o_planeMesh, trimFaces.first);
				zPoint O1 = f.getCenter();
				zVector N1 = f.getNormal();

				zPoint* sectionPositions = 	sectionFn.getRawVertexPositions();			

				for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				{
					float minDist = 100;;
					zPoint p0 = v.getPosition();

					for (int l = 0; l < sectionFn.numVertices(); l++)
					{
						if (p0.distanceTo(sectionPositions[l]) < minDist)
						{
							minDist = p0.distanceTo(sectionPositions[l]);
						}
					}

					if (minDist < 0.25)
					{						
						vertexMap[v.getId()] = t_positions.size();
						

						if (v.getValence() == 1)
						{
														
							float d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

							if (d < 0.1)
							{
								 start.push_back(t_positions.size());
							}
							else end.push_back(t_positions.size());
						}


						t_positions.push_back(p0);
					}
				}

				for (zItGraphEdge e(o_tempThk); !e.end(); e++)
				{
					zIntArray eVerts;
					e.getVertices(eVerts);

					if (vertexMap[eVerts[0]] != -1 && vertexMap[eVerts[1]] != -1)
					{
						t_edgeConnects.push_back(vertexMap[eVerts[0]]);
						t_edgeConnects.push_back(vertexMap[eVerts[1]]);
					}
				}*/

				//zColorArray vertexColor;
				//vertexColor.assign(t_positions.size(), zColor());

				t_Colors[end[0]] = orange;
				t_Colors[end[1]] = orange;

				t_Colors[start[0]] = yellow;
				t_Colors[start[1]] = yellow;

				t_edgeConnects.push_back(end[0]);
				t_edgeConnects.push_back(end[1]);

				t_edgeConnects.push_back(start[0]);
				t_edgeConnects.push_back(start[1]);

				zFnGraph ThkFn(_block.o_sectionGuideGraphs[i]);
				ThkFn.create(t_positions, t_edgeConnects);;
				ThkFn.setEdgeColor(grey);
				
				ThkFn.setVertexColors(t_Colors);

				
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

				zVector X(_block.sectionFrames[i](0, 0), _block.sectionFrames[i](0, 1), _block.sectionFrames[i](0, 2));
				X.normalize();

				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				//fn_guideSmoothMesh.setVertexColorsfromScalars(scalars);

				zPointArray positions;
				zIntArray edgeConnects;
				zColorArray vColors;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);				

			
				// compute inside which convex block the graph exits
				/*zConvexBlock tempConvexB;
				int id = 0;
				zPoint pTemp = positions[0];
				for (auto& lB : _block.leftBlocks)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zItMeshFace f0(o_planeMesh, lB.faces[0]);
					zPoint O0 = f0.getCenter();
					zVector N0 = f0.getNormal();

					float minDist_Plane0 = coreUtils.minDist_Point_Plane(pTemp, O0, N0);
					if (minDist_Plane0 > distanceTolerance) internal_v0 = false;

					zItMeshFace f1(o_planeMesh, lB.faces[2]);
					zPoint O1 = f1.getCenter();
					zVector N1 = f1.getNormal();

					float minDist_Plane1 = coreUtils.minDist_Point_Plane(pTemp, O1, N1);
					if (minDist_Plane1 > distanceTolerance) internal_v1 = false;

					if (internal_v0 && internal_v1)
					{
						tempConvexB = lB;
						zIntPair p(0, id);
						_block.trimConvexBlockType_Index[i] = p;
						break;
					}

					id++;
				}*/

				// compute trim faces
				zIntPair trimFaces(-1, -1);
				zPoint pointOnEndPlane;
				zPoint pointOnStartPlane;

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 0; j < _block.left_sideFaces.size(); j += 1)
					{

						zItMeshFace f(o_planeMesh, _block.left_sideFaces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);

						// intersection
						if (minDist_Plane0 >= 0 && minDist_Plane1 < 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);

							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.left_sideFaces[j];
										pointOnStartPlane = iPt;
									}
									if (col == blue)
									{
										trimFaces.second = _block.left_sideFaces[j];
										pointOnEndPlane = iPt;
									}

								}

							}
						}


						if (minDist_Plane0 < 0 && minDist_Plane1 >= 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.left_sideFaces[j];
										pointOnStartPlane = iPt;
									}
									if (col == blue)
									{
										trimFaces.second = _block.left_sideFaces[j];

										pointOnEndPlane = iPt;
									}
								}

							}
						}
					}
				}
				

				//printf("\n left %i  e %i ", trimFaces.first, trimFaces.second);
					
				
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

					for (int j = 0; j < 2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}


				// create temp graph
				zObjGraph tempGraph;
				//zFnGraph tempFn(_block.o_sectionGraphs[i]);
				zFnGraph tempFn(tempGraph);
				tempFn.create(t_positions, t_edgeConnects);;

				// Remove discontinuous lines
				float dist = 10000;
				int tempStartId;
				for (zItGraphVertex v(tempGraph); !v.end(); v++)
				{
					zItMeshFace f(o_planeMesh, trimFaces.second);
					zPoint O1 = f.getCenter();
					zVector N1 = f.getNormal();

					zPoint p0 = v.getPosition();
					float minDist_Plane = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

					if (minDist_Plane < dist)
					{
						dist = minDist_Plane;
						tempStartId = v.getId();
					}

				}

				zItGraphVertex startV(tempGraph, tempStartId);
				zItGraphHalfEdge walkHe = startV.getHalfEdge();

				t_positions.clear();
				t_edgeConnects.clear();

				t_positions.push_back(walkHe.getStartVertex().getPosition());
				bool exit = false;

				do
				{
					t_edgeConnects.push_back(t_positions.size() - 1);
					t_edgeConnects.push_back(t_positions.size());
					t_positions.push_back(walkHe.getVertex().getPosition());

					if (walkHe.getVertex().checkValency(1))
					{
						exit = true;
					}
					else walkHe = walkHe.getNext();

				} while (!exit);


				zFnGraph sectionFn(_block.o_sectionGraphs[i]);

				sectionFn.create(t_positions, t_edgeConnects);;
				sectionFn.setEdgeColor(magenta);
				sectionFn.setEdgeWeight(3);

				//printf("\n left temp %i  section %i ", tempFn.numVertices(), sectionFn.numVertices());
				
				// compute start- end vertex 
				float distYellow = 10000;
				float distBlue = 10000;
				for (zItGraphVertex v(_block.o_sectionGraphs[i]); !v.end(); v++)
				{
					for (int j = 0; j < 2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();

						zColor col = f.getColor();
						zPoint p0 = v.getPosition();
						float minDist_Plane = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (col == yellow && minDist_Plane < distYellow)
						{
							distYellow = minDist_Plane;

							_block.sectionGraphs_startEndVertex[i].first = v.getId();
							_block.startPos[i] = v.getPosition();

						}

						if (col == blue && minDist_Plane < distBlue)
						{
							distBlue = minDist_Plane;

							_block.sectionGraphs_startEndVertex[i].second = v.getId();
							_block.endPos[i] = v.getPosition();

						}
					}

				}

				_block.sectionTrimFaces[i] = trimFaces;


				// THICKENED mesh

				scalars.clear();
				for (zItMeshVertex v(*o_guideSmoothThickMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				positions.clear();
				edgeConnects.clear();
				vColors.clear();
				fn_guideThickMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);

				// trim with side plane
				t_positions.clear();
				t_edgeConnects.clear();
				positionVertex.clear();
				tVertex_vertexmap.clear();

				zColorArray t_vertexColors;

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					zColor c0 = vColors[edgeConnects[k]];
					zColor c1 = vColors[edgeConnects[k + 1]];

					for (int j = 0; j < 2; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);

								t_vertexColors.push_back(c0);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);

								t_vertexColors.push_back(c1);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}


				
				zObjGraph o_tempThk;
				zFnGraph tempThkFn(o_tempThk);
				tempThkFn.create(t_positions, t_edgeConnects);;

				zIntArray start, end;

				zItMeshFace f1(o_planeMesh, trimFaces.second);
				zPoint O1 = f1.getCenter();
				zVector N1 = f1.getNormal();

				zItMeshFace f0(o_planeMesh, trimFaces.first);
				zPoint O0 = f0.getCenter();
				zVector N0 = f0.getNormal();

				for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				{


					if (v.getValence() == 1)
					{
						
						zPoint p0 = v.getPosition();
						float d = abs(coreUtils.minDist_Point_Plane(p0, O0, N0));

						if (d < distanceTolerance)
						{
							float distToSP = p0.distanceTo(pointOnStartPlane);
							if (distToSP < 0.5) start.push_back(v.getId());

						}

						d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (d < distanceTolerance)
						{
							float distToEP = p0.distanceTo(pointOnEndPlane);
							if (distToEP < 0.5) end.push_back(v.getId());

						}
					}
				}

				//zIntArray vertexMap;
				//vertexMap.assign(tempThkFn.numVertices(), -1);
				//t_positions.clear();
				//t_edgeConnects.clear();

				//zIntArray start, end;
				//zItMeshFace f(o_planeMesh, trimFaces.first);
				//zPoint O1 = f.getCenter();
				//zVector N1 = f.getNormal();

				//zPoint* sectionPositions = sectionFn.getRawVertexPositions();

				//for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				//{
				//	float minDist = 100;;
				//	zPoint p0 = v.getPosition();

				//	for (int l = 0; l < sectionFn.numVertices(); l++)
				//	{
				//		if (p0.distanceTo(sectionPositions[l]) < minDist)
				//		{
				//			minDist = p0.distanceTo(sectionPositions[l]);
				//		}
				//	}

				//	if (minDist < 0.25)
				//	{
				//		vertexMap[v.getId()] = t_positions.size();


				//		if (v.getValence() == 1)
				//		{

				//			float d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

				//			if (d < 0.1)
				//			{
				//				start.push_back(t_positions.size());
				//			}
				//			else end.push_back(t_positions.size());
				//		}


				//		t_positions.push_back(p0);
				//	}
				//}
				////printf("\n %i | %i %i ", i, start.size(), end.size());


				//for (zItGraphEdge e(o_tempThk); !e.end(); e++)
				//{
				//	zIntArray eVerts;
				//	e.getVertices(eVerts);

				//	if (vertexMap[eVerts[0]] != -1 && vertexMap[eVerts[1]] != -1)
				//	{
				//		t_edgeConnects.push_back(vertexMap[eVerts[0]]);
				//		t_edgeConnects.push_back(vertexMap[eVerts[1]]);
				//	}
				//}

				

				t_vertexColors[end[0]] = orange;
				t_vertexColors[end[1]] = orange;

				t_vertexColors[start[0]] = yellow;
				t_vertexColors[start[1]] = yellow;

				t_edgeConnects.push_back(end[0]);
				t_edgeConnects.push_back(end[1]);

				t_edgeConnects.push_back(start[0]);
				t_edgeConnects.push_back(start[1]);

				zFnGraph ThkFn(_block.o_sectionGuideGraphs[i]);
				ThkFn.create(t_positions, t_edgeConnects);;
				ThkFn.setEdgeColor(grey);

				ThkFn.setVertexColors(t_vertexColors);
								
			}

		}

		//printf("\n b %i | sf %i  sv %i ", _block.id, _block.sectionFrames.size(), _block.sectionGraphs_startVertex.size());

	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockSections_Boundary(zPrintBlock& _block)
	{
		zFnMesh  fn_guideSmoothMesh(o_guideSmoothMesh);

		zFnMesh  fn_guideThickMesh(*o_guideSmoothThickMesh);

		_block.sectionTrimFaces.clear();
		_block.sectionTrimFaces.assign(_block.sectionFrames.size(), zIntPair(-1, -1));

		_block.o_sectionGraphs.clear();
		_block.o_sectionGraphs.assign(_block.sectionFrames.size(), zObjGraph());

		_block.sectionGraphs_startEndVertex.clear();
		_block.sectionGraphs_startEndVertex.assign(_block.sectionFrames.size(), zIntPair(-1, -1));

		_block.startPos.clear();
		_block.startPos.assign(_block.sectionFrames.size(), zPoint());

		_block.endPos.clear();
		_block.endPos.assign(_block.sectionFrames.size(), zPoint());

		_block.o_sectionGuideGraphs.clear();
		_block.o_sectionGuideGraphs.assign(_block.sectionFrames.size(), zObjGraph());

		zScalarArray scalars;


		//Right Blocks
		if (_block.rightBlocks.size() > 0 && _block.sectionFrames.size() > 0)
		{
			
			int start = 0;
			int end = (_block.leftBlocks.size() == 0) ? _block.sectionFrames.size() : floor(_block.sectionFrames.size() * 0.5);


			for (int i = start; i < end; i++)
			{

				scalars.clear();
				zPoint O(_block.sectionFrames[i](3, 0), _block.sectionFrames[i](3, 1), _block.sectionFrames[i](3, 2));
				zVector N(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));

				zVector X(_block.sectionFrames[i](0, 0), _block.sectionFrames[i](0, 1), _block.sectionFrames[i](0, 2));
				X.normalize();

				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				
				zPointArray positions;
				zIntArray edgeConnects;
				zColorArray vColors;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);
				

				// compute trim planes
				zIntPair trimFaces(-1, -1);		

				
				zPoint pointOnStartPlane;				

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 0; j < _block.right_sideFaces.size(); j += 1)
					{

						zItMeshFace f(o_planeMesh, _block.right_sideFaces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);

						// intersection
						if (minDist_Plane0 >= 0 && minDist_Plane1 < 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);

							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.right_sideFaces[j];
										pointOnStartPlane = iPt;
									}							
									

								}

							}
						}


						if (minDist_Plane0 < 0 && minDist_Plane1 >= 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.right_sideFaces[j];
										pointOnStartPlane = iPt;
									}								
								}

							}
						}
					}
				}


				//printf("\n right s %i  e %i ", trimFaces.first, trimFaces.second);

				

				//if (trimFaces.first == -1 || trimFaces.second == -1) tempFn.setEdgeColor(magenta);

				// trim with side plane
				zPointArray t_positions;
				zIntArray t_edgeConnects;
				unordered_map <string, int> positionVertex;
				zIntArray tVertex_vertexmap;

				//printf("\n %i %i ", trimFaces.first , trimFaces.second);

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 0; j < 1; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}

				// create temp graph
				zObjGraph tempGraph;
				//zFnGraph tempFn(_block.o_sectionGraphs[i]);
				zFnGraph tempFn(tempGraph);
				tempFn.create(t_positions, t_edgeConnects);;

				// Remove discontinuous lines
				float dist = 10000;
				float distToPointOnStartPlane = 10000;
				int tempStartId;
				for (zItGraphVertex v(tempGraph); !v.end(); v++)
				{
					zItMeshFace f(o_planeMesh, trimFaces.first);
					zPoint O1 = f.getCenter();
					zVector N1 = f.getNormal();

					zPoint p0 = v.getPosition();
					float minDist_Plane = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

					float distToSP = p0.distanceTo(pointOnStartPlane);

					if (distToSP < distToPointOnStartPlane)
					{
						//if (minDist_Plane < dist)
						//{
							dist = minDist_Plane;
							distToPointOnStartPlane = distToSP;
							tempStartId = v.getId();
						//}
					}
					

				}

				zItGraphVertex startV(tempGraph, tempStartId);
				zItGraphHalfEdge walkHe = startV.getHalfEdge();

				t_positions.clear();
				t_edgeConnects.clear();

				t_positions.push_back(walkHe.getStartVertex().getPosition());
				bool exit = false;

				do
				{
					t_edgeConnects.push_back(t_positions.size() - 1);
					t_edgeConnects.push_back(t_positions.size());
					t_positions.push_back(walkHe.getVertex().getPosition());

					if (walkHe.getVertex().checkValency(1))
					{
						exit = true;
					}
					else walkHe = walkHe.getNext();

				} while (!exit);


				zFnGraph sectionFn(_block.o_sectionGraphs[i]);

				sectionFn.create(t_positions, t_edgeConnects);;
				sectionFn.setEdgeColor(magenta);
				sectionFn.setEdgeWeight(3);

				//printf("\n temp %i  section %i ", tempFn.numVertices(), sectionFn.numVertices());

				// compute start- end vertex 

				zItGraphVertexArray boundaryVerts;
				for (zItGraphVertex v(_block.o_sectionGraphs[i]); !v.end(); v++)
				{
					if(v.getValence() == 1) boundaryVerts.push_back(v);
				}

				if (boundaryVerts[0].getPosition().distanceTo(pointOnStartPlane) < boundaryVerts[1].getPosition().distanceTo(pointOnStartPlane))
				{
					_block.sectionGraphs_startEndVertex[i].second = boundaryVerts[0].getId();
					_block.endPos[i] = boundaryVerts[0].getPosition();

					_block.sectionGraphs_startEndVertex[i].first = boundaryVerts[1].getId();
					_block.startPos[i] = boundaryVerts[1].getPosition();
				}
				else
				{
					_block.sectionGraphs_startEndVertex[i].second = boundaryVerts[1].getId();
					_block.endPos[i] = boundaryVerts[1].getPosition();

					_block.sectionGraphs_startEndVertex[i].first = boundaryVerts[0].getId();
					_block.startPos[i] = boundaryVerts[0].getPosition();
				}

				_block.sectionTrimFaces[i] = trimFaces;



				// THICKENED mesh

				scalars.clear();
				for (zItMeshVertex v(*o_guideSmoothThickMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				positions.clear();
				edgeConnects.clear();
				vColors.clear();
				fn_guideThickMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);

				// trim with side plane
				zColorArray t_Colors;

				t_positions.clear();
				t_edgeConnects.clear();
				positionVertex.clear();
				tVertex_vertexmap.clear();

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					zColor c0 = vColors[edgeConnects[k]];
					zColor c1 = vColors[edgeConnects[k + 1]];

					for (int j = 0; j < 1; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);

								t_Colors.push_back(c0);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);

								t_Colors.push_back(c1);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}

				zObjGraph o_tempThk;
				zFnGraph tempThkFn(o_tempThk);
				tempThkFn.create(t_positions, t_edgeConnects);;
				//tempThkFn.setEdgeColor(grey);

				tempThkFn.setVertexColors(t_Colors);

				zItGraphVertexArray thk_boundaryVerts;
				zIntArray start, end;

				zItMeshFace f(o_planeMesh, trimFaces.first);
				zPoint O1 = f.getCenter();
				zVector N1 = f.getNormal();
				for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				{


					if (v.getValence() == 1)
					{
						thk_boundaryVerts.push_back(v);
						zPoint p0 = v.getPosition();
						float d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (d < 0.1)
						{
							float distToSP = p0.distanceTo(pointOnStartPlane);
							if (distToSP < 1) start.push_back(v.getId());

						}
						
					}
				}



				// boundary block
				if (end.size() == 0)
				{
					t_positions.clear(); 
					t_edgeConnects.clear();
					t_Colors.clear();

					zItGraphVertex vStart(o_tempThk, start[0]);
					zItGraphHalfEdge he = vStart.getHalfEdge();

					t_positions.push_back(vStart.getPosition());
					t_Colors.push_back(vStart.getColor());

					bool exitLoop = false;
					do
					{
						if (t_positions.size() > 0)
						{
							t_edgeConnects.push_back(t_positions.size() - 1);
							t_edgeConnects.push_back(t_positions.size());
						}

						t_positions.push_back(he.getVertex().getPosition());

						t_Colors.push_back(he.getVertex().getColor());

						if (he.getVertex().checkValency(1))  exitLoop = true;
						else he = he.getNext();

					} while (!exitLoop);

					t_edgeConnects.push_back(0);
					t_edgeConnects.push_back(t_positions.size() - 1);
					
					t_Colors[0] = yellow;
					t_Colors[t_positions.size() - 1] = yellow;
				}
				

				zFnGraph ThkFn(_block.o_sectionGuideGraphs[i]);
				ThkFn.create(t_positions, t_edgeConnects);;
				ThkFn.setEdgeColor(grey);

				ThkFn.setVertexColors(t_Colors);

				//printf("\n %i bTh %i %i ", i, t_Colors.size(), t_positions.size());

			}

		}

		//Left blocks
		if (_block.leftBlocks.size() > 0 && _block.sectionFrames.size() > 0)
		{

			int start = (_block.rightBlocks.size() == 0) ? 0 : floor(_block.sectionFrames.size() * 0.5);
			int end = _block.sectionFrames.size();


			for (int i = start; i < end; i++)
			{

				scalars.clear();
				zPoint O(_block.sectionFrames[i](3, 0), _block.sectionFrames[i](3, 1), _block.sectionFrames[i](3, 2));
				zVector N(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));

				zVector X(_block.sectionFrames[i](0, 0), _block.sectionFrames[i](0, 1), _block.sectionFrames[i](0, 2));
				X.normalize();

				for (zItMeshVertex v(o_guideSmoothMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				//fn_guideSmoothMesh.setVertexColorsfromScalars(scalars);

				zPointArray positions;
				zIntArray edgeConnects;
				zColorArray vColors;
				fn_guideSmoothMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);


				// compute inside which convex block the graph exits
				/*zConvexBlock tempConvexB;
				int id = 0;
				zPoint pTemp = positions[0];
				for (auto& lB : _block.leftBlocks)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zItMeshFace f0(o_planeMesh, lB.faces[0]);
					zPoint O0 = f0.getCenter();
					zVector N0 = f0.getNormal();

					float minDist_Plane0 = coreUtils.minDist_Point_Plane(pTemp, O0, N0);
					if (minDist_Plane0 > distanceTolerance) internal_v0 = false;

					zItMeshFace f1(o_planeMesh, lB.faces[2]);
					zPoint O1 = f1.getCenter();
					zVector N1 = f1.getNormal();

					float minDist_Plane1 = coreUtils.minDist_Point_Plane(pTemp, O1, N1);
					if (minDist_Plane1 > distanceTolerance) internal_v1 = false;

					if (internal_v0 && internal_v1)
					{
						tempConvexB = lB;
						zIntPair p(0, id);
						_block.trimConvexBlockType_Index[i] = p;
						break;
					}

					id++;
				}*/

				// compute trim faces
				zIntPair trimFaces(-1, -1);
				
				zPoint pointOnStartPlane;

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					for (int j = 0; j < _block.left_sideFaces.size(); j += 1)
					{

						zItMeshFace f(o_planeMesh, _block.left_sideFaces[j]);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

						float minDist_Plane0 = coreUtils.minDist_Point_Plane(p0, O1, N1);
						float minDist_Plane1 = coreUtils.minDist_Point_Plane(p1, O1, N1);

						// intersection
						if (minDist_Plane0 >= 0 && minDist_Plane1 < 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);

							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.left_sideFaces[j];
										pointOnStartPlane = iPt;
									}						
									

								}

							}
						}


						if (minDist_Plane0 < 0 && minDist_Plane1 >= 0)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p0, p1, N1, O1, iPt);
							if (intersect)
							{
								zPointArray pts;
								f.getVertexPositions(pts);

								bool check = coreUtils.pointInPlanarPolygon(iPt, pts, N1);

								if (check)
								{
									if (col == yellow)
									{
										trimFaces.first = _block.left_sideFaces[j];
										pointOnStartPlane = iPt;
									}
									
								}

							}
						}
					}
				}


				//printf("\n left %i  e %i ", trimFaces.first, trimFaces.second);
								

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

					for (int j = 0; j < 1; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}


				// create temp graph
				zObjGraph tempGraph;
				//zFnGraph tempFn(_block.o_sectionGraphs[i]);
				zFnGraph tempFn(tempGraph);
				tempFn.create(t_positions, t_edgeConnects);;

				// Remove discontinuous lines
				float dist = 10000;
				float distToPointOnStartPlane = 10000;
				int tempStartId;
				for (zItGraphVertex v(tempGraph); !v.end(); v++)
				{
					zItMeshFace f(o_planeMesh, trimFaces.first);
					zPoint O1 = f.getCenter();
					zVector N1 = f.getNormal();

					zPoint p0 = v.getPosition();
					float minDist_Plane = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

					float distToSP = p0.distanceTo(pointOnStartPlane);

					if (distToSP < distToPointOnStartPlane)
					{
						//if (minDist_Plane < dist)
						//{
						dist = minDist_Plane;
						distToPointOnStartPlane = distToSP;
						tempStartId = v.getId();
						//}
					}

				}

				zItGraphVertex startV(tempGraph, tempStartId);
				zItGraphHalfEdge walkHe = startV.getHalfEdge();

				t_positions.clear();
				t_edgeConnects.clear();

				t_positions.push_back(walkHe.getStartVertex().getPosition());
				bool exit = false;

				do
				{
					t_edgeConnects.push_back(t_positions.size() - 1);
					t_edgeConnects.push_back(t_positions.size());
					t_positions.push_back(walkHe.getVertex().getPosition());

					if (walkHe.getVertex().checkValency(1))
					{
						exit = true;
					}
					else walkHe = walkHe.getNext();

				} while (!exit);


				zFnGraph sectionFn(_block.o_sectionGraphs[i]);

				sectionFn.create(t_positions, t_edgeConnects);;
				sectionFn.setEdgeColor(magenta);
				sectionFn.setEdgeWeight(3);

				//printf("\n left temp %i  section %i ", tempFn.numVertices(), sectionFn.numVertices());

				// compute start- end vertex 
				zItGraphVertexArray boundaryVerts;
				for (zItGraphVertex v(_block.o_sectionGraphs[i]); !v.end(); v++)
				{
					if (v.getValence() == 1) boundaryVerts.push_back(v);
				}

				if (boundaryVerts[0].getPosition().distanceTo(pointOnStartPlane) < boundaryVerts[1].getPosition().distanceTo(pointOnStartPlane))
				{
					_block.sectionGraphs_startEndVertex[i].second = boundaryVerts[0].getId();
					_block.endPos[i] = boundaryVerts[0].getPosition();

					_block.sectionGraphs_startEndVertex[i].first = boundaryVerts[1].getId();
					_block.startPos[i] = boundaryVerts[1].getPosition();
				}
				else
				{
					_block.sectionGraphs_startEndVertex[i].second = boundaryVerts[1].getId();
					_block.endPos[i] = boundaryVerts[1].getPosition();

					_block.sectionGraphs_startEndVertex[i].first = boundaryVerts[0].getId();
					_block.startPos[i] = boundaryVerts[0].getPosition();
				}

				_block.sectionTrimFaces[i] = trimFaces;


				// THICKENED mesh

				scalars.clear();
				for (zItMeshVertex v(*o_guideSmoothThickMesh); !v.end(); v++)
				{
					zPoint P = v.getPosition();
					float minDist_Plane = coreUtils.minDist_Point_Plane(P, O, N);
					scalars.push_back(minDist_Plane);
				}

				positions.clear();
				edgeConnects.clear();
				vColors.clear();
				fn_guideThickMesh.getIsoContour(scalars, 0.0, positions, edgeConnects, vColors);

				// trim with side plane
				zColorArray t_Colors;

				t_positions.clear();
				t_edgeConnects.clear();
				positionVertex.clear();
				tVertex_vertexmap.clear();

				for (int k = 0; k < edgeConnects.size(); k += 2)
				{
					bool internal_v0 = true;
					bool internal_v1 = true;

					zPoint p0 = positions[edgeConnects[k]];
					zPoint p1 = positions[edgeConnects[k + 1]];

					zColor c0 = vColors[edgeConnects[k]];
					zColor c1 = vColors[edgeConnects[k + 1]];

					for (int j = 0; j < 1; j += 1)
					{

						zItMeshFace f(o_planeMesh, (j == 0) ? trimFaces.first : trimFaces.second);
						zPoint O1 = f.getCenter();
						zVector N1 = f.getNormal();
						zColor col = f.getColor();

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
								internal_v1 = true;
								p1 = iPt;
							}
						}



						if (!internal_v0 && internal_v1)
						{
							zPoint iPt;
							bool intersect = coreUtils.line_PlaneIntersection(p1, p0, N1, O1, iPt);
							if (intersect)
							{
								internal_v0 = true;
								p0 = iPt;
							}
						}
					}


					if (internal_v0 && internal_v1)
					{

						if (p0.distanceTo(p1) > distanceTolerance)
						{

							int v0;

							bool v0Exists = coreUtils.vertexExists(positionVertex, p0, 6, v0);

							if (!v0Exists)
							{
								v0 = t_positions.size();
								t_positions.push_back(p0);
								coreUtils.addToPositionMap(positionVertex, p0, v0, 6);

								t_Colors.push_back(c0);
							}

							int v1;
							bool v1Exists = coreUtils.vertexExists(positionVertex, p1, 6, v1);
							if (!v1Exists)
							{
								v1 = t_positions.size();
								t_positions.push_back(p1);
								coreUtils.addToPositionMap(positionVertex, p1, v1, 6);

								t_Colors.push_back(c1);
							}

							t_edgeConnects.push_back(v0);
							t_edgeConnects.push_back(v1);
						}



					}

				}

				zObjGraph o_tempThk;
				zFnGraph tempThkFn(o_tempThk);
				tempThkFn.create(t_positions, t_edgeConnects);;
				//tempThkFn.setEdgeColor(grey);

				tempThkFn.setVertexColors(t_Colors);

				zItGraphVertexArray thk_boundaryVerts;
				zIntArray start, end;

				zItMeshFace f(o_planeMesh, trimFaces.first);
				zPoint O1 = f.getCenter();
				zVector N1 = f.getNormal();
				for (zItGraphVertex v(o_tempThk); !v.end(); v++)
				{


					if (v.getValence() == 1)
					{
						thk_boundaryVerts.push_back(v);
						zPoint p0 = v.getPosition();
						float d = abs(coreUtils.minDist_Point_Plane(p0, O1, N1));

						if (d < 0.1)
						{
							float distToSP = p0.distanceTo(pointOnStartPlane);
							if (distToSP < 1) start.push_back(v.getId());

						}

					}
				}



				// boundary block
				if (end.size() == 0)
				{
					t_positions.clear();
					t_edgeConnects.clear();
					t_Colors.clear();

					zItGraphVertex vStart(o_tempThk, start[0]);
					zItGraphHalfEdge he = vStart.getHalfEdge();

					t_positions.push_back(vStart.getPosition());
					t_Colors.push_back(vStart.getColor());

					bool exitLoop = false;
					do
					{
						if (t_positions.size() > 0)
						{
							t_edgeConnects.push_back(t_positions.size() - 1);
							t_edgeConnects.push_back(t_positions.size());
						}

						t_positions.push_back(he.getVertex().getPosition());

						t_Colors.push_back(he.getVertex().getColor());

						if (he.getVertex().checkValency(1))  exitLoop = true;
						else he = he.getNext();

					} while (!exitLoop);

					t_edgeConnects.push_back(0);
					t_edgeConnects.push_back(t_positions.size() - 1);

					t_Colors[0] = yellow;
					t_Colors[t_positions.size() - 1] = yellow;
				}


				zFnGraph ThkFn(_block.o_sectionGuideGraphs[i]);
				ThkFn.create(t_positions, t_edgeConnects);;
				ThkFn.setEdgeColor(grey);

				ThkFn.setVertexColors(t_Colors);

				
				
			}

		}

		//printf("\n b %i | sf %i  sv %i ", _block.id, _block.sectionFrames.size(), _block.sectionGraphs_startVertex.size());

	}

	ZSPACE_INLINE bool zTsSDFBridge::checkPrintLayerHeights(zPrintBlock& _block)
	{
		float minLayerHeight = 10;
		float maxLayerHeight = 0;

		bool checkReds = true;

		int r0 = 0;
		int r1 = floor(_block.o_sectionGraphs.size() * 0.5) - 1;
		int r2 = floor(_block.o_sectionGraphs.size() * 0.5);
		int r3 = (_block.o_sectionGraphs.size()) - 1;

		printf("\n r: %i  %i %i %i ", r0, r1, r2, r3);
		int end = (onBoundaryBlock(_block.id)) ? _block.o_sectionGraphs.size() : floor(_block.o_sectionGraphs.size() * 0.5);

		float printLength = 0;

		// PRINT LAYERS
		for (int j = 0; j < end; j++)
		{
			int kStart = ((onBoundaryBlock(_block.id)) && _block.right_BoundaryFaces.size() == 0) ? 1 : 0;
			int kEnd = ((onBoundaryBlock(_block.id)) && _block.left_BoundaryFaces.size() == 0) ? 1 : 2;



			for (int k = kStart; k < kEnd; k++)
			{
				int i = (k == 0) ? j : j + end;

				if (onBoundaryBlock(_block.id)) i = j;

				if (i == r0) continue;

				if (!onBoundaryBlock(_block.id) && i == r2) continue;


				zVector norm(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));
				norm *= -1;
				norm.normalize();

				zVector prevNorm(_block.sectionFrames[i - 1](2, 0), _block.sectionFrames[i - 1](2, 1), _block.sectionFrames[i - 1](2, 2));
				zVector prevOrigin(_block.sectionFrames[i - 1](3, 0), _block.sectionFrames[i - 1](3, 1), _block.sectionFrames[i - 1](3, 2));

				float layerWidth = 0.025;

				int redCounter = 0; 

				for (zItGraphVertex v(_block.o_sectionGuideGraphs[i]); !v.end(); v++)
				{
					

					zPoint p = v.getPosition();

					if (v.getColor() == red) redCounter++;

					zPoint p1 = p + norm * 1.0;

					zPoint intPt;
					bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

					//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

					float layerHeight = intPt.distanceTo(p);
					maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
					minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

										
					//cout << layerHeight << endl;
				}

				
				if (onBoundaryBlock(_block.id))
				{					
					if (redCounter != 7) checkReds = false;
				}
				else
				{
					if (redCounter != 3 ) checkReds = false;
				}


				for (zItGraphEdge e(_block.o_sectionGuideGraphs[i]); !e.end(); e++)
				{
					printLength += e.getLength();
				}
								
				
			}
		}

		printf("\n block %i | %1.4f %1.4f %s | %1.1f ", _block.id, minLayerHeight, maxLayerHeight, (checkReds) ? "reds T" : "reds F", printLength);

		return checkReds;
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlock_bounds(zPrintBlock& _block)
	{
		if (_block.id == -1) return;

		zTransform t = _block.sectionFrames[0];

		zTransform tLocal;
		tLocal.setIdentity();

		zPoint minBB (10000, 10000, 10000);
		zPoint maxBB (-10000, -10000, -10000);

		zPoint localMinBB(10000, 10000, 10000);
		zPoint localMaxBB(-10000, -10000, -10000);

		string folderName = "C:/Users/vishu.b/Desktop/VeniceIO/inc3d_bounds/" + to_string(_block.id);
		mkdir(folderName.c_str());

		int id = 0;

		for (zObjGraph& g : _block.o_sectionGuideGraphs)
		{
			zFnGraph fnGraphGuide(g);

			fnGraphGuide.setTransform(t, true, false);			
			fnGraphGuide.setTransform(tLocal, true, true);

			string outName = folderName;
			outName += "/";
			outName += "guide_";
			outName += to_string(_block.id) + "_" + to_string(id)+ ".json";

			fnGraphGuide.to(outName, zJSON);
			id++;
			/*zPointArray positions;
			fnGraphGuide.getVertexPositions(positions);
			zPoint tempMinBB, tempMaxBB, tempLocalMinBB, tempLocalMaxBB;;
			coreUtils.getBounds(positions, tempMinBB, tempMaxBB);*/

			/*if (tempMinBB.x < minBB.x) minBB.x = tempMinBB.x;
			if (tempMinBB.y < minBB.y) minBB.y = tempMinBB.y;
			if (tempMinBB.z < minBB.z) minBB.z = tempMinBB.z;

			if (tempMaxBB.x > maxBB.x) maxBB.x = tempMaxBB.x;
			if (tempMaxBB.y > maxBB.y) maxBB.y = tempMaxBB.y;
			if (tempMaxBB.z > maxBB.z) maxBB.z = tempMaxBB.z;*/

			
		}

		//cout << "\n block " << _block.id;
		//cout << " | minBB " << minBB;
		//cout << " | maxBB " << maxBB;
		//cout << " | localMinBB " << localMinBB;
		//cout << " | localMaxBB " << localMaxBB;
		//zVector dims = coreUtils.getDimsFromBounds(minBB, maxBB);
		//printf(" %1.4f, %1.4f, %1.4f " , dims.x, dims.y, dims.z) ;
	}

	ZSPACE_INLINE void zTsSDFBridge::computePrintBlockRailInteriorVertex(zPrintBlock& _block, zItMeshHalfEdge& guideMesh_halfedge, bool boundaryBlock)
	{

		//left block
		if (!guideMesh_halfedge.getNext().getNext().getEdge().onBoundary())
		{

			bool addBlock = true;

			if (boundaryBlock)
			{
				if (guideMesh_halfedge.getFace().getColor() == blockColors[1] || guideMesh_halfedge.getFace().getColor() == blockColors[2])
					addBlock = true;
				else addBlock = false;
			}


			if (addBlock)
			{
				zItMeshHalfEdge he = guideMesh_halfedge;

				bool exit = false;

				while (!exit)
				{
					he = he.getNext().getNext().getSym();	

					if (guide_PrintRailEdgesBoolean[he.getEdge().getId()]) exit = true;
				}

				_block.guideMesh_left_railInteriorVertex = he.getVertex().getId();
			}

		}

		//right block
		if (!guideMesh_halfedge.getSym().getNext().getNext().getEdge().onBoundary())
		{
			bool addBlock = true;

			if (boundaryBlock)
			{
				if (guideMesh_halfedge.getSym().getFace().getColor() == blockColors[1] || guideMesh_halfedge.getSym().getFace().getColor() == blockColors[2])
					addBlock = true;
				else addBlock = false;
			}

			if (addBlock)
			{
				zItMeshHalfEdge he = guideMesh_halfedge.getSym();

				bool exit = false;

				while (!exit)
				{
					he = he.getPrev().getPrev().getSym();

					if (guide_PrintRailEdgesBoolean[he.getEdge().getId()]) exit = true;
				}

				_block.guideMesh_right_railInteriorVertex = he.getStartVertex().getId();
			}
		}


		/*zItMeshVertex v (*o_guideMesh, _block.guideMesh_interiorVertex);

		zItMeshHalfEdgeArray cHEdges;
		v.getConnectedHalfEdges(cHEdges);

		zItMeshHalfEdge he;

		for (auto& cHE : cHEdges)
		{
			if (guide_PrintMedialEdgesBoolean[cHE.getEdge().getId()])
			{
				zColor fColor = cHE.getFace().getColor();			

				if (fColor == blockColor)
				{
					he = cHE;					
				}
			}
		}

		bool exit = false;
		
		if (leftBlock) he = he.getSym();

		while (!exit)
		{
			if (leftBlock) he = he.getNext().getNext().getSym();
			else   he = he.getPrev().getPrev().getSym();

			if (guide_PrintRailEdgesBoolean[he.getEdge().getId()]) exit = true;
		}
				

		return (leftBlock) ? he.getVertex().getId() : he.getStartVertex().getId();*/
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

	ZSPACE_INLINE float zTsSDFBridge::computePrintBlockRailLength(zPrintBlock& _block, zPoint& startPlaneOrigin, zVector& startPlaneNorm, zPoint& endPlaneOrigin, zVector& endPlaneNorm, zColor& blockColor, bool leftBlock, const zIntArray& FlipBlockIds, zItMeshHalfEdgeArray& railIntersectionHE, zPointArray& railIntersectionPoints)
	{
		//int railInteriorVertex = computePrintBlockRailInteriorVertex(_block, blockColor, leftBlock);
		
		int railInteriorVertex = (leftBlock) ? _block.guideMesh_left_railInteriorVertex : _block.guideMesh_right_railInteriorVertex;

		//if (_block.id == 22) cout << endl << railInteriorVertex;

		computePrintBlockRailIntersections(_block, railInteriorVertex, startPlaneOrigin, startPlaneNorm, endPlaneOrigin, endPlaneNorm, FlipBlockIds, railIntersectionHE, railIntersectionPoints);

		zItMeshHalfEdge railStartHE = railIntersectionHE[0];
		zItMeshHalfEdge railEndHE = railIntersectionHE[1];

		/*if (_block.id == 22)
		{
			cout << endl << startPlaneOrigin << startPlaneNorm;
			cout << endl << endPlaneOrigin << endPlaneNorm;

			cout << endl << railIntersectionPoints[0] << railIntersectionPoints[1];
		}*/

		// compute total edge length
		float length = 0;

		length += railStartHE.getStartVertex().getPosition().distanceTo(railIntersectionPoints[0]);
		length += railEndHE.getStartVertex().getPosition().distanceTo(railIntersectionPoints[1]);

		zItMeshHalfEdge walkHe = railStartHE;
		walkHe = walkHe.getSym();
		walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();

		bool exit = false;
		while (walkHe != railEndHE)
		{
			length += walkHe.getLength();
			walkHe = (walkHe.onBoundary()) ? walkHe.getNext() : walkHe.getNext().getSym().getNext();
		}

		

		return length;

	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockSDF_Internal(zPrintBlock& _block, int graphId, int printLayerId, bool rightSide, float printWidth, float neopreneOffset, bool addRaft, int raftId, float raftWidth, bool exportBMP)
	{
		if (_block.id == -1) return;
		if (graphId >= _block.o_sectionGraphs.size())return;

		zFnGraph fnGraph(_block.o_sectionGraphs[graphId]);

		/*float pWidth = (addRaft) ? raftWidth : printWidth;
		float inc3dWidth = (addRaft) ? 0.030:0.025;*/

		float pWidth = (addRaft) ? printWidth : printWidth;
		float inc3dWidth = (addRaft) ? 0.025 : 0.025;

		//fnGraph.averageVertices(2);

		zPoint* positions = fnGraph.getRawVertexPositions();


		zFnGraph fnGraphGuide(_block.o_sectionGuideGraphs[graphId]);
		zPoint* positionsGuide = fnGraphGuide.getRawVertexPositions();

		zTransform t = _block.sectionFrames[graphId];
		fnGraph.setTransform(t, true, false);

		fnGraphGuide.setTransform(t, true, false);

		zPoint o(t(3, 0), t(3, 1), t(3, 2));
		zVector n(t(2, 0), t(2, 1), t(2, 2));



		//// clip planes
		int startVId = _block.sectionGraphs_startEndVertex[graphId].first;
		int endVId = _block.sectionGraphs_startEndVertex[graphId].second;


		zIntPair p = _block.sectionTrimFaces[graphId];
		bool boundarySide = (p.first == -1) ? true : false;

		zPointArray fCen;
		fCen.assign(2, zPoint());

		zVectorArray fNorm;
		fNorm.assign(2, zPoint());

		zColorArray fCol;
		fCol.assign(2, zColor());

		zPoint startPlaneOrigin;
		zVector startPlaneNorm;

		zPoint endPlaneOrigin;
		zVector endPlaneNorm;

		int startIndex = (boundarySide) ? 1 : 0;
		for (int i = startIndex; i < 2; i += 1)
		{
			zItMeshFace f(o_planeMesh, (i == 0) ? p.first : p.second);

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
			fNorm[i] = f.getNormal();
			fNorm[i].normalize();

			// point inwards
			fNorm[i] *= -1;

			fCen[i] = f.getCenter();

			if (i == 0)
			{
				startPlaneNorm = fNorm[i];
				startPlaneOrigin = fCen[i];
			}

			if (i == 1)
			{
				endPlaneNorm = fNorm[i];
				endPlaneOrigin = fCen[i];
			}

			fCol[i] = f.getColor();

			/*if (fCol[i] == yellow)
			{

				fCen[i] += fNorm[i] * ((pWidth * 0.5) + neopreneOffset);
			}*/
			//if (fCol[i] == blue) 	fCen[i] += fNorm[i] * ((pWidth * 0.37));

			// revert to original position
			int counter = 0;
			for (auto& v : fVerts)
			{
				v.setPosition(fVertPos[counter]);
				counter++;
			}

			f.updateNormal();

		}



		// TOP BOTTOM HALF EDGES
		zItGraphHalfEdgeArray topHE, bottomHE;
		float topLength, bottomLength;
		polyTopBottomEdges(_block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, startPlaneOrigin, startPlaneNorm, 120);

		printf("\n %i %1.6f  %1.6f ", graphId, topLength, bottomLength);


		// Transform

		zTransform tLocal;
		tLocal.setIdentity();
		fnGraph.setTransform(tLocal, true, true);

		fnGraphGuide.setTransform(tLocal, true, true);

		zPoint v_startPos = positions[startVId];
		zPoint v_endPos = positions[endVId];


		

		// Project start plane
				
		zVector XStart = topHE[0].getVector();
		XStart.normalize();
		zVector YStart = topHE[0].getStartVertex().getPosition() - bottomHE[0].getStartVertex().getPosition();
		YStart.normalize();

		zVector ZStart = XStart ^ YStart;
		ZStart.normalize();

		XStart =  YStart ^ ZStart;
		XStart.normalize();
		

		zPoint cenStart = (topHE[0].getStartVertex().getPosition() + bottomHE[0].getStartVertex().getPosition()) * 0.5;

		//cout <<endl << XStart << cenStart;

		// Project end plane

		zVector XEnd = topHE[topHE.size() - 1].getVector();
		XEnd.normalize();
		zVector YEnd = topHE[topHE.size() - 1].getVertex().getPosition() - bottomHE[bottomHE.size() -1 ].getVertex().getPosition() ;
		YStart.normalize();

		zVector ZEnd = XEnd ^ YEnd;
		ZEnd.normalize();

		XEnd = YEnd ^ ZEnd;
		XEnd.normalize();
		XEnd *= -1;

		zPoint cenEnd = (bottomHE[bottomHE.size() - 1].getVertex().getPosition() + topHE[topHE.size() - 1].getVertex().getPosition()) * 0.5;

		//cout << endl << XEnd << cenEnd;

		int numInfillTriangles = 3;
		float maxTriagleLength = 0.35;

		// field
		zFnMeshScalarField fnField(o_field);

		float offset_outer = 0.5 * pWidth;
		float offset_inner = 1.5 * pWidth;

		// Profile polygon field
		zScalarArray polyField;
		fnField.getScalars_Polygon(polyField, _block.o_sectionGuideGraphs[graphId], false);

		// EDGE field outer

		zScalarArray edgeField_outer = polyField;;
		for (auto& s : edgeField_outer) s += offset_outer;

		zScalarArray trim_outer; 
		fnField.getScalars_3dp_topBottomTrim(trim_outer, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE,  0.5 * pWidth,0, false);

		zScalarArray booleanField_outer;
		fnField.boolean_subtract(edgeField_outer, trim_outer, booleanField_outer, false);


		// PATTERN

		zScalarArray patternField;
		fnField.getScalars_3dp_Pattern(patternField, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, maxTriagleLength, pWidth,  false);

		//fnField.smoothField(patternField, 1, 0.0);

		zScalarArray booleanField_outer2;
		fnField.boolean_union(booleanField_outer, patternField, booleanField_outer2, false);

		for (int i = startIndex; i < 2; i += 1)
		{
			//fNorm[i].normalize();

			zPoint cen = (i == 0) ? cenStart : cenEnd;
			zVector norm = (i == 0) ? XStart : XEnd;

			if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset);
			if (i == 1) cen += norm * ((inc3dWidth * 0.25));

			zScalarArray clipPlaneField;
			fnField.boolean_clipwithPlane(booleanField_outer2, clipPlaneField, cen, norm);

			booleanField_outer2.clear();
			booleanField_outer2 = clipPlaneField;
		}

			
		if (addRaft)
		{
			float offset = (raftWidth - printWidth) * 0.5;
			//printf("\n raft offset: %1.4f ", offset);
			for (auto& s : booleanField_outer2) s -= (offset);

			for (int i = 1; i < 2; i += 1)
			{
				//fNorm[i].normalize();

				zPoint cen = (i == 0) ? cenStart : cenEnd;
				zVector norm = (i == 0) ? XStart : XEnd;

				if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset);
				if (i == 1) cen += norm * ((inc3dWidth * 0.25));

				zScalarArray clipPlaneField;
				fnField.boolean_clipwithPlane(booleanField_outer2, clipPlaneField, cen, norm);

				booleanField_outer2.clear();
				booleanField_outer2 = clipPlaneField;
			}
		}
		

		fnField.smoothField(booleanField_outer2, 1, 0.0);

		// EDGE field inner

		zScalarArray edgeField_inner = polyField;
		for (auto& s : edgeField_inner) s += offset_inner;

		zScalarArray trim_inner;
		fnField.getScalars_3dp_topBottomTrim(trim_inner, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, 1.5 * pWidth,0, false);


		zScalarArray booleanField_inner;
		fnField.boolean_subtract(edgeField_inner, trim_inner, booleanField_inner, false);

		////clip
		
		for (int i = startIndex; i < 2; i += 1)
		{
			//fNorm[i].normalize();

			zPoint cen = (i == 0) ? cenStart : cenEnd;
			zVector norm = (i == 0) ? XStart : XEnd;

			if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset + (pWidth * 1.0));

			if (i == 1 ) cen += norm * ((pWidth * 1.0));
			//if (i == 1 && addRaft) cen += norm * ((inc3dWidth * 0.25) + (pWidth * 1.0));
			//zPoint cen = fCen[i] + (fNorm[i] * pWidth * 1.0);

			zScalarArray clipPlaneField;

			fnField.boolean_clipwithPlane(booleanField_inner, clipPlaneField, cen, fNorm[i]);

			booleanField_inner.clear();
			booleanField_inner = clipPlaneField;
		}
		
		

		//fnField.smoothField(booleanField_inner, 1, 0.0);


		//INFILL

		zScalarArray infillField;
		fnField.getScalars_3dp_Infill(infillField, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, maxTriagleLength, pWidth, false);

		zScalarArray booleanField_infill;
		fnField.boolean_intersect(booleanField_inner, infillField, booleanField_infill, false);

		//TRIANGLE TRIM,
		zScalarArray triangleTrim;
		//fnField.getScalars_3dp_InfillTrim(triangleTrim, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, maxTriagleLength,  pWidth * 0.5, false);
		
		fnField.getScalars_3dp_InfillTrim(triangleTrim, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, maxTriagleLength,(addRaft) ? 0.0125 :  printWidth * 0.5, false, _block.o_trimGraphs[graphId]);

		zFnGraph fnTrimGraph(_block.o_trimGraphs[graphId]);
		fnTrimGraph.setTransform(t, true, true);

		
		//// BOOLEAN FIELDS

		zScalarArray booleanField_trim;

		zScalarArray booleanField_combined;
		fnField.boolean_subtract(booleanField_outer2, booleanField_infill, booleanField_combined, false);

		fnField.boolean_subtract(booleanField_combined, triangleTrim, booleanField_trim, false);
				
		fnField.smoothField(booleanField_trim, 1, 0.0);

		fnField.setFieldValues(booleanField_trim);

		if (exportBMP)
		{
			/*string folderName = "C:/Users/vishu.b/Desktop/VeniceIO_0104/final/blocks/animation/bitmaps/" + to_string(_block.id);
			mkdir(folderName.c_str());

			folderName += "/" + to_string(rightSide);
			mkdir(folderName.c_str());

			string outName = folderName;
			outName += "/bmp_";
			outName += to_string(_block.id) + "_" + to_string(printLayerId) + "_" + to_string(rightSide) + ".bmp";
			fnField.to(outName, zBMP);*/

			/*string folderName = "C:/Users/vishu.b/Desktop/VeniceIO_0104/final/blocks/animation/field/" + to_string(_block.id);
			mkdir(folderName.c_str());

			folderName += "/" + to_string(rightSide);
			mkdir(folderName.c_str());

			fnField.fnMesh.setTransform(t, true, true);

			string outName = folderName;
			outName += "/field_";
			outName += to_string(_block.id) + "_" + to_string(printLayerId) + "_" + to_string(rightSide) + ".json";
			fnField.to(outName, zJSON);

			fnField.fnMesh.setTransform(tLocal, true, true);*/
		}

		

		//// CONTOURS
		if (addRaft)
		{			

			zFnGraph fnRaft(_block.o_raftGraphs[raftId]);
			fnField.getIsocontour(_block.o_raftGraphs[raftId], 0.0);


			// transform 
			fnRaft.setEdgeWeight(2);
			fnRaft.setTransform(t, true, true);

			fnRaft.setEdgeColor(green);
		}
		else
		{			

			zFnGraph fnIsoGraph(_block.o_contourGraphs[graphId]);
			fnField.getIsocontour(_block.o_contourGraphs[graphId], 0.0);

			// transform 
			fnIsoGraph.setEdgeWeight(2);
			//fnIsoGraph.setTransform(t, true, true);

			//fnIsoGraph.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/contour_deck_5_" + to_string(graphId) + ".json", zJSON);

		}

		//fnField.fnMesh.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/field_deck_5_" + to_string(graphId) + ".json", zJSON);
		
		//fnGraphGuide.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/guide_deck" + to_string(graphId) + ".json", zJSON);

		// transform back 
		fnGraph.setTransform(t, true, true);
		fnGraphGuide.setTransform(t, true, true);

		// Export for animation 
		//zColor temp;
		//zColorArray vCols;
		//zDomainColor dCol(zColor(0.950, 0, 0.55, 1), zColor(0, 0.550, 0.950, 1));
		//for (int i = 0; i < booleanField_trim.size(); i++)
		//{
		//	if (booleanField_trim[i] < -0.01)
		//	{
		//		//temp = coreUtils.blendColor(scalars[i], dVal, dCol, zHSV);

		//		temp = dCol.max;
		//	}
		//	else if (booleanField_trim[i] > 0.01)
		//	{

		//		temp = zColor(0.25, 0.25, 0.25, 1) /*dCol.max*/;
		//	}
		//	else temp = dCol.min;;

		//	vCols.push_back(temp);
		//}

		//fnField.fnMesh.setVertexColors(vCols);

		//fnField.fnMesh.setTransform(t, true, true);
		//string oN = "C:/Users/vishu.b/Desktop/SDF/22/sdf_" + to_string(graphId) + ".json";
		//fnField.fnMesh.to(oN, zJSON);

		//fnField.fnMesh.setTransform(tLocal, true, true);

		json j;

		//if(graphId == 1) fnField.fnMesh.to("C:/Users/vishu.b/Desktop/SDF/field_deck.json", zJSON);

		//vector<float> tMatrix;
		//for (int l = 0; l < 16; l++) tMatrix.push_back(t(l));
		//
		//vector<int> midValIndices;
		//vector<int> lowValIndices;
		//for (int l = 0; l < booleanField_trim.size(); l++)
		//{
		//	if (booleanField_trim[l] >= -0.01 && booleanField_trim[l] <= 0.01)
		//	{
		//		midValIndices.push_back(l);
		//	}
		//	if (booleanField_trim[l] < -0.01) lowValIndices.push_back(l);
		//}

		//j["transform"] = tMatrix;
		//j["midValues"] = midValIndices;
		//j["lowValues"] = lowValIndices;

		//// EXPORT	
		//
		//string oN = "C:/Users/vishu.b/Desktop/SDF/22/sdf_" + to_string(graphId) + ".json";
		//ofstream myfile;
		//myfile.open(oN.c_str());

		////myfile.precision(16);
		//myfile << j.dump();
		//myfile.close();

	}

	ZSPACE_INLINE void zTsSDFBridge::computeBlockSDF_Boundary(zPrintBlock& _block, int graphId, float printWidth, float neopreneOffset, bool addRaft , int raftId, float raftWidth)
	{
		if (_block.id == -1) return;
		if (graphId >= _block.o_sectionGraphs.size())return;

		zFnGraph fnGraph(_block.o_sectionGraphs[graphId]);

		float pWidth = (addRaft) ? printWidth : printWidth;
		float inc3dWidth = (addRaft) ? 0.020 : 0.020;		

		zPoint* positions = fnGraph.getRawVertexPositions();


		zFnGraph fnGraphGuide(_block.o_sectionGuideGraphs[graphId]);
		zPoint* positionsGuide = fnGraphGuide.getRawVertexPositions();

		zTransform t = _block.sectionFrames[graphId];
		//cout << endl << t;
		fnGraph.setTransform(t, true, false);

		fnGraphGuide.setTransform(t, true, false);

		zPoint o(t(3, 0), t(3, 1), t(3, 2));
		zVector n(t(2, 0), t(2, 1), t(2, 2));



		//// clip planes
		int startVId = _block.sectionGraphs_startEndVertex[graphId].first;
		int endVId = _block.sectionGraphs_startEndVertex[graphId].second;


		zIntPair p = _block.sectionTrimFaces[graphId];
		bool boundarySide = (p.second == -1) ? true : false;

		zPointArray fCen;
		fCen.assign(2, zPoint());

		zVectorArray fNorm;
		fNorm.assign(2, zPoint());

		zColorArray fCol;
		fCol.assign(2, zColor());

		zPoint startPlaneOrigin;
		zVector startPlaneNorm;

		zPoint endPlaneOrigin;
		zVector endPlaneNorm;

		int endIndex = (boundarySide) ? 1:0;
		for (int i = 0; i < endIndex; i += 1)
		{
			zItMeshFace f(o_planeMesh, (i == 0) ? p.first : p.second);

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
			fNorm[i] = f.getNormal();
			fNorm[i].normalize();

			// point inwards
			fNorm[i] *= -1;

			fCen[i] = f.getCenter();

			fCol[i] = f.getColor();

			if (i == 0)
			{
				startPlaneNorm = fNorm[i];
				startPlaneOrigin = fCen[i];
			}

			if (i == 1)
			{
				endPlaneNorm = fNorm[i];
				endPlaneOrigin = fCen[i];
			}

			//if (fCol[i] == yellow) 	fCen[i] += fNorm[i] * ((printWidth * 0.5) + neopreneOffset);
			//if (fCol[i] == blue) 	fCen[i] += fNorm[i] * ((printWidth * 0.37));

			// revert to original position
			int counter = 0;
			for (auto& v : fVerts)
			{
				v.setPosition(fVertPos[counter]);
				counter++;
			}

			f.updateNormal();

		}


		// TOP BOTOM HALF EDGES
		zItGraphHalfEdgeArray topHE, bottomHE;
		float topLength, bottomLength;
		polyTopBottomEdges(_block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, startPlaneOrigin, startPlaneNorm, 120);

		printf("\n %i %1.6f  %1.6f ", graphId, topLength, bottomLength);


		// Transform

		zTransform tLocal;
		tLocal.setIdentity();
		fnGraph.setTransform(tLocal, true, true);

		fnGraphGuide.setTransform(tLocal, true, true);

		zPoint v_startPos = positions[startVId];
		zPoint v_endPos = positions[endVId];

		

		// Project start plane

		zVector XStart = topHE[0].getVector();
		XStart.normalize();
		zVector YStart = topHE[0].getStartVertex().getPosition() - bottomHE[0].getStartVertex().getPosition();
		YStart.normalize();

		zVector ZStart = XStart ^ YStart;
		ZStart.normalize();

		XStart = YStart ^ ZStart;
		XStart.normalize();


		zPoint cenStart = (topHE[0].getStartVertex().getPosition() + bottomHE[0].getStartVertex().getPosition()) * 0.5;

		//cout <<endl << XStart << cenStart;

		// Project end plane

		zVector XEnd = topHE[topHE.size() - 1].getVector();
		XEnd.normalize();
		zVector YEnd = topHE[topHE.size() - 1].getVertex().getPosition() - bottomHE[bottomHE.size() - 1].getVertex().getPosition();
		YStart.normalize();

		zVector ZEnd = XEnd ^ YEnd;
		ZEnd.normalize();

		XEnd = YEnd ^ ZEnd;
		XEnd.normalize();
		XEnd *= -1;

		zPoint cenEnd = (bottomHE[bottomHE.size() - 1].getVertex().getPosition() + topHE[topHE.size() - 1].getVertex().getPosition()) * 0.5;

		//cout << endl << XEnd << cenEnd;

		int numInfillTriangles = 4;



		// field
		zFnMeshScalarField fnField(o_field);

		float offset_outer = 0.5 * pWidth;
		float offset_inner = 1.5 * pWidth;

		// Profile polygon field
		zScalarArray polyField;
		fnField.getScalars_Polygon(polyField, _block.o_sectionGuideGraphs[graphId], false);

		// EDGE field outer

		zScalarArray edgeField_outer = polyField;;
		for (auto& s : edgeField_outer) s += offset_outer;

		//zScalarArray trim_outer;
		//fnField.getScalars_3dp_topBottomTrim(trim_outer, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, 0.5 * pWidth,0, false);

		zScalarArray booleanField_outer = edgeField_outer;
		//fnField.boolean_subtract(edgeField_outer, trim_outer, booleanField_outer, false);

		// CLIP 
		
		for (int i = 0; i < endIndex; i += 1)
		{
			//fNorm[i].normalize();

			zPoint cen = (i == 0) ? cenStart : cenEnd;
			zVector norm = (i == 0) ? XStart : XEnd;

			if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset);
			if (i == 1) cen += norm * ((inc3dWidth * 0.25));

			zScalarArray clipPlaneField;
			fnField.boolean_clipwithPlane(booleanField_outer, clipPlaneField, cen, norm);

			booleanField_outer.clear();
			booleanField_outer = clipPlaneField;
		}

		if (addRaft)
		{
			float offset = (raftWidth - printWidth) * 0.5;
			//printf("\n raft offset: %1.4f ", offset);

			for (auto& s : booleanField_outer) s -= (offset);

			for (int i = 1; i < 2; i += 1)
			{
				//fNorm[i].normalize();

				zPoint cen = (i == 0) ? cenStart : cenEnd;
				zVector norm = (i == 0) ? XStart : XEnd;

				if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset);
				if (i == 1) cen += norm * ((inc3dWidth * 0.25));

				zScalarArray clipPlaneField;
				fnField.boolean_clipwithPlane(booleanField_outer, clipPlaneField, cen, norm);

				booleanField_outer.clear();
				booleanField_outer = clipPlaneField;
			}
		}

		fnField.smoothField(booleanField_outer, 1, 0.0);
		
		

		// EDGE field inner

		zScalarArray edgeField_inner = polyField;
		for (auto& s : edgeField_inner) s += offset_inner;

		//zScalarArray trim_inner;
		//fnField.getScalars_3dp_topBottomTrim(trim_inner, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, 1.5 * pWidth,0, false);


		zScalarArray booleanField_inner = edgeField_inner;
		//fnField.boolean_subtract(edgeField_inner, trim_inner, booleanField_inner, false);

		//fnField.smoothField(booleanField_inner, 2, 0.0);

		////clip
				
		for (int i = 0; i < endIndex; i += 1)
		{

			zPoint cen = (i == 0) ? cenStart : cenEnd;
			zVector norm = (i == 0) ? XStart : XEnd;

			if (i == 0) cen += norm * ((inc3dWidth * 0.5) + neopreneOffset + (pWidth * 1.0));

			if (i == 1) cen += norm * ((pWidth * 1.0));
			//if (i == 1 && addRaft) cen += norm * ((inc3dWidth * 0.25) + (pWidth * 1.0));

			zScalarArray clipPlaneField;

			fnField.boolean_clipwithPlane(booleanField_inner, clipPlaneField, cen, fNorm[i]);

			booleanField_inner.clear();
			booleanField_inner = clipPlaneField;
		}


		//fnField.smoothField(booleanField_inner, 2, 0.0);


		//INFILL

		//zScalarArray circleTrim;
		//zPoint cen = (topHE[topHE.size() - 1].getVertex().getPosition() + bottomHE[bottomHE.size() - 1].getVertex().getPosition()) * 0.5;
		//fnField.getScalars_Circle(circleTrim, cen, 0.06, 0.0, false);

		zScalarArray infillField;
		fnField.getScalars_3dp_InfillBoundary(infillField, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, 0.24, (addRaft) ? 0.010 : printWidth * 0.50, false);

		zScalarArray booleanField_infill;
		fnField.boolean_intersect(booleanField_inner, infillField, booleanField_infill, false);

		zScalarArray trim_infill;
		fnField.getScalars_3dp_InfillInteriorTrimBoundary(trim_infill, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, 0.24, (addRaft) ? 0.010 : printWidth * 0.50,(graphId %2 == 0), false, _block.o_trimGraphs[graphId]);

		zFnGraph fnTrimGraph(_block.o_trimGraphs[graphId]);
		fnTrimGraph.setTransform(t, true, true);

		zScalarArray booleanField_infill2;
		fnField.boolean_subtract(booleanField_inner, booleanField_infill, booleanField_infill2, false);

		zScalarArray booleanField_infill3;
		fnField.boolean_union(booleanField_infill2, trim_infill, booleanField_infill3, false);

		//zScalarArray booleanField_infill4 = booleanField_infill3;
		////fnField.boolean_subtract(booleanField_infill3, circleTrim, booleanField_infill4, false);

		fnField.smoothField(booleanField_infill3, 1, 0.0);

		////TRIANGLE TRIM,
		zScalarArray triangleTrim;
		fnField.getScalars_3dp_InfillTrimBoundary(triangleTrim, _block.o_sectionGuideGraphs[graphId], topHE, bottomHE, topLength, bottomLength, numInfillTriangles, 0.24, (addRaft) ? 0.010 : printWidth * 0.50, (graphId % 2 == 0), false);

		


		//// BOOLEAN FIELDS

		zScalarArray booleanField_trim;

		zScalarArray booleanField_combined;
		fnField.boolean_subtract(booleanField_outer, booleanField_infill3, booleanField_combined, false);

		fnField.boolean_subtract(booleanField_combined, triangleTrim, booleanField_trim, false);
		fnField.setFieldValues(booleanField_trim);

		

		//// CONTOURS
		if (addRaft)
		{

			zFnGraph fnRaft(_block.o_raftGraphs[raftId]);
			fnField.getIsocontour(_block.o_raftGraphs[raftId], 0.0);


			// transform 
			fnRaft.setEdgeWeight(2);
			fnRaft.setTransform(t, true, true);

			fnRaft.setEdgeColor(green);
		}
		else
		{

			zFnGraph fnIsoGraph(_block.o_contourGraphs[graphId]);
			fnField.getIsocontour(_block.o_contourGraphs[graphId], 0.0);

			// transform 
			fnIsoGraph.setEdgeWeight(2);
			//fnIsoGraph.setTransform(t, true, true);

			fnIsoGraph.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/contour_balustrade_5_" + to_string(graphId) + ".json", zJSON);

		}

		

		

		fnField.fnMesh.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/field_balustrade_5_" + to_string(graphId) + ".json", zJSON);

		fnGraphGuide.to("C:/Users/vishu.b/Desktop/VeniceIO_0104/sdf/guide_balustrade" + to_string(graphId) + ".json", zJSON);

		// transform back 
		//fnGraph.setTransform(t, true, true);
		//fnGraphGuide.setTransform(t, true, true);

		// Export for animation 
		json j;

		//if(graphId == 1) fnField.fnMesh.to("C:/Users/vishu.b/Desktop/SDF/field_balustrade.json", zJSON);

		vector<float> tMatrix;
		for (int l = 0; l < 16; l++) tMatrix.push_back(t(l));

		vector<int> midValIndices;
		vector<int> lowValIndices;
		for (int l = 0; l < booleanField_infill2.size(); l++)
		{
			if (booleanField_infill2[l] >= -0.01 && booleanField_infill2[l] <= 0.01)
			{
				midValIndices.push_back(l);
			}
			if (booleanField_infill2[l] < -0.01) lowValIndices.push_back(l);
		}

		j["transform"] = tMatrix;
		j["midValues"] = midValIndices;
		j["lowValues"] = lowValIndices;

		// EXPORT	

		string oN = "C:/Users/vishu.b/Desktop/SDF/27/sdf_" + to_string(graphId) + ".json";
		ofstream myfile;
		myfile.open(oN.c_str());

		//myfile.precision(16);
		myfile << j.dump();
		myfile.close();
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

	ZSPACE_INLINE void zTsSDFBridge::polyTopBottomEdges(zObjGraph& inPoly, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, zPoint startPlaneOrigin, zVector startPlaneNorm, float hardVertexTolerance)
	{
		zFnGraph inFnGraph(inPoly);
		inFnGraph.setEdgeColor(zColor());

		float dotTolerance = cos(hardVertexTolerance * DEG_TO_RAD);

		zVector up(0, 0, 1);

		bottomHE.clear();
		topHE.clear();

		bottomLength = 0;
		topLength = 0;

		zVector* inPositions = inFnGraph.getRawVertexPositions();
		zColor* inColors = inFnGraph.getRawVertexColors();

		zIntArray startVerts;

		for (int i = 0; i < inFnGraph.numVertices(); i++)
		{
			//float d = coreUtils.minDist_Point_Plane(inPositions[i], startPlaneOrigin, startPlaneNorm);

			//if (abs(d) < distanceTolerance)startVerts.push_back(i);

			if (inColors[i] == yellow) startVerts.push_back(i);;
		}

		
		// Flip if needed, to have lower point in 0 index
		if (inPositions[startVerts[1]].z < inPositions[startVerts[0]].z)
		{
			int temp = startVerts[0];

			startVerts[0] = startVerts[1];
			startVerts[1] = temp;
		}


		zItGraphHalfEdge tHe;
		zItGraphHalfEdge bHe;

		for (int i = 0; i < 2; i++)
		{
			zItGraphVertex v(inPoly, startVerts[i]);

			zItGraphHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			if (i == 0)
			{
				bHe = (cHEdges[0].getVertex().getId() == startVerts[1]) ? cHEdges[1] : cHEdges[0];
			}

			if (i == 1)
			{
				tHe = (cHEdges[0].getVertex().getId() == startVerts[0]) ? cHEdges[1] : cHEdges[0];
			}

		}

		zBoolArray vertVisited;
		vertVisited.assign(inFnGraph.numVertices(), false);

		vertVisited[startVerts[0]] = true;
		vertVisited[startVerts[1]] = true;

		bool exit = false;
		while (!exit)
		{
			bottomHE.push_back(bHe);
			bottomLength += bHe.getLength();

			bHe.getEdge().setColor(zColor(0, 0, 1, 1));
			//bHe.getEdge().setWeight(3);

			zItGraphVertex v = bHe.getVertex();

			if (v.getColor() == orange) exit = true;

			/*zItGraphHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);

			zVector vec0 = cHEdges[0].getVector();
			zVector vec1 = cHEdges[1].getVector();

			vec0.normalize();
			vec1.normalize();

			if (vec0 * vec1 > dotTolerance)  
			{
				printf("\n dot %1.2f %1.2f ", vec0 * vec1, dotTolerance);
				exit = true;
			}*/

			bHe = bHe.getNext();
		}

		exit = false;
		while (!exit)
		{
			topHE.push_back(tHe);
			topLength += tHe.getLength();

			tHe.getEdge().setColor(zColor(1, 0, 0, 1));
			//tHe.getEdge().setWeight(1);

			zItGraphVertex v = tHe.getVertex();
			if (v.getColor() == orange) exit = true;

			//zItGraphHalfEdgeArray cHEdges;
			//v.getConnectedHalfEdges(cHEdges);

			//zVector vec0 = cHEdges[0].getVector();
			//zVector vec1 = cHEdges[1].getVector();

			//vec0.normalize();
			//vec1.normalize();

			//if (vec0 * vec1 > dotTolerance)
			//{	
			//	printf("\n dot %1.2f %1.2f ", vec0 * vec1, dotTolerance);
			//	exit = true;
			//}

			tHe = tHe.getNext();
		}

		//printf("\n length : %1.3f %1.3f ", topLength, bottomLength);

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

	ZSPACE_INLINE void zTsSDFBridge::blockConvexToJSON(zPrintBlock &_block, string dir, string filename)
	{
		
		string folderName = dir;

		//RIGHT
		if (_block.right_BoundaryFaces.size() > 0)
		{
			zPointArray positions;
			zIntArray polyConnects, polyCount;

			zColorArray vertexColors;

			for (auto fId : _block.right_BoundaryFaces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPointArray fVerts;
				f.getVertexPositions(fVerts);

				int numV = positions.size();

				polyConnects.push_back(numV + 0);
				polyConnects.push_back(numV + 1);
				polyConnects.push_back(numV + 2);
				polyConnects.push_back(numV + 3);

				polyCount.push_back(4);

				for (auto vPos : fVerts)
				{
					positions.push_back(vPos);

					vertexColors.push_back(green);
				}
			}

			for (auto fId : _block.right_sideFaces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPointArray fVerts;
				f.getVertexPositions(fVerts);

				int numV = positions.size();

				polyConnects.push_back(numV + 0);
				polyConnects.push_back(numV + 1);
				polyConnects.push_back(numV + 2);
				polyConnects.push_back(numV + 3);

				polyCount.push_back(4);

				for (auto vPos : fVerts)
				{
					positions.push_back(vPos);

					vertexColors.push_back(f.getColor());
				}
			}


			zObjMesh tempMesh;
			zFnMesh tempFn(tempMesh);

			tempFn.create(positions, polyCount, polyConnects);
			tempFn.setVertexColors(vertexColors);

			string outName = folderName;
			outName += "/";
			outName += filename;
			outName += "_";
			outName += to_string(_block.id) + "_" +  to_string(0) + ".json";

			tempFn.to(outName, zJSON);

		}


		//LEFT
		if (_block.left_BoundaryFaces.size() > 0)
		{
			zPointArray positions;
			zIntArray polyConnects, polyCount;

			zColorArray vertexColors;

			for (auto fId : _block.left_BoundaryFaces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPointArray fVerts;
				f.getVertexPositions(fVerts);

				int numV = positions.size();

				polyConnects.push_back(numV + 0);
				polyConnects.push_back(numV + 1);
				polyConnects.push_back(numV + 2);
				polyConnects.push_back(numV + 3);

				polyCount.push_back(4);

				for (auto vPos : fVerts)
				{
					positions.push_back(vPos);

					vertexColors.push_back(green);
				}
			}

			for (auto fId : _block.left_sideFaces)
			{
				zItMeshFace f(o_planeMesh, fId);

				zPointArray fVerts;
				f.getVertexPositions(fVerts);

				int numV = positions.size();

				polyConnects.push_back(numV + 0);
				polyConnects.push_back(numV + 1);
				polyConnects.push_back(numV + 2);
				polyConnects.push_back(numV + 3);

				polyCount.push_back(4);

				for (auto vPos : fVerts)
				{
					positions.push_back(vPos);

					vertexColors.push_back(f.getColor());
				}
			}


			zObjMesh tempMesh;
			zFnMesh tempFn(tempMesh);

			tempFn.create(positions, polyCount, polyConnects);
			tempFn.setVertexColors(vertexColors);

			string outName = folderName;
			outName += "/";
			outName += filename;
			outName += "_";
			outName += to_string(_block.id) + "_" + to_string(1) + ".json";

			tempFn.to(outName, zJSON);

		}

	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToJSON(int blockId, string dir, string filename, float printLyerWidth, float raftLayerWidth)
	{
		if (blockId >= printBlocks.size()) return;
		zPrintBlock _block = printBlocks[blockId];

		int r0 = 0;
		int r1 = floor(_block.o_sectionGraphs.size() * 0.5) - 1;
		int r2 = floor(_block.o_sectionGraphs.size() * 0.5);
		int r3 = (_block.o_sectionGraphs.size()) - 1;

		printf("\n r: %i  %i %i %i ", r0, r1, r2, r3);
		int end = (onBoundaryBlock(blockId)) ? _block.o_sectionGraphs.size() :  floor(_block.o_sectionGraphs.size() * 0.5);

		_block.minLayerHeight = 10;
		_block.maxLayerHeight = 0;

	
		
		string folderName = dir + "/" + to_string(_block.id);
		mkdir(folderName.c_str());
		for (const auto& entry : std::filesystem::directory_iterator(folderName)) std::filesystem::remove_all(entry.path());

		
		
		blockConvexToJSON(_block, folderName, "3dp_bounds");

		// PRINT LAYERS
		for (int j = 0; j < end ; j++)
		{
			int kStart = ((onBoundaryBlock(blockId)) && _block.right_BoundaryFaces.size() == 0) ? 1 : 0;
			int kEnd = ((onBoundaryBlock(blockId)) && _block.left_BoundaryFaces.size() == 0) ? 1 : 2;



			for (int k = kStart; k < kEnd; k++)
			{
				int i = (k == 0) ? j : j + end;

				if(onBoundaryBlock(blockId)) i = j;

				if (i == r0 ) continue;

				if (!onBoundaryBlock(blockId) && i == r2) continue;

				//if (!onBoundaryBlock(blockId))
				//{
					string outName1 = folderName;
					outName1 += "/";
					outName1 += "trim";
					outName1 += "_";
					outName1 += to_string(_block.id) + "_" + to_string(j) + "_" + to_string(k) + ".json";

					zFnGraph fnTrimGraph(_block.o_trimGraphs[i]);
					fnTrimGraph.to(outName1, zJSON);

				//}

				string outName = folderName;
				outName += "/";
				outName += filename;
				outName += "_";
				outName += to_string(_block.id) + "_" + to_string(j) + "_" + to_string(k) + ".json";

				zFnGraph fnIsoGraph(_block.o_contourGraphs[i]);
				fnIsoGraph.to(outName, zJSON);


				// read existing data in the json 
				json jSON;

				ifstream in_myfile;
				in_myfile.open(outName.c_str());

				int lineCnt = 0;

				if (in_myfile.fail())
				{
					cout << " error in opening file  " << outName.c_str() << endl;
				}

				in_myfile >> jSON;
				in_myfile.close();

				// CREATE JSON FILE
				zUtilsJsonHE graphJSON;


				graphJSON.vertexAttributes.clear();
				graphJSON.vertexAttributes = (jSON["VertexAttributes"].get<vector<vector<double>>>());

				// blockId attributes
				vector<vector<double>> blockAttributes;

				zVector norm(_block.sectionFrames[i](2, 0), _block.sectionFrames[i](2, 1), _block.sectionFrames[i](2, 2));
				norm *= -1;
				norm.normalize();

				zVector prevNorm(_block.sectionFrames[i - 1](2, 0), _block.sectionFrames[i - 1](2, 1), _block.sectionFrames[i - 1](2, 2));
				zVector prevOrigin(_block.sectionFrames[i - 1](3, 0), _block.sectionFrames[i - 1](3, 1), _block.sectionFrames[i - 1](3, 2));
							
				float layerWidth = 0.025;
				for (zItGraphVertex v(_block.o_contourGraphs[i]); !v.end(); v++)
				{
					vector<double> v_attrib = graphJSON.vertexAttributes[v.getId()];

					
					zPoint p = v.getPosition();

					zPoint p1 = p + norm * 1.0;

					zPoint intPt;
					bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

					//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

					float layerHeight = intPt.distanceTo(p);
					_block.maxLayerHeight = (layerHeight > _block.maxLayerHeight) ? layerHeight : _block.maxLayerHeight;
					_block.minLayerHeight = (layerHeight < _block.minLayerHeight) ? layerHeight : _block.minLayerHeight;


					v_attrib.push_back(norm.x);
					v_attrib.push_back(norm.y);
					v_attrib.push_back(norm.z);

					v_attrib.push_back(printLyerWidth);
					v_attrib.push_back(layerHeight);

					graphJSON.vertexAttributes[v.getId()] = v_attrib;

				}

				// Json file 
				
				jSON["VertexAttributes"] = graphJSON.vertexAttributes;
				

				// EXPORT	
				ofstream myfile;
				myfile.open(outName.c_str());

				if (myfile.fail())
				{
					cout << " error in opening file  " << outName.c_str() << endl;
					return;
				}

				//myfile.precision(16);
				myfile << jSON.dump();
				myfile.close();

			}
		}

		printf("\n block %i | %1.4f %1.4f ", _block.id, _block.minLayerHeight, _block.maxLayerHeight);

		// RAFT
				
		int kEnd = ((onBoundaryBlock(blockId))) ? 1 : 2;
		for (int k = 0; k < kEnd; k++)
		{
			int i = k;
			//int rId = 0;
			if((onBoundaryBlock(blockId)) && _block.right_BoundaryFaces.size() == 0) k = 1;

			int layerId = 0;
			int prevlayerId = 0;


			string outName_raft = folderName;
			outName_raft += "/";
			outName_raft += filename;
			outName_raft += "_raft_";
			outName_raft += to_string(_block.id) + "_" + to_string(layerId) + "_" + to_string(k) + ".json";

			zFnGraph fnIsoGraph(_block.o_raftGraphs[i]);
			fnIsoGraph.to(outName_raft, zJSON);


			// read existing data in the json 
			json jSON;

			ifstream in_myfile;
			in_myfile.open(outName_raft.c_str());

			int lineCnt = 0;

			if (in_myfile.fail())
			{
				cout << " error in opening file  " << outName_raft.c_str() << endl;
			}

			in_myfile >> jSON;
			in_myfile.close();

			// CREATE JSON FILE
			zUtilsJsonHE graphJSON;


			graphJSON.vertexAttributes.clear();
			graphJSON.vertexAttributes = (jSON["VertexAttributes"].get<vector<vector<double>>>());

			// blockId attributes
			vector<vector<double>> blockAttributes;

			int id = (k == 0) ? 0 : end;

			zVector norm(_block.sectionFrames[id](2, 0), _block.sectionFrames[id](2, 1), _block.sectionFrames[id](2, 2));
			norm *= -1;
			norm.normalize();

			zVector prevNorm(_block.sectionFrames[prevlayerId](2, 0), _block.sectionFrames[prevlayerId](2, 1), _block.sectionFrames[prevlayerId](2, 2));
			zVector prevOrigin(_block.sectionFrames[prevlayerId](3, 0), _block.sectionFrames[prevlayerId](3, 1), _block.sectionFrames[prevlayerId](3, 2));

			float maxLayerHeight = 0;
			float minLayerHeight = 10;
			float layerWidth = raftLayerWidth;
			for (zItGraphVertex v(_block.o_raftGraphs[i]); !v.end(); v++)
			{
				vector<double> v_attrib = graphJSON.vertexAttributes[v.getId()];


				zPoint p = v.getPosition();

				zPoint p1 = p + norm * 1.0;

				zPoint intPt;
				bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

				//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

				/*float layerHeight = intPt.distanceTo(p);
				maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
				minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;*/

				float layerHeight = 0.01;

				v_attrib.push_back(norm.x);
				v_attrib.push_back(norm.y);
				v_attrib.push_back(norm.z);

				v_attrib.push_back(raftLayerWidth);
				v_attrib.push_back(layerHeight);

				graphJSON.vertexAttributes[v.getId()] = v_attrib;

			}

			// Json file 

			jSON["VertexAttributes"] = graphJSON.vertexAttributes;


			// EXPORT	
			ofstream myfile;
			myfile.open(outName_raft.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outName_raft.c_str() << endl;
				return;
			}

			//myfile.precision(16);
			myfile << jSON.dump();
			myfile.close();

		}
		
		blockContoursToIncr3D(blockId, folderName, "3dp", printLyerWidth);


	}

	ZSPACE_INLINE void zTsSDFBridge::blockContoursToIncr3D(int blockId, string dir, string filename, float layerWidth)
	{
		
		for (auto& b : printBlocks)
		{
			if (b.id == -1) continue;			

			if (b.id != blockId) continue;

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

		
			// start - end planes
			myfile << "Right Planes " << "\n";
			myfile << "/* " << "\n";
			
			if (b.rightBlocks.size() > 0)
			{
				int start = 0;
				int end = (b.leftBlocks.size() == 0) ? b.sectionFrames.size() -1 : floor(b.sectionFrames.size() * 0.5) - 1;

				zTransform rStart = b.sectionFrames[start];
				zTransform rEnd = b.sectionFrames[end];

				zItMeshFace rFstart(o_planeMesh, b.right_sectionPlaneFace_GuideSmoothEdge[0].first);
				zItMeshFace rFend(o_planeMesh, b.right_sectionPlaneFace_GuideSmoothEdge[1].first);

				myfile << rStart(3,0) << "," << rStart(3, 1) << "," << rStart(3, 2) << ",";
				myfile << rStart(2, 0) << "," << rStart(2, 1) << "," << rStart(2, 2) << ",";
				myfile << 0 << "\n";

				myfile << rEnd(3, 0) << "," << rEnd(3, 1) << "," << rEnd(3, 2) << ",";
				myfile << rEnd(2, 0) << "," << rEnd(2, 1) << "," << rEnd(2, 2) << ",";
				myfile << 0 << "\n";

				for (int i = 0; i < b.right_sideFaces.size(); i++)
				{
					zItMeshFace f(o_planeMesh, b.right_sideFaces[i]);

					if (f.getColor() == blue)
					{
						myfile << f.getCenter().x << "," << f.getCenter().y << "," << f.getCenter().z << ",";
						myfile << f.getNormal().x << "," << f.getNormal().y << "," << f.getNormal().z << ",";
						myfile <<  1 << "\n";
					}	

					if (f.getColor() == yellow)
					{
						myfile << f.getCenter().x << "," << f.getCenter().y << "," << f.getCenter().z << ",";
						myfile << f.getNormal().x << "," << f.getNormal().y << "," << f.getNormal().z << ",";
						myfile << 2 << "\n";
					}
					
				}
			}
					
			myfile << "*/ " << "\n";

			myfile << "Left Planes " << "\n";
			myfile << "/* " << "\n";

			if (b.leftBlocks.size() > 0)
			{
				int start = (b.rightBlocks.size() == 0) ? 0 : floor(b.sectionFrames.size() * 0.5);
				int end = b.sectionFrames.size() - 1;

				zTransform lStart = b.sectionFrames[start];
				zTransform lEnd = b.sectionFrames[end];

				zItMeshFace lFstart(o_planeMesh, b.left_sectionPlaneFace_GuideSmoothEdge[0].first);
				zItMeshFace lFend(o_planeMesh, b.left_sectionPlaneFace_GuideSmoothEdge[1].first);

				myfile << lStart(3, 0) << "," << lStart(3, 1) << "," << lStart(3, 2) << ",";
				myfile << lStart(2, 0) << "," << lStart(2, 1) << "," << lStart(2, 2) << ",";
				myfile << 0 << "\n";

				myfile << lEnd(3, 0) << "," << lEnd(3, 1) << "," << lEnd(3, 2) << ",";
				myfile << lEnd(2, 0) << "," << lEnd(2, 1) << "," << lEnd(2, 2) << ",";
				myfile << 0 << "\n";

				for (int i = 0; i < b.left_sideFaces.size(); i++)
				{
					zItMeshFace f(o_planeMesh, b.left_sideFaces[i]);

					if (f.getColor() == blue)
					{
						myfile << f.getCenter().x << "," << f.getCenter().y << "," << f.getCenter().z << ",";
						myfile << f.getNormal().x << "," << f.getNormal().y << "," << f.getNormal().z << ",";
						myfile << 1 << "\n";
					}

					if (f.getColor() == yellow)
					{
						myfile << f.getCenter().x << "," << f.getCenter().y << "," << f.getCenter().z << ",";
						myfile << f.getNormal().x << "," << f.getNormal().y << "," << f.getNormal().z << ",";
						myfile << 2 << "\n";
					}

				}

			}

			myfile << "*/ " << "\n";

				
			//// RAFT LAYERS

			//int raft_startVertexId = -1;
			//zItGraphHalfEdge raft_he;
			//zItGraphVertex raft_v;

			//for (int j = 0; j < 1; j++)
			//{

			//	myfile << "Raft Layer " << j << "\n";
			//	myfile << "/* " << "\n";

			//	// RIGHT 
			//	if (b.rightBlocks.size() > 0)
			//	{
			//		int prevlayerId = 0;

			//		int rID = (b.leftBlocks.size() >0)?  j*2 : j;
			//		zFnGraph fnG(b.o_raftGraphs[rID]);
			//		if (fnG.numVertices() == 0)
			//		{
			//			myfile << "*/ " << "\n";
			//			continue;
			//		}

			//		bool flip = false;

			//		if (raft_startVertexId == -1) raft_startVertexId = 0;
			//		else
			//		{
			//			zPoint startP_Prev = raft_v.getPosition();
			//			float dist = 10000;

			//			zPoint* positions = fnG.getRawVertexPositions();
			//			for (int i = 0; i < fnG.numVertices(); i++)
			//			{
			//				float d = positions[i].distanceTo(startP_Prev);

			//				if (d < dist)
			//				{
			//					dist = d;
			//					raft_startVertexId = i;
			//				}
			//			}

			//			zItGraphVertex temp(b.o_raftGraphs[rID], raft_startVertexId);

			//			flip = (temp.getHalfEdge().getVector() * raft_he.getVector() < 0) ? true : false;
			//		}

			//		raft_v = zItGraphVertex(b.o_raftGraphs[rID], raft_startVertexId);
			//		raft_v.setColor(green);

			//		raft_he = raft_v.getHalfEdge();
			//		if (flip) raft_he = raft_he.getSym();

			//		zItGraphHalfEdge start = raft_he;
			//		start.getVertex().setColor(blue);

			//		zVector norm(b.sectionFrames[prevlayerId +1](2, 0), b.sectionFrames[prevlayerId + 1](2, 1), b.sectionFrames[prevlayerId + 1](2, 2));
			//		norm *= -1;
			//		norm.normalize();

			//		zVector prevNorm(b.sectionFrames[prevlayerId](2, 0), b.sectionFrames[prevlayerId](2, 1), b.sectionFrames[prevlayerId](2, 2));
			//		zVector prevOrigin(b.sectionFrames[prevlayerId](3, 0), b.sectionFrames[prevlayerId](3, 1), b.sectionFrames[prevlayerId](3, 2));

			//		/*int startPlane = 0;
			//		zTransform rStart = b.sectionFrames[startPlane];

			//		zVector prevNorm(rStart(2, 0), rStart(2, 1), rStart(2, 2));
			//		zVector prevOrigin(rStart(3, 0), rStart(3, 1), rStart(3, 2));*/

			//		do
			//		{
			//			
			//			zPoint p = raft_he.getVertex().getPosition();

			//			zPoint p1 = p + norm * 1.0;

			//			zPoint intPt;
			//			bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

			//			//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

			//			float layerHeight = intPt.distanceTo(p);
			//			maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
			//			minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

			//			myfile << p.x << "," << p.y << "," << p.z << ",";
			//			myfile << norm.x << "," << norm.y << "," << norm.z << ",";
			//			myfile << layerWidth << ",";
			//			myfile << layerHeight << ",";
			//			myfile << 0 << "\n";

			//			//zItGraphVertex vStart = raft_he.getStartVertex();
			//			/*zItGraphVertex vEnd = raft_he.getVertex();

			//			zItGraphHalfEdgeArray cHEdges;
			//			raft_he.getVertex().getConnectedHalfEdges(cHEdges);

			//			raft_he = (cHEdges[0].getVertex() == vStart) ? cHEdges[1] : cHEdges[0];*/

			//			//raft_he = (raft_he.getNext().getVertex() == vStart) ? raft_he.getPrev() : raft_he.getNext();

			//			raft_he = raft_he.getNext();

			//		} while (raft_he != start);

			//		//printf("\n raft1  working ");

			//	}

			//	// LEFT 
			//	if (b.leftBlocks.size() > 0)
			//	{
			//		int prevlayerId = (b.rightBlocks.size() == 0) ? 0 : floor(b.sectionFrames.size() * 0.5) ;

			//		printf("\n p: %i  %i ", prevlayerId, prevlayerId + 1);
			//		raft_startVertexId = -1;

			//		int lID = (b.rightBlocks.size() > 0) ? j*2 + 1 : j;
			//		zFnGraph fnG(b.o_raftGraphs[lID]);
			//		if (fnG.numVertices() == 0)
			//		{
			//			myfile << "*/ " << "\n";
			//			continue;
			//		}

			//		bool flip = false;

			//		if (raft_startVertexId == -1) raft_startVertexId = 0;
			//		else
			//		{
			//			zPoint startP_Prev = raft_v.getPosition();
			//			float dist = 10000;

			//			zPoint* positions = fnG.getRawVertexPositions();
			//			for (int i = 0; i < fnG.numVertices(); i++)
			//			{
			//				float d = positions[i].distanceTo(startP_Prev);

			//				if (d < dist)
			//				{
			//					dist = d;
			//					raft_startVertexId = i;
			//				}
			//			}

			//			zItGraphVertex temp(b.o_raftGraphs[lID], raft_startVertexId);

			//			flip = (temp.getHalfEdge().getVector() * raft_he.getVector() < 0) ? true : false;
			//		}

			//		raft_v = zItGraphVertex(b.o_raftGraphs[lID], raft_startVertexId);
			//		raft_v.setColor(green);

			//		raft_he = raft_v.getHalfEdge();
			//		if (flip) raft_he = raft_he.getSym();

			//		zItGraphHalfEdge start = raft_he;
			//		start.getVertex().setColor(blue);

			//		zVector norm(b.sectionFrames[prevlayerId +1](2, 0), b.sectionFrames[prevlayerId + 1](2, 1), b.sectionFrames[prevlayerId + 1](2, 2));
			//		norm *= -1;
			//		norm.normalize();

			//		zVector prevNorm(b.sectionFrames[prevlayerId](2, 0), b.sectionFrames[prevlayerId](2, 1), b.sectionFrames[prevlayerId](2, 2));
			//		zVector prevOrigin(b.sectionFrames[prevlayerId](3, 0), b.sectionFrames[prevlayerId](3, 1), b.sectionFrames[prevlayerId](3, 2));

			//		/*int startPlane = (b.rightBlocks.size() == 0) ? 0 : floor(b.sectionFrames.size() * 0.5);;
			//		zTransform lStart = b.sectionFrames[startPlane];

			//		zVector prevNorm(lStart(2, 0), lStart(2, 1), lStart(2, 2));
			//		zVector prevOrigin(lStart(3, 0), lStart(3, 1), lStart(3, 2));*/

			//		do
			//		{
			//			zPoint p = raft_he.getVertex().getPosition();

			//			zPoint p1 = p + norm * 1.0;

			//			zPoint intPt;
			//			bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

			//			//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

			//			float layerHeight = intPt.distanceTo(p);
			//			maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
			//			minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

			//			myfile << p.x << "," << p.y << "," << p.z << ",";
			//			myfile << norm.x << "," << norm.y << "," << norm.z << ",";
			//			myfile << layerWidth << ",";
			//			myfile << layerHeight << ",";
			//			myfile << 1 << "\n";

			//			//zItGraphVertex vStart = raft_he.getStartVertex();
			//			/*zItGraphVertex vEnd = raft_he.getVertex();

			//			zItGraphHalfEdgeArray cHEdges;
			//			raft_he.getVertex().getConnectedHalfEdges(cHEdges);

			//			raft_he = (cHEdges[0].getVertex() == vStart) ? cHEdges[1] : cHEdges[0];*/

			//			//raft_he = (raft_he.getNext().getVertex() == vStart) ? raft_he.getPrev() : raft_he.getNext();

			//			raft_he = raft_he.getNext();

			//		} while (raft_he != start);

			//		//printf("\n raft working ");
			//	}

			//	myfile << "*/ " << "\n";
			//}

			//// PRINT LAYERS
			//int startVertexId = -1;
			//zItGraphHalfEdge he;
			//zItGraphVertex v;
			//float totalLength = 0;

			//int start = 0;
			//int end = (b.rightBlocks.size() == 0 || b.leftBlocks.size() == 0) ? b.sectionFrames.size() : floor(b.sectionFrames.size() * 0.5);			


			//for (int j = 1; j < end ; j++)
			//{		
			//	
			//	myfile << "Layer " << j << "\n";
			//	myfile << "/* " << "\n";


			//	// RIGHT 
			//	if (b.rightBlocks.size() > 0)
			//	{
			//		int rID = j;
			//		zFnGraph fnG(b.o_contourGraphs[rID]);
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
			//			for (int i = 0; i < fnG.numVertices(); i++)
			//			{
			//				float d = positions[i].distanceTo(startP_Prev);

			//				if (d < dist)
			//				{
			//					dist = d;
			//					startVertexId = i;
			//				}
			//			}

			//			zItGraphVertex temp(b.o_contourGraphs[rID], startVertexId);

			//			flip = (temp.getHalfEdge().getVector() * he.getVector() < 0) ? true : false;
			//		}

			//		v = zItGraphVertex(b.o_contourGraphs[rID], startVertexId);
			//		v.setColor(green);

			//		he = v.getHalfEdge();
			//		if (flip) he = he.getSym();

			//		zItGraphHalfEdge start = he;
			//		start.getVertex().setColor(blue);

			//		zVector norm(b.sectionFrames[rID](2, 0), b.sectionFrames[rID](2, 1), b.sectionFrames[rID](2, 2));
			//		norm *= -1;
			//		norm.normalize();

			//		zVector prevNorm(b.sectionFrames[rID - 1](2, 0), b.sectionFrames[rID - 1](2, 1), b.sectionFrames[rID - 1](2, 2));
			//		zVector prevOrigin(b.sectionFrames[rID - 1](3, 0), b.sectionFrames[rID - 1](3, 1), b.sectionFrames[rID - 1](3, 2));
			//		
			//		

			//		do
			//		{
			//			totalLength += he.getLength();
			//			zPoint p = he.getVertex().getPosition();

			//			zPoint p1 = p + norm * 1.0;

			//			zPoint intPt;
			//			bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

			//			//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

			//			float layerHeight = intPt.distanceTo(p);
			//			maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
			//			minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

			//			myfile << p.x << "," << p.y << "," << p.z << ",";
			//			myfile << norm.x << "," << norm.y << "," << norm.z << ",";
			//			myfile << layerWidth << ",";
			//			myfile << layerHeight << ",";
			//			myfile << 0 << "\n";

			//			//zItGraphVertex vStart = he.getStartVertex();
			//			//zItGraphVertex vEnd = he.getVertex();

			//			//zItGraphHalfEdgeArray cHEdges;
			//			//he.getVertex().getConnectedHalfEdges(cHEdges);
			//			//
			//			//he = (cHEdges[0].getVertex() == vStart) ? cHEdges[1] : cHEdges[0];

			//			////he = (he.getNext().getVertex() == vStart) ? he.getPrev() : he.getNext();

			//			he = he.getNext();

			//		} while (he != start);

			//		//printf("\n layer 1 working ");

			//	}
			//	
			//	// LEFT 
			//	if (b.leftBlocks.size() > 0)
			//	{
			//		startVertexId = -1;

			//		int lID = (b.rightBlocks.size() > 0)? j + end :  j;
			//		zFnGraph fnG(b.o_contourGraphs[lID]);
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
			//			for (int i = 0; i < fnG.numVertices(); i++)
			//			{
			//				float d = positions[i].distanceTo(startP_Prev);

			//				if (d < dist)
			//				{
			//					dist = d;
			//					startVertexId = i;
			//				}
			//			}

			//			zItGraphVertex temp(b.o_contourGraphs[lID], startVertexId);

			//			flip = (temp.getHalfEdge().getVector() * he.getVector() < 0) ? true : false;
			//		}
			//						
			//		v = zItGraphVertex(b.o_contourGraphs[lID], startVertexId);
			//		v.setColor(green);

			//		he = v.getHalfEdge();
			//		if (flip) he = he.getSym();

			//		zItGraphHalfEdge start = he;
			//		start.getVertex().setColor(blue);

			//		zVector norm(b.sectionFrames[lID](2, 0), b.sectionFrames[lID](2, 1), b.sectionFrames[lID](2, 2));
			//		norm *= -1;
			//		norm.normalize();

			//		zVector prevNorm(b.sectionFrames[lID - 1](2, 0), b.sectionFrames[lID - 1](2, 1), b.sectionFrames[lID- 1](2, 2));
			//		zVector prevOrigin(b.sectionFrames[lID - 1](3, 0), b.sectionFrames[lID - 1](3, 1), b.sectionFrames[lID - 1](3, 2));

			//		do
			//		{
			//			totalLength += he.getLength();
			//			zPoint p = he.getVertex().getPosition();

			//			zPoint p1 = p + norm * 1.0;

			//			zPoint intPt;
			//			bool check = coreUtils.line_PlaneIntersection(p, p1, prevNorm, prevOrigin, intPt);

			//			//if (!check) printf("\n %i %i no Intersection ",rID, he.getVertex().getId());

			//			float layerHeight = intPt.distanceTo(p);
			//			maxLayerHeight = (layerHeight > maxLayerHeight) ? layerHeight : maxLayerHeight;
			//			minLayerHeight = (layerHeight < minLayerHeight) ? layerHeight : minLayerHeight;

			//			myfile << p.x << "," << p.y << "," << p.z << ",";
			//			myfile << norm.x << "," << norm.y << "," << norm.z << ",";
			//			myfile << layerWidth << ",";
			//			myfile << layerHeight << ",";
			//			myfile << 1 << "\n";

			//			zItGraphVertex vStart = he.getStartVertex();
			//			zItGraphVertex vEnd = he.getVertex();

			//			//zItGraphHalfEdgeArray cHEdges;
			//			//he.getVertex().getConnectedHalfEdges(cHEdges);

			//			//he = (cHEdges[0].getVertex() == vStart) ? cHEdges[1] : cHEdges[0];

			//			////he = (he.getNext().getVertex() == vStart) ? he.getPrev() : he.getNext();

			//			he = he.getNext();

			//		} while (he != start);

			//		//printf("\n layer 0 working ");
			//	}

			//	myfile << "*/ " << "\n";
			//}

			//myfile << "Attributes " << "\n";
			//myfile << "/* " << "\n";
			//myfile << "Units: meters " << "\n";
			//myfile << "Block ID:  " << b.id << "\n";
			//myfile << "max layer height: " << maxLayerHeight << "\n";
			//myfile << "min layer height: " << minLayerHeight << "\n";
			//myfile << "total print length: " << totalLength << "\n";
			//myfile << "*/ " << "\n";

			myfile.close();

			
		}
	}

	ZSPACE_INLINE void zTsSDFBridge::toBRGJSON(string path, zPointArray &points,  zVectorArray &normals, zPointArray& vThickness)
	{
		zFnMesh fnGuidemesh(*o_guideMesh);
		zFnMesh fnGuideThickmesh(*o_guideThickMesh);

		fnGuidemesh.to(path, zJSON);
		fnGuidemesh.setEdgeColor(zColor());

		zPoint* thickPositions = fnGuideThickmesh.getRawVertexPositions();
		
		int numVerts_guide = fnGuidemesh.numVertices();

		printf("\n numvert %i ", numVerts_guide);

		points.clear();
		normals.clear();
		vThickness.clear();

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

		// blockId attributes
		vector<vector<double>> blockAttributes;

		for (int i = 0; i < numPrintBlocks(); i++)
		{
			zPoint p0 = getBlockInteriorPoint(i);
			
			vector<double> b_attrib;
			b_attrib.push_back(p0.x);
			b_attrib.push_back(p0.y);
			b_attrib.push_back(p0.z);


			blockAttributes.push_back(b_attrib);
		}

		//VertexAttributes

		meshJSON.vertexAttributes.clear();
		meshJSON.vertexAttributes = (j["VertexAttributes"].get<vector<vector<double>>>());

	
		for (zItMeshVertex v(*o_guideMesh); !v.end(); v++)
		{
			vector<double> v_attrib = meshJSON.vertexAttributes[v.getId()];

			zPoint p0 = thickPositions[v.getId()];
			zPoint p1 = thickPositions[v.getId() + numVerts_guide];

			v_attrib.push_back(p0.x);
			v_attrib.push_back(p0.y);
			v_attrib.push_back(p0.z);

			v_attrib.push_back(p1.x);
			v_attrib.push_back(p1.y);
			v_attrib.push_back(p1.z);


			meshJSON.vertexAttributes[v.getId()] = v_attrib;
			
			vThickness.push_back(p0);
			vThickness.push_back(p1);


		}

		//Halfedge planes

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

						he.getEdge().setColor(red);
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
		j["VertexAttributes"] = meshJSON.vertexAttributes;

		j["BlockAttributes"] = blockAttributes;

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

			zPoint p1(0, 0, 0);
			zPoint p2(0, 0, 0);
			
			if (v_attrib.size() == 15)
			{
				 p1 = zPoint(v_attrib[9], v_attrib[10], v_attrib[11]);
				 p2 = zPoint(v_attrib[12], v_attrib[13], v_attrib[14]);
			}
			

			vThickness.push_back(p1);
			vThickness.push_back(p2);
			

		}


		//halfEdge attributes
		meshJSON.halfedgeAttributes.clear();
		meshJSON.halfedgeAttributes = (j["HalfedgeAttributes"].get<vector<vector<double>>>());


		for (zItMeshHalfEdge he(*o_guideMesh); !he.end(); he++)
		{

			//if (he.getId() != 8)continue;
			vector<double> he_attrib = meshJSON.halfedgeAttributes[he.getId()];

			zPoint p ;
			zVector n;

		
			
			
			float d = 0;

			he.getEdge().setColor(zColor(0, 0, 0, 1));

			if (/*n.length() > 0 &&*/ guideHalfEdge_planeFace[he.getId()].size() >0)
			{
				zItMeshFace f(o_planeMesh, guideHalfEdge_planeFace[he.getId()][0]);

				if (f.getColor() == grey)
				{
					//n = zVector();
					//p = zVector();
				}
				else
				{
					 p = zPoint(he_attrib[0], he_attrib[1], he_attrib[2]);
					 n = zVector(he_attrib[3], he_attrib[4], he_attrib[5]);

					he.getEdge().setColor(green);

					int fId = guideHalfEdge_planeFace[he.getId()][0];

					zItMeshFace f(o_planeMesh, fId);
					zVector nf = f.getNormal();
					zPoint pf = f.getCenter();

					if (n * nf < 0) n *= -1;

					d = n * nf;

					if (d < devDomain.min) devDomain.min = d;
					if (d > devDomain.max) devDomain.max = d;

					//p = pf;
				}
				
			
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