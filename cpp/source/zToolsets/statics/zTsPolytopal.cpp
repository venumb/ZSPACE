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


#include<headers/zToolsets/statics/zTsPolytopal.h>

//---- ALGLIB METHOD ------------------------------------------------------------------------------


#ifndef ZSPACE_TS_STATICS_POLYTOPAL_ALGLIB
#define ZSPACE_TS_STATICS_POLYTOPAL_ALGLIB



namespace zSpace
{
	//--- USED for LPA , Alglib methods to be defined outside of class.

	ZSPACE_EXTERN std::vector<double> internal_fAreas;

	ZSPACE_INLINE void loadPath_func(const alglib::real_1d_array &Q, double &func, void *ptr)
	{
		func = 0;
		for (int i = 0; i < Q.length(); i++) func += abs(Q[i]) * internal_fAreas[i];

	}

}

#endif

//---- zTsPolytopal ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsPolytopal::zTsPolytopal() {}

	ZSPACE_INLINE zTsPolytopal::zTsPolytopal(zObjGraph &_formObj, zObjMeshArray &_forceObjs, zObjMeshArray  &_polytopalObjs)
	{
		formObj = &_formObj;
		fnForm = zFnGraph(_formObj);

		for (int i = 0; i < _forceObjs.size(); i++)
		{
			forceObjs.push_back(&_forceObjs[i]);
			fnForces.push_back(_forceObjs[i]);
		}

		for (int i = 0; i < _polytopalObjs.size(); i++)
		{
			polytopalObjs.push_back(&_polytopalObjs[i]);
			fnPolytopals.push_back(_polytopalObjs[i]);
		}

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsPolytopal::~zTsPolytopal() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsPolytopal::createForceFromFiles(zStringArray filePaths, zFileTpye type)
	{

		if (type == zJSON)
		{
			for (int i = 0; i < filePaths.size(); i++)
			{
				if (i < fnForces.size())
				{
					fnForces[i].from(filePaths[i], type);
					//fnForces[i].setFaceColor(zColor(1.0, 0.0, 0.0, 0.3));

					cout <<endl  << filePaths[i];
				}

			}
		}

		else if (type == zOBJ)
		{
			for (int i = 0; i < filePaths.size(); i++)
			{
				if (i < fnForces.size()) fnForces[i].from(filePaths[i], type);
			}
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");

	}

	ZSPACE_INLINE void zTsPolytopal::createFormFromForce(zColor edgeCol, bool includeBoundary, double boundaryEdgelength)
	{

		zPointArray positions;
		zIntArray  edgeConnects;

		zSparseMatrix C_fc;
		getPrimal_FaceCellMatrix(zForceDiagram, C_fc);
		cout << "\n C_fc :" << endl << C_fc << endl;

		zSparseMatrix C_ef;
		getPrimal_EdgeFaceMatrix(zForceDiagram, C_ef);
		cout << "\n C_ef :" << endl << C_ef << endl;

		for (int j = 0; j < fnForces.size(); j++)
		{
			positions.push_back(fnForces[j].getCenter());
		}

		edgeConnects.assign(primal_n_f_i * 2, -1);

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

		if (includeBoundary)
		{
			for (int i = 0; i < GFP_SSP_Face.size(); i++)
			{
				if (GFP_SSP_Face[i])
				{			

					int volId = primalFace_VolumeFace[i][0];
					int faceId = primalFace_VolumeFace[i][1];

					zItMeshFace f(*forceObjs[volId], faceId);

					int numVerts = positions.size();					

					positions.push_back(positions[volId] +  primalFaceNormals[i] * boundaryEdgelength);

					edgeConnects.push_back(volId);
					edgeConnects.push_back(numVerts);
				}
			}
		}		

		fnForm.clear();
		fnFormParticles.clear();	

		printf("\n v : %i  e : %i ", positions.size(), edgeConnects.size());

		fnForm.create(positions, edgeConnects);
		fnForm.setEdgeColor(edgeCol);

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			printf("\n %i | %i %i ", e.getId(), e.getHalfEdge(0).getVertex().getId(), e.getHalfEdge(1).getVertex().getId());
		}
	}

	ZSPACE_INLINE void zTsPolytopal::createPolytopalsFromForce(double offset, double param , int subdivs )
	{
		smoothSubDivs = subdivs;

		for (int i = 0; i < fnForces.size(); i++)
		{
			fnPolytopals[i].clear();
			getPolytopal(i, offset, param, subdivs);
		}
	}

	ZSPACE_INLINE void zTsPolytopal::createPolytopalsFromForce_profile(int user_nEdges, double user_edgeLength, double offset, double param, int subdivs)
	{
		smoothSubDivs = subdivs;

		for (int i = 0; i < fnForces.size(); i++)
		{
			fnPolytopals[i].clear();
			getPolytopal_Profile(i, user_nEdges, user_edgeLength, offset, param, subdivs);
		}

		getPolytopal_Beams_Profile(user_nEdges, user_edgeLength, param);
	}

	ZSPACE_INLINE void zTsPolytopal::createSectionPoints(int user_nEdges, double user_edgeLength)
	{
		// get Yvertex
		Y_vertex.clear();
		get_YVertex(Y_vertex);

		userSection_edgeLength = user_edgeLength;
		userSection_numEdge = user_nEdges;

		userSection_points.clear();

		zTransform t_From;
		t_From.setIdentity();

		for (int i = 0; i < internalFaceIndex_primalFace.size(); i++)
		{
			int j = internalFaceIndex_primalFace[i];

			int volId = primalFace_VolumeFace[j][0];
			int faceId = primalFace_VolumeFace[j][1];

			zItMeshFace f(*forceObjs[volId], faceId);
			int numV = f.getNumVertices();

			zPointArray tempPositions;
			getUserSection(numV, tempPositions);

			zTransform t_to;
			t_to.setIdentity();

			zVector O = primalFaceCenters[j];

			zVector Y = primalVertexPositions[Y_vertex[j]] - O;
			Y.normalize();

			zVector Z = primalFaceNormals[j];

			zVector X = Y ^ Z;
			X.normalize();

			t_to(0, 0) = X.x; t_to(1, 0) = X.y; t_to(2, 0) = X.z;
			t_to(0, 1) = Y.x; t_to(1, 1) = Y.y; t_to(2, 1) = Y.z;
			t_to(0, 2) = Z.x; t_to(1, 2) = Z.y; t_to(2, 2) = Z.z;
			t_to(0, 3) = O.x; t_to(1, 3) = O.y; t_to(2, 3) = O.z;

			zTransform t = coreUtils.PlanetoPlane(t_From, t_to);

			for (auto &p : tempPositions)
			{
				p = p * t;
				userSection_points.push_back(p);
			}

		}

	}

	//----3D GS ITERATIVE 

	ZSPACE_INLINE bool zTsPolytopal::equilibrium(bool &computeTargets, double minmax_Edge, zDomainFloat &deviations, double dT, zIntergrationType type, int numIterations, double angleTolerance, bool colorEdges, bool printInfo)
	{

		if (computeTargets)
		{
			computeFormTargets();

			computeTargets = !computeTargets;
		}

		updateFormDiagram(minmax_Edge, dT, type, numIterations);

		// check deviations	
		bool out = checkParallelity(deviations, angleTolerance, colorEdges, printInfo);

		return out;

	}

	//----3D GS ALGEBRAIC

	ZSPACE_INLINE void zTsPolytopal::getPrimal_GlobalElementIndicies(zDiagramType type, int GFP_SSP_Index, int precisionFac)
	{
		if (type == zForceDiagram)
		{

			if (GFP_SSP_Index < -1 && GFP_SSP_Index >= fnForces.size())  throw std::invalid_argument(" invalid GFP / SSP index.");

			primal_n_v = 0;
			primal_n_v_i = 0;

			primal_n_e = 0;
			primal_n_e_i = 0;

			primal_n_f = 0;
			primal_n_f_i = 0;

			primalFace_VolumeFace.clear();
			primalEdge_VolumeEdge.clear();
			primalEdge_PrimalVertices.clear();

			GFP_SSP_Face.clear();
			GFP_SSP_Edge.clear();
			GFP_SSP_Vertex.clear();

			primalVertexPositions.clear();
			primalFaceCenters.clear();
			primalFaceNormals.clear();
			primalFaceAreas.clear();

			primal_internalFaceIndex.clear();
			primal_internalEdgeIndex.clear();
			primal_internalVertexIndex.clear();

			internalFaceIndex_primalFace.clear();
			internalEdgeIndex_primalEdge.clear();
			internalVertexIndex_primalVertex.clear();

			primalVertex_ConnectedPrimalFaces.clear();

			unordered_map <string, int> positionVertex;
			unordered_map <string, int> faceCenterpositionVertex;




			// face map
			for (int j = 0; j < fnForces.size(); j++)
			{

				zPointArray fCenters;
				zVectorArray fNorms;
				zDoubleArray fAreas;

				fnForces[j].getCenters(zFaceData, fCenters);
				fnForces[j].getFaceNormals(fNorms);

				fnForces[j].getPlanarFaceAreas(fAreas);


				for (int i = 0; i < fCenters.size(); i++)
				{
					int globalFaceId = -1;
					bool chkExists = coreUtils.vertexExists(faceCenterpositionVertex, fCenters[i], precisionFac, globalFaceId);



					if (!chkExists)
					{
						coreUtils.addToPositionMap(faceCenterpositionVertex, fCenters[i], primal_n_f, precisionFac);

						zIntArray  volumeFace = { j,i };
						primalFace_VolumeFace.push_back(volumeFace);

						globalFaceId = primal_n_f;

						primalFaceCenters.push_back(fCenters[i]);

						fNorms[i].normalize();
						primalFaceNormals.push_back(fNorms[i]);
						primalFaceAreas.push_back(fAreas[i]);

						// GFP or SSP are external faces
						if (GFP_SSP_Index == -1) GFP_SSP_Face.push_back(true);
						else GFP_SSP_Face.push_back(false);

						primal_n_f++;
					}
					else
					{
						// GFP or SSP are external faces
						if (GFP_SSP_Index == -1) GFP_SSP_Face[globalFaceId] = false;

						primalFace_VolumeFace[globalFaceId].push_back(j);
						primalFace_VolumeFace[globalFaceId].push_back(i);
					}

					string hashKey_volFace = (to_string(j) + "," + to_string(i));
					volumeFace_PrimalFace[hashKey_volFace] = globalFaceId;
				}
			}

			// GFP or SSP are specified volume
			if (GFP_SSP_Index != -1)
			{
				for (int i = 0; i < fnForces[GFP_SSP_Index].numPolygons(); i++)
				{
					string hashKey_volFace = (to_string(GFP_SSP_Index) + "," + to_string(i));

					std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

					if (gotFace != volumeFace_PrimalFace.end())
					{
						int globalFaceId = gotFace->second;
						GFP_SSP_Face[globalFaceId] = true;
					}
				}


			}

			// compute internal face index
			for (int i = 0; i < GFP_SSP_Face.size(); i++)
			{
				if (!GFP_SSP_Face[i])
				{
					primal_internalFaceIndex.push_back(primal_n_f_i);
					internalFaceIndex_primalFace.push_back(i);
					primal_n_f_i++;
				}
				else primal_internalFaceIndex.push_back(-1);
			}

			// compute primal vertex - connected faces


			// vertex and edge maps
			for (int j = 0; j < fnForces.size(); j++)
			{
				// vertex map
				zPoint *pos = fnForces[j].getRawVertexPositions();

				for (zItMeshVertex v(*forceObjs[j]); !v.end(); v++)
				{
					int vId = v.getId();

					zIntArray  cFaces;
					v.getConnectedFaces(cFaces);

					bool boundaryVertex = false;

					for (int i = 0; i < cFaces.size(); i++)
					{
						string hashKey_volFace = (to_string(j) + "," + to_string(cFaces[i]));

						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

						if (gotFace != volumeFace_PrimalFace.end())
						{
							int globalFaceId = gotFace->second;
							if (GFP_SSP_Face[globalFaceId])
							{
								boundaryVertex = true;
								break;
							}
						}
					}

					if (primal_n_f == primal_n_f_i)
					{
						if (v.onBoundary()) boundaryVertex = true;
					}

					int globalVertexId = -1;
					bool chkExists = coreUtils.vertexExists(positionVertex, pos[vId], precisionFac, globalVertexId);

					if (!chkExists)
					{
						coreUtils.addToPositionMap(positionVertex, pos[vId], primal_n_v, precisionFac);

						primalVertexPositions.push_back(pos[vId]);

						globalVertexId = primal_n_v;
						GFP_SSP_Vertex.push_back(false);

						// primal vertex - connected faces
						primalVertex_ConnectedPrimalFaces.push_back(zIntArray ());

						primal_n_v++;
					}

					string hashKey_volVertex = (to_string(j) + "," + to_string(vId));
					volumeVertex_PrimalVertex[hashKey_volVertex] = globalVertexId;

					if (boundaryVertex) GFP_SSP_Vertex[globalVertexId] = true;

					for (int i = 0; i < cFaces.size(); i++)
					{
						string hashKey_volFace = (to_string(j) + "," + to_string(cFaces[i]));

						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

						if (gotFace != volumeFace_PrimalFace.end())
						{
							int globalFaceId = gotFace->second;

							bool chk = false;

							for (int k = 0; k < primalVertex_ConnectedPrimalFaces[globalVertexId].size(); k++)
							{
								if (primalVertex_ConnectedPrimalFaces[globalVertexId][k] == globalFaceId)
								{
									chk = true;
									break;
								}
							}

							if (!chk) primalVertex_ConnectedPrimalFaces[globalVertexId].push_back(globalFaceId);
						}
					}

				}

				//edge map
				for (zItMeshEdge e(*forceObjs[j]); !e.end(); e++)
				{
					int eId = e.getId();

					zIntArray  eFaces;
					e.getFaces(eFaces);

					bool boundaryEdge = false;

					for (int i = 0; i < eFaces.size(); i++)
					{
						string hashKey_volFace = (to_string(j) + "," + to_string(eFaces[i]));

						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

						if (gotFace != volumeFace_PrimalFace.end())
						{
							int globalFaceId = gotFace->second;
							if (GFP_SSP_Face[globalFaceId])
							{
								boundaryEdge = true;
								break;
							}
						}
					}

					if (primal_n_f == primal_n_f_i)
					{
						if (e.onBoundary()) boundaryEdge = true;
					}

					//if (boundaryEdge) continue;

					int v0, v1;

					string hashKey_v0 = (to_string(j) + "," + to_string(e.getHalfEdge(0).getStartVertex().getId()));
					std::unordered_map<std::string, int>::const_iterator got0 = volumeVertex_PrimalVertex.find(hashKey_v0);

					string hashKey_v1 = (to_string(j) + "," + to_string(e.getHalfEdge(0).getVertex().getId()));
					std::unordered_map<std::string, int>::const_iterator got1 = volumeVertex_PrimalVertex.find(hashKey_v1);


					if (got0 != volumeVertex_PrimalVertex.end() && got1 != volumeVertex_PrimalVertex.end())
					{
						v0 = got0->second;
						v1 = got1->second;

						if (v0 > v1) swap(v0, v1);

						string hashKey_e = (to_string(v0) + "," + to_string(v1));
						std::unordered_map<std::string, int>::const_iterator gotGlobalEdge = primalVertices_PrimalEdge.find(hashKey_e);

						int globalEdgeId;

						if (gotGlobalEdge != primalVertices_PrimalEdge.end())
						{
							globalEdgeId = gotGlobalEdge->second;
						}
						else
						{
							primalVertices_PrimalEdge[hashKey_e] = primal_n_e;

							zIntArray  volumeEdge = { j,eId };
							primalEdge_VolumeEdge.push_back(volumeEdge);

							zIntArray  primalVertices = { v0,v1 };
							primalEdge_PrimalVertices.push_back(primalVertices);

							globalEdgeId = primal_n_e;
							GFP_SSP_Edge.push_back(false);
							primal_n_e++;
						}

						string hashKey_volEdge = (to_string(j) + "," + to_string(eId));
						volumeEdge_PrimalEdge[hashKey_volEdge] = globalEdgeId;

						if (boundaryEdge) GFP_SSP_Edge[globalEdgeId] = true;

					}

				}

			}

			// compute internal vertex index
			for (int i = 0; i < GFP_SSP_Vertex.size(); i++)
			{
				if (!GFP_SSP_Vertex[i])
				{
					primal_internalVertexIndex.push_back(primal_n_v_i);
					internalVertexIndex_primalVertex.push_back(i);
					primal_n_v_i++;
				}
				else primal_internalVertexIndex.push_back(-1);
			}

			// compute internal edge index
			for (int i = 0; i < GFP_SSP_Edge.size(); i++)
			{
				if (!GFP_SSP_Edge[i])
				{
					primal_internalEdgeIndex.push_back(primal_n_e_i);
					internalEdgeIndex_primalEdge.push_back(i);
					primal_n_e_i++;
				}
				else primal_internalEdgeIndex.push_back(-1);
			}

			// compute Y_Vertex
			Y_vertex.clear();
			get_YVertex(Y_vertex);

			printf("\n primal Force:  primal_n_v %i, primal_n_v_i %i ,primal_n_e %i, primal_n_e_i %i , primal_n_f %i  primal_n_f_i %i", primal_n_v, primal_n_v_i, primal_n_e, primal_n_e_i, primal_n_f, primal_n_f_i);

			zSparseMatrix C_ev;
			getPrimal_EdgeVertexMatrix(zForceDiagram, C_ev);
			cout << "\n C_ev :" << endl << C_ev << endl;
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	ZSPACE_INLINE void zTsPolytopal::getDual(double threshold, bool includeBoundary , double boundaryEdgelength)
	{
		VectorXd q;
		getDual_ForceDensities_LPA(q);
		cout << "\n q: " << endl << q << endl;

		bool negativeQ = false;
		bool zeroQ = false;
		for (int i = 0; i < q.rows(); i++)
		{
			if (q(i) < 0) negativeQ = true;
			if (abs(q(i)) >= 0 && abs(q(i)) <= 0.000001) zeroQ = true;
		}

		printf("\n q  has negative values : %s ", (negativeQ) ? "true" : "false");
		printf("\n q  has zero values : %s ", (zeroQ) ? "true" : "false");

		double phi = getDual_Loadpath(q);
		cout << endl << "phi: " << phi << endl;

		zSparseMatrix C_fc;
		getPrimal_FaceCellMatrix(zForceDiagram, C_fc);
		//cout << "\n C_fc :" << endl << C_fc << endl;


		zPointArray positions;
		zIntArray  edgeconnects;

		positions.assign(C_fc.cols(), zPoint(3, 0, 0));

		int verticesVisitedCounter = 0;;
		int EdgeVisitedCounter = 0;;


		vector<bool> vertsVisited;
		vertsVisited.assign(C_fc.cols(), false);

		vector<bool> edgeVisited;
		edgeVisited.assign(C_fc.cols(), false);

		zIntArray  currentVertex = { 0 };

		bool exit = false;



		// compute vertex positions
		do
		{
			zIntArray  temp;
			temp.clear();

			for (int i = 0; i < currentVertex.size(); i++)
			{
				if (vertsVisited[currentVertex[i]]) continue;

				int vertexId = currentVertex[i];

				for (int edgeId = 0; edgeId < C_fc.rows(); edgeId++)
				{

					if (C_fc.coeff(edgeId, vertexId) == 1)
					{
						for (int k = 0; k < C_fc.cols(); k++)
						{
							if (k == vertexId) continue;

							if (C_fc.coeff(edgeId, k) == -1)
							{

								if (!vertsVisited[k])
								{

									positions[k] = positions[vertexId] + primalFaceNormals[internalFaceIndex_primalFace[edgeId]] * q[edgeId];
									temp.push_back(k);
								}
							}
						}
					}


					else if (C_fc.coeff(edgeId, vertexId) == -1)
					{
						for (int k = 0; k < C_fc.cols(); k++)
						{
							if (k == vertexId) continue;

							if (C_fc.coeff(edgeId, k) == 1)
							{

								if (!vertsVisited[k])
								{


									positions[k] = positions[vertexId] + primalFaceNormals[internalFaceIndex_primalFace[edgeId]] * q[edgeId] * -1;
									temp.push_back(k);
								}

							}
						}
					}
				}

				vertsVisited[currentVertex[i]] = true;
				verticesVisitedCounter++;
			}

			currentVertex.clear();
			if (temp.size() == 0) exit = true;

			currentVertex = temp;


		} while (verticesVisitedCounter != C_fc.cols() && !exit);

		for (int edgeId = 0; edgeId < C_fc.rows(); edgeId++)
		{


			for (int vId = 0; vId < C_fc.cols(); vId++)
			{
				if (C_fc.coeff(edgeId, vId) == -1)
				{
					edgeconnects.push_back(vId);
					break;
				}

			}

			for (int vId = 0; vId < C_fc.cols(); vId++)
			{
				if (C_fc.coeff(edgeId, vId) == 1)
				{
					edgeconnects.push_back(vId);
					break;
				}
			}


			EdgeVisitedCounter++;
		}


		if (includeBoundary)
		{
			for (int i = 0; i < GFP_SSP_Face.size(); i++)
			{
				if (GFP_SSP_Face[i])
				{

					int volId = primalFace_VolumeFace[i][0];
					int faceId = primalFace_VolumeFace[i][1];

					zItMeshFace f(*forceObjs[volId], faceId);

					int numVerts = positions.size();

					positions.push_back(positions[volId] + primalFaceNormals[i] * boundaryEdgelength);

					edgeconnects.push_back(volId);
					edgeconnects.push_back(numVerts);
				}
			}
		}

		fnForm.create(positions, edgeconnects);

		fnForm.setEdgeColor(zColor(0, 1, 0, 1));

		setDualEdgeWeightsfromPrimal();

		//vector<double> eLengths;
		//fnForm.getEdgeLengths(eLengths);

		//for (auto eLen : eLengths) printf("\n %1.6f ", eLen);

	}

	//----POLYTOPAL

	ZSPACE_INLINE void zTsPolytopal::closePolytopals()
	{
		for (int i = 0; i < fnPolytopals.size(); i++)
		{
			getClosePolytopalMesh(i);
		}
	}

	ZSPACE_INLINE void zTsPolytopal::getPolytopal(int forceIndex, double offset, double param, int subdivs)
	{
		if (forceIndex > fnForces.size()) throw std::invalid_argument(" error: index out of bounds.");

		int fEdges = 3;
		int splits = ceil(userSection_numEdge / fEdges) - 1;

		zPointArray positions;
		zIntArray polyConnects;
		zIntArray polyCounts;

		int n_v = fnForces[forceIndex].numVertices();
		int n_e = fnForces[forceIndex].numHalfEdges();
		int n_f = fnForces[forceIndex].numPolygons();

		zPoint volCenter = fnForces[forceIndex].getCenter();


		zPointArray fCenters;
		fnForces[forceIndex].getCenters(zFaceData, fCenters);

		vector<zVector> fNormals;
		fnForces[forceIndex].getFaceNormals(fNormals);

		for (zItMeshEdge e(*forceObjs[forceIndex]); !e.end(); e++)
		{
			zIntArray  eFaces;
			e.getFaces(eFaces);

			int startId = 0;
			int lastId = 2 + 2 * splits;

			zPointArray tempPosns;
			zPoint pos0 = e.getHalfEdge(1).getVertex().getPosition();
			zPoint pos1 = e.getHalfEdge(0).getVertex().getPosition();

			zVector dir = pos1 - pos0;
			double eLen = dir.length();
			eLen /= (splits + 1);

			dir.normalize();

			tempPosns.push_back(pos0);

			for (int i = 1; i <= splits; i++)
			{
				tempPosns.push_back(pos0 + dir * eLen*i);
				tempPosns.push_back(pos0 + dir * eLen*i);
			}

			tempPosns.push_back(pos1);


			zPoint* formPositions = fnForm.getRawVertexPositions();

			bool firstFaceBoundary = true;

			int numVerts = positions.size();

			for (int i = 0; i < eFaces.size(); i++)
			{

				string hashKey_volFace = (to_string(forceIndex) + "," + to_string(eFaces[i]));

				std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

				if (gotFace != volumeFace_PrimalFace.end())
				{
					int globalFaceId = gotFace->second;
					if (!GFP_SSP_Face[globalFaceId])
					{

						if (i == 0) firstFaceBoundary = false;

						// insert volume centre points
						if (i == 1 && firstFaceBoundary)
						{
							zPoint vCenter_graphPos = formPositions[forceIndex];

							for (int j = 0; j < tempPosns.size(); j++)
							{
								zVector dir_volCenter_0 = tempPosns[j] - volCenter;
								double len0 = dir_volCenter_0.length();
								dir_volCenter_0.normalize();

								zPoint newPos = volCenter + (dir_volCenter_0 * len0 *offset);
								newPos += (vCenter_graphPos - volCenter);

								positions.push_back(newPos);
							}

						}

						zItGraphEdge form_e(*formObj, primal_internalFaceIndex[globalFaceId]);

						zPoint form_eCen = form_e.getCenter();
						zPoint fCenter = fCenters[eFaces[i]];

						// compute corner points
						for (int j = 0; j < tempPosns.size(); j++)
						{

							zVector dir_fCenter_0 = tempPosns[j] - fCenter;
							double len0 = dir_fCenter_0.length();
							dir_fCenter_0.normalize();
							zPoint newPos = fCenter + (dir_fCenter_0 * len0 *offset);
							newPos += form_eCen - fCenter;
							positions.push_back(newPos);
						}



						// insert volume centre points
						if (i == 0)
						{
							zPoint vCenter_graphPos = formPositions[forceIndex];

							for (int j = 0; j < tempPosns.size(); j++)
							{
								zVector dir_volCenter_0 = tempPosns[j] - volCenter;
								double len0 = dir_volCenter_0.length();
								dir_volCenter_0.normalize();

								zPoint newPos = volCenter + (dir_volCenter_0 * len0 *offset);
								newPos += (vCenter_graphPos - volCenter);

								positions.push_back(newPos);
							}
						}

						// poly connects
						for (int j = 0; j <= splits; j++)
						{
							polyConnects.push_back(numVerts + lastId + j * 2);
							polyConnects.push_back(numVerts + lastId + 1 + j * 2);
							polyConnects.push_back(numVerts + startId + 1 + j * 2);
							polyConnects.push_back(numVerts + startId + j * 2);

							polyCounts.push_back(4);
							//printf("\n %i %i %i %i ", numVerts + startId, numVerts + startId + 1, numVerts + lastId + 1, numVerts + lastId);
						}

						if (i == 0)
						{
							startId += (2 + 2 * splits);
							lastId += (2 + 2 * splits);
						}

					}
				}
			}


		}

		printf("\n %i  %i %i %i ", forceIndex, positions.size(), polyCounts.size(), polyConnects.size());

		if (subdivs == 0) fnPolytopals[forceIndex].create(positions, polyCounts, polyConnects);
	}
	
	ZSPACE_INLINE void zTsPolytopal::getPolytopal_Profile(int forceIndex, int user_nEdges, double user_edgeLength, double offset, double param, int subdivs)
	{
		if (forceIndex > fnForces.size()) throw std::invalid_argument(" error: index out of bounds.");

		userSection_edgeLength = user_edgeLength;
		userSection_numEdge = user_nEdges;

		// 3 is hard coded , fix it to be parametric			
		int fEdges = 3;
		int splits = ceil(userSection_numEdge / fEdges) - 1;

		zPointArray positions;
		zIntArray polyConnects;
		zIntArray polyCounts;

		int n_v = fnForces[forceIndex].numVertices();
		int n_e = fnForces[forceIndex].numHalfEdges();
		int n_f = fnForces[forceIndex].numPolygons();

		// compute vertex_face_ profile index map

		zPointArray fCenters;
		fnForces[forceIndex].getCenters(zFaceData, fCenters);

		vector<zVector> fNormals;
		fnForces[forceIndex].getFaceNormals(fNormals);

		zIntArray  v_f_profileIndex;
		v_f_profileIndex.assign(n_v * n_f, -1);

		zIntArray  temp_CornerIds;
		zPointArray sectionPoints;

		for (zItMeshFace f(*forceObjs[forceIndex]); !f.end(); f++)
		{
			string hashKey_volFace = (to_string(forceIndex) + "," + to_string(f.getId()));
			std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);
			if (gotFace != volumeFace_PrimalFace.end())
			{
				int globalFaceId = gotFace->second;
				int vY = Y_vertex[globalFaceId];

				if (userSection_numEdge == 4 || userSection_numEdge == 5 || userSection_numEdge == 6) temp_CornerIds = zIntArray { 0, 2, 4 };

				else if (userSection_numEdge == 8)temp_CornerIds = zIntArray { 0, 3, 5 };

				else if (userSection_numEdge == 9)temp_CornerIds = zIntArray { 0, 3, 6 };


				// get vertex index of the yVertex
				zIntArray  fVerts;
				f.getVertices(fVerts);

				vector<zItMeshHalfEdge> fHEdges;
				f.getHalfEdges(fHEdges);

				zItMeshHalfEdge startHE(*forceObjs[forceIndex]);

				for (auto he : fHEdges)
				{

					string hashKey_volVertex = (to_string(forceIndex) + "," + to_string(he.getStartVertex().getId()));
					std::unordered_map<std::string, int>::const_iterator gotVertex = volumeVertex_PrimalVertex.find(hashKey_volVertex);

					int globalVertexId = gotVertex->second;

					if (globalVertexId == vY)
					{
						startHE = he;
						break;
					}
				}

				int counter = 0;
				zItMeshHalfEdge he = startHE;
				do
				{
					int id = n_f * f.getId() + he.getStartVertex().getId();

					v_f_profileIndex[id] = temp_CornerIds[counter];
					counter++;

					he = he.getNext();

				} while (he != startHE);


				// compute section points at each face
				zPointArray tempPoints;
				getUserSection(3, tempPoints);


				zTransform t_From;
				t_From.setIdentity();

				zTransform t_to;
				t_to.setIdentity();
				

				zVector O =  fCenters[f.getId()];
				
				if (!GFP_SSP_Face[globalFaceId])
				{
					zItGraphEdge form_e(*formObj, primal_internalFaceIndex[globalFaceId]);
					zVector form_eCen = form_e.getCenter();
					O = form_eCen;
				}

				zVector Y = primalVertexPositions[Y_vertex[globalFaceId]] - O;
				Y.normalize();

				zVector Z = /* fNormals[f.getId()];*/ primalFaceNormals[globalFaceId];
				if (fNormals[f.getId()] * Z < 0) Z *= -1;

				if (!GFP_SSP_Face[globalFaceId])
				{
					zItGraphEdge form_e(*formObj, primal_internalFaceIndex[globalFaceId]);
					zVector form_eVec = form_e.getVector();
					form_eVec.normalize();

					Z = form_eVec;
					if (fNormals[f.getId()] * Z < 0) Z *= -1;
				}

				zVector X = Y ^ Z;
				X.normalize();

				t_to(0, 0) = X.x; t_to(1, 0) = X.y; t_to(2, 0) = X.z;
				t_to(0, 1) = Y.x; t_to(1, 1) = Y.y; t_to(2, 1) = Y.z;
				t_to(0, 2) = Z.x; t_to(1, 2) = Z.y; t_to(2, 2) = Z.z;
				t_to(0, 3) = O.x; t_to(1, 3) = O.y; t_to(2, 3) = O.z;

				zTransform t = coreUtils.PlanetoPlane(t_From, t_to);

				for (auto &p : tempPoints)
				{
					p = p * t;
					sectionPoints.push_back(p);
				}


			}


		}


		// compute volume center
		zPoint volCenter = fnForces[forceIndex].getCenter();



		for (zItMeshEdge e(*forceObjs[forceIndex]); !e.end(); e++)
		{
			vector<zItMeshFace> eFaces;
			e.getFaces(eFaces);

			int startId = 0;
			int lastId = 2 + 2 * splits;

			zPointArray tempPosns;

			int v0 = e.getHalfEdge(1).getVertex().getId();
			int v1 = e.getHalfEdge(0).getVertex().getId();



			zPoint pos0 = e.getHalfEdge(1).getVertex().getPosition();
			zPoint pos1 = e.getHalfEdge(0).getVertex().getPosition();

			zVector dir = pos1 - pos0;
			double eLen = dir.length();
			eLen /= (splits + 1);

			dir.normalize();

			tempPosns.push_back(pos0);

			for (int i = 1; i <= splits; i++)
			{
				tempPosns.push_back(pos0 + dir * eLen*i);
				tempPosns.push_back(pos0 + dir * eLen*i);
			}

			tempPosns.push_back(pos1);


			zPoint* formPositions = fnForm.getRawVertexPositions();

			bool firstFaceBoundary = true;

			int numVerts = positions.size();

			for (int i = 0; i < eFaces.size(); i++)
			{

				string hashKey_volFace = (to_string(forceIndex) + "," + to_string(eFaces[i].getId()));
				std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

				int id_0 = v_f_profileIndex[n_f * eFaces[i].getId() + v0];
				int id_1 = v_f_profileIndex[n_f * eFaces[i].getId() + v1];

				int numVerts_perProfile = floor(sectionPoints.size() / fnForces[forceIndex].numPolygons());

				if (id_0 == 0 && id_1 == temp_CornerIds[temp_CornerIds.size() - 1]) id_0 = numVerts_perProfile;
				if (id_1 == 0 && id_0 == temp_CornerIds[temp_CornerIds.size() - 1]) id_1 = numVerts_perProfile;

				if (gotFace != volumeFace_PrimalFace.end())
				{
					int globalFaceId = gotFace->second;
					if (!GFP_SSP_Face[globalFaceId])
					{


						if (i == 0) firstFaceBoundary = false;

						// insert volume centre points
						if (i == 1 && firstFaceBoundary)
						{
							zPoint vCenter_graphPos = formPositions[forceIndex];

							for (int j = 0; j < tempPosns.size(); j++)
							{
								zVector dir_volCenter_0 = tempPosns[j] - volCenter;
								double len0 = dir_volCenter_0.length();
								dir_volCenter_0.normalize();

								zPoint newPos = volCenter + (dir_volCenter_0 * len0 *offset);
								newPos += (vCenter_graphPos - volCenter);

								positions.push_back(newPos);
							}

						}

						int internalFaceId = primal_internalFaceIndex[globalFaceId];

						zItGraphEdge form_e(*formObj, primal_internalFaceIndex[globalFaceId]);

						zPoint form_eCen = form_e.getCenter();
						zPoint fCenter = fCenters[eFaces[i].getId()];
						zVector fNormal = fNormals[eFaces[i].getId()];

						double eLength = form_e.getLength();

						zVector eVec = form_e.getVector();
						eVec.normalize();

						if (eVec* fNormal < 0) eVec *= -1;

						for (int j = 0; j < tempPosns.size(); j++)
						{

							int closestId = -1;
							if (j == 0)
							{
								int id = id_0 % numVerts_perProfile;
								closestId = (eFaces[i].getId()*numVerts_perProfile) + id;
							}

							else if (j == tempPosns.size() - 1)
							{
								int id = id_1 % numVerts_perProfile;
								closestId = (eFaces[i].getId()*numVerts_perProfile) + id;
							}

							else
							{
								int increment = ceil(j * 0.5);

								int id = (id_0 < id_1) ? (id_0 + numVerts_perProfile + increment) % numVerts_perProfile : (id_0 + numVerts_perProfile - increment) % numVerts_perProfile;

								closestId = (eFaces[i].getId()*numVerts_perProfile) + id;
							}

							zPoint newPos = sectionPoints[closestId];
							//newPos += form_eCen - fCenter - (eVec * 0.5 * eLength * param);;
							newPos -= (eVec * 0.5 * eLength * param);;
							positions.push_back(newPos);


						}


						// insert volume centre points
						if (i == 0)
						{
							zPoint vCenter_graphPos = formPositions[forceIndex];

							for (int j = 0; j < tempPosns.size(); j++)
							{
								zVector dir_volCenter_0 = tempPosns[j] - volCenter;
								double len0 = dir_volCenter_0.length();
								dir_volCenter_0.normalize();

								zPoint newPos = volCenter + (dir_volCenter_0 * len0 *offset);
								newPos += (vCenter_graphPos - volCenter);

								positions.push_back(newPos);
							}
						}

						// poly connects
						for (int j = 0; j <= splits; j++)
						{
							polyConnects.push_back(numVerts + lastId + j * 2);
							polyConnects.push_back(numVerts + lastId + 1 + j * 2);
							polyConnects.push_back(numVerts + startId + 1 + j * 2);
							polyConnects.push_back(numVerts + startId + j * 2);

							polyCounts.push_back(4);
						}

						if (i == 0)
						{
							startId += (2 + 2 * splits);
							lastId += (2 + 2 * splits);
						}

					}
				}
			}


		}

		if (subdivs == 0) fnPolytopals[forceIndex].create(positions, polyCounts, polyConnects);
	}

	ZSPACE_INLINE void zTsPolytopal::getPolytopal_Beams_Profile(int user_nEdges, double user_edgeLength, double param)
	{
		zPointArray positions;
		zIntArray polyConnects;
		zIntArray polyCounts;

		for (int i = 0; i < primal_n_f_i; i++)
		{
			int globalFaceId = internalFaceIndex_primalFace[i];

			zItGraphEdge form_e(*formObj, i);

			zPoint form_eCen = form_e.getCenter();

			zVector fNormal = primalFaceNormals[globalFaceId];

			double eLength = form_e.getLength();
			zVector eVec = form_e.getVector();
			eVec.normalize();

			zPointArray sectionPoints;

			// compute section points at each face
			zPointArray tempPoints;
			//getUserSection(fVerts.size(), tempPoints);
			getUserSection(3, tempPoints);

			zTransform t_From;
			t_From.setIdentity();

			zTransform t_to;
			t_to.setIdentity();

			zVector O = form_eCen;

			zVector Y = primalVertexPositions[Y_vertex[globalFaceId]] - O;
			Y.normalize();


			zVector Z = eVec;/*fNormal;*/


			zVector X = Y ^ Z;
			X.normalize();

			t_to(0, 0) = X.x; t_to(1, 0) = X.y; t_to(2, 0) = X.z;
			t_to(0, 1) = Y.x; t_to(1, 1) = Y.y; t_to(2, 1) = Y.z;
			t_to(0, 2) = Z.x; t_to(1, 2) = Z.y; t_to(2, 2) = Z.z;
			t_to(0, 3) = O.x; t_to(1, 3) = O.y; t_to(2, 3) = O.z;

			zTransform t = coreUtils.PlanetoPlane(t_From, t_to);

			for (auto &p : tempPoints)
			{
				p = p * t;
				sectionPoints.push_back(p);
			}


			int numVerts = positions.size();

			int j = 0;
			for (auto &p : sectionPoints)
			{
				zPoint newPos0 = p - (eVec * 0.5 * eLength * param);
				zPoint newPos1 = p + (eVec * 0.5 * eLength * param);

				positions.push_back(newPos0);
				positions.push_back(newPos1);

				polyConnects.push_back(numVerts + 0 + j * 2);
				polyConnects.push_back(numVerts + 1 + j * 2);
				polyConnects.push_back(numVerts + (3 + j * 2) % (sectionPoints.size() * 2));
				polyConnects.push_back(numVerts + (2 + j * 2) % (sectionPoints.size() * 2));

				polyCounts.push_back(4);

				j++;

			}

		}

		fnPolytopals[fnPolytopals.size() - 1].create(positions, polyCounts, polyConnects);

		fnPolytopals[fnPolytopals.size() - 1].setFaceColor(zColor(0.27, 0.15, 0.08, 1));

		//printf("\n%i %i %i %i ", fnPolytopals.size() - 1, fnPolytopals[fnPolytopals.size() - 1].numVertices(), fnPolytopals[fnPolytopals.size() - 1].numEdges(), fnPolytopals[fnPolytopals.size() - 1].numPolygons());

	}

	//--- SET METHODS 

	ZSPACE_INLINE void zTsPolytopal::setVertexOffset(double offset)
	{
		formGraphVertex_Offsets.clear();

		for (int i = 0; i < fnForm.numVertices(); i++) formGraphVertex_Offsets.push_back(offset);
	}

	ZSPACE_INLINE void zTsPolytopal::setVertexOffsets(zDoubleArray &offsets)
	{
		if (offsets.size() != fnForm.numVertices()) throw std::invalid_argument("size of offsets contatiner is not equal to number of graph vertices.");

		formGraphVertex_Offsets = offsets;
	}

	ZSPACE_INLINE void zTsPolytopal::setDualEdgeWeightsfromPrimal(zDomainFloat weightDomain)
	{
		//compute edgeWeights

		float minArea = 100000;
		float maxArea = -100000;

		for (int i = 0; i < primalFaceAreas.size(); i++)
		{
			if (primal_internalFaceIndex[i] == -1) continue;

			if (primalFaceAreas[i] < minArea) minArea = primalFaceAreas[i];
			if (primalFaceAreas[i] > maxArea) maxArea = primalFaceAreas[i];
		}

		zDomainFloat areaDomain(minArea, maxArea);

		zDomainColor colDomain(zColor(0.027, 0, 0.157, 1), zColor(0.784, 0, 0.157, 1));

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{

			if (e.getId() >= internalFaceIndex_primalFace.size()) continue;

			double fArea = primalFaceAreas[internalFaceIndex_primalFace[e.getId()]];

			double wt = coreUtils.ofMap((float) fArea, areaDomain, weightDomain);

			zColor col = coreUtils.blendColor(fArea, areaDomain, colDomain, zRGB);

			e.setWeight(wt);
			e.setColor(col);

			//printf("\n %i %1.2f %1.2f  %1.2f %1.2f ", e.getId(), wt, fArea, areaDomain.min, areaDomain.max);
		}

	}

	//---- UTILITY METHOD

	ZSPACE_INLINE zObjMesh zTsPolytopal::getPolytopals()
	{
		zObjMesh out;

		zPointArray newMesh_verts;
		zIntArray  	newMesh_pCount, newMesh_pConnects;

		for (int j = 0; j < fnForces.size(); j++)
		{
			vector<int> polyConnects;
			vector<int> polyCount;


			fnForces[j].getPolygonData(polyConnects, polyCount);

			int numVerts = newMesh_verts.size();


			//printf(" v: %i , f: %i ", operateZMeshArray[j].numVertices(), operateZMeshArray[j].numPolygons());

			// vertices
			zVector* positions = fnForces[j].getRawVertexPositions();
			for (int i = 0; i < fnForces[j].numVertices(); i++)
			{
				newMesh_verts.push_back(zPoint(positions[j].x, positions[j].y, positions[j].z));
			}

			//pcounts
			for (int i = 0; i < polyCount.size(); i++)
			{
				newMesh_pCount.push_back(polyCount[i]);
			}

			//pconnects
			for (int i = 0; i < polyConnects.size(); i++)
			{
				newMesh_pConnects.push_back(numVerts + polyConnects[i]);
			}

			zFnMesh tempFn(out);
			tempFn.create(newMesh_verts, newMesh_pCount, newMesh_pConnects);

			return out;

		}
	}

	ZSPACE_INLINE void zTsPolytopal::getBisectorPlanes(zVector3DArray &planes)
	{
		planes.clear();

		planes.assign(fnForm.numEdges(), vector<vector<zVector>>());

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			

			zItGraphVertexArray eVerts;
			e.getVertices(eVerts);
		

			for (auto &v : eVerts)
			{
				int volId = v.getId();

				zItGraphHalfEdgeArray cHEdges;
				v.getConnectedHalfEdges(cHEdges);				

				// get force face corresponding to form edge
				int eGlobalFaceId = -1;
				int eLocalFaceId = -1;

				printf("\n %i  f: %i ", volId, fnForces[volId].numPolygons() );

				for (int i = 0; i < fnForces[volId].numPolygons(); i++)
				{
					string hashKey_f = (to_string(volId) + "," + to_string(i));
					std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_f);

					if (gotFace != volumeFace_PrimalFace.end())
					{
						if (gotFace->second == internalFaceIndex_primalFace[e.getId()] )
						{
							eGlobalFaceId = gotFace->second;
							eLocalFaceId = i;
							break;
						}
					}
				}				

				if (eGlobalFaceId != -1)
				{
					zItMeshFace f(*forceObjs[volId], eLocalFaceId);

					zItMeshFaceArray cFaces;
					f.getConnectedFaces(cFaces);			
					
					zVectorArray pNormals;

					zVector fNorm = f.getNormal();
					fNorm.normalize();

					printf("\n Working. %i ", cFaces.size());

					for (auto cF : cFaces)
					{

						string hashKey_f = (to_string(volId) + "," + to_string(cF.getId()));
						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_f);
						
						if (gotFace != volumeFace_PrimalFace.end())
						{
							if (!GFP_SSP_Face[gotFace->second])
							{
								zVector cfNorm = cF.getNormal();
								cfNorm.normalize();

								zVector perp = cfNorm ^ fNorm;

								zVector n = (fNorm + cfNorm)*0.5;
								n.normalize();

								n *= -1;
								pNormals.push_back(n.rotateAboutAxis(perp, 90.0));

								
							}
							
						}						

					}

					planes[e.getId()].push_back(pNormals);
				}
			}
		}

	}

	ZSPACE_INLINE void zTsPolytopal::exportBisectorPlanes(string outfilename)
	{
		zVector3DArray planes;		
		getBisectorPlanes(planes);

		printf("\n Working. %i ", planes.size());

		zInt2DArray edges;
		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			zIntArray eVerts;
			e.getVertices(eVerts);

			edges.push_back(eVerts);
		}

		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<vector<vector<zDoubleArray>>> cutPlanes;

		for (int i = 0; i < planes.size(); i++)
		{			
			vector<vector<zDoubleArray>> ePlanes;

			for (int j = 0; j < planes[i].size(); j++)
			{
				vector<zDoubleArray>pNorms;


				printf(" %i | ", planes[i][j].size());

				for (int k = 0; k < planes[i][j].size(); k++)
				{
					zDoubleArray norm;

					norm.push_back(planes[i][j][k].x);
					norm.push_back(planes[i][j][k].y);
					norm.push_back(planes[i][j][k].z);

					printf(" %1.2f %1.2f %1.2f | ", planes[i][j][k].x, planes[i][j][k].y, planes[i][j][k].z);

					pNorms.push_back(norm);
				}

				ePlanes.push_back(pNorms);
				
			}

			cutPlanes.push_back(ePlanes);
		}

		// output file
		json j;

		// Json file 
		j["Edges"] = edges;	
		j["CutPlanes"] = cutPlanes;

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


	//---- PROTECTED ITERATIVE METHOD UTILITIES

	ZSPACE_INLINE void  zTsPolytopal::computeForcesFaceCenters()
	{
		force_fCenters.clear();

		for (int i = 0; i < fnForces.size(); i++)
		{
			zPointArray fCenters;
			fnForces[i].getCenters(zFaceData, fCenters);

			force_fCenters.push_back(fCenters);
		}
	}

	ZSPACE_INLINE void zTsPolytopal::computeFormTargets()
	{
		targetEdges_form.clear();

		for (int i = 0; i < fnForm.numHalfEdges(); i++)
		{
			targetEdges_form.push_back(zVector());
		}

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			if (e.getId() < internalFaceIndex_primalFace.size())
			{
				zVector t_ij = primalFaceNormals[internalFaceIndex_primalFace[e.getId()]];;
				t_ij.normalize();

				zVector he_0 = e.getHalfEdge(0).getVector();
				if (he_0 * t_ij < 0) targetEdges_form[e.getHalfEdge(0).getId()] = t_ij * -1;
				else targetEdges_form[e.getHalfEdge(0).getId()] = t_ij;

				zVector he_1 = e.getHalfEdge(1).getVector();
				if (he_1 * t_ij < 0) targetEdges_form[e.getHalfEdge(1).getId()] = t_ij * -1;
				else targetEdges_form[e.getHalfEdge(1).getId()] = t_ij;
			}

			else
			{
				zVector he_0 = e.getHalfEdge(0).getVector();
				he_0.normalize();
				targetEdges_form[e.getHalfEdge(0).getId()] = he_0;

				zVector he_1 = e.getHalfEdge(0).getVector();
				he_1.normalize();
				targetEdges_form[e.getHalfEdge(1).getId()] = he_1;
			}

		}

	}

	ZSPACE_INLINE bool zTsPolytopal::checkParallelity(zDomainFloat & deviation, double angleTolerance, bool colorEdges, bool printInfo)
	{
		bool out = true;
		vector<float> deviations;
		deviation = zDomainFloat(10000, -10000);

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			
				//form edge
				int eId_form = e.getHalfEdge(0).getId();
				zVector e_form = e.getHalfEdge(0).getVector();
				e_form.normalize();

				zVector e_target = targetEdges_form[eId_form];

				double a_i = e_form.angle(e_target);

				deviations.push_back(a_i);


				if (a_i > angleTolerance)
				{
					out = false;
				}

				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;
			
			
		}


		if (printInfo)
		{
			printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
		}

		if (colorEdges)
		{
			zDomainColor colDomain(zColor(180, 1, 1), zColor(0, 1, 1));

			for (zItGraphEdge e(*formObj); !e.end(); e++)
			{
				if (e.getId() < internalFaceIndex_primalFace.size())
				{
					zColor col = coreUtils.blendColor(deviations[e.getId()], deviation, colDomain, zHSV);

					if (deviations[e.getId()] < angleTolerance) col = zColor();

					e.setColor(col);
				}
				

			}

		}

		return out;
	}

	ZSPACE_INLINE void zTsPolytopal::updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations )
	{

		zPoint* pos = fnForm.getRawVertexPositions();

		if (fnFormParticles.size() != fnForm.numVertices())
		{
			fnFormParticles.clear();
			formParticlesObj.clear();



			for (zItGraphVertex v(*formObj); !v.end(); v++)
			{
				bool fixed = false;

				int i = v.getId();

				zObjParticle p;
				p.particle = zParticle(pos[i], fixed);
				formParticlesObj.push_back(p);

			}

			for (int i = 0; i < formParticlesObj.size(); i++)
			{
				fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
			}
		}

		if (force_fCenters.size() != fnForces.size()) computeForcesFaceCenters();

		vector<double> edgeLengths;
		fnForm.getEdgeLengths(edgeLengths);

		double minEdgeLength, maxEdgeLength;
		maxEdgeLength = coreUtils.zMax(edgeLengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;



		for (int k = 0; k < numIterations; k++)
		{
			// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
			for (zItGraphVertex v(*formObj); !v.end(); v++)
			{
				int i = v.getId();

				if (fnFormParticles[i].getFixed()) continue;

				// get position of vertex
				zPoint v_i = pos[i];


				// get connected vertices
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				// compute barycenter per vertex
				zPoint b_i;
				for (auto &he : cEdges)
				{
					// get vertex 
					int v1_ID = he.getVertex().getId();

					zPoint v_j = pos[v1_ID];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();;

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_form[he.getId()];;
					t_ij.normalize();
					/*if (e_ij * t_ij < 0)*/ t_ij *= -1;

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				zVector r_i = b_i - v_i;
				zVector forceV = r_i;

				// add forces to particle
				fnFormParticles[i].addForce(forceV);

			}

			// update Particles
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				fnFormParticles[i].integrateForces(dT, type);
				fnFormParticles[i].updateParticle(true);
			}



		}



	}

	//---- ALGEBRAIC METHOD UTILITIES

	ZSPACE_INLINE bool zTsPolytopal::getPrimal_EdgeVertexMatrix(zDiagramType type, zSparseMatrix &out)
	{
		if (type == zForceDiagram)
		{
			if (primal_n_v == 0 || primal_n_e_i == 0) return false;

			vector<bool> edgeVisited;
			edgeVisited.assign(primal_n_e_i, false);

			out = zSparseMatrix(primal_n_e_i, primal_n_v);
			out.setZero();

			vector<zTriplet> coefs; // -1 for from vertex and 1 for to vertex

			for (int j = 0; j < internalEdgeIndex_primalEdge.size(); j++)
			{
				int globalEdgeId = internalEdgeIndex_primalEdge[j];

				int v0 = primalEdge_PrimalVertices[globalEdgeId][0];
				int v1 = primalEdge_PrimalVertices[globalEdgeId][1];

				coefs.push_back(zTriplet(j, v0, -1.0));
				coefs.push_back(zTriplet(j, v1, 1.0));
			}

			out.setFromTriplets(coefs.begin(), coefs.end());

			return true;
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	ZSPACE_INLINE bool zTsPolytopal::getPrimal_EdgeFaceMatrix(zDiagramType type, zSparseMatrix &out)
	{
		if (type == zForceDiagram)
		{
			if (primal_n_f_i == 0 || primal_n_e_i == 0) return false;

			out = zSparseMatrix(primal_n_e_i, primal_n_f_i);
			out.setZero();

			vector<zTriplet> coefs; // -1 for from vertex and 1 for to vertex

			for (int j = 0; j < internalFaceIndex_primalFace.size(); j++)
			{
				int globalFaceId = internalFaceIndex_primalFace[j];

				int volId = primalFace_VolumeFace[globalFaceId][0];
				int faceId = primalFace_VolumeFace[globalFaceId][1];



				zItMeshFace f(*forceObjs[volId], faceId);

				vector<zItMeshHalfEdge> fHEdges;
				f.getHalfEdges(fHEdges);

				for (auto &he : fHEdges)
				{
					int edgeId = he.getEdge().getId();

					int v0 = he.getStartVertex().getId();
					int v1 = he.getVertex().getId();


					string hashKey_v0 = (to_string(volId) + "," + to_string(v0));
					std::unordered_map<std::string, int>::const_iterator gotVertex0 = volumeVertex_PrimalVertex.find(hashKey_v0);

					string hashKey_v1 = (to_string(volId) + "," + to_string(v1));
					std::unordered_map<std::string, int>::const_iterator gotVertex1 = volumeVertex_PrimalVertex.find(hashKey_v1);

					string hashKey_e = (to_string(volId) + "," + to_string(edgeId));
					std::unordered_map<std::string, int>::const_iterator gotEdge = volumeEdge_PrimalEdge.find(hashKey_e);


					if (gotVertex0 != volumeVertex_PrimalVertex.end() && gotVertex1 != volumeVertex_PrimalVertex.end() && gotEdge != volumeEdge_PrimalEdge.end())
					{
						int p_v0 = gotVertex0->second;
						int p_v1 = gotVertex1->second;

						int p_e = gotEdge->second;

						if (GFP_SSP_Edge[p_e]) continue;

						if (p_v0 > p_v1)
						{

							coefs.push_back(zTriplet(primal_internalEdgeIndex[p_e], j, -1.0));
						}
						else
						{
							coefs.push_back(zTriplet(primal_internalEdgeIndex[p_e], j, 1.0));
						}
					}

				}

			}

			out.setFromTriplets(coefs.begin(), coefs.end());

			return true;
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	ZSPACE_INLINE bool zTsPolytopal::getPrimal_FaceCellMatrix(zDiagramType type, zSparseMatrix &out)
	{
		if (type == zForceDiagram)
		{
			if (primal_n_f_i == 0) return false;

			int primal_n_vol = fnForces.size();

			out = zSparseMatrix(primal_n_f_i, primal_n_vol);
			out.setZero();

			vector<zTriplet> coefs;

			for (int j = 0; j < fnForces.size(); j++)
			{
				int volId = j;


				for (zItMeshFace f(*forceObjs[volId]); !f.end(); f++)
				{
					int faceId = f.getId();

					string hashKey_f = (to_string(volId) + "," + to_string(faceId));
					std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_f);

					if (gotFace != volumeFace_PrimalFace.end())
					{
						int p_f = gotFace->second;

						if (GFP_SSP_Face[p_f]) continue;

						int p_volId = primalFace_VolumeFace[p_f][0];
						int p_faceId = primalFace_VolumeFace[p_f][1];

						if (volId == p_volId && faceId == p_faceId && !GFP_SSP_Face[p_f])
						{
							coefs.push_back(zTriplet(primal_internalFaceIndex[p_f], volId, 1.0));
						}
						else coefs.push_back(zTriplet(primal_internalFaceIndex[p_f], volId, -1.0));
					}

				}

			}

			out.setFromTriplets(coefs.begin(), coefs.end());

			return true;
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	ZSPACE_INLINE bool zTsPolytopal::get_EquilibriumMatrix(zDiagramType type, MatrixXd &out)
	{
		double factor = pow(10, 3);

		// Get Normal diagonal matricies
		VectorXd nx(primal_n_f_i);
		VectorXd ny(primal_n_f_i);
		VectorXd nz(primal_n_f_i);

		for (int j = 0; j < primalFaceNormals.size(); j++)
		{
			if (GFP_SSP_Face[j]) continue;

			zVector n = primalFaceNormals[j];

			int id = primal_internalFaceIndex[j];

			nx[id] = n.x;
			ny[id] = n.y;
			nz[id] = n.z;


		}

		zDiagonalMatrix Nx = nx.asDiagonal();
		zDiagonalMatrix Ny = ny.asDiagonal();
		zDiagonalMatrix Nz = nz.asDiagonal();


		// primal _ edgefaceMatrix
		zSparseMatrix C_ef;
		getPrimal_EdgeFaceMatrix(zForceDiagram, C_ef);

		//cout << "\n C_ef " <<  endl  << C_ef << endl;

		// compute A

		out = MatrixXd(3 * primal_n_e_i, primal_n_f_i);



		for (int i = 0; i < 3; i++)
		{
			MatrixXd temp(primal_n_e_i, primal_n_f_i);

			if (i == 0) temp = C_ef * Nx;
			if (i == 1) temp = C_ef * Ny;
			if (i == 2) temp = C_ef * Nz;

			for (int j = 0; j < primal_n_e_i; j++)
			{
				for (int k = 0; k < primal_n_f_i; k++)
				{
					out(i*primal_n_e_i + j, k) = temp(j, k);

					//out(i*primal_n_e_i + j, k) = round(out(i*primal_n_e_i + j, k) *factor) / factor;

					//if (out(i*primal_n_e_i + j, k) >0 &&  out(i*primal_n_e_i + j, k) < 0.000001) out(i*primal_n_e_i + j, k) = 0;
					//if (out(i*primal_n_e_i + j, k) < 0 && out(i*primal_n_e_i + j, k) > -0.000001) out(i*primal_n_e_i + j, k) = 0;
				}
			}
		}

		printf("\n A : %i  %i ", out.rows(), out.cols());

		return true;
	}

	ZSPACE_INLINE void zTsPolytopal::getDual_ForceDensities_LPA(VectorXd &q)
	{
		MatrixXd A;
		get_EquilibriumMatrix(zForceDiagram, A);
		//cout << "\n A " << endl << A << endl;

		string outfilename1 = "C:/Users/vishu.b/desktop/A.csv";
		ofstream myfile1;
		myfile1.open(outfilename1.c_str());

		for (int i = 0; i < A.rows(); i++)
		{
			myfile1 << "\n";
			for (int j = 0; j < A.cols(); j++)
				myfile1 << A(i, j) << ",";
		}
		myfile1.close();

		// Q & s
		real_1d_array Q;
		real_1d_array s;

		Q.setlength(A.cols());
		s.setlength(A.cols());

		internal_fAreas.clear();
		for (int i = 0; i < Q.length(); i++) internal_fAreas.push_back(primalFaceAreas[internalFaceIndex_primalFace[i]]);

		for (int i = 0; i < A.cols(); i++)
		{
			Q(i) = 1.0;
			s(i) = 1;
		}

		real_2d_array c;
		c.setlength(A.rows(), A.cols() + 1);

		integer_1d_array ct;
		ct.setlength(A.rows());

		//AQ =0
		for (int i = 0; i < A.rows(); i++)
		{
			// coefficients
			for (int j = 0; j < A.cols(); j++)
			{
				c(i, j) = A(i, j);
			}

			// right part
			c(i, A.cols()) = 0.0;

			ct(i) = 1;
		}

		// Q >=1
		real_1d_array bndl;
		bndl.setlength(A.cols());

		real_1d_array bndu;
		bndu.setlength(A.cols());

		for (int i = 0; i < A.cols(); i++)
		{
			bndl(i) = 0.1;
			bndu(i) = 3.0;
		}

		//printf("\n %s\n", Q.tostring(2).c_str());
		//printf("\n C \n %s\n", c.tostring(2).c_str());
		//printf("\n CT \n %s\n", ct.tostring().c_str());

		minbleicstate state;
		double epsg = 0.000001;
		double epsf = 0;
		double epsx = 0.000001;
		ae_int_t maxits = 0;
		double diffstep = 1.0e-6;

		minbleiccreatef(Q, diffstep, state);
		minbleicsetbc(state, bndl, bndu);
		minbleicsetlc(state, c, ct);
		minbleicsetscale(state, s);
		minbleicsetcond(state, epsg, epsf, epsx, maxits);

		minbleicoptguardsmoothness(state);
		minbleicoptguardgradient(state, 0.001);

		minbleicreport rep;
		alglib::minbleicoptimize(state, loadPath_func);
		minbleicresults(state, Q, rep);
		printf("\nreultType:  %d\n", int(rep.terminationtype)); // EXPECTED: 4
		//printf("\n Q:  %s\n", Q.tostring(2).c_str()); // EXPECTED: [-1,1]

		optguardreport ogrep;
		minbleicoptguardresults(state, ogrep);
		printf("%s\n", ogrep.nonc0suspected ? "true" : "false"); // EXPECTED: false
		printf("%s\n", ogrep.nonc1suspected ? "true" : "false"); // EXPECTED: false

		q = VectorXd(A.cols());

		for (int i = 0; i < Q.length(); i++) q(i) = Q(i);
		cout << "\n AQ norm: " << endl << (A * q).norm() << endl;
	}

#ifndef USING_CLR
	
	ZSPACE_INLINE void zTsPolytopal::getDual_ForceDensities_MPI(VectorXd &q, double threshold)
	{

		MatrixXd A;
		get_EquilibriumMatrix(zForceDiagram, A);
		//cout << "\n A " << endl << A << endl;	


		mat A_arma = coreUtils.eigenToArma(A);;
		mat A_arma_Inv = arma::pinv(A_arma, threshold);

		cout << "\n Rank : " << arma::rank(A_arma, threshold) << endl;

		mat I_Arma = eye(A.cols(), A.cols());
		arma::vec Xi_Arma = ones(A.cols());

		arma::vec q_arma = (I_Arma - (A_arma_Inv *A_arma)) * Xi_Arma;
		q = coreUtils.armaToEigen(q_arma);

		cout << "\n AQ norm: " << endl << (A * q).norm() << endl;
	}

	ZSPACE_INLINE void zTsPolytopal::getDual_ForceDensities_RREF(VectorXd &q, double threshold )
	{

		MatrixXd A;
		get_EquilibriumMatrix(zForceDiagram, A);
		//cout << "\n A " << endl << A << endl;	


		mat A_arma = coreUtils.eigenToArma(A);;

		int rank;
		mat B;
		vector<bool> independentEdges;
		get_BMatrix(A_arma, threshold, rank, B, independentEdges);

		int numCols = B.n_cols;
		int numRows = B.n_rows;

		arma::vec zeta(numCols);
		for (int i = 0; i < numCols; i++)
		{
			if (i == 1) zeta(i) = 1.0;
			else zeta(i) = 1.0;
		}

		arma::vec qr = (B * zeta);
		//cout << "\n qr: " << endl << qr << endl;

		arma::vec q_arma(A_arma.n_cols);

		int zetaCounter = 0;
		int qrCounter = 0;;
		for (int i = 0; i < independentEdges.size(); i++)
		{

			if (independentEdges[i])
			{
				q_arma(i) = zeta(zetaCounter);
				zetaCounter++;
			}
			else
			{
				q_arma(i) = qr(qrCounter);
				qrCounter++;
			}
		}


		q = coreUtils.armaToEigen(q_arma);
		cout << "\n AQ norm: " << endl << (A * q).norm() << endl;


	}

	ZSPACE_INLINE void zTsPolytopal::get_BMatrix(mat &A, double threshold, int &rank, mat &B, zBoolArray &indEdges)
	{
		mat r = coreUtils.rref(A, threshold);


		indEdges.clear();
		indEdges.assign(r.n_cols, true);

		rank = 0;
		for (int i = 0; i < r.n_cols; i++)
		{
			for (int j = 0; j < r.n_cols; j++)
			{
				if (r(i, j) == 1.0)
				{
					indEdges[j] = false;
					rank++;
					break;
				}
			}

		}

		printf("\n rank : %i ", rank);

		B = mat(rank, A.n_cols - rank);
		int colCounter = 0;

		cout << "\n independent edges : ";
		for (int i = 0; i < indEdges.size(); i++)
		{
			if (indEdges[i])
			{
				printf(" %i ", i);

				for (int j = 0; j < rank; j++)
				{
					// Note B stores 					
					B(j, colCounter) = r(j, i) * -1.0;
				}

				colCounter++;

			}
		}
		cout << "\n ";
		//cout << "\n B: " << endl << B << endl;				   			 		  



	}

#endif

	ZSPACE_INLINE double zTsPolytopal::getDual_Loadpath(VectorXd &q)
	{
		VectorXd fAreas(primal_n_f_i);

		for (int i = 0; i < primal_n_f_i; i++) fAreas(i) = primalFaceAreas[internalFaceIndex_primalFace[i]];

		VectorXd phi = q.transpose().cwiseAbs() * fAreas;

		return phi(0);
	}

	//---- POLYTOPAL METHOD UTILITIES

	ZSPACE_INLINE void zTsPolytopal::getPolytopal(int forceIndex, int subdivs)
	{
		if (forceIndex > fnForces.size()) throw std::invalid_argument(" error: index out of bounds.");

		zPointArray positions;
		zIntArray polyConnects;
		zIntArray polyCounts;

		int n_v = fnForces[forceIndex].numVertices();
		int n_e = fnForces[forceIndex].numHalfEdges();
		int n_f = fnForces[forceIndex].numPolygons();



		zPoint volCenter = fnForces[forceIndex].getCenter();


		zPointArray fCenters;
		fnForces[forceIndex].getCenters(zFaceData, fCenters);

		for (zItMeshEdge e(*forceObjs[forceIndex]); !e.end(); e++)
		{
			zIntArray  eFaces;
			e.getFaces(eFaces);


			zPoint pos0 = e.getHalfEdge(1).getVertex().getPosition();
			zPoint pos1 = e.getHalfEdge(0).getVertex().getPosition();

			zPoint* formPositions = fnForm.getRawVertexPositions();

			if (eFaces.size() == 2)
			{
				int numVerts = positions.size();

				for (int j = 0; j < eFaces.size(); j++)
				{


					string hashKey_f = (to_string(forceIndex) + "," + to_string(eFaces[j]));
					int vId_fCenter = -1;
					bool chkExists_f = coreUtils.existsInMap(hashKey_f, forceVolumeFace_formGraphVertex, vId_fCenter);
					double boundaryOffset = formGraphVertex_Offsets[vId_fCenter];

					zPoint fCenter = fCenters[eFaces[j]];
					zPoint fCenter_graphPos = formPositions[vId_fCenter];

					zVector dir_fCenter_0 = pos0 - fCenter;
					double len0 = dir_fCenter_0.length();
					dir_fCenter_0.normalize();

					zPoint newPos = fCenter + (dir_fCenter_0 * len0 *boundaryOffset);
					newPos += fCenter_graphPos - fCenter;

					positions.push_back(newPos);


					zVector dir_fCenter_1 = pos1 - fCenter;
					double len1 = dir_fCenter_1.length();
					dir_fCenter_1.normalize();

					zPoint newPos1 = fCenter + (dir_fCenter_1 * len1 *boundaryOffset);
					newPos1 += fCenter_graphPos - fCenter;

					positions.push_back(newPos1);
				}

				zVector dir_volCenter_0 = pos0 - volCenter;
				double len0 = dir_volCenter_0.length();
				dir_volCenter_0.normalize();

				string hashKey_v = (to_string(forceIndex) + "," + to_string(-1));
				int vId_vCenter = -1;
				bool chkExists_v = coreUtils.existsInMap(hashKey_v, forceVolumeFace_formGraphVertex, vId_vCenter);

				double centerOffset = formGraphVertex_Offsets[vId_vCenter];
				zPoint vCenter_graphPos = formPositions[vId_vCenter];

				zPoint newPos = volCenter + (dir_volCenter_0 * len0 *centerOffset);
				newPos += (vCenter_graphPos - volCenter);

				positions.push_back(newPos);


				zVector dir_volCenter_1 = pos1 - volCenter;
				double len1 = dir_volCenter_1.length();
				dir_volCenter_1.normalize();

				zPoint newPos1 = volCenter + (dir_volCenter_1 * len1 *centerOffset);
				newPos1 += (vCenter_graphPos - volCenter);

				positions.push_back(newPos1);

				polyConnects.push_back(numVerts);
				polyConnects.push_back(numVerts + 4);
				polyConnects.push_back(numVerts + 5);
				polyConnects.push_back(numVerts + 1);
				polyCounts.push_back(4);

				polyConnects.push_back(numVerts + 5);
				polyConnects.push_back(numVerts + 4);
				polyConnects.push_back(numVerts + 2);
				polyConnects.push_back(numVerts + 3);
				polyCounts.push_back(4);

			}
		}

		if (subdivs == 0) fnPolytopals[forceIndex].create(positions, polyCounts, polyConnects);
		else
		{
			zObjMesh tempObj;
			zFnMesh tempFn(tempObj);

			tempFn.create(positions, polyCounts, polyConnects);
			getPolytopalRulingRemesh(forceIndex, tempObj, subdivs);
		}



	}

	ZSPACE_INLINE void zTsPolytopal::getPolytopalRulingRemesh(int index, zObjMesh &inMeshObj, int SUBDIVS)
	{
		zPointArray positions;
		zIntArray polyConnects;
		zIntArray polyCounts;

		zFnMesh inFnMesh(inMeshObj);
		int n_v_lowPoly = inFnMesh.numVertices();

		inFnMesh.smoothMesh(SUBDIVS);

		int n_v = inFnMesh.numVertices();
		int n_e = inFnMesh.numHalfEdges();
		int n_f = inFnMesh.numPolygons();

		for (int i = 0; i < n_v_lowPoly; i += 6)
		{
			zItMeshVertex vert0(inMeshObj, i);
			zItMeshVertex vert1(inMeshObj, i + 1);
			zItMeshHalfEdge edge0, edge1;

			vector<zItMeshHalfEdge> cEdges0;
			vert0.getConnectedHalfEdges(cEdges0);

			for (auto &he : cEdges0)
			{
				if (!he.onBoundary())
				{
					edge0 = he.getSym();
				}
			}

			vector<zItMeshHalfEdge> cEdges1;
			vert1.getConnectedHalfEdges(cEdges1);

			for (auto &he : cEdges1)
			{
				if (he.onBoundary())
				{
					edge1 = he;
				}
			}

			zPoint v0 = vert0.getPosition();

			zPoint v1 = vert1.getPosition();

			positions.push_back(v0);
			positions.push_back(v1);

			//while (smoothPolytopalMesh.edges[edge0].getVertex()->getVertexId() != i + 2)
			for (int k = 0; k < pow(2, (SUBDIVS + 1)); k++)
			{
				int numVerts = positions.size();

				zPoint v2 = edge0.getStartVertex().getPosition(); ;

				zPoint v3 = edge1.getVertex().getPosition();

				positions.push_back(v2);
				positions.push_back(v3);

				polyConnects.push_back(numVerts - 2);
				polyConnects.push_back(numVerts);
				polyConnects.push_back(numVerts + 1);
				polyConnects.push_back(numVerts - 1);
				polyCounts.push_back(4);

				//vert0 = vert2;
				//vert1 = vert3;

				edge0 = edge0.getPrev();
				edge1 = edge1.getNext();
			}
		}

		fnPolytopals[index].create(positions, polyCounts, polyConnects);
	}

	ZSPACE_INLINE bool zTsPolytopal::computeRulingIntersection(int polytopalIndex, zItMeshVertex &v0, zItMeshVertex &v1, zPoint &closestPt)
	{
		bool out = false;

		zItMeshHalfEdge e0;
		zItMeshHalfEdge e1;

		bool e0HasEdge = false;
		bool e1HasEdge = false;

		vector<zItMeshHalfEdge> cEdges0;
		v0.getConnectedHalfEdges(cEdges0);

		if (cEdges0.size() == 3)
		{
			for (auto &he : cEdges0)
			{
				if (!he.onBoundary())
				{
					e0 = he;
					e0HasEdge = true;
					break;
				}
			}
		}

		vector<zItMeshHalfEdge> cEdges1;
		v1.getConnectedHalfEdges(cEdges1);
		if (cEdges1.size() == 3)
		{
			for (auto &he : cEdges1)
			{
				if (!he.onBoundary())
				{
					e1 = he;
					e1HasEdge = true;
					break;
				}
			}
		}

		if (e0HasEdge  && e1HasEdge)
		{
			zItMeshVertex v2 = v0;
			(v0.getId() % 2 == 0) ? v2++ : v2--;

			zItMeshVertex v3 = v1;
			(v1.getId() % 2 == 0) ? v3++ : v3--;



			zPoint a0 = v2.getPosition();

			zPoint a1 = v0.getPosition();

			zPoint b0 = v3.getPosition();

			zPoint b1 = v1.getPosition();

			double uA = -1;
			double uB = -1;
			out = coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

			if (out)
			{
				if (uA >= uB)
				{
					zVector dir = a1 - a0;
					double len = dir.length();
					dir.normalize();

					if (uA < 0) dir *= -1;
					closestPt = a0 + dir * len * uA;
				}
				else
				{
					zVector dir = b1 - b0;
					double len = dir.length();
					dir.normalize();

					if (uB < 0) dir *= -1;

					closestPt = b0 + dir * len * uB;
				}


			}

		}
		return out;
	}

	ZSPACE_INLINE void zTsPolytopal::getClosePolytopalMesh(int forceIndex)
	{
		if (smoothSubDivs == 0) return;

		int n_v = fnForces[forceIndex].numVertices();
		int n_e = fnForces[forceIndex].numHalfEdges();
		int n_f = fnForces[forceIndex].numPolygons();

		int n_v_smooth = fnPolytopals[forceIndex].numVertices();
		int n_e_smooth = fnPolytopals[forceIndex].numHalfEdges();
		int n_f_smooth = fnPolytopals[forceIndex].numPolygons();

		int numVertsPerStrip = floor(n_v_smooth / (0.5 * n_e));
		int half_NumVertsPerStrip = floor(numVertsPerStrip / 2);


		vector<bool> vertVisited;

		for (int i = 0; i < n_v_smooth; i++)
		{
			vertVisited.push_back(false);
		}

		for (zItMeshEdge e(*forceObjs[forceIndex]); !e.end(); e++)
		{
			int eStripId = e.getId();


			//-- Prev  Edge	

			int ePrev = e.getHalfEdge(0).getPrev().getId(); ;
			int ePrevStripId = floor(ePrev / 2);


			if (ePrev % 2 == 0)
			{
				for (int j = 1, k = 0; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip - 2; j += 2, k += 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], ePrevStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}

						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}



				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = 0; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip - 2; j -= 2, k += 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], ePrevStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}
				}
			}

			//-- Next Edge		

			int eNext = e.getHalfEdge(0).getNext().getId(); ;
			int eNextStripId = floor(eNext / 2);

			if (eNext % 2 == 0)
			{
				for (int j = 0, k = 1; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip; j += 2, k += 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eNextStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zVector cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}

				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = 1; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip; j -= 2, k += 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eNextStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}

				}
			}


			//-- SYM Prev  Edge	


			int eSymPrev = e.getHalfEdge(1).getPrev().getId();
			int eSymPrevStripId = floor(eSymPrev / 2);


			if (eSymPrev % 2 == 0)
			{
				for (int j = 1, k = numVertsPerStrip - 1; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip; j += 2, k -= 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eSymPrevStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}
				}
			}
			else
			{
				for (int j = numVertsPerStrip - 2, k = numVertsPerStrip - 1; j > half_NumVertsPerStrip - 1, k > half_NumVertsPerStrip; j -= 2, k -= 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eSymPrevStripId * numVertsPerStrip + j);

					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}
				}
			}


			//--SYM Next Edge		
			int eSymNext = e.getHalfEdge(1).getNext().getId();
			int eSymNextStripId = floor(eSymNext / 2);

			if (eSymNext % 2 == 0)
			{
				for (int j = 0, k = numVertsPerStrip - 2; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip - 1; j += 2, k -= 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eSymNextStripId * numVertsPerStrip + j);


					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}
				}
			}
			else
			{
				for (int j = numVertsPerStrip - 1, k = numVertsPerStrip - 2; j > half_NumVertsPerStrip, k > half_NumVertsPerStrip - 1; j -= 2, k -= 2)
				{
					zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
					zItMeshVertex v1(*forceObjs[forceIndex], eSymNextStripId * numVertsPerStrip + j);


					if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
					{
						zPoint cPt;
						bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

						if (intersectChk)
						{
							v0.setPosition(cPt);
							v1.setPosition(cPt);

						}
						vertVisited[v0.getId()] = true;
						vertVisited[v1.getId()] = true;
					}

				}
			}
		}

		for (zItMeshVertex v(*forceObjs[forceIndex]); !v.end(); v++)
		{
			zIntArray  cEdges;
			v.getConnectedHalfEdges(cEdges);

			zIntArray  smoothMeshVerts;
			zPointArray intersectPoints;

			// get connected edge strips
			for (int j = 0; j < cEdges.size(); j++)
			{
				int eStripId = floor(cEdges[j] / 2);
				int vertId = eStripId * numVertsPerStrip + half_NumVertsPerStrip;

				if (cEdges[j] % 2 == 0) vertId -= 1;
				smoothMeshVerts.push_back(vertId);
			}

			// comput smooth mesh vertices
			for (int j = 0; j < smoothMeshVerts.size(); j++)
			{
				zItMeshVertex v0(*forceObjs[forceIndex], smoothMeshVerts[j]);
				zItMeshVertex v1(*forceObjs[forceIndex], smoothMeshVerts[(j + 1) % smoothMeshVerts.size()]);

				vertVisited[v0.getId()] = true;

				zPoint cPt;
				bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

				if (intersectChk)
				{
					intersectPoints.push_back(cPt);
				}
			}

			// get average intersection point
			zPoint avgIntersectPoint;

			for (int j = 0; j < intersectPoints.size(); j++)
			{
				avgIntersectPoint += intersectPoints[j];
			}

			avgIntersectPoint = avgIntersectPoint / intersectPoints.size();

			//// update positions
			for (int j = 0; j < smoothMeshVerts.size(); j++)
			{
				zItMeshVertex v0(*forceObjs[forceIndex], smoothMeshVerts[j]);
				v0.setPosition(avgIntersectPoint);
			}

		}


	}

	ZSPACE_INLINE void zTsPolytopal::get_YVertex(zIntArray &Y_vertex)
	{
		Y_vertex.clear();
		Y_vertex.assign(primal_n_f, -1);

		for (int j = 0; j < primal_n_f; j++)
		{
			if (Y_vertex[j] != -1) continue;
			if (GFP_SSP_Face[j])
			{
				continue;
			}


			int volId = primalFace_VolumeFace[j][0];
			int faceId = primalFace_VolumeFace[j][1];

			zItMeshFace f(*forceObjs[volId], faceId);
			zIntArray  fVerts;
			f.getVertices(fVerts);

			int v = -1;
			int num_cFaces = 0;
			int temp_V = -1;

			for (auto vId : fVerts)
			{
				string hashKey_v = (to_string(volId) + "," + to_string(vId));
				std::unordered_map<std::string, int>::const_iterator got_Vertex = volumeVertex_PrimalVertex.find(hashKey_v);

				if (!GFP_SSP_Vertex[got_Vertex->second])
				{
					v = vId;
					break;
				}
				else
				{
					if (primalVertex_ConnectedPrimalFaces[got_Vertex->second].size() > num_cFaces)
					{
						num_cFaces = primalVertex_ConnectedPrimalFaces[got_Vertex->second].size();
						temp_V = vId;
					}

				}
			}


			if (v == -1 && temp_V != -1)
			{
				v = temp_V;
			}


			if (v == -1) continue;


			string hashKey_v = (to_string(volId) + "," + to_string(v));
			std::unordered_map<std::string, int>::const_iterator got_Vertex = volumeVertex_PrimalVertex.find(hashKey_v);

			if (got_Vertex != volumeVertex_PrimalVertex.end())
			{
				int globalVertexId = got_Vertex->second;

				for (int k = 0; k < primalVertex_ConnectedPrimalFaces[globalVertexId].size(); k++)
				{
					int globalfaceId = primalVertex_ConnectedPrimalFaces[globalVertexId][k];

					if (Y_vertex[globalfaceId] == -1 && !GFP_SSP_Face[globalfaceId]) Y_vertex[globalfaceId] = globalVertexId;
				}
			}


		}

		//for (auto &YVert : Y_vertex)printf("\n %i %i ", &YVert - &Y_vertex[0], YVert);

	}

	ZSPACE_INLINE void zTsPolytopal::getUserSection(int face_numEdges, zPointArray &sectionPoints)
	{
		sectionPoints.clear();

		int numPoints = face_numEdges;
		while (numPoints < userSection_numEdge) numPoints += face_numEdges;

		// tri faces
		if (face_numEdges == 3)
		{
			double halfElen = userSection_edgeLength / 2;

			// square profile
			if (userSection_numEdge == 4)
			{
				sectionPoints.push_back(zPoint(0, halfElen, 0));
				sectionPoints.push_back(zPoint(-halfElen, halfElen, 0));
				sectionPoints.push_back(zPoint(-halfElen, -halfElen, 0));
				sectionPoints.push_back(zPoint(0, -halfElen, 0));
				sectionPoints.push_back(zPoint(halfElen, -halfElen, 0));
				sectionPoints.push_back(zPoint(halfElen, halfElen, 0));
			}

			// hexagon or octagon profile
			if (userSection_numEdge == 6 || userSection_numEdge == 8 || userSection_numEdge == 9)
			{


				double theta = 0;

				for (int i = 0; i < numPoints; i++)
				{
					zPoint pos;
					pos.x = (halfElen * cos(theta + HALF_PI));
					pos.y = (halfElen * sin(theta + HALF_PI));
					pos.z = 0;

					sectionPoints.push_back(pos);

					theta += (TWO_PI / numPoints);
				}
			}

			// pentagon profile
			if (userSection_numEdge == 5)
			{

				double theta = 0;

				for (int i = 0; i < numPoints; i++)
				{
					zPoint pos;
					pos.x = (halfElen * cos(theta + HALF_PI));
					pos.y = (halfElen * sin(theta + HALF_PI));
					pos.z = 0;

					if (i == 3)
					{
						zPoint pos1 = (sectionPoints[i - 1] + pos) * 0.5;
						sectionPoints.push_back(pos1);
					}

					sectionPoints.push_back(pos);

					theta += (TWO_PI / numPoints);
				}
			}

		}

	}
	   	 
}