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


#include<headers/zToolsets/statics/zTsSpatialStructure.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSpatialStructures::zTsSpatialStructures() {}

	ZSPACE_INLINE zTsSpatialStructures::zTsSpatialStructures(zObjGraph &_formObj, vector<zObjMesh> &_volumeObjs, vector<zObjMesh>  &_polytopalObjs)
	{
		formObj = &_formObj;
		fnForm = zFnGraph(_formObj);

		for (int i = 0; i < _volumeObjs.size(); i++)
		{
			volumeObjs.push_back(&_volumeObjs[i]);
			fnVolumes.push_back(_volumeObjs[i]);

			polytopalObjs.push_back(&_polytopalObjs[i]);
			fnPolytopals.push_back(_polytopalObjs[i]);

		}

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSpatialStructures::~zTsSpatialStructures() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsSpatialStructures::createVolumeFromFile(string directory, string filename, int numFiles, zFileTpye type)
	{

		if (type == zJSON)
		{

			for (int i = 0; i < numFiles; i++)
			{
				string path = directory + "/" + filename + "_" + to_string(i) + ".json";

				if (i < fnVolumes.size()) fnVolumes[i].from(path, type);
			}



		}

		else if (type == zOBJ)
		{
			for (int i = 0; i < numFiles; i++)
			{
				string path = directory + "/" + filename + "_" + to_string(i) + ".obj";

				if (i < fnVolumes.size()) fnVolumes[i].from(path, type);
			}
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");

	}

	ZSPACE_INLINE void zTsSpatialStructures::createFormFromVolume(double offset, int precisionFac , zColor edgeCol )
	{

		vector<zVector>positions;
		vector<int>edgeConnects;
		unordered_map <string, int> positionVertex;


		volumeFace_formGraphVertex.clear();

		formGraphVertex_volumeMesh.clear();

		formGraphVertex_volumeFace.clear();



		for (int j = 0; j < fnVolumes.size(); j++)
		{

			zColor* faceColors = fnVolumes[j].getRawFaceColors();

			int n_v = fnVolumes[j].numVertices();
			int n_e = fnVolumes[j].numHalfEdges();
			int n_f = fnVolumes[j].numPolygons();

			zVector volCenter = fnVolumes[j].getCenter();

			int vId_volCenter = -1;
			bool chkExists = coreUtils.vertexExists(positionVertex, volCenter, precisionFac, vId_volCenter);


			if (!chkExists)
			{
				coreUtils.addToPositionMap(positionVertex, volCenter, positions.size(), precisionFac);

				vId_volCenter = positions.size();
				positions.push_back(volCenter);

				formGraphVertex_volumeFace.push_back(-1);
				formGraphVertex_volumeFace.push_back(-1);

				formGraphVertex_volumeMesh.push_back(j);
				formGraphVertex_volumeMesh.push_back(-1);
			}

			string hashKey_volface = (to_string(j) + "," + to_string(-1));
			volumeFace_formGraphVertex[hashKey_volface] = vId_volCenter;

			vector<zVector> fCenters;
			fnVolumes[j].getCenters(zFaceData, fCenters);

			for (int i = 0; i < fCenters.size(); i++)
			{
				if (faceColors[i].r != 1.0)
				{
					int vId_fCenter = -2;
					string hashKey_volface = (to_string(j) + "," + to_string(i));
					volumeFace_formGraphVertex[hashKey_volface] = vId_fCenter;

				}

				else
				{
					int vId_fCenter = -1;
					bool chkExists_f = coreUtils.vertexExists(positionVertex, fCenters[i], precisionFac, vId_fCenter);

					if (!chkExists_f)
					{
						coreUtils.addToPositionMap(positionVertex, fCenters[i], positions.size(), precisionFac);

						vId_fCenter = positions.size();
						positions.push_back(fCenters[i]);

						formGraphVertex_volumeFace.push_back(i);
						formGraphVertex_volumeFace.push_back(-1);

						formGraphVertex_volumeMesh.push_back(j);
						formGraphVertex_volumeMesh.push_back(-2);
					}
					else
					{
						formGraphVertex_volumeFace[(vId_fCenter * 2) + 1] = i;
						formGraphVertex_volumeMesh[(vId_fCenter * 2) + 1] = j;
					}

					string hashKey_volface = (to_string(j) + "," + to_string(i));
					volumeFace_formGraphVertex[hashKey_volface] = vId_fCenter;

					edgeConnects.push_back(vId_volCenter);
					edgeConnects.push_back(vId_fCenter);
				}


			}
		}

		fnForm.clear();

		fnForm.create(positions, edgeConnects);

		setVertexOffset(offset);
		fnForm.setEdgeColor(edgeCol);


		// compute intersection point
		for (zItGraphVertex v(*formObj); !v.end(); v++)
		{
			int i = v.getId();

			vector<zItGraphVertex> cVerts;
			v.getConnectedVertices(cVerts);


			int volId_V = formGraphVertex_volumeMesh[i * 2];
			int faceId_V = formGraphVertex_volumeFace[i * 2];



			if (cVerts.size() == 2)
			{
				zItMeshFace volume_face(*volumeObjs[volId_V], faceId_V);

				zVector normF1 = volume_face.getNormal();
				zVector currentPos = v.getPosition();

				zVector p1 = cVerts[0].getPosition();
				zVector p2 = cVerts[1].getPosition();

				zVector interPt;
				bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

				if (chkIntersection)
				{
					v.setPosition(interPt);

					double distTOp1 = interPt.distanceTo(p1);
					double distTOp2 = interPt.distanceTo(p2);
					double distp12 = p1.distanceTo(p2);

					double wt1 = distTOp1 / distp12;
					double wt2 = distTOp2 / distp12;

					formGraphVertex_Offsets[i] = (formGraphVertex_Offsets[cVerts[0].getId()] * wt1) + (formGraphVertex_Offsets[cVerts[1].getId()] * wt2);
				}
			}

		}


	}

	ZSPACE_INLINE void zTsSpatialStructures::createPolytopalsFromVolume()
	{

		for (int i = 0; i < fnVolumes.size(); i++)
		{
			fnPolytopals[i].clear();
			getPolytopal(i);
		}
	}

	//----UPDATE METHOD

	ZSPACE_INLINE bool zTsSpatialStructures::equilibrium(bool computeTargets, double minmax_Edge, double dT, zIntergrationType type, int numIterations, double angleTolerance, bool colorEdges, bool printInfo)
	{

		if (computeTargets)
		{
			computeFormTargets();

			computeTargets = !computeTargets;
		}

		updateFormDiagram(minmax_Edge, dT, type, numIterations);

		// check deviations
		zDomainFloat dev;
		bool out = checkParallelity(dev, angleTolerance, colorEdges, printInfo);

		return out;

	}

	//--- SET METHODS 

	ZSPACE_INLINE void zTsSpatialStructures::setVertexOffset(double offset)
	{
		formGraphVertex_Offsets.clear();

		for (int i = 0; i < fnForm.numVertices(); i++) formGraphVertex_Offsets.push_back(offset);
	}

	ZSPACE_INLINE void zTsSpatialStructures::setVertexOffsets(vector<double> &offsets)
	{
		if (offsets.size() != fnForm.numVertices()) throw std::invalid_argument("size of offsets contatiner is not equal to number of graph vertices.");

		formGraphVertex_Offsets = offsets;
	}

	ZSPACE_INLINE void zTsSpatialStructures::setFormEdgeWeightsfromVolume(zDomainFloat weightDomain)
	{
		//compute edgeWeights
		vector<vector<double>> volMesh_fAreas;

		zDomainFloat areaDomain(10000, -100000);
		zDomainColor colDomain(zColor(0.784, 0, 0.157, 1), zColor(0.027, 0, 0.157, 1));

		for (int i = 0; i < fnVolumes.size(); i++)
		{
			vector<double> fAreas;
			fnVolumes[i].getPlanarFaceAreas(fAreas);

			double temp_MinArea = coreUtils.zMin(fAreas);
			areaDomain.min = (temp_MinArea < areaDomain.min) ? temp_MinArea : areaDomain.min;

			double temp_maxArea = coreUtils.zMax(fAreas);
			areaDomain.max = (temp_maxArea > areaDomain.max) ? temp_maxArea : areaDomain.max;

			volMesh_fAreas.push_back(fAreas);
		}

		for (zItGraphVertex v(*formObj); !v.end(); v++)
		{
			int i = v.getId();
			if (formGraphVertex_volumeFace[i * 2] == -1 && formGraphVertex_volumeFace[(i * 2) + 1] == -1) continue;

			vector<zItGraphHalfEdge> cEdges;
			v.getConnectedHalfEdges(cEdges);

			int volID = formGraphVertex_volumeMesh[i * 2];
			int faceID = formGraphVertex_volumeFace[i * 2];

			double fArea = volMesh_fAreas[volID][faceID];

			for (int j = 0; j < cEdges.size(); j++)
			{
				double val = coreUtils.ofMap((float)fArea, areaDomain, weightDomain);

				zColor col = coreUtils.blendColor((float)fArea, areaDomain, colDomain, zRGB);

				cEdges[j].getEdge().setWeight(val);
				cEdges[j].getEdge().setColor(col);
			}

		}
	}

	ZSPACE_INLINE void zTsSpatialStructures::extrudeConnectionFaces(int volumeIndex)
	{


		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zVector> fCenters;
		fnVolumes[volumeIndex].getCenters(zFaceData, fCenters);

		fnVolumes[volumeIndex].getVertexPositions(positions);



		for (zItMeshFace f(*volumeObjs[volumeIndex]); !f.end(); f++)
		{
			double faceVal = f.getColor().r;

			vector<int> fVerts;
			f.getVertices(fVerts);

			if (faceVal != 1.0)
			{
				for (int j = 0; j < fVerts.size(); j++) polyConnects.push_back(fVerts[j]);

				polyCounts.push_back(fVerts.size());
			}
			else
			{
				vector<zVector> fVertPositions;
				f.getVertexPositions(fVertPositions);

				string hashKey_f = (to_string(volumeIndex) + "," + to_string(f.getId()));
				int vId_fCenter = -1;
				bool chkExists_f = coreUtils.existsInMap(hashKey_f, volumeFace_formGraphVertex, vId_fCenter);

				zItGraphVertex vForm(*formObj, vId_fCenter);

				double boundaryOffset = formGraphVertex_Offsets[vId_fCenter];

				// get current size of positions
				int numVerts = positions.size();

				// append new positions
				for (int j = 0; j < fVertPositions.size(); j++)
				{
					zVector dir = fCenters[f.getId()] - fVertPositions[j];
					double len = dir.length();
					dir.normalize();

					zVector newPos = fVertPositions[j] + dir * len * boundaryOffset;

					newPos += vForm.getPosition() - fCenters[f.getId()];

					positions.push_back(newPos);
				}

				// compute polyconnects and polycounts
				for (int j = 0; j < fVerts.size(); j++)
				{
					int currentId = j;
					int nextId = (j + 1) % fVerts.size();

					polyConnects.push_back(fVerts[currentId]);
					polyConnects.push_back(fVerts[nextId]);
					polyConnects.push_back(numVerts + nextId);
					polyConnects.push_back(numVerts + currentId);

					polyCounts.push_back(4);

				}

			}



		}


		if (positions.size() > 0)
		{
			fnPolytopals[volumeIndex].create(positions, polyCounts, polyConnects);
		}



	}

	ZSPACE_INLINE void zTsSpatialStructures::getPolytopal(int volumeIndex)
	{
		if (volumeIndex > fnVolumes.size()) throw std::invalid_argument(" error: index out of bounds.");

		string hashKey_volcenter = (to_string(volumeIndex) + "," + to_string(-1));
		int vId_volCenter = -1;
		bool chkExists_f = coreUtils.existsInMap(hashKey_volcenter, volumeFace_formGraphVertex, vId_volCenter);

		zVector volCenter = fnVolumes[volumeIndex].getCenter();

		zItGraphVertex vForm(*formObj, vId_volCenter);

		zVector vCenter_graphPos = vForm.getPosition();
		double volCenterOffset = formGraphVertex_Offsets[vId_volCenter];

		extrudeConnectionFaces(volumeIndex);

		// scale original points from scale center

		for (zItMeshVertex v(*volumeObjs[volumeIndex]); !v.end(); v++)
		{
			if (v.onBoundary()) continue;


			zVector dir = volCenter - v.getPosition();
			double len = dir.length();
			dir.normalize();

			zVector newPos = v.getPosition() + dir * len * volCenterOffset;

			newPos += vCenter_graphPos - volCenter;

			v.setPosition(newPos);


		}

	}

	ZSPACE_INLINE void zTsSpatialStructures::computeVolumesFaceCenters()
	{
		volume_fCenters.clear();

		for (int i = 0; i < fnVolumes.size(); i++)
		{
			vector<zVector> fCenters;
			fnVolumes[i].getCenters(zFaceData, fCenters);

			volume_fCenters.push_back(fCenters);
		}
	}

	ZSPACE_INLINE void zTsSpatialStructures::computeFormTargets()
	{
		targetEdges_form.clear();

		for (int i = 0; i < fnForm.numHalfEdges(); i++)
		{
			targetEdges_form.push_back(zVector());
		}

		for (zItGraphVertex v(*formObj); !v.end(); v++)
		{
			// get position of vertex
			zVector v_i = v.getPosition();

			int i = v.getId();

			int volId_V = formGraphVertex_volumeMesh[i * 2];
			int faceId_V = formGraphVertex_volumeFace[i * 2];

			if (v.checkValency(1))	continue;
			if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] != -1) continue;

			// get connected vertices
			vector<zItGraphHalfEdge> cEdges;
			v.getConnectedHalfEdges(cEdges);


			for (auto &he : cEdges)
			{
				// get vertex 
				int v1_ID = he.getVertex().getId();

				zVector v_j = he.getVertex().getPosition();
				zVector e_ij = v_i - v_j;
				e_ij.normalize();

				// get volume and face Id of the connected Vertex
				int volId = formGraphVertex_volumeMesh[v1_ID * 2];
				int faceId = formGraphVertex_volumeFace[v1_ID * 2];

				zItMeshFace forceFace(*volumeObjs[volId], faceId);

				zVector t_ij = forceFace.getNormal();;
				t_ij.normalize();

				if (e_ij * t_ij > 0) t_ij *= -1;

				targetEdges_form[he.getId()] = t_ij;

				int symEdge = he.getSym().getId();
				targetEdges_form[symEdge] = (t_ij * -1);

			}
		}

	}

	ZSPACE_INLINE bool zTsSpatialStructures::checkParallelity(zDomainFloat & deviation, double angleTolerance, bool colorEdges, bool printInfo)
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

				zColor col = coreUtils.blendColor(deviations[e.getId()], deviation, colDomain, zHSV);

				if (deviations[e.getId()] < angleTolerance) col = zColor();

				e.setColor(col);

			}

		}

		return out;
	}

	ZSPACE_INLINE bool zTsSpatialStructures::updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations)
	{
		bool out = true;

		zVector* pos = fnForm.getRawVertexPositions();

		if (fnFormParticles.size() != fnForm.numVertices())
		{
			fnFormParticles.clear();
			formParticlesObj.clear();


			for (zItGraphVertex v(*formObj); !v.end(); v++)
			{
				bool fixed = false;

				int i = v.getId();

				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] != -1)
				{
					fixed = true;
					v.setColor(zColor());
				}

				zObjParticle p;
				p.particle = zParticle(formObj->graph.vertexPositions[i], fixed);
				formParticlesObj.push_back(p);

			}

			for (int i = 0; i < formParticlesObj.size(); i++)
			{
				fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
			}
		}

		if (volume_fCenters.size() != fnVolumes.size()) computeVolumesFaceCenters();

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
				zVector v_i = pos[i];

				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				// get connected vertices
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				// compute barycenter per vertex
				zVector b_i;
				for (auto &he : cEdges)
				{
					// get vertex 
					int v1_ID = he.getVertex().getId();

					zVector v_j = pos[v1_ID];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();;

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					int symEdge = he.getSym().getId();

					zVector t_ij = targetEdges_form[symEdge];;
					t_ij.normalize();
					if (e_ij * t_ij < 0) t_ij *= -1;

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				zVector r_i = b_i - v_i;
				zVector forceV = r_i;


				// add forces to particle
				fnFormParticles[i].addForce(forceV);

			}


			// In plane force for valence 1 vertices
			//for (int i = 0; i < fnFormParticles.size(); i++)
			//{
			//	if (fnForm.checkVertexValency(i, 1))
			//	{
			//		// add force to keep point in the plane of the face

			//		int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
			//		int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

			//		zVector fNorm = fnForces[volId_V].getFaceNormal(faceId_V);
			//		fNorm.normalize();

			//		zVector V = fnForm.getVertexPosition(i);

			//		zVector e = V - force_fCenters[volId_V][faceId_V];

			//		double minDist = coreUtils.minDist_Point_Plane(V, force_fCenters[volId_V][faceId_V], fNorm);

			//		if (minDist > 0)
			//		{
			//			if (e * fNorm >= 0)	fNorm *= -1;


			//			zVector forceV = (fNorm * minDist);
			//			fnFormParticles[i].addForce(forceV);
			//		}

			//		//forceV += (e * -1);

			//	}
			//}


			// update Particles
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				fnFormParticles[i].integrateForces(dT, type);
				fnFormParticles[i].updateParticle(true);
			}

			// update fixed particle positions ( ones with a common face) 
			for (zItGraphVertex v(*formObj); !v.end(); v++)
			{
				int i = v.getId();

				if (!fnFormParticles[i].getFixed()) continue;

				vector<zItGraphVertex> cVerts;
				v.getConnectedVertices(cVerts);


				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				zItMeshFace fForce(*volumeObjs[volId_V], faceId_V);

				zVector normF1 = fForce.getNormal();

				zVector currentPos = v.getPosition();

				if (cVerts.size() == 2)
				{
					zVector p1 = cVerts[0].getPosition();

					zVector p2 = cVerts[1].getPosition();

					zVector interPt;

					zVector newPos = (p1 + p2) *0.5;

					v.setPosition(newPos);

					/*bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						fnForm.setVertexPosition(i,interPt);
					}*/
				}

			}

		}
	}

}