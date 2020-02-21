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


#include<headers/zInterface/functionsets/zFnMeshDynamics.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnMeshDynamics::zFnMeshDynamics()
	{
		fnType = zFnType::zMeshDynamicsFn;
		meshObj = nullptr;
	}

	ZSPACE_INLINE zFnMeshDynamics::zFnMeshDynamics(zObjMesh &_meshObj)
	{
		fnType = zFnType::zMeshDynamicsFn;
		meshObj = &_meshObj;


	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnMeshDynamics::~zFnMeshDynamics() {}

	//---- OVERRIDE METHODS
	
	ZSPACE_INLINE zFnType zFnMeshDynamics::getType()
	{
		return zFnType::zMeshDynamicsFn;
	}

	ZSPACE_INLINE void zFnMeshDynamics::from(string path, zFileTpye type)
	{
		if (type == zOBJ) fromOBJ(path);
		else if (type == zJSON) fromJSON(path);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnMeshDynamics::to(string path, zFileTpye type)
	{
		if (type == zOBJ) toOBJ(path);
		else if (type == zJSON) toJSON(path);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnMeshDynamics::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		meshObj->getBounds(minBB, maxBB);
	}

	ZSPACE_INLINE void zFnMeshDynamics::clear()
	{

		zFnMesh::clear();

		fnParticles.clear();
		particlesObj.clear();
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnMeshDynamics::makeDynamic(bool fixBoundary)
	{
		fnParticles.clear();
		particlesObj.clear();

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			bool fixed = false;

			if (fixBoundary) fixed = (v.onBoundary());

			zObjParticle p;
			p.particle = zParticle(*v.getRawPosition(), fixed);
			particlesObj.push_back(p);

			if (!fixed) setVertexColor(zColor(0, 0, 1, 1));
		}

		for (int i = 0; i < particlesObj.size(); i++)
		{
			fnParticles.push_back(zFnParticle(particlesObj[i]));
		}

	}

	ZSPACE_INLINE void zFnMeshDynamics::create(zObjMesh &_meshObj, bool fixBoundary)
	{
		meshObj = &_meshObj;
		makeDynamic(fixBoundary);

	}

	//---- FORCE METHODS 

	ZSPACE_INLINE void zFnMeshDynamics::addGravityForce(zVector grav)
	{
		for (int i = 0; i < fnParticles.size(); i++)
		{
			fnParticles[i].addForce(grav);
		}
	}

	ZSPACE_INLINE void zFnMeshDynamics::addEdgeForce(const zDoubleArray &weights)
	{

		if (weights.size() > 0 && weights.size() != meshObj->mesh.vertices.size()) throw std::invalid_argument("cannot apply edge force.");

		for (zItMeshVertex v(*meshObj); !v.end(); v++)
		{
			int i = v.getId();

			if (v.isActive())
			{
				if (fnParticles[i].getFixed()) continue;

				vector<zItMeshHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zVector eForce;

				for (auto &he : cEdges)
				{


					int v1 = he.getVertex().getId();
					zVector e = meshObj->mesh.vertexPositions[v1] - meshObj->mesh.vertexPositions[i];

					double len = e.length();
					e.normalize();

					if (weights.size() > 0) e *= weights[i];

					eForce += (e * len);
				}

				fnParticles[i].addForce(eForce);

			}
			else  fnParticles[i].setFixed(true);;

		}

	}

	ZSPACE_INLINE void zFnMeshDynamics::addPlanarityForce(vector<double> &fVolumes, vector<zVector> fCenters, double tolerance)
	{
		if (fVolumes.size() != meshObj->mesh.faces.size()) throw std::invalid_argument("sizes of face Volumes and mesh faces dont match.");
		if (fCenters.size() != meshObj->mesh.faces.size()) throw std::invalid_argument("sizes of face Centers and mesh faces dont match.");

		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			int i = f.getId();

			if (f.isActive() && fVolumes[i] > tolerance)
			{
				vector<int> fVerts;
				f.getVertices(fVerts);

				vector<zVector> fVertPositions;
				f.getVertexPositions(fVertPositions);

				zVector fNormal = meshObj->mesh.faceNormals[i];

				for (int j = 0; j < fVertPositions.size(); j++)
				{
					if (fnParticles[fVerts[j]].getFixed()) continue;

					double dist = meshObj->mesh.coreUtils.minDist_Point_Plane(fVertPositions[j], fCenters[i], fNormal);

					zVector pForce = fNormal * dist * -1.0;
					fnParticles[fVerts[j]].addForce(pForce);

				}
			}

		}
	}

	//---- UPDATE METHODS 

	ZSPACE_INLINE void zFnMeshDynamics::update(double dT, zIntergrationType type, bool clearForce, bool clearVelocity, bool clearDerivatives)
	{
		for (int i = 0; i < fnParticles.size(); i++)
		{
			fnParticles[i].integrateForces(dT, type);
			fnParticles[i].updateParticle(clearForce, clearVelocity, clearDerivatives);
		}

		computeMeshNormals();
	}

}