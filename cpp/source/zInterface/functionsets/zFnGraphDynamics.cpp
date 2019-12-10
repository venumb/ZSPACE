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


#include<headers/zInterface/functionsets/zFnGraphDynamics.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnGraphDynamics::zFnGraphDynamics()
	{
		fnType = zFnType::zGraphDynamicsFn;
		graphObj = nullptr;

	}

	ZSPACE_INLINE zFnGraphDynamics::zFnGraphDynamics(zObjGraph &_graphObj)
	{
		graphObj = &_graphObj;
		fnType = zFnType::zGraphDynamicsFn;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnGraphDynamics::~zFnGraphDynamics() {}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE zFnType zFnGraphDynamics::getType()
	{
		return zFnType::zGraphDynamicsFn;
	}

	ZSPACE_INLINE void zFnGraphDynamics::from(string path, zFileTpye type, bool staticGeom)
	{
		if (type == zTXT)
		{
			fromTXT(path);
			setStaticContainers();
		}
		else if (type == zMAYATXT)
		{
			fromMAYATXT(path);
			setStaticContainers();
		}
		else if (type == zJSON)
		{
			fromJSON(path);
			setStaticContainers();
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnGraphDynamics::to(string path, zFileTpye type)
	{
		if (type == zTXT) toTXT(path);
		else if (type == zJSON) toJSON(path);

		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	ZSPACE_INLINE void zFnGraphDynamics::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		graphObj->getBounds(minBB, maxBB);
	}

	ZSPACE_INLINE void zFnGraphDynamics::clear()
	{
		zFnGraph::clear();

		fnParticles.clear();
		particlesObj.clear();
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnGraphDynamics::makeDynamic(bool fixBoundary)
	{
		fnParticles.clear();

		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			bool fixed = false;

			if (fixBoundary) fixed = (v.checkValency(1));

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

	ZSPACE_INLINE void zFnGraphDynamics::create(zObjGraph &_graphObj, bool fixBoundary)
	{
		graphObj = &_graphObj;

		makeDynamic(fixBoundary);

	}

	//---- FORCE METHODS 

	ZSPACE_INLINE void zFnGraphDynamics::addGravityForce(zVector grav)
	{
		for (int i = 0; i < fnParticles.size(); i++)
		{
			fnParticles[i].addForce(grav);
		}
	}

	ZSPACE_INLINE void zFnGraphDynamics::addEdgeForce(const zDoubleArray &weights)
	{

		if (weights.size() > 0 && weights.size() != graphObj->graph.vertices.size()) throw std::invalid_argument("cannot apply edge force.");

		for (zItGraphVertex v(*graphObj); !v.end(); v++)
		{
			int i = v.getId();

			if (v.isActive())
			{
				if (fnParticles[i].getFixed()) continue;

				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zVector eForce;

				for (auto &he : cEdges)
				{
					int v1 = he.getVertex().getId();
					zVector e = graphObj->graph.vertexPositions[v1] - graphObj->graph.vertexPositions[i];

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

	//---- UPDATE METHODS 

	ZSPACE_INLINE void zFnGraphDynamics::update(double dT, zIntergrationType type, bool clearForce , bool clearVelocity, bool clearDerivatives)
	{
		for (int i = 0; i < fnParticles.size(); i++)
		{
			fnParticles[i].integrateForces(dT, type);
			fnParticles[i].updateParticle(clearForce, clearVelocity, clearDerivatives);
		}
	}

	//---- PRIVATE METHODS

	ZSPACE_INLINE void zFnGraphDynamics::setStaticContainers()
	{
		graphObj->graph.staticGeometry = true;

		vector<vector<int>> edgeVerts;

		for (zItGraphEdge e(*graphObj); !e.end(); e++)
		{
			vector<int> verts;
			e.getVertices(verts);

			edgeVerts.push_back(verts);
		}

		graphObj->graph.setStaticEdgeVertices(edgeVerts);
	}
}