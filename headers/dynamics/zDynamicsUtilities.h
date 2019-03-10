#pragma once

#include <headers/dynamics/zParticle.h>

#include <headers/geometry/zGraph.h>
#include <headers/geometry/zMesh.h>

namespace zSpace
{

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zDynamicsUtilities
	*	\brief The physics and dynamics utility methods of the library.
	*  @{
	*/

	//--------------------------
	//---- FORCE METHODS 
	//--------------------------

	/*! \brief This method adds the input gravity force to all the particles in the input container.
	*
	*	\param		[in]		inParticles	- Input particle container.
	*	\param		[in]		grav		- Input gravity force.
	*	\since version 0.0.1
	*/
	inline void addGravityForce(vector<zParticle> &inParticles, zVector grav = zVector(0, 0, -9.8))
	{
		for (int i = 0; i < inParticles.size(); i++)
		{
			inParticles[i].addForce(grav);
		}
	}

	/*! \brief This method adds the edge forces to all the particles in the input container based on the input graph/ mesh.
	*
	*	\tparam				T					- Type to work with zGraph and zMesh.	
	*	\param		[in]	inParticles			- Input particle container.
	*	\param		[in]	inHEDataStructure	- Input graph or mesh.
	*	\param		[in]	weights				- Input container of weights per force.
	*	\since version 0.0.1
	*/
	template<typename T>
	inline void addEdgeForce(vector<zParticle> &inParticles, T &inHEDataStructure , vector<double> &weights);
	
	/*! \brief This method adds the planarity force to all the particles in the input container based on the face volumes of the input mesh.
	*
	*	\param		[in]	inParticles			- Input particle container.
	*	\param		[in]	inMesh				- Input mesh.
	*	\param		[in]	fVolumes			- container of face volumes. Use getMeshFaceVolumes method to compute it.
	*	\param		[in]	fCenters			- container of face centers. Use getCenters method to compute it.
	*	\param		[in]	tolerance			- tolerance value. Default it is set to 0.001.
	*	\since version 0.0.1
	*/
	inline void addPlanarityForce(vector<zParticle> &inParticles, zMesh &inMesh, vector<double> &fVolumes, vector<zVector> fCenters, double tolerance = 0.001)
	{
		if (inParticles.size() != inMesh.vertexActive.size()) throw std::invalid_argument("cannot apply planarity force.");

		if (fVolumes.size() != inMesh.faceActive.size()) throw std::invalid_argument("sizes of face Volumes and mesh faces dont match.");
		if (fCenters.size() != inMesh.faceActive.size()) throw std::invalid_argument("sizes of face Centers and mesh faces dont match.");
		
		for (int i = 0; i < inMesh.faceActive.size(); i++)
		{
			if (inMesh.faceActive[i] && fVolumes[i] > tolerance)
			{
				vector<int> fVerts;
				inMesh.getVertices(i, zFaceData, fVerts);

				vector<zVector> fVertPositions;
				inMesh.getVertexPositions(i, zFaceData, fVertPositions);

				zVector fNormal = inMesh.faceNormals[i];

				for (int j = 0; j < fVertPositions.size(); j++)
				{
					if (inParticles[fVerts[j]].fixed) continue;
					
					double dist = minDist_Point_Plane(fVertPositions[j], fCenters[i], fNormal);				

					zVector pForce = fNormal * dist * -1.0;
					inParticles[fVerts[j]].addForce(pForce);
				}
			}

		}
	}

		
	/** @}*/

	/** @}*/

}


#ifndef DOXYGEN_SHOULD_SKIP_THIS


//--------------------------
//---- TEMPLATE SPECIALIZATION DEFINITIONS 
//--------------------------

//---------------//

//---- graph specilization for addSpringForce
template<>
inline void zSpace::addEdgeForce(vector<zParticle> &inParticles, zGraph &inGraph, vector<double> &weights)
{
	if (inParticles.size() != inGraph.vertexActive.size()) throw std::invalid_argument("cannot apply edge force.");

	if (weights.size() >0 && weights.size() != inGraph.vertexActive.size()) throw std::invalid_argument("cannot apply edge force.");
	
	for (int i = 0; i < inGraph.vertexActive.size(); i++)
	{
		if (inGraph.vertexActive[i])
		{
			if (inParticles[i].fixed) continue;

			vector<int> cEdges;
			inGraph.getConnectedEdges(i, zVertexData, cEdges);

			zVector sForce; 

			for (int j = 0; j < cEdges.size(); j++)
			{
				int v1 = inGraph.edges[cEdges[j]].getVertex()->getVertexId();
				zVector e = inGraph.vertexPositions[v1] - inGraph.vertexPositions[i];
				
				double len = e.length();
				e.normalize();

				if (weights.size() > 0) e *= weights[i];

				sForce += (e * len);
			}

			inParticles[i].addForce(sForce);

		}
		else inParticles[i].fixed = true;;
		
	}
}

//---- mesh specilization for addSpringForce
template<>
inline void zSpace::addEdgeForce(vector<zParticle> &inParticles, zMesh &inMesh, vector<double> &weights)
{
	if (inParticles.size() != inMesh.vertexActive.size()) throw std::invalid_argument("cannot apply edge force.");

	if (weights.size() > 0 && weights.size() != inMesh.vertexActive.size()) throw std::invalid_argument("cannot apply edge force.");

	for (int i = 0; i < inMesh.vertexActive.size(); i++)
	{
		if (inMesh.vertexActive[i])
		{
			if (inParticles[i].fixed) continue;

			vector<int> cEdges;
			inMesh.getConnectedEdges(i, zVertexData, cEdges);

			zVector sForce;

			for (int j = 0; j < cEdges.size(); j++)
			{
				int v1 = inMesh.edges[cEdges[j]].getVertex()->getVertexId();
				zVector e = inMesh.vertexPositions[v1] - inMesh.vertexPositions[i];
				
				double len = e.length();
				e.normalize();

				if (weights.size() > 0) e *= weights[i];

				sForce += (e * len);
			}

			inParticles[i].addForce(sForce);

		}
		else inParticles[i].fixed = true;;

	}
	
}

//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

