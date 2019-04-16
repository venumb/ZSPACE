#pragma once

#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnParticle.h>


namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnMeshDynamics
	*	\brief A mesh function set for dynamics.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnMeshDynamics : public zFnMesh
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------
		
		/*!	\brief container of particle function set  */
		vector<zFnParticle> fnParticles;	

		/*!	\brief container of  particle objects  */
		vector<zObjParticle> particlesObj;
		
	public:	

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnMeshDynamics() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zFnMeshDynamics(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;				
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnMeshDynamics() {}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------
			

		void clear() override
		{

			zFnMesh::clear();			

			fnParticles.clear();
			particlesObj.clear();
		}
		
		//--------------------------
		//---- CREATE METHODS
		//--------------------------
				

		/*! \brief This method creates the particles object from the mesh object already attached to zFnMeshDynamics.
		*
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void makeDynamic( bool fixBoundary = false)
		{
			fnParticles.clear();
			particlesObj.clear();
			

			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				bool fixed = false;

				if (fixBoundary) fixed = (onBoundary(i, zVertexData));
	
				zObjParticle p;
				p.particle = zParticle(meshObj->mesh.vertexPositions[i], fixed);
				particlesObj.push_back(p);

				if (!fixed) setVertexColor(zColor(0, 0, 1, 1));
			}

			for (int i = 0; i < particlesObj.size(); i++)
			{
				fnParticles.push_back(zFnParticle(particlesObj[i]));
			}

		}

		/*! \brief This method creates a particle system from the input mesh object.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void makeDynamic(zObjMesh &_meshObj, bool fixBoundary = false)
		{
			meshObj = &_meshObj;
			makeDynamic(fixBoundary);

		}

		
		//--------------------------
		//---- FORCE METHODS 
		//--------------------------

		/*! \brief This method adds the input gravity force to all the particles in the input container.
		*
		*	\param		[in]		grav		- Input gravity force.
		*	\since version 0.0.2
		*/
		void addGravityForce(zVector grav = zVector(0, 0, -9.8))
		{
			for (int i = 0; i < fnParticles.size(); i++)
			{
				fnParticles[i].addForce(grav);
			}
		}

		/*! \brief This method adds the edge forces to all the particles in the input container based on the input graph/ mesh.
		*
		*	\param		[in]	weights				- Input container of weights per vertex.
		*	\since version 0.0.2
		*/
		void addEdgeForce(const vector<double> &weights = vector<double>())
		{

			if (weights.size() > 0 && weights.size() != meshObj->mesh.vertexActive.size()) throw std::invalid_argument("cannot apply edge force.");
			
			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				if (meshObj->mesh.vertexActive[i])
				{
					if (fnParticles[i].getFixed()) continue;

					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					zVector eForce;

					for (int j = 0; j < cEdges.size(); j++)
					{
						int v1 = meshObj->mesh.edges[cEdges[j]].getVertex()->getVertexId();
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

		/*! \brief This method adds the planarity force to all the particles in the input container based on the face volumes of the input mesh.
		*
		*	\param		[in]	fVolumes			- container of face volumes. Use getMeshFaceVolumes method to compute it.
		*	\param		[in]	fCenters			- container of face centers. Use getCenters method to compute it.
		*	\param		[in]	tolerance			- tolerance value. Default it is set to 0.001.
		*	\since version 0.0.2
		*/
		void addPlanarityForce(vector<double> &fVolumes, vector<zVector> fCenters, double tolerance = 0.001)
		{
			if (fVolumes.size() != meshObj->mesh.faceActive.size()) throw std::invalid_argument("sizes of face Volumes and mesh faces dont match.");
			if (fCenters.size() != meshObj->mesh.faceActive.size()) throw std::invalid_argument("sizes of face Centers and mesh faces dont match.");

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				if (meshObj->mesh.faceActive[i] && fVolumes[i] > tolerance)
				{
					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					vector<zVector> fVertPositions;
					getVertexPositions(i, zFaceData, fVertPositions);

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

		//--------------------------
		//---- UPDATE METHODS 
		//--------------------------

		/*! \brief This method updates the position and velocity of all the  particle.
		*
		*	\param		[in]	dT					- timestep.
		*	\param		[in]	type				- integration type - zEuler or zRK4.
		*	\param		[in]	clearForce			- clears the force if true.
		*	\param		[in]	clearVelocity		- clears the velocity if true.
		*	\param		[in]	clearDerivatives	- clears the derivatives if true.
		*	\since version 0.0.2
		*/
		void update(double dT, zIntergrationType type = zEuler, bool clearForce = true, bool clearVelocity = false, bool clearDerivatives = false)
		{
			for (int i = 0; i < fnParticles.size(); i++)
			{
				fnParticles[i].integrateForces(dT, type);
				fnParticles[i].updateParticle(clearForce, clearVelocity, clearDerivatives);				
			}
		}

	};
}

