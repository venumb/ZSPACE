#pragma once

#include <headers/framework/dynamics/zParticle.h>

#include <headers/api/functionsets/zFnGraph.h>

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

	/*! \class zFnGraphDynamics
	*	\brief A graph function set for dynamics.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnGraphDynamics : public zFnGraph
	{
	private:
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
		zFnGraphDynamics() 
		{
			fnType = zFnType::zGraphDynamicsFn;
			graphObj = nullptr;
			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.2
		*/
		zFnGraphDynamics(zObjGraph &_graphObj)
		{
			graphObj = &_graphObj;
			fnType = zFnType::zGraphDynamicsFn;			
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnGraphDynamics() {}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------
				

		void clear() override
		{
			zFnGraph::clear();		

			fnParticles.clear();
			particlesObj.clear();
		}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the particles object from the graph object already attached to zFnGraphDynamics.
		*
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void makeDynamic(bool fixBoundary = false)
		{
			fnParticles.clear();

			for (zItGraphVertex v(*graphObj); !v.end(); v.next())			
			{
				bool fixed = false;

				if (fixBoundary) fixed = (v.checkVertexValency(1));

				zObjParticle p;
				p.particle = zParticle(*v.getRawVertexPosition(), fixed);
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
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void create(zObjGraph &_graphObj, bool fixBoundary = false)
		{
			graphObj = &_graphObj;

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
		*	\param		[in]	inHEDataStructure	- Input graph or mesh.
		*	\param		[in]	weights				- Input container of weights per force.
		*	\since version 0.0.2
		*/
		void addEdgeForce(const vector<double> &weights = vector<double>())
		{

			if (weights.size() > 0 && weights.size() != graphObj->graph.vertices.size()) throw std::invalid_argument("cannot apply edge force.");

			for (zItGraphVertex v(*graphObj); !v.end(); v.next())
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



