#pragma once

#include<headers/api/object/zObjParticle.h>
#include<headers/api/functionsets/zFn.h>

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

	/*! \class zFnParticle
	*	\brief A particle function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnParticle : protected zFn
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a particle object  */
		zObjParticle *particleObj;

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnParticle()
		{
			fnType = zFnType::zMeshFn;

			particleObj = nullptr;
			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_particleObj			- input particle object.
		*	\since version 0.0.2
		*/
		zFnParticle(zObjParticle &_particleObj)
		{
			particleObj = &_particleObj;
			fnType = zFnType::zParticleFn;

			
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnParticle() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		
		/*! \brief This method creates a particle from the input parameters.
		*
		*	\param		[in]	_p			- position of the particle.
		*	\param		[in]	_fixed		- state of the particle, by deafult true/ active;
		*	\param		[in]	_m			- mass of the particle.
		*	\param		[in]	_v			- velocity of the particle.
		*	\param		[in]	_f			- force of the particle.
		*	\since version 0.0.2
		*/
		void create(zVector &_p, bool _fixed = false, double _m = 1.0, zVector _v = zVector(), zVector _f = zVector())
		{
			

			if (particleObj == NULL)
			{
				zObjParticle pObj;
				pObj.particle = zParticle(_p, _fixed, _m, _v, _f);

				particleObj = &pObj;
			}
			else particleObj->particle = zParticle(_p, _fixed, _m, _v, _f);
		}

		


		//--------------------------
		//---- SET METHODS
		//--------------------------
		
		
		/*! \brief This method sets the mass of the particle.
		*
		*	\param		[in]	mass		- input mass.
		*	\since version 0.0.2
		*/
		void setMass(double mass)
		{
			particleObj->particle.m = mass;
		}

		/*! \brief This method sets the position of the particle.
		*
		*	\param		[in]	pos			- input position.
		*	\since version 0.0.2
		*/
		void setPosition(zVector *pos)
		{
			particleObj->particle.s.p = pos;
		}

		/*! \brief This method sets the velocity of the particle.
		*
		*	\param		[in]	vel			- input velocity.
		*	\since version 0.0.2
		*/
		void setVelocity(zVector &vel)
		{
			particleObj->particle.s.v = vel;
		}

		/*! \brief This method sets the state of the particle.
		*
		*	\param		[in]	state			- input state.
		*	\since version 0.0.2
		*/
		void setState(zParticleState &state)
		{
			particleObj->particle.s = state;
		}

		/*! \brief This method sets the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.2
		*/
		void setForce(zVector &force)
		{
			particleObj->particle.f = force;
		}

		/*! \brief This method sets the fixed boolean  of the particle.
		*
		*	\param		[in]	_fixed		- input fixed boolean.
		*	\since version 0.0.2
		*/
		void setFixed(bool _fixed)
		{
			particleObj->particle.fixed = _fixed;
		}

		

		//--------------------------
		//---- GET METHODS
		//--------------------------

		
		/*! \brief This method retruns the mass of the particle.
		*
		*	\return			double	- mass of the particle.
		*	\since version 0.0.2
		*/
		double getMass()
		{
			return particleObj->particle.m;
		}

		/*! \brief This method retruns the position of the particle.
		*
		*	\return			zVector	- position of the particle.
		*	\since version 0.0.2
		*/
		zVector getPosition()
		{			
			return *particleObj->particle.s.p;
		}

		/*! \brief This method retruns the velocity of the particle.
		*
		*	\return			zVector	- velocity of the particle.
		*	\since version 0.0.2
		*/
		zVector getVelocity()
		{
			return particleObj->particle.s.v;
		}

		/*! \brief This method retruns the current state of the particle.
		*
		*	\return			zParticleState	- current state of the particle.
		*	\since version 0.0.2
		*/
		zParticleState getState()
		{
			return particleObj->particle.s;
		}

		/*! \brief This method retruns the force of the particle.
		*
		*	\return			zVector	- force of the particle.
		*	\since version 0.0.2
		*/
		zVector getForce()
		{
			return particleObj->particle.f;
		}

		/*! \brief This method retruns the boolean if the article is fixed or not.
		*
		*	\return			bool	- true is the particle is fixed.
		*	\since version 0.0.2
		*/
		bool getFixed()
		{
			return particleObj->particle.fixed;
		}

		

		//--------------------------
		//---- FORCE METHODS
		//--------------------------

		

		/*! \brief This method adds the input force to the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.2
		*/
		void addForce(zVector &force)
		{
			particleObj->particle.f += force;
		}

		/*! \brief This method clears the force of the particle.
		*
		*	\since version 0.0.2
		*/
		void clearForce()
		{
			particleObj->particle.f = zVector();
		}

		/*! \brief This method adds the all input forces to the force of the particle.
		*
		*	\param		[in]	forces		- container of input forces.
		*	\since version 0.0.2
		*/
		void addForces(vector<zVector> &forces)
		{
			for (int i = 0; i < forces.size(); i++) particleObj->particle.f += forces[i];
		}

		


		//--------------------------
		//---- INTEGRATION METHODS
		//--------------------------

		

		/*! \brief This method computes the derivatives.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/ and http://paulbourke.net/miscellaneous/particle/particlelib.c
		*	\param		[in]	deriv	- current derivatives.
		*	\param		[in]	dT		- timestep.
		*	\since version 0.0.2
		*/
		zParticleDerivative getDerivatives(zParticleDerivative &deriv, double dT)
		{
			zParticleDerivative out;

			//acceleration
			out.dV = (particleObj->particle.f / particleObj->particle.m) * dT;

			//veclocity
			out.dP = deriv.dV + (out.dV * dT);

			return out;
		}

		/*! \brief This method intergrates the force and compute the derivatives.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/ and http://paulbourke.net/miscellaneous/particle/particlelib.c
		*	\param		[in]	dT		- time step.
		*	\param		[in]	type	- integration type.
		*	\since version 0.0.2
		*/
		void integrateForces(double dT, zIntergrationType type = zEuler)
		{

			if (getFixed())
			{			
				return;
			}

			// Semi-Implicit Euler Integrate
			if (type == zEuler)
			{
				particleObj->particle.derivative = getDerivatives(particleObj->particle.derivative, dT);				
			}

			// Renge Kutta 4th Derivative Integrate
			else if (type == zRK4)
			{

				zParticleDerivative a, b, c, d, temp;

				a = getDerivatives(particleObj->particle.derivative, dT) * 0.5;

				temp = particleObj->particle.derivative + a;
				b = getDerivatives(temp, dT) * 0.5;

				temp = particleObj->particle.derivative + b;
				c = getDerivatives(temp, dT);

				temp = particleObj->particle.derivative + c;
				d = getDerivatives(temp, dT);

				particleObj->particle.derivative = a * (0.167) + b * (0.334) + c * (0.334) + d * (0.167);

			}

			// pixel Integrate
			else if (type == zPixel)
			{

			}

			else throw std::invalid_argument(" error: invalid zIntergrationType ");

		}

		/*! \brief This method intergrates the force, compute the derivatives and returns a zVector of the updated position. Generally used to check if the updated position is in bounds before updating the particle.
		*
		*	\param		[in]	dT		- time step.
		*	\param		[in]	type	- integration type.
		*	\since version 0.0.2
		*/
		zVector getUpdatePosition(double dT, zIntergrationType type = zEuler)
		{
			integrateForces(dT, type);
			return (*particleObj->particle.s.p + particleObj->particle.derivative.dP);
		}

		/*! \brief This method updates the position and velocity of the particle.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/
		*	\param		[in]	clearForce			- clears the force if true.
		*	\param		[in]	clearVelocity		- clears the velocity if true.
		*	\param		[in]	clearDerivatives	- clears the derivatives if true.
		*	\since version 0.0.2
		*/
		void updateParticle(bool clearForce = false, bool clearVelocity = false, bool clearDerivatives = false)
		{
			if (getFixed()) return;

			*particleObj->particle.s.p += particleObj->particle.derivative.dP;
			particleObj->particle.s.v += particleObj->particle.derivative.dV;

			if (clearForce) particleObj->particle.f = zVector();
			if (clearVelocity) particleObj->particle.s.v = zVector();
			if (clearDerivatives) particleObj->particle.derivative = zParticleDerivative();
		}

		
	};
}