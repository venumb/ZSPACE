#pragma once

#include <headers/core/zVector.h>
#include <headers/core/zMatrix.h>
#include <headers/core/zColor.h>

namespace zSpace
{

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zDynamicsClasses
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/

	/*! \struct zParticleState
	*	\brief A struct to store the postion and velocity of the particle.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	struct zParticleState
	{
		/*!	\brief stores pointer to position. */
		zVector *p = nullptr;

		/*!	\brief stores the velocity. */
		zVector v;
	};


	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zDynamicsClasses
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/

	/*! \struct zParticleDerivative
	*	\brief A strcut to store the postion and velocity derivatives of the particle.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	struct zParticleDerivative
	{
		/*!	\brief stores the position derivative. */
		zVector dP;

		/*!	\brief stores the velocity derivative. */
		zVector dV;

		/*! \brief This operator is used for derivative addition.
		*
		*	\param		[in]	d1						- derivative to be added.
		*	\return				zParticleDerivative		- resultant derivative.
		*	\since version 0.0.1
		*/
		zParticleDerivative operator +(zParticleDerivative d1)
		{
			zParticleDerivative out; 

			out.dP = this->dP + d1.dP;
			out.dV = this->dV + d1.dV;

			return out;
		}

		/*! \brief This operator is used for derivative scalar multiplication.
		*
		*	\param		[in]	val						- scalar value.
		*	\since version 0.0.1
		*/
		zParticleDerivative operator *(double val)
		{
			zParticleDerivative out;

			out.dP = this->dP * val;
			out.dP = this->dV * val;

			return out;

		}

		/*! \brief This operator is used for derivative scalar multiplication  and assigment of the result to the current derivative.
		*
		*	\param		[in]	val						- scalar value.
		*	\since version 0.0.1
		*/
		void operator *=(double val)
		{
			this->dP *= val;
			this->dV *= val;
		}

	};

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zDynamicsClasses
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/

	/*! \class zParticle
	*	\brief A particle class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zParticle
	{
	private:

		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief stores mass of the particle. */
		double m;

		/*!	\brief stores state of the particle. */
		zParticleState s;		

		/*!	\brief stores force acting on the particle. */
		zVector f;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief stores if the particle is fixed or active to move. */
		bool fixed;

		/*!	\brief stores the particle derivative. */
		zParticleDerivative derivative;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zParticle()
		{
			m = 1.0;
			fixed = true;

			s.p = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_p			- position of the particle.
		*	\param		[in]	_fixed		- state of the particle, by deafult true/ active;
		*	\param		[in]	_m			- mass of the particle.
		*	\param		[in]	_v			- velocity of the particle.
		*	\param		[in]	_f			- force of the particle.		
		*	\since version 0.0.1
		*/
		zParticle(zVector *_p, bool _fixed = true, double _m = 1.0 , zVector _v = zVector(), zVector _f = zVector())
		{
			s.p = _p;
			s.v = _v;
			f = _f;
			m = _m;

			fixed = _fixed;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zParticle() {}
		
		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method retruns the mass of the particle.
		*
		*	\return			double	- mass of the particle.
		*	\since version 0.0.1
		*/
		double getMass()
		{
			return m;
		}

		/*! \brief This method retruns the position of the particle.
		*
		*	\return			zVector	- position of the particle.
		*	\since version 0.0.1
		*/
		zVector* getPosition()
		{
			return s.p;
		}

		/*! \brief This method retruns the velocity of the particle.
		*
		*	\return			zVector	- velocity of the particle.
		*	\since version 0.0.1
		*/
		zVector getVelocity()
		{
			return s.v;
		}

		/*! \brief This method retruns the current state of the particle.
		*
		*	\return			zParticleState	- current state of the particle.
		*	\since version 0.0.1
		*/
		zParticleState getState()
		{
			return s;
		}

		/*! \brief This method retruns the force of the particle.
		*
		*	\return			zVector	- force of the particle.
		*	\since version 0.0.1
		*/
		zVector getForce()
		{
			return f;
		}
		
		/*! \brief This method sets the mass of the particle.
		*
		*	\param		[in]	mass		- input mass.
		*	\since version 0.0.1
		*/
		void setMass(double mass)
		{
			m = mass;
		}

		/*! \brief This method sets the position of the particle.
		*
		*	\param		[in]	pos			- input position.
		*	\since version 0.0.1
		*/
		void setPosition(zVector *pos)
		{
			s.p = pos;
		}

		/*! \brief This method sets the velocity of the particle.
		*
		*	\param		[in]	vel			- input velocity.
		*	\since version 0.0.1
		*/
		void setVelocity(zVector &vel)
		{
			s.v = vel;
		}

		/*! \brief This method sets the state of the particle.
		*
		*	\param		[in]	state			- input state.
		*	\since version 0.0.1
		*/
		void setState(zParticleState &state)
		{
			s = state;
		}

		/*! \brief This method sets the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.1
		*/
		void setForce(zVector &force)
		{
			f = force;
		}

		//--------------------------
		//---- FORCE METHODS
		//--------------------------

		/*! \brief This method adds the input force to the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.1
		*/
		void addForce(zVector &force)
		{
			f += force;
		}

		/*! \brief This method adds the all input forces to the force of the particle.
		*
		*	\param		[in]	forces		- container of input forces.
		*	\since version 0.0.1
		*/
		void addForces(vector<zVector> &forces)
		{
			for(int i =0; i< forces.size(); i++) f += forces[i];
		}

		//--------------------------
		//---- INTEGRATE METHODS
		//--------------------------
		
		/*! \brief This method computes the derivatives.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/ and http://paulbourke.net/miscellaneous/particle/particlelib.c
		*	\param		[in]	type	- integration type. Works only for RK4.
		*	\since version 0.0.1
		*/
		zParticleDerivative getDerivative( zParticleDerivative &deriv, double dT)
		{
			zParticleDerivative out;

			//acceleration
			out.dV = (f / m) * dT;

			//veclocity
			out.dP = deriv.dV + (out.dV * dT);

			return out;
		}

		/*! \brief This method intergrates the force and cmpute the derivatives.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/ and http://paulbourke.net/miscellaneous/particle/particlelib.c
		*	\param		[in]	dT		- time step.
		*	\param		[in]	type	- integration type.
		*	\since version 0.0.1
		*/
		void integrateForces(double dT , zIntergrationType type = zEuler)
		{
			if (fixed) return;

			// Semi-Implicit Euler Integrate
			if (type == zEuler)
			{
				derivative = getDerivative(derivative, dT);
			}

			// Renge Kutta 4th Derivative Integrate
			else if (type == zRK4)
			{
				
				zParticleDerivative a, b, c, d, temp;		

				a = getDerivative(derivative, dT) * 0.5;				
				
				temp = derivative + a;
				b = getDerivative( temp, dT) * 0.5;				
				
				temp = derivative + b;
				c = getDerivative(temp, dT);				
							
				temp = derivative + c;
				d = getDerivative(temp, dT);				

				derivative = a * (0.167) + b *(0.334) + c * (0.334) + d*(0.167);			

			}

			// pixel Integrate
			else if (type == zPixel)
			{

			}

			else throw std::invalid_argument(" error: invalid zIntergrationType ");
			
		}


		/*! \brief This method updates the position and velocity of the particle.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/
		*	\param		[in]	clearForce		- clears the force if true.
		*	\param		[in]	clearVelocity	- clears the velocity if true.
		*	\since version 0.0.1
		*/
		void updateParticle( bool clearForce = false, bool clearVelocity = false)
		{
			if (fixed) return;

			*s.p += derivative.dP;
			s.v += derivative.dV;

			if (clearForce) f = zVector();
			if (clearVelocity) s.v = zVector();
		}

	};
}