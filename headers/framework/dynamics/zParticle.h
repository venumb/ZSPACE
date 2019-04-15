#pragma once

#include <headers/framework/core/zVector.h>
#include <headers/framework/core/zMatrix.h>
#include <headers/framework/core/zColor.h>

namespace zSpace
{

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes and utility methods of the library.
	*  @{
	*/


	/*! \struct zParticleState
	*	\brief A struct to store the postion and velocity of the particle.
	*	\since version 0.0.1
	*/

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

	/*! \struct zParticleDerivative
	*	\brief A strcut to store the postion and velocity derivatives of the particle.
	*	\since version 0.0.1
	*/

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

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief stores mass of the particle. */
		double m;

		/*!	\brief stores state of the particle. */
		zParticleState s;

		/*!	\brief stores force acting on the particle. */
		zVector f;

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
			fixed = false;

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
		zParticle(zVector &_p, bool _fixed = false, double _m = 1.0 , zVector _v = zVector(), zVector _f = zVector())
		{
			s.p = &_p;
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

	};
}