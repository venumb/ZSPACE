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

#ifndef ZSPACE_PARTICLE_H
#define ZSPACE_PARTICLE_H

#pragma once

#include <headers/zCore/base/zVector.h>
#include <headers/zCore/base/zMatrix.h>
#include <headers/zCore/base/zColor.h>

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/


	/*! \struct zParticleState
	*	\brief A struct to store the postion and velocity of the particle.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE zParticleState
	{
		/*!	\brief stores pointer to position. */
		zVector *p = nullptr;

		/*!	\brief stores the velocity. */
		zVector v;
	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/

	/*! \struct zParticleDerivative
	*	\brief A strcut to store the postion and velocity derivatives of the particle.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE zParticleDerivative
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

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zDynamics
	*	\brief The physics and dynamics classes of the library.
	*  @{
	*/	

	/*! \class zParticle
	*	\brief A particle class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zParticle
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
		zParticle();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_p			- position of the particle.
		*	\param		[in]	_fixed		- state of the particle, by deafult true/ active;
		*	\param		[in]	_m			- mass of the particle.
		*	\param		[in]	_v			- velocity of the particle.
		*	\param		[in]	_f			- force of the particle.		
		*	\since version 0.0.1
		*/
		zParticle(zVector &_p, bool _fixed = false, double _m = 1.0, zVector _v = zVector(), zVector _f = zVector());

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zParticle();

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/dynamics/zParticle.cpp>
#endif

#endif