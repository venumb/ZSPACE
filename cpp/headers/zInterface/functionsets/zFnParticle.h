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

#ifndef ZSPACE_FN_PARTICLE_H
#define ZSPACE_FN_PARTICLE_H

#pragma once

#include<headers/zInterface/objects/zObjParticle.h>
#include<headers/zInterface/functionsets/zFn.h>

namespace zSpace
{
	/** \addtogroup zInterface
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

	class ZSPACE_API zFnParticle : protected zFn
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
		zFnParticle();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_particleObj			- input particle object.
		*	\since version 0.0.2
		*/
		zFnParticle(zObjParticle &_particleObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnParticle();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		zFnType getType() override;

		void from(string path, zFileTpye type, bool staticGeom = false) override;

		void to(string path, zFileTpye type) override;

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		void clear() override;

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
		void create(zPoint &_p, bool _fixed = false, double _m = 1.0, zVector _v = zVector(), zVector _f = zVector());

		//--------------------------
		//---- SET METHODS
		//--------------------------
				
		/*! \brief This method sets the mass of the particle.
		*
		*	\param		[in]	mass		- input mass.
		*	\since version 0.0.2
		*/
		void setMass(double mass);

		/*! \brief This method sets the position of the particle.
		*
		*	\param		[in]	pos			- input position.
		*	\since version 0.0.2
		*/
		void setPosition(zVector *pos);

		/*! \brief This method sets the velocity of the particle.
		*
		*	\param		[in]	vel			- input velocity.
		*	\since version 0.0.2
		*/
		void setVelocity(zVector &vel);

		/*! \brief This method sets the state of the particle.
		*
		*	\param		[in]	state			- input state.
		*	\since version 0.0.2
		*/
		void setState(zParticleState &state);

		/*! \brief This method sets the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.2
		*/
		void setForce(zVector &force);

		/*! \brief This method sets the fixed boolean  of the particle.
		*
		*	\param		[in]	_fixed		- input fixed boolean.
		*	\since version 0.0.2
		*/
		void setFixed(bool _fixed);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method returns the mass of the particle.
		*
		*	\return			double	- mass of the particle.
		*	\since version 0.0.2
		*/
		double getMass();

		/*! \brief This method returns the position of the particle.
		*
		*	\return			zPoint	- position of the particle.
		*	\since version 0.0.2
		*/
		zPoint getPosition();

		/*! \brief This method returns the pointer to the position.
		*
		*	\return			zPoint*	- pointer to the position.
		*	\since version 0.0.2
		*/
		zPoint* getRawPosition();

		/*! \brief This method returns the velocity of the particle.
		*
		*	\return			zVector	- velocity of the particle.
		*	\since version 0.0.2
		*/
		zVector getVelocity();

		/*! \brief This method returns the current state of the particle.
		*
		*	\return			zParticleState	- current state of the particle.
		*	\since version 0.0.2
		*/
		zParticleState getState();

		/*! \brief This method returns the force of the particle.
		*
		*	\return			zVector	- force of the particle.
		*	\since version 0.0.2
		*/
		zVector getForce();

		/*! \brief This method returns the boolean if the article is fixed or not.
		*
		*	\return			bool	- true is the particle is fixed.
		*	\since version 0.0.2
		*/
		bool getFixed();

		//--------------------------
		//---- FORCE METHODS
		//--------------------------

		/*! \brief This method adds the input force to the force of the particle.
		*
		*	\param		[in]	force		- input force.
		*	\since version 0.0.2
		*/
		void addForce(zVector &force);

		/*! \brief This method clears the force of the particle.
		*
		*	\since version 0.0.2
		*/
		void clearForce();

		/*! \brief This method adds the all input forces to the force of the particle.
		*
		*	\param		[in]	forces		- container of input forces.
		*	\since version 0.0.2
		*/
		void addForces(vector<zVector> &forces);

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
		zParticleDerivative getDerivatives(zParticleDerivative &deriv, double dT);

		/*! \brief This method intergrates the force and compute the derivatives.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/ and http://paulbourke.net/miscellaneous/particle/particlelib.c
		*	\param		[in]	dT		- time step.
		*	\param		[in]	type	- integration type.
		*	\since version 0.0.2
		*/
		void integrateForces(double dT, zIntergrationType type = zEuler);

		/*! \brief This method intergrates the force, compute the derivatives and returns a zVector of the updated position. Generally used to check if the updated position is in bounds before updating the particle.
		*
		*	\param		[in]	dT		- time step.
		*	\param		[in]	type	- integration type.
		*	\since version 0.0.2
		*/
		zVector getUpdatePosition(double dT, zIntergrationType type = zEuler);

		/*! \brief This method updates the position and velocity of the particle.
		*
		*	\details Based on https://gafferongames.com/post/integration_basics/
		*	\param		[in]	clearForce			- clears the force if true.
		*	\param		[in]	clearVelocity		- clears the velocity if true.
		*	\param		[in]	clearDerivatives	- clears the derivatives if true.
		*	\since version 0.0.2
		*/
		void updateParticle(bool clearForce = false, bool clearVelocity = false, bool clearDerivatives = false);

		//--------------------------
		//---- TRANSFORM METHODS OVERRIDES
		//--------------------------

		void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) override;

		void setScale(zFloat4 &scale) override;

		void setRotation(zFloat4 &rotation, bool appendRotations = false) override;

		void setTranslation(zVector &translation, bool appendTranslations = false) override;

		void setPivot(zVector &pivot) override;

		void getTransform(zTransform &transform) override;

	protected:

		//--------------------------
		//---- TRANSFORM  METHODS
		//--------------------------
		void transformObject(zTransform &transform) override;
		
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnParticle.cpp>
#endif

#endif