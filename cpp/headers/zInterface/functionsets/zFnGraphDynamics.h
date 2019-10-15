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

#ifndef ZSPACE_FN_GRAPH_DYNAMICS_H
#define ZSPACE_FN_GRAPH_DYNAMICS_H

#pragma once

#include <headers/zInterface/functionsets/zFnParticle.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

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

	/*! \class zFnGraphDynamics
	*	\brief A graph function set for dynamics.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zFnGraphDynamics : public zFnGraph
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
		zFnGraphDynamics();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.2
		*/
		zFnGraphDynamics(zObjGraph &_graphObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnGraphDynamics();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------
		
		zFnType getType() override;

		void from(string path, zFileTpye type, bool staticGeom = false);

		void to(string path, zFileTpye type);

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		void clear() override;

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the particles object from the graph object already attached to zFnGraphDynamics.
		*
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void makeDynamic(bool fixBoundary = false);

		/*! \brief This method creates a particle system from the input mesh object.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void create(zObjGraph &_graphObj, bool fixBoundary = false);		

		//--------------------------
		//---- FORCE METHODS 
		//--------------------------

		/*! \brief This method adds the input gravity force to all the particles in the input container.
		*
		*	\param		[in]		grav		- Input gravity force.
		*	\since version 0.0.2
		*/
		void addGravityForce(zVector grav = zVector(0, 0, -9.8));

		/*! \brief This method adds the edge forces to all the particles in the input container based on the input graph/ mesh.
		*
		*	\param		[in]	inHEDataStructure	- Input graph or mesh.
		*	\param		[in]	weights				- Input container of weights per force.
		*	\since version 0.0.2
		*/
		void addEdgeForce(const zDoubleArray &weights = zDoubleArray());

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
		void update(double dT, zIntergrationType type = zEuler, bool clearForce = true, bool clearVelocity = false, bool clearDerivatives = false);

	private:

		//--------------------------
		//---- PRIVATE METHODS
		//--------------------------

		/*! \brief This method sets the edge vertex position conatiners for static graphs.
		*
		*	\since version 0.0.2
		*/
		void setStaticContainers();
				
	};

	

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnGraphDynamics.cpp>
#endif

#endif
