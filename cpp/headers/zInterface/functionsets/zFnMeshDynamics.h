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

#ifndef ZSPACE_FN_MESH_DYNAMICS_H
#define ZSPACE_FN_MESH_DYNAMICS_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnParticle.h>


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

	/*! \class zFnMeshDynamics
	*	\brief A mesh function set for dynamics.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zFnMeshDynamics : public zFnMesh
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
		zFnMeshDynamics();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zFnMeshDynamics(zObjMesh &_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnMeshDynamics();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------
		
		zFnType getType() override;

		void from(string path, zFileTpye type);

		void to(string path, zFileTpye type);

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		void clear() override;
		
		//--------------------------
		//---- CREATE METHODS
		//--------------------------
				

		/*! \brief This method creates the particles object from the mesh object already attached to zFnMeshDynamics.
		*
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void makeDynamic(bool fixBoundary = false);

		/*! \brief This method creates a particle system from the input mesh object.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	fixBoundary			- true if the boundary vertices are to be fixed.
		*	\since version 0.0.2
		*/
		void create(zObjMesh &_meshObj, bool fixBoundary = false);

		
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
		*	\param		[in]	weights				- Input container of weights per vertex.
		*	\since version 0.0.2
		*/
		void addEdgeForce(const zDoubleArray &weights = zDoubleArray());

		/*! \brief This method adds the planarity force to all the particles in the input container based on the face volumes of the input mesh.
		*
		*	\param		[in]	fVolumes			- container of face volumes. Use getMeshFaceVolumes method to compute it.
		*	\param		[in]	fCenters			- container of face centers. Use getCenters method to compute it.
		*	\param		[in]	tolerance			- tolerance value. Default it is set to 0.001.
		*	\since version 0.0.2
		*/
		void addPlanarityForce(vector<double> &fVolumes, vector<zVector> fCenters, double tolerance = EPS);

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

	};
}



#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnMeshDynamics.cpp>
#endif

#endif