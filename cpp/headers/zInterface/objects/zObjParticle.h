// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//

#ifndef ZSPACE_OBJ_PARTICLE_H
#define ZSPACE_OBJ_PARTICLE_H

#pragma once

#include <headers/zInterface/objects/zObj.h>
#include <headers/zCore/dynamics/zParticle.h>


namespace zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjParticle
	*	\brief The particle object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zObjParticle :public zObj
	{
	private:

		/*! \brief boolean for displaying the particle forces */
		bool displayForces;

		/*! \brief force display scale */
		double forceScale;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief particle */
		zParticle particle;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjParticle();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjParticle();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets display vertices, edges and face booleans.
		*
		*	\param		[in]	_displayForces					- input display forces booelan.
		*	\param		[in]	_forceScale					- input scale of forces.
		*	\since version 0.0.2
		*/
		void setDisplayElements(bool _displayForces, double _forceScale);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		void draw() override;
#endif

	protected:
		//--------------------------
		//---- PROTECTED DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.2
		*/
		void drawForces();

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/** \addtogroup Container
	*	\brief  The container typedef of the library.
	*  @{
	*/

	/*! \typedef zObjParticleArray
	*	\brief A vector of zObjParticle.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjParticle> zObjParticleArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjParticle.cpp>
#endif

#endif