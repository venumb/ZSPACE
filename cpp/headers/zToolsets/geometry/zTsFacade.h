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

#ifndef ZSPACE_TS_GEOMETRY_FACADE_H
#define ZSPACE_TS_GEOMETRY_FACADE_H



#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zInterface/functionsets/zFnMeshField.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


namespace zSpace
{
	
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	
	/*! \class zTsFacade
	*	\brief A toolset for Facades.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsFacade
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to input guide mesh Object  */
		zObjMesh* o_guideMesh;

	
				
		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*!	\brief container of indicies of fixed vertices  */
		vector<int> fixedVertices;

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> fixedVerticesBoolean;

		/*!	\brief container of face principal curvatures  */
		zCurvatureArray faceCurvatures;
		

		//--------------------------
		//---- GUIDE MESH ATTRIBUTES
		//--------------------------

		/*!	\brief container of  guide particle objects  */
		vector<zObjParticle> o_guideParticles;

		/*!	\brief container of guide particle function set  */
		vector<zFnParticle> fnGuideParticles;	
			

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zTsFacade();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zTsFacade();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the guide mesh object.
		*
		*	\param		[in]	_o_guideMesh			- input guide mesh object.
		*	\since version 0.0.4
		*/
		void setGuideMesh(zObjMesh& _o_guideMesh);
		

		//--------------------------
		//---- GET METHODS
		//--------------------------

	

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------

		/*! \brief This method computes the curvature of the guide mesh faces.
		*
		*	\param		[in]	colFaces				- input block index.
		*	\return				bool					- true if boundary block else false.
		*	\since version 0.0.4
		*/
		void computeFaceCurvature(bool colFaces = true, zDomainColor colDomain = zDomainColor());
		
		//--------------------------
		//---- UTILITY METHODS
		//--------------------------


		//--------------------------
		//---- IO METHODS
		//--------------------------
			
	
		
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsFacade.cpp>
#endif

#endif