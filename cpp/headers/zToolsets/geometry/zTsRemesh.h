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

#ifndef ZSPACE_TS_GEOMETRY_REMESH_H
#define ZSPACE_TS_GEOMETRY_REMESH_H

#pragma once
#include <headers/zInterface/functionsets/zFnMesh.h>

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

	/*! \class zTsRemesh
	*	\brief A remesh tool set class for remeshing triangular meshes.
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/


	class ZSPACE_TOOLS zTsRemesh 
	{
	protected:		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnMesh fnMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsRemesh();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsRemesh(zObjMesh &_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsRemesh();

		//--------------------------
		//---- REMESH METHODS
		//--------------------------

		/*! \brief This method splits an edge longer than the given input value at its midpoint and  triangulates the mesh. the adjacent triangles are split into 2-4 triangles.
		*
		*	\param		[in]	maxEdgeLength	- maximum edge length.
		*	\since version 0.0.2
		*/
		void splitLongEdges(double maxEdgeLength);

		/*! \brief This method collapses an edge shorter than the given minimum edge length value if the collapsing doesnt produce adjacent edges longer than the maximum edge length.
		*
		*	\param		[in]	minEdgeLength		- minimum edge length.
		*	\param		[in]	maxEdgeLength		- maximum edge length.
		*	\since version 0.0.2
		*/
		void collapseShortEdges(double minEdgeLength, double maxEdgeLength);

		/*! \brief This method equalizes the vertex valences by flipping edges of the input triangulated mesh. Target valence for interior vertex is 4 and boundary vertex is 6.
		*
		*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
		*	\since version 0.0.2
		*/
		void equalizeValences();

		/*! \brief This method applies an iterative smoothing to the mesh by  moving the vertex but constrained to its tangent plane.
		*
		*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
		*	\since version 0.0.2
		*/
		void tangentialRelaxation();
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsRemesh.cpp>
#endif

#endif