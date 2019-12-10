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

#ifndef ZSPACE_TS_GEOMETRY_GRAPH_POLYHEDRA_H
#define ZSPACE_TS_GEOMETRY_GRAPH_POLYHEDRA_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/model/zModel.h>

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

	/*! \class zTsGraphPolyhedra
	*	\brief A function set to convert graph data to polyhedra.
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsGraphPolyhedra
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to graph Object  */
		zObjGraph *graphObj;

		/*!	\brief DISCRIPTION  */
		zModel *model;

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief DISCRIPTION  */
		zUtilsDisplay display;

		/*!	\brief DISCRIPTION  */
		zItGraphVertexArray sortedGraphVertices;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray conHullCol;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray dualMeshCol;

		/*!	\brief DISCRIPTION  */
		vector<zIntPairArray> c_graphEdge_dualCellFace;

		/*!	\brief DISCRIPTION  */
		vector<pair<zPoint, zPoint>> dualConnectivityLines;

		/*!	\brief DISCRIPTION  */
		zIntArray nodeId;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnGraph fnGraph;

		// TMP!!
		int snapSteps = 0;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zTsGraphPolyhedra();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.4
		*/
		zTsGraphPolyhedra(zObjGraph &_graphObj, zModel &_model);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zTsGraphPolyhedra();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void createGraphFromFile(string &_path, zFileTpye _type, bool _staticGeom = false);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void createGraphFromMesh(zObjMesh &_inMeshObj, zVector &_verticalForce);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void create();

		//--------------------------
		//---- DRAW METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void setDisplayGraphElements(bool _drawGraph, bool _drawVertIds = false, bool _drawEdgeIds = false);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void setDisplayHullElements(bool _drawConvexHulls, bool _drawFaces = true, bool _drawVertIds = false, bool _drawEdgeIds = false, bool _drawFaceIds = false);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void setDisplayPolyhedraElements(bool _drawDualMesh, bool _drawFaces = true, bool _drawVertIds = false, bool _drawEdgeIds = false, bool _drawFaceIds = false);

	private:
		//--------------------------
		//---- PRIVATE CREATE METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void createDualMesh(zItGraphVertex &_graphVertex);
		   		 
		//--------------------------
		//---- PRIVATE UTILITY METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void sortGraphVertices(zItGraphVertexArray &_graphVertices);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void cleanConvexHull(zItGraphVertex &_vIt, int _maxPolygons, zPointArray &_hullPts);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void colorDualFaceConnectivity();

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void snapDualCells(zItGraphVertexArray &bsf, zItGraphVertexArray &gCenters);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsGraphPolyhedra.cpp>
#endif

#endif