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

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief DISCRIPTION  */
		zUtilsDisplay display;

		/*!	\brief DISCRIPTION  */
		vector<zIntArray> graphEdge_DualCellFace;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to graph Object  */
		zObjGraph *graphObj;

		/*!	\brief form function set  */
		zFnGraph fnGraph;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray conHullsCol;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray graphMeshCol;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray dualMeshCol;

		/*!	\brief DISCRIPTION  */
		zObjGraphArray dualGraphCol;

		/*!	\brief DISCRIPTION  */
		zIntArray internalVertexIds;

		/*!	\brief DISCRIPTION  */
		int firstInternalVertexId;

		/*!	\brief DISCRIPTION  */
		int n_nodes;

		/*!	\brief DISCRIPTION  */
		zPointArray cellCenters;

		bool drawDualMeshFaces;

		vector<zPointArray> dualFaceCenter;

		//TMP
		int snap = 0;
		zVectorArray tmp1;
		zVectorArray tmp2;



	
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
		zTsGraphPolyhedra(zObjGraph &_graphObj);


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
		void create();


		//--------------------------
		//---- DRAW METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void drawGraph(bool drawIds = false);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void drawConvexHulls();


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void drawGraphMeshes();


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void drawDual(bool drawDualMeshFaces = true, bool drawIds = false);


	private:

		//--------------------------
		//---- PRIVATE CREATE METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void createDualMesh(zPointArray &positions, int numEdges, zSparseMatrix &C_ev, zSparseMatrix &C_fc, zObjMesh &dualMesh);


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void createDualGraph(zPointArray &positions, int numEdges, zSparseMatrix &C_fc, zObjGraph &dualGraph);


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void getCellCenter(zItGraphVertex &graphVertIt); //not implemented yet


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void getContainedFace(); // not impolemented yet


		//--------------------------
		//---- PRIVATE UTILITY METHODS
		//--------------------------

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void getInternalVertex();


		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void cyclicSort(zIntArray &unsorted, zPoint &cen,zPointArray &pts, zVector refDir, zVector normal);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void snapDualCells(zItGraphVertexArray &bsf, zItGraphVertexArray &gCenters);

		/*! \brief DISCRIPTION
		*
		*	\since version 0.0.4
		*/
		void drawDualFaceConnectivity();


	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsGraphPolyhedra.cpp>
#endif

#endif