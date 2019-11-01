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

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to graph Object  */ //DOES THIS NEED TO BE A POINTER?
		zObjGraph *graphObj;

		

		/*!	\brief form function set  */
		zFnGraph fnGraph;

		zObjMeshArray conHullsCol;

		zObjMeshArray graphMeshCol;

		zObjMeshArray dualMeshCol;

		zObjGraphArray dualGraphCol;

		vector<zIntArray> graphEdge_DualCellFace;

		int n_nodes = 0;

		zPointArray tmpP;

		

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
		//---- UTILITY METHODS
		//--------------------------
		void createGraphMesh();

		void getCellCenter();

		void getContainedFace();

		void drawConvexHulls();

		void drawGraphMeshes();

		void drawDual();

	private:		
		//--------------------------
		//---- ALGEBRAIC METHOD UTILITIES
		//--------------------------

		void createDualMesh(zPointArray &positions, int numEdges, zSparseMatrix &C_ev, zSparseMatrix &C_fc, zObjMesh &dualMesh);

		void createDualGraph(zPointArray &positions, int numEdges, zSparseMatrix &C_fc, zObjGraph &dualGraph);

		/*! \brief This method gets the edge-vertex connectivity matrix of the primal. It corresponds to the face-cell connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_EdgeVertexMatrix(zObjMesh inMesh,  zSparseMatrix &out);


		/*! \brief This method gets the edge-face connectivity matrix of the primal. It corresponds to the face-edge connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_EdgeFaceMatrix(zDiagramType type, zSparseMatrix &out);

		/*! \brief This method gets the face-cell connectivity matrix of the primal. It corresponds to the edge-vertex connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_FaceCellMatrix(zDiagramType type, zSparseMatrix &out);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsGraphPolyhedra.cpp>
#endif

#endif