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
#include <headers/zInterface/functionsets/zFnParticle.h>
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

		/*!	\brief container of  form particle objects  */
		vector<zObjParticleArray> formParticlesObj;		

		/*!	\brief container of form particle function set  */
		vector<vector<zFnParticle>> fnFormParticles;

		/*!	\brief DISCRIPTION  */
		zUtilsCore coreUtils;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray convexHullMeshes;

		/*!	\brief DISCRIPTION  */
		zObjMeshArray dualMeshes;

		/*!	\brief DISCRIPTION  */
		vector<zIntPairArray> c_graphHalfEdge_dualCellFace;

		vector<zVectorArray> dualCellFace_NormTargets;

		vector<zDoubleArray> dualCellFace_AreaTargets;

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

		zDomainDouble deviations[2];

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
		//---- UPDATE METHODS
		//--------------------------

		bool equilibrium(bool &compTargets, double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = EPS, double areaTolerance = EPS, bool printInfo = false);


#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else

		//--------------------------
		//---- DISPLAY SET METHODS
		//--------------------------

		/*! \brief This method sets the display model not for Unreal.
		*
		*	\param		[in]	_index				- input housing unit index
		*	\since version 0.0.4
		*/
		void setDisplayModel(zModel&_model);

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

	protected:
		
		/*!	\brief DISCRIPTION  */
			zModel *model;
#endif

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
		//---- PRIVATE COMPUTE / UPDATE METHODS
		//--------------------------
		   	
		void computeTargets();

		void updateDual(double &dT, zIntergrationType &type, int &numIterations);

		bool checkParallelity(zDomainDouble &deviations, double &angleTolerance,  bool &printInfo);

		bool checkArea(zDomainDouble &deviations, double &areaTolerance, bool &printInfo);

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
		void cleanConvexHull(zItGraphVertex &_vIt);

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