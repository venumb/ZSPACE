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

#ifndef ZSPACE_CONFIG_VOXEL_H
#define ZSPACE_CONFIG_VOXEL_H



#pragma once

#include <headers/zCore/data/zDatabase.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zInterOp/include/zRhinoInclude.h>

#include <headers/zConfigurator/base/zCfCommands.h>
#include <headers/zConfigurator/unit/zCfUnit.h>

namespace zSpace
{
	/** \addtogroup zConfigurator
	*	\brief Collection of tool sets for configurator.
	*  @{
	*/

	/** \addtogroup base
	*	\brief base tool sets for configurator.
	*  @{
	*/

	/*! \class zCfVoxels
	*	\brief A toolset for voxels.
	*	\details 
	*	\since version 0.0.4
	*/
	

	/** @}*/

	/** @}*/
	class ZSPACE_CF zCfVoxels
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- VOXEL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to center line graph Object  */
		zObjGraph o_centerline;

		/*!	\brief pointer container to voxel Objects  */
		zObjMeshArray o_voxels;

		/*!	\brief pointer container to booleans for voxel program :  true - interior, false - exterior  */
		zBoolArray voxelInteriorProgram;

		/*!	\brief number of voxels  */
		int n_voxels;

		/*!	\brief voxelDimensions  */
		zVector voxelDims;

		/*!	\brief global number of faces  */
		int global_n_f = 0;

		/*!	\brief global number of internal faces  */
		int global_n_f_i = 0; 

		unordered_map <string, int> voxelFace_GlobalFace;
		vector< zIntArray > globalFace_VoxelFace;
		zBoolArray Global_BoundaryFace;
		zIntArray global_internalFaceIndex; // -1 for external faces 
		zIntArray internalFaceIndex_globalFace;

		//--------------------------
		//---- SETOUT ATTRIBUTES
		//--------------------------

		/*!	\brief global setout point  */
		zPoint globalSetoutAnchor;
		

		/*!	\brief global setout point  */
		zIntArray voxelSetoutFaceId;

		//--------------------------
		//---- DATABASE ATTRIBUTES
		//--------------------------

		/*!	\brief database needed to acces the data.  */
		zDatabase* db;
		

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		vector<zCfUnit> units;

		//--------------------------
		//---- HISTORY ATTRIBUTES
		//--------------------------

		/*!	\brief container of voxel history  */
		vector<zStringArray> voxelHistory;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zCfVoxels();


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zCfVoxels();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets global setout Anchor.
		*	
		*	\param		[in]	_anchor			- input global anchor point.
		*	\since version 0.0.4
		*/
		void setGlobalSetoutAnchor(zPoint _anchor = zPoint(0, 0, 1));

		/*! \brief This method sets SQLite database.
		*
		*	\param		[in]	DatabaseFileName			- input file path to the SQL database.
		*	\since version 0.0.4
		*/
		void setDatabase(char* DatabaseFileName);

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the voxels from the input files.
		*
		*	\param [in]		filePaths		- input container of file paths.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.4
		*/
		void createVoxelsFromFiles(zStringArray filePaths, zFileTpye type);

		/*! \brief This method creates the voxels from the input files.
		*
		*	\param [in]		filePaths		- input container of file paths.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.4
		*/
		void createVoxelsFromTransforms(vector<zTransformationMatrix> &unitTransforms, zStringArray &unitAttributes,  zVector &_voxelDims);

		/*! \brief This method creates the center line graph based on the input voxels.
		*
		*	\since version 0.0.4
		*/
		void createCenterLineFromVoxels(zColor edgeCol = zColor(0.75, 0.5, 0, 1));

	
		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method returns the number of voxels .
		* 
		*	\return				int - number of voxels.
		*	\since version 0.0.4
		*/
		int numVoxels();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets pointer to the internal centerline graph object.
		*
		*	\return				zObjGraph*					- pointer to internal centerline graph object.
		*	\since version 0.0.4
		*/
		zObjGraph* getRawGraph();

		/*! \brief This method gets pointer to the internal voxel mesh object at the input index.
		*
		*	\param [in]		id				- input voxel id.
		*	\return			zObjMesh*		- pointer to internal voxel mesh object if it exists.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawVoxel(int id);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method computes global indicies of faces, edges and vertices of the primal, required to build the connectivity matrix.
		*
		*	\param		[in]	edgeCol						- input edge color for the form.
		*	\since version 0.0.4
		*/
		void computeGlobalElementIndicies( int precisionFac = 6);

		/*! \brief This method gets the face-cell connectivity matrix of the global. It corresponds to the edge-vertex connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.4
		*/
		bool getGlobal_FaceCellMatrix( zSparseMatrix& out);

		/*! \brief This method computes the setout face of each voxel.
		*
		*	\param		[in]	fNorm						- input vector orientation for setout face normal
		*	\since version 0.0.4
		*/
		void computeVoxelSetoutFace(zVector fNorm = zVector(0, 0, -1));

		/*! \brief This method computes the voxel create history.
		*
		*	\since version 0.0.4
		*/
		void computeVoxelHistory();

		/*! \brief This method computes the number of voxels from transforms and voxel dimensions.
		*
		* *	\return				int - number of voxels.
		*	\since version 0.0.4
		*/
		int computeNumVoxelfromTransforms(vector<zTransformationMatrix>& transforms);

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include <source/zConfigurator/base/zCfVoxels.cpp>
#endif

#endif