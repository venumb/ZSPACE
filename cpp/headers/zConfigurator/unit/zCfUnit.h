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

#ifndef ZSPACE_CONFIG_UNIT_H
#define ZSPACE_CONFIG_UNIT_H



#pragma once

#include <headers/zCore/data/zDatabase.h>

#include <headers/zConfigurator/kit/zCfColumn.h>


namespace zSpace
{
	/** \addtogroup zConfigurator
	*	\brief Collection of tool sets for configurator.
	*  @{
	*/

	/** \addtogroup Unit
	*	\brief Unit tool sets for configurator.
	*  @{
	*/

	/*! \class zCfUnit
	*	\brief A toolset for units.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_CF zCfUnit
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

		/*!	\brief pointer to center line graph object  */
		zObjGraph* centerlineObj;

		/*!	\brief pointer container to voxel objects  */
		zObjMeshPointerArray voxelObjs;
		
		/*!	\brief pointer container to booleans for voxel program :  true - interior, false - exterior  */
		zBoolArray* voxelInteriorProgram;

		//--------------------------
		//---- UNIT ATTRIBUTES
		//--------------------------

		/*!	\brief ids of voxels beloing to the unit  */
		zIntArray voxelIDS;

		/*!	\brief pointer to unit transformation matrix  */
		zTransformationMatrix* unitTransform;

		/*!	\brief unit spaceplan object */
		zObjMesh unitSpacePlan;

		/*!	\brief unit type  */
		string unitType; 

		/*!	\brief unit space plan type  */
		string unitSpacePlanType;

		/*!	\brief unit ID  */
		string unitID;

		/*!	\brief combined voxel mesh object  */
		zObjMesh combinedVoxelObj;

		//--------------------------
		//---- DIMENSION ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to voxel dimensions  */
		zVector* voxelDims;

		/*!	\brief wall panel built up thickness  */
		float wallPanel_width = 0.1;

		/*!	\brief primary column width  */
		float column_primary_width = 0.1;

		/*!	\brief primary column length  */
		float column_primary_length = 0.1;

		/*!	\brief primary beam width  */
		float beam_width = 0.1;

		/*!	\brief beam height  */
		float beam_height = 0.3;

		//--------------------------
		//---- DATABASE ATTRIBUTES
		//--------------------------

		/*!	\brief database needed to acces the data.  */
		zDatabase* db;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief container of column objects.  */
		vector<zCfColumn> primaryColumns;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zCfUnit();

			//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zCfUnit();


		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the combined voxel mesh from the individual voxels of the unit.
		*
		*	\since version 0.0.4
		*/
		void createCombinedVoxelMesh();

		/*! \brief This method creates the combined voxel mesh from the individual voxels of the unit.
		*
		*  	\param		[in]	xSubdiv		- input number of subdivisions in x.
		*  	\param		[in]	ySubdiv		- input number of subdivisions in y.
		*	\since version 0.0.4
		*/
		void createSpacePlanMesh(int xSubdiv = 3, int ySubdiv = 3);

		/*! \brief This method creates the combined voxel mesh from the individual voxels of the unit.
		*
		*	\since version 0.0.4
		*/
		void createPrimaryColumns();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the voxel objects.
		*
		*	\param		[in]	_voxelObjs			- input container of voxel objects.
		*	\since version 0.0.4
		*/
		void setVoxels(zObjMeshArray& _voxelObjs);

		/*! \brief This method sets the centerline graph object.
		*
		*	\param		[in]	_centerLineObj		- input center line graph object.
		*	\since version 0.0.4
		*/
		void setCenterlineGraph(zObjGraph& _centerLineObj);

		/*! \brief This method sets the centerline graph object.
		*
		*	\param		[in]	_voxelIDS			- input container of voxelIds of the unit.
		*	\since version 0.0.4
		*/
		void setVoxelIds(zIntArray& _voxelIDS);

		/*! \brief This method sets the voxel interiro program array.
		*
		*	\param		[in]	voxelInteriorProgram			- input container of booleans indicating interior / exterior program of the unit.
		*	\since version 0.0.4
		*/
		void setVoxelInteriorProgram(zBoolArray& _voxelInteriorProgram);

		/*! \brief This method sets the transformation matrix of the unit.
		*
		* 	\param		[in]	_unitTransform		- input unit transformation matrix.
		*	\since version 0.0.4
		*/
		void setUnitTransform(zTransformationMatrix& _unitTransform);

		/*! \brief This method sets the voxel dimensions of the unit.
		*
		*  	\param		[in]	_voxelDims			- input voxel dimensions.
		*	\since version 0.0.4
		*/
		void setVoxelDimensions(zVector& _voxelDims);

		/*! \brief This method sets the string of the unit.
		*
		*	\param		[in]	_unitStrings		- input string of unit ID, type, spaceplan type etc. ( eg: 0_base_sp1)
		*	\since version 0.0.4
		*/
		void setUnitString(string& _unitString);

		/*! \brief This method sets the database of the unit.
		*
		*	\param		[in]	_db		- input database.
		*	\since version 0.0.4
		*/
		void setDatabase(zDatabase& _db);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets pointer to the internal combined voxel mesh object.
		*
		*	\return				zObjMesh*					- pointer to internal combined voxel meshobject.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawCombinedVoxelMesh();

		/*! \brief This method gets pointer to the internal space plan mesh object.
		*
		*	\return				zObjMesh*					- pointer to internal space plan meshobject.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawSpacePlanMesh();

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

	private:

		/*! \brief This method computes the unit attributes.
		*
		*	\param		[in]	_unitString		- input string. ( eg: 0_base_sp1)
		*	\since version 0.0.4
		*/
		void computeUnitAttributes(string &_unitString);


	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include <source/zConfigurator/unit/zCfUnit.cpp>
#endif

#endif