#pragma once
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

#ifndef ZSPACE_CONFIG_KIT_H
#define ZSPACE_CONFIG_KIT_H



#pragma once

#include <headers/zCore/data/zDatabase.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <headers/zInterOp/include/zRhinoInclude.h>

#include <headers/zConfigurator/base/zCfCommands.h>
#include <headers/zConfigurator/base/zCfEnumerators.h>

namespace zSpace
{
	/** \addtogroup zConfigurator
	*	\brief Collection of tool sets for configurator.
	*  @{
	*/

	/** \addtogroup Kit
	*	\brief Kit tool sets for configurator.
	*  @{
	*/

	/*! \class zCfKit
	*	\brief A kit object.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_CF zCfKit
	{

	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief transformation matrix  */
		zTransformationMatrix transform;

		/*!	\brief kit component object  */
		zObjMesh o_component;

		/*!	\brief kit component type  */
		zCfComponentType componentType;

		/*!	\brief kit component ID  */
		string componentID;

		/*!	\brief component Unit ID  */
		string unitID;

		/*!	\brief component Voxel ID  */
		string voxelID;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zCfKit();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zCfKit();

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------

		/*! \brief This method creates the kit object.
		*
		*	\param [in]		_transform			- input transformation matrix.
		*	\param [in]		type				- type of kit component.
		*	\since version 0.0.4
		*/
		virtual void create(zTransformationMatrix &_transform , string type) = 0;

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets pcomponent ID.
		*
		*	\param [in]		_componentId	- input component id;
		*	\since version 0.0.4
		*/
		void setComponentID(string &_componentID);

		/*! \brief This method sets pcomponent ID.
		*
		*	\param	[in]	_unitID	- input unit id;
		*	\since version 0.0.4
		*/
		void setUnitID(string& _unitID);

		/*! \brief This method sets pcomponent ID.
		*
		*	\param	[in]	_voxelID	- input voxel id;
		*	\since version 0.0.4
		*/
		void setVoxelID(string& _voxelID);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets pointer to the internal component mesh object.
		*
		*	\return				zObjMesh*					- pointer to internal component mesh object.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawComponentMesh();

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include <source/zConfigurator/kit/zCfKit.cpp>
#endif

#endif
