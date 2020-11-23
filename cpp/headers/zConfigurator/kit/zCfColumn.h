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

#ifndef ZSPACE_CONFIG_COLUMN_H
#define ZSPACE_CONFIG_COLUMN_H



#pragma once

#include <headers/zConfigurator/kit/zCfKit.h>


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

	/*! \class zCfColumn
	*	\brief A kit for column.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_CF zCfColumn : public zCfKit
	{


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
		zCfColumn();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zCfColumn();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void create(zTransformationMatrix& _transform, string type) override;

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the column from the input files.
		*
		*	\param [in]		_transform		- input transformation matrix.
		*	\param [in]		offsetX			- input offset in x axis.
		* 	\param [in]		offsetY			- input offset in y axis.
		*	\since version 0.0.4
		*/
		void createSplitColumn(zTransformationMatrix& _transform, float offsetX, float offsetY);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include <source/zConfigurator/kit/zCfColumn.cpp>
#endif

#endif
