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

#ifndef ZSPACE_OBJ_POINTCLOUD_H
#define ZSPACE_OBJ_POINTCLOUD_H

#pragma once

#include <headers/zInterface/objects/zObj.h>
#include <headers/zCore/geometry/zPointCloud.h>

namespace zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjPointCloud
	*	\brief The point cloud object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zObjPointCloud :public zObj
	{
	protected:

		/*! \brief boolean for displaying the vertices */
		bool displayVertices;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief point cloud */
		zPointCloud pCloud;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjPointCloud();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjPointCloud();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets display vertices booleans.
		*
		*	\param		[in]	_displayVerts				- input display vertices boolean.
		*	\since version 0.0.2
		*/
		void setDisplayElements(bool _displayVerts);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		void draw() override;
#endif

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends pointcloud to the buffer.
		*
		*	\since version 0.0.1
		*/
		void appendToBuffer();

	protected:
		//--------------------------
		//---- PROTECTED DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the point cloud.
		*
		*	\since version 0.0.2
		*/
		void drawPointCloud();
	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/** \addtogroup Container
	*	\brief  The container typedef of the library.
	*  @{
	*/

	/*! \typedef zObjPointCloudArray
	*	\brief A vector of zObjPointCloud.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjPointCloud> zObjPointCloudArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjPointCloud.cpp>
#endif

#endif