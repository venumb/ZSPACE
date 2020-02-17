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

#ifndef ZSPACE_OBJ_SPATIALBIN_H
#define ZSPACE_OBJ_SPATIALBIN_H

#pragma once

#include <headers/zInterface/objects/zObjPointCloud.h>
#include <headers/zCore/field/zField3D.h>
#include <headers/zCore/field/zBin.h>

#include <vector>
using namespace std;

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

	/*! \class zObjSpatialBin
	*	\brief The spatial binning class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/	

	class ZSPACE_API zObjSpatialBin : public zObjPointCloud
	{
	protected:
		
		/*! \brief boolean for displaying the bins */
		bool displayBounds;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField3D<float> field;	

		/*!	\brief bins			*/
		vector<zBin> bins;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zObjSpatialBin();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.3
		*/
		~zObjSpatialBin();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets display bins booleans.
		*
		*	\param		[in]	_displayBounds				- input display bounds boolean.
		*	\since version 0.0.3
		*/
		void setDisplayBounds(bool _displayBounds);
	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		void draw() override;
#endif

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

	protected:
		//--------------------------
		//---- PROTECTED DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the point cloud.
		*
		*	\since version 0.0.3
		*/
		void drawBins();

		/*! \brief This method displays the bounds of bins.
		*
		*	\since version 0.0.3
		*/
		void drawBounds();

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

	/*! \typedef zObjSpatialBinArray
	*	\brief A vector of zObjSpatialBin.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjSpatialBin> zObjSpatialBinArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/
}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjSpatialBin.cpp>
#endif

#endif

