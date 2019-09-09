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
		bool showBounds;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField3D<double> field;	

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

		/*! \brief This method sets show bins booleans.
		*
		*	\param		[in]	_showBounds				- input show bounds boolean.
		*	\since version 0.0.3
		*/
		void setShowBounds(bool _showBounds);
	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override;

		void getBounds(zVector &minBB, zVector &maxBB) override;

		//--------------------------
		//---- DISPLAY METHODS
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

}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjSpatialBin.cpp>
#endif

#endif

