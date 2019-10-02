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

#ifndef ZSPACE_ITERATOR_POINTCLOUD_H
#define ZSPACE_ITERATOR_POINTCLOUD_H

#pragma once

#include<headers/zInterface/iterators/zIt.h>
#include<headers/zInterface/objects/zObjPointCloud.h>

namespace zSpace
{
		
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zPointCloudIterators
	*	\brief The point cloud iterator classes of the library.
	*  @{
	*/

	/*! \class zItPointCloudVertex
	*	\brief The point cloud vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItPointCloudVertex : public zIt
	{
	protected:

		zItVertex iter;

		/*!	\brief pointer to a pointcloud object  */
		zObjPointCloud *pointsObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItPointCloudVertex();
	
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\since version 0.0.3
		*/
		zItPointCloudVertex(zObjPointCloud &_pointsObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\param		[in]	_index				- input index in mesh vertex list.
		*	\since version 0.0.3
		*/
		zItPointCloudVertex(zObjPointCloud &_pointsObj, int _index);

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void begin() override;

		void operator++(int) override;

		void operator--(int) override;

		bool end() override;

		void reset() override;		

		int size() override;

		void deactivate() override;
	
		//--------------------------
		//---- GET METHODS
		//--------------------------
		
		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();		

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItVertex		- raw iterator
		*	\since version 0.0.3
		*/
		zItVertex  getRawIter();

		/*! \brief This method gets position of the vertex.
		*
		*	\return				zPoint					- vertex position.
		*	\since version 0.0.3
		*/
		zPoint getPosition();

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\return				zPoint*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zPoint* getRawPosition();

		/*! \brief This method gets color of the vertex.
		*
		*	\return				zColor					- vertex color.
		*	\since version 0.0.3
		*/
		zColor getColor();

		/*! \brief This method gets pointer to the color of the vertex.
		*
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.3
		*/
		zColor* getRawColor();
		
		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id);

		/*! \brief This method sets position of the vertex.
		*
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.3
		*/
		void setPosition(zPoint &pos);

		/*! \brief This method sets color of the vertex.
		*
		*	\param		[in]	col						- vertex color.
		*	\since version 0.0.3
		*/
		void setColor(zColor col);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------
		
		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive();
		

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItPointCloudVertex &other);

		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItPointCloudVertex &other);		

	};
		

	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/iterators/zItPointCloud.cpp>
#endif

#endif