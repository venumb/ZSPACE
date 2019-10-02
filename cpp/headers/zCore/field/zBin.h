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

#ifndef ZSPACE_BIN_H
#define ZSPACE_BIN_H

#pragma once

#include<headers/zCore/base/zVector.h>
#include<headers/zCore/base/zColor.h>

namespace zSpace
{

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/


	/** \addtogroup zFields
	*	\brief The field classes of the library.
	*  @{
	*/

	/*! \class zBin
	*	\brief A class to hold vertex indicies inside a bin .
	*	\since version 0.0.3
	*/

	/** @}*/
	/** @}*/


	class ZSPACE_CORE zBin
	{
	protected:
		/*!	\brief container of vertex indicies inside the bin per object */
		vector<vector<int>> ids;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.3
		*/
		zBin();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.3
		*/
		~zBin();

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method adds an object to the bin.
		*
		*	\since version 0.0.3
		*/
		void  addObject();

		/*! \brief This method adds the input vertex index of the input object to the bin.
		*
		*	\param		[in]	vertexId		- input vertexId.
		*	\param		[in]	objectId		- input objectId.
		*	\since version 0.0.3
		*/
		void addVertexIndex(int vertexId, int objectId);

		/*! \brief This method clears the vertex indicies container
		*
		*	\since version 0.0.3
		*/
		void clear();

		/*! \brief This method returns if the bin contains any index or not.
		*
		*	\return				bool		- true if the bin cntains atleat one vertex index, else false.
		*	\since version 0.0.3
		*/
		bool contains();

	};


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/field/zBin.cpp>
#endif

#endif