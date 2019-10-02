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

#ifndef ZSPACE_ITERATOR_H
#define ZSPACE_ITERATOR_H

#pragma once

#include<headers/zInterface/objects/zObj.h>
#include<headers/zInterface/functionsets/zFn.h>

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterator classes of the library.
	*  @{
	*/

	/*! \class zIt
	*	\brief The base iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zIt
	{

	protected:
		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;
				
	public:
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zIt();				

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.3
		*/
		~zIt();

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------
				
		/*! \brief This method sets the iterator pointing to first element the contatiner.
		*
		*	\since version 0.0.3
		*/
		virtual void begin() = 0;

		/*! \brief This method advance to the next element in the iteration.
		*
		*	\since version 0.0.3
		*/
		virtual void operator++(int) = 0;

		/*! \brief This method advance to the previous element in the iteration.
		*
		*	\since version 0.0.3
		*/
		virtual void operator--(int) = 0;

		/*! \brief This method indicates if all the elements are tranversed.
		*
		*	\return		bool	- true if all the elements are tranversed.
		*	\since version 0.0.3
		*/
		virtual bool end() = 0;

		/*! \brief This method resets the iterator to the first element.
		*
		*	\since version 0.0.3
		*/
		virtual void reset() {}			

		/*! \brief This method gets the size f the element container.
		*
		*	\return		int	- size of element container.
		*	\since version 0.0.3
		*/
		virtual int size() = 0;

		/*! \brief This method deactivates the element attached to the iterator.
		*
		*	\since version 0.0.3
		*/
		virtual void deactivate() = 0;

	};
}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/iterators/zIt.cpp>
#endif

#endif