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

#ifndef ZSPACE_MODEL_H
#define ZSPACE_MODEL_H

#pragma once
#include <headers/zInterface/objects/zObj.h>

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) 
	// Do Nothing
#else

namespace zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zViewer
	*	\brief The viewer related classes of the library.
	*  @{
	*/

	/*! \class zModel
	*	\brief The model class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	class ZSPACE_API zModel 
	{
	private:
		//--------------------------
		//----  PRIVATE ATTRIBUTES
		//--------------------------
				

		/*! \brief boolean for displaying the buffer point object colors */
		bool showBufPointColors;
		
		/*! \brief boolean for displaying the buffer object as points */
		bool showBufPoints;

		/*! \brief boolean for displaying the buffer line object colors */
		bool showBufLinesColors;

		/*! \brief boolean for displaying the buffer object as lines */
		bool showBufLines;

		/*! \brief boolean for displaying the buffer triangle object colors */
		bool showBufTrisColors;

		/*! \brief boolean for displaying the buffer object as triangles */
		bool showBufTris;

		/*! \brief boolean for displaying the buffer Quads object colors */
		bool showBufQuadsColors;

		/*! \brief boolean for displaying the buffer object as quads */
		bool showBufQuads;


	public:
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief display utilities object	*/
		zUtilsDisplay displayUtils;

		/*! \brief container of scene objects	*/
		vector<zObj *> sceneObjects;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zModel();

		/*! \brief Overloaded constructor.
		*
		*	\since version 0.0.2
		*/
		zModel(int _buffersize);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zModel();

		//--------------------------
		//---- OBJECT METHODS
		//--------------------------

		/*! \brief This method adds the input object to the scene.
		*
		*	\param		[in]	obj				- input object.
		*	\since version 0.0.2
		*/
		void addObject(zObj &obj);

		/*! \brief This method adds the input container of objects to the scene.
		*
		*	\param		[in]	objs				- input  contatiner of objects.
		*	\since version 0.0.2
		*/
		void addObjects(zObjArray &objs);

		//--------------------------
		//---- DRAW METHODS
		//--------------------------

		/*! \brief This method draws the scene objects.
		*
		*	\since version 0.0.2
		*/
		void draw();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show buffer points boolean.
		*
		*	\param		[in]	_showBufPoints				- input show buffer points boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufPoints(bool _showBufPoints, bool showColors = false);

		/*! \brief This method sets show buffer lines boolean.
		*
		*	\param		[in]	_showBufLines				- input show buffer line boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufLines(bool _showBufLines, bool showColors = false);

		/*! \brief This method sets show buffer triangles boolean.
		*
		*	\param		[in]	_showBufTris				- input show buffer tris boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufTris(bool _showBufTris, bool showColors = false);

		/*! \brief This method sets show buffer quads boolean.
		*
		*	\param		[in]	_showBufQuads				- input show buffer quads boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufQuads(bool _showBufQuads, bool showColors = false);

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/model/zModel.cpp>
#endif

#endif

#endif