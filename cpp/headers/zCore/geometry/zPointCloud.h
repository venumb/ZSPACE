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

#ifndef ZSPACE_POINTCLOUD_H
#define ZSPACE_POINTCLOUD_H

#pragma once

#include <headers/zCore/base/zVector.h>
#include <headers/zCore/utilities/zUtilsCore.h>
#include <headers/zCore/geometry/zHEGeomTypes.h>

#include <headers/zCore/base/zTypeDef.h>

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zPointCloud
	*	\brief A point cloud class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zPointCloud
	{
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of active vertices */
		int n_v;

		/*!	\brief vertex container */
		zVertexArray vertices;

		/*!	\brief container which stores vertex positions. 	*/
		zPointArray vertexPositions;

		/*!	\brief container which stores vertex colors. 	*/
		zColorArray vertexColors;

		/*!	\brief container which stores vertex weights. 	*/
		zDoubleArray vertexWeights;

		/*!	\brief position to vertexId map. Used to check if vertex exists with the haskey being the vertex position.	 */
		unordered_map <string, int> positionVertex;

		/*!	\brief stores the start vertex ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_VertexId;

		/*!	\brief stores the start vertex color ID in the VBO, when attache to the zBufferObject.	*/
		int VBO_VertexColorId;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zPointCloud();		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zPointCloud();
			   
		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the the point cloud from the input contatiners.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\since version 0.0.2
		*/
		void create(zPointArray &_positions);

		/*! \brief This methods clears all the graph containers.
		*
		*	\since version 0.0.2
		*/
		void clear();

		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zPoint &pos);

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\return					bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zPoint &pos, int &outVertexId, int precisionfactor = 6);

		/*! \brief This method sets the number of vertices in zGraph  the input value.
		*	\param		[in]		_n_v	- number of vertices.
		*	\since version 0.0.1
		*/
		void setNumVertices(int _n_v);

		//--------------------------
		//---- MAP METHODS
		//--------------------------

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/
		void addToPositionMap(zPoint &pos, int index, int precisionfactor = 6);

		/*! \brief This method removes the position given by input vector from the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/
		void removeFromPositionMap(zPoint &pos, int precisionfactor = 6);

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/geometry/zPointCloud.cpp>
#endif

#endif