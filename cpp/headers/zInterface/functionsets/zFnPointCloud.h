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

#ifndef ZSPACE_FN_POINTCLOUD_H
#define ZSPACE_FN_POINTCLOUD_H

#pragma once

#include<headers/zInterface/objects/zObjPointCloud.h>
#include<headers/zInterface/functionsets/zFn.h>
#include<headers/zInterface/iterators/zItPointCloud.h>

namespace zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnPointCloud
	*	\brief A point cloud function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zFnPointCloud : public zFn
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a mesh object  */
		zObjPointCloud *pointsObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnPointCloud();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\since version 0.0.2
		*/
		zFnPointCloud(zObjPointCloud &_pointsObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnPointCloud();

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		zFnType getType() override;

		void from(string path, zFileTpye type, bool staticGeom = false) override;

		void to(string path, zFileTpye type) override;

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		void clear() override;

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a mesh from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\since version 0.0.1
		*/
		void create(zPointArray(&_positions));

		//--------------------------
		//---- APPEND METHODS
		//--------------------------

		/*! \brief This method adds the input point to the point cloud.
		*	\param		[in]	_position		- input position to be added.
		*	\since version 0.0.2
		*/
		void addPosition(zPoint &_position);

		/*! \brief This method adds the input point container to the point cloud.
		*	\param		[in]	_positions		- input container positions to be added.
		*	\since version 0.0.2
		*/
		void addPositions(zPointArray &_positions);

		//--------------------------
		//---- QUERY METHODS
		//--------------------------

		/*! \brief This method returns the number of points in the pointcloud.
		*	\return				number of points.
		*	\since version 0.0.2
		*/
		int numVertices();

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets point color of all the point with the input color.
		*
		*	\param		[in]	col				- input color.
		*	\since version 0.0.2
		*/
		void setVertexColor(zColor col);

		/*! \brief This method sets point color of all the point with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of points in the point cloud.
		*	\since version 0.0.2
		*/
		void setVertexColors(zColorArray& col);

		//--------------------------
		//--- GET METHODS 
		//--------------------------

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexPositions(zPointArray& pos);

		/*! \brief This method gets pointer to the internal vertex positions container.
		*
		*	\return				zVector*					- pointer to internal vertex position container.
		*	\since version 0.0.2
		*/
		zPoint* getRawVertexPositions();

		/*! \brief This method gets vertex color of all the vertices.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexColors(zColorArray& col);

		/*! \brief This method gets pointer to the internal vertex color container.
		*
		*	\return				zColor*					- pointer to internal vertex color container.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColors();

		//--------------------------
		//---- TRANSFORM METHODS OVERRIDES
		//--------------------------


		void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) override;

		void setScale(zFloat4 &scale) override;

		void setRotation(zFloat4 &rotation, bool appendRotations = false) override;

		void setTranslation(zVector &translation, bool appendTranslations = false) override;

		void setPivot(zVector &pivot) override;

		void getTransform(zTransform &transform) override;


	protected:

		//--------------------------
		//---- PROTECTED OVERRIDE METHODS
		//--------------------------	

		void transformObject(zTransform &transform) override;

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method imports a point cloud from an TXT file.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		void fromCSV(string infilename);

		/*! \brief This method exports the input point cloud to a TXT file format.
		*
		*	\param [in]		inPositions			- input container of position.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		void toCSV(string outfilename);

	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnPointCloud.cpp>
#endif

#endif