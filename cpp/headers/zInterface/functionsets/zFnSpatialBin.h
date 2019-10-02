// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Authors : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Federico Borello <federico.borello@zaha-hadid.com>
//

#ifndef ZSPACE_FN_SPATIALBIN_H
#define ZSPACE_FN_SPATIALBIN_H

#pragma once

#include<headers/zInterface/objects/zObjSpatialBin.h>

#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/objects/zObjPointCloud.h>

#include<headers/zInterface/functionsets/zFnPointCloud.h>


namespace  zSpace
{

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/	

	/*! \class zFnSpatialBin
	*	\brief A spatial binning function-set.
	*	\since version 0.0.3
	*/
	
	/** @}*/
	/** @}*/

	class ZSPACE_API zFnSpatialBin
	{
	protected:
		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief stores pointers of objects */
		vector<zObj*> objects;

		/*!	\brief pointer to a field 3D object  */
		zObjSpatialBin *binObj;		

	public:


		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<zIntArray> ringNeighbours;

		/*!	\brief container of adjacent neighbourhood indicies.  */
		vector<zIntArray> adjacentNeighbours;

		/*!	\brief point cloud function set  */
		zFnPointCloud fnPoints;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.3
		*/
		zFnSpatialBin();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_binObj		- input spatial bin object.
		*	\since version 0.0.3
		*/
		zFnSpatialBin(zObjSpatialBin &_binObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.3
		*/
		~zFnSpatialBin();

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		void clear();

		//--------------------------
		//---- CREATE METHOD
		//--------------------------

		/*! \brief This method creates the spatial bins from the input bounds.
		*
		*	\param  	[out]	minBB		- bounding box minimum.
		*	\param		[out]	maxBB		- bounding box maximum.
		*	\param  	[in]	_res		- resolution of bins.
		*	\since version 0.0.3
		*/
		void create(const zPoint &_minBB = zPoint(0, 0, 0), const zPoint &_maxBB = zPoint(10, 10, 10), int _res = 30);

		//--------------------------
		//---- GET METHODS
		//--------------------------

			/*! \brief This method gets the number of objects in the spatial bin.
		*
		*	\return			int	- number of objects in the spatial bin.
		*	\since version 0.0.3
		*/
		int numObjects();

		/*! \brief This method gets the number of bins in the spatial bin.
		*
		*	\return			int	- number of bins in the spatial bin.
		*	\since version 0.0.3
		*/
		int numBins();

		/*! \brief This method gets the ring neighbours of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourhoodRing(int index, int numRings, zIntArray &ringNeighbours);

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourAdjacents(int index, zIntArray &adjacentNeighbours);

		/*! \brief This method gets the index of the bin for the input X,Y and Z indicies.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\param		[in]	index_Y		- input index in Y.
		*	\param		[in]	index_Z		- input index in Z.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(int index_X, int index_Y, int index_Z, int &index);

		/*! \brief This method gets the index of the bin at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(zPoint &pos, int &index);
	
		/*! \brief This method gets the indicies of the bin at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\param		[out]	index_Z		- output index in Z.
		*	\return				bool		- true if position is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndices(zPoint &pos, int &index_X, int &index_Y, int &index_Z);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method adds a new object to the bin 
		*
		*	\param		[in]	inObj			- input object. Works for objects with position informations - pointcloud, graphs and meshes.
		*	\since version 0.0.3
		*/
		template<typename T>
		void addObject(T &inObj);		

		/*! \brief This method clears the bins.
		*
		*	\since version 0.0.3
		*/
		void clearBins();


		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------

	protected:

		/*! \brief This method checks if the bounds of the input object is less than the current bounds of the bins. If not the bins bunds is resized.
		*
		*	\param		[in]	inObj		- input object.
		*	\return				bool		- true if bounds of bins is resized else false.
		*	\since version 0.0.3
		*/
		template<typename T>
		bool boundsCheck(T &inObj);

		/*! \brief This method partitions the  new object to the bin
		*
		*	\param		[in]	inObj			- input object. Works for objects with position informations - pointcloud, graphes and meshes.
		*	\since version 0.0.3
		*/
		template<typename T>
		void partitionToBins(T &inObj, int objectId);

		/*! \brief This method partitions the input position.
		*
		*	\param		[in]	inPos		- input position to be added to bin.
		*	\param		[in]	pointId		- input index of the position in the container.
		*	\param		[in]	objectId	- input index of the object the position belongs to.
		*	\since version 0.0.3
		*/
		void partitionToBin(zPoint &inPos, int pointId, int objectId);

		/*! \brief This method creates the point cloud from the field parameters.
		*
		*	\since version 0.0.3
		*/
		void createPointCloud();
		
	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------


	//---------------//

	//---- pointcloud specilization for addObject

	template<>
	inline bool zFnSpatialBin::boundsCheck(zObjPointCloud &inObj)
	{
		zVector minBB, maxBB;
		binObj->getBounds(minBB, maxBB);

		zVector minBB_Obj, maxBB_Obj;
		inObj.getBounds(minBB_Obj, maxBB_Obj);

		bool resized = false;

		if (minBB_Obj.x < minBB.x) { resized = true; minBB.x = minBB_Obj.x; }
		if (minBB_Obj.y < minBB.y) { resized = true; minBB.y = minBB_Obj.y; }
		if (minBB_Obj.z < minBB.z) { resized = true; minBB.z = minBB_Obj.z; }

		if (maxBB_Obj.x > maxBB.x) { resized = true; maxBB.x = maxBB_Obj.x; }
		if (maxBB_Obj.y > maxBB.y) { resized = true; maxBB.y = maxBB_Obj.y; }
		if (maxBB_Obj.z > maxBB.z) { resized = true; maxBB.z = maxBB_Obj.z; }

		if (resized)
		{
			int resX = ((maxBB.x - minBB.x) / binObj->field.unit_X) + 1;
			int resY = ((maxBB.y - minBB.y) / binObj->field.unit_Y) + 1;
			int resZ = ((maxBB.z - minBB.z) / binObj->field.unit_Z) + 1;

			int res = (resX > resY) ? resX : resY;
			res = (res > resZ) ? res : resZ;

			clear();
			create(minBB, maxBB, res);

			// partition to bin existing objects
			int i = 0;
			for (auto &obj : objects)
			{
				partitionToBins(obj, i);
				i++;
			}

		}

		return resized;
	}

	//---- graph specilization for addObject

	template<>
	inline bool zFnSpatialBin::boundsCheck(zObjGraph &inObj)
	{
		zVector minBB, maxBB;
		binObj->getBounds(minBB, maxBB);

		zVector minBB_Obj, maxBB_Obj;
		inObj.getBounds(minBB_Obj, maxBB_Obj);

		bool resized = false;

		if (minBB_Obj.x < minBB.x) { resized = true; minBB.x = minBB_Obj.x; }
		if (minBB_Obj.y < minBB.y) { resized = true; minBB.y = minBB_Obj.y; }
		if (minBB_Obj.z < minBB.z) { resized = true; minBB.z = minBB_Obj.z; }

		if (maxBB_Obj.x > maxBB.x) { resized = true; maxBB.x = maxBB_Obj.x; }
		if (maxBB_Obj.y > maxBB.y) { resized = true; maxBB.y = maxBB_Obj.y; }
		if (maxBB_Obj.z > maxBB.z) { resized = true; maxBB.z = maxBB_Obj.z; }

		if (resized)
		{
			int resX = ((maxBB.x - minBB.x) / binObj->field.unit_X) + 1;
			int resY = ((maxBB.y - minBB.y) / binObj->field.unit_Y) + 1;
			int resZ = ((maxBB.z - minBB.z) / binObj->field.unit_Z) + 1;

			int res = (resX > resY) ? resX : resY;
			res = (res > resZ) ? res : resZ;

			clear();
			create(minBB, maxBB, res);

			// partition to bin existing objects
			int i = 0;
			for (auto &obj : objects)
			{
				partitionToBins(obj, i);
				i++;
			}

		}

		return resized;
	}

	//---- mesh specilization for addObject

	template<>
	inline bool zFnSpatialBin::boundsCheck(zObjMesh &inObj)
	{
		zVector minBB, maxBB;
		binObj->getBounds(minBB, maxBB);

		zVector minBB_Obj, maxBB_Obj;
		inObj.getBounds(minBB_Obj, maxBB_Obj);

		bool resized = false;

		if (minBB_Obj.x < minBB.x) { resized = true; minBB.x = minBB_Obj.x; }
		if (minBB_Obj.y < minBB.y) { resized = true; minBB.y = minBB_Obj.y; }
		if (minBB_Obj.z < minBB.z) { resized = true; minBB.z = minBB_Obj.z; }

		if (maxBB_Obj.x > maxBB.x) { resized = true; maxBB.x = maxBB_Obj.x; }
		if (maxBB_Obj.y > maxBB.y) { resized = true; maxBB.y = maxBB_Obj.y; }
		if (maxBB_Obj.z > maxBB.z) { resized = true; maxBB.z = maxBB_Obj.z; }

		if (resized)
		{
			int resX = ((maxBB.x - minBB.x) / binObj->field.unit_X) + 1;
			int resY = ((maxBB.y - minBB.y) / binObj->field.unit_Y) + 1;
			int resZ = ((maxBB.z - minBB.z) / binObj->field.unit_Z) + 1;

			int res = (resX > resY) ? resX : resY;
			res = (res > resZ) ? res : resZ;

			clear();
			create(minBB, maxBB, res);

			// partition to bin existing objects
			int i = 0;
			for (auto &obj : objects)
			{
				partitionToBins(obj, i);
				i++;
			}

		}

		return resized;
	}

	//---------------//

	//---- pointcloud specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjPointCloud & inObj, int objectId)
	{
		zVector *positions = &inObj.pCloud.vertexPositions[0];

		for (int i = 0; i < inObj.pCloud.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}

	//---- graph specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjGraph & inObj, int objectId)
	{
		zVector *positions = &inObj.graph.vertexPositions[0];

		for (int i = 0; i < inObj.graph.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}

	//---- mesh specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjMesh & inObj, int objectId)
	{
		zVector *positions = &inObj.mesh.vertexPositions[0];

		for (int i = 0; i < inObj.mesh.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}	

	//---------------//

	//---- pointcloud specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjPointCloud & inObj)
	{

		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);
	
		partitionToBins(inObj, objects.size() - 1);

	}

	//---- graph specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjGraph & inObj)
	{
		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);

		partitionToBins(inObj, objects.size() - 1);
	}

	//---- mesh specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjMesh & inObj)
	{
		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);

		partitionToBins(inObj, objects.size() - 1);
	}

	//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
	
	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnSpatialBin.cpp>
#endif

#endif