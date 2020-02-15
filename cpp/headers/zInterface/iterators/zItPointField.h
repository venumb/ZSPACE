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

#ifndef ZSPACE_ITERATOR_POINTFIELD_H
#define ZSPACE_ITERATOR_POINTFIELD_H

#pragma once

#include<headers/zInterface/iterators/zItPointCloud.h>
#include<headers/zInterface/objects/zObjPointField.h>

namespace zSpace
{
	class ZSPACE_API zItPointScalarField;
	class ZSPACE_API zItPointVectorField;

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

	/*! \typedef zItPointScalarFieldArray
	*	\brief A vector of zItPointScalarField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItPointScalarField> zItPointScalarFieldArray;

	/*! \typedef zItPointVectorFieldArray
	*	\brief A vector of zItPointVectorField.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zItPointVectorField> zItPointVectorFieldArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zPointFieldIterators
	*	\brief The 3D point field iterator classes of the library.
	*  @{
	*/

	/*! \class zItPointScalarField
	*	\brief The 3D point vector field iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_API zItPointScalarField : public zIt
	{
	protected:

		std::vector<float>::iterator iter;

		/*!	\brief pointer to a mesh fieldobject  */
		zObjPointScalarField  *fieldObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItPointScalarField();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\since version 0.0.3
		*/
		zItPointScalarField(zObjPointScalarField &_fieldObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_index				- input index in field values container.
		*	\since version 0.0.3
		*/
		zItPointScalarField(zObjPointScalarField &_fieldObj, int _index);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_pos				- input positionn.
		*	\param		[in]	closestIndex		- true if closest index is required.
		*	\since version 0.0.3
		*/
		zItPointScalarField(zObjPointScalarField &_fieldObj, zVector &_pos, bool closestIndex = false);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	pos			- input position.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		zItPointScalarField(zObjPointScalarField &_fieldObj, int _index_X, int _index_Y, int _index_Z);

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
		//--- FIELD TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the ring neighbours of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Ring(int numRings, zItPointScalarFieldArray &ringNeighbours);

		/*! \brief This method gets the ring neighbour indicies of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Ring(int numRings, zIntArray &ringNeighbours);

		/*! \brief This method gets the ring Points of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours);

		/*! \brief This method gets the immediate adjacent neighbours of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(zItPointScalarFieldArray &adjacentNeighbours);

		/*! \brief This method gets the immediate adjacent neighbour indicies of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(zIntArray &adjacentNeighbours);

		/*! \brief This method gets the immediate adjacent Points  of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of adjacent neighbour positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours);

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\return				zVector		- field position.
		*	\since version 0.0.2
		*/
		zVector getPosition();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the indices of the iterator.
		*
		*	\param		[out]	index_X		- output index X
		*	\param		[out]	index_Y		- output index Y
		*	\param		[out]	index_Z		- output index Z
		*	\since version 0.0.3
		*/
		void getIndices(int &index_X, int& index_Y, int& index_Z);

		/*! \brief This method gets the value of the field at the current element.
		*
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\param		[out]	val			- field value.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		zScalar getValue();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the value of the field at the current element.
		*
		*	\param		[in]	val			- field value.
		*	\param		[in]	append		- value if added to existing if true.
		*	\since version 0.0.2
		*/
		void setValue(zScalar val, bool append = false);

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItPointScalarField &other);

		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItPointScalarField &other);

	};

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zPointFieldIterators
	*	\brief The 3D point field iterator classes of the library.
	*  @{
	*/

	/*! \class zItPointVectorField
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshVectorField
	*	\brief The 2D mesh vector field iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
		
	class ZSPACE_API zItPointVectorField : public zIt
	{
	protected:

		std::vector<zVector>::iterator iter; 

		/*!	\brief pointer to a mesh fieldobject  */
		zObjPointVectorField *fieldObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItPointVectorField();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\since version 0.0.3
		*/
		zItPointVectorField(zObjPointVectorField &_fieldObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_index				- input index in field values container.
		*	\since version 0.0.3
		*/
		zItPointVectorField(zObjPointVectorField &_fieldObj, int _index);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_pos				- input positionn.
		*	\param		[in]	closestIndex		- true if closest index is required.
		*	\since version 0.0.3
		*/
		zItPointVectorField(zObjPointVectorField &_fieldObj, zVector &_pos, bool closestIndex = false);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	pos			- input position.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		zItPointVectorField(zObjPointVectorField &_fieldObj, int _index_X, int _index_Y);

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
		//--- FIELD TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the ring neighbours of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Ring(int numRings, zItPointVectorFieldArray &ringNeighbours);

		/*! \brief This method gets the ring neighbour indicies of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Ring(int numRings, zIntArray &ringNeighbours);
	
		/*! \brief This method gets the ring Points of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Ring(int numRings, zPointArray &ringNeighbours);
		
		/*! \brief This method gets the immediate adjacent neighbours of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(zItPointVectorFieldArray &adjacentNeighbours);

		/*! \brief This method gets the immediate adjacent neighbour indicies of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(zIntArray &adjacentNeighbours);
		
		/*! \brief This method gets the immediate adjacent Points  of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of adjacent neighbour positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Adjacents(zPointArray &adjacentNeighbours);

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\return				zVector		- field position.
		*	\since version 0.0.2
		*/
		zVector getPosition();
		

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId();

		/*! \brief This method gets the indices of the iterator.
		*
		*	\param		[out]	index_X		- output index X
		*	\param		[out]	index_Y		- output index Y
		*	\param		[out]	index_Z		- output index Z
		*	\since version 0.0.3
		*/
		void getIndices(int &index_X, int& index_Y, int& index_Z);

		/*! \brief This method gets the value of the field at the current element.
		*
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\param		[out]	val			- field value.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		zVector getValue();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the value of the field at the current element.
		*
		*	\param		[in]	val			- field value.
		*	\param		[in]	append		- value if added to existing if true.
		*	\since version 0.0.2
		*/
		void setValue(zVector &val, bool append = false);

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItPointVectorField &other);

		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItPointVectorField &other);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/iterators/zItPointField.cpp>
#endif

#endif