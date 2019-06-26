#pragma once

#include<headers/api/iterators/zItMesh.h>
#include<headers/api/object/zObjMeshField.h>

namespace zSpace
{
	template<typename T>
	class zItMeshField : public zIt
	{
	protected:

		std::vector<T>::iterator iter;

		/*!	\brief pointer to a mesh fieldobject  */
		zObjMeshField<T> *fieldObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItMeshField()
		{
			fieldObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\since version 0.0.3
		*/
		zItMeshField(zObjMeshField<T> &_fieldObj)
		{
			fieldObj = &_fieldObj;
			iter = fieldObj->field.fieldValues.begin();

		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_index				- input index in field values container.
		*	\since version 0.0.3
		*/
		zItMeshField(zObjMeshField<T> &_fieldObj, int _index)
		{
			fieldObj = &_fieldObj;
			iter = fieldObj->field.fieldValues.begin();

			if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_pos				- input positionn.
		*	\param		[in]	closestIndex		- true if closest index is required.
		*	\since version 0.0.3
		*/
		zItMeshField(zObjMeshField<T> &_fieldObj, zVector &_pos , bool closestIndex = false)
		{
			fieldObj = &_fieldObj;
			iter = fieldObj->field.fieldValues.begin();

			int _index_X = floor((_pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
			int _index_Y = floor((_pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

			int _index = _index_X * fieldObj->field.n_Y + _index_Y;

			if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	pos			- input position.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		zItMeshField(zObjMeshField<T> &_fieldObj, int _index_X, int _index_Y)
		{
			fieldObj = &_fieldObj;
			iter = fieldObj->field.fieldValues.begin();

			int _index = _index_X * fieldObj->field.n_Y + _index_Y;

			if (_index < 0 && _index >= fieldObj->field.fieldValues.size()) throw std::invalid_argument(" error: index out of bounds");
				
			
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = fieldObj->field.fieldValues.begin();
		}

		virtual void next() override
		{
			iter++;
		}

		virtual void prev() override
		{
			iter--;
		}

		virtual bool end() override
		{
			return (iter == fieldObj->field.fieldValues.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = fieldObj->field.fieldValues.begin();
		}


		virtual int size() override
		{

			return fieldObj->field.fieldValues.size();
		}

		//--------------------------
		//--- FIELD TOPOLOGY QUERY METHODS 
		//--------------------------


		/*! \brief This method gets the ring neighbours of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Ring(int numRings, vector<zItMeshField> &ringNeighbours);
		

		/*! \brief This method gets the ring Points of the field at the current element.
		*
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbours positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Ring(int numRings, vector<zVector> &ringNeighbours);
		
		/*! \brief This method gets the immediate adjacent neighbours of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbours.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(vector<zItMeshField> &adjacentNeighbours);
		
		/*! \brief This method gets the immediate adjacent Points  of the field at the current element.
		*
		*	\param		[out]	adjacentNeighbours	- contatiner of adjacent neighbour positions.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Adjacents(vector<zVector> &adjacentNeighbours);
		



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
		int getId()
		{
			return distance(fieldObj->field.fieldValues.begin(), iter);
		}

		/*! \brief This method gets the value of the field at the current element.
		*
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\param		[out]	val			- field value.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		T getValue()
		{
			return *iter;
		}

	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- double specilization for getNeighbour_Ring
	template<>
	inline void zItMeshField<double>::getNeighbour_Ring(int numRings, vector<zItMeshField>& ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(zItMeshField(*fieldObj, newId));
			}

		}
	}
		   
	//---- vector specilization for getNeighbour_Ring
	template<>
	inline void zItMeshField<zVector>::getNeighbour_Ring(int numRings, vector<zItMeshField>& ringNeighbours)
	{
		ringNeighbours.clear();

		int idX = floor(getId() / fieldObj->field.n_Y);
		int idY = getId() % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X - 1) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size()) ringNeighbours.push_back(zItMeshField(*fieldObj, newId));
			}

		}
	}
			
	//---------------//
	
	//---- vector specilization for getNeighbourPosition_Ring
	template<>
	inline void zItMeshField<zVector>::getNeighbourPosition_Ring( int numRings, vector<zVector>& ringNeighbours)
	{
		ringNeighbours.clear();

		vector<zItMeshField> rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fVector : rNeighbour)
		{
			ringNeighbours.push_back(fVector.getPosition());
		}

	}

	//---- zVector specilization for getNeighbourPosition_Ring
	template<>
	inline void zItMeshField<double>::getNeighbourPosition_Ring( int numRings, vector<zVector>& ringNeighbours)
	{
		ringNeighbours.clear();

		vector<zItMeshField> rNeighbour;
		getNeighbour_Ring(numRings, rNeighbour);

		for (auto &fScalar : rNeighbour)
		{
			ringNeighbours.push_back(fScalar.getPosition());
		}

	}

	//---------------//

	//---- double specilization for getNeighbour_Adjacents
	template<>
	inline void zItMeshField<double>::getNeighbour_Adjacents( vector<zItMeshField>& adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(zItMeshField(*fieldObj, newId));
				}
			}

		}

	}
	

	//---- vector specilization for getNeighbour_Adjacents
	template<>
	inline void zItMeshField<zVector>::getNeighbour_Adjacents( vector<zItMeshField>& adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		int numRings = 1;
		int index = getId();

		int idX = floor(index / fieldObj->field.n_Y);
		int idY = index % fieldObj->field.n_Y;

		int startIdX = -numRings;
		if (idX == 0) startIdX = 0;

		int startIdY = -numRings;
		if (idY == 0) startIdY = 0;

		int endIdX = numRings;
		if (idX == fieldObj->field.n_X) endIdX = 0;

		int endIdY = numRings;
		if (idY == fieldObj->field.n_Y) endIdY = 0;

		for (int i = startIdX; i <= endIdX; i++)
		{
			for (int j = startIdY; j <= endIdY; j++)
			{
				int newId_X = idX + i;
				int newId_Y = idY + j;

				int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


				if (newId < size())
				{
					if (i == 0 || j == 0) adjacentNeighbours.push_back(zItMeshField(*fieldObj, newId));
				}
			}

		}

	}

	//---------------//

	//---- double specilization for getNeighbourPosition_Adjacents
	template<>
	inline void zItMeshField<double>::getNeighbourPosition_Adjacents( vector<zVector>& adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		vector<zItMeshField> aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fScalar : aNeighbour)
		{
			adjacentNeighbours.push_back(fScalar.getPosition());
		}

	}
	
	//---- vector specilization for getNeighbourPosition_Adjacents
	template<>
	inline void zItMeshField<double>::getNeighbourPosition_Adjacents( vector<zVector>& adjacentNeighbours)
	{
		adjacentNeighbours.clear();

		vector<zItMeshField> aNeighbour;
		getNeighbour_Adjacents(aNeighbour);

		for (auto &fVector : aNeighbour)
		{
			adjacentNeighbours.push_back(fVector.getPosition());
		}

	}

	//---------------//

	//---- double specilization for getPosition
	template<>
	inline zVector zItMeshField<double>::getPosition()
	{

		if (fieldObj->field.valuesperVertex)
		{
			zItMeshVertex v(*fieldObj, getId());
			return v.getVertexPosition();
		}
		else
		{
			zItMeshFace f(*fieldObj, getId());
			return f.getCenter();
		}
	}

	//---- vector specilization for getPosition
	template<>
	inline zVector zItMeshField<zVector>::getPosition()
	{

		if (fieldObj->field.valuesperVertex)
		{
			zItMeshVertex v(*fieldObj, getId());
			return v.getVertexPosition();
		}
		else
		{
			zItMeshFace f(*fieldObj, getId());
			return f.getCenter();
		}
	}


#endif /* DOXYGEN_SHOULD_SKIP_THIS */
}