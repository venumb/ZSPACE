#pragma once

#include<headers/api/iterators/zIt.h>
#include<headers/api/object/zObjPointCloud.h>

namespace zSpace
{
		
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshVertex
	*	\brief The mesh vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItPointCloudVertex : public zIt
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
		zItPointCloudVertex()
		{
			pointsObj = nullptr;
		}
	
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\since version 0.0.3
		*/
		zItPointCloudVertex(zObjPointCloud &_pointsObj)
		{
			pointsObj = &_pointsObj;

			iter = pointsObj->pCloud.vertices.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\param		[in]	_index				- input index in mesh vertex list.
		*	\since version 0.0.3
		*/
		zItPointCloudVertex(zObjPointCloud &_pointsObj, int _index)
		{
			pointsObj = &_pointsObj;

			iter = pointsObj->pCloud.vertices.begin();

			if (_index < 0 && _index >= pointsObj->pCloud.vertices.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = pointsObj->pCloud.vertices.begin();
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
			return (iter == pointsObj->pCloud.vertices.end()) ? true : false;
		}


		virtual void reset() override 
		{
			iter = pointsObj->pCloud.vertices.begin();
		}		
		

		virtual int size() override
		{		

			return pointsObj->pCloud.vertices.size();
		}

		virtual void deactivate() override 
		{			
			iter->reset();			
		}   	
			

	
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
			return iter->getId();
		}
		

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItVertex		- raw iterator
		*	\since version 0.0.3
		*/
		zItVertex  getRawIter()
		{
			return iter;
		}

		/*! \brief This method gets position of the vertex.
		*
		*	\return				zVector					- vertex position.
		*	\since version 0.0.3
		*/
		zVector getVertexPosition()
		{
			return pointsObj->pCloud.vertexPositions[getId()];
		}

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zVector* getRawVertexPosition()
		{
			return &pointsObj->pCloud.vertexPositions[getId()];
		}
				

		/*! \brief This method gets color of the vertex.
		*
		*	\return				zColor					- vertex color.
		*	\since version 0.0.3
		*/
		zColor getVertexColor()
		{
			return pointsObj->pCloud.vertexColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the vertex.
		*
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.3
		*/
		zColor* getRawVertexColor()
		{
			return &pointsObj->pCloud.vertexColors[getId()];
		}
		
		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id)
		{
			iter->setId(_id);
		}				

		/*! \brief This method sets position of the vertex.
		*
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.3
		*/
		void setVertexPosition(zVector &pos)
		{
			pointsObj->pCloud.vertexPositions[getId()] = pos;
		}

		/*! \brief This method sets color of the vertex.
		*
		*	\param		[in]	col						- vertex color.
		*	\since version 0.0.3
		*/
		void setVertexColor(zColor col)
		{
			pointsObj->pCloud.vertexColors[getId()] = col;
		}

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------
		
		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive()
		{
			return iter->isActive();
		}
		

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItPointCloudVertex &other)
		{
			return (getId() == other.getId());
		}


		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItPointCloudVertex &other)
		{
			return (getId() != other.getId());
		}

		

	};
		

	
}