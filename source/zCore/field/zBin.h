#pragma once

#include<headers/framework/core/zVector.h>
#include<headers/framework/core/zColor.h>



namespace zSpace
{

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
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


	class zBin
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
		zBin()
		{
			ids.clear();
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.3
		*/

		~zBin() {}

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method adds an object to the bin.
		*
		*	\since version 0.0.3
		*/
		void  addObject()
		{
			ids.push_back(vector<int>());			
		}

		/*! \brief This method adds the input vertex index of the input object to the bin.
		*
		*	\param		[in]	vertexId		- input vertexId.
		*	\param		[in]	objectId		- input objectId.
		*	\since version 0.0.3
		*/
		void addVertexIndex(int vertexId, int objectId)
		{
			if (objectId < ids.size())	ids[objectId].push_back(vertexId);
			else throw std::invalid_argument(" error: objectId out of bounds.");

		}

		/*! \brief This method clears the vertex indicies container
		*
		*	\since version 0.0.3
		*/
		void clear()
		{
			ids.clear();
		}

		/*! \brief This method returns if the bin contains any index or not.
		*
		*	\return				bool		- true if the bin cntains atleat one vertex index, else false.
		*	\since version 0.0.3
		*/
		bool contains()
		{
			return (ids.size() > 0);
		}

	};


}