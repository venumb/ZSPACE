#pragma once

#include<headers/api/object/zObj.h>
#include<headers/api/functionsets/zFn.h>

namespace zSpace
{
	/** \addtogroup API
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

	class  zIt
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
		zIt()
		{		
			
		}
				

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.3
		*/
		~zIt() {}

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------
		
		
		/*! \brief This method sets the iterator pointing to first element the contatiner.
		*
		*	\since version 0.0.3
		*/
		virtual void begin() { }

		/*! \brief This method advance to the next element in the iteration.
		*
		*	\since version 0.0.3
		*/
		virtual void next() {}

		/*! \brief This method advance to the previous element in the iteration.
		*
		*	\since version 0.0.3
		*/
		virtual void prev() {}

		/*! \brief This method indicates if all the elements are tranversed.
		*
		*	\return		bool	- true if all the elements are tranversed.
		*	\since version 0.0.3
		*/
		virtual bool end() { return false; }

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
		virtual int size() { return 0; }		

		/*! \brief This method deactivates the element attached to the iterator.
		*
		*	\since version 0.0.3
		*/
		virtual void deactivate() {}
	};
}