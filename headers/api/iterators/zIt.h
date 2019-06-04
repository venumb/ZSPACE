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

		virtual void begin() { }

		virtual void next() {}

		virtual void prev() {}

		virtual bool end() { return false; }

		virtual void reset() {}			

		virtual int size() { return 0; }		

		virtual void deactivate() {}
	};
}