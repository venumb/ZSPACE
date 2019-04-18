#pragma once

#include <headers/framework/utilities/zUtilsCore.h>
#include <headers/framework/utilities/zUtilsDisplay.h>
#include <headers/framework/utilities/zUtilsJson.h>

namespace zSpace
{
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObj
	*	\brief The base object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class  zObj
	{
	protected:
		
		/*! \brief pointer to display utilities object	*/
		zUtilsDisplay *displayUtils;

		/*! \brief pointer to core utilities object	*/
		zUtilsCore *coreUtils;
		
		/*! \brief boolean for displaying the object		*/
		bool showObject;


	public:
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------		
			

		/*! \brief stores a 4X4 matrix transform			*/
		zTransform transform;	
		

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObj()
		{
			showObject = true;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObj() {}

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------
		
		/*! \brief This method displays the object type.
		*
		*	\since version 0.0.2
		*/
		virtual void draw() {};		
		
	
		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show object boolean.
		*
		*	\param		[in]	_showObject				- input show object boolean.
		*	\since version 0.0.2
		*/
		void setShowObject(bool _showObject)
		{
			showObject = _showObject;
		}

		/*! \brief This method sets display utils.
		*
		*	\param		[in]	_displayUtils			- input display utils.
		*	\param		[in]	_coreUtils				- input core utils.
		*	\since version 0.0.2
		*/
		void setUtils(zUtilsDisplay &_displayUtils, zUtilsCore &_coreUtils)
		{
			displayUtils = &_displayUtils;
			coreUtils = &_coreUtils;
		}

		

	};


	
	

	


}

