#pragma once
#pragma once

#include <headers/api/object/zObject.h>
#include <headers/framework/field/zField2D.h>

#include <vector>
using namespace std;

namespace zSpace
{


	/** \addtogroup zObjects
	*	\brief The object and function set classes of the library.
	*  @{
	*/

	/*! \class zGraphObject
	*	\brief The graph object class.
	*	\since version 0.0.2
	*/

	/** @}*/
	
	template<typename T>
	class zObjField2D : public zObject
	{
	private:
		
	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField2D<T> field;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjField2D()
		{
			displayUtils = nullptr;			
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjField2D() {}

	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			
		}		

	};




}

