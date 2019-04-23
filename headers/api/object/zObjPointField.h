#pragma once
#pragma once

#include <headers/api/object/zObjPointCloud.h>
#include <headers/framework/field/zField3D.h>

#include <vector>
using namespace std;

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

	/*! \class zObjPointField
	*	\brief The 3D point field object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	template<typename T>
	class zObjPointField : public zObjPointCloud
	{
	private:
		
	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField3D<T> field;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjPointField()
		{
			displayUtils = nullptr;	

			showVertices = false;
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjPointField() {}

	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			zObjPointCloud::draw();
		}		

	};




}

