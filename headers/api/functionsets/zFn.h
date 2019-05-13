#pragma once


#include <headers/api/object/zObj.h>

namespace zSpace
{
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFn
	*	\brief The base function set class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	class  zFn
	{

	protected:
		
		/*!	\brief function type  */
		zFnType fnType;
		

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFn() 
		{
			fnType = zInvalidFn;	
					
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFn() {}

		//--------------------------
		//---- VIRTUAL METHODS
		//--------------------------

		/*! \brief This method return the function set type.
		*
		*	\return 	zFnType			- type of function set.
		*	\since version 0.0.2
		*/
		virtual zFnType getType() { return fnType; }

		/*! \brief This method imports the object linked to function type.
		*
		*	\param	[in]	path			- output file name including the directory path and extension.
		*	\param	[in]	type			- type of file to be imported.
		*	\param	[in]	staticGeom		- true if the object is static. Helps speed up display especially for meshes object. Default set to false.
		*	\since version 0.0.2
		*/
		virtual void from(string path, zFileTpye type, bool staticGeom = false) {}

		/*! \brief This method exports the object linked to function type.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported.
		*	\since version 0.0.2
		*/
		virtual void to(string path, zFileTpye type) {}

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		virtual void clear() {}


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the object transform to the input transform.
		*
		*	\param [in]		inTransform			- input transform.
		*	\param [in]		decompose			- decomposes transform to rotation and translation if true.
		*	\param [in]		updatePositions		- updates the object positions if true.
		*	\since version 0.0.2
		*/
		virtual void setTransform(zTransform &inTransform,bool decompose = true, bool updatePositions = true){}
	
		
		/*! \brief This method sets the scale components of the object.
		*
		*	\param [in]		scale		- input scale values.
		*	\since version 0.0.2
		*/
		virtual void setScale(double3 &scale){}
	

		/*! \brief This method sets the rotation components of the object.
		*
		*	\param [in]		rotation			- input rotation values.
		*	\param [in]		append		- true if the input values are added to the existing rotations.
		*	\since version 0.0.2
		*/
		virtual void setRotation(double3 &rotation, bool append = false){}
	

		/*! \brief This method sets the translation components of the object.
		*
		*	\param [in]		translation			- input translation vector.
		*	\param [in]		append				- true if the input values are added to the existing translation.
		*	\since version 0.0.2
		*/
		virtual void setTranslation(zVector &translation, bool append = false){}
		

		/*! \brief This method sets the pivot of the object.
		*
		*	\param [in]		pivot				- input pivot position.
		*	\since version 0.0.2
		*/
		virtual void setPivot(zVector &pivot) {}
	
		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the object transform.
		*
		*	\since version 0.0.2
		*/
		virtual  void getTransform(zTransform &transform) {}



		//--------------------------
		//---- TRANSFORMATION METHODS
		//--------------------------
				
	protected:
		

		/*! \brief This method scales the object with the input scale transformation matix.
		*
		*	\since version 0.0.2
		*/
		virtual void transformObject(zTransform &transform) {  }

	};


}