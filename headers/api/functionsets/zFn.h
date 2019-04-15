#pragma once


#include <headers/api/object/zObject.h>

namespace zSpace
{
	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFn
	*	\brief The base function set class.
	*	\since version 0.0.2
	*/

	/** @}*/
	class  zFn
	{

	protected:
		enum zFnType { zInvalidFn = 160 , zPointCloudFn, zMeshFn, zGraphFn, zField2DFn, zTyField3DFn, zParticleFn } ;

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
		*	\since version 0.0.1
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

		/*! \brief This method exports the object linked to function type.
		*
		*	\param [in]		path			- output file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported.
		*	\since version 0.0.2
		*/
		virtual void from(string path, zFileTpye type) {}

		/*! \brief This method imports the object linked to function type.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		virtual void to(string path, zFileTpye type) {}

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		virtual void clear() {};
	};


}