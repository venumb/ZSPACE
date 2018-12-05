#pragma once

#include <headers/core/zVector.h>
#include <headers/core/zMatrix.h>

#include <vector>
using namespace std;

namespace zSpace
{
	


	/** \addtogroup zObject
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryClasses
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \struct zObject
	*	\brief A basic object class holding informations of a transforma nd object type.
	*	\tparam				T			- Type to work with the library classes - zVector, zGraph, zMesh, zField2D, zField3D.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template<typename T>
	struct  zObject
	{

	public:
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief stores a 4X4 matrix transform			*/
		zTransform transform;	

		/*! \brief object type		*/
		T objectType;


		zObject(T &_object)
		{
			object = _object;
		}

	
	};


	template<typename T>
	class zFieldObject
	{
		double influence;
		
		vector<zObject<T>> fieldObject;

		zFieldObject() {}

		zFieldObject(double _influence, zObject<T> &_fieldObject)
		{
			influence = _influence;
			fieldObject = _fieldObject
		}

		~zFieldObject() {}
		
	};

	


}

