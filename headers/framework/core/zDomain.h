#pragma once

#include<headers/framework/core/zDefinitions.h>
#include<headers/framework/core/zMatrix.h>
#include<headers/framework/core/zVector.h>
#include<headers/framework/core/zColor.h>

namespace  zSpace
{

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \struct zDomain
	*	\brief A struct for storing domain values.
	*	\since version 0.0.2
	*/

	/** @}*/
	/** @}*/

	template <typename T>
	struct zDomain
	{
		T min;
		T max;

		zDomain(){}
		
		zDomain(T _min, T _max)
		{
			min = _min;
			max = _max;
		}
	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zDomainTypedef
	*	\brief  The domain typedef of the library.
	*  @{
	*/

	/*! \typedef zDomainInt
	*	\brief A domain  of integers.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<int> zDomainInt;

	/*! \typedef zDomainDouble
	*	\brief A domain  of double.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<double> zDomainDouble;

	/*! \typedef zDomainFloat
	*	\brief A domain  of float.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<double> zDomainFloat;

	/*! \typedef zDomainColor
	*	\brief A domain  of color.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<zColor> zDomainColor;

	/*! \typedef zDomainVector
	*	\brief A domain  of vectors.
	*
	*	\since version 0.0.2
	*/
	typedef zDomain<zVector> zDomainVector;


	/** @}*/
	/** @}*/
	/** @}*/
}