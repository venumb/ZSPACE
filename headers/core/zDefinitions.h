#pragma once


namespace zSpace
{

	/** \addtogroup zDefinitions
	*	\brief Collection of all the definitions in the library.
	*  @{
	*/

	/*!
	*	\brief Defines the value of PI.
	*/
	#ifndef PI
	#define PI       3.14159265358979323846
	#endif

	/*!
	*	\brief Defines the value of 2 times PI.
	*/
	#ifndef TWO_PI
	#define TWO_PI   6.28318530717958647693
	#endif

	/*!
	*	\brief Defines the value of 4 times PI.
	*/
	#ifndef FOUR_PI
	#define FOUR_PI 12.56637061435917295385
	#endif

	/*!
	*	\brief Defines the value of 0.5 times PI.
	*/
	#ifndef HALF_PI
	#define HALF_PI  1.57079632679489661923
	#endif

	/*!
	*	\brief Defines the conversion of degrees to radians.
	*/
	#ifndef DEG_TO_RAD
	#define DEG_TO_RAD (PI/180.0)
	#endif

	/*!
	*	\brief Defines the conversion of radians to degrees.
	*/
	#ifndef RAD_TO_DEG
	#define RAD_TO_DEG (180.0/PI)
	#endif


	/** @}*/ // end of group zEnumerators
}
