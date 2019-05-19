#pragma once

#include<headers/framework/core/zDefinitions.h>
#include<headers/framework/core/zMatrix.h>
#include<headers/framework/core/zVector.h>

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

	/*! \class zQuaternion
	*	\brief A Quaternion  class.
	*	\details Adapted from Quaternion class described here https://github.com/dgpdec/course.
	*	\since version 0.0.2
	*/

	/** @}*/
	/** @}*/

	class zQuaternion
	{
	protected:

		/*!	\brief scalar part of quaternion			*/
		double s;		

		/*!	\brief vector part of quaternion(imaginary)		*/
		zVector v;
	

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------


		/*! \brief Default Constructor.
		*	\since version 0.0.2
		*/
		zQuaternion()
		{
			s = 0.0;
			v = zVector();
		}

		/*! \brief Overloaded Constructor.
		*	\param		[in]	_s			- scalar part.
		*	\param		[in]	_v			- vector part.
		*	\since version 0.0.2
		*/
		zQuaternion(double _s, double _vi, double _vj, double _vk)
		{
			s = _s;
			v = zVector(_vi, _vj, _vk);
		}

		/*! \brief Overloaded Constructor.
		*	\param		[in]	_s			- scalar part.
		*	\param		[in]	_v			- vector part.
		*	\since version 0.0.2
		*/
		zQuaternion(double _s, zVector _v)
		{
			s = _s;
			v = _v;
		}

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator assigns scalar component of the quaternion
		*
		*	\param		[in]	_s		- scalar component.
		*	\since version 0.0.2
		*/
		zQuaternion operator=(double _s)
		{
			s = _s;
			v = zVector();
		}

		/*! \brief This operator assigns vector component of the quaternion
		*
		*	\param		[in]	_v		- vector component.
		*	\since version 0.0.2
		*/
		zQuaternion operator=(zVector &_v)
		{
			s = 0.0;
			v = _v;
		}

		/*! \brief This operator returns the indexed component (0-based indexing: double, i, j, k)
		*
		*	\return			double&		- reference to the specified component (0-based indexing: double, i, j, k)
		*	\since version 0.0.2
		*/
		double& operator[](int index)
		{
			return (&s)[index];
		}

		/*! \brief This operator returns the indexed component (0-based indexing: double, i, j, k)
		*
		*	\return			double&		- const reference to the specified component (0-based indexing: double, i, j, k)
		*	\since version 0.0.2
		*/
		const double& operator[](int index) const			
		{
			return (&s)[index];
		}

		/*! \brief This operator is used for quaternion addition.
		*
		*	\param		[in]	q			- quaternion which is added to the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the addition.
		*	\since version 0.0.2
		*/
		zQuaternion operator+(const zQuaternion  &q)
		{
			return zQuaternion(s + q.s, v + q.v);
		}

		/*! \brief This operator is used for quaternion subtraction.
		*
		*	\param		[in]	q			- quaternion which is subtracted from the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the subtraction.
		*	\since version 0.0.2
		*/
		zQuaternion operator-(const zQuaternion &q)
		{
			return zQuaternion(s - q.s, v - q.v);
		}

		/*! \brief This operator is used for quaternion multiplication with a scalar.
		*
		*	\param		[in]	c			- scalar value which is multiplied to the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the multiplication.
		*	\since version 0.0.2
		*/
		zQuaternion operator*(double c) 
		{
			return zQuaternion(s*c, v *c);
		}

		/*! \brief This operator is used for quaternion Hamiton Product.
		*
		*	\param		[in]	q			- quaternion which is multiplied with the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the multiplication.
		*	\since version 0.0.2
		*/
		zQuaternion operator*(zQuaternion &q)
		{			
			return zQuaternion(s*q.s - (v * q.v), ((q.v *s) + (v * q.s)) + (v^q.v));
		}

		/*! \brief This operator is used for quaternion division with a scalar.
		*
		*	\param		[in]	q			- scalar value which divides the current quaternion.
		*	\return				zQuaternion	- resultant quaternion after the division.
		*	\since version 0.0.2
		*/
		zQuaternion operator/(double c)
		{
			return zQuaternion(s/c, v/c);
		}

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This overloaded operator is used for addition and assigment of the result to the current quaternion.
		*
		*	\param		[in]	q		- quaternion which is added to the current quaternion.
		*	\since version 0.0.1
		*/
		void operator +=(const zQuaternion &q)
		{
			s += q.s;
			v += q.v;
		}	

		/*! \brief This overloaded operator is used for subtraction and assigment of the result to the current quaternion.
		*
		*	\param		[in]	q		- quaternion which is subtracted from the current quaternion.
		*	\since version 0.0.1
		*/
		void operator -=(const zQuaternion &q)
		{
			s -= q.s;
			v -= q.v;
		}


		/*! \brief This operator is used for multiplication with a scalar and assigment of the result to the current quaternion.
		*
		*	\param		[in]	c			- scalar value which is multiplied to the current quaternion.		
		*	\since version 0.0.2
		*/
		void operator*=(double c)
		{
			s *= c;
			v *=c;
		}

		/*! \brief This operator is used for quaternion Hamiton Product and assigment of the result to the current quaternion..
		*
		*	\param		[in]	q			- quaternion which is multiplied with the current quaternion.
		*	\since version 0.0.2
		*/
		void operator*=(zQuaternion q)
		{
			*this = (*this * q);
		}

		/*! \brief This operator is used for division with a scalar and assigment of the result to the current quaternion.
		*
		*	\param		[in]	c			- scalar value which is used for division of the current quaternion.
		*	\since version 0.0.2
		*/
		void operator/=(double c)
		{
			s /= c;
			v /= c;
		}



		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the real/scalar component of the quaternion.
		*	\return				double		- scalar component.
		*	\since version 0.0.2
		*/
		double getRe()
		{
			return s;
		}

		/*! \brief This method gets the imaginary/ vector component of the quaternion.
		*	\return				zVector		- vector component.
		*	\since version 0.0.2
		*/
		zVector getIm()
		{
			return v;
		}

		/*! \brief This method gets the conjuate of the quaternion.
		*	\return				zQuaternion		- conjugate quaternion.
		*	\since version 0.0.2
		*/
		zQuaternion getConjugate()
		{
			return zQuaternion(s, v * -1);
		}

		/*! \brief This method gets the inverse of the quaternion.
		*	\return				zQuaternion		- inverse quaternion.
		*	\since version 0.0.2
		*/
		zQuaternion getInverse()
		{
			return (this->getConjugate()) / this->length2();
		}

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method gets the Euclidean length of the quaternion.
		*	\return				double		- Euclidean length.
		*	\since version 0.0.2
		*/
		double length() 			
		{
			return sqrt(s*s + v.x*v.x + v.y*v.y + v.z*v.z);
		}

		/*! \brief This method gets the Euclidean length sqaured of the quaternion.
		*	\return				double		- Euclidean length squared.
		*	\since version 0.0.2
		*/
		double length2()
		{
			return ((s*s) + (v*v));
		}

		/*! \brief This method normalizes the quaternion .
		*	\since version 0.0.2
		*/
		void normalize()
		{
			*this /= length2();
		}

		/*! \brief This method computers  spherical-linear interpolation
		*	\param		[in]	q0			- input start quaternion.
		*	\param		[in]	q1			- input end quaternion.
		*	\param		[in]	t			- input time step.
		*	\since version 0.0.2
		*/
		zQuaternion slerp( zQuaternion& q0, zQuaternion& q1, double t)			
		{
			// interpolate length
			double m0 = q0.length();
			double m1 = q1.length();
			double m = (1 - t)*m0 + t * m1;

			// interpolate direction
			zQuaternion p0 = q0 / m0;
			zQuaternion p1 = q1 / m1;
			double theta = acos((p0.getConjugate()*p1).getRe());
			zQuaternion p = p0*(sin((1 - t)*theta)) + (p1*sin(t*theta)) / sin(theta);

			return p*m ;
		}

	};

}