#pragma once


#include <stdexcept>
#include <vector>
using namespace std;

#define PI 3.14159265

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core classes of the library.
	*  @{
	*/

	/*! \class zVector
	*	\brief A vector math class. 
	*	\since version 0.0.1
	*/

	/** @}*/ // end of group core
	class zVector
	{
	public:
		double x;			/*!< x component				*/
		double y;			/*!< y component				*/
		double z;			/*!< z component				*/

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		
		zVector()
		{
			x = 0;
			y = 0;
			z = 0;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_x		- x component of the zVector.
		*	\param		[in]	_z		- y component of the zVector.
		*	\param		[in]	_z		- z component of the zVector.
		*	\since version 0.0.1
		*/
		
		zVector(double _x, double _y, double _z)
		{
			x = _x;
			y = _y;
			z = _z;
		}

		//---- DESTRUCTOR

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		
		~zVector() {}

		//---- OPERATORS

		/*! \brief This operator checks for equality of two zVectors.
		*
		*	\param		[in]	v1		- zVector against which the equality is checked.
		*	\return				bool	- true if vectors are equal.
		*	\since version 0.0.1
		*/
		
		bool operator==(zVector v1)
		{
			bool out = false;
			if (x == v1.x && y == v1.y && z == v1.z) out = true;

			return out;
		}

		/*! \brief This operator is used for vector addition.
		*
		*	\param		[in]	v1		- zVector which is added to the current vector.
		*	\return				zVector	- resultant vector after the addition.
		*	\since version 0.0.1
		*/
		
		zVector operator+(zVector v1)
		{
			return zVector(x + v1.x, y + v1.y, z + v1.z);
		}

		/*! \brief This operator is used for vector subtraction.
		*
		*	\param		[in]	v1		- zVector which is subtracted from the current vector.
		*	\return				zVector	- resultant vector after the subtraction.
		*	\since version 0.0.1
		*/
	
		zVector operator -(zVector v1)
		{
			return zVector(x - v1.x, y - v1.y, z - v1.z);
		}

		/*! \brief This operator is used for vector dot product.
		*
		*	\param		[in]	v1		- zVector which is used for the dot product with the current vector.
		*	\return				double	- resultant value after the dot product.
		*	\since version 0.0.1
		*/
	
		double operator *(zVector v1)
		{
			return (x * v1.x + y * v1.y + z * v1.z);
		}

		/*! \brief This operator is used for vector cross procduct.
		*
		*	\param		[in]	v1		- zVector which is used for the cross product with the current vector.
		*	\return				zVector	- resultant vector after the cross product.
		*	\since version 0.0.1
		*/
	
		zVector operator ^(zVector v1)
		{
			return zVector(y * v1.z - z * v1.y, z*v1.x - x * v1.z, x*v1.y - y * v1.x);
		}

		/*! \brief This operator is used for scalar addition of a vector.
		*
		*	\param		[in]	val		- scalar value to be added to the current vector.
		*	\return				zVector	- resultant vector after the scalar addition.
		*	\since version 0.0.1
		*/
	
		zVector operator +(double val)
		{
			return  zVector(x + val, y + val, z + val);
		}

		/*! \brief This operator is used for scalar subtraction of a vector.
		*
		*	\param		[in]	val		- scalar value to be subtracted from the current vector.
		*	\return				zVector	- resultant vector after the scalar subtraction.
		*	\since version 0.0.1
		*/
	
		zVector operator -(double val)
		{
			return  zVector(x - val, y - val, z - val);
		}

		/*! \brief This operator is used for scalar muliplication of a vector.
		*
		*	\param		[in]	val		- scalar value to be multiplied with the current vector.
		*	\return				zVector	- resultant vector after the scalar multiplication.
		*	\since version 0.0.1
		*/
	
		zVector operator *(double val)
		{
			return  zVector(x * val, y * val, z * val);
		}

		/*! \brief This operator is used for scalar division of a vector.
		*
		*	\param		[in]	val		- scalar value used to divide from the current vector.
		*	\return				zVector	- resultant vector after the scalar division.
		*	\since version 0.0.1
		*/
	
		zVector operator /(double val)
		{
			if (val == 0)
				throw std::invalid_argument("val can't be zero");

			return  zVector(x / val, y / val, z / val);
		}

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This overloaded operator is used for vector addition and assigment of the result to the current vector.
		*
		*	\param		[in]	v1		- zVector which is added to the current vector.
		*	\since version 0.0.1
		*/
		
		void operator +=(zVector v1)
		{
			x += v1.x;
			y += v1.y;
			z += v1.z;

		}

		/*! \brief This overloaded operator is used for vector subtraction and assigment of the result to the current vector.
		*
		*	\param		[in]	v1		- zVector which is subtacted from the current vector.
		*	\since version 0.0.1
		*/
	
		void operator -=(zVector v1)
		{
			x -= v1.x;
			y -= v1.y;
			z -= v1.z;

		}

		
		/*! \brief This overloaded operator is used for scalar addition and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be added to the current vector.
		*	\since version 0.0.1
		*/
		
		void operator +=(double val)
		{
			x += val;
			y += val;
			z += val;

		}

		/*! \brief This overloaded operator is used for scalar subtraction and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be sbtracted from the current vector.
		*	\since version 0.0.1
		*/
	
		void operator -=(double val)
		{
			x -= val;
			y -= val;
			z -= val;

		}

		/*! \brief This overloaded operator is used for scalar multiplication and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value to be multiplied to the current vector.
		*	\since version 0.0.1
		*/
	
		void operator *=(double val)
		{
			x *= val;
			y *= val;
			z *= val;

		}

		/*! \brief This overloaded operator is used for scalar division and assigment of the result to the current vector.
		*
		*	\param		[in]	val		- scalar value used to divide from the current vector.
		*	\since version 0.0.1
		*/
	
		void operator /=(double val)
		{
			x /= val;
			y /= val;
			z /= val;

		}

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method returns the magnitude/length of the zVector.
		*
		*	\return				double		- value of the maginute of the vector.
		*	\since version 0.0.1
		*/
	
		double length()
		{
			return sqrt(x*x + y * y + z * z);
		}

		/*! \brief This method normalizes the vector to unit length.
		*	\since version 0.0.1
		*/
	
		void normalize()
		{
			double length = this->length();

			x /= length;
			y /= length;
			z /= length;
		}

		/*! \brief This method returns the distance between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				double		- value of the distance between the vectors.
		*	\since version 0.0.1
		*/
	
		double distanceTo(zVector v1)
		{
			return sqrt(pow((x - v1.x), 2.0) + pow((y - v1.y), 2.0) + pow((z - v1.z), 2.0));
		}

		/*! \brief This method returns the angle between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				double		- value of the angle between the vectors.
		*	\since version 0.0.1
		*/
	
		double angle(zVector v1)
		{
			double length = this->length();
			double length1 = v1.length();

			double dotProduct = (x * v1.x + y * v1.y + z * v1.z);

			double angle = acos(dotProduct / (length * length1));

			return angle * (180.0 / PI);

		}

		/*! \brief This method returns the dihedral angle between the two input zVectors using current zVector as edge reference.
		*
		*	\param		[in]	v1			- input vector.
		*	\param		[in]	v2			- input vector.
		*	\return				double		- value of the dihedral angle between the vectors.
		*	\since version 0.0.1
		*/
	
		double dihedralAngle(zVector v1, zVector v2)
		{
			zVector edgeRef(x, y, z);
			zVector tmp(v1.x, v1.y, v1.z);

			
			v1.normalize();
			v2.normalize();
			double dot = v1 * v2;
			dot = (dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot));
			double  dtheta = atan2((tmp^v2).length(), tmp*v2);
			while (dtheta > PI)
				dtheta -= PI * 2;
			while (dtheta < -PI)
				dtheta += PI * 2;

			return(dtheta * (180.0 / PI) * ((v1^v2)*edgeRef < 0 ? -1 : 1));
		}

	
		/*! \brief This method returns the component value of the current zVector.
		*
		*	\param		[in]	i			- index. ( 0 - x component, 0 - y component, 2 - z component).
		*	\return				double		- value of the dihedral angle between the vectors.
		*	\since version 0.0.1
		*/
	
		double  getComponent(int i)
		{
			if (i == 0)return x;
			if (i == 1)return y;
			if (i == 2)return z;
		}



	
	};
}

