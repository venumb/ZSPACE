#pragma once

#include <stdexcept>
#include <vector>
using namespace std;



#include<headers/framework/core/zDefinitions.h>
#include<headers/framework/core/zMatrix.h>

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

	/*! \class zVector
	*	\brief A 3 dimensional vector math class. 
	*	\since version 0.0.1
	*/

	/** @}*/ 
	/** @}*/

	class zVector
	{
	protected:
		
		/*!	\brief pointer to components	*/
		pDouble3 vals;

	public:
		/*!	\brief x component				*/
		double x;			

		/*!	\brief y component				*/
		double y;	

		/*!	\brief z component				*/
		double z;			

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
			
			vals[0] = &x;
			vals[1] = &y;
			vals[2] = &z;
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
			
			vals[0] = &x;
			vals[1] = &y;
			vals[2] = &z;
					
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	vals	- input array of values.
		*	\since version 0.0.2
		*/
		zVector(double3 &_vals)
		{	

			x = _vals[0];
			y = _vals[1];
			z = _vals[2];
			
			vals[0] = &x;
			vals[1] = &y;
			vals[2] = &z;
		}

		//---- DESTRUCTOR

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		
		~zVector() 	{}

		//---- OPERATORS

		/*! \brief This operator checks for equality of two zVectors.
		*
		*	\param		[in]	v1		- zVector against which the equality is checked.
		*	\return				bool	- true if vectors are equal.
		*	\since version 0.0.1
		*/		
		bool operator==(const zVector &v1)
		{
			bool out = false;
			if (x == v1.x && y == v1.y && z == v1.z) out = true;

			return out;
		}

		/*! \brief This method returns the component value of the current zVector.
		*
		*	\param		[in]	index		- index. ( 0 - x component, 1 - y component, 2 - z component).
		*	\return				double		- value of the component.
		*	\since version 0.0.1
		*/
		double  operator[](int index)
		{
			if (index >= 0 && index <= 2)return *vals[index];
		}

		/*! \brief This operator is used for vector addition.
		*
		*	\param		[in]	v1		- zVector which is added to the current vector.
		*	\return				zVector	- resultant vector after the addition.
		*	\since version 0.0.1
		*/		
		zVector operator+(const zVector &v1)
		{
			return zVector(x + v1.x, y + v1.y, z + v1.z);
		}

		/*! \brief This operator is used for vector subtraction.
		*
		*	\param		[in]	v1		- zVector which is subtracted from the current vector.
		*	\return				zVector	- resultant vector after the subtraction.
		*	\since version 0.0.1
		*/	
		zVector operator -(const zVector &v1)
		{
			return zVector(x - v1.x, y - v1.y, z - v1.z);
		}

		/*! \brief This operator is used for vector dot product.
		*
		*	\param		[in]	v1		- zVector which is used for the dot product with the current vector.
		*	\return				double	- resultant value after the dot product.
		*	\since version 0.0.1
		*/	
		double operator *(const zVector &v1)
		{
			return (x * v1.x + y * v1.y + z * v1.z);
		}

		/*! \brief This operator is used for vector cross procduct.
		*
		*	\param		[in]	v1		- zVector which is used for the cross product with the current vector.
		*	\return				zVector	- resultant vector after the cross product.
		*	\since version 0.0.1
		*/	
		zVector operator ^(const zVector &v1)
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

		/*! \brief This operator is used for 4x4 / 3X3 matrix muliplication of a vector.
		*
		*	\param		[in]	inMatrix	- input 4X4 / 3X3 zMatrixd to be multiplied with the current vector.
		*	\return				zVector		- resultant vector after the matrix multiplication.
		*	\since version 0.0.1
		*/
		zVector operator*(zMatrixd inMatrix)
		{
			if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
			if (inMatrix.getNumCols() < 3 || inMatrix.getNumCols() > 4) 	throw std::invalid_argument("input Matrix is not a 3X3 or 4X4 matrix.");

			zMatrixd vecMatrix = this->toColumnMatrix(inMatrix.getNumCols());

			zMatrixd outVecMatrix =  inMatrix * vecMatrix;

			return this->fromColumnMatrix(outVecMatrix);
		}

		/*! \brief This operator is used for 4x4 matrix muliplication of a vector.
		*
		*	\param		[in]	inMatrix	- input 4X4 zTransform to be multiplied with the current vector.
		*	\return				zVector		- resultant vector after the matrix multiplication.
		*	\since version 0.0.1
		*/
		zVector operator*(zTransform inTrans)
		{
			Vector4d p(x, y, z, 1);
			
			Vector4d newP = inTrans * p;

			zVector out(newP(0), newP(1), newP(2));

			return out;
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
		void operator +=(const zVector &v1)
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
		void operator -=(const zVector &v1)
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

		/*! \brief This method returns the squared length of the zVector.
		*
		*	\return				double		- value of the squared maginute of the vector.
		*	\since version 0.0.1
		*/
		double length2()
		{
			return (x*x + y * y + z * z);
		}

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


		/*! \brief This method returns the square distance between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				double		- value of the distance between the vectors.
		*	\since version 0.0.1
		*/
		double squareDistanceTo(zVector &v1)
		{
			return (this->operator- (v1)) * (this->operator- (v1));
		}

		/*! \brief This method returns the distance between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				double		- value of the distance between the vectors.
		*	\since version 0.0.1
		*/	
		double distanceTo(zVector &v1)
		{
			return sqrt(squareDistanceTo(v1));
		}
			

		/*! \brief This method returns the angle between the current zVector and input zVector.
		*
		*	\param		[in]	v1			- input vector.
		*	\return				double		- value of the angle between the vectors.
		*	\since version 0.0.1
		*/	
		double angle(zVector &v1)
		{
			// check if they are parallel
			zVector a(x,y,z);
			zVector b(v1.x, v1.y, v1.z);

			a.normalize();
			b.normalize();

			if (a*b == 1) return 0;
			else if (a*b == -1) return 180;
			else
			{
				
				double dotProduct = a * b;
				double angle = acos(dotProduct);

				return angle * RAD_TO_DEG;
			}
			
			

		}

		/*! \brief This method returns the angle between the current zVector and input zVector in the range of 0 to 360.
		*
		*	\param		[in]	v1			- input vector.
		*	\param		[in]	normal		- input reference normal or axis of rotation.
		*	\return				double		- value of the angle between the vectors.
		*	\since version 0.0.1
		*/
		double angle360(zVector &v1, zVector &normal)
		{
			// check if they are parallel
			zVector a(x, y, z);
			zVector b(v1.x, v1.y, v1.z);

			a.normalize();
			b.normalize();

			if (a*b == 1) return 0;
			else if (a*b == -1) return 180;
			else
			{
				double length = this->length();
				double length1 = v1.length();

				double dot = a*b;

				zVector cross = a ^ b;
				double det = normal * (cross);

				double angle = atan2(det,dot);
				if(angle < 0) angle += TWO_PI;

			
				return angle * RAD_TO_DEG;
			}



		}

		/*! \brief This method returns the dihedral angle between the two input zVectors using current zVector as edge reference.
		*
		*	\details Based on https://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleMeshDerivativesCheatSheet.pdf
		*	\param		[in]	v1			- input vector.
		*	\param		[in]	v2			- input vector.
		*	\return				double		- value of the dihedral angle between the vectors.
		*	\since version 0.0.1
		*/	
		double dihedralAngle(zVector &v1, zVector &v2)
		{
			zVector e(x, y, z);
			
			v1.normalize();
			v2.normalize();
			double dot = v1 * v2;

			zVector cross = v1 ^ v2;
			double  dtheta = atan2(e * cross, dot);	

			return(dtheta * (180.0 / PI));
		}

	
		/*! \brief This method returns the contangetn of the angle between the current and input vector. 
		*
		*	\details Based on http://multires.caltech.edu/pubs/diffGeoOps.pdf and http://rodolphe-vaillant.fr/?e=69
		*	\param		[in]	v			- input vector.
		*	\return				double		- cotangent of angle between the vectors.
		*	\since version 0.0.1
		*/
		double cotan(zVector &v)
		{
			zVector u(x, y, z);

			double dot = u * v;

			//using http://multires.caltech.edu/pubs/diffGeoOps.pdf
			//double denom = (u*u) * (v*v) - (dot*dot);	
			//if (denom == 0) return 0.0;
			//else return dot / sqrt(denom);

			//using http://rodolphe-vaillant.fr/?e=69
			double denom = (u ^ v).length();

			if (denom == 0) return 0.0;
			else return dot / denom;
		}

		/*! \brief This method gets the components as a array of doubles of the current zVector.
		*
		*	\param		[out]	vals		- output compnent values. ( 0 - x component, 1 - y component, 2 - z component).
		*	\since version 0.0.2
		*/
		void getComponents(double3 &_vals)
		{
			_vals[0] = x;
			_vals[1] = y;
			_vals[2] = z;
		}

		/*! \brief This method gets the raw pointer to the components.
		*
		*	\return				double*		- pointer to the first value of the components.
		*	\since version 0.0.2
		*/
		double* getRawComponents()
		{
			return *vals;
		}

		/*! \brief This method returns the row matrix of the current zVector.
		*
		*	\param		[in]	cols			- number of columns in the output vector. Needs to be 4 or 3.
		*	\return				zMatrixd		- row matrix of the vector.
		*	\since version 0.0.1
		*/
		zMatrixd toRowMatrix(int cols = 4)
		{
			vector<double> vals;
		
			if(cols == 4) vals = { x,y,z,1 };
			if(cols == 3) vals = { x,y,z };

			return zMatrixd(1, cols, vals);
		}

		/*! \brief This method returns the column matrix of the current zVector.
		*
		*	\param		[in]	rows			- number of rows in the output vector. Needs to be 4 or 3.
		*	\return				zMatrixd		- column matrix of the vector.
		*	\since version 0.0.1
		*/
		zMatrixd toColumnMatrix(int rows = 4)
		{
			vector<double> vals = { x,y,z,1 };

			if (rows == 4) vals = { x,y,z,1 };
			if (rows == 3) vals = { x,y,z };
			return zMatrixd(rows, 1, vals);
		}

		/*! \brief This method returns the vector from the input row matrix.
		*
		*	\param		[in]		zMatrixd	- input row matrix. Works with 1X3 or 1X4 matrix.
		*	\return					zVector		- zVector of the row matrix.
		*	\since version 0.0.1
		*/
		zVector fromRowMatrix(zMatrixd &inMatrix)
		{
			if (inMatrix.getNumRows() != 1) throw std::invalid_argument("input Matrix is not a row matrix.");
			if (inMatrix.getNumCols() < 3 || inMatrix.getNumCols() > 4) throw std::invalid_argument("cannot convert row matrix to vector.");

			return zVector(inMatrix(0,0), inMatrix(0, 1), inMatrix(0, 2));
		}

		/*! \brief This method returns the vector from the input column matrix.
		*
		*	\param		[in]		zMatrixd	- input column matrix. Works with 3X1 or 4X1 matrix.
		*	\return					zVector		- zVector of the column matrix.
		*	\since version 0.0.1
		*/
		zVector fromColumnMatrix(zMatrixd &inMatrix)
		{
			if (inMatrix.getNumCols() != 1) throw std::invalid_argument("input Matrix is not a column matrix.");
			if (inMatrix.getNumRows() < 3 || inMatrix.getNumRows() > 4) throw std::invalid_argument("cannot convert column matrix to vector.");

			return zVector(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		}
		

		/*! \brief This method returns the rotated vector of the current vector about an input axis by the the input angle.
		*
		*	\param		[in]		axisVec			- axis of rotation.
		*	\param		[in]		angle			- rotation angle.
		*	\return					zVector			- rotated vector.
		*	\since version 0.0.1
		*/
		zVector rotateAboutAxis(zVector axisVec, double angle = 0)
		{
			axisVec.normalize();

			double theta = DEG_TO_RAD * angle;

			zMatrixd rotationMatrix = zMatrixd(3, 3);

			rotationMatrix(0, 0) = cos(theta) + (axisVec.x * axisVec.x) * (1 - cos(theta));
			rotationMatrix(1, 0) = (axisVec.x * axisVec.y) * (1 - cos(theta)) - (axisVec.z) * (sin(theta));
			rotationMatrix(2, 0) = (axisVec.x * axisVec.z) * (1 - cos(theta)) + (axisVec.y) * (sin(theta));

			rotationMatrix(0, 1) = (axisVec.x * axisVec.y) * (1 - cos(theta)) + (axisVec.z) * (sin(theta));
			rotationMatrix(1, 1) = cos(theta) + (axisVec.y * axisVec.y) * (1 - cos(theta));
			rotationMatrix(2, 1) = (axisVec.y * axisVec.z) * (1 - cos(theta)) - (axisVec.x) * (sin(theta));

			rotationMatrix(0, 2) = (axisVec.x * axisVec.z) * (1 - cos(theta)) - (axisVec.y) * (sin(theta));
			rotationMatrix(1, 2) = (axisVec.z * axisVec.y) * (1 - cos(theta)) + (axisVec.x) * (sin(theta));
			rotationMatrix(2, 2) = cos(theta) + (axisVec.z * axisVec.z) * (1 - cos(theta));


			zVector out = this->operator* ( rotationMatrix);

			return out;
		}


	
	
	};
}

