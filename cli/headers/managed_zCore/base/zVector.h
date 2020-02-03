// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#ifdef ZSPACE_MANAGED_LIBRARY

#pragma once


#include<headers/managed_zCore/managedObject/zManagedObj.h>
#include<headers/zCore/zCore.h>

using namespace System;


namespace zSpaceManaged
{
	/// <summary>
	/// Class <c>zVector</c> 
	/// A 3 dimensional vector math class.
	/// </summary>		

	public ref class zVector : public zManagedObj<zSpace::zVector>
	{
	protected:
				

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/// <summary>
		/// Default Constructor.
		/// </summary>
		zVector();

		/// <summary>
		///	Overloaded constructor.
		///	<param name="_x"> x component of the zVector.</param>
		///	<param name="_y"> y component of the zVector.</param>
		///	<param name="_z"> z component of the zVector.</param>
		///	</summary>		
		zVector(double _x, double _y, double _z);	

		/// <summary>
		/// Copy Constructor.
		/// </summary>
		zVector(zSpace::zVector &v1);

		//--------------------------
		//---- OPERATORS
		//--------------------------	

		/// <summary>
		///	This operator checks for equality of two zVectors.
		///	<param name= "v1"> zVector against which the equality is checked.</param>
		///	<returns> true if vectors are equal.</param>
		///	</summary>	
		bool operator==(zVector ^ v1);

		/// <summary>
		///	This method returns the component value at the input index of the current zVector.
		///	<param name= "index">  0 - x component, 1 - y component, 2 - z component.</param>
		///	<returns> value of the component.</returns>
		///	</summary>	
		double operator[](int index);

		/// <summary>
		///	This operator is used for vector addition.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "v2"> zVector which is added to the current vector.</param>
		///	<returns> resultant vector after the addition.</returns>
		///	</summary>
		static zVector^ operator+( zVector ^v1,const zVector ^v2);

		/// <summary>
		///	This operator is used for vector subtraction.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "v2"> zVector which is subtracted from the current vector.</param>
		///	<returns> resultant vector after the subtraction.</returns>
		///	</summary>
		static zVector^ operator-( zVector ^v1, const zVector ^v2);

		/// <summary>
		///	This operator is used for vector dot product.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "v2"> zVector which is used for the dot product with the current vector.</param>
		///	<returns> resultant value after the dot product.</returns>
		///	</summary>
		static double operator*( zVector ^v1,const zVector ^v2);

		/// <summary>
		///	This operator is used for vector cross procduct.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "v2"> zVector which is used for the cross product with the current vector.</param>
		///	<returns> resultant vector after the cross product.</returns>
		///	</summary>
		static zVector^ operator^( zVector ^v1, const zVector ^v2);

		/// <summary>
		///	This operator is used for scalar addition of a vector.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "val"> scalar value to be added to the current vector.</param>
		///	<returns> resultant vector after the scalar addition.</returns>
		///	</summary>
		static zVector^ operator+( zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar subtraction of a vector.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "val"> scalar value to be subtracted from the current vector.</param>
		///	<returns> resultant vector after the scalar subtraction.</returns>
		///	</summary>
		static zVector^ operator-( zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar muliplication of a vector.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "val"> scalar value to be multiplied with the current vector.</param>
		///	<returns> resultant vector after the scalar multiplication.</returns>
		///	</summary>
		static zVector^ operator*( zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar division of a vector.
		///	<param name= "v1"> The current vector.</param>
		///	<param name= "val"> scalar value used to divide the current vector.</param>
		///	<returns> resultant vector after the scalar division.</returns>
		///	</summary>
		static zVector^ operator/( zVector ^v1, double val);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/// <summary>
		///	This overloaded operator is used for vector addition and assigment of the result to the current vector.
		///	<param name= "v1"> zVector which is added to the current vector.</param>
		///	<returns> resultant vector after the vector additon.</returns>
		///	</summary>
		zVector^ operator+=(const zVector ^v1);

		/// <summary>
		///	This overloaded operator is used for vector subtraction and assigment of the result to the current vector.
		///	<param name= "v1"> zVector which is subtracted from the current vector.</param>
		///	<returns> resultant vector after the vector additon.</returns>
		///	</summary>
		zVector^ operator-=(const zVector ^v1);

		/// <summary>
		///	This overloaded operator is used for scalar addition and assigment of the result to the current vector.
		///	<param name= "val"> scalar value to be added the current vector.</param>
		///	<returns> resultant vector after the scalar additon.</returns>
		///	</summary>
		zVector^ operator+=(double val);

		/// <summary>
		///	This overloaded operator is used for scalar subtraction and assigment of the result to the current vector.
		///	<param name= "val"> scalar value to be subtracted from the current vector.</param>
		///	<returns> resultant vector after the scalar subtraction.</returns>
		///	</summary>
		zVector^ operator-=(double val);

		/// <summary>
		///	This overloaded operator is used for scalar multiplication and assigment of the result to the current vector.
		///	<param name= "val"> scalar value to be multipled with the current vector.</param>
		///	<returns> resultant vector after the scalar multiplication.</returns>
		///	</summary>
		zVector^ operator*=(double val);

		/// <summary>
		///	This overloaded operator is used for scalar division and assigment of the result to the current vector.
		///	<param name= "val"> scalar value used to divide from the current vector.</param>
		///	<returns> resultant vector after the scalar division.</returns>
		///	</summary>
		zVector^ operator/=(double val);

		//--------------------------
		//---- METHODS
		//--------------------------

		/// <summary>
		///	This method returns the squared length of the vector.
		///	<returns> value of the squared maginute of the vector.</returns>
		///	</summary>
		double length2();

		/// <summary>
		///	This method returns the magnitude/length of the vector.
		///	<returns> value of the maginute of the vector.</returns>
		///	</summary>
		double length();

		/// <summary>
		///	This method normalizes the vector to unit length.
		///	</summary>
		void normalize();

		/// <summary>
		///	This method returns the squared distance between the current vector and input vector.
		///	<param name= "v1"> input vector.</param>
		///	<returns> squared value of the distance between the vectors..</returns>
		///	</summary>
		double squareDistanceTo(zVector ^v1);

		/// <summary>
		///	This method returns the distance between the current vector and input vector.
		///	<param name= "v1"> input vector.</param>
		///	<returns> value of the distance between the vectors.</returns>
		///	</summary>
		double distanceTo(zVector ^v1);

		/// <summary>
		///	This method returns the angle between the current vector and input vector.
		///	<param name= "v1"> input vector.</param>
		///	<returns> value of the angle between the vectors.</returns>
		///	</summary>
		double angle(zVector ^v1);

		/// <summary>
		///	This method returns the angle between the current vector and input vector in the range of 0 to 360 in the plane given by the input normal.
		///	<param name= "v1"> input vector.</param>
		///	<param name= "normal"> input reference normal or axis of rotation..</param>
		///	<returns> value of the angle between the vectors.</returns>
		///	</summary>
		double angle360(zVector ^v1, zVector ^normal);

		/// <summary>
		///	This method returns the dihedral angle between the two input vectors using current vector as edge reference.
		/// <remarks> Based on https://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleMeshDerivativesCheatSheet.pdf </remarks>
		///	<param name= "v1"> input vector.</param>
		///	<param name= "v2"> input vector.</param>
		///	<returns> value of the dihedral angle between the vectors.</returns>
		///	</summary>
		double dihedralAngle(zVector ^v1, zVector ^v2);

		/// <summary>
		///	This method returns the contangent of the angle between the current and input vector. 
		/// <remarks> Based on http://multires.caltech.edu/pubs/diffGeoOps.pdf and http://rodolphe-vaillant.fr/?e=69 </remarks>
		///	<param name= "v"> input vector.</param>
		///	<returns> value of the cotangent of angle between the vectors.</returns>
		///	</summary>
		double cotan(zVector ^v);

		/// <summary>
		///	This method returns the rotated vector of the current vector about an input axis by the the input angle.
		///	<param name= "axisVec"> axis of rotation.</param>
		///	<param name= "angle"> rotation angle.</param>
		///	<returns> rotated vector.</returns>
		///	</summary>
		zVector^ rotateAboutAxis(zVector ^axisVec, double angle);

		//--------------------------
		//---- PROPERTY
		//--------------------------

		property double x
		{
		public:
			double get()
			{
				return m_zInstance->x;
			}

			void set(double value)
			{
				m_zInstance->x = value;
			}
		}

		property double y
		{
		public:
			double get()
			{
				return m_zInstance->y;
			}

			void set(double value)
			{
				m_zInstance->y = value;
			}
		}

		property double z
		{
		public:
			double get()
			{
				return m_zInstance->z;
			}

			void set(double value)
			{
				m_zInstance->z = value;
			}
		}
		
	};

}

#endif