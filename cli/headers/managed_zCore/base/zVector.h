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

	/// <summary>
	///	Overloaded constructor.
	///	<param name="_x"> x component of the zVector.</param>
	///	<param name="_y"> y component of the zVector.</param>
	///	<param name="_z"> z component of the zVector.</param>
	///	</summary>
	

	public ref class zVector : public zManagedObj<zSpace::zVector>
	{
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
		zVector(const zVector ^ &v1);

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
		///	<returns> value of the component.</param>
		///	</summary>	
		double operator[](int index);

		/// <summary>
		///	This operator is used for vector addition.
		///	<param name= "v1"> zVector which is added to the current vector.</param>
		///	<returns> resultant vector after the addition..</param>
		///	</summary>
		static zVector^ operator+(const zVector ^v1,const zVector ^v2);

		/// <summary>
		///	This operator is used for vector subtraction.
		///	<param name= "v1"> zVector which is subtracted from the current vector.</param>
		///	<returns> resultant vector after the subtraction.</param>
		///	</summary>
		static zVector^ operator-(const zVector ^v1, const zVector ^v2);

		/// <summary>
		///	This operator is used for vector dot product.
		///	<param name= "v1"> zVector which is used for the dot product with the current vector.</param>
		///	<returns> resultant value after the dot product.</param>
		///	</summary>
		static double operator*(const zVector ^v1,const zVector ^v2);

		/// <summary>
		///	This operator is used for vector cross procduct.
		///	<param name= "v1"> zVector which is used for the cross product with the current vector.</param>
		///	<returns> resultant vector after the cross product.</param>
		///	</summary>
		static zVector^ operator^(const zVector ^v1, const zVector ^v2);

		/// <summary>
		///	This operator is used for scalar addition of a vector.
		///	<param name= "val"> scalar value to be added to the current vector.</param>
		///	<returns> resultant vector after the scalar addition.</param>
		///	</summary>
		static zVector^ operator+(const zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar subtraction of a vector.
		///	<param name= "val"> scalar value to be subtracted from the current vector.</param>
		///	<returns> resultant vector after the scalar subtraction.</param>
		///	</summary>
		static zVector^ operator-(const zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar muliplication of a vector.
		///	<param name= "val"> scalar value to be multiplied with the current vector.</param>
		///	<returns> resultant vector after the scalar multiplication.</param>
		///	</summary>
		static zVector^ operator*(const zVector ^v1, double val);

		/// <summary>
		///	This operator is used for scalar division of a vector.
		///	<param name= "val"> scalar value used to divide the current vector.</param>
		///	<returns> resultant vector after the scalar division.</param>
		///	</summary>
		static zVector^ operator/(const zVector ^v1, double val);

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