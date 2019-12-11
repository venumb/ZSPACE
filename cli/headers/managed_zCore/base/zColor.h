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

	public ref class zColor : public zManagedObj<zSpace::zColor>
	{
	public:

		zColor();

		zColor(double _r, double _g, double _b, double _a);

		zColor(double _h, double _s, double _v);

		/// <summary>
		/// Copy Constructor.
		/// </summary>
		zColor(const zColor ^ &c1);

		void toHSV();

		void toRGB();

		bool operator==(zColor c1);

		property double r
		{
		public:
			double get()
			{
				return m_zInstance->r;
			}

			void set(double value)
			{
				m_zInstance->r = value;
			}
		}

		property double g
		{
		public:
			double get()
			{
				return m_zInstance->g;
			}

			void set(double value)
			{
				m_zInstance->g = value;
			}
		}

		property double b
		{
		public:
			double get()
			{
				return m_zInstance->b;
			}

			void set(double value)
			{
				m_zInstance->b = value;
			}
		}

		property double a
		{
		public:
			double get()
			{
				return m_zInstance->a;
			}

			void set(double value)
			{
				m_zInstance->a = value;
			}
		}

		property double h
		{
		public:
			double get()
			{
				return m_zInstance->h;
			}

			void set(double value)
			{
				m_zInstance->h = value;
			}
		}

		property double s
		{
		public:
			double get()
			{
				return m_zInstance->s;
			}

			void set(double value)
			{
				m_zInstance->s = value;
			}
		}
		

		property double v
		{
		public:
			double get()
			{
				return m_zInstance->v;
			}

			void set(double value)
			{
				m_zInstance->v = value;
			}
		}
	};

}