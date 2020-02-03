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

using namespace System;
using namespace System::Runtime::InteropServices;

namespace zSpaceManaged 
{

	template<class T>
	public ref class zManagedObj
	{
	protected:

		T* m_zInstance;

	public:
		zManagedObj() {}

		zManagedObj(T* cppObj)
			: m_zInstance(cppObj)
		{
		}

		virtual ~zManagedObj()
		{
			if (m_zInstance != nullptr)
			{
				delete m_zInstance;
			}
		}

		!zManagedObj()
		{
			if (m_zInstance != nullptr)
			{
				delete m_zInstance;
			}
		}

		T* GetObject()
		{
			return m_zInstance;
		}
		
		static const char* string_to_char_array(String^ string)
		{
			const char* str = (const char*)(Marshal::StringToHGlobalAnsi(string)).ToPointer();
			return str;
		}

	};
}

#endif