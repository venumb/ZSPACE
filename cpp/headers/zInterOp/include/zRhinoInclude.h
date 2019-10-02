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

#if defined(ZSPACE_RHINO_INTEROP) 

#ifndef ZSPACE_RHINO_INCLUDE_H
#define ZSPACE_RHINO_INCLUDE_H

#pragma once

//--------------------------
//---- windows include
//--------------------------
#include<Unknwnbase.h>
#include<commdlg.h>

#include <iomanip>

#if defined(WIN32) 
#undef WIN32
#endif

#include <rhinoSdkStdafxPreamble.h>

#define NOMINMAX
#include <Windows.h>

#define RHINO_V6_READY

#pragma warning (disable: 4251)

#include <rhinoSdk.h>

#endif

#endif