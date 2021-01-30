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



#ifndef ZSPACE_INTEROP_RHINOLIB_
#define ZSPACE_INTEROP_RHINOLIB_

#pragma once
#include <headers/zInterOp/include/zRhinoInclude.h>
#include <delayimp.h>
#include "headers/zInterOp/core/zRhinoCore.h"



#ifndef DLI_HOOK
#define DLI_HOOK

static FARPROC WINAPI DliRhinoLibrary(unsigned dliNotify, PDelayLoadInfo pdli)
{
	static const wchar_t* RhinoLibraryPath = L"C:\\Program Files\\Rhino 7\\System\\RhinoLibrary.dll";

	if (dliNotify == dliNotePreLoadLibrary && _stricmp(pdli->szDll, "RhinoLibrary.dll") == 0)
		return (FARPROC)LoadLibraryEx(RhinoLibraryPath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);

	return 0;
}



 const PfnDliHook __pfnDliNotifyHook2 = DliRhinoLibrary;

#endif


extern "C" HRESULT StartupInProcess(int argc, wchar_t** argv, const STARTUPINFO* pStartUpInfo, HWND hHostWnd);
extern "C" HRESULT ShutdownInProcess();

namespace zSpace
{
	zRhinoCore::zRhinoCore(){}

	zRhinoCore::~zRhinoCore(){}

	void zRhinoCore::startUp()
	{

		StartupInProcess(0, nullptr, nullptr, HWND_DESKTOP);
	}

	void zRhinoCore::shutDown()
	{
		ShutdownInProcess();
	}


}

#endif