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


#include <filesystem>
namespace fs = std::filesystem;

/*! \brief This method compares the time of creation of the two input file.
*
*	\param		[in]	first			- input path 1.
*	\param		[in]	second			- input path 2.
*	\return				bool			- true if first file is created before the second file .
*	\since version 0.0.2
*/
inline bool compare_time_creation(fs::path& first, const fs::path& second)
{
	auto timeFirst = fs::last_write_time(first); 
	

	auto timeSec = fs::last_write_time(second);

	return (timeFirst < timeSec);
}

