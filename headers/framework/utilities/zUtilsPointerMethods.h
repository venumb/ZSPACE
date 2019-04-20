#pragma once

#include <filesystem>
namespace fs = std::experimental::filesystem;

/*! \brief This method compares the time of creation of the two input file.
*
*	\param		[in]	first			- input path 1.
*	\param		[in]	second			- input path 2.
*	\return				bool			- true if first file is created before the second file .
*	\since version 0.0.2
*/
bool compare_time_creation(fs::path& first, const fs::path& second)
{
	auto timeFirst = fs::last_write_time(first);

	auto timeSec = fs::last_write_time(second);

	return (timeFirst < timeSec);
}