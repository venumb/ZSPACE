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

#ifndef ZSPACE_UTILS_BMP_H
#define ZSPACE_UTILS_BMP_H

#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

#include<headers/zCore/base/zInline.h>

namespace zSpace
{
	

#pragma pack(push, 1)
	
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/
	
	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/** \addtogroup zBMP
	*	\brief The bitmap utility structs of the library.
	*  @{
	*/

	/*! \struct BMPFileHeader
	*
	*	\brief A bitmap file header.
	*	\details  All BMP images starts with a five elements file header. This has information about the file type, file size and location of the pixel data. Explanation taken from https://solarianprogrammer.com/2018/11/19/cpp-reading-writing-bmp-images/
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	
	struct ZSPACE_CORE BMPFileHeader
	{
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*!	\brief File type always BM which is 0x4D42  */
		uint16_t file_type{ 0x4D42 };          

		/*!	\brief Size of the file (in bytes)  */
		uint32_t file_size{ 0 }; 			
		 
		/*!	\brief Reserved, always 0  */
		uint16_t reserved1{ 0 };   

		/*!	\brief Reserved, always 0  */
		uint16_t reserved2{ 0 }; 

		/*!	\brief Start position of pixel data (bytes from the beginning of the file)  */
		uint32_t offset_data{ 0 };             
		
	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/** \addtogroup zBMP
	*	\brief The bitmap utility structs of the library.
	*  @{
	*/

	/*! \struct BMPInfoHeader
	*
	*	\brief A bitmap information header.
	*	\details  Also named the info header. This has information about the width/height of the image, bits depth and so on. Explanation taken from https://solarianprogrammer.com/2018/11/19/cpp-reading-writing-bmp-images/
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE BMPInfoHeader
	{
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*!	\brief Size of this header (in bytes) */
		uint32_t size{ 0 };

		/*!	\brief/ width of bitmap in pixels */
		int32_t width{ 0 }; 

		/*!	\brief height of bitmap in pixels (if positive, bottom-up, with origin in lower left corner. if negative, top-down, with origin in upper left corner) */
		int32_t height{ 0 };                     
			
		/*!	\brief No. of planes for the target device, this is always 1 */
		uint16_t planes{ 1 };   

		/*!	\brief No. of bits per pixel */
		uint16_t bit_count{ 0 };  

		/*!	\brief 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images */
		uint32_t compression{ 0 };   

		/*!	\brief size of bitmap image. 0 - for uncompressed images */
		uint32_t size_image{ 0 };  

		/*!	\brief pixels in x per meter of bitmap image. 0 - for uncompressed images */
		int32_t x_pixels_per_meter{ 0 };

		/*!	\brief pixels in y per meter of bitmap image. 0 - for uncompressed images */
		int32_t y_pixels_per_meter{ 0 };

		/*!	\brief No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count */
		uint32_t colors_used{ 0 }; 

		/*!	\brief No. of colors used for displaying the bitmap. If 0 all colors are required */
		uint32_t colors_important{ 0 };          
		
	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/** \addtogroup zBMP
	*	\brief The bitmap utility structs of the library.
	*  @{
	*/

	/*! \struct BMPColorHeader
	*
	*	\brief A bitmap color header.
	*	\details Contains informations about the color space and bit masks. Explanation taken from https://solarianprogrammer.com/2018/11/19/cpp-reading-writing-bmp-images/
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	struct ZSPACE_CORE BMPColorHeader
	{
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*!	\brief Bit mask for the red channel */
		uint32_t red_mask{ 0x00ff0000 };   

		/*!	\brief Bit mask for the green channel */
		uint32_t green_mask{ 0x0000ff00 };

		/*!	\brief Bit mask for the blue channel */
		uint32_t blue_mask{ 0x000000ff };  

		/*!	\brief Bit mask for the alpha channel */
		uint32_t alpha_mask{ 0xff000000 };   

		/*!	\brief Default "sRGB" (0x73524742) */
		uint32_t color_space_type{ 0x73524742 }; 

		/*!	\brief Unused data for sRGB color space  */  
		uint32_t unused[16]{ 0 };                
		
	};
#pragma pack(pop)

	   
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \struct zUtilsBMP
	*
	*	\brief A bitmap struct to define methods to read, write and create bitmaps.
	*	\details  Explanation taken from https://solarianprogrammer.com/2018/11/19/cpp-reading-writing-bmp-images/
	*/

	/** @}*/

	/** @}*/
	   
	class ZSPACE_CORE zUtilsBMP
	{
	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief Stores BMP file header  */
		BMPFileHeader file_header;

		/*!	\brief Stores BMP information header  */
		BMPInfoHeader bmp_info_header;

		/*!	\brief Stores BMP color header information  */
		BMPColorHeader bmp_color_header;

		/*!	\brief Storeds BMP data  */
		std::vector<uint8_t> data;

		//--------------------------
		//----  CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zUtilsBMP();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	fname			- file name.
		*	\since version 0.0.1
		*/
		zUtilsBMP(const char *fname);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	width			- width of bitmap.
		*	\param		[in]	height			- height of bitmap.
		*	\param		[in]	has_alpha		- true if there needs to be a alpha channel.
		*	\since version 0.0.1
		*/
		zUtilsBMP(int32_t width, int32_t height, bool has_alpha = true);


		//--------------------------
		//----  DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zUtilsBMP();

		//--------------------------
		//----  METHODS
		//--------------------------

		/*! \brief This methods reads the input bitmap file.
		*
		*	\param [in]		fname			- input file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		void read(const char *fname);
	
		/*! \brief This methods writes the bitmap to the output file.
		*
		*	\param [in]		fname			- output file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		void write(const char *fname);

	private:
		//--------------------------
		//----  PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief Stores row stride  */
		uint32_t row_stride{ 0 };

		//--------------------------
		//----  PRIVATE METHODS
		//--------------------------

		/*! \brief This methods writes the headers of the bitmap file.
		*
		*	\param [in]		of			- stream of the output file.
		*	\since version 0.0.1
		*/
		void write_headers(std::ofstream &of);
		
		/*! \brief This methods writes the headers and data of the bitmap file.
		*
		*	\param [in]		of			- stream of the output file.
		*	\since version 0.0.1
		*/
		void write_headers_and_data(std::ofstream &of);

		/*! \brief This methods checks if the pixel data is stored as BGRA and if the color space type is sRGB
		*
		*	\param [in]		bmp_color_header	- input bitmap color header.
		*	\since version 0.0.1
		*/
		void check_color_header(BMPColorHeader &bmp_color_header);

		/*! \brief This methods alings the strides. It adds 1 to the row_stride until it is divisible with align_stride
		*
		*	\param [in]		align_stride	- input stride.
		*	\since version 0.0.1
		*/
		uint32_t make_stride_aligned(uint32_t align_stride);
			
	};
	

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/utilities/zUtilsBMP.cpp>
#endif

#endif