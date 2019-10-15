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

#include<headers/zCore/utilities/zUtilsBMP.h>

namespace zSpace
{

	//----  CONSTRUCTOR

	ZSPACE_INLINE zUtilsBMP::zUtilsBMP() {}

	ZSPACE_INLINE zUtilsBMP::zUtilsBMP(const char *fname)
	{
		read(fname);

	}

	ZSPACE_INLINE zUtilsBMP::zUtilsBMP(int32_t width, int32_t height, bool has_alpha)
	{
		if (width <= 0 || height <= 0)
		{
			throw std::runtime_error("The image width and height must be positive numbers.");
		}

		bmp_info_header.width = width;
		bmp_info_header.height = height * -1;
		if (has_alpha)
		{
			bmp_info_header.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
			file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);

			bmp_info_header.bit_count = 32;
			bmp_info_header.compression = 3;
			row_stride = width * 4;
			data.resize(row_stride * height);
			file_header.file_size = file_header.offset_data + data.size();
		}
		else
		{
			bmp_info_header.size = sizeof(BMPInfoHeader);
			file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

			bmp_info_header.bit_count = 24;
			bmp_info_header.compression = 0;
			row_stride = width * 3;
			data.resize(row_stride * height);

			uint32_t new_stride = make_stride_aligned(4);
			file_header.file_size = file_header.offset_data + data.size() + bmp_info_header.height * (new_stride - row_stride);
		}

	}

	//----  DESTRUCTOR

	ZSPACE_INLINE zUtilsBMP::~zUtilsBMP() {}

	//----  METHODS

	ZSPACE_INLINE void zUtilsBMP::read(const char *fname)
	{
		std::ifstream inp{ fname, std::ios_base::binary };
		if (inp)
		{
			inp.read((char*)&file_header, sizeof(file_header));
			if (file_header.file_type != 0x4D42)
			{
				throw std::runtime_error("Error! Unrecognized file format.");

			}
			inp.read((char*)&bmp_info_header, sizeof(bmp_info_header));

			// The BMPColorHeader is used only for transparent images
			if (bmp_info_header.bit_count == 32)
			{
				// Check if the file has bit mask color information
				if (bmp_info_header.size >= (sizeof(BMPInfoHeader) + sizeof(BMPColorHeader)))
				{
					inp.read((char*)&bmp_color_header, sizeof(bmp_color_header));
					// Check if the pixel data is stored as BGRA and if the color space type is sRGB
					check_color_header(bmp_color_header);

				}
				else
				{
					std::cerr << "Warning! The file \"" << fname << "\" does not seem to contain bit mask information\n";
					throw std::runtime_error("Error! Unrecognized file format.");

				}

			}

			// Jump to the pixel data location
			inp.seekg(file_header.offset_data, inp.beg);

			// Adjust the header fields for output.
			// Some editors will put extra info in the image file, we only save the headers and the data.
			if (bmp_info_header.bit_count == 32)
			{
				bmp_info_header.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
				file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);

			}
			else {
				bmp_info_header.size = sizeof(BMPInfoHeader);
				file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

			}
			file_header.file_size = file_header.offset_data;

			if (bmp_info_header.height < 0)
			{
				throw std::runtime_error("The program can treat only BMP images with the origin in the bottom left corner!");

			}

			data.resize(bmp_info_header.width * bmp_info_header.height * bmp_info_header.bit_count / 8);

			// Here we check if we need to take into account row padding
			if (bmp_info_header.width % 4 == 0) {
				inp.read((char*)data.data(), data.size());
				file_header.file_size += data.size();

			}
			else
			{
				row_stride = bmp_info_header.width * bmp_info_header.bit_count / 8;
				uint32_t new_stride = make_stride_aligned(4);
				std::vector<uint8_t> padding_row(new_stride - row_stride);

				for (int y = 0; y < bmp_info_header.height; ++y)
				{
					inp.read((char*)(data.data() + row_stride * y), row_stride);
					inp.read((char*)padding_row.data(), padding_row.size());

				}
				file_header.file_size += data.size() + bmp_info_header.height * padding_row.size();

			}

		}
		else
		{
			throw std::runtime_error("Unable to open the input image file.");

		}

	}

	ZSPACE_INLINE void zUtilsBMP::write(const char *fname)
	{
		std::ofstream of{ fname, std::ios_base::binary };
		if (of)
		{
			if (bmp_info_header.bit_count == 32)
			{
				write_headers_and_data(of);
			}
			else if (bmp_info_header.bit_count == 24)
			{
				if (bmp_info_header.width % 4 == 0)
				{
					write_headers_and_data(of);
				}
				else
				{
					uint32_t new_stride = make_stride_aligned(4);
					std::vector<uint8_t> padding_row(new_stride - row_stride);

					write_headers(of);

					for (int y = 0; y < bmp_info_header.height; ++y)
					{
						of.write((const char*)(data.data() + row_stride * y), row_stride);
						of.write((const char*)padding_row.data(), padding_row.size());
					}

				}

			}
			else
			{
				throw std::runtime_error("The program can treat only 24 or 32 bits per pixel BMP files");
			}

		}
		else
		{
			throw std::runtime_error("Unable to open the output image file.");
		}

	}

	//----  PRIVATE METHODS

	ZSPACE_INLINE void zUtilsBMP::write_headers(std::ofstream &of)
	{
		of.write((const char*)&file_header, sizeof(file_header));
		of.write((const char*)&bmp_info_header, sizeof(bmp_info_header));
		if (bmp_info_header.bit_count == 32) {
			of.write((const char*)&bmp_color_header, sizeof(bmp_color_header));

		}

	}

	ZSPACE_INLINE void zUtilsBMP::write_headers_and_data(std::ofstream &of)
	{
		write_headers(of);
		of.write((const char*)data.data(), data.size());

	}

	ZSPACE_INLINE void zUtilsBMP::check_color_header(BMPColorHeader &bmp_color_header)
	{
		BMPColorHeader expected_color_header;
		if (expected_color_header.red_mask != bmp_color_header.red_mask ||
			expected_color_header.blue_mask != bmp_color_header.blue_mask ||
			expected_color_header.green_mask != bmp_color_header.green_mask ||
			expected_color_header.alpha_mask != bmp_color_header.alpha_mask)
		{
			throw std::runtime_error("Unexpected color mask format! The program expects the pixel data to be in the BGRA format");
		}

		if (expected_color_header.color_space_type != bmp_color_header.color_space_type)
		{
			throw std::runtime_error("Unexpected color space type! The program expects sRGB values");
		}

	}

	ZSPACE_INLINE uint32_t zUtilsBMP::make_stride_aligned(uint32_t align_stride)
	{
		uint32_t new_stride = row_stride;
		while (new_stride % align_stride != 0)
		{
			new_stride++;
		}
		return new_stride;
	}

}