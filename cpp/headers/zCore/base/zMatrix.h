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

#ifndef ZSPACE_MATRIX_H
#define ZSPACE_MATRIX_H

#pragma once

#include "headers/zCore/base/zInline.h"

#include <stdexcept>
#include <vector>
#include <queue>
#include <tuple>

#include <array>

using namespace std;

#ifndef __CUDACC__
	#include<depends/Eigen/Core>
	#include<depends/Eigen/Dense>
	#include<depends/Eigen/Sparse>
	#include<depends/Eigen/Eigen>
	#include<depends/Eigen/Sparse>
	using namespace Eigen;

#ifndef USING_CLR
	#include <depends/Armadillo/armadillo>
	using namespace arma;
#endif

#endif

namespace zSpace
{ 
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \typedef zMatrix4Row
	*	\brief An array of float of size 4.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix4Row[4];

	/*! \typedef zMatrix4Col
	*	\brief An array of float of size 4.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix4Col[4];

	/*! \typedef zMatrix3Diag
	*	\brief An array of float of size 4.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix4Diag[4];

	/*! \typedef zMatrix3Row
	*	\brief An array of float of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix3Row[3];

	/*! \typedef zMatrix3Col
	*	\brief An array of float of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix3Col[3];

	/*! \typedef zMatrix3Diag
	*	\brief An array of float of size 3.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix3Diag[3];

	/*! \typedef zMatrix2Row
*	\brief An array of float of size 2.
*
*	\since version 0.0.2
*/
	typedef float zMatrix2Row[2];

	/*! \typedef zMatrix2Col
	*	\brief An array of float of size 2.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix2Col[2];

	/*! \typedef zMatrix3Diag
	*	\brief An array of float of size 2.
	*
	*	\since version 0.0.2
	*/
	typedef float zMatrix2Diag[2];

	/** @}*/

	/** @}*/


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zMatrix2
	*	\brief A 2x2 matrix math class.
	*
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_CORE zMatrix2
	{

	protected:
		/*!	\brief list of values in the matrix			*/
		float mat[4];

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default Constructor .
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2();


		/*! \brief Overloaded Constructor for a square matrix.
		*
		*	\param		[in]	_mat		- input values of the matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2(const float _mat[4]);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zMatrix2();


		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index in the matrix value container given the row and column indicies.
		*
		*	\param		[in]	int		- row index.
		*	\param		[in]	int		- column index.
		*	\return				int		- index in value container.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getIndex(int rowIndex, int colIndex);

		/*! \brief This method gets the row and column index in the matrix given the input container index.
		*
		*	\param		[in]	int		- index.
		*	\param		[out]	int		- row index.
		*	\param		[out]	int		- column index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getIndices(int index, int &rowIndex, int &colIndex);

		/*! \brief This method gets the row values as container of values at the input row index.
		*
		*	\param		[in]	index		- input row index.
		*	\return				zMatrix2Row	- container of row values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getRow(int index, zMatrix2Row &row);

		/*! \brief This method gets the column values at the input column index.
		*
		*	\param		[in]	index		- input column index.
		*	\return				zMatrix2Col	- container of column values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getCol(int index, zMatrix2Col &col);

		/*! \brief This method gets the diagonal values.
		*
		*	\return				zMatrix2Diag	- container of diagonal values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getDiagonal(zMatrix2Diag &diag);

		/*! \brief This method gets the mats values container.
		*
		*	\param		[out]	_mat		- size 4 container of matrix values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float* getMatrixValues();

		/*! \brief This method gets pointer to the internal matrix values container.
		*
		*	\param		[out]	float*		- pointer to internal matrix value container.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE float* getRawMatrixValues();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the row values at the input row index with the input value.
		*	
		*	\param		[in]	index		- input row index.
		*	\param		[in]	val			- value of the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, float val);

		/*! \brief This method sets the row values at the input row index with the container vector of values.
		*	
		*	\param		[in]	index		- input row index.
		*	\param		[in]	row			- container of values for the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, zMatrix2Row &row);

		/*! \brief This method sets the column values at the input column index with the input value.
		*	
		*	\param		[in]	index		- input col index.
		*	\param		[in]	val			- value of the col.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, float val);

		/*! \brief This method sets the col values at the input col index with the input container of values.
		*	
		*	\param		[in]	index		- input col index.
		*	\param		[in]	col			- container of values for the column.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, const zMatrix2Col &col);


		/*! \brief This method sets the diagonal values of a square matrix with the input value.
		*	
		*	\param		[in]	val			- value of the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(float val);

		/*! \brief This method sets the diagonal values of the matrix with the input container of values.
		*	
		*	\param		[in]	diag			- container of values for the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(const zMatrix2Diag &diag);

		/*! \brief This method sets values of the matrix to zero.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setZero();

		/*! \brief This method sets values of the matrix to one.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setOne();

		/*! \brief This method sets the matrix to identity.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setIdentity();

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator returns the reference  to the coefficient at at the given row and column index.
		*
		*	\param		[in]	row		- input row index.
		*	\param		[in]	col		- input col index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int row, int col);

		/*! \brief This operator returns the reference  to the coefficient at at the given index.
		*
		*	\param		[in]	id		- input index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int index);

		/*! \brief This operator is used for matrix addition.
		*
		*	\param		[in]	m1		 - zMatrix2 which is added to the current matrix.
		*	\return				zMatrix2 - resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator+ (zMatrix2 &m1);

		/*! \brief This operator is used for scalar addition to a matrix.
		*
		*	\param		[in]	s1			- scalar value which is added to the current matrix.
		*	\return				zMatrix2	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator+ (float s1);

		/*! \brief This operator is used for matrix subtraction.
		*
		*	\param		[in]	m1		    - zMatrix2 which is subtracted from the current matrix.
		*	\return				zMatrix2	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator- (zMatrix2 &m1);

		/*! \brief This operator is used for scalar subtraction from a matrix.
		*
		*	\param		[in]	s1			- scalar value which is subtracted from the current matrix.
		*	\return				zMatrix2	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator- (float s1);

		/*! \brief This operator is used for matrix multiplication.
		*
		*	\param		[in]	m1			- zMatrix2 which is multiplied with the current matrix.
		*	\return				zMatrix2	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator* (zMatrix2 &m1);

		/*! \brief This operator is used for scalar multiplication with a matrix.
		*
		*	\param		[in]	s1			- scalar value which is multiplied with the current matrix.
		*	\return				zMatrix2	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 operator* (float s1);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This operator is used for matrix addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix2 which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (zMatrix2 &m1);

		/*! \brief This operator is used for scalar addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (float s1);

		/*! \brief This operator is used for matrix subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix2 which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (zMatrix2 &m1);

		/*! \brief This operator is used for scalar subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (float s1);

		/*! \brief This operator is used for scalar multiplication and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is multiplied to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator*= (float s1);

		/*! \brief This operator is used for scalar division and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which divides the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator/= (float s1);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method is used for matrix multiplication with a column matrix.
		*
		*	\param		[in]	m1			- column matrix which is multiplied with the current matrix.
		*	\param		[out]	out			- resultant column matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void multiply(zMatrix2Col &m1, zMatrix2Col &out);

		/*! \brief This method returns the transpose of the input matrix.
		*
		*	\return				zMatrix2	- resultant transpose matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 transpose();

		/*! \brief This method returns the determinant of the input matrix if it is a square matrix.
		*
		*	\details Based on https://www.geeksforgeeks.org/determinant-of-a-matrix/
		*	\return				float		- determinant value.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float det();

	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zMatrix3
	*	\brief A 3x3 matrix math class.
	*
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	class ZSPACE_CORE zMatrix3
	{

	protected:
		/*!	\brief list of values in the matrix			*/
		float mat[9];

	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default Constructor .
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3();

		/*! \brief Overloaded Constructor.
		*
		*	\param		[in]	_mat		- input values of the matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3(const float _mat[9]);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zMatrix3();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index in the matrix value container given the row and column indicies.
		*
		*	\param		[in]	int		- row index.
		*	\param		[in]	int		- column index.
		*	\return				int		- index in value container.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getIndex(int rowIndex, int colIndex);

		/*! \brief This method gets the row and column index in the matrix given the input container index.
		*
		*	\param		[in]	int		- index.
		*	\param		[out]	int		- row index.
		*	\param		[out]	int		- column index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getIndices(int index, int &rowIndex, int &colIndex);

		/*! \brief This method gets the row values as container of values at the input row index.
		*
		*	\param		[in]	index		- input row index.
		*	\return				zMatrix3Row	- container of row values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getRow(int index, zMatrix3Row &row);

		/*! \brief This method gets the column values at the input column index.
		*
		*	\param		[in]	index		- input column index.
		*	\return				zMatrix3Col	- container of column values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getCol(int index, zMatrix3Col &col);

		/*! \brief This method gets the diagonal values.
		*
		*	\return				zMatrix3Diag	- container of diagonal values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getDiagonal(zMatrix3Diag &diag);

		/*! \brief This method gets the mats values container.
		*
		*	\param		[out]	_mat		- size 4 container of matrix values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float* getMatrixValues();

		/*! \brief This method gets pointer to the internal matrix values container.
		*
		*	\param		[out]	float*		- pointer to internal matrix value container.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE float* getRawMatrixValues();


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the row values at the input row index with the input value.
		*
		*	\param		[in]	index		- input row index.
		*	\param		[in]	val			- value of the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, float val);

		/*! \brief This method sets the row values at the input row index with the container vector of values.
		*
		*	\param		[in]	index		- input row index.
		*	\param		[in]	row			- container of values for the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, const zMatrix3Row &row);

		/*! \brief This method sets the column values at the input column index with the input value.
		*
		*	\param		[in]	index		- input col index.
		*	\param		[in]	val			- value of the col.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, float val);

		/*! \brief This method sets the col values at the input col index with the input container of values.
		*
		*	\param		[in]	index		- input col index.
		*	\param		[in]	col			- container of values for the column.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, const zMatrix3Col &col);

		/*! \brief This method sets the diagonal values of a square matrix with the input value.
		*
		*	\param		[in]	val			- value of the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(float val);

		/*! \brief This method sets the diagonal values of the matrix with the input container of values.
		*
		*	\param		[in]	diag		- container of values for the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(const zMatrix3Diag &diag);

		/*! \brief This method sets values of the matrix to zero.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setZero();

		/*! \brief This method sets values of the matrix to one.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setOne();

		/*! \brief This method sets the matrix to identity.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setIdentity();

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator returns the reference  to the coefficient at at the given row and column index.
		*
		*	\param		[in]	row		- input row index.
		*	\param		[in]	col		- input col index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int row, int col);

		/*! \brief This operator returns the reference  to the coefficient at at the given index.
		*
		*	\param		[in]	id		- input index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int index);

		/*! \brief This operator is used for matrix addition.
		*
		*	\param		[in]	m1		 - zMatrix3 which is added to the current matrix.
		*	\return				zMatrix3 - resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator+ (zMatrix3 &m1);

		/*! \brief This operator is used for scalar addition to a matrix.
		*
		*	\param		[in]	s1			- scalar value which is added to the current matrix.
		*	\return				zMatrix3	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator+ (float s1);

		/*! \brief This operator is used for matrix subtraction.
		*
		*	\param		[in]	m1		    - zMatrix3 which is subtracted from the current matrix.
		*	\return				zMatrix3	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator- (zMatrix3 &m1);

		/*! \brief This operator is used for scalar subtraction from a matrix.
		*
		*	\param		[in]	s1			- scalar value which is subtracted from the current matrix.
		*	\return				zMatrix3	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator- (float s1);

		/*! \brief This operator is used for matrix multiplication.
		*
		*	\param		[in]	m1			- zMatrix3 which is multiplied with the current matrix.
		*	\return				zMatrix3	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator* (zMatrix3 &m1);


		/*! \brief This operator is used for scalar multiplication with a matrix.
		*
		*	\param		[in]	s1			- scalar value which is multiplied with the current matrix.
		*	\return				zMatrix3	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 operator* (float s1);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------
		
		/*! \brief This operator is used for matrix addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix3 which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (zMatrix3 &m1);

		/*! \brief This operator is used for scalar addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (float s1);

		/*! \brief This operator is used for matrix subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix3 which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (zMatrix3 &m1);

		/*! \brief This operator is used for scalar subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (float s1);

		/*! \brief This operator is used for scalar multiplication and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is multiplied to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator*= (float s1);

		/*! \brief This operator is used for scalar division and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which divides the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator/= (float s1);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method is used for matrix multiplication with a column matrix.
		*
		*	\param		[in]	m1			- column matrix which is multiplied with the current matrix.
		*	\param		[out]	out			- resultant column matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void multiply(zMatrix3Col &m1, zMatrix3Col &out);

		/*! \brief This method returns the transpose of the input matrix.
		*
		*	\return				zMatrix3	- resultant transpose matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 transpose();

		/*! \brief This method returns the minor/sub matrix of the input matrix removing the row and column values given by the input colindex and rowIndex.
		*
		*	\param		[in]	colIndex	- column index to be removed.
		*	\return				zMatrix2	- resultant leftover matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix2 minor(int colIndex, int rowIndex = 0);

		/*! \brief This method returns the cofactor matrix of the input matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\return				zMatrix3	- resultant cofactor zMatrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 cofactor();

		/*! \brief This method returns the adjoint matrix of the input matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\return				zMatrix3	- resultant adjoint zMatrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 adjoint();

		/*! \brief This method returns the inverse matrix of the input matrix, if it exists.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\param		[out]	zMatrix3	- resultant inverse zMatrix.
		*	\return				bool		- true if inverse matrix exists.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool inverse(zMatrix3 &outMatrix);

		/*! \brief This method returns the determinant of the input matrix.
		*
		*	\details Based on https://www.geeksforgeeks.org/determinant-of-a-matrix/
		*	\return				float		- determinant value.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float det();

	};

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zMatrix4
	*	\brief A 4x4 matrix math class.
	*	
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/	
	class ZSPACE_CORE zMatrix4
	{
	
	protected:
		/*!	\brief list of values in the matrix			*/
		float mat[16];

	public:

		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default Constructor .
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4();


		/*! \brief Overloaded Constructor for a square matrix.
		*
		*	\param		[in]	_mat		- input values of the matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4(const float _mat[16]);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE ~zMatrix4();
	

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index in the matrix value container given the row and column indicies.
		*
		*	\param		[in]	int		- row index.
		*	\param		[in]	int		- column index.
		*	\return				int		- index in value container.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE int getIndex(int rowIndex, int colIndex);

		/*! \brief This method gets the row and column index in the matrix given the input container index.
		*
		*	\param		[in]	int		- index.
		*	\param		[out]	int		- row index.
		*	\param		[out]	int		- column index.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getIndices(int index, int &rowIndex, int &colIndex);

		/*! \brief This method gets the row values as container of values at the input row index.
		*
		*	\param		[in]	index		- input row index.
		*	\return				zMatrix4Row	- container of row values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getRow(int index, zMatrix4Row &row);

		/*! \brief This method gets the column values at the input column index.
		*
		*	\param		[in]	index		- input column index.
		*	\return				zMatrix4Col	- container of column values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getCol(int index, zMatrix4Col &col);

		/*! \brief This method gets the diagonal values.
		*
		*	\return				zMatrix4Diag	- container of diagonal values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void getDiagonal(zMatrix4Diag &diag);

		/*! \brief This method gets the mats values container.
		*
		*	\param		[out]	_mat		- size 4 container of matrix values.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float* getMatrixValues();

		/*! \brief This method gets pointer to the internal matrix values container.
		*
		*	\param		[out]	float*		- pointer to internal matrix value container.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE float* getRawMatrixValues();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the row values at the input row index with the input value.
		*
		*	\param		[in]	index		- input row index.
		*	\param		[in]	val			- value of the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, float val);

		/*! \brief This method sets the row values at the input row index with the input vector of values.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	vals		- container of values for the row.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setRow(int index, const zMatrix4Row row);

		/*! \brief This method sets the row values at the input row index with the container vector of values.
		*
		*	\param		[in]	index		- input column index.
		*	\param		[in]	row			- value for the column.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, float val);

		/*! \brief This method sets the col values at the input col index with the input container of values.
		*
		*	\param		[in]	index		- input col index.
		*	\param		[in]	col			- container of values for the column.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setCol(int index, const zMatrix4Col col);

		/*! \brief This method sets the diagonal values of a square matrix with the input value.
		*
		*	\param		[in]	val			- value of the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(float val);

		/*! \brief This method sets the diagonal values of the matrix with the input container of values.
		*
		*	\param		[in]	diag			- vector of values for the diagonal.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setDiagonal(const zMatrix4Diag diag);

		/*! \brief This method sets values of the matrix to zero.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setZero();

		/*! \brief This method sets values of the matrix to one.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setOne();

		/*! \brief This method sets the matrix to identity if it is a square matrix.
		*
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void setIdentity();

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator returns the reference  to the coefficient at at the given row and column index.
		*
		*	\param		[in]	row		- input row index.
		*	\param		[in]	col		- input col index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int row, int col);

		/*! \brief This operator returns the reference  to the coefficient at at the given index.
		*
		*	\param		[in]	id		- input index.
		*	\return				float	- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float& operator()(int index);

		/*! \brief This operator is used for matrix addition.
		*
		*	\param		[in]	m1		 - zMatrix4 which is added to the current matrix.
		*	\return				zMatrix4 - resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator+ (zMatrix4 &m1);

		/*! \brief This operator is used for scalar addition to a matrix.
		*
		*	\param		[in]	s1			- scalar value which is added to the current matrix.
		*	\return				zMatrix4	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator+ (float s1);

		/*! \brief This operator is used for matrix subtraction.
		*
		*	\param		[in]	m1		    - zMatrix4 which is subtracted from the current matrix.
		*	\return				zMatrix4	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator- (zMatrix4 &m1);

		/*! \brief This operator is used for scalar subtraction from a matrix.
		*
		*	\param		[in]	s1			- scalar value which is subtracted from the current matrix.
		*	\return				zMatrix4	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator- (float s1);

		/*! \brief This operator is used for matrix multiplication.
		*
		*	\param		[in]	m1			- zMatrix2 which is multiplied with the current matrix.
		*	\return				zMatrix4	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator* (zMatrix4 &m1);

		/*! \brief This operator is used for scalar multiplication with a matrix.
		*
		*	\param		[in]	s1			- scalar value which is multiplied with the current matrix.
		*	\return				zMatrix2	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 operator* (float s1);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This operator is used for matrix addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix4 which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (zMatrix4 &m1);

		/*! \brief This operator is used for scalar addition and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator+= (float s1);

		/*! \brief This operator is used for matrix subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	m1		- zMatrix4 which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (zMatrix4 &m1);

		/*! \brief This operator is used for scalar subtraction and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is subtracted from the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator-= (float s1);

		/*! \brief This operator is used for scalar multiplication and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which is multiplied to the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator*= (float s1);

		/*! \brief This operator is used for scalar division and assigment of the result to the current matrix.
		*
		*	\param		[in]	s1		- scalar value which divides the current matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void operator/= (float s1);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method is used for matrix multiplication with a column matrix.
		*
		*	\param		[in]	m1			- column matrix which is multiplied with the current matrix.
		*	\param		[out]	out			- resultant column matrix after the multiplication.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE void multiply(zMatrix4Col &m1, zMatrix4Col &out);

		/*! \brief This method returns the transpose of the input matrix.
		*
		*	\return				zMatrix4	- resultant transpose matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 transpose();

		/*! \brief This method returns the minor/sub matrix of the input matrix removing the row and column values given by the input colindex and rowIndex.
		*
		*	\param		[in]	colIndex	- column index to be removed.
		*	\return				zMatrix4	- resultant leftover matrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix3 minor(int colIndex, int rowIndex = 0);

		/*! \brief This method returns the cofactor matrix of the input matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\return				zMatrix3	- resultant cofactor zMatrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 cofactor();

		/*! \brief This method returns the adjoint matrix of the input matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\return				zMatrix		- resultant adjoint zMatrix.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 adjoint();

		/*! \brief This method returns the inverse matrix of the input matrix, if it exists.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\param		[out]	zMatrix3	- resultant inverse zMatrix.
		*	\return				bool		- true if inverse matrix exists.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool inverse(zMatrix4 &outMatrix);

		/*! \brief This method returns the determinant of the input matrix if it is a square matrix.
		*
		*	\details Based on https://www.geeksforgeeks.org/determinant-of-a-matrix/
		*	\return				float		- determinant value.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE float det();

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/


	/** \addtogroup Matrix
	*	\brief  The matrix typedef of the library.
	*  @{
	*/

	/*! \typedef zTransform
	*	\brief A 4x4 matrix.
	*
	*	\since version 0.0.2
	*/	
#ifndef __CUDACC__
	typedef Eigen::Matrix4f zTransform;
#else
	typedef zSpace::zMatrix4 zTransform;
#endif

	/*! \typedef zPlane
	*	\brief A 4x4 matrix.
	*
	*	\since version 0.0.2
	*/
#ifndef __CUDACC__
	typedef Eigen::Matrix4f zPlane;
#else
	typedef zSpace::zMatrix4 zPlane;
#endif
	
#ifndef __CUDACC__
	/*! \typedef zSparseMatrix
	*	\brief A  Eigen library column-major sparse matrix type of float.
	*
	*	\since version 0.0.2
	*/
	typedef Eigen::SparseMatrix<double> zSparseMatrix; 

	/*! \typedef zTriplet
	*	\brief A  Eigen library triplet of float.
	*
	*	\since version 0.0.2
	*/
	typedef Eigen::Triplet<double> zTriplet;

	/*! \typedef zDiagonalMatrix
	*	\brief A  Eigen library diagonal matrix of float.
	*
	*	\since version 0.0.2
	*/
	typedef DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> zDiagonalMatrix;
	
#endif
	
	/** @}*/
	/** @}*/ 
	/** @}*/
	/** @}*/
	// end of group zCore


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
#else
#include<source/zCore/base/zMatrix.cpp>
#endif

#endif
