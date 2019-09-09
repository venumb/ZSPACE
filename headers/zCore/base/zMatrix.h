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

#include<headers/zCore/base/zInline.h>

#include <stdexcept>
#include <vector>
using namespace std;

#include<depends/Eigen/Core>
#include<depends/Eigen/Dense>
#include<depends/Eigen/Sparse>
#include<depends/Eigen/Eigen>
#include<depends/Eigen/Sparse>
using namespace Eigen;

#include <depends/Armadillo/armadillo>
using namespace arma;



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

	/*! \class zMatrix
	*	\brief A template matrix math class.
	*	
	*	\tparam				T			- Type to work with standard c++ numerical datatypes. 
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	template <typename T>
	class ZSPACE_CORE zMatrix
	{
	
	protected:
		/*!	\brief list of values in the matrix			*/
		vector<T> mat;	

		/*!	\brief number of rows in the matrix			*/
		int rows;	

		/*!	\brief number of columns in the matrix		*/
		int cols;		

	public:

		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default Constructor sets to a 4x4 matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\since version 0.0.1
		*/
		zMatrix();		

		/*! \brief Overloaded Constructor.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_rows		- number of rows.
		*	\param		[in]	_cols		- number of columns.
		*	\param		[in]	_mat		- values of the matrix.
		*	\since version 0.0.1
		*/
		zMatrix(int _rows, int _cols, vector<T> &_mat);		

		/*! \brief Overloaded Constructor for a square matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_dims		- number of rows and cols.
		*	\since version 0.0.1
		*/
		zMatrix(int _dims);		

		/*! \brief Overloaded Constructor for a matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_rows		- number of rows.
		*	\param		[in]	_cols		- number of cols.
		*	\since version 0.0.1
		*/
		zMatrix(int _rows, int _cols);	

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zMatrix();
	

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the number of rows in the matrix.
		*	\return				int		- number of rows.
		*	\since version 0.0.1
		*/
		int getNumRows();		

		/*! \brief This method gets the number of columns in the matrix.
		*	\return				int		- number of columns.
		*	\since version 0.0.1
		*/
		int getNumCols();		

		/*! \brief This method gets the index in the matrix value container given the row and column indicies.
		*	\param		[in]	int		- row index.
		*	\param		[in]	int		- column index.
		*	\return				int		- index in value container.
		*	\since version 0.0.1
		*/
		int getIndex(int rowIndex, int colIndex);	

		/*! \brief This method gets the row and column index in the matrix given the input container index.
		*	\param		[in]	int		- index.
		*	\param		[out]	int		- row index.
		*	\param		[out]	int		- column index.
		*	\since version 0.0.1
		*/
		void getIndices(int index, int &rowIndex, int &colIndex);

		/*! \brief This method gets the row values as container of values at the input row index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\return				vector<T>	- vector of row values.
		*	\since version 0.0.1
		*/
		vector<T> getRow(int index);

		/*! \brief This method gets the row matrix at the input row index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\return				zMatrix<T>	- row matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> getRowMatrix(int index);

		/*! \brief This method gets the column values at the input column index.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input column index.
		*	\return				vector<T>	- vector of column values.
		*	\since version 0.0.1
		*/
		vector<T> getCol(int index);

		/*! \brief This method gets the column matrix at the input column index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input column index.
		*	\return				zMatrix<T>	- column matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> getColumnMatrix(int index);

		/*! \brief This method gets the diagonal values if it is a square matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				vector<T>	- vector of diagonal values.
		*	\since version 0.0.1
		*/
		vector<T> getDiagonal();

		/*! \brief This method gets the mats values container.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[out]	_mat		- vector of matrix values.
		*	\since version 0.0.1
		*/
		void getMatrixValues(vector<T> & _mat);

		/*! \brief This method gets pointer to the internal matrix values container.
		*
		*	\tparam				T					- Type to work with standard c++ numerical datatypes.
		*	\return				T*					- pointer to internal matrix value container.
		*	\since version 0.0.2
		*/
		T* getRawMatrixValues();


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the matrix values with the input container of values.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_mat		- vector of matrix values.
		*	\since version 0.0.1
		*/
		void setMatrixValues(vector<T> & _mat);

		/*! \brief This method sets the row values at the input row index with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	val			- value of the row.
		*	\since version 0.0.1
		*/
		void setRow(int index, T val);

		/*! \brief This method sets the row values at the input row index with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	vals		- vector of values for the row.
		*	\since version 0.0.1
		*/
		void setRow(int index, vector<T>& vals);

		/*! \brief This method sets the row values at the input row index with the input row Matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	rowMatrix	- row matrix.
		*	\since version 0.0.1
		*/
		void setRow(int index, zMatrix<T> &rowMatrix);

		/*! \brief This method sets the col values at the input col index with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input col index.
		*	\param		[in]	val			- value of the col.
		*	\since version 0.0.1
		*/
		void setCol(int index, T val);

		/*! \brief This method sets the col values at the input col index with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input col index.
		*	\param		[in]	vals		- vector of values for the col.
		*	\since version 0.0.1
		*/
		void setCol(int index, vector<T>& vals);

		/*! \brief This method sets the row values at the input column index with the input column Matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	colMatrix	- column matrix.
		*	\since version 0.0.1
		*/
		void setCol(int index, zMatrix<T> &colMatrix);

		/*! \brief This method sets the diagonal values of a square matrix with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val			- value of the diagonal.
		*	\since version 0.0.1
		*/
		void setDiagonal(T val);

		/*! \brief This method sets the diagonal values of a square matrix with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- vector of values for the diagonal.
		*	\since version 0.0.1
		*/
		void setDiagonal(vector<T> vals);

		/*! \brief This method sets values of the matrix to zero.
		*	\since version 0.0.1
		*/
		void setZero();

		/*! \brief This method sets values of the matrix to one.
		*	\since version 0.0.1
		*/
		void setOne();

		/*! \brief This method sets the matrix to identity if it is a square matrix.
		*	\since version 0.0.1
		*/
		void setIdentity();

		//--------------------------
		//---- OPERATORS
		//--------------------------

		/*! \brief This operator returns the reference  to the coefficient at at the given row and column index.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	row		- input row index.
		*	\param		[in]	col		- input col index.
		*	\return				T		- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		T& operator()(int row, int col);

		/*! \brief This operator returns the reference  to the coefficient at at the given index.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	id		- input index.
		*	\return				T		- reference  to the coefficient.
		*	\since version 0.0.1
		*/
		T& operator()(int index);

		/*! \brief This operator is used for matrix addition.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\return				zMatrix	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator+ (zMatrix<T> &m1);

		/*! \brief This operator is used for scalar addition to a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\return				zMatrix	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator+ (T s1);

		/*! \brief This operator is used for matrix subtraction.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is subtracted from the current matrix.
		*	\return				zMatrix	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator- (zMatrix<T> &m1);

		/*! \brief This operator is used for scalar subtraction from a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is subtracted from the current matrix.
		*	\return				zMatrix	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator- (T s1);


		/*! \brief This operator is used for matrix multiplication.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is multiplied with the current matrix.
		*	\return				zMatrix	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator* (zMatrix<T> &m1);

		/*! \brief This operator is used for scalar multiplication with a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is multiplied with the current matrix.
		*	\return				zMatrix	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/
		zMatrix<T> operator* (T s1);

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------


		/*! \brief This operator is used for matrix addition and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\since version 0.0.1
		*/
		void operator+= (zMatrix<T> &m1);


		/*! \brief This operator is used for scalar addition and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/
		void operator+= (T s1);


		/*! \brief This operator is used for matrix subtraction and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\since version 0.0.1
		*/
		void operator-= (zMatrix<T> &m1);

		/*! \brief This operator is used for scalar subtraction and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/
		void operator-= (T s1);

		/*! \brief This operator is used for scalar multiplication and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is multiplied to the current matrix.
		*	\since version 0.0.1
		*/
		void operator*= (T s1);


		/*! \brief This operator is used for scalar division and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which divides the current matrix.
		*	\since version 0.0.1
		*/
		void operator/= (T s1);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method returns the transpose of the input matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\return				zMatrix	- resultant transpose matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> transpose();


		/*! \brief This method returns the minor/sub matrix of the input square matrix removing the row and column values given by the input colindex and rowIndex.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	colIndex- column index to be removed.
		*	\return				zMatrix	- resultant leftover matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> minorMatrix(int colIndex, int rowIndex = 0);
			   
		/*! \brief This method returns the cofactor matrix of the input square matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				zMatrix		- resultant cofactor zMatrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> cofactorMatrix();


		/*! \brief This method returns the adjoint matrix of the input square matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				zMatrix		- resultant adjoint zMatrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> adjointMatrix();

		/*! \brief This method returns the inverse matrix of the input square matrix, if it exists.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[out]	zMatrix		- resultant inverse zMatrix.
		*	\return				bool		- true if inverse matrix exists.
		*	\since version 0.0.1
		*/
		bool inverseMatrix(zMatrix<T>& outMat);

		/*! \brief This method returns the determinant of the input matrix if it is a square matrix.
		*
		*	\details Based on https://www.geeksforgeeks.org/determinant-of-a-matrix/
		*	\tparam				T		- Type to work with int, double, float
		*	\return				T		- determinant value.
		*	\since version 0.0.1
		*/
		T det();		

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zMatrixTypedef
	*	\brief  The matrix typedef of the library.
	*  @{
	*/

	/*! \typedef zMatrixi
	*	\brief A matrix  of integers.
	*
	*	\since version 0.0.1
	*/
	typedef zMatrix<int> zMatrixi;
	
	
	/*! \typedef zMatrixd
	*	\brief A matrix  of doubles.
	*
	*	\since version 0.0.1
	*/	
	typedef zMatrix<double> zMatrixd;
	
	/*! \typedef zMatrixf
	*	\brief A matrix  of floats.
	*
	*	\since version 0.0.1
	*/	
	typedef zMatrix<float> zMatrixf;

	/*! \typedef zTransform
	*	\brief A 4x4 matrix.
	*
	*	\since version 0.0.2
	*/	
	typedef Eigen::Matrix4d zTransform;

	/*! \typedef zSparseMatrix
	*	\brief A  Eigen library column-major sparse matrix type of double.
	*
	*	\since version 0.0.2
	*/
	typedef Eigen::SparseMatrix<double> zSparseMatrix; 

	/*! \typedef zTriplet
	*	\brief A  Eigen library triplet of double.
	*
	*	\since version 0.0.2
	*/
	typedef Eigen::Triplet<double> zTriplet;

	/*! \typedef zDiagonalMatrix
	*	\brief A  Eigen library diagonal matrix of double.
	*
	*	\since version 0.0.2
	*/
	typedef DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> zDiagonalMatrix;
	
	//typedef zMatrix<double> zTransform;
	
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
