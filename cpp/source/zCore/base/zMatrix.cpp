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


#include<headers/zCore/base/zMatrix.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	template <typename T>
	ZSPACE_INLINE zMatrix<T>::zMatrix()
	{
		rows = 4;
		cols = 4;

		mat.clear();

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				mat.push_back(0);
				if (i == j) mat[mat.size() - 1] = 1;

			}
		}
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T>::zMatrix(int _rows, int _cols, vector<T> &_mat)
	{
		if (_rows*_cols != _mat.size()) throw std::invalid_argument("input _mat size not equal to (_rows * _cols).");

		rows = _rows;
		cols = _cols;

		mat.clear();
		mat = _mat;
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T>::zMatrix(int _dims)
	{
		rows = _dims;
		cols = _dims;

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				mat.push_back(0);
				if (i == j)mat[mat.size() - 1] = 1;

			}
		}
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T>::zMatrix(int _rows, int _cols)
	{
		rows = _rows;
		cols = _cols;

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				mat.push_back(0);
			}
		}
	}

	//---- DESTRUCTOR

	template <typename T>
	ZSPACE_INLINE zMatrix<T>::~zMatrix(){}

	//---- GET METHODS

	template <typename T>
	ZSPACE_INLINE int zMatrix<T>::getNumRows()
	{
		return this->rows;
	}

	template <typename T>
	ZSPACE_INLINE int zMatrix<T>::getNumCols()
	{
		return this->cols;
	}

	template <typename T>
	ZSPACE_INLINE int zMatrix<T>::getIndex(int rowIndex, int colIndex)
	{
		if (rowIndex < 0 || rowIndex > this->getNumRows()) throw std::invalid_argument("input rowIndex out of bounds.");
		if (colIndex < 0 || colIndex > this->getNumCols()) throw std::invalid_argument("input colIndex out of bounds.");

		return rowIndex * this->getNumCols() + colIndex;
	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::getIndices(int index, int &rowIndex, int &colIndex)
	{
		if (index < 0 || index > this->getNumRows() * this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

		rowIndex =

			rowIndex = floor(index / this->getNumCols());
		colIndex = index % this->getNumCols();
	}

	template <typename T>
	ZSPACE_INLINE vector<T> zMatrix<T>::getRow(int index)
	{
		if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

		vector<T> out;

		for (int i = 0; i < this->getNumCols(); i++)
		{
			out.push_back(this->operator()(index, i));
		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::getRowMatrix(int index)
	{
		if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

		zMatrix<T> out(1, this->getNumCols());

		for (int i = 0; i < this->getNumCols(); i++)
		{
			out(0, i) = (this->operator()(index, i));
		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE vector<T> zMatrix<T>::getCol(int index)
	{
		if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

		vector<T> out;

		for (int i = 0; i < this->getNumRows(); i++)
		{
			out.push_back(this->operator()(i, index));
		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::getColumnMatrix(int index)
	{
		if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

		zMatrix<T> out(this->getNumRows(), 1);

		for (int i = 0; i < this->getNumRows(); i++)
		{
			out(i, 0) = (this->operator()(i, index));
		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE vector<T> zMatrix<T>::getDiagonal()
	{
		if (this->getNumRows() != this->getNumCols()) throw std::invalid_argument("input is not a square matrix.");

		vector<T> out;

		for (int i = 0; i < this->getNumRows(); i++)
		{
			for (int j = 0; j < this->getNumCols(); j++)
			{
				if (i == j) out.push_back(this->operator()(i, j));
			}

		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::getMatrixValues(vector<T> & _mat)
	{
		_mat = mat;
	}
	
	template <typename T>
	ZSPACE_INLINE T* zMatrix<T>::getRawMatrixValues()
	{
		return &mat[0];
	}

	//---- SET METHODS

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setMatrixValues(vector<T> & _mat)
	{
		if (this->getNumCols() * this->getNumRows() != _mat.size()) throw std::invalid_argument("input _mat size not equal to (rows * cols).");

		mat.clear();
		mat = _mat;
	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setRow(int index, T val)
	{
		if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

		for (int i = 0; i < this->getNumCols(); i++)
		{
			this->operator()(index, i) = val;


		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setRow(int index, vector<T>& vals)
	{
		if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");
		if (vals.size() != this->getNumCols()) throw std::invalid_argument("input values size dont match with number of columns.");

		for (int i = 0; i < this->getNumCols(); i++)
		{
			this->operator()(index, i) = vals[i];
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setRow(int index, zMatrix<T> &rowMatrix)
	{
		if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");
		if (rowMatrix.getNumCols() != this->getNumCols()) throw std::invalid_argument("input values size dont match with number of columns.");
		if (rowMatrix.getNumRows() != 1) throw std::invalid_argument("input in not row matrix.");

		for (int i = 0; i < this->getNumCols(); i++)
		{
			this->operator()(index, i) = rowMatrix(0, i);
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setCol(int index, T val)
	{
		if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

		for (int i = 0; i < this->getNumRows(); i++)
		{
			this->operator()(i, index) = val;
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setCol(int index, vector<T>& vals)
	{
		if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
		if (vals.size() != this->getNumRows()) throw std::invalid_argument("input values size dont match with number of rows.");

		for (int i = 0; i < this->getNumRows(); i++)
		{
			this->operator()(i, index) = vals[i];
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setCol(int index, zMatrix<T> &colMatrix)
	{
		if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
		if (colMatrix.getNumRows() != this->getNumRows()) throw std::invalid_argument("input values size dont match with number of rows.");
		if (colMatrix.getNumCols() != 1) throw std::invalid_argument("input in not column matrix.");

		for (int i = 0; i < this->getNumRows(); i++)
		{
			this->operator()(i, index) = colMatrix(i, 0);
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setDiagonal(T val)
	{
		if (this->getNumRows() != this->getNumCols()) throw std::invalid_argument("input not a square matrix.");

		for (int i = 0; i < this->getNumRows(); i++)
		{
			for (int j = 0; j < this->getNumCols(); j++)
			{
				if (i == j)this->operator()(i, j) = val;
			}

		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setDiagonal(vector<T> vals)
	{
		if (this->getNumRows() != this->getNumCols()) throw std::invalid_argument("input not a square matrix.");
		if (vals.size() != this->getNumRows()) throw std::invalid_argument("input values size dont match with number of rows/columns.");

		for (int i = 0; i < this->getNumRows(); i++)
		{
			for (int j = 0; j < this->getNumCols(); j++)
			{
				if (i == j) this->operator()(i, j) = vals[i];
			}

		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setZero()
	{

		for (int i = 0; i < getNumRows(); i++)
		{
			for (int j = 0; j < getNumCols(); j++)
			{
				this->operator()(i, j) = 0;
			}
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setOne()
	{

		for (int i = 0; i < getNumRows(); i++)
		{
			for (int j = 0; j < getNumCols(); j++)
			{
				this->operator()(i, j) = 1;
			}
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::setIdentity()
	{
		_ASSERT_EXPR(getNumRows() == getNumCols(), "input not a square matrix.");

		this->setZero();
		this->setDiagonal(1);
	}

	//---- OPERATORS

	template <typename T>
	ZSPACE_INLINE T& zMatrix<T>::operator()(int row, int col)
	{
		if (row < 0 && row > this->getNumRows()) throw std::invalid_argument("input row out of bounds.");
		if (col < 0 && col > this->getNumRows()) throw std::invalid_argument("input col out of bounds.");

		return mat[(row * this->getNumCols()) + col];
	}

	template <typename T>
	ZSPACE_INLINE T& zMatrix<T>::operator()(int index)
	{
		if (index < 0 && index >= this->getNumRows() * this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

		return mat[index];
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator+ (zMatrix<T> &m1)
	{
		if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
		if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");

		vector<T> out;

		for (int i = 0; i < this->getNumRows() *this->getNumCols(); i++)
		{
			T val = this->operator()(i) + m1(i);
			out.push_back(val);
		}

		return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator+ (T s1)
	{

		vector<T> out;

		for (int i = 0; i < this->getNumRows() *this->getNumCols(); i++)
		{
			T val = this->operator()(i) + s1;
			out.push_back(val);
		}

		return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator- (zMatrix<T> &m1)
	{
		if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
		if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");


		vector<T> out;

		for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
		{
			T val = this->operator()(i) - m1(i);
			out.push_back(val);
		}

		return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator- (T s1)
	{
		vector<T> out;

		for (int i = 0; i < this->getNumRows() *this->getNumCols(); i++)
		{
			T val = this->operator()(i) - s1;
			out.push_back(val);
		}

		return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator* (zMatrix<T> &m1)
	{
		if (this->getNumCols() != m1.getNumRows()) throw std::invalid_argument("number of columns in current matrix not equal to number of rows in m1.");

		vector<T> out;
		out.assign(this->getNumRows() * m1.getNumCols(), T());

		int count = 0;

		for (int i = 0; i < this->getNumRows(); i++)
		{
			vector<T> rVals = this->getRow(i);

			for (int j = 0; j < m1.getNumCols(); j++)
			{
				vector<T> cVals = m1.getCol(j);

				T val = 0;

				for (int k = 0; k < rVals.size(); k++)
				{
					val += rVals[k] * cVals[k];
				}

				out[count] = (val);
				count++;
			}
		}

		return zMatrix<T>(this->getNumRows(), m1.getNumCols(), out);
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::operator* (T s1)
	{
		vector<T> out;

		for (int i = 0; i < getNumRows() *getNumCols(); i++)
		{
			T val = this->operator()(i) * s1;
			out.push_back(val);
		}

		return zMatrix<T>(getNumRows(), getNumCols(), out);
	}

	//---- OVERLOADED OPERATORS

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::operator+= (zMatrix<T> &m1)
	{
		if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
		if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");

		for (int i = 0; i < getNumRows() *getNumCols(); i++)
		{

			this->operator()(i) += m1(i);
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::operator+= (T s1)
	{
		for (int i = 0; i < getNumRows() *getNumCols(); i++)
		{
			this->operator()(i) += s1;
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::operator-= (zMatrix<T> &m1)
	{
		if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
		if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");

		for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
		{

			this->operator()(i) -= m1(i);
		}

	}

	template <typename T>
	ZSPACE_INLINE
	void zMatrix<T>::operator-= (T s1)
	{

		for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
		{

			this->operator()(i) -= s1;
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::operator*= (T s1)
	{

		for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
		{

			this->operator()(i) *= s1;
		}

	}

	template <typename T>
	ZSPACE_INLINE void zMatrix<T>::operator/= (T s1)
	{

		for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
		{

			mat[i] /= s1;
		}

	}

	//---- METHODS

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::transpose()
	{
		int _numRows = this->getNumCols();
		int _numCols = this->getNumRows();

		zMatrix<T> out = zMatrix<T>(_numRows, _numCols);

		for (int i = 0; i < this->getNumCols(); i++)
		{
			vector<T> colVals = this->getCol(i);
			out.setRow(i, colVals);
		}

		return out;
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::minorMatrix(int colIndex, int rowIndex )
	{
		if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");
		if (colIndex < 0 && colIndex > getNumCols()) throw std::invalid_argument("input colIndex out of bounds.");
		if (rowIndex < 0 && rowIndex > getNumRows()) throw std::invalid_argument("input rowIndex out of bounds.");

		vector<T> vals;

		for (int i = 0; i < this->getNumRows(); i++)
		{
			if (i != rowIndex)
			{
				for (int j = 0; j < this->getNumCols(); j++)
				{
					if (j != colIndex) vals.push_back(this->operator()(i, j));

				}
			}

		}

		return zMatrix<T>(this->getNumRows() - 1, this->getNumCols() - 1, vals);

	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::cofactorMatrix()
	{
		if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");

		vector<T> vals;

		if (this->getNumCols() == 1)
		{
			vals.push_back(1);

			return	zMatrix<T>(this->getNumRows(), this->getNumCols(), vals);

		}

		int sign = 1;

		for (int i = 0; i < this->getNumRows(); i++)
		{
			for (int j = 0; j < this->getNumCols(); j++)
			{

				zMatrix<T> minorM = this->minorMatrix(j, i);

				// sign of adj[j][i] positive if sum of row 
				// and column indexes is even. 
				sign = ((i + j) % 2 == 0) ? 1 : -1;

				vals.push_back(sign * minorM.det());

			}
		}

		return zMatrix<T>(this->getNumRows(), this->getNumCols(), vals);;
	}

	template <typename T>
	ZSPACE_INLINE zMatrix<T> zMatrix<T>::adjointMatrix()
	{
		if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");

		zMatrix<T> cofactorMatrix = this->cofactorMatrix();

		return cofactorMatrix.transpose();
	}

	template <typename T>
	ZSPACE_INLINE bool zMatrix<T>::inverseMatrix(zMatrix<T>& outMat)
	{

		if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");

		double det = this->det();

		if (det == 0)
		{
			printf("\n Singular matrix, can't find its inverse");
			outMat = zMatrix<T>(this->getNumRows(), this->getNumCols());
			return false;
		}
		else
		{
			zMatrix<T> adjoint = adjointMatrix();
			vector<T> vals;

			for (int i = 0; i < this->getNumRows(); i++)
			{
				for (int j = 0; j < this->getNumRows(); j++)
					vals.push_back(adjoint(i, j) / det);
			}

			outMat = zMatrix<T>(this->getNumRows(), this->getNumCols(), vals);
			return true;
		}

	}

	template <typename T>
	ZSPACE_INLINE T zMatrix<T>::det()
	{
		if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");

		T D = 0;

		if (this->getNumCols() == 1) return this->operator()(0, 0);


		int sign = 1;  // To store sign multiplier 

					   // Iterate for each element of first row 
		for (int i = 0; i < this->getNumCols(); i++)
		{
			// Getting Cofactor of mat[0][f] 

			zMatrix<T> minorM = this->minorMatrix(i);

			D += sign * this->operator()(0, i) * minorM.det();

			// terms are to be added with alternate sign 
			sign = -sign;
		}

		return D;
	}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// explicit instantiation
	template class zMatrix<int>;

	template class zMatrix<float>;

	template class zMatrix<double>;
	
#endif
	
}

