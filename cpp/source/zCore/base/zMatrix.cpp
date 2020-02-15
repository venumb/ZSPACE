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

//---- zMATRIX 2X2

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zMatrix2::zMatrix2()
	{
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 2; j++)
			{

				if (i == j) mat[i * 2 + j] = 1;
				else mat[i * 2 + j] = (0);

			}
		}
	}

	ZSPACE_INLINE zMatrix2::zMatrix2(const float _mat[4])
	{
		for (int i = 0; i < 4; i++) mat[i] = _mat[i];
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zMatrix2::~zMatrix2() {}

	//---- GET METHODS

	ZSPACE_INLINE int zMatrix2::getIndex(int rowIndex, int colIndex)
	{
		return rowIndex * 2 + colIndex;
	}

	ZSPACE_INLINE void zMatrix2::getIndices(int index, int &rowIndex, int &colIndex)
	{

#ifndef __CUDACC__
		rowIndex = floor(index / 2);
#else
		rowIndex = floorf(index / 2);
#endif		
		colIndex = index % 2;
	}

	ZSPACE_INLINE void zMatrix2::getRow(int index, zMatrix2Row &row)
	{
		for (int i = 0; i < 2; i++)	row[i] = (this->operator()(index, i));
	}

	ZSPACE_INLINE void zMatrix2::getCol(int index, zMatrix2Col &col)
	{
		for (int i = 0; i < 2; i++)	col[i] = (this->operator()(i, index));
	}

	ZSPACE_INLINE void zMatrix2::getDiagonal(zMatrix2Diag &diag)
	{
		for (int i = 0; i < 3; i++)	diag[i] = (this->operator()(i, i));
	}

	ZSPACE_INLINE float* zMatrix2::getMatrixValues()
	{		
		return  mat;
	}

	ZSPACE_INLINE float* zMatrix2::getRawMatrixValues()
	{
		return &mat[0];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zMatrix2::setRow(int index, float val)
	{
		for (int i = 0; i < 2; i++) this->operator()(index, i) = val;
	}

	ZSPACE_INLINE void zMatrix2::setRow(int index, zMatrix2Row &row)
	{
		for (int i = 0; i < 2; i++) this->operator()(index, i) = i;
	}

	ZSPACE_INLINE void zMatrix2::setCol(int index, float val)
	{
		for (int i = 0; i < 2; i++) this->operator()(i, index) = val;
	}

	ZSPACE_INLINE void zMatrix2::setCol(int index, const zMatrix2Col &col)
	{
		for (int i = 0; i <= 2; i++) this->operator()(i, index) = col[i];
	}

	ZSPACE_INLINE void zMatrix2::setDiagonal(float val)
	{
		for (int i = 0; i < 2; i++)	this->operator()(i, i) = val;
	}

	ZSPACE_INLINE void zMatrix2::setDiagonal(const zMatrix2Diag &diag)
	{
		for (int i = 0; i < 2; i++)	this->operator()(i, i) = diag[i];
	}

	ZSPACE_INLINE void zMatrix2::setZero()
	{
		for (int i = 0; i < 4; i++) mat[i] = 0;
	}

	ZSPACE_INLINE void zMatrix2::setOne()
	{
		for (int i = 0; i < 4; i++) mat[i] = 1;
	}

	ZSPACE_INLINE void zMatrix2::setIdentity()
	{
		this->setZero();
		this->setDiagonal(1);
	}

	//---- OPERATORS

	ZSPACE_INLINE float& zMatrix2::operator()(int row, int col)
	{
		return mat[row * 2 + col];
	}

	ZSPACE_INLINE float& zMatrix2::operator()(int index)
	{
		return mat[index];
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator+ (zMatrix2 &m1)
	{
		zMatrix2 out;
		for (int i = 0; i < 4; i++)out(i) = this->operator()(i) + m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator+ (float s1)
	{
		zMatrix2 out;
		for (int i = 0; i < 4; i++)out(i) = this->operator()(i) + s1;
		return out;
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator- (zMatrix2 &m1)
	{
		zMatrix2 out;
		for (int i = 0; i < 4; i++)out(i) = this->operator()(i) - m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator- (float s1)
	{
		zMatrix2 out;
		for (int i = 0; i < 4; i++)out(i) = this->operator()(i) - s1;
		return out;
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator* (zMatrix2 &m1)
	{
		zMatrix2 out;
		int count = 0;

		for (int i = 0; i < 2; i++)
		{
			zMatrix2Row rVals;
			this->getRow(i, rVals);

			for (int j = 0; j < 2; j++)
			{
				zMatrix2Col cVals;
				m1.getCol(j,cVals);

				float val = 0;
				for (int k = 0; k < 2; k++)	val += rVals[k] * cVals[k];
				out(count) = val;
				count++;
			}
		}

		return out;
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::operator* (float s1)
	{
		zMatrix2 out;
		for (int i = 0; i < 4; i++)out(i) = this->operator()(i) * s1;
		return out;
	}

	//---- OVERLOADED OPERATORS

	ZSPACE_INLINE void zMatrix2::operator+= (zMatrix2 &m1)
	{
		for (int i = 0; i < 4; i++) this->operator()(i) += m1(i);
	}

	ZSPACE_INLINE void zMatrix2::operator+= (float s1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) += s1;
	}

	ZSPACE_INLINE void zMatrix2::operator-= (zMatrix2 &m1)
	{
		for (int i = 0; i < 4; i++) this->operator()(i) -= m1(i);
	}

	ZSPACE_INLINE void zMatrix2::operator-= (float s1)
	{
		for (int i = 0; i < 4; i++) this->operator()(i) -= s1;
	}

	ZSPACE_INLINE void zMatrix2::operator*= (float s1)
	{
		for (int i = 0; i < 4; i++) this->operator()(i) *= s1;
	}

	ZSPACE_INLINE void zMatrix2::operator/= (float s1)
	{
		for (int i = 0; i < 4; i++) this->operator()(i) /= s1;
	}

	//---- METHODS

	ZSPACE_INLINE void zMatrix2::multiply(zMatrix2Col &m1, zMatrix2Col &out)
	{
		int count = 0;

		for (int i = 0; i < 2; i++)
		{
			zMatrix2Row rVals;
			this->getRow(i, rVals);

			float val = 0;
			for (int k = 0; k < 2; k++)	val += rVals[k] * m1[k];
			out[count] = val;
			count++;
		}
	}

	ZSPACE_INLINE zMatrix2 zMatrix2::transpose()
	{
		zMatrix2 out;

		for (int i = 0; i < 3; i++)
		{
			zMatrix2Col colVals;
			this->getCol(i, colVals);
			out.setRow(i, colVals);
		}

		return out;
	}

	ZSPACE_INLINE float zMatrix2::det()
	{
		return (mat[0]*mat[3]) - (mat[1] * mat[2]) ;
	}

}

//---- zMATRIX 3X3

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zMatrix3::zMatrix3()
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{

				if (i == j) mat[i * 3 + j] = 1;
				else mat[i * 3 + j] = (0);

			}
		}
	}

	ZSPACE_INLINE zMatrix3::zMatrix3(const float _mat[9])
	{
		for (int i = 0; i < 9; i++) mat[i] = _mat[i];
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zMatrix3::~zMatrix3() {}

	//---- GET METHODS

	ZSPACE_INLINE int zMatrix3::getIndex(int rowIndex, int colIndex)
	{
		return rowIndex * 3 + colIndex;
	}

	ZSPACE_INLINE void zMatrix3::getIndices(int index, int &rowIndex, int &colIndex)
	{

#ifndef __CUDACC__
		rowIndex = floor(index / 3);
#else
		rowIndex = floorf(index / 3);
#endif
		colIndex = index % 3;
	}

	ZSPACE_INLINE void zMatrix3::getRow(int index, zMatrix3Row &row)
	{	
		for (int i = 0; i < 3; i++)	row[i] = (this->operator()(index, i));
	}

	ZSPACE_INLINE void zMatrix3::getCol(int index, zMatrix3Col &col)
	{	
		for (int i = 0; i < 3; i++)	col[i] = (this->operator()(i, index));
	}

	ZSPACE_INLINE void zMatrix3::getDiagonal(zMatrix3Diag &diag)
	{			
		for (int i = 0; i < 3; i++)	diag[i] = (this->operator()(i, i));
	}

	ZSPACE_INLINE float* zMatrix3::getMatrixValues()
	{
		return mat;
	}

	ZSPACE_INLINE float* zMatrix3::getRawMatrixValues( )
	{
		return &mat[0];
	}

	//---- SET METHODS

	ZSPACE_INLINE void zMatrix3::setRow(int index, float val)
	{
		for (int i = 0; i < 3; i++) this->operator()(index, i) = val;
	}

	ZSPACE_INLINE void zMatrix3::setRow(int index, const zMatrix3Row &row)
	{
		for (int i = 0; i < 3; i++) this->operator()(index, i) = row[i];
	}

	ZSPACE_INLINE void zMatrix3::setCol(int index, float val)
	{
		for (int i = 0; i < 3; i++) this->operator()(i, index) = val;
	}

	ZSPACE_INLINE void zMatrix3::setCol(int index, const zMatrix3Col &col)
	{
		for (int i = 0; i <= 3; i++) this->operator()(i, index) = col[i];
	}

	ZSPACE_INLINE void zMatrix3::setDiagonal(float val)
	{
		for (int i = 0; i < 3; i++)	this->operator()(i, i) = val;
	}

	ZSPACE_INLINE void zMatrix3::setDiagonal(const zMatrix3Diag &diag)
	{
		for (int i = 0; i < 3; i++)	this->operator()(i, i) = diag[i];
	}

	ZSPACE_INLINE void zMatrix3::setZero()
	{
		for (int i = 0; i < 9; i++) mat[i] = 0;
	}

	ZSPACE_INLINE void zMatrix3::setOne()
	{
		for (int i = 0; i < 9; i++) mat[i] = 1;
	}

	ZSPACE_INLINE void zMatrix3::setIdentity()
	{
		this->setZero();
		this->setDiagonal(1);
	}

	//---- OPERATORS

	ZSPACE_INLINE float& zMatrix3::operator()(int row, int col)
	{
		return mat[row * 3 + col];
	}

	ZSPACE_INLINE float& zMatrix3::operator()(int index)
	{
		return mat[index];
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator+ (zMatrix3 &m1)
	{
		zMatrix3 out;
		for (int i = 0; i < 9; i++)out(i) = this->operator()(i) + m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator+ (float s1)
	{
		zMatrix3 out;
		for (int i = 0; i < 9; i++)out(i) = this->operator()(i) + s1;
		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator- (zMatrix3 &m1)
	{
		zMatrix3 out;
		for (int i = 0; i < 9; i++)out(i) = this->operator()(i) - m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator- (float s1)
	{
		zMatrix3 out;
		for (int i = 0; i < 9; i++)out(i) = this->operator()(i) - s1;
		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator* (zMatrix3 &m1)
	{
		zMatrix3 out;
		int count = 0;

		for (int i = 0; i < 3; i++)
		{
			zMatrix3Row rVals;
			this->getRow(i, rVals);

			for (int j = 0; j < 3; j++)
			{
				zMatrix3Col cVals;
				m1.getCol(j,cVals);

				float val = 0;
				for (int k = 0; k < 3; k++)	val += rVals[k] * cVals[k];
				out(count) = val;
				count++;
			}
		}

		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::operator* (float s1)
	{
		zMatrix3 out;
		for (int i = 0; i < 9; i++)out(i) = this->operator()(i) * s1;
		return out;
	}

	//---- OVERLOADED OPERATORS

	ZSPACE_INLINE void zMatrix3::operator+= (zMatrix3 &m1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) += m1(i);
	}

	ZSPACE_INLINE void zMatrix3::operator+= (float s1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) += s1;
	}

	ZSPACE_INLINE void zMatrix3::operator-= (zMatrix3 &m1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) -= m1(i);
	}

	ZSPACE_INLINE void zMatrix3::operator-= (float s1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) -= s1;
	}

	ZSPACE_INLINE void zMatrix3::operator*= (float s1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) *= s1;
	}

	ZSPACE_INLINE void zMatrix3::operator/= (float s1)
	{
		for (int i = 0; i < 9; i++) this->operator()(i) /= s1;
	}

	//---- METHODS

	ZSPACE_INLINE void zMatrix3::multiply(zMatrix3Col &m1, zMatrix3Col &out)
	{
		int count = 0;

		for (int i = 0; i < 3; i++)
		{
			zMatrix3Row rVals;
			this->getRow(i, rVals);

			float val = 0;
			for (int k = 0; k < 3; k++)	val += rVals[k] * m1[k];
			out[count] = val;
			count++;
		}
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::transpose()
	{
		zMatrix3 out;

		for (int i = 0; i < 3; i++)
		{
			zMatrix3Col colVals;
			this->getCol(i, colVals);
			out.setRow(i, colVals);
		}

		return out;
	}	

	ZSPACE_INLINE zMatrix2 zMatrix3::minor(int colIndex, int rowIndex)
	{
		zMatrix2 out;
		int iCount = 0;
		int jCount = 0;

		for (int i = 0; i < 3; i++)
		{
			if (i != rowIndex)
			{
				for (int j = 0; j < 3; j++)
				{
					jCount = 0;
					if (j != colIndex)
					{
						out(iCount, jCount) = (this->operator()(i, j));
						jCount++;
					}
				}

				iCount++;
			}
		}

		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::cofactor()
	{
		zMatrix3 out;
		int sign = 1;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{

				zMatrix2 minorM = this->minor(j, i);

				// sign of adj[j][i] positive if sum of row 
				// and column indexes is even. 
				sign = ((i + j) % 2 == 0) ? 1 : -1;

				out(i, j) = (sign * minorM.det());

			}
		}

		return out;;
	}

	ZSPACE_INLINE zMatrix3 zMatrix3::adjoint()
	{
		zMatrix3 cofactorMatrix = this->cofactor();

		return cofactorMatrix.transpose();
	}

	ZSPACE_INLINE bool zMatrix3::inverse(zMatrix3 &outMatrix)
	{
		float det = this->det();

		if (det == 0)
		{
			printf("\n Singular matrix, can't find its inverse");
			outMatrix.setIdentity();
			return false;
		}
		else
		{
			zMatrix3 adjointM = adjoint();			

			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					outMatrix(i,j) = (adjointM(i, j) / det);
			}
						
			return true;
		}
	}

	ZSPACE_INLINE float zMatrix3::det()
	{
		float D = 0;

		int sign = 1;  // To store sign multiplier 

		 // Iterate for each element of first row 
		for (int i = 0; i < 3; i++)
		{
			// Getting Cofactor of mat[0][f] 
			zMatrix2 minorM = this->minor(i);
			D += sign * this->operator()(0, i) * minorM.det();

			// terms are to be added with alternate sign 
			sign = -sign;
		}

		return D;
	}

}

//---- zMATRIX 4X4

namespace zSpace
{
	//---- CONSTRUCTOR
	
	ZSPACE_INLINE zMatrix4::zMatrix4()
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				
				if (i == j) mat[i * 4 + j] = 1;
				else mat[i * 4 + j] = (0);

			}
		}
	}
	
	ZSPACE_INLINE zMatrix4::zMatrix4(const float _mat[16])
	{	
		for (int i = 0; i < 16; i++) mat[i] = _mat[i];
	}	

	//---- DESTRUCTOR
	
	ZSPACE_INLINE zMatrix4::~zMatrix4(){}

	//---- GET METHODS
	
	ZSPACE_INLINE int zMatrix4::getIndex(int rowIndex, int colIndex)
	{
		return rowIndex * 4 + colIndex;
	}
	
	ZSPACE_INLINE void zMatrix4::getIndices(int index, int &rowIndex, int &colIndex)
	{
#ifndef __CUDACC__
		rowIndex = floor(index / 4);
#else
		rowIndex = floorf(index / 4);
#endif
		colIndex = index % 4;
	}
	
	ZSPACE_INLINE void zMatrix4::getRow(int index, zMatrix4Row &row)
	{
		for (int i = 0; i < 4; i++)	row[i] = (this->operator()(index, i));
	}
	
	ZSPACE_INLINE void zMatrix4::getCol(int index, zMatrix4Col &col)
	{
		for (int i = 0; i < 4; i++)	col[i] = (this->operator()(i, index));
	}
	
	ZSPACE_INLINE void zMatrix4::getDiagonal(zMatrix4Diag &diag)
	{
		for (int i = 0; i < 4; i++)	diag[i] = (this->operator()(i, i));	
	}
	
	ZSPACE_INLINE float* zMatrix4::getMatrixValues()
	{
		return mat;
	}

	ZSPACE_INLINE float* zMatrix4::getRawMatrixValues()
	{
		return &mat[0];
	}

	//---- SET METHODS
		
	ZSPACE_INLINE void zMatrix4::setRow(int index, float val)
	{
		for (int i = 0; i < 4; i++) this->operator()(index, i) = val;		
	}

	ZSPACE_INLINE void zMatrix4::setRow(int index, const zMatrix4Row row)
	{		
		for (int i = 0; i < 4; i++) this->operator()(index, i) = row[i];
	}

	ZSPACE_INLINE void zMatrix4::setCol(int index, float val)
	{
		for (int i = 0; i < 4; i++) this->operator()(i, index) = val;
	}
	
	ZSPACE_INLINE void zMatrix4::setCol(int index, const zMatrix4Col col)
	{
		for (int i = 0; i <= 4; i++) this->operator()(i, index) = col[i];
	}

	ZSPACE_INLINE void zMatrix4::setDiagonal(float val)
	{
		for (int i = 0; i < 4; i++)	this->operator()(i, i) = val;
	}

	ZSPACE_INLINE void zMatrix4::setDiagonal(const zMatrix4Diag diag)
	{
		for (int i = 0; i < 4; i++)	this->operator()(i, i) = diag[i];
	}

	ZSPACE_INLINE void zMatrix4::setZero()
	{
		for (int i = 0; i < 16; i++) mat[i] = 0;
	}

	ZSPACE_INLINE void zMatrix4::setOne()
	{
		for (int i = 0; i < 16; i++) mat[i] = 1;
	}

	ZSPACE_INLINE void zMatrix4::setIdentity()
	{
		this->setZero();
		this->setDiagonal(1);
	}

	//---- OPERATORS

	ZSPACE_INLINE float& zMatrix4::operator()(int row, int col)
	{
		return mat[row * 4 + col];
	}

	ZSPACE_INLINE float& zMatrix4::operator()(int index)
	{
		return mat[index];
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator+ (zMatrix4 &m1)
	{
		zMatrix4 out;
		for (int i = 0; i < 16; i++)out(i) = this->operator()(i) + m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator+ (float s1)
	{
		zMatrix4 out;
		for (int i = 0; i < 16; i++)out(i) = this->operator()(i) + s1;
		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator- (zMatrix4 &m1)
	{
		zMatrix4 out;
		for (int i = 0; i < 16; i++)out(i) = this->operator()(i) - m1(i);
		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator- (float s1)
	{
		zMatrix4 out;
		for (int i = 0; i < 16; i++)out(i) = this->operator()(i) - s1;
		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator* (zMatrix4 &m1)
	{
		zMatrix4 out;
		int count = 0;

		for (int i = 0; i < 4; i++)
		{
			zMatrix4Row rVals;
			this->getRow(i,rVals);

			for (int j = 0; j < 4; j++)
			{
				zMatrix4Col cVals;
				m1.getCol(j,cVals);

				float val = 0;
				for (int k = 0; k < 4; k++)	val += rVals[k] * cVals[k];
				out(count) = val;
				count++;
			}
		}

		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::operator* (float s1)
	{
		zMatrix4 out;
		for (int i = 0; i < 16; i++)out(i) = this->operator()(i) * s1;
		return out;
	}

	//---- OVERLOADED OPERATORS
	
	ZSPACE_INLINE void zMatrix4::operator+= (zMatrix4 &m1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) += m1(i);
	}
	
	ZSPACE_INLINE void zMatrix4::operator+= (float s1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) += s1;
	}
	
	ZSPACE_INLINE void zMatrix4::operator-= (zMatrix4 &m1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) -= m1(i);
	}
	
	ZSPACE_INLINE
	void zMatrix4::operator-= (float s1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) -= s1;
	}
	
	ZSPACE_INLINE void zMatrix4::operator*= (float s1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) *= s1;
	}
	
	ZSPACE_INLINE void zMatrix4::operator/= (float s1)
	{
		for (int i = 0; i < 16; i++) this->operator()(i) /= s1;
	}

	//---- METHODS
	
	ZSPACE_INLINE void zMatrix4::multiply(zMatrix4Col &m1, zMatrix4Col &out)
	{		
		int count = 0;

		for (int i = 0; i < 4; i++)
		{
			zMatrix4Row rVals;
			this->getRow(i, rVals);

			float val = 0;
			for (int k = 0; k < 4; k++)	val += rVals[k] * m1[k];
			out[count] = val;
			count++;
		}		
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::transpose()
	{
		zMatrix4 out;

		for (int i = 0; i < 4; i++)
		{
			zMatrix4Col colVals;
			this->getCol(i,colVals);
			out.setRow(i, colVals);
		}

		return out;
	}

	ZSPACE_INLINE zMatrix3 zMatrix4::minor(int colIndex, int rowIndex)
	{
		zMatrix3 out;
		int iCount = 0;
		int jCount = 0;

		for (int i = 0; i < 4; i++)
		{
			if (i != rowIndex)
			{
				for (int j = 0; j < 4; j++)
				{
					jCount = 0;					
					if (j != colIndex)
					{
						out(iCount, jCount) = (this->operator()(i, j));
						jCount++;
					}
				}

				iCount++;
			}
		}

		return out;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::cofactor()
	{
		zMatrix4 out;
		int sign = 1;

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j <4; j++)
			{

				zMatrix3 minorM = this->minor(j, i);

				// sign of adj[j][i] positive if sum of row 
				// and column indexes is even. 
				sign = ((i + j) % 2 == 0) ? 1 : -1;

				out(i,j) = (sign * minorM.det());

			}
		}

		return out;;
	}

	ZSPACE_INLINE zMatrix4 zMatrix4::adjoint()
	{
		zMatrix4 cofactorMatrix = this->cofactor();

		return cofactorMatrix.transpose();
	}

	ZSPACE_INLINE bool zMatrix4::inverse(zMatrix4 &outMatrix)
	{
		float det = this->det();

		if (det == 0)
		{
			printf("\n Singular matrix, can't find its inverse");
			outMatrix.setIdentity();
			return false;
		}
		else
		{
			zMatrix4 adjointMat = adjoint();

			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
					outMatrix(i, j) = (adjointMat(i, j) / det);
			}

			return true;
		}
	}

	ZSPACE_INLINE float zMatrix4::det()
	{
		float D = 0;

		int sign = 1;  // To store sign multiplier 

		 // Iterate for each element of first row 
		for (int i = 0; i < 4; i++)
		{
			// Getting Cofactor of mat[0][f] 
			zMatrix3 minorM = this->minor(i);
			D += sign * this->operator()(0, i) * minorM.det();

			// terms are to be added with alternate sign 
			sign = -sign;
		}

		return D;
	}

}

