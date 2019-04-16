
#pragma once

#include <stdexcept>
#include <vector>
using namespace std;

namespace zSpace
{

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
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
	class zMatrix
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
		zMatrix()
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

		/*! \brief Overloaded Constructor.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_rows		- number of rows.
		*	\param		[in]	_cols		- number of columns.
		*	\param		[in]	_mat		- values of the matrix.
		*	\since version 0.0.1
		*/
		zMatrix(int _rows, int _cols, vector<T> &_mat)
		{
			if (_rows*_cols != _mat.size()) throw std::invalid_argument("input _mat size not equal to (_rows * _cols).");

			rows = _rows;
			cols = _cols;

			mat.clear();
			mat = _mat;
		}


		/*! \brief Overloaded Constructor for a square matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_dims		- number of rows and cols.
		*	\since version 0.0.1
		*/
		zMatrix(int _dims)
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

		/*! \brief Overloaded Constructor for a matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_rows		- number of rows.
		*	\param		[in]	_cols		- number of cols.
		*	\since version 0.0.1
		*/
		zMatrix(int _rows, int _cols)
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

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zMatrix()
		{

		}


		//--------------------------
		//---- ADD REMOVE METHODS
		//--------------------------

		/*! \brief This method gets the column values at the input column index.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input column index.
		*	\return				vector<T>	- vector of column values.
		*	\since version 0.0.1
		*/
		void addRow(vector<T> vals)
		{
			/*if (vals.size() != this->getNumCols()) throw std::invalid_argument("cannot add row as the number columns and size of vals dont match");

			

			for (int i = 0; i < this->getNumRows(); i++)
			{
				out.push_back(this->operator()(i, index));
			}*/

			//return out;
		}

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the number of rows in the matrix.
		*	\return				int		- number of rows.
		*	\since version 0.0.1
		*/
		int getNumRows()
		{
			return this->rows;
		}


		/*! \brief This method gets the number of columns in the matrix.
		*	\return				int		- number of columns.
		*	\since version 0.0.1
		*/
		int getNumCols()
		{
			return this->cols;
		}

		/*! \brief This method gets the index in the matrix value container given the row and column indicies.
		*	\param		[in]	int		- row index.
		*	\param		[in]	int		- column index.
		*	\return				int		- index in value container.
		*	\since version 0.0.1
		*/
		int getIndex(int rowIndex, int colIndex)
		{
			if (rowIndex < 0 || rowIndex > this->getNumRows()) throw std::invalid_argument("input rowIndex out of bounds.");
			if (colIndex < 0 || colIndex > this->getNumCols()) throw std::invalid_argument("input colIndex out of bounds.");

			return rowIndex * this->getNumCols() + colIndex;
		}

		/*! \brief This method gets the row and column index in the matrix given the input container index.
		*	\param		[in]	int		- index.
		*	\param		[out]	int		- row index.
		*	\param		[out]	int		- column index.
		*	\since version 0.0.1
		*/
		void getIndices(int index, int &rowIndex, int &colIndex)
		{
			if (index < 0 || index > this->getNumRows() * this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
			
			rowIndex = 

			rowIndex = floor(index / this->getNumCols());
			colIndex = index % this->getNumCols();		
		}

		/*! \brief This method gets the row values as container of values at the input row index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\return				vector<T>	- vector of row values.
		*	\since version 0.0.1
		*/
		vector<T> getRow(int index)
		{
			if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

			vector<T> out;

			for (int i = 0; i < this->getNumCols(); i++)
			{
				out.push_back(this->operator()(index, i));
			}

			return out;
		}

		/*! \brief This method gets the row matrix at the input row index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\return				zMatrix<T>	- row matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> getRowMatrix(int index)
		{
			if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

			zMatrix<T> out (1 , this->getNumCols());

			for (int i = 0; i < this->getNumCols(); i++)
			{
				out(0,i)  = (this->operator()(index, i));
			}

			return out;
		}

		/*! \brief This method gets the column values at the input column index.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input column index.
		*	\return				vector<T>	- vector of column values.
		*	\since version 0.0.1
		*/
		vector<T> getCol(int index)
		{
			if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

			vector<T> out;

			for (int i = 0; i < this->getNumRows(); i++)
			{
				out.push_back(this->operator()(i, index));
			}

			return out;
		}

		/*! \brief This method gets the column matrix at the input column index.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input column index.
		*	\return				zMatrix<T>	- column matrix.
		*	\since version 0.0.1
		*/
		zMatrix<T> getColumnMatrix(int index)
		{
			if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

			vector<T> out;

			for (int i = 0; i < this->getNumRows(); i++)
			{
				out(i, 0) = (this->operator()(i, index));
			}

			return out;
		}

		/*! \brief This method gets the diagonal values if it is a square matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				vector<T>	- vector of diagonal values.
		*	\since version 0.0.1
		*/

		vector<T> getDiagonal()
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


		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the mats values with the input list of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	_mat		- vector of matrix values.
		*	\since version 0.0.1
		*/

		void setMat(vector<T> & _mat)
		{
			if (this->getNumCols() * this->getNumRows() != _mat.size()) throw std::invalid_argument("input _mat size not equal to (rows * cols).");
						
			mat.clear();
			mat = _mat;
		}

		/*! \brief This method sets the row values at the input row index with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	val			- value of the row.
		*	\since version 0.0.1
		*/
		void setRow(int index, T val)
		{
			if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");

			for (int i = 0; i < this->getNumCols(); i++)
			{
				this->operator()(index, i) = val;


			}

		}


		/*! \brief This method sets the row values at the input row index with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	vals		- vector of values for the row.
		*	\since version 0.0.1
		*/
		void setRow(int index, vector<T>& vals)
		{
			if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");
			if (vals.size() != this->getNumCols()) throw std::invalid_argument("input values size dont match with number of columns.");
			
			for (int i = 0; i < this->getNumCols(); i++)
			{
				this->operator()(index, i) = vals[i];
			}

		}

		/*! \brief This method sets the row values at the input row index with the input row Matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	rowMatrix	- row matrix.
		*	\since version 0.0.1
		*/
		void setRow(int index, zMatrix<T> &rowMatrix)
		{
			if (index < 0 || index > this->getNumRows()) throw std::invalid_argument("input index out of bounds.");
			if (rowMatrix.getNumCols() != this->getNumCols()) throw std::invalid_argument("input values size dont match with number of columns.");
			if (rowMatrix.getNumRows() != 1) throw std::invalid_argument("input in not row matrix.");

			for (int i = 0; i < this->getNumCols(); i++)
			{
				this->operator()(index, i) = rowMatrix(0,i);
			}

		}

		/*! \brief This method sets the col values at the input col index with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input col index.
		*	\param		[in]	val			- value of the col.
		*	\since version 0.0.1
		*/
		void setCol(int index, T val)
		{
			if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
		
			for (int i = 0; i < this->getNumRows(); i++)
			{
				this->operator()(i, index) = val;
			}

		}


		/*! \brief This method sets the col values at the input col index with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input col index.
		*	\param		[in]	vals		- vector of values for the col.
		*	\since version 0.0.1
		*/
		void setCol(int index, vector<T>& vals)
		{
			if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
			if (vals.size() != this->getNumRows()) throw std::invalid_argument("input values size dont match with number of rows.");

			for (int i = 0; i < this->getNumRows(); i++)
			{
				this->operator()(i, index) = vals[i];
			}

		}


		/*! \brief This method sets the row values at the input column index with the input column Matrix.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	index		- input row index.
		*	\param		[in]	colMatrix	- column matrix.
		*	\since version 0.0.1
		*/
		void setCol(int index, zMatrix<T> &colMatrix)
		{
			if (index < 0 || index > this->getNumCols()) throw std::invalid_argument("input index out of bounds.");
			if (colMatrix.getNumRows() != this->getNumRows()) throw std::invalid_argument("input values size dont match with number of rows.");
			if (colMatrix.getNumCols() != 1) throw std::invalid_argument("input in not column matrix.");

			for (int i = 0; i < this->getNumRows(); i++)
			{
				this->operator()(i, index) = rowMatrix(i, 0);
			}

		}

		/*! \brief This method sets the diagonal values of a square matrix with the input value.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val			- value of the diagonal.
		*	\since version 0.0.1
		*/
		void setDiagonal(T val)
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

		/*! \brief This method sets the diagonal values of a square matrix with the input vector of values.
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- vector of values for the diagonal.
		*	\since version 0.0.1
		*/

		void setDiagonal(vector<T> vals)
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


		/*! \brief This method sets values of the matrix to zero.
		*	\since version 0.0.1
		*/

		void setZero()
		{

			for (int i = 0; i < getNumRows(); i++)
			{
				for (int j = 0; j < getNumCols(); j++)
				{
					this->operator()(i, j) = 0;
				}
			}

		}


		/*! \brief This method sets values of the matrix to one.
		*	\since version 0.0.1
		*/

		void setOne()
		{

			for (int i = 0; i < getNumRows(); i++)
			{
				for (int j = 0; j < getNumCols(); j++)
				{
					this->operator()(i, j) = 1;
				}
			}

		}

		/*! \brief This method sets the matrix to identity if it is a square matrix.
		*	\since version 0.0.1
		*/

		void setIdentity()
		{
			_ASSERT_EXPR(getNumRows() == getNumCols(), "input not a square matrix.");

			this->setZero();
			this->setDiagonal(1);
		}

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

		T& operator()(int row, int col)
		{
			if (row < 0 && row > this->getNumRows()) throw std::invalid_argument("input row out of bounds.");
			if (col < 0 && col > this->getNumRows()) throw std::invalid_argument("input col out of bounds.");

			return mat[(row * this->getNumCols()) + col];
		}

		/*! \brief This operator returns the reference  to the coefficient at at the given index.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	id		- input index.
		*	\return				T		- reference  to the coefficient.
		*	\since version 0.0.1
		*/

		T& operator()(int index)
		{
			if (index < 0 && index >= this->getNumRows() * this->getNumCols()) throw std::invalid_argument("input index out of bounds.");

			return mat[index];
		}

		/*! \brief This operator is used for matrix addition.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\return				zMatrix	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator+ (zMatrix<T> &m1)
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


		/*! \brief This operator is used for scalar addition to a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\return				zMatrix	- resultant matrix after the addition.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator+ (T s1)
		{

			vector<T> out;

			for (int i = 0; i < this->getNumRows() *this->getNumCols(); i++)
			{
				T val = this->operator()(i) + s1;
				out.push_back(val);
			}

			return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
		}

		/*! \brief This operator is used for matrix subtraction.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is subtracted from the current matrix.
		*	\return				zMatrix	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator- (zMatrix<T> &m1)
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

		/*! \brief This operator is used for scalar subtraction from a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is subtracted from the current matrix.
		*	\return				zMatrix	- resultant matrix after the subtraction.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator- (T s1)
		{
			vector<T> out;

			for (int i = 0; i < this->getNumRows() *this->getNumCols(); i++)
			{
				T val = this->operator()(i) - s1;
				out.push_back(val);
			}

			return zMatrix<T>(this->getNumRows(), this->getNumCols(), out);
		}

		/*! \brief This operator is used for matrix multiplication.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is multiplied with the current matrix.
		*	\return				zMatrix	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator* (zMatrix<T> &m1)
		{
			if (this->getNumCols()!= m1.getNumRows()) throw std::invalid_argument("number of columns in current matrix not equal to number of rows in m1.");
			
			vector<T> out;

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

					out.push_back(val);
				}
			}

			return zMatrix<T>(this->getNumRows(), m1.getNumCols(), out);
		}


		/*! \brief This operator is used for scalar multiplication with a matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is multiplied with the current matrix.
		*	\return				zMatrix	- resultant matrix after the multiplication.
		*	\since version 0.0.1
		*/

		zMatrix<T> operator* (T s1)
		{
			vector<T> out;

			for (int i = 0; i < getNumRows() *getNumCols(); i++)
			{
				T val = this->operator()(i) * s1;
				out.push_back(val);
			}

			return zMatrix<T>(getNumRows(), getNumCols(), out);
		}

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------


		/*! \brief This operator is used for matrix addition and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\since version 0.0.1
		*/

		void operator+= (zMatrix<T> &m1)
		{
			if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
			if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");

			for (int i = 0; i < getNumRows() *getNumCols(); i++)
			{

				this->operator()(i) += m1(i);
			}

		}


		/*! \brief This operator is used for scalar addition and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/

		void operator+= (T s1)
		{
			for (int i = 0; i < getNumRows() *getNumCols(); i++)
			{
				this->operator()(i) += s1;
			}

		}

		/*! \brief This operator is used for matrix subtraction and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	m1		- zMatrix which is added to the current matrix.
		*	\since version 0.0.1
		*/

		void operator-= (zMatrix<T> &m1)
		{
			if (this->getNumRows() != m1.getNumRows()) throw std::invalid_argument("number of rows not equal.");
			if (this->getNumCols() != m1.getNumCols()) throw std::invalid_argument("number of cols not equal.");

			for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
			{

				this->operator()(i) -= m1(i);
			}

		}


		/*! \brief This operator is used for scalar subtraction and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is added to the current matrix.
		*	\since version 0.0.1
		*/

		void operator-= (T s1)
		{

			for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
			{

				this->operator()(i) -= s1;
			}

		}


		/*! \brief This operator is used for scalar multiplication and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which is multiplied to the current matrix.
		*	\since version 0.0.1
		*/

		void operator*= (T s1)
		{

			for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
			{

				this->operator()(i) *= s1;
			}

		}


		/*! \brief This operator is used for scalar division and assigment of the result to the current matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	s1		- scalar value which divides the current matrix.
		*	\since version 0.0.1
		*/

		void operator/= (T s1)
		{

			for (int i = 0; i < this->getNumRows() * this->getNumCols(); i++)
			{

				mat[i] /= s1;
			}

		}


		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method returns the transpose of the input matrix.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\return				zMatrix	- resultant transpose matrix.
		*	\since version 0.0.1
		*/

		zMatrix<T> transpose()
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


		/*! \brief This method returns the minor/sub matrix of the input square matrix removing the row and column values given by the input colindex and rowIndex.
		*
		*	\tparam				T		- Type to work with int, double, float
		*	\param		[in]	colIndex- column index to be removed.
		*	\return				zMatrix	- resultant leftover matrix.
		*	\since version 0.0.1
		*/

		zMatrix<T> minorMatrix(int colIndex, int rowIndex = 0)
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


		/*! \brief This method returns the cofactor matrix of the input square matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				zMatrix		- resultant cofactor zMatrix.
		*	\since version 0.0.1
		*/

		zMatrix<T> cofactorMatrix()
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


		/*! \brief This method returns the adjoint matrix of the input square matrix.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\return				zMatrix		- resultant adjoint zMatrix.
		*	\since version 0.0.1
		*/

		zMatrix<T> adjointMatrix()
		{
			if (getNumCols() != getNumRows()) throw std::invalid_argument("input Matrix is not a square.");

			zMatrix<T> cofactorMatrix = this->cofactorMatrix();

			return cofactorMatrix.transpose();
		}

		/*! \brief This method returns the inverse matrix of the input square matrix, if it exists.
		*
		*   \details Based on  https://www.geeksforgeeks.org/adjoint-inverse-matrix/
		*	\tparam				T   - Type to work with standard c++ numerical datatypes.
		*	\param		[out]	zMatrix		- resultant inverse zMatrix.
		*	\return				bool		- true if inverse matrix exists.
		*	\since version 0.0.1
		*/

		bool inverseMatrix(zMatrix<T>& outMat)
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
					for (int j = 0; j<this->getNumRows(); j++)
						vals.push_back(adjoint(i, j) / det);
				}

				outMat = zMatrix<T>(this->getNumRows(), this->getNumCols(), vals);
				return true;
			}

		}

		/*! \brief This method returns the determinant of the input matrix if it is a square matrix.
		*
		*	\details Based on https://www.geeksforgeeks.org/determinant-of-a-matrix/
		*	\tparam				T		- Type to work with int, double, float
		*	\return				T		- determinant value.
		*	\since version 0.0.1
		*/

		T det()
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


	

		

	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
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
	*	\brief A 4X4 matrix  of doubles.
	*
	*	\since version 0.0.1
	*/
	typedef zMatrix<double> zTransform;
	
	/** @}*/
	/** @}*/ 
	/** @}*/
	// end of group zCore


}


