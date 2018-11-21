#pragma once

#include<headers/core/zVector.h>
#include<headers/core/zMatrix.h>
#include<headers/core/zColor.h>

namespace zSpace
{
	/** \addtogroup zSpaceDisplay
	*	\brief Collection of general display and console print utility methods along with bufferobject class. It uses OPENGL framework for the display operations. 
	*  @{
	*/
	
	//--------------------------
	//---- MATRIX PRINT UTILITIES
	//--------------------------

	/*! \brief This methods prints the zMatrix values to the console.
	*
	*	\param		[in]	inMatrix	- matrix to be printed
	*	\since version 0.0.1
	*/
	template <typename T>
	void printMatrix(zMatrix<T> &inMatrix)
	{
		for (int i = 0; i < inMatrix.getNumRows(); i++)
		{
			printf("\n ");

			for (int j = 0; j < inMatrix.getNumCols(); j++)
			{
				cout << inMatrix(i, j) << " ";
			}

		}

		printf("\n ");
	}

	//--------------------------
	//---- VECTOR PRINT UTILITIES
	//--------------------------

	/*! \brief This methods prints the zVector values to the console.
	*
	*	\param		[in]	inVec	- vector to be printed
	*	\since version 0.0.1
	*/

	void printVector(zVector &inVec)
	{
		printf("\n  %1.2f %1.2f %1.2f \n", inVec.x, inVec.y, inVec.z);
	}

	//--------------------------
	//---- COLOR PRINT UTILITIES
	//--------------------------

	/*! \brief This methods prints the zColor values to the console.
	*
	*	\param		[in]	inCol	- vector to be printed
	*	\param		[in]	colType	- zColorType.
	*	\since version 0.0.1
	*/

	void printColor(zColor &inCol, zColorType colType)
	{
		if (colType = zRGB) printf("\n  %1.2f %1.2f %1.2f %1.2f \n", inCol.r, inCol.g, inCol.b, inCol.a);
		if (colType = zHSV) printf("\n  %1.2f %1.2f %1.2f \n", inCol.h, inCol.s, inCol.v);
	}

	/** @}*/ 

}
