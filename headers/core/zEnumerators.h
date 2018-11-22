#pragma once


	/*!	\namespace	zSpace
	*	\brief		namespace for the library.
	*	\since		version 0.0.1
	*/
namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core classes, enumerators ,defintions and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zCoreEnumerators
	*	\brief Collection of all the enumerators in the library.
	*  @{
	*/

	/*! \enum	zHEData
	*	\brief	Data types of a halfedge data structure.
	*	\since	version 0.0.1
	*/
	
	enum zHEData { zVertexData = 0, zEdgeData, zFaceData};

	/*! \enum	zScalarfieldData
	*	\brief	Algorithm types for field navigation.
	*	\since	version 0.0.1
	*/

	enum zScalarfieldData { zGradientDescent=30, zGradientAscent };

	/*! \enum	zColorData
	*	\brief	Data types of color.
	*	\since	version 0.0.1
	*/
	
	enum zColorType { zRGB = 40, zHSV };

	/*! \enum	zDataLevel
	*	\brief	Grid levels for data in London.
	*	\since	version 0.0.1
	*/

	enum zDataLevel { zLsoaData = 50, zMsoaData, zPostcodeData };

	/*! \enum	zIntergrationType
	*	\brief	Integration types.
	*	\since	version 0.0.1
	*/

	enum zIntergrationType { zEuler = 60, zRK4 , zPixel };

	/*! \enum	zDiffusionType
	*	\brief	Diffusion or averaging types.
	*	\since	version 0.0.1
	*/

	enum zDiffusionType { zLaplacian = 70, zAverage, zLaplacianAdjacent};

	/*! \enum	zSlimeParameter
	*	\brief	Slime mold parameters.
	*	\since	version 0.0.1
	*/

	enum zSlimeParameter { zSO = 80, zSA, zRA, zdepT};


	/** @}*/

	/** @}*/ 
}
