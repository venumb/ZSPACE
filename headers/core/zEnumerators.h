#pragma once


	/*!	\namespace	zSpace
	*	\brief		namespace for the library.
	*	\since		version 0.0.1
	*/
namespace zSpace
{
	/*! \enum	zHEData
	*	\brief	Different data types of a halfedge data structure.
	*	\since	version 0.0.1
	*/
	
	enum zHEData { zVertexData = 0, zEdgeData, zFaceData};

	/*! \enum	zScalarfieldData
	*	\brief	Different field navigation algorithm.
	*	\since	version 0.0.1
	*/

	enum zScalarfieldData { zGradientDescent=30, zGradientAscent };

	/*! \enum	zColorData
	*	\brief	Different data type of a color.
	*	\since	version 0.0.1
	*/
	
	enum zColorType { zRGB = 40, zHSV };

	/*! \enum	zDataLevel
	*	\brief	Different grid levels for data in London.
	*	\since	version 0.0.1
	*/

	enum zDataLevel { zLsoaData = 50, zMsoaData, zPostcodeData };

	/*! \enum	zIntergrationType
	*	\brief	Different integration types.
	*	\since	version 0.0.1
	*/

	enum zIntergrationType { zEuler = 60, zRK4 , zPixel };

	/*! \enum	zDiffusionType
	*	\brief	Different diffusion of averaging types.
	*	\since	version 0.0.1
	*/

	enum zDiffusionType { zLaplacian = 70, zAverage, zLaplacianAdjacent};

	/*! \enum	zSlimeParameter
	*	\brief	Different slime mould parameters.
	*	\since	version 0.0.1
	*/

	enum zSlimeParameter { zSO = 80, zSA, zRA, zdepT};

}
