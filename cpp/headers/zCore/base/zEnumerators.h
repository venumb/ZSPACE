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

#ifndef ZSPACE_ENUMERATORS
#define ZSPACE_ENUMERATORS

	/*!	\namespace	zSpace
	*	\brief		namespace for the library.
	*	\since		version 0.0.1
	*/
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

	/** \addtogroup zEnumerators
	*	\brief  The enumerators of the library.
	*  @{
	*/

	/*! \enum	zHEData
	*	\brief	Data types of a halfedge data structure.
	*	\since	version 0.0.1
	*/	
	enum zHEData { zVertexData = 0, zHalfEdgeData, zEdgeData, zFaceData};

	/*! \enum	zScalarfieldData
	*	\brief	Algorithm types for field navigation.
	*	\since	version 0.0.1
	*/
	enum zScalarfieldData { zGradientDescent=30, zGradientAscent };

	/*! \enum	zColourData
	*	\brief	Data types of color.
	*	\since	version 0.0.1
	*/	
	enum zColorType { zRGB = 40, zHSV };


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


	/*! \enum	zSQLCommand
	*	\brief SQLite database command types.
	*	\since	version 0.0.1
	*/
	enum zSQLCommand { zCreate = 90, zInsert, zSelect, zSelectExists, zSelectCreate, zSelectCount, zUpdate, zDrop, zExportCSV };

	/*! \enum	zDataLevel
	*	\brief	Grid levels for data in London.
	*	\since	version 0.0.1
	*/
	enum zDataLevel { zLsoaData = 50, zMsoaData, zPostcodeData };

	
	/*! \enum	zDataBuilding
	*	\brief	Grid levels for data in London.
	*	\since	version 0.0.1
	*/
	enum zDataBuilding { zCommercialBuilding = 100, zResidentialBuilding, zPublicBuilding, zUniversityBuilding, zUndefinedBuilding};
	
	/*! \enum	zDataStreet
	*	\brief	Grid levels for data in London.
	*	\since	version 0.0.1
	*/
	enum zDataStreet { zTrunkStreet = 20, zPrimaryStreet, zSecondaryStreet, zTertiaryStreet, zResidentialStreet, zPedestrianStreet, zCycleStreet, zServiceStreet, zUndefinedStreet };

	/*! \enum	zDatatypes
	*	\brief	Grid levels for data in London.
	*	\since	version 0.0.1
	*/
	enum zDatatypes { zInt = 110, zFloat, zDouble, zString };

	/*! \enum	zWalkTypes
	*	\brief	walk types for Shortest Path Trees.
	*	\since	version 0.0.1
	*/
	enum zWalkType { zEdgePath = 120, zEdgeVisited};

	/*! \enum	zFieldValueType
	*	\brief	field value sampling types.
	*	\since	version 0.0.1
	*/
	enum zFieldValueType { zFieldIndex = 130, zFieldNeighbourWeighted, zFieldAdjacentWeighted, zFieldContainedWeighted};

	/*! \enum	zFieldStreamType
	*	\brief	field stream sampling types.
	*	\since	version 0.0.1
	*/
	enum zFieldStreamType { zForward = 140, zBackward, zForwardBackward};

	/*! \enum	zFileTpye
	*	\brief	input and ouput file types.
	*	\since	version 0.0.2
	*/
	enum zFileTpye { zJSON = 150, zOBJ, zTXT,zMAYATXT, zCSV, zBMP, zPNG, zJPEG, zMESH , zGRAPH};

	/*! \enum	zFnType
	*	\brief	functionset types.
	*	\since	version 0.0.2
	*/
	enum zFnType { zInvalidFn = 160, zPointsFn, zMeshFn, zGraphFn, zMeshFieldFn, zPointFieldFn, zParticleFn, zMeshDynamicsFn, zGraphDynamicsFn};

	/*! \enum	zDiagramType
	*	\brief	diagram types for vaults.
	*	\since	version 0.0.2
	*/
	enum zDiagramType { zFormDiagram = 190, zForceDiagram, zResultDiagram };

	/*! \enum	zRobotType
	*	\brief	robot types.
	*	\since	version 0.0.2
	*/
	enum zRobotType { zRobotABB = 200, zRobotKuka, zRobotNachi };

	/*! \enum	zRobotRotationType
	*	\brief	robot rotation types.
	*	\since	version 0.0.2
	*/
	enum zRobotRotationType { zJoint = 210, zJointHome, zJointMinimum, zJointMaximum };

	/*! \enum	zRobotMoveType
	*	\brief	robot move types.
	*	\since	version 0.0.2
	*/
	enum zRobotMoveType { zMoveLinear = 220, zMoveJoint };

	/*! \enum	zRobotEEControlType
	*	\brief	robot end effector control types.
	*	\since	version 0.0.2
	*/
	enum zRobotEEControlType { zEEOn = 230, zEEOff, zEENeutral };

	/*! \enum	zSpectralVertexType
	*	\brief	spectral processing vertex types.
	*	\since	version 0.0.2
	*/
	enum zSpectralVertexType { zRegular = 240, zMinima, zMaxima, zSaddle };

	/*! \enum	zConnectivityType
	*	\brief	connectivty matrix types.
	*	\since	version 0.0.4
	*/
  enum zConnectivityType { zVertexVertex = 250, zVertexEdge, zFaceVertex, zFaceEdge };


	/** @}*/

	/** @}*/

	/** @}*/
}

#endif