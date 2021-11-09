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

#ifndef ZSPACE_TS_STREAMLINES_2D_H
#define ZSPACE_TS_STREAMLINES_2D_H

#pragma once


#include <headers/zInterface/functionsets/zFnMeshField.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>


namespace zSpace
{

	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/** \addtogroup zTsStreamlines
	*	\brief tool sets for field stream lines.
	*  @{
	*/

	/*! \class zStreamLine
	*	\brief An class to store data of a stream lines.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zStreamLine
	{
	protected:

		

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to stream graph Object  */
		zObjGraph graphObj;

		/*!<stores parent stream index of the stream. -1 if there is no parent.*/
		int parent;

		/*!<container stores the child stream indices of the stream.*/
		vector<int> child;

		/*!<container stores index of closest stream per vertex of thge stream graph. 2 indices per edge - left closest and right closest. -1 if there is no closest stream.*/
		vector<int> closestStream;

		/*!<container stores index of closest stream edge per vertex of thge stream graph. 2 indices per edge - left closest and right closest. -1 if there is no closest stream*/
		vector<int> closestStream_Edge;

		/*!<container stores index of closest stream point per vertex of thge stream graph.2 indices per edge - left closest and right closest. -1 if there is no closest stream*/
		vector<zVector> closestStream_Point;

		bool isValid = false;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief default constructor.
		*
		*	\since version 0.0.1
		*/
		zStreamLine();

		/*! \brief this method sets the parent.
		*
		*	\param		[in]	_parentId			- input parent index.
		*	\since version 0.0.1
		*/
		void setParent( int _parentId);

	};


	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/** \addtogroup zTsStreamlines
	*	\brief tool sets for field stream lines.
	*  @{
	*/

	/*! \class zTsStreams2D
	*	\brief A streamlines tool set for creating streams on a 2D field.
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&integrationType=pdf)
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsStreams2D 
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		
		/*!	\brief pointer to a 2D field  */
		zObjMeshField<zVector> *fieldObj;

		/*!<\brief seperation distance between stream lines.*/
		double *dSep;

		/*!<\brief test seperation distance between stream lines.*/
		double *dTest;

		/*!<\brief minimum length of stream.*/
		double *minLength;

		/*!<\brief maximum length of stream.*/
		double *maxLength;

		/*!<\brief streamType - zForwardbackward / zForward/ zBackward.*/
		zFieldStreamType streamType;

		/*!<\brief timestep for integration.*/
		double *dT;

		/*!<\brief integration integrationType - zEuler / zRK4.*/
		zIntergrationType integrationType;

		/*!<\brief angle of stream rotation.*/
		double *angle;

		/*!<\brief boolean is true if the backward direction is flipped.*/
		bool *flipBackward;

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
				

		/*!	\brief field mesh function set  */
		zFnMeshField<zVector> fnField;

		/*!	\brief 2 dimensional container of stream positions per field index.  */
		vector<vector<zVector>> fieldIndex_streamPositions;		

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zTsStreams2D();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_field			- input vector field 2D.
		*	\since version 0.0.1
		*/
		zTsStreams2D(zObjMeshField<zVector> &_fieldObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zTsStreams2D();
		
		//--------------------------
		//----  SET METHODS
		//--------------------------

		/*! \brief This method sets the separation distance.
		*
		*	\param	[in]	_dSep		- input separation distance.
		*	\since version 0.0.1
		*/
		void setSeperationDistance(double &_dSep);

		/*! \brief This method sets the test separation distance.
		*
		*	\param	[in]	_dSep		- input test separation distance.
		*	\since version 0.0.1
		*/
		void setTestSeperationDistance(double &_dTest);

		/*! \brief This method sets the minimum length of the streams.
		*
		*	\param	[in]	_minLength		- input maximum length.
		*	\since version 0.0.1
		*/
		void setMinLength(double &_minLength);

		/*! \brief This method sets the maximum length of the streams.
		*
		*	\param	[in]	_maxLength		- input maximum length.
		*	\since version 0.0.1
		*/
		void setMaxLength(double &_maxLength);

		/*! \brief This method sets the stream type.
		*
		*	\param	[in]	_streamType		- input stream type
		*	\since version 0.0.1
		*/
		void setIntegrationType(zFieldStreamType _streamType);

		/*! \brief This method sets the integration time step.
		*
		*	\param	[in]	_dT		- input time step
		*	\since version 0.0.1
		*/
		void setTimeStep(double &_dT);

		/*! \brief This method sets the integration type.
		*
		*	\param	[in]	_integrationType		- input integration type.
		*	\since version 0.0.1
		*/
		void setIntegrationType(zIntergrationType _integrationType);

		/*! \brief This method sets the rotation angle.
		*
		*	\param	[in]	_angle		- input angle
		*	\since version 0.0.1
		*/
		void setAngle(double &_angle);

		/*! \brief This method sets the flip backwards boolean.
		*
		*	\param	[in]	_flipBackward		- input flip backwards boolean.
		*	\since version 0.0.1
		*/
		void setFlipBackwards(bool &_flipBackward);
		
		//--------------------------
		//----  2D STREAM LINES METHODS
		//--------------------------

		/*! \brief This method creates the stream lines and stores them as a graph.
		*
		*	\param	[out]	streams							- container of streams.
		*	\param	[in]	start_seedPoints				- container of start seed positions. If empty a random position in the field is considered.
		*	\param	[in]	seedStreamsOnly					- generates streams from the seed points only if true.
		*	\since version 0.0.1
		*/
		void createStreams(vector<zStreamLine>& streams, vector<zVector> &start_seedPoints, bool seedStreamsOnly = false);

		//--------------------------
		//----  2D STREAM LINES METHODS WITH INFLUENCE SCALAR FIELD
		//--------------------------
		
		/*! \brief This method creates the stream lines and stores them as a graph.
		*
		*	\param	[out]	streams							- container of streams.
		*	\param	[in]	start_seedPoints				- container of start seed positions. If empty a random position in the field is considered.
		*	\param	[in]	influenceField					- input scalar field.
		*	\param	[in]	min_Power						- input minimum power value.
		*	\param	[in]	max_Power						- input maximum power value.
		*	\param	[in]	seedStreamsOnly					- generates streams from the seed points only if true.
		*	\since version 0.0.1
		*/
		void createStreams_Influence(vector<zStreamLine>& streams, vector<zVector> &start_seedPoints, zFnMeshField<zScalar>& fnInfluenceField, double min_Power, double max_Power, bool seedStreamsOnly = false);

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------
	protected:

		/*! \brief This method creates a single stream line as a graph.
		*
		*	\param	[in]	streamGraph						- stream graph created from the field.
		*	\param	[in]	seedPoint						- input seed point.
		*	\return			bool							- true if the graph is created.
		*	\since version 0.0.1
		*/
		bool createStreamGraph(zObjGraph &streamGraphObj, zVector &seedPoint);

		/*! \brief This method computes the seed points.
		*
		*	\param	[in]	currentStream					- input current stream.
		*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
		*	\param	[in]	seedPoints						- container of seed points.
		*	\since version 0.0.1
		*/
		void getSeedPoints(zStreamLine& currentStream, int vertexId, vector<zVector> &seedPoints);
		
		/*! \brief This method creates a single stream line as a graph based on a influence scalar field.
		*
		*	\param	[in]	streamGraph						- stream graph created from the field.
		*	\param	[in]	seedPoint						- input seed point.
		*	\param	[in]	influenceField					- input scalar field.
		*	\param	[in]	min_Power						- input minimum power value.
		*	\param	[in]	max_Power						- input maximum power value.
		*	\return			bool							- true if the graph is created.
		*	\since version 0.0.1
		*/
		bool createStreamGraph_Influence(zObjGraph &streamGraphObj, zVector &seedPoint, zFnMeshField<zScalar>& fnInfluenceField, double min_Power, double max_Power);

		/*! \brief This method computes the seed points.
		*
		*	\param	[in]	influenceField					- input scalar field.
		*	\param	[in]	currentStream				- input current stream line.
		*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
		*	\param	[in]	min_Power						- input minimum power value.
		*	\param	[in]	max_Power						- input maximum power value.
		*	\param	[in]	seedPoints						- container of seed points.
		*	\since version 0.0.1
		*/
		void getSeedPoints_Influence(zFnMeshField<zScalar>& fnInfluenceField, zStreamLine& currentStream, int vertexId, double min_Power, double max_Power, vector<zVector> &seedPoints);

		//--------------------------
		//----  2D FIELD UTILITIES
		//--------------------------	

		/*! \brief This method checks if the input position is in the bounds of the field.
		*
		*	\param	[in]	inPoint		- input point.
		*	\return			bool		- true if the input position is in bounds.
		*	\since version 0.0.1
		*/
		bool checkFieldBounds(zVector &inPoint);

		/*! \brief This method checks if the input position is a valid stream position.
		*
		*	\param	[in]	inPoint							- input point.
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		bool checkValidStreamPosition(zVector &inPoint, double &dTest);

		/*! \brief This method checks if the input position is a valid seed position.
		*
		*	\param	[in]	inPoint							- input point.
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		bool checkValidSeedPosition(zVector &inPoint, double &dSep);

		/*! \brief This method adds the input position to the field stream position container.
		*
		*	\param	[in]	inPoint							- input point.
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		void addToFieldStreamPositions(zVector &inPoint);
	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/streamlines/zTsStreams2D.cpp>
#endif

#endif