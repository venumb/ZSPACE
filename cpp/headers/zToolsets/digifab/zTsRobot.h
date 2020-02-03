// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Authors : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Federico Borello < federico.borello@zaha-hadid.com>
//

#ifndef ZSPACE_TS_DIGIFAB_ROBOT_H
#define ZSPACE_TS_DIGIFAB_ROBOT_H

#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>

#include <iostream>
using namespace std;

namespace zSpace
{

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsDigiFab
	*	\brief toolsets for digital fabrication related utilities.
	*  @{
	*/

	/** \addtogroup zInverseKinematics
	*	\brief toolsets for inverse kinematics chain.
	*  @{
	*/

	/*! \struct zDHparameter
	*	\brief A struct to store the DH parameters of IK chain.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	struct zDHparameter
	{
		double d;
		double a;
		double alpha;
		double theta;

	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsDigiFab
	*	\brief toolsets for digital fabrication related utilities.
	*  @{
	*/

	/** \addtogroup zInverseKinematics
	*	\brief toolsets for inverse kinematics chain.
	*  @{
	*/

	/*! \struct zJointRotation
	*	\brief A struct to store the joint rotation related data.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	struct zJointRotation
	{
		double rotation;

		double home;

		double minimum;
		double maximum;

		double pulse;
		double mask;
		double offset;
	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsDigiFab
	*	\brief toolsets for digital fabrication related utilities.
	*  @{
	*/

	/** \addtogroup zInverseKinematics
	*	\brief toolsets for inverse kinematics chain.
	*  @{
	*/

	/*! \struct zGCode
	*	\brief A struct to store the G-Code related data.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	struct zGCode
	{
		vector<zJointRotation> rotations;
		double vel;
		int moveType; // 0 = movJ, 1 = movL 

		int endEffectorControl; // 0 - neutral , 1 - on , 2 -0ff

		zVector robotTCP_position;
		zVector target_position;


		double distDifference;
		bool targetReached;

	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsDigiFab
	*	\brief toolsets for digital fabrication related utilities.
	*  @{
	*/

	/** \addtogroup zInverseKinematics
	*	\brief toolsets for inverse kinematics chain.
	*  @{
	*/

	/*! \class zTsRobot
	*	\brief A class for inverse kinematics chain of a 6 Axis Robot.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class  ZSPACE_TOOLS zLink
	{
	public:

		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*! \brief stores transformation matrix of the link */
		zTransform T;	

		/*! \brief stores base transformation matrix of the link */
		zTransform TBase;			

		//--------------------------
		//---- DH ATTRIBUTES
		//--------------------------

		zDHparameter linkDH;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.2
		*/
		zLink();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	DH				- input DH Parameters.	
		*	\since version 0.0.2
		*/
		zLink(zDHparameter &DH);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		~zLink();
		
		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method updates the transform T
		*	\since version 0.0.2
		*/
		void updateTransform();

	};


	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsDigiFab
	*	\brief toolsets for digital fabrication related utilities.
	*  @{
	*/

	/** \addtogroup zInverseKinematics
	*	\brief toolsets for inverse kinematics chain.
	*  @{
	*/
	
	/*! \class zTsRobot
	*	\brief A class for inverse kinematics chain of a 6 Axis Robot.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsRobot
	{
	protected:
		
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- JOINT ATTRIBUTES
		//--------------------------

		/*!	\brief pointer container to joint mesh objects  */
		vector<zObjMesh*> jointMeshObjs;

		/*!	\brief pointer to joint graph object  */
		zObjGraph *jointGraphObj;

		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*!	\brief stores robot scale  */
		double robot_scale = 1.0;

		/*!	\brief contatiner of robot links DH Parameters  */
		vector<zDHparameter> robot_DH;		

		/*!	\brief contatiner of robot links  */
		vector<zLink> Bars;		

		/*!	\brief contatiner of robot joint mesh transform  */
		vector<zTransform> robotMesh_transforms;

		/*!	\brief robot base matrix  */
		zTransform robot_base_matrix;

		/*!	\brief robot target matrix  */
		zTransform robot_target_matrix;

		/*!	\brief robot end effector matrix  */
		zTransform robot_endEffector_matrix;


		/*!	\brief contatiner of robot GCode  */
		vector<zGCode> robot_gCode;
	

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------		

		/*!	\brief container of joint mesh function set  */
		vector<zFnMesh> fnMeshJoints;

		/*!	\brief joint graph function set  */
		zFnGraph fnGraphJoint;

		/*!	\brief contatiner of robot joint rotations  */
		vector<zJointRotation> jointRotations;

		/*!	\brief contatiner of robot joint transforms  */
		vector<zTransform> robotJointTransforms;

		/*!	\brief contatiner of robot mesh dihedral angles  */
		vector<vector<double>> robot_jointMeshes_edgeAngle;

		/*!	\brief contatiner of robot target transforms  */
		vector<zTransform> robotTargets;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsRobot();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_jointGraphObj			- input joint graph object.
		*	\param		[in]	_jointMeshObjs			- input container of joint mesh objects.
		*	\since version 0.0.2
		*/
		zTsRobot(zObjGraph &_jointGraphObj, vector<zObjMesh> &_jointMeshObjs);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsRobot();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the robot from the input parameters.
		*
		*	\param		[in]	_d				- container of d values of DH Parameter per joint.
		*	\param		[in]	_a				- container of a values of DH Parameter per joint.
		*	\param		[in]	_alpha			- container of alpla values of DH Parameter per joint.
		*	\param		[in]	_theta			- container of theta values of DH Parameter per joint.
		*	\param		[in]	_robot_scale	- scale of robot.
		*/
		void createRobot(vector<double>(&_d), vector<double>(&_a), vector<double>(&_alpha), vector<double>(&_theta), double _robot_scale = 1.0);
	
		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createRobotfromFile(string path, zFileTpye type);


		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		directory		- input file directory path.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createRobotJointMeshesfromFile(string directory, zFileTpye type, bool endeffector = false);

		/*! \brief This method creates the robot targets from the input file.
		*
		*	\param [in]		directory		- input file directory path.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createTargetsfromFile(string infilename, zFileTpye type);

		/*! \brief This method create a new target from input position and rotations vectors.
		*
		*	\param [in]		_position		- target position.
		*	\param [in]		_rotationX		- target X rotation axis.
		*	\param [in]		_rotationY		- target Y rotation axis.
		*	\param [in]		_rotationZ		- target Z rotation axis.
		*	\since version 0.0.2
		*/
		void addTarget(zVector& _position, zVector& _rotationX, zVector& _rotationY, zVector _rotationZ);

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method set the robot target matrix.
		*
		*	\param [in]		target			- input target matrix.	
		*	\since version 0.0.2
		*/
		void setTarget(zTransform &target);

		/*! \brief This method set the robot end effector matrix.
		*
		*	\param [in]		EE			- input endeffector matrix.
		*	\since version 0.0.2
		*/
		void setEndEffector(zTransform &EE);

		//--------------------------
		//----KINMATICS METHODS
		//--------------------------

		/*! \brief This Methods computes the joint positionsof the zRobot.
		*
		*	\since version 0.0.2
		*/
		void computeJoints();

		/*! \brief This Methods computes the forward kinematics chain of zRobot for the specified joint rotation values.
		*
		*	\param		[in]	type		- robot joint rotation type.
		*	\since version 0.0.2
		*/
		zVector forwardKinematics(zRobotRotationType rotType = zJoint);

		/*! \brief This Methods computes the inverse kinematics chain of the robot .
		*
		*	\since version 0.0.2
		*/
		zVector inverseKinematics();

		//--------------------------
		//----MESH METHODS
		//--------------------------
		
		/*! \brief This methods sets the joint mesh dihedral edge angles container.
		*
		*	\param		[in]	lightVec		- vector of light.
		*	\since version 0.0.2
		*/
		void setJointMeshDihedralEdges();

		/*! \brief This methods sets the joint mesh occlusion face color .
		*
		*	\param		[in]	lightVec		- vector of light.
		*	\since version 0.0.2
		*/
		void setJointMeshColor(zVector lightVec);

		/*! \brief This methods sets the joint mesh transformation matrix to the joint transform of the robot .
		*
		*	\since version 0.0.2
		*/
		void setJointMeshTransform(bool updatePositions = true);

		//--------------------------
		//----GCODE METHODS
		//--------------------------
		
		/*! \brief This method stores the robot gcode.
		*
		*	\param [in]		target_position			- target position to be stored.
		*	\param [in]		velocity				- robot velocity.
		*	\param [in]		moveType				- robot move type - zMoveLinear/zMoveJoint .
		*	\since version 0.0.2
		*/
		void gCode_store(zVector &target_position, double velocity, zRobotMoveType moveType, zRobotEEControlType endEffectorControl);

		/*! \brief This method exports the robot gcode to the input folder.
		*
		*	\param [in]		directoryPath			- input directory path.
		*	\param [in]		type					- robot type.
		*	\since version 0.0.2
		*/
		void gCode_to(string directoryPath, zRobotType type);


	protected:

		/*! \brief This method create a graph connecting all the joint positions.
		*
		*	\since version 0.0.2
		*/
		void createRobotJointGraph();

		/*! \brief This method updates the joint graph based on the jint rotation matrices.
		*
		*	\since version 0.0.2
		*/
		void update_robotJointGraph();

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method imports robot from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromJSON(string infilename);

		/*! \brief This method imports robot targets from a TXT file format .
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromTXT(string infilename);

		/*! \brief This method exports robot gcode for an ABB robot.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toABBGcode(string infilename);

		/*! \brief This method exports robot gcode for a NACHI MZ07 robot.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toNACHI_MZ07Gcode(string infilename);
	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/digiFab/zTsRobot.cpp>
#endif

#endif