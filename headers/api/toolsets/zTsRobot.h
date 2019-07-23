#pragma once

#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>

namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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

	/** @}*/

	struct zDHparameter
	{
		double d;
		double a;
		double alpha;
		double theta;

	};

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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

	/** @}*/

	class  zLink
	{
	public:

		//---- ATTRIBUTES

		/*! \brief stores transformation matrix of the link */
		zTransform T;	

		/*! \brief stores base transformation matrix of the link */
		zTransform TBase;			

		//---- DH ATTRIBUTES

		zDHparameter linkDH;

		//---- CONSTRUCTOR

		/*! \brief Default constructor.
		*	\since version 0.0.2
		*/
		zLink()
		{
			linkDH.alpha = linkDH.d = linkDH.theta = linkDH.a = 0.0;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	DH				- input DH Parameters.	
		*	\since version 0.0.2
		*/
		zLink(zDHparameter &DH)
		{
			linkDH = DH;			

			updateTransform();
		}

		//---- DESTRUCTOR

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		~zLink(){}

		//---- METHODS

		/*! \brief This method updates the transform T
		*	\since version 0.0.2
		*/
		void updateTransform()
		{
			T(0, 0) = cos(linkDH.theta);
			T(0, 1) = -sin(linkDH.theta) * cos(linkDH.alpha);
			T(0, 2) = sin(linkDH.theta) * sin(linkDH.alpha);
			T(0, 3) = linkDH.a * cos(linkDH.theta);

			T(1, 0) = sin(linkDH.theta);
			T(1, 1) = cos(linkDH.theta) * cos(linkDH.alpha);
			T(1, 2) = -cos(linkDH.theta)*sin(linkDH.alpha);
			T(1, 3) = linkDH.a * sin(linkDH.theta);

			T(2, 0) = 0;
			T(2, 1) = sin(linkDH.alpha);
			T(2, 2) = cos(linkDH.alpha);
			T(2, 3) = linkDH.d;

			T(3, 0) = 0;
			T(3, 1) = 0;
			T(3, 2) = 0;
			T(3, 3) = 1;
		}

	};


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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

	/** @}*/

	class zTsRobot
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

		//--------------------------
		//---- MESH ATTRIBUTES
		//--------------------------

	

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
		zTsRobot() 
		{
			robot_scale = 1.0;

			for (int i = 0; i < DOF; i++)
			{
				zDHparameter DH;
				DH.alpha = DH.d = DH.theta = DH.a = 0.0;
				robot_DH.push_back(DH);

				zJointRotation jointRot;
				jointRot.home = jointRot.minimum = jointRot.maximum = jointRot.rotation = 0;
				jointRot.pulse = jointRot.mask = jointRot.offset = 0.0;
				jointRotations.push_back(jointRot);

				robotJointTransforms.push_back(zTransform(4,4));
				robotMesh_transforms.push_back(zTransform(4,4));
			}

			for (int i = 0; i < DOF; i++) jointMeshObjs.push_back( nullptr);
			
			jointGraphObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_jointGraphObj			- input joint graph object.
		*	\param		[in]	_jointMeshObjs			- input container of joint mesh objects.
		*	\since version 0.0.2
		*/
		zTsRobot(zObjGraph &_jointGraphObj, vector<zObjMesh> &_jointMeshObjs)
		{
			jointGraphObj = &_jointGraphObj;
			fnGraphJoint = zFnGraph(_jointGraphObj);

			jointMeshObjs.clear();
			fnMeshJoints.clear();

			for (int i = 0; i < _jointMeshObjs.size(); i++)
			{
				jointMeshObjs.push_back(&_jointMeshObjs[i]);
				fnMeshJoints.push_back(_jointMeshObjs[i]);		
			}

			

		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsRobot() {}

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
		void createRobot(vector<double>(&_d), vector<double>(&_a), vector<double>(&_alpha), vector<double>(&_theta), double _robot_scale = 1.0)
		{
			robot_scale = _robot_scale;

			for (int i = 0; i < DOF; i++)
			{
				zDHparameter DH;

				DH.alpha = _alpha[i] * DEG_TO_RAD;
				DH.d = _d[i] * robot_scale;
				DH.theta = _theta[i] * DEG_TO_RAD;
				DH.a = _a[i] * robot_scale;

				Bars.push_back(zLink(DH));

				robot_DH.push_back(DH);
			}


			for (int i = 0; i < DOF; i++)
			{
				zJointRotation jointRot;
				jointRot.home = jointRot.minimum = jointRot.maximum = jointRot.rotation = 0;
				jointRot.pulse = jointRot.mask = jointRot.offset = 0.0;
				jointRotations.push_back(jointRot);


				robotJointTransforms.push_back(zTransform(4,4));
				robotMesh_transforms.push_back(zTransform(4,4));
			}
			
			createRobotJointGraph();
		
			forwardKinematics(zJointHome);	

			setJointMeshTransform(false);
			
		}
	
		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createRobotfromFile(string path, zFileTpye type)
		{
			if (type == zJSON)
			{
				fromJSON(path);
				
				createRobotJointGraph();

				computeJoints();

				forwardKinematics(zJointHome);	

				setJointMeshTransform(false);
			}
			else throw std::invalid_argument(" invalid file type.");
		}


		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		directory		- input file directory path.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createRobotJointMeshesfromFile(string directory, zFileTpye type)
		{
			fnMeshJoints.clear();

			if (type == zJSON)
			{
				// import Base
				for (int i = 0; i < 1; i++)
				{
					string path = directory;
					path.append("/r_base.json");
					fnMeshJoints[0].from(path, type, true);

					
				}
				
				// import joints
				for (int i = 1; i <= DOF; i++)
				{
					string path = directory;
					path.append("/r_");
					path.append(to_string(i));
					path.append(".json");
					fnMeshJoints[i].from(path, type, true);

				}

				// import EE
				for (int i = 0; i < 1; i++)
				{
					string path = directory;
					path.append("/r_EE.json");
					fnMeshJoints[7].from(path, type, false);

				}		

				setJointMeshColor(zVector(0, 0, 1));

			}

			else if (type == zOBJ)
			{
				// import Base
				for (int i = 0; i < 1; i++)
				{
					string path = directory;
					path.append("/r_base.obj");
					fnMeshJoints[0].from(path, type, true);
				
					
				}

				// import joints
				for (int i = 1; i <= DOF; i++)
				{
					string path = directory;
					path.append("/r_");
					path.append(to_string(i));
					path.append(".obj");
					fnMeshJoints[i].from(path, type, true);
										
				}

				//// import EE
				for (int i = 0; i < 1; i++)
				{
					string path = directory;
					path.append("/r_EE.obj");
					fnMeshJoints[7].from(path, type);
					
				}

				setJointMeshColor(zVector(0,0,-1));
				
			}

			else throw std::invalid_argument(" error: invalid zFileTpye type");

		
		}


		/*! \brief This method creates the robot targets from the input file.
		*
		*	\param [in]		directory		- input file directory path.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createTargetsfromFile(string infilename, zFileTpye type)
		{
			if (type == zTXT)
			{
				fromTXT(infilename);
			}
		}

		/*! \brief This method create a new target from input position and rotations vectors.
		*
		*	\param [in]		_position		- target position.
		*	\param [in]		_rotationX		- target X rotation axis.
		*	\param [in]		_rotationY		- target Y rotation axis.
		*	\param [in]		_rotationZ		- target Z rotation axis.
		*	\since version 0.0.2
		*/
		void addTarget(zVector& _position, zVector& _rotationX, zVector& _rotationY, zVector _rotationZ)
		{
			zTransform target;

			target.setIdentity();

			target(0, 0) = _rotationX.x;
			target(0, 1) = _rotationX.y;
			target(0, 2) = _rotationX.z;

			target(1, 0) = _rotationY.x;
			target(1, 1) = _rotationY.y;
			target(1, 2) = _rotationY.z;

			target(2, 0) = _rotationZ.x;
			target(2, 1) = _rotationZ.y;
			target(2, 2) = _rotationZ.z;

			target(3, 0) = _position.x;
			target(3, 1) = _position.y;
			target(3, 2) = _position.z;

			robotTargets.push_back(target);
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method set the robot target matrix.
		*
		*	\param [in]		target			- input target matrix.	
		*	\since version 0.0.2
		*/
		void setTarget(zTransform &target)
		{
			robot_target_matrix = target.transpose();
		}

		/*! \brief This method set the robot end effector matrix.
		*
		*	\param [in]		EE			- input endeffector matrix.
		*	\since version 0.0.2
		*/
		void setEndEffector(zTransform &EE)
		{
			zTransform temp = coreUtils.toLocalMatrix(EE);			
			robot_endEffector_matrix = temp.transpose();
		}

		//--------------------------
		//----KINMATICS METHODS
		//--------------------------

		/*! \brief This Methods computes the joint positionsof the zRobot.
		*
		*	\since version 0.0.2
		*/
		void computeJoints()
		{
			zTransform Tbase;
			Tbase.setIdentity();

			for (int i = 0; i < DOF; i++)
			{		 
				Tbase = Tbase * Bars[i].T;
				robotJointTransforms[i] = Tbase;
			}

			update_robotJointGraph();

			setJointMeshTransform(true);
			
		}

		/*! \brief This Methods computes the forward kinematics chain of zRobot for the specified joint rotation values.
		*
		*	\param		[in]	type		- robot joint rotation type.
		*	\since version 0.0.2
		*/
		zVector forwardKinematics(zRobotRotationType rotType = zJoint)
		{
			zVector out;

			if (rotType != zJoint)
			{
				for (int i = 0; i < DOF; i++)
				{
					if (rotType == zJointHome) jointRotations[i].rotation = jointRotations[i].home;
					if (rotType == zJointMinimum) jointRotations[i].rotation = jointRotations[i].minimum;
					if (rotType == zJointMaximum) jointRotations[i].rotation = jointRotations[i].maximum;
				}
			}

			for (int i = 0; i < DOF; i++)
			{
				if (isnan(jointRotations[i].rotation))
				{
					printf("\n joint %i roation isNAN.", i);
				}

				Bars[i].linkDH.theta = (jointRotations[i].rotation)  * DEG_TO_RAD;
			}

			for (int i = 0; i < DOF; i++)Bars[i].updateTransform();

			computeJoints();
			out = zVector(robotJointTransforms[DOF - 1](0, 3), robotJointTransforms[DOF - 1](1, 3), robotJointTransforms[DOF - 1](2, 3));

			return out;

		}

		/*! \brief This Methods computes the inverse kinematics chain of the robot .
		*
		*	\since version 0.0.2
		*/
		zVector inverseKinematics()
		{
			// compute target for joint 6
			zTransform Target_J6 = robot_target_matrix * robot_endEffector_matrix;

			// CALCULATE WRIST CENTER
			zVector wristC = zVector(Target_J6(0, 3), Target_J6(1, 3), Target_J6(2, 3));

			wristC -= (zVector(Target_J6(0, 2), Target_J6(1, 2), Target_J6(2, 2)) * Bars[5].linkDH.d * 1);

			// CALCULATE FIRST 3 ANGLES
			double th0 = atan2(wristC.y, wristC.x);

			// x and y of wc in joint 1's basis
			double r = sqrt(wristC.x * wristC.x + wristC.y * wristC.y) - Bars[0].linkDH.a;
			double s = Bars[0].linkDH.d - wristC.z;

			// relevant joint lengths
			double a1 = Bars[1].linkDH.a;
			double a2 = sqrt(Bars[2].linkDH.a * Bars[2].linkDH.a + Bars[3].linkDH.d * Bars[3].linkDH.d);

			// 3rd angle
			double d = (r * r + s * s - a1 * a1 - a2 * a2) / (2.0 * a1 * a2);
			double th2 = atan2(sqrt(1.0 - d * d), d); // negate for alt sol'n

			// 2nd angle
			double k1 = a1 + a2 * cos(th2);
			double k2 = a2 * sin(th2);
			double th1 = atan2(s, r) - atan2(k2, k1);


			jointRotations[0].rotation = th0 * RAD_TO_DEG;
			jointRotations[1].rotation = (th1)* RAD_TO_DEG;

			double dq2 = atan2(Bars[2].linkDH.a, Bars[3].linkDH.d);
			jointRotations[2].rotation = (((th2 + dq2) * RAD_TO_DEG) - 90 /*- home_rotations[2]*/) * 1;

			

			// SET FORWARD
			forwardKinematics();


			// SOLVE LAST 3 ANGLES

			zTransform r03 = Bars[0].T * Bars[1].T * Bars[2].T;		

			zTransform r36 = r03.transpose() * Target_J6;
			double t = r36(2, 2);

			double th3 = atan2(-r36(1, 2), -r36(0, 2));
			double th4 = atan2(1.0 * sqrt(1.0 - t * t), t);
			double th5 = atan2(-r36(2, 1), r36(2, 0));

			jointRotations[3].rotation = (th3 * RAD_TO_DEG);
			jointRotations[4].rotation = (th4 * RAD_TO_DEG) ;
			jointRotations[5].rotation = (th5 * RAD_TO_DEG);

			// SET FORWARD
			forwardKinematics();

			zVector out(robotJointTransforms[DOF - 1](0, 3), robotJointTransforms[DOF - 1](1, 3), robotJointTransforms[DOF - 1](2, 3));

			return out;
		}

		//--------------------------
		//----MESH METHODS
		//--------------------------
		
		/*! \brief This methods sets the joint mesh dihedral edge angles container.
		*
		*	\param		[in]	lightVec		- vector of light.
		*	\since version 0.0.2
		*/
		void setJointMeshDihedralEdges()
		{
			robot_jointMeshes_edgeAngle.clear();
			for (int i = 0; i < DOF + 2 /*fnMeshJoints.size()*/; i++)
			{

				if (fnMeshJoints[i].numVertices() == 0) continue;

				vector<double> dihedralAngles;

				fnMeshJoints[i].getEdgeDihedralAngles(dihedralAngles);

				robot_jointMeshes_edgeAngle.push_back(dihedralAngles);
			}
		}

		/*! \brief This methods sets the joint mesh occlusion face color .
		*
		*	\param		[in]	lightVec		- vector of light.
		*	\since version 0.0.2
		*/
		void setJointMeshColor(zVector lightVec)
		{
			for (int i = 0; i < DOF + 2 /*fnMeshJoints.size()*/; i++)
			{

				if (fnMeshJoints[i].numVertices() == 0) continue;

				fnMeshJoints[i].setFaceColorOcclusion(lightVec, true);
			}

			setJointMeshDihedralEdges();
		}

		/*! \brief This methods sets the joint mesh transformation matrix to the joint transform of the robot .
		*
		*	\since version 0.0.2
		*/
		void setJointMeshTransform(bool updatePositions = true)
		{
			
			for (int i = 0; i < DOF; i++)
			{
				robotMesh_transforms[i] = robotJointTransforms[i].transpose();			

				fnMeshJoints[i + 1].setTransform(robotMesh_transforms[i],false, updatePositions);				

				// update EE
				if( i == DOF -1)fnMeshJoints[i + 2].setTransform(robotMesh_transforms[i], false, updatePositions);
			}
		}


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
		void gCode_store(zVector &target_position, double velocity, zRobotMoveType moveType, zRobotEEControlType endEffectorControl)
		{
			zGCode inGCode;
			inGCode.vel = velocity;
			inGCode.moveType = moveType;
			inGCode.endEffectorControl = endEffectorControl;
			
			zTransform TCP = robotJointTransforms[DOF - 1] * robot_endEffector_matrix;

			inGCode.robotTCP_position = zVector(TCP(0, 3), TCP(1, 3), TCP(2, 3));
			inGCode.target_position = target_position;

			inGCode.distDifference = inGCode.robotTCP_position.distanceTo(inGCode.target_position);

			inGCode.targetReached = true;
			if (inGCode.distDifference > 0.1) inGCode.targetReached = false;

			for (int i = 0; i < DOF; i++)
			{
				//if (robot_gCode.size() > 0 && i == 3)
				//{
				//	int numGPoints = robot_gCode.size() - 1;
				//	bool prevRot = (robot_gCode[numGPoints].rotations[i].rotation >= 0) ? true : false;

				//	bool currentRot = (jointRotations[i].rotation >= 0) ? true : false;

				//	if (prevRot != currentRot)
				//	{
				//		if (prevRot) jointRotations[i].rotation += 360;
				//		else jointRotations[i].rotation -= 360;
				//	}
				//}

				inGCode.rotations.push_back(jointRotations[i]);

				if ((jointRotations[i].rotation + jointRotations[i].offset) < jointRotations[i].minimum || (jointRotations[i].rotation + jointRotations[i].offset) > jointRotations[i].maximum) inGCode.targetReached = false;
			}

			robot_gCode.push_back(inGCode);
		}

		/*! \brief This method exports the robot gcode to the input folder.
		*
		*	\param [in]		directoryPath			- input directory path.
		*	\param [in]		type					- robot type.
		*	\since version 0.0.2
		*/
		void gCode_to(string directoryPath, zRobotType type)
		{
			if (type == zRobotABB)
			{
				string filename = directoryPath;
				filename.append("/ABB_GCode.prg");
				toABBGcode(filename);
			}

			if (type == zRobotNachi)
			{
				string filename = directoryPath;
				filename.append("/MZ07-01-A.080");
				toNACHI_MZ07Gcode(filename);
			}
		}


	protected:

		/*! \brief This method create a graph connecting all the joint positions.
		*
		*	\since version 0.0.2
		*/
		void createRobotJointGraph()
		{
			vector<zVector> positions;
			vector<int> edgeConnects;

			positions.push_back(zVector());			
			positions.push_back(zVector(0,0, robotJointTransforms[0](2, 3)));

			for (int i = 0; i < DOF; i++)
			{
				zVector pos; 
				pos.x = robotJointTransforms[i](0, 3);
				pos.y = robotJointTransforms[i](1, 3);
				pos.z = robotJointTransforms[i](2, 3);
								

				positions.push_back(pos);
			}

			for (int i = 1; i < positions.size(); i++)
			{
				edgeConnects.push_back(i - 1);
				edgeConnects.push_back(i);
			}

			fnGraphJoint.create(positions, edgeConnects);

		}

		/*! \brief This method updates the joint graph based on the jint rotation matrices.
		*
		*	\since version 0.0.2
		*/
		void update_robotJointGraph()
		{
			zVector p(zVector(0, 0, robotJointTransforms[0](2, 3)));
			

			zItGraphVertex v(*jointGraphObj, 1);
			v.setVertexPosition(p);
			v.next();

			for (int i = 0; i < DOF; i++, v.next())
			{
				zVector pos;
				pos.x = robotJointTransforms[i](0, 3);
				pos.y = robotJointTransforms[i](1, 3);
				pos.z = robotJointTransforms[i](2, 3);

				v.setVertexPosition( pos);				
			}
		}		

		/*! \brief This method computes the tranformation to the world space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform toWorldMatrix(zTransform &inMatrix)
		{

			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
			zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
			zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
			zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));


			outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
			outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
			outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

			outMatrix(0, 3) = Cen.x; outMatrix(1, 3) = Cen.y; outMatrix(2, 3) = Cen.z;

			return outMatrix;
		}

		/*! \brief This method computes the tranformation to the local space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input 4X4 zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform toLocalMatrix(zTransform &inMatrix)
		{

			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
			zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
			zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
			zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));

			zVector orig(0, 0, 0);
			zVector d = Cen - orig;

			outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
			outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
			outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

			outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);



			return outMatrix;
		}

		/*! \brief This method computes the tranformation from one 4X4 matrix to another.
		*
		*	\param		[in]	from		- input 4X4 zMatrix.
		*	\param		[in]	to			- input 4X4 zMatrix.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform PlanetoPlane(zTransform &from, zTransform &to)
		{
			zTransform world = toWorldMatrix(to);
			zTransform local = toLocalMatrix(from);

			zTransform out = world * local;

			return out;
		}

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method imports robot from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromJSON(string infilename)
		{
			json j;
			zUtilsJsonRobot robotJSON;

			ifstream in_myfile;
			in_myfile.open(infilename.c_str());

			int lineCnt = 0;

			if (in_myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;
			}

			in_myfile >> j;
			in_myfile.close();


			// READ Data from JSON			
		
			robotJSON.scale = (j["scale"].get<double>());
			robot_scale = robotJSON.scale;

			//DH
			robotJSON.d = (j["d"].get<vector<double>>());

			robotJSON.a = (j["a"].get<vector<double>>());

			robotJSON.alpha = (j["alpha"].get<vector<double>>());

			robotJSON.theta = (j["theta"].get<vector<double>>());

			for (int i = 0; i < DOF; i++)
			{
				zDHparameter DH;

				DH.alpha = robotJSON.alpha[i] * DEG_TO_RAD;
				DH.d = robotJSON.d[i] * robot_scale;
				DH.theta = robotJSON.theta[i] * DEG_TO_RAD;
				DH.a = robotJSON.a[i] * robot_scale;				

				Bars.push_back(zLink(DH));

				robot_DH.push_back(DH);
			}

			// joint rotation

			robotJSON.jointRotations_home = (j["jointRotations_home"].get<vector<double>>());

			robotJSON.jointRotations_minimum = (j["jointRotations_minimum"].get<vector<double>>());

			robotJSON.jointRotations_maximum = (j["jointRotations_maximum"].get<vector<double>>());

			robotJSON.jointRotations_pulse = (j["jointRotations_pulse"].get<vector<double>>());

			robotJSON.jointRotations_mask = (j["jointRotations_mask"].get<vector<double>>());

			robotJSON.jointRotations_offset = (j["jointRotations_offset"].get<vector<double>>());


			for (int i = 0; i < DOF; i++)
			{
				zJointRotation jointRot;
				jointRot.home = robotJSON.jointRotations_home[i];
				jointRot.minimum = robotJSON.jointRotations_minimum[i];
				jointRot.maximum = robotJSON.jointRotations_maximum[i];
				jointRot.rotation = 0;
				jointRot.pulse = robotJSON.jointRotations_pulse[i];
				jointRot.mask = robotJSON.jointRotations_mask[i];
				jointRot.offset = robotJSON.jointRotations_offset[i];
				jointRotations.push_back(jointRot);


				robotJointTransforms.push_back(zTransform());
				robotMesh_transforms.push_back(zTransform());
			}

		}

		/*! \brief This method imports robot targets from a TXT file format .
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromTXT(string infilename)
		{
			robotTargets.clear();

			ifstream myfile;
			myfile.open(infilename.c_str());

			int lineCnt = 0;
			

			if (myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return ;				
			}

			int lay0 = 0;
			int lay1 = 0;
			int nPtsPerLayer = 0;;

			while (!myfile.eof()/* && lineCnt < 15*/)
			{
				string str;
				getline(myfile, str);

				vector<string> perlineData = coreUtils.splitString(str, ",");
					
			

				if (perlineData.size() == 12)
				{
					zTransform mat;
					mat.setIdentity();

					//x
					mat(0, 0) = atof(perlineData[0].c_str());
					mat(0, 1) = atof(perlineData[1].c_str());
					mat(0, 2) = atof(perlineData[2].c_str());

					////y
					mat(1, 0) = atof(perlineData[3].c_str());
					mat(1, 1) = atof(perlineData[4].c_str());
					mat(1, 2) = atof(perlineData[5].c_str());

					//z
					mat(2, 0) = atof(perlineData[6].c_str());
					mat(2, 1) = atof(perlineData[7].c_str());
					mat(2, 2) = atof(perlineData[8].c_str());
					

					//cen
					mat(3, 0) = atof(perlineData[9].c_str());
					mat(3, 1) = atof(perlineData[10].c_str());
					mat(3, 2) = atof(perlineData[11].c_str());					

					robotTargets.push_back(mat);
					
				}

				lineCnt++;
			}			
		}

		/*! \brief This method exports robot gcode for an ABB robot.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toABBGcode(string infilename)
		{
			printf(" ----------- writing \n ");

			ofstream myfile;
			myfile.open(infilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;
			}

			//myfile << " \n ! GCode for ABB ROBOT \n " << endl;
			//myfile << " \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n " << endl;

			// start MODULE
			myfile << "\n MODULE Module1 \n";

			// boolean variables
			myfile << "\n !Flag, end the program \n";
			myfile << " \n VAR bool bProgEnd; \n ";

			// constant variable for home position
			myfile << "\n !Constant for the joint calibrate position \n ";
			myfile << "\n CONST jointtarget calib_pos := [[0, 0, 0, 0, 0, 0], [0, 9E9, 9E9, 9E9, 9E9, 9E9]]; \n";

			// PROCEDURE Main
			myfile << "\n PROC Main() \n ";
			myfile << "\n Init; \n ";

			myfile << "\n mv_Calib; \n ";

			// write custom gcode
			myfile << "\n mv_Custom; \n ";
			// complete custom gcode

			myfile << "\n mv_Calib; \n ";
			myfile << "\n ENDPROC \n ";

			// PROCEDURE init
			myfile << "\n PROC Init() \n ";

			// define varaiale initial value if any
			myfile << "\n !Defined setting of the variables \n";
			myfile << "\n bProgEnd := FALSE; \n ";

			myfile << "\n ENDPROC \n ";

			// PROCEDURE mv_Calib
			myfile << "\n PROC mv_Calib() \n ";

			myfile << "\n MoveAbsJ calib_pos,v100,z10,tool0; \n ";

			myfile << "\n ENDPROC \n ";

			// PROCEDURE mv_Custom
			myfile << "\n PROC mv_Custom() \n ";
			myfile << "\n  CONST num count := " << robot_gCode.size() << ";";

			myfile << "\n  CONST jointtarget poses {count} := ";
			myfile << "\n [ ";

			for (int i = 0; i < robot_gCode.size(); i++)
			{

				myfile << "\n  [[  ";

				for (int j = 0; j < DOF; j++)
				{
					myfile << to_string(robot_gCode[i].rotations[j].rotation + robot_gCode[i].rotations[j].offset);
					if (j < DOF - 1) myfile << ", ";
				}

				myfile << "], [0, 9E9, 9E9, 9E9, 9E9, 9E9]] ";

				if (i != robot_gCode.size() - 1) myfile << ",";
			}

			myfile << "\n ]; ";

			myfile << "\n FOR i FROM 1 TO count DO ";
			myfile << "\n MoveAbsJ poses{i}, v100, z10, tool0;";
			myfile << "\n ENDFOR";

			/*for (int i = 0; i < robot_gCode.size(); i++)
			{
				myfile << "\n";

				if(robot_gCode[i].moveType == zRobot_Move_joint) myfile << "MoveAbsJ ";
				if(robot_gCode[i].moveType == zRobot_Move_linear) myfile << "MoveL ";

				myfile<< "p" << i << ", v" << to_string((int)robot_gCode[i].vel) << ", z10, tool0; \n";
			}*/

			myfile << "\n ENDPROC \n ";

			//End MODULE
			myfile << "\n ENDMODULE \n";

			//close file
			myfile.close();
		}

		/*! \brief This method exports robot gcode for a NACHI MZ07 robot.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toNACHI_MZ07Gcode(string infilename)
		{
			printf("\n----------- writing \n ");

			ofstream myfile;
			myfile.open(infilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;
			}

			//for (int i = 0; i < robot_gCode.size(); i++)
			//{
			//	if (!robot_gCode[i].targetReached)
			//	{
			//		cout << " some or all points out of range" << endl;

			//		cout << "--------------------------- EXPORT GCODE ----------------- " << endl;
			//		cout << "Ensure you have inspected robot reach at all points previously " << endl;
			//		cout << "un-reachable points revert to previous reach-able points" << endl;

			//		return;
			//	}
			//}

			for (int i = 0; i < robot_gCode.size(); i++)
			{
				myfile << "MOVEX A=6, AC=0, SM=0, M1J, P,";

				myfile << "(";

				for (int j = 0; j < DOF; j++)
				{
					myfile << to_string((robot_gCode[i].rotations[j].rotation + robot_gCode[i].rotations[j].offset) * robot_gCode[i].rotations[j].mask);

					if (j < DOF - 1) myfile << ",";
				}

				myfile << "),";

				myfile << "S=" << to_string(robot_gCode[i].vel) << ",";

				myfile << "H=3, MS, CONF=0001" << endl;
			}

			//close file
			myfile.close();

			printf("\nG-CODE Exported Succesfully\n ");
		}
	};

}