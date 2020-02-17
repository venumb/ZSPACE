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

#ifndef ZSPACE_TS_PATHNETWORK_SLIME_H
#define ZSPACE_TS_PATHNETWORK_SLIME_H

#pragma once

#include <headers/zInterface/functionsets/zFnPointCloud.h>
#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnMeshField.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zInterface/iterators/zItMeshField.h>

namespace zSpace
{
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications. 
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/** \addtogroup zTsSlimeMould
	*	\brief The slime mould related tool sets to create the simulation.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zSlimeAgent
	*	\brief A slime agent class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zSlimeAgent
	{

	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief particle object  */
		zObjParticle particlesObj;

		/*!<Sensor offset of the agent.*/
		double *SO;

		/*!<Sensor Angle of the agent.*/
		double *SA;

		/*!<Agent rotation angle.*/
		double *RA;

		/*!<deposition per step.*/
		double *depT;

		/*!<probability of random change in direction.*/
		double *pCD;

		/*!<sensitivity threshold.*/
		double *sMin;

	public:

		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------
		zFnParticle fnParticle; 
				
		
		/*!<trail color of the agent.*/
		zColor tCol;

		/*!<trail weight of the agent.*/
		double tWeight;			
		
		/*!<trail maximum points of the agent.*/
		int tMax;	

		/*!<trail points counter of the agent.*/
		int tCounter;

		/*!<stores trail positions of agent.*/
		vector<zVector> trail;

		/*!<stores trail rotation matrix of agent.*/
		Matrix3d tMat; 

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zSlimeAgent();		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zSlimeAgent();

		//--------------------------
		//---- CREATE METHOD
		//--------------------------

		/*! \brief This method creates the agent with the input parameters.
		*
		*	\param		[in]	_pos				- start positions of the agent.
		*	\param		[in]	_SO					- sensor offset.
		*	\param		[in]	_SA					- sensor angle.
		*	\param		[in]	_RA					- rotation angle.
		*	\param		[in]	_depT				- deposition rate.
		*	\param		[in]	_pCD				- probability of random change in direction.
		*	\param		[in]	_sMin				- sensitivity threshold.
		*	\since version 0.0.1
		*/
		void create(zVector &_pos, double &_SO, double &_SA, double &_RA, double &_depT, double &_pCD, double &_sMin);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method returns the forward direction for the agent.
		*
		*	\return 	zVector				- vector of forward direction.
		*	\since version 0.0.1
		*/
		zVector getF();

		/*! \brief This method returns the forward right direction for the agent.
		*
		*	\return 	zVector				- vector of forward right direction.
		*	\since version 0.0.1
		*/
		zVector getFR();

		/*! \brief This method returns the forward left direction for the agent.
		*
		*	\return 	zVector				- vector of forward left direction.
		*	\since version 0.0.1
		*/
		zVector getFL();

		/*! \brief This method returns the sensor offset value for the agent.
		*
		*	\return 	double				- sensor offset value .
		*	\since version 0.0.1
		*/
		double getSO();

		/*! \brief This method returns the sensor angle value for the agent.
		*
		*	\return 	double				- sensor angle value .
		*	\since version 0.0.1
		*/
		double getSA();

		/*! \brief This method returns the rotation angle value for the agent.
		*
		*	\return 	double				- rotation angle value .
		*	\since version 0.0.1
		*/
		double getRA();

		/*! \brief This method returns the sensititvity threshold value for the agent.
		*
		*	\return 	double				- sensititvity threshold value .
		*	\since version 0.0.1
		*/
		double getSMin();

		/*! \brief This method returns the deposition per step value for the agent.
		*
		*	\return 	double				- deposition per step value.
		*	\since version 0.0.1
		*/
		double getDepT();

		//--------------------------
		//---- SET METHODS
		//--------------------------
			
		/*! \brief This method returns the direction for the agent based on input values of F, Fr and FL.
		*
		*	\param		[in]	a_F					- value of chemA in forward direction.
		*	\param		[in]	a_FR				- value of chemA in forward right direction.
		*	\param		[in]	a_FL				- value of chemA in forward left direction.
		*	\param		[in]	chemoRepulsive		- the agents will repel from the chemical stimulant if true.
		*	\return 	zVector						- vector of direction.
		*	\since version 0.0.1
		*/
		void setVelocity(double a_F, double a_FR, double a_FL, bool chemoRepulsive = false);

		/*! \brief This method sets the sensor offset value for the agent.
		*
		*	\param		[in]	_SO				- sensor offset value .
		*	\since version 0.0.2
		*/
		void setSO(double &_SO);

		/*! \brief This method sets the sensor angle value for the agent.
		*
		*	\param		[in]	_SA			- sensor angle value .
		*	\since version 0.0.2
		*/
		void setSA(double &_SA);

		/*! \brief This method sets the rotation angle value for the agent.
		*
		*	\param		[in]	_RA				- rotation angle value .
		*	\since version 0.0.2
		*/
		void setRA(double &_RA);

		/*! \brief This method sets the deposition per step value for the agent.
		*
		*	\param		[in]	_depT				-  deposition per step value .
		*	\since version 0.0.2
		*/
		void setDepT(double &_depT);

		/*! \brief This method sets the sensitivity threshold value for the agent.
		*
		*	\param		[in]	_sMin				-  sensitivity threshold value .
		*	\since version 0.0.2
		*/
		void setSMin(double &_sMin);

	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/** \addtogroup zTsSlimeMould
	*	\brief  The slime mould related tool sets to create the simulation.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zSlimeEnvironment
	*	\brief A slime environment class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zSlimeEnvironment : public zFnMeshField<zScalar>
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;		

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------
						

		/*!<field resolution in X.*/
		int resX;	
		
		/*!<field resolution in Y.*/
		int resY;
		
		/*!<pixel size.*/
		double pix;				

	
		/*!<chemical A in the field. The size of the array should match fieldresoultion.*/
		vector<float> chemA;	

		/*!<true if cell is occupied else false. The size of the array should match fieldresoultion.*/
		vector< bool> occupied;		

		/*!<indicates food source indices in environment.*/
		vector<bool> bAttractants;

		/*!<indicates food source indices in environment.*/
		vector<bool> bRepellants;	

		/*!<mininmum and maximum value of A.*/
		double minA, maxA;	

		/*!<Environment cellID of mininmum and maximum value of A.*/
		int id_minA, id_maxA;
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zSlimeEnvironment();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\since version 0.0.2
		*/
		zSlimeEnvironment(zObjMeshField<zScalar> &_fieldObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zSlimeEnvironment();

		//--------------------------
		//---- GET SET METHODS
		//--------------------------
		
		/*! \brief This method return the value of chem A at the input position.
		*
		*	\param		[in]	pos		- input position .
		*	\return				double	- value of ChemA.
		*	\since version 0.0.1
		*/	
		double getChemAatPosition(zVector &pos);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method diffuses chemA in the environment.
		*
		*	\since version 0.0.1
		*/	
		void diffuseEnvironment(double decayT, double diffuseDamp, zDiffusionType diffType = zAverage);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method computes the minimum and maximum value of chemcial A in the environment.
		*
		*	\param		[in]	min		- input minimum percentile. Needs be between 0 and 1 .
		*	\param		[in]	max		- input maximum percentile. Needs be between 0 and 1 .
		*	\since version 0.0.1
		*/	
		void minMax_chemA(double min = 0, double max = 1);
		
	};


	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsPathNetworks
	*	\brief tool sets for path network optimization.
	*  @{
	*/

	/** \addtogroup zTsSlimeMould
	*	\brief  The slime mould related function sets to create the simulation.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zTsSlime
	*	\brief A slime mould simulation tool set.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsSlime
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to point cloud object  */
		zObjPointCloud *pointsObj;

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!<slime environment .*/
		zSlimeEnvironment environment;	
		
		/*!<slime agents.*/
		vector<zSlimeAgent>  agents;	

		/*!<stores point cloud function set for the slime agent positions.*/
		zFnPointCloud fnPositions;
		
		/*!<indicates attractant food source indices in environment.*/
		vector<int> attractants;

		/*!<indicates repellant food source indices in environment.*/
		vector<int> repellants;			

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/					
		zTsSlime();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input mesh field object.
		*	\param		[in]	_pointsObj			- input pointcloud object.
		*	\since version 0.0.2
		*/
		zTsSlime(zObjMeshField<zScalar> &_fieldObj, zObjPointCloud &_pointsObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------
		
		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zTsSlime();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the environment form the input parameters.
		*
		*	\param		[in]	_minBB		- bounding box minimum.
		*	\param		[in]	_maxBB		- bounding box maximum.
		*	\param		[in]	_resX		- resolution X of environment.
		*	\param		[in]	_resY		- resolution Y of environment.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.2
		*/
		void createEnvironment(zVector _minBB, zVector _maxBB, int _resX, int _resY, int _NR = 1);

		/*! \brief This methods creates the environment form the input parameters.
		*	\param		[in]	_pix		- pixel size of environment.
		*	\param		[in]	_res		- resolution of environment.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.1
		*/
		void createEnvironment(double _pix, int _res, zVector _minBB = zVector(), int _NR = 1);


		/*! \brief This methods creates the agents with the input parameters.
		*	\param		[in]	_p			- population percentage between 0 and 1.
		*	\param		[in]	_SO			- agent sensor offset.
		*	\param		[in]	_SA			- agent sensor angle.
		*	\param		[in]	_RA			- agent rotation angle.
		*	\param		[in]	_depT		- agent deposition per step value.
		*	\param		[in]	_pCD		- agent probability of random change in direction value.
		*	\param		[in]	_sMin		- agent sensitivity threshold value.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.1
		*/
		void createAgents(double _p, double &_SO, double &_SA, double &_RA, double &_depT, double &_pCD, double &_sMin);

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method defines the motor stage of the simulation.
		*
		*	\param		[in]	dT					- time step.
		*	\param		[in]	type				- integration type.
		*	\param		[in]	agentTrail			- stores the agent trail if true.
		*	\since version 0.0.1
		*/
		void slime_Motor(double dT, zIntergrationType integrateType, bool agentTrail);

		/*! \brief This method defines the motor stage of the simulation.
		*
		*	\param		[in]	chemoRepulsive		- the agents will repel from the chemical stimulant if true.
		*	\since version 0.0.1
		*/
		void slime_Sensor(bool chemoRepulsive = false);

		/*! \brief This method contains the agent with in the bounds of the environment.
		*
		*	\param		[in]	index		- input agent index .
		*	\param		[in]	dT			- time step.
		*	\param		[in]	type		- integration type.
		*	\since version 0.0.1
		*/
		void containBounds(int  index, double dT, zIntergrationType integrateType);

		/*! \brief This method adds a food source at the input index of the environment.
		*
		*	\param		[in]	id			- environment ID to be made a fod source.
		*	\param		[in]	depA		- deposition rate of chem A at food source .
		*	\param		[in]	attractant	- type of food source. True if attractant , else false.
		*	\return				bool		- true if the food source is created.
		*	\since version 0.0.1
		*/
		bool  addFoodSource(int id, double depA, int nR, bool attractant);

		/*! \brief This method adds a repellant food source at boundary cells of the environment.
		*
		*	\param		[in]	depA				- deposition rate of chem A at food source .
		*	\param		[in]	distFromBoundary	- number of cells from the boundary to made a food source.
		*	\since version 0.0.1
		*/
		void makeBoundaryRepellant(double depA, int distFromBoundary);

		/*! \brief This method deposits chemical A at the input environment Id.
		*
		*	\param		[in]	id		- environment ID.
		*	\param		[in]	depA	- deposition rate of chem A.
		*	\since version 0.0.1
		*/
		void depositChemicalA(int id, double depA);

		/*! \brief This method deposits chemical A at the input food source.
		*
		*	\param		[in]	depA	- deposition rate of chem A.
		*	\param		[in]	wProj	- prepattern stimuli projection weight.
		*	\since version 0.0.1
		*/
		void depositAtFoodSource(double depA, double wProj);

		/*! \brief This method computes a random unoccupied cell in the environment.
		*
		*	\param		[in]	boundaryOffset		- excludes the rows of input boundary offset.
		*	\since version 0.0.1
		*/
		int randomUnoccupiedCell(int boundaryOffset = 3);

		/*! \brief This method clears the food sources. 
		*
		*	\param		[in]	Attractants		- clears all attractants if true.
		*	\param		[in]	Repellants		- clears all repellants if true.
		*	\since version 0.0.1
		*/
		void clearFoodSources(bool Attractants = true, bool Repellants = true);

		/*! \brief This method removes agents randomly form the simulation if the total number of agents is higher than input minimum.
		*
		*	\param		[in]	minAgents		- minimum number of agents.
		*	\since version 0.0.1
		*/
		void randomRemoveAgents(int minAgents);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------
			   

		/*! \brief This method computes the color value of each cell in the environment based on chemical A or agent occupied cells.
		*
		*	\param		[in]	dispAgents		- display cells occupied by agents if true, else displays environment chemical.
		*	\param		[in]	usePercentile	- color cells using the percentile value. Works only with chemical A based coloring.
		*	\since version 0.0.1
		*/
		void computeEnvironmentColor(bool dispAgents = false, bool usePercentile = false);

		/*! \brief This method computes the agent parameters based on a input scalar field.
		*
		*	\param		[in]	agentId			- input agent ID.
		*	\param		[in]	fieldScalars	- input container of scalars.
		*	\param		[in]	parameterValue	- input parameter value.
		*	\param		[in]	minVal			- minimum value of parameter.
		*	\param		[in]	maxVal			- maximum value of parameter.
		*	\param		[in]	parm			- zSlimeParameter type ( zSO / zSA / zRA /zdepT).
		*	\since version 0.0.1
		*/
		void updateAgentParametersFromField(int agentId, vector<double> &fieldScalars, double &parameterValue, double minVal, double maxVal, zSlimeParameter parm);
				
	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/pathNetworks/zTsSlimeMould.cpp>
#endif

#endif