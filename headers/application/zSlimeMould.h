#pragma once


#include <headers/core/zUtilities.h>
#include <headers/core/zColor.h>

#include <headers/geometry/zMesh.h>

#include <headers/geometry/zField.h>
#include <headers/geometry/zFieldUtilities.h>

#include <headers/dynamics/zParticle.h>

#include <headers/IO/zExchange.h>

namespace zSpace
{
	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zSlimeMould
	*	\brief The slime mould simulation related classes of the library.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zSlimeAgent
	*	\brief A slime agent class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	class zSlimeAgent
	{
	public:

		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------
		zParticle particle; 
	
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
		zSlimeAgent()
		{

			particle = zParticle();
		
			double randX = randomNumber_double(-1, 1);
			double randY = randomNumber_double(-1, 1);

			if (randX == randY && randY == 0)
				randX = (randomNumber_double(0, 1) > 0.5)? 1 :  - 1;

			zVector velocity = zVector(randX, randY, 0);
			particle.setVelocity(velocity);

			tCol = zColor(0, 0, 0, 1);
			tWeight = 1;

			tCounter = 0;
			tMax = 1000;

			tMat.setIdentity();

			SO = nullptr;
			SA = nullptr;
			RA = nullptr;

			depT = nullptr;
			pCD = nullptr;
			sMin = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zSlimeAgent(){}

		
		//--------------------------
		//---- GET SET METHODS
		//--------------------------

		/*! \brief This method returns the forward direction for the agent.
		*
		*	\return 	zVector				- vector of forward direction.
		*	\since version 0.0.1
		*/
		zVector getF()
		{
				
			particle.getVelocity().normalize();
			zVector F = particle.getVelocity() * (*SO);

			return (*particle.getPosition() + F);			
		}

		/*! \brief This method returns the forward right direction for the agent.
		*
		*	\return 	zVector				- vector of forward right direction.
		*	\since version 0.0.1
		*/
		zVector getFR()
		{
			zVector axis(0, 0, 1);

			particle.getVelocity().normalize();
			zVector velocity = particle.getVelocity() * (*SO);
			

			zVector FR = velocity.rotateAboutAxis(axis, -(*SA));
		
			return (*particle.getPosition() + FR);
		}

		/*! \brief This method returns the forward left direction for the agent.
		*
		*	\return 	zVector				- vector of forward left direction.
		*	\since version 0.0.1
		*/
		zVector getFL()
		{
			zVector axis(0, 0, 1);

			particle.getVelocity().normalize();
			zVector velocity = particle.getVelocity() * (*SO);	

			zVector FL = velocity.rotateAboutAxis(axis, (*SA));
	
			return  (*particle.getPosition() + FL);
		}

		/*! \brief This method returns the direction for the agent based on input values of F, Fr and FL.
		*
		*	\param		[in]	a_F					- value of chemA in forward direction.
		*	\param		[in]	a_FR				- value of chemA in forward right direction.
		*	\param		[in]	a_FL				- value of chemA in forward left direction.
		*	\param		[in]	chemoRepulsive		- the agents will repel from the chemical stimulant if true.
		*	\return 	zVector						- vector of direction.
		*	\since version 0.0.1
		*/
		void setVelocity(double a_F, double a_FR, double a_FL, bool chemoRepulsive = false)
		{
			zVector out;
			zVector axis(0, 0, 1);


			if (a_F > a_FL &&  a_F > a_FR)
			{
				out = particle.getVelocity();
			}
			else if (a_F < a_FL &&  a_F < a_FR)
			{
				double rand = randomNumber_double(0, 1);
				float ang = (rand > 0.5) ? (-(*RA)) : ((*RA));

				out = particle.getVelocity().rotateAboutAxis(axis, ang);
			}
			else if (a_FL < a_FR)
			{
				out = particle.getVelocity().rotateAboutAxis(axis, -(*RA));
			}
			else if (a_FR < a_FL)
			{
				out = particle.getVelocity().rotateAboutAxis(axis, (*RA));
			}
			else
			{
				out = particle.getVelocity();
			}

			out.normalize();

			if (chemoRepulsive) out *= -1;
			particle.setVelocity(out);		

			//printf("\n inside  %1.2f %1.2f %1.2f ", particle.getVelocity().x, particle.getVelocity().y, particle.getVelocity().z);
		}


	};

	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zSlimeMould
	*	\brief The slime mould simulation related classes of the library.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zSlimeEnvironment
	*	\brief A slime environment class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zSlimeEnvironment
	{
	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------
		
		/*!<scalar field.*/
		zField2D<double> field;
		
		/*!<field mesh.*/
		zMesh fieldMesh;

		/*!<field resolution in X.*/
		int resX;	
		
		/*!<field resolution in Y.*/
		int resY;
		
		/*!<pixel size.*/
		double pix;				

	
		/*!<chemical A in the field. The size of the array should match fieldresoultion.*/
		vector<double> chemA;	

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
		*	\since version 0.0.1
		*/
		zSlimeEnvironment()
		{
			resX = resY = 200;
			pix = 1.0;

			chemA.clear();
			occupied.clear();
			bAttractants.clear();
			bRepellants.clear();

			for (int i = 0; i < resX*resY; i++)
			{
				chemA.push_back(0);
				occupied.push_back(false);
				bAttractants.push_back(false);
				bRepellants.push_back(false);
			}

			field = zField2D<double>(pix, pix, resX, resY);
			from2DFIELD(fieldMesh, field);

			minMax_chemA();
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_res		- resolution of environment.
		*	\param		[in]	_pix		- pixel size of environment.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.1
		*/
		zSlimeEnvironment(int _res, double _pix, int _NR = 1)
		{
			resX = resY = _res;
			pix = _pix;

			field = zField2D<double>(pix, pix, resX, resY, _NR);
			from2DFIELD(fieldMesh, field);

			minMax_chemA();

			chemA.clear();
			occupied.clear();
			bAttractants.clear();
			bRepellants.clear();

			for (int i = 0; i < resX*resY; i++)
			{
				chemA.push_back(0);
				occupied.push_back(false);
				bAttractants.push_back(false);
				bRepellants.push_back(false);
			}

			minMax_chemA();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_res		- resolution of environment.
		*	\param		[in]	_minBB		- bounding box minimum.
		*	\param		[in]	_maxBB		- bounding box maximum.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.1
		*/
		zSlimeEnvironment(int _resX, int _resY, zVector _minBB, zVector _maxBB, int _NR = 1)
		{
			resX = _resX;
			resY = _resY;

			field = zField2D<double>(_minBB, _maxBB, _resX, _resY, _NR);
			from2DFIELD(fieldMesh, field);

			chemA.clear();
			occupied.clear();
			bAttractants.clear();
			bRepellants.clear();

			for (int i = 0; i < _resX*_resY; i++)
			{
				chemA.push_back(0);
				occupied.push_back(false);
				bAttractants.push_back(false);
				bRepellants.push_back(false);
			}

			minMax_chemA();
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zSlimeEnvironment(){}

		//--------------------------
		//---- GET SET METHODS
		//--------------------------
		
		/*! \brief This method return the value of chem A at the input position.
		*
		*	\param		[in]	pos		- input position .
		*	\return				double	- value of ChemA.
		*	\since version 0.0.1
		*/	
		double getChemAatPosition(zVector &pos)
		{
			int id;
			bool check = field.getIndex(pos,id);
	
			if(check) 	return chemA[id];
			else return -1.0;
		}

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method diffuses chemA in the environment.
		*
		*	\since version 0.0.1
		*/	
		void diffuseEnvironment(double decayT, double diffuseDamp, zDiffusionType diffType = zAverage)
		{
			vector<double> temp_chemA;

			for (int i = 0; i < field.fieldValues.size(); i++)
			{
				double lapA = 0;

				for (int j = 0; j < field.ringNeighbours[i].size(); j++)
				{
					int id = field.ringNeighbours[i][j];
								

					if (diffType == zLaplacian)
					{
						if (id != i) lapA += (chemA[id] * 1);
						else lapA += (chemA[id] * -8);
					}
					else if (diffType == zAverage)
					{
						lapA += (chemA[id] * 1);
					}
				}

				if (diffType == zLaplacian)
				{
					double newA = chemA[i] + (lapA * diffuseDamp);
					temp_chemA.push_back(newA);
				}
				else if (diffType == zAverage)
				{
					if (lapA != 0) lapA /= (field.ringNeighbours[i].size());
					temp_chemA.push_back(lapA);
				}
			}

			for (int i = 0; i < field.fieldValues.size(); i++)
			{
				chemA[i] = (1 - decayT) *temp_chemA[i];				
			}
		}

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method computes the minimum and maximum value of chemcial A in the environment.
		*
		*	\param		[in]	min		- input minimum percentile. Needs be between 0 and 1 .
		*	\param		[in]	max		- input maximum percentile. Needs be between 0 and 1 .
		*	\since version 0.0.1
		*/	
		void minMax_chemA(double min = 0, double max = 1)
		{
			minA = 10000;
			maxA = -10000;

			id_minA = -1;
			id_maxA = -1;

			vector<double> data = chemA;
			std::sort(data.begin(), data.end());

			int max_percentile = floor(max *  chemA.size());
			maxA = data[max_percentile - 1];
			id_maxA = max_percentile - 1;

			int min_percentile = floor(min *  chemA.size());
			minA = data[min_percentile];
			id_minA = min_percentile;
		}
		
	};

	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zSlimeMould
	*	\brief The slime mould simulation related classes of the library.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zSlime
	*	\brief A slime mould class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zSlime
	{
	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!<slime environment .*/
		zSlimeEnvironment environment;	
		
		/*!<slime agents.*/
		vector<zSlimeAgent>  agents;	

		/*!<stores slime agent positions.*/
		vector<zVector> agentPositions;
		
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
		zSlime()
		{
			environment = zSlimeEnvironment();
			agents.clear();
			attractants.clear();
			repellants.clear();
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------
		
		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zSlime() {}

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method set the agents positions as particles. To be used after all the agent positions are initialised.
		*
		*	\since version 0.0.1
		*/
		void setAgentParticles()
		{
			for (int i = 0; i < agentPositions.size(); i++)
			{
				double randX = randomNumber_double(-1, 1);
				double randY = randomNumber_double(-1, 1);

				if (randX == randY && randY == 0)
					randX = (randomNumber_double(0, 1) > 0.5) ? 1 : -1;

				zVector velocity = zVector(randX, randY, 0);
				agents[i].particle = zParticle(&agentPositions[i], false, 1.0, velocity);
			}
		}

		/*! \brief This method defines the motor stage of the simulation.
		*
		*	\param		[in]	dT					- time step.
		*	\param		[in]	type				- integration type.
		*	\param		[in]	agentTrail			- stores the agent trail if true.
		*	\since version 0.0.1
		*/
		void slime_Motor(double dT, zIntergrationType integrateType, bool agentTrail)
		{
		
			for (int i = 0; i < environment.resX * environment.resY; i++)
			{
				environment.occupied[i] = false;
			}		
			
			for (int i = 0; i < agents.size(); i++)
			{

				if (!agents[i].particle.fixed)
				{

					containBounds(agents[i], dT, integrateType);

					// get new position
					zVector nPos = agents[i].particle.getUpdatePosition(dT, integrateType);
					
					int fieldID = -1;;
					bool check = environment.field.getIndex(nPos, fieldID);

					if (check)
					{
						if (!environment.occupied[fieldID] && environment.chemA[fieldID] >= 0)
						{
							//trail
							if (agentTrail)
							{
								if (agents[i].trail.size() == agents[i].tMax)
								{
									int aId = agents[i].tCounter % agents[i].tMax;

									agents[i].trail[aId] = *agents[i].particle.getPosition();
									agents[i].tCounter++;
								}
								else
								{
									agents[i].trail.push_back(*agents[i].particle.getPosition());
									agents[i].tCounter = agents[i].trail.size();
								}
							}

							
							agents[i].particle.updateParticle(true,false,true);							

							depositChemicalA(fieldID, *(agents[i].depT));

							environment.occupied[fieldID] = true;
						}
						else
						{
							double randX = randomNumber_double(-1, 1);
							double randY = randomNumber_double(-1, 1);
							zVector velocity = zVector(randX, randY, 0);
							agents[i].particle.setVelocity(velocity);
						}
					}				
					else
					{
						double randX = randomNumber_double(-1, 1);
						double randY = randomNumber_double(-1, 1);
						zVector velocity = zVector(randX, randY, 0);
						agents[i].particle.setVelocity(velocity);
					}	
				}
			}
		}

		/*! \brief This method defines the motor stage of the simulation.
		*
		*	\param		[in]	chemoRepulsive		- the agents will repel from the chemical stimulant if true.
		*	\since version 0.0.1
		*/
		void slime_Sensor(bool chemoRepulsive = false)
		{
			for (int i = 0; i < agents.size(); i++)
			{
				if (!agents[i].particle.fixed)
				{
					zVector F = agents[i].getF();
					double a_F = environment.getChemAatPosition(F);

					

					zVector FR = agents[i].getFR();
					double a_FR = environment.getChemAatPosition(FR);

					zVector FL = agents[i].getFL();
					double a_FL = environment.getChemAatPosition(FL);

					agents[i].setVelocity(a_F, a_FR, a_FL, chemoRepulsive);


					
				}
			}
		}

		/*! \brief This method contains the agent with in the bounds of the environment.
		*
		*	\param		[in]	agent		- input zSlimeAgent .
		*	\param		[in]	dT			- time step.
		*	\param		[in]	type		- integration type.
		*	\since version 0.0.1
		*/
		void containBounds(zSlimeAgent & agent, double dT, zIntergrationType integrateType)
		{
			zVector nPos = agent.particle.getUpdatePosition(dT, integrateType);
			int id_new;
			bool check = environment.field.getIndex(*agent.particle.getPosition(), id_new);

			if (check)
			{
				if (environment.chemA[id_new] < 0)
				{
					zVector f = agent.particle.getForce() * -1;
					agent.particle.setForce(f);
				}
			}
		
		}

		/*! \brief This method adds a food source at the input index of the environment.
		*
		*	\param		[in]	id			- environment ID to be made a fod source.
		*	\param		[in]	depA		- deposition rate of chem A at food source .
		*	\param		[in]	attractant	- type of food source. True if attractant , else false.
		*	\return				bool		- true if the food source is created.
		*	\since version 0.0.1
		*/
		bool  addFoodSource(int id, double depA, int nR, bool attractant)
		{
			bool out = false;

			if (id < environment.bAttractants.size()) {

				vector<int> neighbourRing;
				environment.field.getNeighbourhoodRing(id, nR, neighbourRing);

				for (int j = 0; j < neighbourRing.size(); j++)
				{
					if (attractant)
					{
						attractants.push_back(neighbourRing[j]);
						environment.bAttractants[neighbourRing[j]] = true;
						depositChemicalA(neighbourRing[j], depA);
					}
					else
					{
						repellants.push_back(neighbourRing[j]);
						environment.bRepellants[neighbourRing[j]] = true;
						depositChemicalA(neighbourRing[j], -depA);
					}

					environment.occupied[neighbourRing[j]] = true;
				}

				out = true;
			}

			return out;
		}

		/*! \brief This method adds a repellant food source at boundary cells of the environment.
		*
		*	\param		[in]	depA				- deposition rate of chem A at food source .
		*	\param		[in]	distFromBoundary	- number of cells from the boundary to made a food source.
		*	\since version 0.0.1
		*/
		void makeBoundaryRepellant(double depA, int distFromBoundary)
		{
			for (int i = 0; i < environment.resX; i++)
			{
				for (int j = 0; j < environment.resY; j++)
				{

					bool repellant = false;

					if (i <= distFromBoundary || i >= environment.resX - distFromBoundary) repellant = true;

					if (j <= distFromBoundary || j >= environment.resY - distFromBoundary) repellant = true;



					if (repellant)
					{
						int id = (i * environment.resY) + j;
						addFoodSource(id, depA, 1, false);
					}
				}
			}
		}

		/*! \brief This method deposits chemical A at the input environment Id.
		*
		*	\param		[in]	id		- environment ID.
		*	\param		[in]	depA	- deposition rate of chem A.
		*	\since version 0.0.1
		*/
		void depositChemicalA(int id, double depA)
		{
			environment.chemA[id] += depA;
		}

		/*! \brief This method deposits chemical A at the input food source.
		*
		*	\param		[in]	depA	- deposition rate of chem A.
		*	\param		[in]	wProj	- prepattern stimuli projection weight.
		*	\since version 0.0.1
		*/
		void depositAtFoodSource(double depA, double wProj)
		{
			for (int i = 0; i < attractants.size(); i++)
			{
				double depT = (1 + wProj) * depA;

				depositChemicalA(attractants[i], (depT));
			}

			for (int i = 0; i < repellants.size(); i++)
			{
				double depT = (1 + wProj) * depA;

				depositChemicalA(repellants[i], (-1 * depT));
			}
		}

		/*! \brief This method computes a random unoccupied cell in the environment.
		*
		*	\param		[in]	boundaryOffset		- excludes the rows of input boundary offset.
		*	\since version 0.0.1
		*/
		int randomUnoccupiedCell(int boundaryOffset = 3)
		{
			bool exit = false;
			int outId = -1;

			while (!exit)
			{
				int rndX = randomNumber(boundaryOffset, environment.resX - boundaryOffset);
				int rndY = randomNumber(boundaryOffset, environment.resY - boundaryOffset);

				int id = (rndX * environment.resY) + rndY;
				if (!environment.occupied[id])
				{
					exit = true;
					outId = id;
				}
			}

			return outId;
		}

		/*! \brief This method clears the food sources. 
		*
		*	\param		[in]	Attractants		- clears all attractants if true.
		*	\param		[in]	Repellants		- clears all repellants if true.
		*	\since version 0.0.1
		*/
		void clearFoodSources(bool Attractants = true, bool Repellants = true)
		{
			if (Attractants)
			{
				for (int i = 0; i < attractants.size(); i++)
				{
					environment.bAttractants[attractants[i]] = false;
				}
				attractants.clear();
			}

			if (Repellants)
			{
				for (int i = 0; i < repellants.size(); i++)
				{
					environment.bRepellants[repellants[i]] = false;
				}
				repellants.clear();
			}
		}

		/*! \brief This method removes agents randomly form the simulation if the total number of agents is higher than input minimum.
		*
		*	\param		[in]	minAgents		- minimum number of agents.
		*	\since version 0.0.1
		*/
		void randomRemoveAgents(int minAgents)
		{
			if (agents.size() > minAgents)
			{
				int id = randomNumber(0, agents.size() - 1);
				agents[id].particle.fixed = true;
			}
		}

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method computes the color value of each cell in the environment based on chemical A or agent occupied cells.
		*
		*	\param		[in]	dispAgents		- display cells occupied by agents if true, else displays environment chemical.
		*	\param		[in]	usePercentile	- color cells using the percentile value. Works only with chemical A based coloring.
		*	\since version 0.0.1
		*/
		void computeEnvironmentColor(bool dispAgents = false, bool usePercentile = false)
		{
			

			if (!dispAgents)
			{
				for (int i = 0; i < environment.chemA.size(); i++)
				{
					double val;

					if (!usePercentile)
					{
						val = ofMap(environment.chemA[i], environment.minA, environment.maxA, 0.0, 1.0);
						environment.fieldMesh.faceColors[i] = zColor(val, val, val, 1);
					}
					

					if (usePercentile)
					{
						if (environment.chemA[i] >= 0 && environment.chemA[i] <= environment.maxA)
						{
							val = ofMap(environment.chemA[i], 0.0, environment.maxA, 0.5, 1.0);
							environment.fieldMesh.faceColors[i] = zColor(val, val, val, 1);
						}
						else if (environment.chemA[i] > environment.maxA)
						{
							val = 1;
							environment.fieldMesh.faceColors[i] = zColor(val, val, val, 1);
						}
						else if (environment.chemA[i] < 0 && environment.chemA[i] >= environment.minA)
						{
							val = ofMap(environment.chemA[i], environment.minA, 0.0, 1.0, 0.5);

							environment.fieldMesh.faceColors[i] = zColor(1 - val, 1 - val, 1 - val, 1);
						}
						else
						{
							val = 0;
							environment.fieldMesh.faceColors[i] = zColor(val, val, val, 1);
						}
					}
					


				}

				environment.fieldMesh.computeVertexColorfromFaceColor();

			}

			else
			{

				for (int i = 0; i < environment.resX * environment.resY; i++)
				{
					environment.fieldMesh.faceColors[i] = zColor(0, 0, 0, 1);
				}

				for (int i = 0; i < agents.size(); i++)
				{
					int outId;
					bool check = environment.field.getIndex(*agents[i].particle.getPosition(),outId);

					if (check)
					{
						if (!agents[i].particle.fixed)
						{
							if (environment.bRepellants[outId]) agents[i].particle.fixed = true;

							environment.fieldMesh.faceColors[outId] = zColor(1, 0, 0, 1);
						}
					}
							
				}

				environment.fieldMesh.computeVertexColorfromFaceColor();
			}
		}


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
		void updateAgentParametersFromField(int agentId, vector<double> &fieldScalars, double &parameterValue, double minVal, double maxVal, zSlimeParameter parm)
		{
			if (!agents[agentId].particle.fixed)
			{
				int fieldId;
				bool check = environment.field.getIndex(*agents[agentId].particle.getPosition(), fieldId);

				if (check)
				{
					double val = ofMap(fieldScalars[fieldId], 0.0, 1.0, minVal, maxVal);

					parameterValue = val;

					if (parm == zSO) agents[agentId].SO = &parameterValue;
					if (parm == zSA) agents[agentId].SA = &parameterValue;
					if (parm == zRA) agents[agentId].RA = &parameterValue;
					if (parm == zdepT) agents[agentId].depT = &parameterValue;
				}
			}
		}
				
	};
}

