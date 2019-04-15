#pragma once

#include <headers/api/functionsets/zFnPointCloud.h>
#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>
#include <headers/api/functionsets/zFnField2D.h>
#include <headers/api/functionsets/zFnParticle.h>

namespace zSpace
{
	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/** \addtogroup zTsSlimeMould
	*	\brief The slime mould related tool sets to create the simulation.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zTsSlimeAgent
	*	\brief A slime agent class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	class zTsSlimeAgent
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
		zTsSlimeAgent()
		{
				

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

		
		/*! \brief Overloaded constructor.
		*
		*	\since version 0.0.2
		*/
		zTsSlimeAgent(zVector &_pos, double &_SO, double &_SA, double &_RA, double &_depT, double &_pCD, double &_sMin)
		{
			particlesObj.particle = zParticle(_pos, false);
			fnParticle = zFnParticle(particlesObj);
					

			tCol = zColor(0, 0, 0, 1);
			tWeight = 1;

			tCounter = 0;
			tMax = 1000;

			tMat.setIdentity();

			SO = &_SO;
			SA = &_SA;
			RA = &_RA;

			depT = &_depT;
			pCD = &_pCD;
			sMin = &_sMin;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zTsSlimeAgent(){}

		
		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method returns the forward direction for the agent.
		*
		*	\return 	zVector				- vector of forward direction.
		*	\since version 0.0.1
		*/
		zVector getF()
		{
				
			fnParticle.getVelocity().normalize();			
			zVector F = fnParticle.getVelocity() * (*SO);

			return (fnParticle.getPosition() + F);
		}

		/*! \brief This method returns the forward right direction for the agent.
		*
		*	\return 	zVector				- vector of forward right direction.
		*	\since version 0.0.1
		*/
		zVector getFR()
		{
			zVector axis(0, 0, 1);

			fnParticle.getVelocity().normalize();
			zVector velocity = fnParticle.getVelocity() * (*SO);
			

			zVector FR = velocity.rotateAboutAxis(axis, -(*SA));
		
			return (fnParticle.getPosition() + FR);
		}

		/*! \brief This method returns the forward left direction for the agent.
		*
		*	\return 	zVector				- vector of forward left direction.
		*	\since version 0.0.1
		*/
		zVector getFL()
		{
			zVector axis(0, 0, 1);

			fnParticle.getVelocity().normalize();
			zVector velocity = fnParticle.getVelocity() * (*SO);

			zVector FL = velocity.rotateAboutAxis(axis, (*SA));
	
			return  (fnParticle.getPosition() + FL);
		}



		/*! \brief This method returns the sensor offset value for the agent.
		*
		*	\return 	double				- sensor offset value .
		*	\since version 0.0.1
		*/
		double getSO()
		{
			return *SO;
		}

		/*! \brief This method returns the sensor angle value for the agent.
		*
		*	\return 	double				- sensor angle value .
		*	\since version 0.0.1
		*/
		double getSA()
		{
			return *SA;
		}

		/*! \brief This method returns the rotation angle value for the agent.
		*
		*	\return 	double				- rotation angle value .
		*	\since version 0.0.1
		*/
		double getRA()
		{
			return *RA;
		}

		/*! \brief This method returns the sensititvity threshold value for the agent.
		*
		*	\return 	double				- sensititvity threshold value .
		*	\since version 0.0.1
		*/
		double getSMin()
		{
			return *sMin;
		}

		/*! \brief This method returns the deposition per step value for the agent.
		*
		*	\return 	double				- deposition per step value.
		*	\since version 0.0.1
		*/
		double getDepT()
		{
			return *depT;
		}


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
		void setVelocity(double a_F, double a_FR, double a_FL, bool chemoRepulsive = false)
		{
			zVector out;
			zVector axis(0, 0, 1);


			if (a_F > a_FL &&  a_F > a_FR)
			{
				out = fnParticle.getVelocity();
			}
			else if (a_F < a_FL &&  a_F < a_FR)
			{
				double rand = coreUtils.randomNumber_double(0, 1);
				float ang = (rand > 0.5) ? (-(*RA)) : ((*RA));

				out = fnParticle.getVelocity().rotateAboutAxis(axis, ang);
			}
			else if (a_FL < a_FR)
			{
				out = fnParticle.getVelocity().rotateAboutAxis(axis, -(*RA));
			}
			else if (a_FR < a_FL)
			{
				out = fnParticle.getVelocity().rotateAboutAxis(axis, (*RA));
			}
			else
			{
				out = fnParticle.getVelocity();
			}

			out.normalize();

			if (chemoRepulsive) out *= -1;
			fnParticle.setVelocity(out);

			//printf("\n inside  %1.2f %1.2f %1.2f ", particle.getVelocity().x, particle.getVelocity().y, particle.getVelocity().z);
		}

		/*! \brief This method sets the sensor offset value for the agent.
		*
		*	\param		[in]	_SO				- sensor offset value .
		*	\since version 0.0.2
		*/
		void setSO(double &_SO)
		{
			SO = &_SO;
		}

		/*! \brief This method sets the sensor angle value for the agent.
		*
		*	\param		[in]	_SA			- sensor angle value .
		*	\since version 0.0.2
		*/
		void setSA(double &_SA)
		{
			SA= &_SA;
		}

		/*! \brief This method sets the rotation angle value for the agent.
		*
		*	\param		[in]	_RA				- rotation angle value .
		*	\since version 0.0.2
		*/
		void setRA(double &_RA)
		{
			RA = &_RA;
		}

		/*! \brief This method sets the deposition per step value for the agent.
		*
		*	\param		[in]	_depT				-  deposition per step value .
		*	\since version 0.0.2
		*/
		void setDepT(double &_depT)
		{
			depT = &_depT;
		}

		/*! \brief This method sets the sensitivity threshold value for the agent.
		*
		*	\param		[in]	_sMin				-  sensitivity threshold value .
		*	\since version 0.0.2
		*/
		void setSMin(double &_sMin)
		{
			sMin = &_sMin;
		}

	};

	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
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

	class zTsSlimeEnvironment
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;		

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------
		
		/*!<scalar field.*/
		zFnField2D<double> fnField;			

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
		*	\since version 0.0.2
		*/
		zTsSlimeEnvironment()
		{
			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\param		[in]	_fieldMeshObj		- input mesh object.
		*	\since version 0.0.2
		*/
		zTsSlimeEnvironment(zObjField2D<double> &_fieldObj, zObjMesh &_fieldMeshObj)
		{
			fnField = zFnField2D<double>(_fieldObj, _fieldMeshObj);
		}
			

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zTsSlimeEnvironment(){}

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
			bool check = fnField.getIndex(pos,id);
	
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

			for (int i = 0; i < fnField.numFieldValues(); i++)
			{
				double lapA = 0;

				for (int j = 0; j < fnField.ringNeighbours[i].size(); j++)
				{
					int id = fnField.ringNeighbours[i][j];
								

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
					if (lapA != 0) lapA /= (fnField.ringNeighbours[i].size());
					temp_chemA.push_back(lapA);
				}
			}

			for (int i = 0; i < fnField.numFieldValues(); i++)
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

	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/** \addtogroup zSlimeMould
	*	\brief  The slime mould related function sets to create the simulation.
	*	\details Based on Physarum Transport Networks (http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf , https://pdfs.semanticscholar.org/21fe/f305fae61a9dbf9b63c0450b881dbd3ca154.pdf).
	*  @{
	*/

	/*! \class zFnSlime
	*	\brief A slime mould function set.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zTsSlime
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		zObjPointCloud *pointsObj;

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!<slime environment .*/
		zTsSlimeEnvironment environment;	
		
		/*!<slime agents.*/
		vector<zTsSlimeAgent>  agents;	

		/*!<stores slime agent positions.*/
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
		zTsSlime()
		{
			environment = zTsSlimeEnvironment();
			agents.clear();
			attractants.clear();
			repellants.clear();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\param		[in]	_fieldMeshObj		- input mesh object.
		*	\param		[in]	_pointsObj			- input pointcloud object.
		*	\since version 0.0.2
		*/
		zTsSlime(zObjField2D<double> &_fieldObj, zObjMesh &_fieldMeshObj, zObjPointCloud &_pointsObj)
		{
			environment = zTsSlimeEnvironment(_fieldObj, _fieldMeshObj);

			pointsObj = &_pointsObj;
			fnPositions = zFnPointCloud(_pointsObj);
			
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
		~zTsSlime() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the environment form the input parameters.
		*
		*	\param		[in]	_minBB		- bounding box minimum.
		*	\param		[in]	_maxBB		- bounding box maximum.
		*	\param		[in]	_res		- resolution of environment.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.2
		*/
		void createEnvironment(zVector _minBB, zVector _maxBB, int _resX, int _resY, int _NR = 1)
		{
			environment.resX = _resX;
			environment.resY = _resY;

		
			environment.fnField.create(_minBB, _maxBB, _resX, _resY, _NR);

			environment.chemA.clear();
			environment.occupied.clear();
			environment.bAttractants.clear();
			environment.bRepellants.clear();

			for (int i = 0; i < _resX*_resY; i++)
			{
				environment.chemA.push_back(0);
				environment.occupied.push_back(false);
				environment.bAttractants.push_back(false);
				environment.bRepellants.push_back(false);
			}

			environment.minMax_chemA();

			
		}

		/*! \brief This methods creates the environment form the input parameters.
		*	\param		[in]	_pix		- pixel size of environment.
		*	\param		[in]	_res		- resolution of environment.
		*	\param		[in]	_NR			- neighbourhod ring to be calculated.
		*	\since version 0.0.1
		*/
		void createEnvironment( double _pix, int _res, zVector _minBB = zVector(), int _NR = 1)
		{
			environment.resX = environment.resY = _res;
			environment.pix = _pix;

		
			environment.fnField.create(_pix, _pix, _res, _res, _minBB, _NR);

			environment.minMax_chemA();

			environment.chemA.clear();
			environment.occupied.clear();
			environment.bAttractants.clear();
			environment.bRepellants.clear();

			for (int i = 0; i < environment.resX * environment.resY; i++)
			{
				environment.chemA.push_back(0);
				environment.occupied.push_back(false);
				environment.bAttractants.push_back(false);
				environment.bRepellants.push_back(false);
			}

			environment.minMax_chemA();
		}


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
		void createAgents(double _p, double &_SO, double &_SA, double &_RA, double &_depT , double &_pCD, double &_sMin)
		{

			int numAgents = (int)environment.resX * environment.resY * _p;

			for (int i = 0; i < numAgents; i++)
			{
				int rnd = randomUnoccupiedCell();

				zVector pos = environment.fnField.getPosition(rnd);
				fnPositions.addPoint(pos);
				
				environment.occupied[rnd] = true;
			}		
					   
			agents.clear();

			for (int i = 0; i < fnPositions.numPoints(); i++)
			{
				agents.push_back(zTsSlimeAgent(pointsObj->pCloud.points[i], _SO, _SA, _RA, _depT, _pCD, _sMin));

				double randX = coreUtils.randomNumber_double(-1, 1);
				double randY = coreUtils.randomNumber_double(-1, 1);

				if (randX == randY && randY == 0)
					randX = (coreUtils.randomNumber_double(0, 1) > 0.5) ? 1 : -1;

				zVector velocity = zVector(randX, randY, 0);

				agents[i].fnParticle.setVelocity(velocity);
			}

		}

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
		void slime_Motor(double dT, zIntergrationType integrateType, bool agentTrail)
		{
		
			for (int i = 0; i < environment.resX * environment.resY; i++)
			{
				environment.occupied[i] = false;
			}		
			
			for (int i = 0; i < agents.size(); i++)
			{

				if (!agents[i].fnParticle.getFixed())
				{

					containBounds(i, dT, integrateType);

					// get new position
					zVector nPos = agents[i].fnParticle.getUpdatePosition(dT, integrateType);
					
					int fieldID = -1;;
					bool check = environment.fnField.getIndex(nPos, fieldID);

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

									agents[i].trail[aId] = agents[i].fnParticle.getPosition();
									agents[i].tCounter++;
								}
								else
								{
									agents[i].trail.push_back(agents[i].fnParticle.getPosition());
									agents[i].tCounter = agents[i].trail.size();
								}
							}

							
							agents[i].fnParticle.updateParticle(true,false,true);

							depositChemicalA(fieldID, agents[i].getDepT());

							environment.occupied[fieldID] = true;
						}
						else
						{
							double randX = coreUtils.randomNumber_double(-1, 1);
							double randY = coreUtils.randomNumber_double(-1, 1);
							zVector velocity = zVector(randX, randY, 0);
							agents[i].fnParticle.setVelocity(velocity);
						}
					}				
					else
					{
						double randX = coreUtils.randomNumber_double(-1, 1);
						double randY = coreUtils.randomNumber_double(-1, 1);
						zVector velocity = zVector(randX, randY, 0);
						agents[i].fnParticle.setVelocity(velocity);
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
				if (!agents[i].fnParticle.getFixed())
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
		*	\param		[in]	index		- input agent index .
		*	\param		[in]	dT			- time step.
		*	\param		[in]	type		- integration type.
		*	\since version 0.0.1
		*/
		void containBounds(int  index, double dT, zIntergrationType integrateType)
		{
			if (index > agents.size()) throw std::invalid_argument(" error: index out of bounds.");


			zVector nPos = agents[index].fnParticle.getUpdatePosition(dT, integrateType);
			int id_new;
			zVector cPos = agents[index].fnParticle.getPosition();

			bool check = environment.fnField.getIndex(cPos, id_new);

			if (check)
			{
				if (environment.chemA[id_new] < 0)
				{
					zVector f = agents[index].fnParticle.getForce() * -1;
					agents[index].fnParticle.setForce(f);
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
				environment.fnField.getNeighbourhoodRing(id, nR, neighbourRing);

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
				int rndX = coreUtils.randomNumber(boundaryOffset, environment.resX - boundaryOffset);
				int rndY = coreUtils.randomNumber(boundaryOffset, environment.resY - boundaryOffset);

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
				int id = coreUtils.randomNumber(0, agents.size() - 1);
				agents[id].fnParticle.setFixed(true);
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
						val = coreUtils.ofMap(environment.chemA[i], environment.minA, environment.maxA, 0.0, 1.0);
						environment.fnField.fnMesh.setFaceColor(i,  zColor(val, val, val, 1));
					}
					

					if (usePercentile)
					{
						if (environment.chemA[i] >= 0 && environment.chemA[i] <= environment.maxA)
						{
							val = coreUtils.ofMap(environment.chemA[i], 0.0, environment.maxA, 0.5, 1.0);
							environment.fnField.fnMesh.setFaceColor(i, zColor(val, val, val, 1));
						}
						else if (environment.chemA[i] > environment.maxA)
						{
							val = 1;
							environment.fnField.fnMesh.setFaceColor(i, zColor(val, val, val, 1));
						}
						else if (environment.chemA[i] < 0 && environment.chemA[i] >= environment.minA)
						{
							val = coreUtils.ofMap(environment.chemA[i], environment.minA, 0.0, 1.0, 0.5);

							environment.fnField.fnMesh.setFaceColor(i,  zColor(1 - val, 1 - val, 1 - val, 1));
						}
						else
						{
							val = 0;
							environment.fnField.fnMesh.setFaceColor(i, zColor(val, val, val, 1));
						}
					}
					


				}

				environment.fnField.fnMesh.computeVertexColorfromFaceColor();

			}

			else
			{

				for (int i = 0; i < environment.resX * environment.resY; i++)
				{
					environment.fnField.fnMesh.setFaceColor(i, zColor(0, 0, 0, 1));
				}

				for (int i = 0; i < agents.size(); i++)
				{
					int outId;
					zVector cPos = agents[i].fnParticle.getPosition();
					bool check = environment.fnField.getIndex(cPos,outId);

					if (check)
					{
						if (!agents[i].fnParticle.getFixed())
						{
							if (environment.bRepellants[outId]) agents[i].fnParticle.setFixed(true);

							environment.fnField.fnMesh.setFaceColor(outId,  zColor(1, 0, 0, 1));
						}
					}
							
				}

				environment.fnField.fnMesh.computeVertexColorfromFaceColor();
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
			if (!agents[agentId].fnParticle.getFixed())
			{
				int fieldId;
				zVector cPos = agents[agentId].fnParticle.getPosition();
				bool check = environment.fnField.getIndex(cPos, fieldId);

				if (check)
				{
					double val = coreUtils.ofMap(fieldScalars[fieldId], 0.0, 1.0, minVal, maxVal);

					parameterValue = val;

					if (parm == zSO) agents[agentId].setSO(parameterValue);
					if (parm == zSA) agents[agentId].setSA(parameterValue);
					if (parm == zRA) agents[agentId].setRA(parameterValue);
					if (parm == zdepT) agents[agentId].setDepT(parameterValue);
				}
			}
		}
				
	};
}

