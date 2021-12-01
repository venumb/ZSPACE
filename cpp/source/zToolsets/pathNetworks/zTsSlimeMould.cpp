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


#include<headers/zToolsets/pathNetworks/zTsSlimeMould.h>

//---- zSlimeAgent ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zSlimeAgent::zSlimeAgent()
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

	//---- DESTRUCTOR

	ZSPACE_INLINE zSlimeAgent::~zSlimeAgent() {}

	//---- CREATE METHOD

	ZSPACE_INLINE void zSlimeAgent::create(zVector &_pos, double &_SO, double &_SA, double &_RA, double &_depT, double &_pCD, double &_sMin)
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

	//---- GET METHODS

	ZSPACE_INLINE zVector zSlimeAgent::getF()
	{
		fnParticle.getVelocity().normalize();
		zVector F = fnParticle.getVelocity() * (*SO);

		return (fnParticle.getPosition() + F);
	}

	ZSPACE_INLINE zVector zSlimeAgent::getFR()
	{
		zVector axis(0, 0, 1);

		fnParticle.getVelocity().normalize();
		zVector velocity = fnParticle.getVelocity() * (*SO);


		zVector FR = velocity.rotateAboutAxis(axis, -(*SA));

		return (fnParticle.getPosition() + FR);
	}

	ZSPACE_INLINE zVector zSlimeAgent::getFL()
	{
		zVector axis(0, 0, 1);

		fnParticle.getVelocity().normalize();
		zVector velocity = fnParticle.getVelocity() * (*SO);

		zVector FL = velocity.rotateAboutAxis(axis, (*SA));

		return  (fnParticle.getPosition() + FL);
	}

	ZSPACE_INLINE double zSlimeAgent::getSO()
	{
		return *SO;
	}

	ZSPACE_INLINE double zSlimeAgent::getSA()
	{
		return *SA;
	}

	ZSPACE_INLINE double zSlimeAgent::getRA()
	{
		return *RA;
	}

	ZSPACE_INLINE double zSlimeAgent::getSMin()
	{
		return *sMin;
	}

	ZSPACE_INLINE double zSlimeAgent::getDepT()
	{
		return *depT;
	}

	//---- SET METHODS

	ZSPACE_INLINE void zSlimeAgent::setVelocity(double a_F, double a_FR, double a_FL, bool chemoRepulsive)
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

		//printf("\n inside %i  %1.2f %1.2f %1.2f ", fnParticle.getVelocity().x, fnParticle.getVelocity().y, fnParticle.getVelocity().z);
	}

	ZSPACE_INLINE void zSlimeAgent::setSO(double &_SO)
	{
		SO = &_SO;
	}

	ZSPACE_INLINE void zSlimeAgent::setSA(double &_SA)
	{
		SA = &_SA;
	}

	ZSPACE_INLINE void zSlimeAgent::setRA(double &_RA)
	{
		RA = &_RA;
	}

	ZSPACE_INLINE void zSlimeAgent::setDepT(double &_depT)
	{
		depT = &_depT;
	}

	ZSPACE_INLINE void zSlimeAgent::setSMin(double &_sMin)
	{
		sMin = &_sMin;
	}

}

//---- zSlimeEnvironment ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zSlimeEnvironment::zSlimeEnvironment()
	{

	}

	ZSPACE_INLINE zSlimeEnvironment::zSlimeEnvironment(zObjMeshField<zScalar> &_fieldObj)
	{
		fieldObj = &_fieldObj;
		fnMesh = zFnMesh(_fieldObj);
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zSlimeEnvironment::~zSlimeEnvironment() {}

	//---- GET SET METHODS

	ZSPACE_INLINE double zSlimeEnvironment::getChemAatPosition(zVector &pos)
	{

		//zItMeshScalarField s(*fieldObj, pos);
		//int id = s.getId();

		int fieldID = -1;;
		bool check = checkPositionBounds(pos, fieldID);

		if(check) 	
		return chemA[fieldID];
		else return -1.0;
	}

	//---- METHODS

	ZSPACE_INLINE void zSlimeEnvironment::diffuseEnvironment(double decayT, double diffuseDamp, zDiffusionType diffType)
	{
		vector<double> temp_chemA;

		for (int i = 0; i < numFieldValues(); i++)
		{
			double lapA = 0;

			for (int j = 0; j < ringNeighbours[i].size(); j++)
			{
				int id = ringNeighbours[i][j];


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
				if (lapA != 0) lapA /= (ringNeighbours[i].size());
				temp_chemA.push_back(lapA);
			}
		}

		for (int i = 0; i < numFieldValues(); i++)
		{
			chemA[i] = (1 - decayT) *temp_chemA[i];
		}
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zSlimeEnvironment::minMax_chemA(double min, double max)
	{
		minA = 10000;
		maxA = -10000;

		id_minA = -1;
		id_maxA = -1;

		vector<float> data = chemA;
		std::sort(data.begin(), data.end());

		int max_percentile = floor(max *  chemA.size());
		maxA = data[max_percentile - 1];
		id_maxA = max_percentile - 1;

		int min_percentile = floor(min *  chemA.size());
		minA = data[min_percentile];
		id_minA = min_percentile;
	}

}

//---- zTsSlime ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSlime::zTsSlime()
	{
		environment = zSlimeEnvironment();
		agents.clear();
		attractants.clear();
		repellants.clear();
	}

	ZSPACE_INLINE zTsSlime::zTsSlime(zObjMeshField<zScalar> &_fieldObj, zObjPointCloud &_pointsObj)
	{
		environment = zSlimeEnvironment(_fieldObj);

		pointsObj = &_pointsObj;
		fnPositions = zFnPointCloud(_pointsObj);

		agents.clear();
		attractants.clear();
		repellants.clear();
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSlime::~zTsSlime() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zTsSlime::createEnvironment(zVector _minBB, zVector _maxBB, int _resX, int _resY, int _NR)
	{
		environment.resX = _resX;
		environment.resY = _resY;


		environment.create(_minBB, _maxBB, _resX, _resY, _NR, true, false);

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

	ZSPACE_INLINE void zTsSlime::createEnvironment(double _pix, int _res, zVector _minBB, int _NR)
	{
		environment.resX = environment.resY = _res;
		environment.pix = _pix;


		environment.create(_pix, _pix, _res, _res, _minBB, _NR);

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

	ZSPACE_INLINE void zTsSlime::createAgents(double _p, double &_SO, double &_SA, double &_RA, double &_depT, double &_pCD, double &_sMin)
	{

		int numAgents = (int)environment.resX * environment.resY * _p;

		for (int i = 0; i < numAgents; i++)
		{
			int rnd = randomUnoccupiedCell();

			zItMeshScalarField s(*environment.fieldObj, rnd);

			zVector pos = s.getPosition();
			fnPositions.addPosition(pos);

			zItPointCloudVertex v(*pointsObj, i);
			v.setColor(zColor(1, 1, 1, 1));

			environment.occupied[rnd] = true;
		}


		agents.clear();
		agents.assign(fnPositions.numVertices(), zSlimeAgent());

		zVector* pos = fnPositions.getRawVertexPositions();

		for (int i = 0; i < fnPositions.numVertices(); i++)
		{
			agents[i].create(pos[i], _SO, _SA, _RA, _depT, _pCD, _sMin);

			double randX = coreUtils.randomNumber_double(-1, 1);
			double randY = coreUtils.randomNumber_double(-1, 1);

			if (randX == randY && randY == 0)
				randX = (coreUtils.randomNumber_double(0, 1) > 0.5) ? 1 : -1;

			zVector velocity = zVector(randX, randY, 0);

			agents[i].fnParticle.setVelocity(velocity);
		}

	}

	//---- METHODS

	ZSPACE_INLINE void zTsSlime::slime_Motor(double dT, zIntergrationType integrateType, bool agentTrail)
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

				bool check = environment.checkPositionBounds(nPos, fieldID);

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


						agents[i].fnParticle.updateParticle(false, false, false);

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

	ZSPACE_INLINE void zTsSlime::slime_Sensor(bool chemoRepulsive)
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

	ZSPACE_INLINE void zTsSlime::containBounds(int  index, double dT, zIntergrationType integrateType)
	{
		if (index > agents.size()) throw std::invalid_argument(" error: index out of bounds.");


		zVector nPos = agents[index].fnParticle.getUpdatePosition(dT, integrateType);
		int id_new;
		zVector cPos = agents[index].fnParticle.getPosition();

		bool check = environment.checkPositionBounds(cPos, id_new);

		if (check)
		{
			if (environment.chemA[id_new] < 0)
			{
				zVector f = agents[index].fnParticle.getForce() * -1;
				agents[index].fnParticle.setForce(f);
			}
		}

	}

	ZSPACE_INLINE bool  zTsSlime::addFoodSource(int id, double depA, int nR, bool attractant)
	{
		bool out = false;

		if (id < environment.bAttractants.size()) {

			zItMeshScalarField s(*environment.fieldObj, id);

			vector<int> neighbourRing;
			s.getNeighbour_Ring(nR, neighbourRing);

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

	ZSPACE_INLINE void zTsSlime::makeBoundaryRepellant(double depA, int distFromBoundary)
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

	ZSPACE_INLINE void zTsSlime::depositChemicalA(int id, double depA)
	{
		environment.chemA[id] += depA;
	}

	ZSPACE_INLINE void zTsSlime::depositAtFoodSource(double depA, double wProj)
	{
		for (int i = 0; i < attractants.size(); i++)
		{
			double dep = (1 + wProj) * depA;

			depositChemicalA(attractants[i], (dep));
		}

		for (int i = 0; i < repellants.size(); i++)
		{
			double dep = (1 + wProj) * depA;

			depositChemicalA(repellants[i], (-1 * dep));
		}
	}

	ZSPACE_INLINE int zTsSlime::randomUnoccupiedCell(int boundaryOffset)
	{
		bool exit = false;
		int outId = -1;

		while (!exit)
		{
			int rndX = coreUtils.randomNumber(boundaryOffset, environment.resX - boundaryOffset - 1);
			int rndY = coreUtils.randomNumber(boundaryOffset, environment.resY - boundaryOffset - 1);

			int id = (rndX * environment.resY) + rndY;
			if (!environment.occupied[id])
			{
				exit = true;
				outId = id;
			}
		}

		return outId;
	}

	ZSPACE_INLINE void zTsSlime::clearFoodSources(bool Attractants, bool Repellants)
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

	ZSPACE_INLINE void zTsSlime::randomRemoveAgents(int minAgents)
	{
		if (agents.size() > minAgents)
		{
			int id = coreUtils.randomNumber(0, agents.size() - 1);
			agents[id].fnParticle.setFixed(true);
		}
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zTsSlime::computeEnvironmentColor(bool dispAgents, bool usePercentile)
	{
		environment.setFieldValues(environment.chemA);
	}

	ZSPACE_INLINE void zTsSlime::updateAgentParametersFromField(int agentId, vector<double> &fieldScalars, double &parameterValue, double minVal, double maxVal, zSlimeParameter parm)
	{
		if (!agents[agentId].fnParticle.getFixed())
		{
			int fieldId;
			zVector cPos = agents[agentId].fnParticle.getPosition();
			bool check = environment.checkPositionBounds(cPos, fieldId);

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

}