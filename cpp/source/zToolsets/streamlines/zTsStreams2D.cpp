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


#include<headers/zToolsets/streamlines/zTsStreams2D.h>

//---- zStreamLine ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zStreamLine::zStreamLine()
	{
		parent = -1; // no parent

	}

	ZSPACE_INLINE void zStreamLine::setParent( int _parentId)
	{		
		parent = _parentId;
	}
}

//---- zTsStreams2D ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsStreams2D::zTsStreams2D()
	{
		fieldObj = nullptr;

		dSep = nullptr;
		dTest = nullptr;
		minLength = nullptr;
		maxLength = nullptr;

		dT = nullptr;
		angle = nullptr;
		flipBackward = nullptr;

		streamType = zForward;
		integrationType = zEuler;

	}

	ZSPACE_INLINE zTsStreams2D::zTsStreams2D(zObjMeshField<zVector> &_fieldObj)
	{
		fieldObj = &_fieldObj;
		fnField = zFnMeshField<zVector>(_fieldObj);

		fieldIndex_streamPositions.clear();

		for (int i = 0; i < fnField.numFieldValues(); i++)
		{
			vector<zVector> temp;
			fieldIndex_streamPositions.push_back(temp);
		}

		double tempDSEP = 1.0;
		dSep = &tempDSEP;

		double tempDTest = 0.8;
		dTest = &tempDTest;

		double tempMinLength = 5.0;
		minLength = &tempMinLength;

		double tempMaxLength = 5.0;
		maxLength = &tempMaxLength;

		double tempDT = 0.1;
		dT = &tempDT;

		double tempAngle = 0.0;
		angle = &tempAngle;

		bool tempFlip = false;
		flipBackward = &tempFlip;

		streamType = zForwardBackward;
		integrationType = zEuler;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsStreams2D::~zTsStreams2D() {}

	//----  SET METHODS

	ZSPACE_INLINE void zTsStreams2D::setSeperationDistance(double &_dSep)
	{
		dSep = &_dSep;
	}

	ZSPACE_INLINE void zTsStreams2D::setTestSeperationDistance(double &_dTest)
	{
		dTest = &_dTest;
	}

	ZSPACE_INLINE void zTsStreams2D::setMinLength(double &_minLength)
	{
		minLength = &_minLength;
	}

	ZSPACE_INLINE void zTsStreams2D::setMaxLength(double &_maxLength)
	{
		maxLength = &_maxLength;
	}

	ZSPACE_INLINE void zTsStreams2D::setIntegrationType(zFieldStreamType _streamType)
	{
		streamType = _streamType;
	}

	ZSPACE_INLINE void zTsStreams2D::setTimeStep(double &_dT)
	{
		dT = &_dT;
	}

	ZSPACE_INLINE void zTsStreams2D::setIntegrationType(zIntergrationType _integrationType)
	{
		integrationType = _integrationType;
	}

	ZSPACE_INLINE void zTsStreams2D::setAngle(double &_angle)
	{
		angle = &_angle;
	}

	ZSPACE_INLINE void zTsStreams2D::setFlipBackwards(bool &_flipBackward)
	{
		flipBackward = &_flipBackward;
	}

	//----  2D STREAM LINES METHODS

	ZSPACE_INLINE void zTsStreams2D::createStreams(vector<zStreamLine>& streams, vector<zVector> &start_seedPoints, bool seedStreamsOnly)
	{
		streams.clear();
		streams.assign(5000, zStreamLine());

		int streamCounter = 0;

		// make first stream line
		if (streamCounter == 0)
		{
			bool noStartSeedPoints = false;

			if (start_seedPoints.size() == 0)
			{
				noStartSeedPoints = true;

				zVector minBB, maxBB;
				fnField.getBoundingBox(minBB, maxBB);

				zVector seedPoint = zVector(coreUtils.randomNumber_double(minBB.x, maxBB.x), coreUtils.randomNumber_double(minBB.y, maxBB.y), 0);

				start_seedPoints.push_back(seedPoint);
			}

			for (int i = 0; i < start_seedPoints.size(); i++)
			{
				
				bool chk = createStreamGraph(streams[streamCounter].graphObj, start_seedPoints[i]);

				if (chk)
				{
					//streams.push_back(temp);

					streams[streamCounter].isValid = true;
					streamCounter++;				

				}
			}




		}


		if (seedStreamsOnly)
		{
			printf("\n %i streamLines created. ", streamCounter);
			return;
		}

		// compute other stream lines.

		int currentStreamGraphId = 0;
		int currentStreamGraphVertexId = 0;

		bool finished = false;

		while (!finished)
		{
			vector<zVector> seedPoints;

			
			zFnGraph fnGraph(streams[currentStreamGraphId].graphObj);

			if (streams[currentStreamGraphId].isValid)
			{
				getSeedPoints(streams[currentStreamGraphId], currentStreamGraphVertexId, seedPoints);


				for (int i = 0; i < seedPoints.size(); i++)
				{

					int id = streamCounter;
					//streams.push_back(zStreamLine());

					bool chk = createStreamGraph(streams[id].graphObj, seedPoints[i]);

					if (chk)
					{
						streams[currentStreamGraphId].child.push_back(id);
						streams[id].setParent(currentStreamGraphId);

						streams[id].isValid = true;
						streamCounter++;
					}
				}
			}

			

			currentStreamGraphVertexId++;

			if (currentStreamGraphVertexId >= fnGraph.numVertices()) currentStreamGraphId++, currentStreamGraphVertexId = 0;

			if (currentStreamGraphId >= streams.size()) finished = true;

			if (streamCounter >= streams.size()) finished = true;
		}

		printf("\n %i streamLines created. ", streamCounter);

	}

	//----  2D STREAM LINES METHODS WITH INFLUENCE SCALAR FIELD

	ZSPACE_INLINE void zTsStreams2D::createStreams_Influence(vector<zStreamLine>& streams, vector<zVector> &start_seedPoints, zFnMeshField<zScalar>& fnInfluenceField, double min_Power, double max_Power, bool seedStreamsOnly)
	{
		streams.clear();
		streams.assign(100, zStreamLine());

		vector<vector<int>> childgraphs;
		vector<int> parentGraph;
		bool alternate = false;

		int streamCounter = 0; 

		// make first stream line
		if (streamCounter == 0)
		{
			bool noStartSeedPoints = false;

			if (start_seedPoints.size() == 0)
			{
				noStartSeedPoints = true;

				zVector minBB, maxBB;
				fnField.getBoundingBox(minBB, maxBB);

				zVector seedPoint = zVector(coreUtils.randomNumber_double(minBB.x, maxBB.x), coreUtils.randomNumber_double(minBB.y, maxBB.y), 0);

				start_seedPoints.push_back(seedPoint);
			}

			for (int i = 0; i < start_seedPoints.size(); i++)
			{
				//streams.push_back(zStreamLine());

				bool chk = createStreamGraph_Influence(streams[streamCounter].graphObj, start_seedPoints[i], fnInfluenceField, min_Power, max_Power);

				if (chk)
				{
					//streams.push_back(temp);

					streams[streamCounter].isValid = true;
					streamCounter++;

					alternate = !alternate;

				}
			}




		}

		vector<bool> alternateGraph;
		for (int i = 0; i < streamCounter; i++) alternateGraph.push_back(false);

		if (seedStreamsOnly)
		{
			printf("\n %i streamLines created. ", streamCounter);

			for (int i = 0; i < streamCounter; i++)
			{
				zFnGraph fnGraph(streams[i].graphObj);

				zColor col = (alternateGraph[i]) ? zColor(1, 0, 0, 1) : zColor(0, 0, 1, 1);
				fnGraph.setEdgeColor(col, true);

			}

			return;
		}

		// compute other stream lines.

		int currentStreamGraphId = 0;
		int currentStreamGraphVertexId = 0;

		bool finished = false;


		int numGraphs = streams.size();
		int startGraph = 0;





		while (!finished)
		{
			vector<zVector> seedPoints;
			zFnGraph fnCurrentGraph(streams[currentStreamGraphId].graphObj);
			
			if (streams[currentStreamGraphId].isValid)
			{				
				
				getSeedPoints_Influence(fnInfluenceField, streams[currentStreamGraphId], currentStreamGraphVertexId, min_Power, max_Power, seedPoints);

				for (int i = 0; i < seedPoints.size(); i++)
				{

					int id = streamCounter;
					//streams.push_back(zStreamLine());

					
					bool chk = createStreamGraph_Influence(streams[id].graphObj, seedPoints[i], fnInfluenceField, min_Power, max_Power);

					if (chk)
					{
						streams[currentStreamGraphId].child.push_back(id);
						streams[id].setParent(currentStreamGraphId);

						streams[id].isValid = true;
						streamCounter++;

						alternateGraph.push_back(!alternateGraph[currentStreamGraphId]);

						alternate = !alternate;
					}
				}
			}
			

			currentStreamGraphVertexId++;

			if (currentStreamGraphVertexId >= fnCurrentGraph.numVertices()) currentStreamGraphId++, currentStreamGraphVertexId = 0;

			if (currentStreamGraphId >= streams.size()) finished = true;


		}

		for (int i = 0; i < streamCounter; i++)
		{

			zFnGraph fnGraph(streams[i].graphObj);
			zColor col = (alternateGraph[i]) ? zColor(1, 0, 0, 1) : zColor(0, 0, 1, 1);

			fnGraph.setEdgeColor(col, true);

		}


		printf("\n %i streamLines created. ", streamCounter);


	}

	//---- PROTECTED METHODS

	ZSPACE_INLINE bool zTsStreams2D::createStreamGraph(zObjGraph &streamGraphObj, zVector &seedPoint)
	{

		vector<zVector> positions;
		vector<int> edgeConnects;


		// move forward
		if (streamType == zForward || streamType == zForwardBackward)
		{
			bool exit = false;

			zVector startForward = seedPoint;

			zObjParticle p;
			p.particle = zParticle(startForward);

			zFnParticle seedForward(p);

			//zFnParticle seedForward;
			//seedForward.create(startForward);

			double currentLength = 0.0;

			//printf("\n working!");

			while (!exit)
			{
				bool firstVertex = (startForward == seedPoint) ? true : false;

				zVector curPos = seedForward.getPosition();

				if (firstVertex)
				{

					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;
				bool checkBounds = fnField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);



				if (!checkBounds)
				{
					//printf("\n bounds working!");

					exit = true;
					continue;
				}

				// local minima or maxima point
				if (fieldForce.length() == 0)
				{
					//printf("\n force working!");

					exit = true;
					continue;
				}

				// update particle force
				fieldForce.normalize();
				fieldForce *= (*dSep * 1.0);

				zVector axis(0, 0, 1);

				double rotateAngle = *angle;
				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);

				seedForward.addForce(fieldForce);

				// update seed particle
				seedForward.integrateForces(*dT, integrationType);
				seedForward.updateParticle(true);
				zVector newPos = seedForward.getPosition();
				checkBounds = checkFieldBounds(newPos);

				if (!checkBounds)
				{
					//printf("\n bounds 2 working!");

					//printf("\n %1.2f %1.2f %1.2f ", newPos.x, newPos.y, newPos.z);

					exit = true;
					continue;



				}

				int index = -1;
				bool checkRepeat = coreUtils.checkRepeatElement(newPos, positions, index);
				if (checkRepeat)
				{
					//printf("\n repeat working!");

					exit = true;
					continue;



				}


				bool validStreamPoint = checkValidStreamPosition(newPos, *dTest);

				if (!validStreamPoint)
				{
					exit = true;

					//printf("\n validity working!");
				}

				// check length
				if (currentLength + curPos.distanceTo(newPos) > *maxLength)
				{
					exit = true;

					//printf("\n length working!");
				}

				// add new stream point
				if (!exit)
				{


					if (positions.size() > 0)
					{
						edgeConnects.push_back(positions.size());
						edgeConnects.push_back(positions.size() - 1);
					}

					positions.push_back(newPos);

					currentLength += curPos.distanceTo(newPos);

				}




			}
		}


		if (streamType == zBackward || streamType == zForwardBackward)
		{
			// move backwards
			bool exit = false;

			zVector startBackward = seedPoint;

			zObjParticle p;
			p.particle = zParticle(startBackward);

			zFnParticle seedBackward(p);

			//zFnParticle seedBackward;
			//seedBackward.create(startBackward);


			double currentLength = 0.0;

			while (!exit)
			{
				bool firstVertex = (startBackward == seedPoint) ? true : false;


				zVector curPos = seedBackward.getPosition();


				// insert first point if the stream is inly for backward direction.
				if (firstVertex && streamType == zBackward)
				{
					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;

				bool checkBounds = fnField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}
				// local minima or maxima point
				if (fieldForce.length() == 0)
				{
					exit = true;
					continue;
				}

				// update particle force
				fieldForce.normalize();
				fieldForce *= (*dSep * 1.0);

				zVector axis(0, 0, 1);
				fieldForce *= -1;

				double rotateAngle = *angle;
				if (!*flipBackward) rotateAngle = 180.0 - *angle;

				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);

				seedBackward.addForce(fieldForce);

				// update seed particle
				seedBackward.integrateForces(*dT, integrationType);
				seedBackward.updateParticle(true);
				zVector newPos = seedBackward.getPosition();

				checkBounds = checkFieldBounds(newPos);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}

				int index = -1;
				bool checkRepeat = coreUtils.checkRepeatElement(newPos, positions, index);
				if (checkRepeat)
				{
					exit = true;
					continue;
				}

				bool validStreamPoint = checkValidStreamPosition(newPos, *dTest);


				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > *maxLength) exit = true;

				// add new stream point
				if (!exit)
				{


					if (positions.size() > 0)
					{
						(firstVertex) ? edgeConnects.push_back(0) : edgeConnects.push_back(positions.size() - 1);
						edgeConnects.push_back(positions.size());

					}

					positions.push_back(newPos);

					currentLength += curPos.distanceTo(newPos);
				}

			}
		}


		/*printf("\n v: %i e:%i ", positions.size(), edgeConnects.size());*/

		// create stream graph
		bool out = false;

		if (edgeConnects.size() > 0)
		{

			zFnGraph tempFn(streamGraphObj);
			tempFn.create(positions, edgeConnects);

			vector<double> lengths;
			double length = tempFn.getEdgeLengths(lengths);

			if (length > *minLength)
			{
				for (int i = 0; i < positions.size(); i++)
				{
					addToFieldStreamPositions(positions[i]);

				}

				out = true;
			}

		}



		return out;

	}

	ZSPACE_INLINE void zTsStreams2D::getSeedPoints(zStreamLine& currentStream, int vertexId, vector<zVector> &seedPoints)
	{
		zFnGraph tempFn(currentStream.graphObj);
		if (tempFn.numEdges() == 0) return;
		
		zItGraphVertex v(currentStream.graphObj, vertexId);

		if (v.checkValency(1)) return;

		

		zVector up(0, 0, 1);
		zVector norm;

		zVector vPos = v.getPosition();



		zItGraphHalfEdge curEdge = v.getHalfEdge();



		if (curEdge.getVertex().isActive())
		{
			zVector v1 = curEdge.getVertex().getPosition();
			zVector e1 = v1 - vPos;
			e1.normalize();

			norm += e1 ^ up;
		}

		if (curEdge.getPrev().isActive())
		{
			zVector v2 = curEdge.getPrev().getStartVertex().getPosition();
			zVector e2 = vPos - v2;
			e2.normalize();

			norm += e2 ^ up;
		}


		if (norm.length() == 0) return;

		norm *= 0.5;
		norm.normalize();

		zVector tempSeedPoint = vPos + (norm* *dSep);
		bool out = checkValidSeedPosition(tempSeedPoint, *dSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

		tempSeedPoint = vPos + (norm* *dSep*-1);
		out = checkValidSeedPosition(tempSeedPoint, *dSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

	}

	ZSPACE_INLINE bool zTsStreams2D::createStreamGraph_Influence(zObjGraph &streamGraphObj, zVector &seedPoint, zFnMeshField<zScalar>& fnInfluenceField, double min_Power, double max_Power)
	{

		vector<zVector> positions;
		vector<int> edgeConnects;


		// move forward
		if (streamType == zForward || streamType == zForwardBackward)
		{
			bool exit = false;

			zVector startForward = seedPoint;

			zObjParticle p;
			p.particle = zParticle(startForward);

			zFnParticle seedForward (p);
			//seedForward.create(startForward);


			double currentLength = 0;

			while (!exit)
			{
				bool firstVertex = (startForward == seedPoint) ? true : false;

				zVector curPos = seedForward.getPosition();

				if (firstVertex)
				{

					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;
				bool checkBounds = fnField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);



				if (!checkBounds)
				{

					exit = true;
					continue;
				}

				// local minima or maxima point
				if (fieldForce.length() == 0)
				{

					exit = true;
					continue;
				}


				// get dSep
				float influenceFieldValue;
				fnInfluenceField.getFieldValue(curPos, zFieldNeighbourWeighted, influenceFieldValue);
				double power = coreUtils.ofMap(influenceFieldValue, -1.0f, 1.0f, (float) min_Power,(float) max_Power);

				double distSep = *dSep / pow(2, power);


				// update particle force

				zVector axis(0, 0, 1);

				double rotateAngle = *angle;

				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);

				fieldForce.normalize();
								
				fieldForce *= (distSep * 1.0);

				seedForward.addForce(fieldForce);

				// update seed particle
				seedForward.integrateForces(*dT, integrationType);
				seedForward.updateParticle(true);
				zVector newPos = seedForward.getPosition();
				checkBounds = checkFieldBounds(newPos);

				if (!checkBounds)
				{

					exit = true;
					continue;


				}

				if (exit) cout << "\n e t";
				else cout << "\n e f";

				int index = -1;
				bool checkRepeat = coreUtils.checkRepeatElement(newPos, positions, index);

				if (checkRepeat)
				{

					exit = true;
					continue;


				}

				if (exit) cout << "\n e t";
				else cout << "\n e f";

				bool validStreamPoint = checkValidStreamPosition(newPos, *dTest);

				if (!validStreamPoint)
				{

					exit = true;
				}

				if (exit) cout << "\n e t";
				else cout << "\n e f";
				
				// check length
				if (currentLength + curPos.distanceTo(newPos) > *maxLength *0.5)  exit = true;

				if (exit) cout << "\n e t";
				else cout << "\n e f";

				// add new stream point
				if (!exit)
				{


					if (positions.size() > 0)
					{
						edgeConnects.push_back(positions.size());
						edgeConnects.push_back(positions.size() - 1);


					}

					positions.push_back(newPos);

					currentLength += curPos.distanceTo(newPos);
				}




			}
		}


		if (streamType == zBackward || streamType == zForwardBackward)
		{
			// move backwards
			bool exit = false;

			zVector startBackward = seedPoint;

			zObjParticle p;
			p.particle = zParticle(startBackward);

			zFnParticle seedBackward(p);
			//seedBackward.create(startBackward);

			double currentLength = 0.0;

			while (!exit)
			{
				bool firstVertex = (startBackward == seedPoint) ? true : false;


				zVector curPos = seedBackward.getPosition();


				// insert first point if the stream is inly for backward direction.
				if (firstVertex && streamType == zBackward)
				{
					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;

				bool checkBounds = fnField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}
				// local minima or maxima point
				if (fieldForce.length() == 0)
				{
					exit = true;
					continue;
				}

				// get dSep
				float influenceFieldValue;
				fnInfluenceField.getFieldValue(curPos, zFieldNeighbourWeighted, influenceFieldValue);
				double power = coreUtils.ofMap(influenceFieldValue, -1.0f, 1.0f, (float) min_Power, (float) max_Power);

				double distSep = *dSep / pow(2, power);

				// update particle force

				zVector axis(0, 0, 1);
				fieldForce *= -1;

				double rotateAngle = *angle;

				if (!*flipBackward) rotateAngle = 180.0 - *angle;

				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);

				fieldForce.normalize();
				fieldForce *= (distSep * 1.0);

				seedBackward.addForce(fieldForce);

				// update seed particle
				seedBackward.integrateForces(*dT, integrationType);
				seedBackward.updateParticle(true);
				zVector newPos = seedBackward.getPosition();

				checkBounds = checkFieldBounds(newPos);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}

				int index = -1;
				bool checkRepeat = coreUtils.checkRepeatElement(newPos, positions, index);
				if (checkRepeat)
				{
					exit = true;
					continue;
				}

				bool validStreamPoint = checkValidStreamPosition(newPos, *dTest);


				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > *maxLength * 0.5)  exit = true;

				// add new stream point
				if (!exit)
				{


					if (positions.size() > 0)
					{
						(firstVertex) ? edgeConnects.push_back(0) : edgeConnects.push_back(positions.size() - 1);
						edgeConnects.push_back(positions.size());

					}

					positions.push_back(newPos);

					currentLength += curPos.distanceTo(newPos);
				}

			}
		}


		printf("\n v: %i e:%i ", positions.size(), edgeConnects.size());

		// create stream graph
		bool out = false;
		if (edgeConnects.size() > 0)
		{


			zFnGraph tempFn(streamGraphObj);
			tempFn.create(positions, edgeConnects);

			vector<double> lengths;
			double length = tempFn.getEdgeLengths(lengths);

			if (length > *minLength)
			{
				for (int i = 0; i < positions.size(); i++)
				{
					int curFieldIndex;
					bool checkBounds = fnField.checkPositionBounds(positions[i], curFieldIndex);

					if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(positions[i]);

				}

				out = true;
			}

		}



		return out;

	}

	ZSPACE_INLINE void zTsStreams2D::getSeedPoints_Influence(zFnMeshField<zScalar>& fnInfluenceField, zStreamLine& currentStream, int vertexId, double min_Power, double max_Power, vector<zVector> &seedPoints)
	{
		zFnGraph tempFn(currentStream.graphObj);
		if (tempFn.numEdges() == 0) return;

		zItGraphVertex v(currentStream.graphObj, vertexId);

		if (v.checkValency(1)) return;

		zVector up(0, 0, 1);
		zVector norm;

		zVector vPos = v.getPosition();

		// get dSep
		float influenceFieldValue;
		fnInfluenceField.getFieldValue(vPos, zFieldNeighbourWeighted, influenceFieldValue);

		double power = coreUtils.ofMap(influenceFieldValue, -1.0f, 1.0f, (float) min_Power, (float) max_Power);
		power = floor(power);
		double distSep = *dSep /*/ pow(2, power)*/;

		zItGraphHalfEdge curEdge = v.getHalfEdge();


		//if (curEdge.getVertex().isActive())
		//{
			zVector v1 = curEdge.getVertex().getPosition();
			zVector e1 = v1 - vPos;
			e1.normalize();

			norm += e1 ^ up;
		//}

		//if (curEdge.getPrev().isActive())
		//{
			zVector v2 = curEdge.getPrev().getStartVertex().getPosition();
			zVector e2 = vPos - v2;
			e2.normalize();

			norm += e2 ^ up;
		//}


		if (norm.length() == 0) return;

		norm *= 0.5;
		norm.normalize();

		zVector tempSeedPoint = vPos + (norm*distSep);
		bool out = checkValidSeedPosition(tempSeedPoint, distSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

		tempSeedPoint = vPos + (norm*distSep*-1);
		out = checkValidSeedPosition(tempSeedPoint, distSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

	}

	//----  2D FIELD UTILITIES

	ZSPACE_INLINE bool zTsStreams2D::checkFieldBounds(zVector &inPoint)
	{
		zVector minBB, maxBB;
		fnField.getBoundingBox(minBB, maxBB);	

		return coreUtils.pointInBounds(inPoint, minBB, maxBB);
	}

	ZSPACE_INLINE bool zTsStreams2D::checkValidStreamPosition(zVector &inPoint, double &dTest)
	{

		int newFieldIndex;
		bool check = fnField.checkPositionBounds(inPoint, newFieldIndex);

		bool validStreamPoint = true;

		for (int j = 0; j < fieldIndex_streamPositions[newFieldIndex].size(); j++)
		{
			zVector streamPoint = fieldIndex_streamPositions[newFieldIndex][j];


			double dist = streamPoint.distanceTo(inPoint);

			if (dist < dTest)
			{
				validStreamPoint = false;
				break;
			}
		}

		// check in neighbour if validStreamPoint is true
		if (validStreamPoint)
		{
			vector<int> ringNeighbours;

			zItMeshVectorField s(*fnField.fieldObj, newFieldIndex);

			s.getNeighbour_Ring( 1, ringNeighbours);

			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				if (ringNeighbours[i] == newFieldIndex) continue;

				for (int j = 0; j < fieldIndex_streamPositions[ringNeighbours[i]].size(); j++)
				{
					zVector streamPoint = fieldIndex_streamPositions[ringNeighbours[i]][j];

					double dist = streamPoint.distanceTo(inPoint);

					if (dist < dTest)
					{
						validStreamPoint = false;


						break;
					}
				}

			}
		}

		return validStreamPoint;
	}

	ZSPACE_INLINE bool zTsStreams2D::checkValidSeedPosition(zVector &inPoint, double &dSep)
	{


		int newFieldIndex;
		bool checkBounds = fnField.checkPositionBounds(inPoint, newFieldIndex);

		if (!checkBounds) return false;

		bool validSeedPoint = true;

		for (int j = 0; j < fieldIndex_streamPositions[newFieldIndex].size(); j++)
		{
			zVector streamPoint = fieldIndex_streamPositions[newFieldIndex][j];

			double dist = streamPoint.distanceTo(inPoint);

			if (dist < dSep)
			{
				validSeedPoint = false;
				break;
			}
		}

		// check in neighbour if validStreamPoint is true
		if (validSeedPoint)
		{
			vector<int> ringNeighbours;

			zItMeshVectorField s(*fnField.fieldObj, newFieldIndex);
			s.getNeighbour_Ring( 1, ringNeighbours);

			for (int i = 0; i < ringNeighbours.size(); i++)
			{
				if (ringNeighbours[i] == newFieldIndex) continue;

				for (int j = 0; j < fieldIndex_streamPositions[ringNeighbours[i]].size(); j++)
				{
					zVector streamPoint = fieldIndex_streamPositions[ringNeighbours[i]][j];

					double dist = streamPoint.distanceTo(inPoint);

					if (dist < dSep)
					{
						validSeedPoint = false;
						break;
					}
				}

			}
		}

		return validSeedPoint;
	}

	ZSPACE_INLINE void zTsStreams2D::addToFieldStreamPositions(zVector &inPoint)
	{

		int curFieldIndex;
		bool checkBounds = fnField.checkPositionBounds(inPoint, curFieldIndex);

		if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(inPoint);
	}
}