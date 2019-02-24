#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zGraphMeshUtilities.h>
#include <headers/geometry/zMeshModifiers.h>

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

	/** \addtogroup zStreamLines2D
	*	\brief Collection of methods for stream lines of a 2D Field.
	*  @{
	*/

	/*! \struct zStreams
	*	\brief A strcut to store data of a stream lines.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	struct zStream
	{
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!<stores the stream line as a graph.*/
		zGraph streamGraph;

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

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief default constructor.
		*
		*	\since version 0.0.1
		*/
		zStream()
		{		

		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_streamGraph		- input stream graph.
		*	\since version 0.0.1
		*/
		zStream( zGraph &_streamGraph)
		{
			streamGraph = _streamGraph;

			parent = -1; // no parent

		}


		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_streamGraph		- input stream graph.
		*	\param		[in]	_parentId			- input parent index.
		*	\since version 0.0.1
		*/
		zStream(zGraph &_streamGraph , int _parentId)
		{
			streamGraph = _streamGraph;

			parent = _parentId; 
		}

	};



	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zStreamLines2D
	*	\brief Collection of methods for stream lines of a 2D Field.
	*  @{
	*/

	//--------------------------
	//----  2D FIELD UTILITIES
	//--------------------------	
	/*! \brief This method checks if the input position is in the bounds of the field.
	*
	*	\param	[in]	inPoint		- input point.
	*	\param	[in]	inField		- input field.
	*	\return			bool		- true if the input position is in bounds.
	*/
	bool checkFieldBounds(zVector &inPoint, zField2D<zVector>& inField)
	{
		zVector minBB, maxBB;
		inField.getBoundingBox(minBB, maxBB);

		return pointInBounds(inPoint, minBB, maxBB);
	}

	/*! \brief This method checks if the input position is a valid stream position.
	*
	*	\param	[in]	inPoint							- input point.
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\return			bool							- true if the input position is valid stream point.
	*/
	bool checkValidStreamPosition(zVector &inPoint, zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dTest)
	{
			
		int newFieldIndex;
		bool check = inField.getIndex(inPoint, newFieldIndex);

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
			inField.getNeighbourhoodRing(newFieldIndex, 1, ringNeighbours);

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
	
	/*! \brief This method checks if the input position is a valid seed position.
	*
	*	\param	[in]	inPoint							- input point.
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\return			bool							- true if the input position is valid stream point.
	*/
	bool checkValidSeedPosition(zVector &inPoint, zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dSep)
	{
			

		int newFieldIndex;
		bool checkBounds = 	inField.getIndex(inPoint, newFieldIndex);

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
			inField.getNeighbourhoodRing(newFieldIndex, 1, ringNeighbours);

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

	void addToFieldStreamPositions(zVector &inPoint, zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions)
	{
		if (fieldIndex_streamPositions.size() == 0 || fieldIndex_streamPositions.size() != inField.numFieldValues())
		{
			fieldIndex_streamPositions.clear();

			for (int i = 0; i < inField.numFieldValues(); i++)
			{
				vector<zVector> temp;
				fieldIndex_streamPositions.push_back(temp);
			}
		}

		int curFieldIndex;
		bool checkBounds = inField.getIndex(inPoint, curFieldIndex);

		if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(inPoint);
	}

	
	//--------------------------
	//----  2D STREAM LINES METHODS
	//--------------------------

	/*! \brief This method creates a single stream line as a graph.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	streamGraph						- stream graph created from the field.
	*	\param	[in]	seedPoint						- input seed point.
	*	\param	[in]	inField							- input field.	
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\param	[in]	streamType						- field stream type. ( zForward / zbackward / zForwardbackward)
	*	\param	[in]	dT								- time step.
	*	\param	[in]	type							- integration type.
	*	\param	[in]	NormalDir						- integrates the stream lines normal to the field direction.
	*	\return			bool							- true if the graph is created.
	*	\since version 0.0.1	
	*/	
	bool createStreamGraph( zGraph &streamGraph, zVector &seedPoint, zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dSep, double &dTest, double minLength, double maxLength, zFieldStreamType streamType, double dT, zIntergrationType type, bool NormalDir = false)
	{
		

		vector<zVector> positions;
		vector<int> edgeConnects;
		
			   		 	
		// move forward
		if (streamType == zForward || streamType == zForwardBackward)
		{
			bool exit = false;

			zVector startForward = seedPoint;

			zParticle seedForward(&startForward);

			double currentLength = 0.0;

			while (!exit)
			{
				bool firstVertex = (startForward == seedPoint) ? true : false;

				zVector curPos = *seedForward.getPosition();

				if (firstVertex)
				{

					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;
				bool checkBounds = inField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);
				
				

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
				fieldForce *= (dSep * 1.0);

				if (NormalDir) fieldForce = fieldForce ^ zVector(0, 0, 1);

				seedForward.addForce(fieldForce);

				// update seed particle
				seedForward.integrateForces(dT, type);
				seedForward.updateParticle(true);
				zVector newPos = *seedForward.getPosition();
				checkBounds = checkFieldBounds(newPos, inField);			

				if (!checkBounds)
				{
					exit = true;
					continue;

					
				}

				bool checkRepeat = checkRepeatElement(newPos, positions);
				if (checkRepeat)
				{
					exit = true;
					continue;

					
				}

				
				bool validStreamPoint = checkValidStreamPosition(newPos, inField, fieldIndex_streamPositions, dTest);

				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > maxLength) exit = true;
				
				// add new stream point
				if(!exit)
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

			zParticle seedBackward(&startBackward);

			double currentLength = 0.0;

			while (!exit)
			{
				bool firstVertex = (startBackward == seedPoint) ? true : false;


				zVector curPos = *seedBackward.getPosition();

			
				// insert first point if the stream is inly for backward direction.
				if (firstVertex && streamType == zBackward)
				{
					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;
				
				bool checkBounds = inField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);

				if(!checkBounds)
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
				fieldForce *= (dSep * -1.0);

				if (NormalDir) fieldForce = fieldForce ^ zVector(0, 0, 1);

				seedBackward.addForce(fieldForce);

				// update seed particle
				seedBackward.integrateForces(dT, type);
				seedBackward.updateParticle(true);
				zVector newPos = *seedBackward.getPosition();

				checkBounds = checkFieldBounds(newPos, inField);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}

				bool checkRepeat = checkRepeatElement(newPos, positions);
				if (checkRepeat)
				{
					exit = true;
					continue;
				}

				bool validStreamPoint = checkValidStreamPosition(newPos, inField, fieldIndex_streamPositions, dTest);


				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > maxLength) exit = true;

				// add new stream point
				if(!exit)
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
			streamGraph = zGraph(positions, edgeConnects);

			vector<double> lengths;
			double length = getEdgeLengths(streamGraph, lengths);
			
			if (length > minLength)
			{
				for (int i = 0; i < positions.size(); i++)
				{
					addToFieldStreamPositions(positions[i], inField, fieldIndex_streamPositions);
			
				}

				out = true;
			}
			
		}

		

		return out;

	}

	/*! \brief This method computes the seed points.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	currentStreamGraph				- input current stream graph.
	*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	seedPoints						- container of seed points.
	*	\since version 0.0.1
	*/
	void getSeedPoints(zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions, zGraph& currentStreamGraph, int vertexId, double &dSep, vector<zVector> &seedPoints)
	{
		if (currentStreamGraph.checkVertexValency(vertexId, 1)) return;

		zVector up(0, 0, 1);
		zVector norm;

		zVector v = currentStreamGraph.vertexPositions[vertexId];



		zEdge *curEdge = currentStreamGraph.vertices[vertexId].getEdge();



		if (curEdge->getVertex())
		{
			zVector v1 = currentStreamGraph.vertexPositions[curEdge->getVertex()->getVertexId()];
			zVector e1 = v1 - v;
			e1.normalize();

			norm += e1 ^ up;
		}

		if (curEdge->getPrev())
		{
			zVector v2 = currentStreamGraph.vertexPositions[curEdge->getPrev()->getSym()->getVertex()->getVertexId()];
			zVector e2 = v - v2;
			e2.normalize();

			norm += e2 ^ up;
		}


		if (norm.length() == 0) return;

		norm *= 0.5;
		norm.normalize();

		zVector tempSeedPoint = v + (norm*dSep);
		bool out = checkValidSeedPosition(tempSeedPoint, inField, fieldIndex_streamPositions, dSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

		tempSeedPoint = v + (norm*dSep*-1);
		out = checkValidSeedPosition(tempSeedPoint, inField, fieldIndex_streamPositions, dSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

	}

	/*! \brief This method creates the stream lines and stores them as a graph.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	streams							- container of streamlines.
	*	\param	[in]	start_seedPoints				- container of start seed positions. If empty a random position in the field is considered.
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\param	[in]	streamType						- field stream type. ( zForward / zbackward / zForwardbackward)
	*	\param	[in]	dT								- time step.
	*	\param	[in]	type							- integration type.
	*	\param	[in]	seedStreamsOnly					- generates streams from the seed points only if true.
	*	\param	[in]	NormalDir						- integrates the stream lines normal to the field direction.
	*	\since version 0.0.1
	*/
	void createStreams(vector<zStream>& streams, vector<zVector> &start_seedPoints, zField2D<zVector>& inField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dSep, double &dTest, double minLength, double maxLength, zFieldStreamType streamType = zForwardBackward, double dT =1.0, zIntergrationType type = zRK4, bool seedStreamsOnly = false, bool NormalDir = false  )
	{
		streams.clear();

		// setup fieldIndex_streamPositions
		if (fieldIndex_streamPositions.size() == 0 || fieldIndex_streamPositions.size() != inField.numFieldValues())
		{
			fieldIndex_streamPositions.clear();

			for (int i = 0; i < inField.numFieldValues(); i++)
			{
				vector<zVector> temp;
				fieldIndex_streamPositions.push_back(temp);
			}
		}	

		// make first stream line
		if (streams.size() == 0)
		{
			bool noStartSeedPoints = false;

			if (start_seedPoints.size() == 0)
			{		
				noStartSeedPoints = true;

				zVector minBB, maxBB;
				inField.getBoundingBox(minBB, maxBB);

				zVector seedPoint = zVector(randomNumber_double(minBB.x, maxBB.x), randomNumber_double(minBB.y, maxBB.y), 0);

				start_seedPoints.push_back(seedPoint);
			}

			for (int i = 0; i < start_seedPoints.size(); i++)
			{
				zGraph temp;

				double temp_DSep = (!noStartSeedPoints) ? 0.2 : dSep;
				bool chk = createStreamGraph(temp, start_seedPoints[i], inField, fieldIndex_streamPositions, temp_DSep, dTest, minLength, maxLength, streamType, dT, type, NormalDir);

			
				if (chk)
				{
					streams.push_back(zStream(temp));
				}
			}
		

			

		}


		if (seedStreamsOnly)
		{
			printf("\n %i streamLines created. ", streams.size());
			return;
		}

		// compute other stream lines.
		
		int currentStreamGraphId = 0;
		int currentStreamGraphVertexId = 0;

		bool finished = false;

		while (!finished)
		{
			vector<zVector> seedPoints;
			
			getSeedPoints(inField, fieldIndex_streamPositions, streams[currentStreamGraphId].streamGraph, currentStreamGraphVertexId, dSep, seedPoints);
		

			for (int i = 0; i < seedPoints.size(); i++)
			{
				zGraph temp;
				bool chk = createStreamGraph(temp, seedPoints[i], inField, fieldIndex_streamPositions, dSep, dTest, minLength, maxLength, streamType, dT, type, NormalDir);

				if (chk)
				{
					streams[currentStreamGraphId].child.push_back(streams.size());
					streams.push_back(zStream(temp, currentStreamGraphId));
				}
			}				

			currentStreamGraphVertexId++;

			if (currentStreamGraphVertexId == streams[currentStreamGraphId].streamGraph.numVertices()) currentStreamGraphId++ , currentStreamGraphVertexId = 0;

			if (currentStreamGraphId >= streams.size()) finished = true;
				
		}
		
		printf("\n %i streamLines created. ", streams.size());

	}


	//--------------------------
	//----  2D STREAM LINES METHODS WITH INFLUENCE SCALAR FIELD
	//--------------------------

	/*! \brief This method creates a single stream line as a graph.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	streamGraph						- stream graph created from the field.
	*	\param	[in]	seedPoint						- input seed point.
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\param	[in]	streamType						- field stream type. ( zForward / zbackward / zForwardbackward)
	*	\param	[in]	dT								- time step.
	*	\param	[in]	type							- integration type.
	*	\param	[in]	NormalDir						- integrates the stream lines normal to the field direction.
	*	\return			bool							- true if the graph is created.
	*	\since version 0.0.1
	*/
	bool createStreamGraph_Influence(zGraph &streamGraph, zVector &seedPoint, zField2D<zVector>& inField, zField2D<double>& influenceField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dSep, double &min_Power, double &max_Power, double &dTest_Factor, double minLength, double maxLength, zFieldStreamType streamType, double dT, zIntergrationType type, double angle = 0,  bool flipBackward = true)
	{
				
		vector<zVector> positions;
		vector<int> edgeConnects;


		// move forward
		if (streamType == zForward || streamType == zForwardBackward)
		{
			bool exit = false;

			zVector startForward = seedPoint;

			zParticle seedForward(&startForward);

			double currentLength = 0;

			while (!exit)
			{
				bool firstVertex = (startForward == seedPoint) ? true : false;

				zVector curPos = *seedForward.getPosition();

				if (firstVertex)
				{

					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;
				bool checkBounds = inField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);



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
				double influenceFieldValue;
				influenceField.getFieldValue(curPos,zFieldNeighbourWeighted, influenceFieldValue);
				double power = ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);

				double distSep = dSep / pow(2, power);

				
				// update particle force
		
				zVector axis(0, 0, 1);

				double rotateAngle = angle;
				//if (currentLength > maxLength * 0.25 && currentLength < maxLength * 0.5) rotateAngle += 90;;
				//if (currentLength > maxLength * 0.75 )fieldForce *= -1;

				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);
			
				fieldForce.normalize();
				fieldForce *= (distSep * 1.0);

				seedForward.addForce(fieldForce);

				// update seed particle
				seedForward.integrateForces(dT, type);
				seedForward.updateParticle(true);
				zVector newPos = *seedForward.getPosition();
				checkBounds = checkFieldBounds(newPos, inField);

				if (!checkBounds)
				{
					exit = true;
					continue;


				}

				bool checkRepeat = checkRepeatElement(newPos, positions);
				if (checkRepeat)
				{
					exit = true;
					continue;


				}

				double dTest = distSep * dTest_Factor;
				bool validStreamPoint = checkValidStreamPosition(newPos, inField, fieldIndex_streamPositions, dTest);

				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > maxLength*0.5 )  exit = true;

				// add new stream point
				if(!exit )
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

			zParticle seedBackward(&startBackward);

			double currentLength = 0.0;

			while (!exit)
			{
				bool firstVertex = (startBackward == seedPoint) ? true : false;


				zVector curPos = *seedBackward.getPosition();


				// insert first point if the stream is inly for backward direction.
				if (firstVertex && streamType == zBackward)
				{
					positions.push_back(curPos);
				}

				// get field focrce
				zVector fieldForce;

				bool checkBounds = inField.getFieldValue(curPos, zFieldNeighbourWeighted, fieldForce);

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
				double influenceFieldValue;
				influenceField.getFieldValue(curPos, zFieldNeighbourWeighted, influenceFieldValue);
				double power = ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);

				double distSep = dSep / pow(2, power);

				// update particle force
								
				zVector axis(0, 0, 1);
				fieldForce *= -1;
				
				double rotateAngle = angle ;

				//if (currentLength > maxLength * 0.25 && currentLength < maxLength * 0.5) rotateAngle += 60;
				//if (currentLength > maxLength * 0.75) rotateAngle += 60;

				if (!flipBackward) rotateAngle = 180.0 - angle  ;

				fieldForce = fieldForce.rotateAboutAxis(axis, rotateAngle);

				fieldForce.normalize();
				fieldForce *= (distSep * 1.0);

				seedBackward.addForce(fieldForce);

				// update seed particle
				seedBackward.integrateForces(dT, type);
				seedBackward.updateParticle(true);
				zVector newPos = *seedBackward.getPosition();

				checkBounds = checkFieldBounds(newPos, inField);

				if (!checkBounds)
				{
					exit = true;
					continue;
				}

				bool checkRepeat = checkRepeatElement(newPos, positions);
				if (checkRepeat)
				{
					exit = true;
					continue;
				}

				double dTest = distSep * dTest_Factor;
				bool validStreamPoint = checkValidStreamPosition(newPos, inField, fieldIndex_streamPositions, dTest);


				if (!validStreamPoint) exit = true;

				// check length
				if (currentLength + curPos.distanceTo(newPos) > maxLength*0.5)  exit = true;

				// add new stream point
				if(!exit)
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
		
			streamGraph = zGraph(positions, edgeConnects);

			vector<double> lengths;
			double length = getEdgeLengths(streamGraph, lengths);

			if (length > minLength)
			{
				for (int i = 0; i < positions.size(); i++)
				{
					int curFieldIndex;
					bool checkBounds = inField.getIndex(positions[i], curFieldIndex);

					if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(positions[i]);

				}

				out = true;
			}

		}



		return out;

	}

	/*! \brief This method computes the seed points.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	currentStreamGraph				- input current stream graph.
	*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	seedPoints						- container of seed points.
	*	\since version 0.0.1
	*/
	void getSeedPoints_Influence(zField2D<zVector>& inField, zField2D<double>& influenceField, vector<vector<zVector>> &fieldIndex_streamPositions, zGraph& currentStreamGraph, int vertexId, double &dSep, double &min_Power, double &max_Power, vector<zVector> &seedPoints)
	{
		if (currentStreamGraph.checkVertexValency(vertexId, 1)) return;

		zVector up(0, 0, 1);
		zVector norm;

		zVector v = currentStreamGraph.vertexPositions[vertexId];

		// get dSep
		double influenceFieldValue;
		influenceField.getFieldValue(v, zFieldNeighbourWeighted, influenceFieldValue);
		
		double power = ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);
		power = floor(power);
		double distSep = dSep / pow(2, power);
		
		


		zEdge *curEdge = currentStreamGraph.vertices[vertexId].getEdge();



		if (curEdge->getVertex())
		{
			zVector v1 = currentStreamGraph.vertexPositions[curEdge->getVertex()->getVertexId()];
			zVector e1 = v1 - v;
			e1.normalize();

			norm += e1 ^ up;
		}

		if (curEdge->getPrev())
		{
			zVector v2 = currentStreamGraph.vertexPositions[curEdge->getPrev()->getSym()->getVertex()->getVertexId()];
			zVector e2 = v - v2;
			e2.normalize();

			norm += e2 ^ up;
		}


		if (norm.length() == 0) return;

		norm *= 0.5;
		norm.normalize();

		zVector tempSeedPoint = v + (norm*distSep);
		bool out = checkValidSeedPosition(tempSeedPoint, inField, fieldIndex_streamPositions, distSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

		tempSeedPoint = v + (norm*distSep*-1);
		out = checkValidSeedPosition(tempSeedPoint, inField, fieldIndex_streamPositions, distSep);
		if (out)  seedPoints.push_back(tempSeedPoint);

	}

	/*! \brief This method creates the stream lines and stores them as a graph.
	*
	*	\details Based on evenly spaced streamlines (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.9498&rep=rep1&type=pdf)
	*	\param	[in]	streamGraphs					- container of streamlines as graph.
	*	\param	[in]	start_seedPoints				- container of start seed positions. If empty a random position in the field is considered.
	*	\param	[in]	inField							- input field.
	*	\param	[in]	fieldIndex_streamPositions		- 2 dimensional container of stream positions per field index.
	*	\param	[in]	dSep							- minimal distance between seed points.
	*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
	*	\param	[in]	streamType						- field stream type. ( zForward / zbackward / zForwardbackward)
	*	\param	[in]	dT								- time step.
	*	\param	[in]	type							- integration type.
	*	\param	[in]	seedStreamsOnly					- generates streams from the seed points only if true.
	*	\param	[in]	NormalDir						- integrates the stream lines normal to the field direction.
	*	\since version 0.0.1
	*/
	void createStreams_Influence(vector<zStream>& streams, vector<zVector> &start_seedPoints, zField2D<zVector>& inField, zField2D<double>& influenceField, vector<vector<zVector>> &fieldIndex_streamPositions, double &dSep, double &min_Power, double &max_Power, double &dTest_Factor, double minLength, double maxLength, zFieldStreamType streamType = zForwardBackward, double dT = 1.0, zIntergrationType type = zRK4, bool seedStreamsOnly = false, double angle = 0, bool flipBackward = true)
	{
		streams.clear();

		vector<vector<int>> childgraphs;
		vector<int> parentGraph;
		bool alternate = false;

		// setup fieldIndex_streamPositions
		if (fieldIndex_streamPositions.size() == 0 || fieldIndex_streamPositions.size() != inField.numFieldValues())
		{
			fieldIndex_streamPositions.clear();

			for (int i = 0; i < inField.numFieldValues(); i++)
			{
				vector<zVector> temp;
				fieldIndex_streamPositions.push_back(temp);
			}
		}

		// make first stream line
		if (streams.size() == 0)
		{
			bool noStartSeedPoints = false;

			if (start_seedPoints.size() == 0)
			{
				noStartSeedPoints = true;

				zVector minBB, maxBB;
				inField.getBoundingBox(minBB, maxBB);

				zVector seedPoint = zVector(randomNumber_double(minBB.x, maxBB.x), randomNumber_double(minBB.y, maxBB.y), 0);

				start_seedPoints.push_back(seedPoint);
			}

			for (int i = 0; i < start_seedPoints.size(); i++)
			{
				zGraph temp;
				
				bool chk = createStreamGraph_Influence(temp, start_seedPoints[i], inField, influenceField, fieldIndex_streamPositions, dSep, min_Power, max_Power, dTest_Factor, minLength, maxLength, streamType, dT, type, angle, flipBackward);
			
				if (chk)
				{
					streams.push_back(zStream(temp));

					alternate = !alternate;

				}
			}




		}
		
		vector<bool> alternateGraph;
		for (int i = 0; i < streams.size(); i++) alternateGraph.push_back(false);

		if (seedStreamsOnly)
		{
			printf("\n %i streamLines created. ", streams.size());

			for (int i = 0; i < streams.size(); i++)
			{
				zColor col = (alternateGraph[i]) ? zColor(1, 0, 0, 1) : zColor(0, 0, 1, 1);
				setEdgeColor(streams[i].streamGraph, col, true);

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

			getSeedPoints_Influence(inField, influenceField, fieldIndex_streamPositions, streams[currentStreamGraphId].streamGraph, currentStreamGraphVertexId, dSep, min_Power, max_Power, seedPoints);


			for (int i = 0; i < seedPoints.size(); i++)
			{
				zGraph temp;

				double ang = (alternate) ? angle + 60.0 : angle;

				bool chk = createStreamGraph_Influence(temp, seedPoints[i], inField, influenceField, fieldIndex_streamPositions, dSep, min_Power, max_Power, dTest_Factor, minLength, maxLength, streamType, dT, type, ang, flipBackward);

				if (chk)
				{
					streams[currentStreamGraphId].child.push_back(streams.size());
					streams.push_back(zStream(temp, currentStreamGraphId));
					
					alternateGraph.push_back(!alternateGraph[currentStreamGraphId]);

					alternate = !alternate;
				}
			}

			currentStreamGraphVertexId++;

			if (currentStreamGraphVertexId == streams[currentStreamGraphId].streamGraph.numVertices()) currentStreamGraphId++, currentStreamGraphVertexId = 0;

			if (currentStreamGraphId >= streams.size()) finished = true;

			
		}

		for (int i = 0; i < streams.size(); i++)
		{
			zColor col = (alternateGraph[i]) ? zColor(1, 0, 0, 1) : zColor(0, 0, 1, 1);
			setEdgeColor(streams[i].streamGraph, col, true);
		
		}
		

		printf("\n %i streamLines created. ", streams.size());


		/*printf("\n Parents ");
		for (int i = 0; i < streams.size(); i++)
		{
			printf("\n %i : p %i ", i, streams[i].parent);
			printf("\n %i : child ", i);

			for (int j = 0; j < streams[i].child.size(); j++)
			{
				printf(" %i ", streams[i].child[j]);
			}

			printf("\n");
		}*/

		

	}



	/** @}*/

	/** @}*/
}
