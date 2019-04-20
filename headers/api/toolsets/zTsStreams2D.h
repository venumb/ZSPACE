#pragma once


#include <headers/api/functionsets/zFnMeshField.h>

#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>
#include <headers/api/functionsets/zFnParticle.h>


namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/** \addtogroup zTsStreamlines
	*	\brief tool sets for field stream lines.
	*  @{
	*/

	/*! \class zStream
	*	\brief An class to store data of a stream lines.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	   
	class zStream
	{
	protected:

		/*!	\brief pointer to stream graph Object  */
		zObjGraph *graphObj;

	public:
		//--------------------------
		//----  PUBLIC ATTRIBUTES
		//--------------------------

		/*!<stores the stream line as a graph.*/
		zFnGraph fnGraph;

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
			graphObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.1
		*/
		zStream( zObjGraph &_graphObj)
		{
			graphObj = &_graphObj;
			fnGraph = zFnGraph(_graphObj);

			parent = -1; // no parent

		}


		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\param		[in]	_parentId			- input parent index.
		*	\since version 0.0.1
		*/
		zStream(zObjGraph &_graphObj, int _parentId)
		{
			graphObj = &_graphObj;
			fnGraph = zFnGraph(_graphObj);

			parent = _parentId; 
		}

	};


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

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
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class zTsStreams2D 
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
		zTsStreams2D() 
		{
			fieldObj = nullptr;

			dSep = nullptr;
			dTest = nullptr;
			minLength = nullptr;
			maxLength = nullptr;

			dT = nullptr;
			angle = nullptr;
			flipBackward = nullptr;

			streamType = zForwardBackward;
			integrationType = zEuler;

		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_field			- input vector field 2D.
		*	\param		[in]	_fieldMesh		- input mesh.
		*	\since version 0.0.1
		*/
		zTsStreams2D(zObjMeshField<zVector> &_fieldObj, zObjMesh &_fieldMeshObj)
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


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zTsStreams2D() {}
		
		//--------------------------
		//----  SET METHODS
		//--------------------------

		/*! \brief This method sets the separation distance.
		*
		*	\param	[in]	_dSep		- input separation distance.
		*	\since version 0.0.1
		*/
		void setSeperationDistance(double &_dSep)
		{
			dSep = &_dSep;
		}

		/*! \brief This method sets the test separation distance.
		*
		*	\param	[in]	_dSep		- input test separation distance.
		*	\since version 0.0.1
		*/
		void setTestSeperationDistance(double &_dTest)
		{
			dTest = &_dTest;
		}

		/*! \brief This method sets the minimum length of the streams.
		*
		*	\param	[in]	_minLength		- input maximum length.
		*	\since version 0.0.1
		*/
		void setMinLength(double &_minLength)
		{
			minLength = &_minLength;
		}

		/*! \brief This method sets the maximum length of the streams.
		*
		*	\param	[in]	_maxLength		- input maximum length.
		*	\since version 0.0.1
		*/
		void setMaxLength(double &_maxLength)
		{
			maxLength = &_maxLength;
		}

		/*! \brief This method sets the stream type.
		*
		*	\param	[in]	_streamType		- input stream type
		*	\since version 0.0.1
		*/
		void setIntegrationType(zFieldStreamType _streamType)
		{
			streamType = _streamType;
		}

		/*! \brief This method sets the integration time step.
		*
		*	\param	[in]	_dT		- input time step
		*	\since version 0.0.1
		*/
		void setTimeStep(double &_dT)
		{
			dT = &_dT;
		}

		/*! \brief This method sets the integration type.
		*
		*	\param	[in]	_integrationType		- input integration type.
		*	\since version 0.0.1
		*/
		void setIntegrationType(zIntergrationType _integrationType)
		{
			integrationType = _integrationType;
		}

		/*! \brief This method sets the rotation angle.
		*
		*	\param	[in]	_angle		- input angle
		*	\since version 0.0.1
		*/
		void setAngle(double &_angle)
		{
			angle = &_angle;
		}

		/*! \brief This method sets the flip backwards boolean.
		*
		*	\param	[in]	_flipBackward		- input flip backwards boolean.
		*	\since version 0.0.1
		*/
		void setFlipBackwards(bool &_flipBackward)
		{
			flipBackward = &_flipBackward;
		}
		
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
		void createStreams(vector<zStream>& streams, vector<zVector> &start_seedPoints, bool seedStreamsOnly = false)
		{
			streams.clear();

			// make first stream line
			if (streams.size() == 0)
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
					zObjGraph temp;

					bool chk = createStreamGraph(temp, start_seedPoints[i]);


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

				getSeedPoints(streams[currentStreamGraphId], currentStreamGraphVertexId, seedPoints);


				for (int i = 0; i < seedPoints.size(); i++)
				{
					zObjGraph temp;
					bool chk = createStreamGraph(temp, seedPoints[i]);

					if (chk)
					{
						streams[currentStreamGraphId].child.push_back(streams.size());
						streams.push_back(zStream(temp, currentStreamGraphId));
					}
				}

				currentStreamGraphVertexId++;

				if (currentStreamGraphVertexId == streams[currentStreamGraphId].fnGraph.numVertices()) currentStreamGraphId++, currentStreamGraphVertexId = 0;

				if (currentStreamGraphId >= streams.size()) finished = true;

			}

			printf("\n %i streamLines created. ", streams.size());

		}

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
		void createStreams_Influence(vector<zStream>& streams, vector<zVector> &start_seedPoints,  zFnMeshField<double>& fnInfluenceField, double min_Power, double max_Power, bool seedStreamsOnly = false)
		{
			streams.clear();

			vector<vector<int>> childgraphs;
			vector<int> parentGraph;
			bool alternate = false;


			// make first stream line
			if (streams.size() == 0)
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
					zObjGraph temp;

					bool chk = createStreamGraph_Influence(temp, start_seedPoints[i], fnInfluenceField, min_Power, max_Power);

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
					streams[i].fnGraph.setEdgeColor(col, true);

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

				getSeedPoints_Influence( fnInfluenceField, streams[currentStreamGraphId], currentStreamGraphVertexId, min_Power, max_Power, seedPoints);

				for (int i = 0; i < seedPoints.size(); i++)
				{
					zObjGraph temp;
								

					bool chk = createStreamGraph_Influence(temp, seedPoints[i],  fnInfluenceField, min_Power, max_Power);

					if (chk)
					{
						streams[currentStreamGraphId].child.push_back(streams.size());
						streams.push_back(zStream(temp, currentStreamGraphId));

						alternateGraph.push_back(!alternateGraph[currentStreamGraphId]);

						alternate = !alternate;
					}
				}

				currentStreamGraphVertexId++;

				if (currentStreamGraphVertexId == streams[currentStreamGraphId].fnGraph.numVertices()) currentStreamGraphId++, currentStreamGraphVertexId = 0;

				if (currentStreamGraphId >= streams.size()) finished = true;


			}

			for (int i = 0; i < streams.size(); i++)
			{
				zColor col = (alternateGraph[i]) ? zColor(1, 0, 0, 1) : zColor(0, 0, 1, 1);
						
				streams[i].fnGraph.setEdgeColor(col, true);
	
			}


			printf("\n %i streamLines created. ", streams.size());


		}

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
		bool createStreamGraph(zObjGraph &streamGraphObj, zVector &seedPoint)
		{

			vector<zVector> positions;
			vector<int> edgeConnects;


			// move forward
			if (streamType == zForward || streamType == zForwardBackward)
			{
				bool exit = false;

				zVector startForward = seedPoint;

				zFnParticle seedForward;
				seedForward.create(startForward);				

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

				zFnParticle seedBackward;
				seedBackward.create(startBackward);
			

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

		/*! \brief This method computes the seed points.
		*
		*	\param	[in]	currentStream					- input current stream.
		*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
		*	\param	[in]	seedPoints						- container of seed points.
		*	\since version 0.0.1
		*/
		void getSeedPoints(zStream& currentStream, int vertexId, vector<zVector> &seedPoints)
		{
			if (currentStream.fnGraph.checkVertexValency(vertexId, 1)) return;

			zVector up(0, 0, 1);
			zVector norm;

			zVector v = currentStream.fnGraph.getVertexPosition(vertexId);



			zEdge *curEdge = currentStream.fnGraph.getEdge(vertexId,zVertexData);



			if (curEdge->getVertex())
			{
				zVector v1 = currentStream.fnGraph.getVertexPosition(curEdge->getVertex()->getVertexId());
				zVector e1 = v1 - v;
				e1.normalize();

				norm += e1 ^ up;
			}

			if (curEdge->getPrev())
			{
				zVector v2 = currentStream.fnGraph.getVertexPosition(curEdge->getPrev()->getSym()->getVertex()->getVertexId());
				zVector e2 = v - v2;
				e2.normalize();

				norm += e2 ^ up;
			}


			if (norm.length() == 0) return;

			norm *= 0.5;
			norm.normalize();

			zVector tempSeedPoint = v + (norm* *dSep);
			bool out = checkValidSeedPosition(tempSeedPoint, *dSep);
			if (out)  seedPoints.push_back(tempSeedPoint);

			tempSeedPoint = v + (norm* *dSep*-1);
			out = checkValidSeedPosition(tempSeedPoint, *dSep);
			if (out)  seedPoints.push_back(tempSeedPoint);

		}
		
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
		bool createStreamGraph_Influence(zObjGraph &streamGraphObj, zVector &seedPoint, zFnMeshField<double>& fnInfluenceField, double min_Power, double max_Power)
		{

			vector<zVector> positions;
			vector<int> edgeConnects;


			// move forward
			if (streamType == zForward || streamType == zForwardBackward)
			{
				bool exit = false;

				zVector startForward = seedPoint;

				zFnParticle seedForward;
				seedForward.create(startForward);
				

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
					double influenceFieldValue;
					fnInfluenceField.getFieldValue(curPos, zFieldNeighbourWeighted, influenceFieldValue);
					double power = coreUtils.ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);

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

					int index = -1;
					bool checkRepeat = coreUtils.checkRepeatElement(newPos, positions, index);
					if (checkRepeat)
					{
						
						exit = true;
						continue;


					}

					bool validStreamPoint = checkValidStreamPosition(newPos, *dTest);

					if (!validStreamPoint)
					{
						
						exit = true;
					}

					// check length
					if (currentLength + curPos.distanceTo(newPos) > *maxLength *0.5)  exit = true;

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

				zFnParticle seedBackward;
				seedBackward.create(startBackward);				

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
					double influenceFieldValue;
					fnInfluenceField.getFieldValue(curPos, zFieldNeighbourWeighted, influenceFieldValue);
					double power = coreUtils.ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);

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
						int curFieldIndex;
						bool checkBounds = fnField.getIndex(positions[i], curFieldIndex);

						if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(positions[i]);

					}

					out = true;
				}

			}



			return out;

		}

		/*! \brief This method computes the seed points.
		*
		*	\param	[in]	influenceField					- input scalar field.
		*	\param	[in]	currentStreamGraph				- input current stream graph.
		*	\param	[in]	vertexId						- vertex index in the vertices contatiner of the stream graph.
		*	\param	[in]	min_Power						- input minimum power value.
		*	\param	[in]	max_Power						- input maximum power value.
		*	\param	[in]	seedPoints						- container of seed points.
		*	\since version 0.0.1
		*/
		void getSeedPoints_Influence(zFnMeshField<double>& fnInfluenceField, zStream& currentStream, int vertexId, double min_Power, double max_Power, vector<zVector> &seedPoints)
		{
			if (currentStream.fnGraph.checkVertexValency(vertexId, 1)) return;

			zVector up(0, 0, 1);
			zVector norm;

			zVector v = currentStream.fnGraph.getVertexPosition(vertexId);

			// get dSep
			double influenceFieldValue;
			fnInfluenceField.getFieldValue(v, zFieldNeighbourWeighted, influenceFieldValue);

			double power = coreUtils.ofMap(influenceFieldValue, -1.0, 1.0, min_Power, max_Power);
			power = floor(power);
			double distSep = *dSep / pow(2, power);




			zEdge *curEdge = currentStream.fnGraph.getEdge(vertexId, zVertexData);



			if (curEdge->getVertex())
			{
				zVector v1 = currentStream.fnGraph.getVertexPosition(curEdge->getVertex()->getVertexId());
				zVector e1 = v1 - v;
				e1.normalize();

				norm += e1 ^ up;
			}

			if (curEdge->getPrev())
			{
				zVector v2 = currentStream.fnGraph.getVertexPosition(curEdge->getPrev()->getSym()->getVertex()->getVertexId());
				zVector e2 = v - v2;
				e2.normalize();

				norm += e2 ^ up;
			}


			if (norm.length() == 0) return;

			norm *= 0.5;
			norm.normalize();

			zVector tempSeedPoint = v + (norm*distSep);
			bool out = checkValidSeedPosition(tempSeedPoint, distSep);
			if (out)  seedPoints.push_back(tempSeedPoint);

			tempSeedPoint = v + (norm*distSep*-1);
			out = checkValidSeedPosition(tempSeedPoint, distSep);
			if (out)  seedPoints.push_back(tempSeedPoint);

		}

		//--------------------------
		//----  2D FIELD UTILITIES
		//--------------------------	

		/*! \brief This method checks if the input position is in the bounds of the field.
		*
		*	\param	[in]	inPoint		- input point.
		*	\return			bool		- true if the input position is in bounds.
		*	\since version 0.0.1
		*/
		bool checkFieldBounds(zVector &inPoint)
		{
			zVector minBB, maxBB;
			fnField.getBoundingBox(minBB, maxBB);

			return coreUtils.pointInBounds(inPoint, minBB, maxBB);
		}

		/*! \brief This method checks if the input position is a valid stream position.
		*
		*	\param	[in]	inPoint							- input point.
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		bool checkValidStreamPosition(zVector &inPoint, double &dTest)
		{

			int newFieldIndex;
			bool check = fnField.getIndex(inPoint, newFieldIndex);

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
				fnField.getNeighbour_Ring(newFieldIndex, 1, ringNeighbours);

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
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		bool checkValidSeedPosition(zVector &inPoint, double &dSep)
		{


			int newFieldIndex;
			bool checkBounds = fnField.getIndex(inPoint, newFieldIndex);

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
				fnField.getNeighbour_Ring(newFieldIndex, 1, ringNeighbours);

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

		/*! \brief This method adds the input position to the field stream position container.
		*
		*	\param	[in]	inPoint							- input point.
		*	\param	[in]	dTest							- dtest is a percentage of dsep. It is the minimal distance under which the integration of the streamline will be stopped in the current direction.
		*	\return			bool							- true if the input position is valid stream point.
		*	\since version 0.0.1
		*/
		void addToFieldStreamPositions(zVector &inPoint)
		{

			int curFieldIndex;
			bool checkBounds = fnField.getIndex(inPoint, curFieldIndex);

			if (checkBounds) fieldIndex_streamPositions[curFieldIndex].push_back(inPoint);
		}
	};



	


	


}
