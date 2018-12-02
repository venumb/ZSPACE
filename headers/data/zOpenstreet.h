#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zMeshModifiers.h>

#include <headers/geometry/zScalarField.h>
#include <headers/geometry/zScalarFieldUtilities.h>

#include <headers/data/zDatabase.h>

namespace zSpace
{

	/** \addtogroup zData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zCityData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/*! \struct zWays
	*	\brief A struct for storing  information of OSM ways and street graph.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	struct zWays
	{
		/*! \brief stores index of the way in a ways container.		*/
		int id;

		/*! \brief stores index of the OSM way.		*/
		string OS_wayId;

		/*! \brief container of streetgraph edges which correspond to the OSM way ID.		*/
		vector<int> streetGraph_edgeId;

		/*! \brief stores type of the building as given by OSM.		*/
		zDataStreet streetType;

	};


	/** \addtogroup zData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zCityData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/*! \struct zBuildings
	*	\brief A class for accessing the openstreet data and other city level data stored in a SQL database , CSV files etc.
	*	\details Uses open source data from https://www.openstreetmap.org
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	struct zBuildings
	{
		/*! \brief stores index of the way in a buildings container.		*/
		int id;

		/*! \brief stores index of the OSM way.		*/
		string OS_wayId;
		
		/*! \brief container of building graph edges which correspond to the OSM way ID.		*/
		vector<int> buildingGraph_edgeId;

		/*! \brief stores type of the building as given by OSM.		*/
		zDataBuilding buildingType;	

	};



	/** \addtogroup zData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zCityData
	*	\brief The data classes and utility methods of the library.
	*  @{
	*/

	/*! \class zOpenStreet
	*	\brief A struct for storing  information of OSM buildings and building graph.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	class zOpenStreet
	{

	public:

		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		//---- STREET ATTRIBUTES

		/*! \brief stores current number of ways		*/
		int n_zWays;

		/*! \brief graph of the street network given by OSM		*/
		zGraph streetGraph;

		/*! \brief zWays container		*/
		zWays *way;

		/*!	\brief street graph edgeId to wayId map. Used to get correponding wayId given the street graph EdgeId.  */
		map <int, string> streetEdges_Way;

		/*!	\brief wayId to street graph EdgeId map. Used to get correponding street graph EdgeId given the wayId.  */
		unordered_map <string, int> way_streetEdges;

		//---- BUILDING ATTRIBUTES

		/*! \brief stores current number of buildings		*/
		int n_zBuildings;

		/*! \brief graph of the buildings given by OSM		*/
		zGraph buildingGraph;

		/*! \brief zBuildings container		*/
		zBuildings *buildings;

		/*!	\brief building graph edgeId to wayId map. Used to get correponding wayId given the building graph EdgeId.  */
		map <int, string> buildingEdges_Building;

		/*!	\brief wayId to building graph EdgeId map. Used to get correponding building graph EdgeId given the wayId.  */
		unordered_map <string, int> Building_buildingEdges;

		//---- BOUNDS ATTRIBUTES

		/*!	\brief bounds of OSM data in latitute and logitude.  */
		double lat_lon[4];

		/*!	\brief minimum bounds of OSM data in 3D space given by a zVector.  */
		zVector minBB;

		/*!	\brief maximum bounds of OSM data in 3D space given by a zVector.  */
		zVector maxBB;

		//---- DATABASE ATTRIBUTES

		/*!	\brief database needed to acces the OSM and other data.  */
		zDatabase *zDB;

		//---- FIELD ATTRIBUTES

		/*!	\brief scalar field covering the bounds of data.  */
		zScalarField2D scalarfield;

		/*!	\brief mesh of the scalar field.  */
		zMesh  fieldMesh;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zOpenStreet()
		{
			zDB = NULL;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	DatabaseFileName		- file path to the SQL database.
		*	\since version 0.0.1
		*/
		zOpenStreet(char* DatabaseFileName)
		{
			zDB = new zDatabase(DatabaseFileName);
			zDB->close();
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zOpenStreet() {}


		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------


		/*! \brief This method computes the bounding box in 3D space of the OSM data from the  lat_lon container of the bounds.
		*
		*	\param		[in]	scaleFactor		- scale of the map to be displayed.
		*	\since version 0.0.1
		*/
		void computeBoundingBox(double scaleFactor)
		{
			double diagonalDist = computeDistance(this->lat_lon[0], this->lat_lon[1], this->lat_lon[2], this->lat_lon[3]) * scaleFactor;

			double distLat = computeDistance(this->lat_lon[0], this->lat_lon[1], this->lat_lon[2], this->lat_lon[1]) * scaleFactor;

			double distLon = computeDistance(this->lat_lon[0], this->lat_lon[1], this->lat_lon[0], this->lat_lon[3]) * scaleFactor;

			minBB = zVector(-distLon * 0.5, -distLat * 0.5, 0);
			maxBB = zVector(distLon * 0.5, distLat * 0.5, 0);

			printf("\n diagonal distance: %1.2f ", diagonalDist);
		}

		/*! \brief This method computes the distance between two geo-points given by input latitute and longitude.
		*
		*	\details based on https://www.movable-type.co.uk/scripts/latlong.html using Haversine formula.
		*	\param		[in]	lat1		- latitude of geo-point 1.
		*	\param		[in]	lon1		- longitude of geo-point 1.
		*	\param		[in]	lat2		- latitude of geo-point 2.
		*	\param		[in]	lon2		- longitude of geo-point 2.
		*	\return				double		- distance between points in kilometers.
		*	\since version 0.0.1
		*/
		double computeDistance(double &lat1, double &lon1, double &lat2, double &lon2)
		{
			double R = 6378.137; // Radius of earth in KM

			double dLat = (lat2 * PI / 180) - (lat1 * PI / 180);
			double dLon = (lon2 * PI / 180) - (lon1 * PI / 180);

			double a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1 * PI / 180) * cos(lat2 * PI / 180) * sin(dLon / 2) * sin(dLon / 2);
			double c = 2 * atan2(sqrt(a), sqrt(1 - a));
			double d = R * c;

			return d; // in kilometers
		}



		/*! \brief This method computes the scalar field from the bounds and input resolution. It also computes the field mesh.
		*
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*/
		void fieldFromBounds(int _n_X = 100, int _n_Y = 100)
		{
			zVector offset(0.5, 0.5, 0);

			scalarfield = zScalarField2D(this->minBB - offset, this->maxBB + offset, _n_X, _n_Y);

			fieldMesh = fromScalarField2D(this->scalarfield);

		}



		//--------------------------
		//---- STREET GRAPH METHODS
		//--------------------------

		/*! \brief This method gets the street type from the OSM data for the given wayId if it corresponds to a street.
		*
		*	\param		[in]	wayId		- input wayId.
		*	\since version 0.0.1
		*/
		zDataStreet getStreetType(string wayId)
		{
			vector<string> outStm;
			vector<string> sqlStm = { "SELECT v FROM ways_tags WHERE ways_tags.id = " + wayId + ";" };
			bool stat = zDB->sqlCommand(sqlStm, zSelect, false, outStm, false);



			zDataStreet out = zUndefinedStreet;

			if (outStm.size() > 0)
			{
				if (outStm[0] == "trunk" || outStm[0] == "trunk_link") out = zTrunkStreet;

				if (outStm[0] == "primary" || outStm[0] == "primary_link") out = zPrimaryStreet;

				if (outStm[0] == "secondary" || outStm[0] == "secondary_link") out = zSecondaryStreet;

				if (outStm[0] == "tertiary" || outStm[0] == "tertiary_link") out = zTertiaryStreet;

				if (outStm[0] == "residential" || outStm[0] == "living_street") out = zResidentialStreet;

				if (outStm[0] == "pedestrian" || outStm[0] == "footway") out = zPrimaryStreet;

				if (outStm[0] == "cycleway") out = zCycleStreet;

				if (outStm[0] == "service") out = zServiceStreet;

			}


			return out;

		}

		/*! \brief This method creates the street graph from the OSM data.
		*
		*	\param		[in]	edgeCol		- input color to be assigned to the edges of the graph.
		*	\since version 0.0.1
		*/
		void createStreetGraph(zColor edgeCol = zColor(0, 0, 0, 1))
		{
			vector<zVector>(positions);
			vector<int>(edgeConnects);

			unordered_map <string, int> node_streetVertices;
			map <int, string> streetVertices_Node;


			vector<string> outStm_nodes;
			vector<string> sqlStm_nodes = { " SELECT * FROM nodes WHERE id IN (SELECT node_id FROM ways_nodes WHERE way_id IN (SELECT id FROM ways_tags WHERE ways_tags.k = \"highway\")) ;" };
			bool stat = zDB->sqlCommand(sqlStm_nodes, zSelect, false, outStm_nodes, false);


			printf("\n outStm_nodes: %i", outStm_nodes.size());

			for (int i = 0; i < outStm_nodes.size(); i += 3)
			{


				string hashKey = outStm_nodes[i];;
				node_streetVertices[hashKey] = positions.size();

				streetVertices_Node[positions.size()] = hashKey;

				double lat = atof(outStm_nodes[i + 1].c_str());
				double lon = atof(outStm_nodes[i + 2].c_str());

				zVector pos = fromCoordinates(lat, lon);

				positions.push_back(pos);


			}

			vector<string> outStm_ways;
			vector<string> sqlStm_ways = { " SELECT * FROM ways_nodes WHERE way_id IN (SELECT id FROM ways_tags WHERE ways_tags.k = \"highway\");" };
			stat = zDB->sqlCommand(sqlStm_ways, zSelect, false, outStm_ways, false);


			way = new zWays[outStm_ways.size()];
			n_zWays = 0;

			for (int i = 0; i < outStm_ways.size() - 3; i += 3)
			{
				if (i != 0) i -= 3;

				string wayId = outStm_ways[i];

				way[n_zWays].id = n_zWays;
				way[n_zWays].OS_wayId = wayId;

				way[n_zWays].streetType = getStreetType(wayId);


				while (outStm_ways[i] == wayId && i < outStm_ways.size() - 3)
				{
					string hashKey = outStm_ways[i + 1];
					std::unordered_map<string, int>::const_iterator got = node_streetVertices.find(hashKey);

					if (got != node_streetVertices.end())
					{
						if (outStm_ways[i + 3] == wayId)
						{
							string hashKey1 = outStm_ways[i + 3 + 1].c_str();
							std::unordered_map<string, int>::const_iterator got1 = node_streetVertices.find(hashKey1);

							if (got1 != node_streetVertices.end())
							{
								way[n_zWays].streetGraph_edgeId.push_back(edgeConnects.size());
								streetEdges_Way[edgeConnects.size()] = wayId;
								way_streetEdges[wayId] = edgeConnects.size();
								edgeConnects.push_back(got->second);

								way[n_zWays].streetGraph_edgeId.push_back(edgeConnects.size());
								streetEdges_Way[edgeConnects.size()] = wayId;
								way_streetEdges[wayId] = edgeConnects.size();
								edgeConnects.push_back(got1->second);

							}

						}
					}

					i += 3;
				}

				n_zWays++;
			}





			streetGraph = zGraph(positions, edgeConnects);



			setEdgeColor(streetGraph, edgeCol, true);

		}


		//--------------------------
		//---- BUILDING GRAPH METHODS
		//--------------------------

		/*! \brief This method gets the street type from the OSM data for the given wayId if it corresponds to a street.
		*
		*	\param		[in]	wayId		- input wayId.
		*	\since version 0.0.1
		*/
		zDataBuilding getBuildingType(string wayId)
		{
			vector<string> outStm;
			vector<string> sqlStm = { "SELECT v FROM ways_tags WHERE ways_tags.id = " + wayId + ";" };
			bool stat = zDB->sqlCommand(sqlStm, zSelect, false, outStm, false);



			zDataBuilding out = zUndefinedBuilding;

			if (outStm.size() > 0)
			{
				if (outStm[0] == "office" || outStm[0] == "commercial" || outStm[0] == "hotel" || outStm[0] == "theatre" || outStm[0] == "retail") out = zCommercialBuilding;
				if (outStm[0] == "Commercial" || outStm[0] == "Arts_&_Office_Complex" || outStm[0] == "shop" || outStm[0] == "offices") out = zCommercialBuilding;

				if (outStm[0] == "apartments" || outStm[0] == "flats" || outStm[0] == "residential" || outStm[0] == "council_flats") out = zResidentialBuilding;
				if (outStm[0] == "house" || outStm[0] == "hall_of_residence" || outStm[0] == "Student hostel") out = zResidentialBuilding;

				if (outStm[0] == "gallery" || outStm[0] == "hospital" || outStm[0] == "church" || outStm[0] == "place_of_worship" || outStm[0] == "museum" || outStm[0] == "cathedral") out = zPublicBuilding;
				if (outStm[0] == "railway_station" || outStm[0] == "Community_Building" || outStm[0] == "train_station" || outStm[0] == "station" || outStm[0] == "civic") out = zPublicBuilding;

				if (outStm[0] == "school" || outStm[0] == "university" || outStm[0] == "college" || outStm[0] == "Nursery,_School") out = zUniversityBuilding;

			}


			return out;

		}

		/*! \brief This method creates the building graph from the OSM data.
		*
		*	\param		[in]	edgeCol		- input color to be assigned to the edges of the graph.
		*	\since version 0.0.1
		*/
		void createBuildingGraph(zColor edgeCol = zColor(0, 0, 0, 1))
		{
			vector<zVector>(positions);
			vector<int>(edgeConnects);

			unordered_map <string, int> node_buildingVertices;
			map <int, string> buildingVertices_Node;


			vector<string> outStm_nodes;
			vector<string> sqlStm_nodes = { " SELECT * FROM nodes WHERE id IN (SELECT node_id FROM ways_nodes WHERE way_id IN (SELECT id FROM ways_tags WHERE ways_tags.k = \"building\")) ;" };
			bool stat = zDB->sqlCommand(sqlStm_nodes, zSelect, false, outStm_nodes, false);

			printf("\n outStm_nodes: %i", outStm_nodes.size());

			for (int i = 0; i < outStm_nodes.size(); i += 3)
			{


				string hashKey = outStm_nodes[i];;
				node_buildingVertices[hashKey] = positions.size();

				buildingVertices_Node[positions.size()] = hashKey;

				double lat = atof(outStm_nodes[i + 1].c_str());
				double lon = atof(outStm_nodes[i + 2].c_str());

				zVector pos = fromCoordinates(lat, lon);

				positions.push_back(pos);


			}

			vector<string> outStm_ways;
			vector<string> sqlStm_ways = { " SELECT * FROM ways_nodes WHERE way_id IN (SELECT id FROM ways_tags WHERE ways_tags.k = \"building\");" };
			stat = zDB->sqlCommand(sqlStm_ways, zSelect, false, outStm_ways, false);

			printf("\n outStm_ways: %i", outStm_ways.size());

			buildings = new zBuildings[outStm_ways.size()];
			n_zBuildings = 0;

			for (int i = 0; i < outStm_ways.size() - 3; i += 3)
			{
				if (i != 0) i -= 3;

				string wayId = outStm_ways[i];

				buildings[n_zBuildings].id = n_zBuildings;
				buildings[n_zBuildings].OS_wayId = wayId;

				buildings[n_zBuildings].buildingType = getBuildingType(wayId);

				while (outStm_ways[i] == wayId && i < outStm_ways.size() - 3)
				{
					string hashKey = outStm_ways[i + 1];
					std::unordered_map<string, int>::const_iterator got = node_buildingVertices.find(hashKey);

					if (got != node_buildingVertices.end())
					{
						if (outStm_ways[i + 3] == wayId)
						{


							string hashKey1 = outStm_ways[i + 3 + 1].c_str();
							std::unordered_map<string, int>::const_iterator got1 = node_buildingVertices.find(hashKey1);


							if (got1 != node_buildingVertices.end())
							{
								buildings[n_zBuildings].buildingGraph_edgeId.push_back(edgeConnects.size());
								buildingEdges_Building[edgeConnects.size()] = wayId;
								Building_buildingEdges[wayId] = edgeConnects.size();
								edgeConnects.push_back(got->second);

								buildings[n_zBuildings].buildingGraph_edgeId.push_back(edgeConnects.size());
								buildingEdges_Building[edgeConnects.size()] = wayId;
								Building_buildingEdges[wayId] = edgeConnects.size();
								edgeConnects.push_back(got1->second);

							}

						}
					}

					i += 3;
				}

				n_zBuildings++;
			}


			buildingGraph = zGraph(positions, edgeConnects);
			setEdgeColor(buildingGraph, edgeCol, true);
		}


		//--------------------------
		//---- GET DATA METHODS
		//--------------------------

		/*! \brief This method computes the 3D position based on the input latitude and longitude, using the bounds of the OSM data as the domain.
		*
		*	\param		[in]	lat		- input latitude.
		*	\param		[in]	lon		- input longitude.
		*	\return				zVector	- output vector.
		*	\since version 0.0.1
		*/
		zVector fromCoordinates(double &lat, double &lon)
		{
			double mappedX = ofMap(lon, lat_lon[1], lat_lon[3], minBB.x, maxBB.x);
			double mappedY = ofMap(lat, lat_lon[0], lat_lon[2], minBB.y, maxBB.y);

			//printf("\n X : %1.2f Y: %1.2f", mappedX, mappedY);

			return zVector(mappedX, mappedY, 0);
		}

		/*! \brief This method gets graph and data attributes from input shape CSV data files.
		*
		*	\param		[in]	infile_Nodes		- input file name including the directory path and extension for position information.
		*	\param		[in]	infile_Attribute	- input file name including the directory path and extension for attribute information.
		*	\param		[in]	attributeData		- container for sttribute data as a string.
		*	\param		[out]	outgraph			- out graph.
		*	\since version 0.0.1
		*/
		void fromCoordinates_ShapeCSV(string infile_Nodes, string infile_Attribute, vector<vector<string>> &attributeData, zGraph &outgraph)
		{
			attributeData.clear();

			vector<zVector> positions;
			vector<int> edgeConnects;

			unordered_map <string, int> positionVertex;

			vector<string> shapeIds;

			// nodes
			ifstream myfile;
			myfile.open(infile_Nodes.c_str());

			if (myfile.fail())
			{
				cout << " \n error in opening file  " << infile_Nodes.c_str() << endl;
				return;

			}

			while (!myfile.eof())
			{
				string str;
				getline(myfile, str);

				vector<string> perlineData = splitString(str, ",");

			
				for (int i = 0; i < perlineData.size(); i++)
				{

					if (perlineData[0] == " " || perlineData[0] == "\"shapeid\"") continue;

					string shapeId = (perlineData[0]);					

					vector<int> tempConnects;

					//printf("\n%s : ", shapeId.c_str());
					bool exit = false;									
				
					
					do
					{
						if (perlineData[0] == " ")continue;
						
			

						vector<string> x = splitString(perlineData[2], "\"");
						vector<string> y = splitString(perlineData[1], "\"");

						//printf("\n %s %s |", x[0].c_str(), y[0].c_str());

						double lat = atof(x[0].c_str());
						double lon = atof(y[0].c_str());

						// get mapped position

						zVector p0 = fromCoordinates(lat, lon);

						// check if position alread exists. 
						int v0;
						bool vExists = vertexExists(positionVertex, p0, v0);

						if (!vExists)
						{
							v0 = positions.size();
							positions.push_back(p0);

							string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
							positionVertex[hashKey] = v0;
						}

					
						tempConnects.push_back(v0);



						if (!myfile.eof())
						{
							getline(myfile, str);
							perlineData.clear();
							perlineData = splitString(str, ",");
						}
						else exit = true;

						if (myfile.eof()) exit = true;
						else
						{					
							if (perlineData[0] != shapeId) 	exit = true;
						}
							
						
							
					} while (!exit);
													

					for (int k = 0; k < tempConnects.size() - 1; k++)
					{
						edgeConnects.push_back(tempConnects[k]);
						edgeConnects.push_back(tempConnects[k + 1]);

						
					}					
									
					
				}
			}

			myfile.close();

			printf("\n positions: %i ", positions.size());
			outgraph = zGraph(positions, edgeConnects);


			// attributes

	

			myfile.open(infile_Attribute.c_str());

			if (myfile.fail())
			{
				cout << " \n error in opening file  " << infile_Attribute.c_str() << endl;
				return;

			}

			while (!myfile.eof())
			{
				string str;
				getline(myfile, str);

				vector<string> perlineData = splitString(str, ",");

				if (perlineData.size() < 1) continue;

				if (perlineData[0] == " ") continue;

				if (perlineData[0] == "\"shapeid\"") continue;
				

				vector<string> data;

				for (int i = 0; i < perlineData.size(); i++)
				{
					
					vector<string> attrib = splitString(perlineData[i], "\"");
					data.push_back(attrib[0]);

				}

				if( data.size() > 0) attributeData.push_back(data);

			}

			myfile.close();
				

		}
	

		/*! \brief This method gets data positions and attributes from input CSV shape data file .
		*
		*	\param		[in]	infile_Nodes		- input file name including the directory path and extension for position information.
		*	\param		[in]	attributeData		- container for sttribute data as a string.
		*	\param		[out]	positions			- out point cloud.
		*	\since version 0.0.1
		*/
		void fromCoordinates_ShapeCSV(string infile_Nodes,vector<vector<string>> &attributeData, vector<zVector> &positions)
		{
			attributeData.clear();
			positions.clear();
		
			// nodes
			ifstream myfile;
			myfile.open(infile_Nodes.c_str());

			if (myfile.fail())
			{
				cout << " \n error in opening file  " << infile_Nodes.c_str() << endl;
				return;

			}

			while (!myfile.eof())
			{
				string str;
				getline(myfile, str);

				vector<string> perlineData = splitString(str, ",");


				for (int i = 0; i < perlineData.size(); i++)
				{

					if (perlineData[0] == " " || perlineData[0] == "\"shapeid\"") continue;

					string shapeId = (perlineData[0]);

					vector<string> x = splitString(perlineData[2], "\"");
					vector<string> y = splitString(perlineData[1], "\"");

					//printf("\n %s %s |", x[0].c_str(), y[0].c_str());

					double lat = atof(x[0].c_str());
					double lon = atof(y[0].c_str());

					// get mapped position

					zVector p0 = fromCoordinates(lat, lon);
					
					positions.push_back(p0);

					
					if (perlineData.size() > 3)
					{
						vector<string> data;

						vector<string> shapeId = splitString(perlineData[0], "\"");
						data.push_back(shapeId[0]);

						for (int i = 3; i < perlineData.size(); i++)
						{

							vector<string> attrib = splitString(perlineData[i], "\"");
							data.push_back(attrib[0]);

						}

						if (data.size() > 0) attributeData.push_back(data);
					}

					


				}
			}

			myfile.close();
				
		}


		//--------------------------
		//---- FIELD DATA METHODS
		//--------------------------

	
		/*! \brief This method updates the scalars in the scalar field based on input graph connectivity.
		*
		*	\param		[in]	inGraph		- input graph.
		*	\since version 0.0.1
		*/
		void updateScalars_GraphConnectivity(zGraph& inGraph)
		{
			for (int i = 0; i < scalarfield.getNumScalars(); i++)
			{
				scalarfield.setWeight(0.0,i);
			}

			for (int i = 0; i < inGraph.vertexActive.size(); i++)
			{
				if (!inGraph.vertexActive[i]) continue;

				int fieldIndex = scalarfield.getIndex(inGraph.vertexPositions[i]);

				int valence = inGraph.getVertexValence((double)i, zVertexData);		
								
				scalarfield.setWeight(valence, fieldIndex);			

			}
		}
	

		/*! \brief This method updates the scalars in the scalar field based on input CSV data file with format - lat,lon,data.
		*
		*	\tparam				T			- Type to work with standard c++ datatypes of int, float,double, string. 
		*	\param		[in]	infilename	- input file name including the directory path and extension.
		*	\param		[out]	data		- output data.
		*	\since version 0.0.1
		*/		
		template <typename T>
		void updateScalars_fromCSV(string infilename, vector<T> &data);
		
				
		/*! \brief This method updates the scalars in the scalar field and gets data positions based on input CSV data file with format - lat,lon,data.
		*
		*	\tparam				T				- Type to work with standard c++ datatypes of int, float,double, string.
		*	\param		[in]	infilename		- input file name including the directory path and extension.
		*	\param		[out]	dataPositions	- output positions in the bounds of the map.
		*	\param		[out]	data			- output data.
		*	\since version 0.0.1
		*/
		template <typename T>
		void updateScalars_fromCSV(string infilename, vector<zVector> &dataPositions, vector<T> &data);
		

	};
}




//--------------------------
//---- TEMPLATE SPECIALIZATION DEFINITIONS 
//--------------------------

//---- string specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<string> &data)
{
	data.clear();

	for (int i = 0; i < scalarfield.getNumScalars(); i++)
	{
		scalarfield.setWeight(0.0, i);
	}

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");
		

		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = fromCoordinates(lat, lon);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				string inData = (perlineData[2]);

				data.push_back(inData);

				double weight;
				weight = scalarfield.getWeight(fieldIndex);
				weight++;
				
				scalarfield.setWeight(weight, fieldIndex);

			}



		}

	}

	myfile.close();
}

//---- double specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<double> &data)
{
	data.clear();

	for (int i = 0; i < scalarfield.getNumScalars(); i++)
	{
		scalarfield.setWeight(0.0, i);
	}

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");
		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = fromCoordinates(lat, lon);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;				

				scalarfield.setWeight(weight, fieldIndex);

			}



		}

	}

	myfile.close();
}

//---- float specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<float> &data)
{
	data.clear();

	for (int i = 0; i < scalarfield.getNumScalars(); i++)
	{
		scalarfield.setWeight(0.0, i);
	}

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");
		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = fromCoordinates(lat, lon);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				scalarfield.setWeight(weight, fieldIndex);

			}



		}

	}

	myfile.close();
}

//---- int specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<int> &data)
{
	data.clear();

	for (int i = 0; i < scalarfield.getNumScalars(); i++)
	{
		scalarfield.setWeight(0.0, i);
	}

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");

		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = fromCoordinates(lat, lon);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atoi(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				scalarfield.setWeight(weight, fieldIndex);

			}



		}

	}

	myfile.close();
}




//---- string specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename,  vector<zSpace::zVector> &dataPositions, vector<string> &data)
{
	dataPositions.clear();
	data.clear();

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " \n error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");

		if (perlineData.size() != 3) continue;
		if (perlineData[0] == " ") continue;		

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				
				// get mapped position
				zVector pos = fromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				string inData = (perlineData[2]);

				data.push_back(inData);

				double weight;
				weight = scalarfield.getWeight(fieldIndex);
				weight++;

				scalarfield.setWeight(weight, fieldIndex);



			}



		}

	}

	myfile.close();
}

//---- double specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<zSpace::zVector> &dataPositions, vector<double> &data)
{
	dataPositions.clear();
	data.clear();

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " \n error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");

		if (perlineData.size() != 3) continue;
		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());



			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = fromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				scalarfield.setWeight(weight, fieldIndex);



			}



		}

	}

	myfile.close();
}

//---- float specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<zSpace::zVector> &dataPositions, vector<float> &data)
{
	dataPositions.clear();
	data.clear();

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " \n error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");

		if (perlineData.size() != 3) continue;
		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());



			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = fromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				scalarfield.setWeight(weight, fieldIndex);



			}



		}

	}

	myfile.close();
}

//---- int specialization
template <>
void zSpace::zOpenStreet::updateScalars_fromCSV(string infilename, vector<zSpace::zVector> &dataPositions, vector<int> &data)
{
	dataPositions.clear();
	data.clear();

	ifstream myfile;
	myfile.open(infilename.c_str());

	if (myfile.fail())
	{
		cout << " \n error in opening file  " << infilename.c_str() << endl;
		return;

	}

	while (!myfile.eof())
	{
		string str;
		getline(myfile, str);

		vector<string> perlineData = splitString(str, ",");

		if (perlineData.size() != 3) continue;
		if (perlineData[0] == " ") continue;

		for (int i = 0; i < perlineData.size(); i++)
		{
			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());



			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = fromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex = scalarfield.getIndex(pos);

				// get data
				double inData = atoi(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				scalarfield.setWeight(weight, fieldIndex);



			}



		}

	}

	myfile.close();
}