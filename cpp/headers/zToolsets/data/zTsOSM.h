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

#ifndef ZSPACE_TS_DATA_OSM_H
#define ZSPACE_TS_DATA_OSM_H

#pragma once

#include <headers/zCore/data/zDatabase.h>

#include <headers/zInterface/functionsets/zFnMeshField.h>
#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>


namespace zSpace
{

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsData
	*	\brief toolsets for data related utilities.
	*  @{
	*/

	/** \addtogroup zOSM
	*	\brief The data classes and structs for open street map toolset.
	*  @{
	*/

	/*! \struct zWays
	*	\brief A struct for storing  information of OSM ways and street graph.
	*	\since version 0.0.2
	*/

	/** @}*/

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

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsData
	*	\brief toolsets for data related utilities.
	*  @{
	*/

	/** \addtogroup zOSM
	*	\brief The data classes and structs for open street map toolset.
	*  @{
	*/

	/*! \struct zBuildings
	*	\brief A struct for storing building related data. 
	*	\since version 0.0.2
	*/

	/** @}*/

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

		double height;

		string postcode;
		

	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsData
	*	\brief toolsets for data related utilities.
	*  @{
	*/

	/** \addtogroup zOSM
	*	\brief The data classes and structs for open street map toolset.
	*  @{
	*/

	/*! \struct zPostcode
	*	\brief A struct for storing postcode related data. 
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zPostcode
	{
		string postcode;

		vector<double> residentialPrices;

		zVector point;
	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsData
	*	\brief toolsets for data related utilities.
	*  @{
	*/

	/** \addtogroup zOSM
	*	\brief The data classes and structs for open street map toolset.
	*  @{
	*/

	/*! \class zTsOSM
	*	\brief A class for accessing and visualizing the openstreet data and other city level data stored in a SQL database , CSV files etc.
	*	\details Uses open source data from https://www.openstreetmap.org
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	
	class ZSPACE_TOOLS zTsOSM
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to a 2D field  */
		zObjMeshScalarField  *fieldObj;

		//--------------------------
		//---- STREET ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a street graph object  */
		zObjGraph *streetObj;

		/*! \brief stores current number of ways		*/
		int n_zWays;

		/*! \brief zWays container		*/
		zWays *way;

		/*!	\brief street graph edgeId to wayId map. Used to get correponding wayId given the street graph EdgeId.  */
		map <int, string> streetEdges_Way;

		/*!	\brief OSMwayId to wayId. Used to get correponding street graph edges given the osm way id.  */
		unordered_map <string, int> OSMwaysID_zWayId;

		//--------------------------
		//---- BUILDING ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a building graph object  */
		
		zObjGraph *buildingGraphObj;

		zObjMesh *buildingObj;

		/*! \brief stores current number of buildings		*/
		int n_zBuildings;

		/*! \brief zBuildings container		*/
		zBuildings *buildings;

		/*!	\brief building graph edgeId to wayId map. Used to get correponding wayId given the building graph EdgeId.  */
		map <int, string> buildingEdges_Building;

		/*!	\brief OSMwayId to zbuildingId. Used to get correponding building graph edges given the osm way id.  */
		unordered_map <string, int> OSMwaysID_zBuildingsId;

		//--------------------------
		//---- POSTCODE ATTRIBUTES
		//--------------------------
		unordered_map <string, int> postcode_zPostcodeId;

		//--------------------------
		//---- BOUNDS ATTRIBUTES
		//--------------------------
		
		/*!	\brief minimum bounds of OSM data in 3D space given by a zVector.  */
		zVector minBB;

		/*!	\brief maximum bounds of OSM data in 3D space given by a zVector.  */
		zVector maxBB;

		//--------------------------
		//---- DATABASE ATTRIBUTES
		//--------------------------

		/*!	\brief database needed to acces the OSM and other data.  */
		zDatabase *zDB;

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief bounds of OSM data in latitute and logitude.  */
		double lat_lon[4];

		double scaleFactor;

		/*!	\brief field function set  */
		zFnMeshField<zScalar> fnField;

		/*!	\brief graph function set for streets */
		zFnGraph fnStreet;

		/*!	\brief graph function set for buildings */
		zFnGraph fnGraphBuilding;
		zFnMesh fnBuilding;

		/*!	\brief 2 dimensional container of stream positions per field index.  */
		vector<vector<zVector>> fieldIndex_streamPositions;

		vector<zVector> buildingCenters;

		vector<zPostcode> postcodes;

		vector<zVector> tubeStations;

		vector<zVector> parks;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsOSM();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	DatabaseFileName		- file path to the SQL database.
		*	\since version 0.0.1
		*/
		zTsOSM(char* DatabaseFileName, zObjMeshField<zScalar> &_fieldObj, zObjGraph &_streetObj, zObjMesh &_buildingObj, zObjGraph &_buildingGraphObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsOSM();

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------

		/*! \brief This method sets the scale factor.
		*
		*	\param		[in]	scaleFactor		- scale of the map to be displayed.
		*	\since version 0.0.1
		*/
		void setScale(double _scaleFactor);

		/*! \brief This method computes the bounding box in 3D space of the OSM data from the  lat_lon container of the bounds.
		*
		*	\since version 0.0.1
		*/
		void computeBoundingBox();

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
		double computeDistance(double &lat1, double &lon1, double &lat2, double &lon2);

		/*! \brief This method computes the 3D position based on the input latitude and longitude, using the bounds of the OSM data as the domain.
		*
		*	\param		[in]	lat		- input latitude.
		*	\param		[in]	lon		- input longitude.
		*	\return				zVector	- output vector.
		*	\since version 0.0.1
		*/
		zVector computePositionFromCoordinates(double &lat, double &lon);

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method create the scalar field from the bounds and input resolution.
		*
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*/
		void createFieldFromBounds(int _n_X = 100, int _n_Y = 100);

		/*! \brief This method creates the streets from the OSM data.
		*
		*	\param		[in]	edgeCol		- input color to be assigned to the edges of the graph.
		*	\since version 0.0.1
		*/
		void createStreets(zColor streetCol = zColor(0, 0, 0, 1));

		/*! \brief This method creates the building graph from the OSM data.
		*
		*	\param		[in]	edgeCol		- input color to be assigned to the edges of the graph.
		*	\since version 0.0.1
		*/
		void createGraphBuildings(zColor buildingCol = zColor(0, 0, 0, 1));

		void exportGraphBuildings(string outfile);

		void exportPartGraphBuildings(string outfile);

		void createBuildings(zColor buildingCol = zColor(0, 0, 0, 1));

		void getPostcodesPrices();

		void getTubeStations();

		void exportTubeStations(string outfile);

		void getParks();

		void exportParks(string outfile);

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the street type from the OSM data for the given wayId if it corresponds to a street.
		*
		*	\param		[in]	wayId		- input wayId.
		*	\since version 0.0.1
		*/
		zDataStreet getStreetType(string wayId);

		/*! \brief This method extracts the edges from the street graph based on the input key and value for OSM ways tags.
		*
		*	\param		[in]	k			- key of the relations tag.
		*	\param		[in]	v			- value of the relations tag.
		*	\param		[out]	graphEdges	- edge indicies correponding to k and v.
		*	\since version 0.0.1
		*/
		void getStreetEdgesFromWays(string k, string v, vector<int> &graphEdges);

		/*! \brief This method extracts the edges from the street graph based on the input key and value for OSM relations tags.
		*
		*	\param		[in]	k			- key of the relations tag.
		*	\param		[in]	v			- value of the relations tag.
		*	\param		[out]	graphEdges	- edge indicies correponding to k and v.
		*	\since version 0.0.1
		*/
		void getStreetEdgesFromRelations(string k, string v, vector<int> &graphEdges);

		/*! \brief This method gets the street type from the OSM data for the given wayId if it corresponds to a street.
		*
		*	\param		[in]	wayId		- input wayId.
		*	\since version 0.0.1
		*/
		zDataBuilding getBuildingType(string wayId);

		//--------------------------
		//---- DATA FACTORY METHODS
		//--------------------------

		/*! \brief This method gets graph and data attributes from input shape CSV data files.
		*
		*	\param		[in]	infile_Nodes		- input file name including the directory path and extension for position information.
		*	\param		[in]	infile_Attribute	- input file name including the directory path and extension for attribute information.
		*	\param		[in]	attributeData		- container for sttribute data as a string.
		*	\since version 0.0.1
		*/
		zObjGraph fromShapeCSV(string infile_Nodes, string infile_Attribute, vector<vector<string>> &attributeData);

		/*! \brief This method gets mesh and data attributes from input shape CSV data files.
		*
		*	\param		[in]	infile_Nodes		- input file name including the directory path and extension for position information.
		*	\param		[in]	infile_Attribute	- input file name including the directory path and extension for attribute information.
		*	\param		[in]	attributeData		- container for sttribute data as a string.
		*	\since version 0.0.1
		*/
		zObjMesh fromCoordinates_ShapeCSV(string infile_Nodes, string infile_Attribute, vector<vector<string>> &attributeData);

		//--------------------------
		//---- FIELD DATA METHODS
		//--------------------------

		/*! \brief This method updates the scalars in the scalar field based on input graph connectivity.
		*
		*	\param		[in]	inGraph		- input graph.
		*	\since version 0.0.1
		*/
		void updateScalars_GraphConnectivity(zObjGraph& inGraph);

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

		/*! \brief This method exports the relations from an OSM file to 3 CSV files.
		*
		*	\param		[in]	infilename						- input file name including the directory path and extension.
		*	\param		[out]	outfilename_Relations			- output CSV file of relation ids.
		*	\param		[out]	outfilename_Relations_members	- output CSV file of relation members.
		*	\param		[out]	outfilename_RelationTags		- output CSV file of relation tags.
		*	\since version 0.0.1
		*/
		void exportOSMRelationsToCSV(string infilename, string outfilename_Relations, string outfilename_Relations_members, string outfilename_RelationTags);


	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- string specialization for updateScalars_fromCSV 
	template<>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<string>& data)
	{
		data.clear();

		int i = 0;

		for (zItMeshScalarField s(*fnField.fieldObj); !s.end(); s++, i++)
		{
			s.setValue(0.0, i);
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				string inData = (perlineData[2]);

				data.push_back(inData);

				double weight =  s.getValue();
				weight++;

				s.setValue(weight, fieldIndex);

			}

		}

		myfile.close();
	}
	
	//---- double specialization for updateScalars_fromCSV 
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<double> &data)
	{
		data.clear();

		int i = 0;
		for (zItMeshScalarField s(*fnField.fieldObj); !s.end(); s++, i++)
		{
			s.setValue(0.0, i);
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);

				// get field index
				int fieldIndex;;

				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight = inData;
				s.setValue(weight, fieldIndex);

			}







		}

		myfile.close();
	}

	//---- float specialization for updateScalars_fromCSV 
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<float> &data)
	{
		data.clear();

		int i = 0;
		for (zItMeshScalarField s(*fnField.fieldObj); !s.end(); s++, i++)
		{
			s.setValue(0.0, i);
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight = inData;
				s.setValue(weight, fieldIndex);

			}


		}

		myfile.close();
	}

	//---- int specialization for updateScalars_fromCSV 
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<int> &data)
	{
		data.clear();

		int i = 0;
		for (zItMeshScalarField s(*fnField.fieldObj); !s.end(); s++, i++)
		{
			s.setValue(0.0, i);
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{
				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atoi(perlineData[2].c_str());

				data.push_back(inData);

				double weight = inData;			

				s.setValue(weight, fieldIndex);

			}

		}

		myfile.close();
	}

	//---------------//

	//---- string specialization for updateScalars_fromCSV  with datapositions
	template<>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<zVector>& dataPositions, vector<string>& data)
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());



			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				string inData = (perlineData[2]);

				data.push_back(inData);
				
				double weight = s.getValue();
				weight++;

				s.setValue(weight, fieldIndex);



			}

		}

		myfile.close();
	}

	//---- double specialization for updateScalars_fromCSV  with datapositions
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<zVector> &dataPositions, vector<double> &data)
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				s.setValue(weight);
			}






		}

		myfile.close();
	}

	//---- float specialization for updateScalars_fromCSV  with datapositions
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<zVector> &dataPositions, vector<float> &data)
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;

			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());



			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atof(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				s.setValue(weight, fieldIndex);


			}



		}

		myfile.close();
	}

	//---- int specialization for updateScalars_fromCSV  with datapositions
	template <>
	inline void zTsOSM::updateScalars_fromCSV(string infilename, vector<zVector> &dataPositions, vector<int> &data)
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

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() != 3) continue;
			if (perlineData[0] == " ") continue;


			double lat = atof(perlineData[0].c_str());
			double lon = atof(perlineData[1].c_str());

			if (lat >= lat_lon[0] && lat <= lat_lon[2] && lon >= lat_lon[1] && lon <= lat_lon[3])
			{

				// get mapped position
				zVector pos = computePositionFromCoordinates(lat, lon);
				dataPositions.push_back(pos);

				// get field index
				int fieldIndex;
				bool boundsCheck = fnField.checkPositionBounds(pos, fieldIndex);

				if (boundsCheck) continue;

				zItMeshScalarField s(*fnField.fieldObj, fieldIndex);

				// get data
				double inData = atoi(perlineData[2].c_str());

				data.push_back(inData);

				double weight;
				weight = inData;;

				s.setValue(weight);

			}

		}

		myfile.close();
	}

	
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/data/zTsOSM.cpp>
#endif

#endif