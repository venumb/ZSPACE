#pragma once




#include <headers/dynamics/zParticle.h>


namespace zSpace
{
	/** \addtogroup zIO
	*	\brief The data transfer classes and utility methods of the library.
	*  @{
	*/

	/*! \class zIOFactory
	*	\brief A file transfer factory class. 
	*	\since version 0.0.1
	*/

	/** @}*/

	class zIOFactory
	{


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zIOFactory() {}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zIOFactory() {}
				




		//--------------------------
		//---- GRAPH ATTRIBUTE METHODS
		//--------------------------

		/*! \brief This method exports the graph attribute data to a CSV file.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	outfilename		- output file name including the directory path and extension.
		*	\param		[in]	type			- zVertexData / zEdgedata
		*	\param		[in]	inGraph			- input graph.
		*	\param		[out]	data			- output data.
		*	\since version 0.0.1
		*/
		template <typename T>
		inline void toCSV(string outfilename, zHEData type, zGraph& inGraph, vector<T> &data)
		{

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				return;

			}

			if (type == zVertexData)
			{
				if (data.size() != inGraph.vertexActive.size())
				{
					myfile.close();
					throw std::invalid_argument(" error: data size and number of vertices dont match.");
				}



				myfile << "\n ";

				// vertex 
				for (int i = 0; i < inGraph.vertexActive.size(); i++)
				{
					myfile << "\n " << i << "," << data[i];

				}

			}

			if (type == zEdgeData)
			{
				if (data.size() != inGraph.edgeActive.size())
				{
					myfile.close();
					throw std::invalid_argument(" error: data size and number of edges dont match.");
				}



				myfile << "\n ";

				// edge 
				for (int i = 0; i < inGraph.edgeActive.size(); i++)
				{
					myfile << "\n " << i << "," << data[i];

				}

			}

			myfile.close();
		}


		/*! \brief This method imports the graph attribute data from a CSV file.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	infilename		- input file name including the directory path and extension.
		*	\param		[in]	type			- zVertexData / zEdgedata
		*	\param		[in]	inGraph			- input graph.
		*	\param		[out]	data			- input data.
		*	\since version 0.0.1
		*/
		template <typename T>
		inline void fromCSV(string infilename, zHEData type, zGraph& inGraph, vector<T> &data);

		//--------------------------
		//---- STREAM LINES METHODS
		//--------------------------
		
		/*! \brief This method exports zStream to a JSON file format using JSON Modern Library.
		*
		*	\param		[in]	inStream				- input zStream.
		*	\param		[in]	outfilename				- output file name including the directory path and extension.
		*	\param		[in]	vColors					- export vertex color information if true.
		*	\since version 0.0.1
		*/
		/*inline void toJSON(zStream &inStream, string outfilename, bool vColors = false)
		{
			// remove inactive elements
			if (inStream.streamGraph.numVertices() != inStream.streamGraph.vertexActive.size()) inStream.streamGraph.removeInactiveElements(zVertexData);
			if (inStream.streamGraph.numEdges() != inStream.streamGraph.edgeActive.size()) inStream.streamGraph.removeInactiveElements(zEdgeData);


			// output file
			zGraphJSON inGraphJSON;
			json j;

			inGraphJSON.to_json(j, inStream.streamGraph, vColors);

			// add stream attributes;
			vector<vector<int>> closestStream_Attributes;
			vector<vector<double>> closestStream_Point;

			for (int i = 0; i < inStream.closestStream.size(); i += 2)
			{
				vector<int> stream_attrib;

				stream_attrib.push_back(inStream.closestStream[i]);
				stream_attrib.push_back(inStream.closestStream[i + 1]);

				stream_attrib.push_back(inStream.closestStream_Edge[i]);
				stream_attrib.push_back(inStream.closestStream_Edge[i + 1]);



				closestStream_Attributes.push_back(stream_attrib);
			}

			for (int i = 0; i < inStream.closestStream_Point.size(); i += 2)
			{
				vector<double> stream_closestPoint_attrib;

				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i].x);
				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i].y);
				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i].z);

				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i + 1].x);
				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i + 1].y);
				stream_closestPoint_attrib.push_back(inStream.closestStream_Point[i + 1].z);

				closestStream_Point.push_back(stream_closestPoint_attrib);
			}

			// Json file 
			j["ClosestStreams"] = closestStream_Attributes;
			j["ClosestStreamPoints"] = closestStream_Point;

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				return;
			}

			//myfile.precision(16);
			myfile << j.dump();
			myfile.close();
		}*/


		/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		inStream			- stream to be created from the JSON file.
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		/*inline void fromJSON(zStream &inStream, string infilename)
		{
			json j_in;
			zGraphJSON inGraphJSON;

			ifstream in_myfile;
			in_myfile.open(infilename.c_str());

			int lineCnt = 0;

			if (in_myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;
			}



			in_myfile >> j_in;
			in_myfile.close();

			inGraphJSON.from_json(j_in, inStream.streamGraph);

			// stream attributes

			vector<vector<int>> closestStream_Attributes;
			vector<vector<double>> closestStream_Point;


			closestStream_Attributes = (j_in["ClosestStreams"].get<vector<vector<int>>>());
			closestStream_Point = ((j_in["ClosestStreamPoints"].get<vector<vector<double>>>()));

			for (int i = 0; i < closestStream_Point.size(); i++)
			{
				for (int k = 0; k < closestStream_Point[i].size(); k++)
				{
					// position
					if (closestStream_Point[i].size() == 6)
					{
						zVector pos1(closestStream_Point[i][k + 0], closestStream_Point[i][k + 1], closestStream_Point[i][k + 2]);
						zVector pos2(closestStream_Point[i][k + 3], closestStream_Point[i][k + 4], closestStream_Point[i][k + 5]);

						inStream.closestStream_Point.push_back(pos1);
						inStream.closestStream_Point.push_back(pos2);
						k += 5;
					}
				}

			}

			for (int i = 0; i < closestStream_Attributes.size(); i++)
			{
				for (int k = 0; k < closestStream_Attributes[i].size(); k++)
				{
					if (closestStream_Attributes[i].size() == 4)
					{
						inStream.closestStream.push_back(closestStream_Attributes[i][k + 0]);
						inStream.closestStream.push_back(closestStream_Attributes[i][k + 1]);

						inStream.closestStream_Edge.push_back(closestStream_Attributes[i][k + 2]);
						inStream.closestStream_Edge.push_back(closestStream_Attributes[i][k + 3]);

						k += 3;
					}
				}
			}


			printf("\n inStream: %i %i ", inStream.streamGraph.numVertices(), inStream.streamGraph.numEdges());
		}*/


		
	



	};
	
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- string specialization for fromCSV
	template <>
	inline void zIOFactory::fromCSV(string infilename, zHEData type, zGraph& inGraph, vector<string> &data)
	{
		//data.clear();

		//ifstream myfile;
		//myfile.open(infilename.c_str());

		//if (myfile.fail())
		//{
		//	cout << " error in opening file  " << infilename.c_str() << endl;
		//	return;

		//}


		//while (!myfile.eof())
		//{
		//	string str;
		//	getline(myfile, str);

		//	vector<string> perlineData = splitString(str, ",");

		//	if (perlineData.size() != 2) continue;

		//	// get data
		//	string inData = (perlineData[1]);

		//	data.push_back(inData);

		//}

		//myfile.close();

		//if (type == zVertexData)
		//{
		//	if (data.size() != inGraph.vertexActive.size())
		//	{

		//		throw std::invalid_argument(" error: data size and number of vertices dont match.");
		//	}

		//}


		//if (type == zEdgeData)
		//{
		//	if (data.size() != inGraph.edgeActive.size())
		//	{
		//		throw std::invalid_argument(" error: data size and number of edges dont match.");
		//	}
		//}




	}

	//---- int specialization for fromCSV
	template <>
	inline void zIOFactory::fromCSV(string infilename, zHEData type, zGraph& inGraph, vector<int> &data)
	{
		//data.clear();

		//ifstream myfile;
		//myfile.open(infilename.c_str());

		//if (myfile.fail())
		//{
		//	cout << " error in opening file  " << infilename.c_str() << endl;
		//	return;

		//}

		//while (!myfile.eof())
		//{
		//	string str;
		//	getline(myfile, str);

		//	vector<string> perlineData = splitString(str, ",");

		//	if (perlineData.size() != 2) continue;

		//	// get data
		//	int inData = atoi(perlineData[1].c_str());

		//	data.push_back(inData);

		//}
		//myfile.close();

		//if (type == zVertexData)
		//{
		//	if (data.size() != inGraph.vertexActive.size())
		//	{
		//		throw std::invalid_argument(" error: data size and number of vertices dont match.");
		//	}

		//}


		//if (type == zEdgeData)
		//{
		//	if (data.size() != inGraph.edgeActive.size())
		//	{
		//		throw std::invalid_argument(" error: data size and number of edges dont match.");
		//	}
		//}

	}

	//---- float specialization for fromCSV
	template <>
	inline void zIOFactory::fromCSV(string infilename, zHEData type, zGraph& inGraph, vector<float> &data)
	{
		//data.clear();

		//ifstream myfile;
		//myfile.open(infilename.c_str());

		//if (myfile.fail())
		//{
		//	cout << " error in opening file  " << infilename.c_str() << endl;
		//	return;

		//}

		//while (!myfile.eof())
		//{
		//	string str;
		//	getline(myfile, str);

		//	vector<string> perlineData = splitString(str, ",");

		//	if (perlineData.size() != 2) continue;

		//	// get data
		//	float inData = atof(perlineData[1].c_str());

		//	data.push_back(inData);

		//}
		//myfile.close();

		//if (type == zVertexData)
		//{
		//	if (data.size() != inGraph.vertexActive.size())
		//	{
		//		throw std::invalid_argument(" error: data size and number of vertices dont match.");
		//	}

		//}


		//if (type == zEdgeData)
		//{
		//	if (data.size() != inGraph.edgeActive.size())
		//	{
		//		throw std::invalid_argument(" error: data size and number of edges dont match.");
		//	}
		//}


	}

	//---- double specialization for fromCSV
	template <>
	inline void zIOFactory::fromCSV(string infilename, zHEData type, zGraph& inGraph, vector<double> &data)
	{
	//	data.clear();

	//	ifstream myfile;
	//	myfile.open(infilename.c_str());

	//	if (myfile.fail())
	//	{
	//		cout << " error in opening file  " << infilename.c_str() << endl;
	//		return;

	//	}

	//	while (!myfile.eof())
	//	{
	//		string str;
	//		getline(myfile, str);

	//		vector<string> perlineData = splitString(str, ",");

	//		if (perlineData.size() != 2) continue;

	//		// get data
	//		double inData = atof(perlineData[1].c_str());

	//		data.push_back(inData);

	//	}

	//	myfile.close();

	//	if (type == zVertexData)
	//	{
	//		if (data.size() != inGraph.vertexActive.size())
	//		{
	//			throw std::invalid_argument(" error: data size and number of vertices dont match.");
	//		}

	//	}


	//	if (type == zEdgeData)
	//	{
	//		if (data.size() != inGraph.edgeActive.size())
	//		{
	//			throw std::invalid_argument(" error: data size and number of edges dont match.");
	//		}
	//	}




	}

	//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */


}





