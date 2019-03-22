#pragma once

#include<headers/IO/zJSON.h>
#include <headers/IO/zBitmap.h>

#include <headers/dynamics/zParticle.h>


namespace zSpace
{
	/** \addtogroup zIO
	*	\brief The data transfer classes and utility methods of the library.
	*  @{
	*/
	
	//--------------------------
	//---- MESH METHODS
	//--------------------------

	/** \addtogroup zIO_Mesh
	*	\brief Collection of input - output methods for zMesh.
	*  @{
	*/

	/*! \brief This method exports zMesh as an OBJ file.
	*
	*	\param [in]		inMesh				- mesh to be exported.
	*	\param [in]		infilename			- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void toOBJ(zMesh &inMesh, string outfilename)
	{

		// remove inactive elements
		if (inMesh.numVertices() != inMesh.vertexActive.size()) inMesh.removeInactiveElements(zVertexData);
		if (inMesh.numEdges() != inMesh.edgeActive.size()) inMesh.removeInactiveElements(zEdgeData);
		if (inMesh.numPolygons() != inMesh.faceActive.size()) inMesh.removeInactiveElements(zFaceData);

		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		

		// vertex positions
		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			 myfile << "\n v " << inMesh.vertexPositions[i].x << " " << inMesh.vertexPositions[i].y << " " << inMesh.vertexPositions[i].z;

		}

		// vertex nornmals
		for (int i = 0; i < inMesh.vertexNormals.size(); i++)
		{
			myfile << "\n vn " << inMesh.vertexNormals[i].x << " " << inMesh.vertexNormals[i].y << " " << inMesh.vertexNormals[i].z;

		}
		
		myfile << "\n";

		// face connectivity
		for (int i = 0; i < inMesh.numPolygons(); i++)
		{
			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			myfile << "\n f ";

			for (int j = 0; j < fVerts.size(); j++)
			{
				myfile << fVerts[j] + 1;

				if (j != fVerts.size() - 1) myfile << " ";
			}

		}

		
		

		myfile.close();

		cout << endl << " OBJ exported. File:   " << outfilename.c_str() << endl;

	}

	/*! \brief This method exports zMesh to a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inMesh				- input mesh.
	*	\param [in]		outfilename			- output file name including the directory path and extension.
	*	\param [in]		vColors				- export vertex color information if true.
	*	\since version 0.0.1
	*/
	inline void toJSON(zMesh &inMesh, string outfilename,  bool vColors = false)
	{

		// remove inactive elements
		if (inMesh.numVertices() != inMesh.vertexActive.size()) inMesh.removeInactiveElements(zVertexData);
		if (inMesh.numEdges() != inMesh.edgeActive.size()) inMesh.removeInactiveElements(zEdgeData);
		if (inMesh.numPolygons() != inMesh.faceActive.size()) inMesh.removeInactiveElements(zFaceData);

		// output file
		zMeshJSON inMeshJSON;
		json j;

		inMeshJSON.to_json(j, inMesh,  vColors);

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
		
	}


	/*! \brief This method imports zMesh from an OBJ file.
	*
	*	\param [in]		inMesh				- mesh created from the obj file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void fromOBJ(zMesh &inMesh, string infilename)
	{
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		vector<zVector>  vertexNormals;
		vector<zVector>  faceNormals;

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

			vector<string> perlineData = splitString(str, " ");

			if (perlineData.size() > 0)
			{
				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						positions.push_back(pos);
					}
					//printf("\n working vertex");
				}

				// vertex normal
				if (perlineData[0] == "vn")
				{
					//printf("\n working vertex normal ");
					if (perlineData.size() == 4)
					{
						zVector norm;
						norm.x = atof(perlineData[1].c_str());
						norm.y = atof(perlineData[2].c_str());
						norm.z = atof(perlineData[3].c_str());

						vertexNormals.push_back(norm);
					}
					//printf("\n working vertex");
				}

				// face
				if (perlineData[0] == "f")
				{

					zVector norm;



					for (int i = 1; i < perlineData.size(); i++)
					{
						vector<string> faceData = splitString(perlineData[i], "/");

						//vector<string> cleanFaceData = splitString(faceData[0], "/\/");

						int id = atoi(faceData[0].c_str()) - 1;
						polyConnects.push_back(id);

						//printf(" %i ", id);

						int normId = atoi(faceData[faceData.size() - 1].c_str()) - 1;
						norm += vertexNormals[normId];

					}

					norm /= (perlineData.size() - 1);
					norm.normalize();
					faceNormals.push_back(norm);

					polyCounts.push_back(perlineData.size() - 1);
					//printf("\n working face ");
				}
			}
		}

		myfile.close();

	
		inMesh = zMesh(positions, polyCounts, polyConnects,false);;
		printf("\n inMesh: %i %i %i", inMesh.numVertices(), inMesh.numEdges(), inMesh.numPolygons());

		setFaceNormals(inMesh,faceNormals);

		inMesh.computeMeshNormals();
	}

	/*! \brief This method imports zMesh from a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inMesh				- mesh created from the JSON file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\param [in]		vColors				- import vertex color information if true.
	*	\param [in]		fColors				- import face color information if true.
	*	\since version 0.0.1
	*/
	inline void fromJSON(zMesh &inMesh, string infilename)
	{
		json j_in;
		zMeshJSON inMeshJSON;

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

		inMeshJSON.from_json(j_in, inMesh);


		// add to maps 
		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			inMesh.addToPositionMap(inMesh.vertexPositions[i], i);
		}


		for (int i = 0; i < inMesh.numEdges(); i += 2)
		{
			int v1 = inMesh.edges[i].getVertex()->getVertexId();
			int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();

			inMesh.addToVerticesEdge(v1, v2, i);
		}
		
		// set colors

		//setVertexColor(inMesh, zColor(1, 0, 0, 1));
		//setEdgeColor(inMesh, zColor(0, 0, 0, 1));
		//setFaceColor(inMesh, zColor(0.5, 0.5, 0.5,1));
	}

	/*! \brief This method creates a mesh from the input scalar field.
	*
	*	\tparam				T				- Type to work with standard c++ numerical datatypes and zVector.
	*	\param [in]		inMesh				- mesh created from the input field.
	*	\param	[in]	inField				- input zField2D
	*	\since version 0.0.1
	*/
	template <typename T>
	void from2DFIELD(zMesh &inMesh, zField2D<T> &inField)
	{
		
		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;

		zVector minBB, maxBB;
		double unit_X, unit_Y;
		int n_X, n_Y;

		inField.getUnitDistances(unit_X, unit_Y);
		inField.getResolution(n_X, n_Y);

		inField.getBoundingBox(minBB, maxBB);

		zVector unitVec = zVector(unit_X, unit_Y, 0);
		zVector startPt = minBB;

		int resX = n_X + 1;
		int resY = n_Y + 1;

		for (int i = 0; i < resX; i++)
		{
			for (int j = 0; j < resY; j++)
			{
				zVector pos;
				pos.x = startPt.x + i * unitVec.x;
				pos.y = startPt.y + j * unitVec.y;

				positions.push_back(pos);
			}
		}



		for (int i = 0; i < resX - 1; i++)
		{
			for (int j = 0; j < resY - 1; j++)
			{
				int v0 = (i * resY) + j;
				int v1 = ((i + 1) * resY) + j;

				int v2 = v1 + 1;
				int v3 = v0 + 1;

				polyConnects.push_back(v0);
				polyConnects.push_back(v1);
				polyConnects.push_back(v2);
				polyConnects.push_back(v3);

				polyCounts.push_back(4);
			}
		}


		inMesh = zMesh(positions, polyCounts, polyConnects);;

		printf("\n scalarfieldMesh: %i %i %i", inMesh.numVertices(), inMesh.numEdges(), inMesh.numPolygons());
				
	}
	
	/** @}*/
	
	//--------------------------
	//---- GRAPH METHODS
	//--------------------------

	/** \addtogroup zIO_Graph
	*	\brief Collection of input - output methods for zGraph.
	*  @{
	*/

	/*! \brief This method imports zGraph from an TXT file.
	*
	*	\param [in]		inGraph				- graph created from the txt file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void fromTXT(zGraph &inGraph, string infilename)
	{
		vector<zVector>positions;
		vector<int>edgeConnects;


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

			vector<string> perlineData = splitString(str, " ");

			if (perlineData.size() > 0)
			{
				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						positions.push_back(pos);
					}
					//printf("\n working vertex");
				}


				// face
				if (perlineData[0] == "e")
				{

					for (int i = 1; i < perlineData.size(); i++)
					{
						int id = atoi(perlineData[i].c_str()) - 1;
						edgeConnects.push_back(id);					
					}
				}
			}
		}

		myfile.close();


		inGraph = zGraph(positions, edgeConnects);;
		printf("\n inGraph: %i %i ", inGraph.numVertices(), inGraph.numEdges());

		
	}

	/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inGraph				- graph created from the JSON file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/	
	inline void fromJSON(zGraph &inGraph, string infilename)
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

		inGraphJSON.from_json(j_in, inGraph);

		printf("\n inGraph: %i %i ", inGraph.numVertices(), inGraph.numEdges());

		// add to maps 
		for (int i = 0; i < inGraph.vertexPositions.size(); i++)
		{
			inGraph.addToPositionMap(inGraph.vertexPositions[i], i);
		}


		for (int i = 0; i < inGraph.numEdges(); i += 2)
		{
			int v1 = inGraph.edges[i].getVertex()->getVertexId();
			int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

			inGraph.addToVerticesEdge(v1, v2, i);
		}

		// set colors
		setVertexColor(inGraph, zColor(1, 0, 0, 1));
		setEdgeColor(inGraph, zColor(0, 0, 0, 1));
	}


	/*! \brief This method creates zGraph from a input zMesh.
	*
	*	\param [in]		inGraph				- graph created from the JSON file.
	*	\param [in]		inMesh				- input mesh.
	*	\since version 0.0.1
	*/
	inline void fromMESH(zGraph &inGraph, zMesh &inMesh)
	{
		
		vector<int>edgeConnects;		

		for (int i = 0; i < inMesh.numEdges(); i += 2)
		{
			edgeConnects.push_back(inMesh.edges[i + 1].getVertex()->getVertexId());
			edgeConnects.push_back(inMesh.edges[i ].getVertex()->getVertexId());
		}


		inGraph = zGraph(inMesh.vertexPositions, edgeConnects);

	}	
	

	/*! \brief This method exports zGraph to a TXT file format. 
	*
	*	\param [in]		inGraph				- input graph.
	*	\param [in]		outfilename			- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void toTXT(zGraph &inGraph, string outfilename)
	{
		// remove inactive elements
		if (inGraph.numVertices() != inGraph.vertexActive.size()) inGraph.removeInactiveElements(zVertexData);
		if (inGraph.numEdges() != inGraph.edgeActive.size()) inGraph.removeInactiveElements(zEdgeData);
		

		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		myfile << "\n ";

		// vertex positions
		for (int i = 0; i < inGraph.vertexActive.size(); i++)
		{
			if (!inGraph.vertexActive[i]) continue;

			myfile << "\n v " << inGraph.vertexPositions[i].x << " " << inGraph.vertexPositions[i].y << " " << inGraph.vertexPositions[i].z;

		}

		myfile << "\n ";

		// edge connectivity
		for (int i = 0; i < inGraph.edgeActive.size(); i+= 2)
		{
			if (!inGraph.edgeActive[i]) continue;			

			myfile << "\n e ";
			
			myfile << inGraph.edges[i].getVertex()->getVertexId() << " ";
			myfile << inGraph.edges[i].getVertex()->getVertexId();			

		}

		myfile << "\n ";

		myfile.close();

		cout << endl << " TXT exported. File:   " << outfilename.c_str() << endl;
	}

	/*! \brief This method exports zGraph to a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inGraph				- input graph.
	*	\param [in]		outfilename			- output file name including the directory path and extension.
	*	\param [in]		vColors				- export vertex color information if true.
	*	\since version 0.0.1
	*/
	inline void toJSON(zGraph &inGraph, string outfilename, bool vColors = false )
	{
		// remove inactive elements
		if (inGraph.numVertices() != inGraph.vertexActive.size()) inGraph.removeInactiveElements(zVertexData);
		if (inGraph.numEdges() != inGraph.edgeActive.size()) inGraph.removeInactiveElements(zEdgeData);


		// output file
		zGraphJSON inGraphJSON;
		json j;

		inGraphJSON.to_json(j, inGraph, vColors);

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
	}



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

	/** @}*/


	//--------------------------
	//---- STREAM METHODS
	//--------------------------

	/** \addtogroup zIO_Streams
	*	\brief Collection of input - output methods for particles.
	*  @{
	*/
	
	/*! \brief This method exports zStream to a JSON file format using JSON Modern Library.
	*
	*	\param		[in]	inStream				- input zStream.
	*	\param		[in]	outfilename				- output file name including the directory path and extension.
	*	\param		[in]	vColors					- export vertex color information if true.
	*	\since version 0.0.1
	*/
	inline void toJSON(zStream &inStream, string outfilename, bool vColors = false)
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
	}


	/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inStream			- stream to be created from the JSON file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void fromJSON(zStream &inStream, string infilename)
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

		for (int i = 0; i < closestStream_Point.size(); i ++)
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
	}
		

	/** @}*/

	//--------------------------
	//---- PARTICLES METHODS
	//--------------------------

	/** \addtogroup zIO_Dynamics
	*	\brief Collection of input - output methods for particles.
	*  @{
	*/

	/*! \brief This method creates a container of particles with the positions initiaised at the input positions.
	*
	*	\param		[in]	inPartices				- container of particles created from the input positions.
	*	\param		[in]	inPoints				- input container of positions.
	*	\param		[in]	fixed					- input contatiner indicating if a particle is active or fixed.
	*	\param		[in]	clear					- true if the input contatiner of particle is to be cleared.
	*	\since version 0.0.1
	*/
	inline void fromPOSITIONS(vector<zParticle> &inParticles, vector<zVector> &inPoints, vector<bool> fixed, bool clear = true)
	{
		if(fixed.size() > 0 && fixed.size()!= inPoints.size() ) throw std::invalid_argument(" error: size of inPoints and active dont match.");

		if(clear) inParticles.clear();

		for (int i = 0; i < inPoints.size(); i++)
		{
			bool pActive = (fixed.size() > 0) ? fixed[i] : true;

			inParticles.push_back(zParticle(&inPoints[i], pActive));
		}		
	}

	/*! \brief This method creates a container of particles with the positions initiaised at the mesh vertex positions.
	*
	*	\param		[in]	inPartices				- container of particles created from the input positions.
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	fixBoundary				- true if the boundary vertices are to be fixed.
	*	\param		[in]	clear					- true if the input contatiner of particle is to be cleared.
	*	\since version 0.0.1
	*/
	inline void fromMESH(vector<zParticle> &inParticles, zMesh &inMesh, bool fixBoundary = false, bool clear = true)
	{
		
		if (clear) inParticles.clear();

		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			bool fixed = false;

			if (fixBoundary) fixed = (inMesh.onBoundary(i, zVertexData)) ;

			inParticles.push_back(zParticle(&inMesh.vertexPositions[i], fixed));

			if (!fixed) setVertexColor( inMesh, zColor(0, 0, 1, 1));
		}

		
	}

	/*! \brief This method creates a container of particles with the positions initiaised at the graph vertex positions.
	*
	*	\param		[in]	inPartices				- container of particles created from the input positions.
	*	\param		[in]	inGraph					- input graph.
	*	\param		[in]	fixBoundary				- true if the boundary vertices are to be fixed.
	*	\param		[in]	clear					- true if the input contatiner of particle is to be cleared.
	*	\since version 0.0.1
	*/
	inline void fromGRAPH(vector<zParticle> &inParticles, zGraph &inGraph, bool fixBoundary = false, bool clear = true)
	{

		if (clear) inParticles.clear();

		for (int i = 0; i < inGraph.vertexPositions.size(); i++)
		{
			bool fixed = false;

			if (fixBoundary) fixed = inGraph.checkVertexValency(i,1);

			inParticles.push_back(zParticle(&inGraph.vertexPositions[i], fixed));

			if (!fixed) setVertexColor(inGraph, zColor(0, 0, 1, 1));
		}


	}

	/** @}*/




	//--------------------------
	//---- POINT CLOUD METHODS
	//--------------------------

	/** \addtogroup zIO_PointCloud
	*	\brief Collection of input - output methods for point clouds.
	*  @{
	*/

	/*! \brief This method imports a point cloud from an TXT file.
	*
	*	\param [in]		inPositions			- container of positions created from the txt file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void fromTXT(vector<zVector> &inPositions, string infilename)
	{
		inPositions.clear();

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

			vector<string> perlineData = splitString(str, " ");

			if (perlineData.size() > 0)
			{
				// vertex
				if (perlineData[0] == "v")
				{
					if (perlineData.size() == 4)
					{
						zVector pos;
						pos.x = atof(perlineData[1].c_str());
						pos.y = atof(perlineData[2].c_str());
						pos.z = atof(perlineData[3].c_str());

						inPositions.push_back(pos);
					}
					//printf("\n working vertex");
				}

				
			}
		}

		myfile.close();


		
		printf("\n inPositions: %i ", inPositions.size());


	}

	/*! \brief This method exports the input point cloud to a TXT file format.
	*
	*	\param [in]		inPositions			- input container of position.
	*	\param [in]		outfilename			- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	inline void toTXT(vector<zVector> &inPositions, string outfilename)
	{
		


		// output file
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		// vertex positions
		for (int i = 0; i < inPositions.size(); i++)
		{
			
			myfile << "\n v " << inPositions[i].x << " " << inPositions[i].y << " " << inPositions[i].z;

		}

		myfile.close();

		cout << endl << " TXT exported. File:   " << outfilename.c_str() << endl;
	}

	/** @}*/


	//--------------------------
	//---- 2D FIELD METHODS
	//--------------------------

	/** \addtogroup zIO_Field2D
	*	\brief Collection of input - output methods for zField2D.
	*  @{
	*/

	/*! \brief This method exports the input field to a bitmap file format based on the face color of the correspoding field mesh.
	*
	*	\tparam			T				- Type to work with double and zVector.
	*	\param [in]		inField			- input field .
	*	\param [in]		inFeldMesh		- input field mesh.
	*	\param [in]		outfilename		- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	template <typename T>
	void toBMP(zField2D<T> &inField, zMesh &inFieldMesh, string outfilename)
	{
		int resX, resY;
		inField.getResolution(resX, resY);

		BMP bmp(resX, resY);

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{
				int faceId; 
				inField.getIndex(x, y, faceId);

				// blue
				bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 0] = inFieldMesh.faceColors[faceId].b * 255;

				// green
				bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 1] = inFieldMesh.faceColors[faceId].g * 255;

				// red
				bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 2] = inFieldMesh.faceColors[faceId].r * 255;

				// alpha
				if (channels == 4)
				{
					bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 3] = inFieldMesh.faceColors[faceId].a * 255;
				}

			}

		}

		bmp.write(outfilename.c_str());
	}


	/*! \brief This method imorts the input bitmap file and creates the corresponding field and  field mesh. The Bitmap needs to be in grey-scale colors only to update field values.
	*
	*	\tparam					T				- Type to work with double and zVector.
	*	\param		[in]		inField			- input field .
	*	\param		[in]		inFeldMesh		- input field mesh.
	*	\param		[in]		minBB			- minimum bounds of the field.
	*	\param		[in]		maxBB			- maximum bounds of the field.
	*	\param		[in]		infilename		- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	template <typename T>
	void fromBMP(zField2D<T> &inField, zMesh &inFieldMesh, zVector &minBB, zVector &maxBB,  string infilename )
	{

		BMP bmp(infilename.c_str());

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		int resX = bmp.bmp_info_header.width;
		int resY = bmp.bmp_info_header.height;
		
		if (resX == 0 || resY == 0) return;

		inField = zField2D<T>(minBB, maxBB, resX, resY);
		from2DFIELD(inFieldMesh, inField);

		

		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{
				int faceId;
				inField.getIndex(x, y, faceId);

				//printf("\n %i %i %i ", x, y, faceId);

				// blue
				double b  = (double) bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 0] / 255;
		
				// green
				double g = (double) bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 1] /  255;

				// red
				double r = (double) bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 2] / 255;

				inFieldMesh.faceColors[faceId] = zColor(r, g, b, 1);

				// alpha
				if (channels == 4)
				{
					double a  = (double) bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 3] / 255;

					inFieldMesh.faceColors[faceId] =  zColor(r, g, b, a) ;
				}

				inField.setFieldValue(r, faceId);
				
			}
		}
				
	}

	/** @}*/

	/** @}*/
}


//--------------------------
//---- TEMPLATE SPECIALIZATION DEFINITIONS 
//--------------------------


//---- string specialization
template <>
inline void zSpace::fromCSV(string infilename, zSpace::zHEData type, zSpace::zGraph& inGraph, vector<string> &data)
{
	data.clear();

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

		if (perlineData.size() != 2) continue;

		// get data
		string inData = (perlineData[1]);

		data.push_back(inData);

	}
	
	myfile.close();

	if (type == zVertexData)
	{
		if (data.size() != inGraph.vertexActive.size())
		{
			
			throw std::invalid_argument(" error: data size and number of vertices dont match.");
		}

	}


	if (type == zEdgeData)
	{
		if (data.size() != inGraph.edgeActive.size())
		{
			throw std::invalid_argument(" error: data size and number of edges dont match.");
		}
	}




}

//---- int specialization
template <>
inline void zSpace::fromCSV(string infilename, zSpace::zHEData type, zSpace::zGraph& inGraph, vector<int> &data)
{
	data.clear();

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

		if (perlineData.size()!= 2) continue;

		// get data
		int inData = atoi(perlineData[1].c_str());

		data.push_back(inData);

	}
	myfile.close();

	if (type == zVertexData)
	{
		if (data.size() != inGraph.vertexActive.size())
		{
			throw std::invalid_argument(" error: data size and number of vertices dont match.");
		}

	}


	if (type == zEdgeData)
	{
		if (data.size() != inGraph.edgeActive.size())
		{
			throw std::invalid_argument(" error: data size and number of edges dont match.");
		}
	}

}


//---- float specialization
template <>
inline void zSpace::fromCSV(string infilename, zSpace::zHEData type, zSpace::zGraph& inGraph, vector<float> &data)
{
	data.clear();

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

		if (perlineData.size() != 2) continue;

		// get data
		float inData = atof(perlineData[1].c_str());

		data.push_back(inData);

	}
	myfile.close();

	if (type == zVertexData)
	{
		if (data.size() != inGraph.vertexActive.size())
		{
			throw std::invalid_argument(" error: data size and number of vertices dont match.");
		}

	}


	if (type == zEdgeData)
	{
		if (data.size() != inGraph.edgeActive.size())
		{
			throw std::invalid_argument(" error: data size and number of edges dont match.");
		}
	}


}

//---- double specialization
template <>
inline void zSpace::fromCSV(string infilename, zSpace::zHEData type, zSpace::zGraph& inGraph, vector<double> &data)
{
	data.clear();

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

		if (perlineData.size() != 2) continue;

		// get data
		double inData = atof(perlineData[1].c_str());

		data.push_back(inData);

	}

	myfile.close();

	if (type == zVertexData)
	{
		if (data.size() != inGraph.vertexActive.size())
		{
			throw std::invalid_argument(" error: data size and number of vertices dont match.");
		}

	}


	if (type == zEdgeData)
	{
		if (data.size() != inGraph.edgeActive.size())
		{
			throw std::invalid_argument(" error: data size and number of edges dont match.");
		}
	}


	

}
