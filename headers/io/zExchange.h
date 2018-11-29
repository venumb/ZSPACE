#pragma once

#include<headers\IO\zJSON.h>

namespace zSpace
{
	/** \addtogroup zIO
	*	\brief The data transfer classes and utility methods of the library.
	*  @{
	*/

	//--------------------------
	//---- IMPORT METHODS
	//--------------------------


	//---- MESH METHODS

	/** \addtogroup zIO_Mesh
	*	\brief Collection of input - output methods for zMesh.
	*  @{
	*/

	/*! \brief This method imports zMesh from an OBJ file.
	*
	*	\param [in]		inMesh				- mesh create from the obj file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	void fromOBJ(zMesh &inMesh, string infilename)
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
	}

	/*! \brief This method imports zMesh from a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inMesh				- mesh created from the JSON file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/

	void fromJSON(zMesh &inMesh, string infilename)
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
	}
	
	/** @}*/
	
	//---- GRAPH METHODS
		
	/** \addtogroup zIO_Graph
	*	\brief Collection of input - output methods for zGraph.
	*  @{
	*/

	/*! \brief This method imports zGraph from an TXT file.
	*
	*	\param [in]		inGraph				- mesh create from the obj file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	void fromTXT(zGraph &inGraph, string infilename)
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
		printf("\n inGraph: %i %i %i", inGraph.numVertices(), inGraph.numEdges());

		
	}

	/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
	*
	*	\param [in]		inGraph				- graph created from the JSON file.
	*	\param [in]		infilename			- input file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	
	void fromJSON(zGraph &inGraph, string infilename)
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
	}


	/*! \brief This method creates zGraph from a input zMesh.
	*
	*	\param [in]		inGraph				- graph created from the JSON file.
	*	\param [in]		inMesh				- input mesh.
	*	\since version 0.0.1
	*/
	void fromMESH(zGraph &inGraph, zMesh &inMesh)
	{
		
		vector<int>edgeConnects;		

		for (int i = 0; i < inMesh.numEdges(); i += 2)
		{
			edgeConnects.push_back(inMesh.edges[i + 1].getVertex()->getVertexId());
			edgeConnects.push_back(inMesh.edges[i ].getVertex()->getVertexId());
		}


		inGraph = zGraph(inMesh.vertexPositions, edgeConnects);

	}
	
	/** @}*/

	//--------------------------
	//---- EXPORT METHODS
	//--------------------------

	//---- MESH METHODS

	/** \addtogroup zIO_Mesh
	*	\brief Collection of input - output methods for zMesh.
	*  @{
	*/

	/*! \brief This method exports zMesh as an OBJ file.
	*
	*	\param [in]		inMesh				- mesh create from the obj file.
	*	\param [in]		infilename			- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	void toOBJ(zMesh &inMesh, string outfilename)
	{
		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;

		}

		myfile << "\n ";

		// vertex positions
		for (int i = 0; i < inMesh.vertexPositions.size(); i++)
		{
			myfile<< "\n v " << inMesh.vertexPositions[i].x << " " << inMesh.vertexPositions[i].y << " " << inMesh.vertexPositions[i].z;

		}

		myfile << "\n ";

		// face connectivity
		for (int i = 0; i < inMesh.numPolygons(); i++)
		{
			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			myfile << "\n f ";

			for (int j = 0; j < fVerts.size(); j++)
			{
				myfile << fVerts[j] + 1;

				if( j!= fVerts.size() -1 ) myfile << " ";
			}
			
		}	

		myfile << "\n ";

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
	
	void toJSON(zMesh &inMesh, string outfilename, bool vColors = false)
	{
		zMeshJSON inMeshJSON;
		json j;

		inMeshJSON.to_json(j, inMesh, vColors);

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

	/** @}*/

	//---- GRAPH METHODS

	/** \addtogroup zIO_Graph
	*	\brief Collection of input - output methods for zGraph.
	*  @{
	*/

	/*! \brief This method exports zGraph to a TXT file format. 
	*
	*	\param [in]		inGraph				- input graph.
	*	\param [in]		outfilename			- output file name including the directory path and extension.
	*	\since version 0.0.1
	*/
	void toTXT(zGraph &inGraph, string outfilename)
	{
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
	void toJSON(zGraph &inGraph, string outfilename, bool vColors = false)
	{
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

	/** @}*/

	/** @}*/
}
