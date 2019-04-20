#pragma once


#include<headers/api/object/zObjGraph.h>

#include<headers/api/functionsets/zFn.h>
#include<headers/api/functionsets/zFnMesh.h>

namespace zSpace
{
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnGraph
	*	\brief A graph function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnGraph : protected zFn
	{
	private:
		
		/*! \brief This method sets the edge vertex position conatiners for static meshes.
		*
		*	\since version 0.0.2
		*/
		void setStaticContainers()
		{
			graphObj->graph.staticGeometry = true;

			vector<vector<zVector>> edgePositions;

			for (int i = 0; i < numEdges(); i++)
			{
				vector<zVector> vPositions;
				getVertexPositions(i, zEdgeData, vPositions);

				edgePositions.push_back(vPositions);
			}

			graphObj->graph.setStaticEdgePositions(edgePositions);			
		}

	protected:
		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;


		//--------------------------
		//---- REMOVE INACTIVE
		//--------------------------

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData .
		*	\since version 0.0.2
		*/
		void removeInactive(zHEData type = zVertexData)
		{
			//  Vertex
			if (type == zVertexData)
			{

				zVertex *resized = new zVertex[graphObj->graph.max_n_v];

				int vertexActiveID = 0;
				int numOrginalVertexActive = graphObj->graph.vertexActive.size();

				for (int i = 0; i < numVertices(); i++)
				{

					while (!graphObj->graph.vertexActive[i])
					{
						graphObj->graph.vertexActive.erase(graphObj->graph.vertexActive.begin() + i);

						graphObj->graph.vertexPositions.erase(graphObj->graph.vertexPositions.begin() + i);

						graphObj->graph.vertexColors.erase(graphObj->graph.vertexColors.begin() + i);

						vertexActiveID++;
					}

					resized[i].setVertexId(i);


					// get connected edges and repoint their pointers
					if (vertexActiveID < numOrginalVertexActive)
					{
						vector<int> cEdges;
						getConnectedEdges(vertexActiveID, zVertexData, cEdges);

						for (int j = 0; j < cEdges.size(); j++)
						{
							graphObj->graph.edges[cEdges[j]].getSym()->setVertex(&resized[i]);
						}


						resized[i].setEdge(&graphObj->graph.edges[graphObj->graph.vertices[vertexActiveID].getEdge()->getEdgeId()]);

						graphObj->graph.edges[graphObj->graph.vertices[vertexActiveID].getEdge()->getEdgeId()].getSym()->setVertex(&resized[i]);
					}


					vertexActiveID++;

				}

				//printf("\n m: %i %i ", numVertices(), vertexActive.size());


				delete[] graphObj->graph.vertices;

				graphObj->graph.vertices = resized;

				printf("\n removed inactive vertices. ");

			}

			//  Edge
			else if (type == zEdgeData) {

				zEdge *resized = new zEdge[graphObj->graph.max_n_e];

				int edgeActiveID = 0;
				int numOrginalEdgeActive = graphObj->graph.edgeActive.size();

				int inactiveCounter = 0;

				// clear vertices edge map
				graphObj->graph.verticesEdge.clear();


				for (int i = 0; i < numEdges(); i += 2)
				{

					while (!graphObj->graph.edgeActive[i])
					{
						graphObj->graph.edgeActive.erase(graphObj->graph.edgeActive.begin() + i);
						edgeActiveID++;
					}


					resized[i].setEdgeId(i);
					resized[i + 1].setEdgeId(i + 1);

					// get connected edges and repoint their pointers
					if (edgeActiveID < numOrginalEdgeActive)
					{


						resized[i].setSym(&resized[i + 1]);

						if (graphObj->graph.edges[edgeActiveID].getNext())
						{
							resized[i].setNext(&resized[graphObj->graph.edges[edgeActiveID].getNext()->getEdgeId()]);

							graphObj->graph.edges[edgeActiveID].getNext()->setPrev(&resized[i]);
						}
						if (graphObj->graph.edges[edgeActiveID].getPrev())
						{
							resized[i].setPrev(&resized[graphObj->graph.edges[edgeActiveID].getPrev()->getEdgeId()]);

							graphObj->graph.edges[edgeActiveID].getPrev()->setNext(&resized[i]);
						}


						if (graphObj->graph.edges[edgeActiveID].getVertex()) resized[i].setVertex(&graphObj->graph.vertices[graphObj->graph.edges[edgeActiveID].getVertex()->getVertexId()]);
						graphObj->graph.vertices[graphObj->graph.edges[edgeActiveID].getVertex()->getVertexId()].setEdge(resized[i].getSym());


						//sym edge
						if (graphObj->graph.edges[edgeActiveID + 1].getNext())
						{
							resized[i + 1].setNext(&resized[graphObj->graph.edges[edgeActiveID + 1].getNext()->getEdgeId()]);

							graphObj->graph.edges[edgeActiveID + 1].getNext()->setPrev(&resized[i + 1]);

						}
						if (graphObj->graph.edges[edgeActiveID + 1].getPrev())
						{
							resized[i + 1].setPrev(&resized[graphObj->graph.edges[edgeActiveID + 1].getPrev()->getEdgeId()]);

							graphObj->graph.edges[edgeActiveID + 1].getPrev()->setNext(&resized[i + 1]);
						}

						if (graphObj->graph.edges[edgeActiveID + 1].getVertex()) resized[i + 1].setVertex(&graphObj->graph.vertices[graphObj->graph.edges[edgeActiveID + 1].getVertex()->getVertexId()]);
						graphObj->graph.vertices[graphObj->graph.edges[edgeActiveID + 1].getVertex()->getVertexId()].setEdge(resized[i + 1].getSym());



						// rebuild vertices edge map
						int v2 = resized[i].getVertex()->getVertexId();
						int v1 = resized[i + 1].getVertex()->getVertexId();

						graphObj->graph.addToVerticesEdge(v1, v2, i);



					}

					edgeActiveID += 2;

				}

				//printf("\n m: %i %i ", numEdges(), edgeActive.size());

				delete[] graphObj->graph.edges;
				graphObj->graph.edges = resized;

				printf("\n removed inactive edges. ");

			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method imports zGraph from an TXT file.
		*
		*	\param [in]		inGraph				- graph created from the txt file.
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		inline void fromTXT(string infilename)
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

				

				vector<string> perlineData = graphObj->graph.coreUtils.splitString(str, " ");

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


			if (!planarGraph) graphObj->graph = zGraph(positions, edgeConnects);;
			if (planarGraph)
			{
				graphNormal.normalize();

				zVector x(1, 0, 0);
				zVector sortRef = graphNormal ^ x;

				graphObj->graph = zGraph(positions, edgeConnects, graphNormal, sortRef);
			}
			printf("\n graphObj->graph: %i %i ", numVertices(), numEdges());


		}

		/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		inGraph				- graph created from the JSON file.
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		inline void fromJSON(string infilename)
		{
			json j;
			zUtilsJsonHE graphJSON;

			// read data to json
			ifstream in_myfile;
			in_myfile.open(infilename.c_str());

			int lineCnt = 0;

			if (in_myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;
			}

			in_myfile >> j;
			in_myfile.close();

			// read data to json graph

			// Vertices
			graphJSON.vertices.clear();
			graphJSON.vertices = (j["Vertices"].get<vector<int>>());

			//Edges
			graphJSON.halfedges.clear();
			graphJSON.halfedges = (j["Halfedges"].get<vector<vector<int>>>());



			// update  graph

			graphObj->graph.vertices = new zVertex[graphJSON.vertices.size() * 2];
			graphObj->graph.edges = new zEdge[graphJSON.halfedges.size() * 2];


			graphObj->graph.setNumVertices(graphJSON.vertices.size());
			graphObj->graph.setNumEdges(floor(graphJSON.halfedges.size()));


			graphObj->graph.vertexActive.clear();
			for (int i = 0; i < graphJSON.vertices.size(); i++)
			{
				graphObj->graph.vertices[i].setVertexId(i);
				if (graphJSON.vertices[i] != -1) graphObj->graph.vertices[i].setEdge(&graphObj->graph.edges[graphJSON.vertices[i]]);

				graphObj->graph.vertexActive.push_back(true);
			}

			int k = 0;
			graphObj->graph.edgeActive.clear();
			for (int i = 0; i < graphJSON.halfedges.size(); i++)
			{
				graphObj->graph.edges[i].setEdgeId(i);

				if (graphJSON.halfedges[i][k] != -1) 	graphObj->graph.edges[i].setPrev(&graphObj->graph.edges[graphJSON.halfedges[i][k]]);
				if (graphJSON.halfedges[i][k + 1] != -1) graphObj->graph.edges[i].setNext(&graphObj->graph.edges[graphJSON.halfedges[i][k + 1]]);
				if (i % 2 == 0)
				{
					if (graphJSON.halfedges[i + 1][k + 2] != -1) graphObj->graph.edges[i].setVertex(&graphObj->graph.vertices[graphJSON.halfedges[i + 1][k + 2]]);
				}
				else
				{
					if (graphJSON.halfedges[i - 1][k + 2] != -1) graphObj->graph.edges[i].setVertex(&graphObj->graph.vertices[graphJSON.halfedges[i - 1][k + 2]]);
				}

				if (i % 2 == 0) graphObj->graph.edges[i].setSym(&graphObj->graph.edges[i]);
				else  graphObj->graph.edges[i].setSym(&graphObj->graph.edges[i - 1]);

				graphObj->graph.edgeActive.push_back(true);
			}



			// Vertex Attributes
			graphJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
			//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

			graphObj->graph.vertexPositions.clear();
			for (int i = 0; i < graphJSON.vertexAttributes.size(); i++)
			{
				for (int k = 0; k < graphJSON.vertexAttributes[i].size(); k++)
				{
					
					// position and color

					if (graphJSON.vertexAttributes[i].size() == 6)
					{
						zVector pos(graphJSON.vertexAttributes[i][k], graphJSON.vertexAttributes[i][k + 1], graphJSON.vertexAttributes[i][k + 2]);
						graphObj->graph.vertexPositions.push_back(pos);

						zColor col(graphJSON.vertexAttributes[i][k + 3], graphJSON.vertexAttributes[i][k + 4], graphJSON.vertexAttributes[i][k + 5], 1);
						graphObj->graph.vertexColors.push_back(col);

						k += 5;
					}
				}
			}


			// Edge Attributes
			graphJSON.halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();
			printf("\n graphObj->graph: %i %i ", numVertices(), numEdges());

			// add to maps 
			for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
			{
				graphObj->graph.addToPositionMap(graphObj->graph.vertexPositions[i], i);
			}


			for (int i = 0; i < numEdges(); i += 2)
			{
				int v1 = graphObj->graph.edges[i].getVertex()->getVertexId();
				int v2 = graphObj->graph.edges[i + 1].getVertex()->getVertexId();

				graphObj->graph.addToVerticesEdge(v1, v2, i);
			}


		}


		/*! \brief This method exports zGraph to a TXT file format.
		*
		*	\param [in]		inGraph				- input graph.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		inline void toTXT(string outfilename)
		{
			// remove inactive elements
			if (numVertices() != graphObj->graph.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edgeActive.size()) removeInactiveElements(zEdgeData);


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
			for (int i = 0; i < graphObj->graph.vertexActive.size(); i++)
			{
				if (!graphObj->graph.vertexActive[i]) continue;

				myfile << "\n v " << graphObj->graph.vertexPositions[i].x << " " << graphObj->graph.vertexPositions[i].y << " " << graphObj->graph.vertexPositions[i].z;

			}

			myfile << "\n ";

			// edge connectivity
			for (int i = 0; i < graphObj->graph.edgeActive.size(); i += 2)
			{
				if (!graphObj->graph.edgeActive[i]) continue;

				myfile << "\n e ";

				myfile << graphObj->graph.edges[i].getVertex()->getVertexId() << " ";
				myfile << graphObj->graph.edges[i].getVertex()->getVertexId();

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
		*	\since version 0.0.2
		*/
		inline void toJSON(string outfilename)
		{
			// remove inactive elements
			if (numVertices() != graphObj->graph.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edgeActive.size()) removeInactiveElements(zEdgeData);


			// output file
			zUtilsJsonHE graphJSON;
			json j;

			// create json

			// Vertices
			for (int i = 0; i < numVertices(); i++)
			{
				if (graphObj->graph.vertices[i].getEdge()) graphJSON.vertices.push_back(graphObj->graph.vertices[i].getEdge()->getEdgeId());
				else graphJSON.vertices.push_back(-1);
			}

			//Edges
			for (int i = 0; i < numEdges(); i++)
			{
				vector<int> HE_edges;

				if (graphObj->graph.edges[i].getPrev()) HE_edges.push_back(graphObj->graph.edges[i].getPrev()->getEdgeId());
				else HE_edges.push_back(-1);

				if (graphObj->graph.edges[i].getNext()) HE_edges.push_back(graphObj->graph.edges[i].getNext()->getEdgeId());
				else HE_edges.push_back(-1);

				if (graphObj->graph.edges[i].getVertex()) HE_edges.push_back(graphObj->graph.edges[i].getVertex()->getVertexId());
				else HE_edges.push_back(-1);

				graphJSON.halfedges.push_back(HE_edges);
			}


			// vertex Attributes
			for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
			{
				vector<double> v_attrib;

				v_attrib.push_back(graphObj->graph.vertexPositions[i].x);
				v_attrib.push_back(graphObj->graph.vertexPositions[i].y);
				v_attrib.push_back(graphObj->graph.vertexPositions[i].z);

				
				v_attrib.push_back(graphObj->graph.vertexColors[i].r);
				v_attrib.push_back(graphObj->graph.vertexColors[i].g);
				v_attrib.push_back(graphObj->graph.vertexColors[i].b);
			


				graphJSON.vertexAttributes.push_back(v_attrib);
			}


			// Json file 
			j["Vertices"] = graphJSON.vertices;
			j["Halfedges"] = graphJSON.halfedges;
			j["VertexAttributes"] = graphJSON.vertexAttributes;
			j["HalfedgeAttributes"] = graphJSON.halfedgeAttributes;
			

			// export json

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

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
		/*!	\brief boolean true is its a planar graph  */
		bool planarGraph;

		/*!	\brief stores normal of the the graph if planar  */
		zVector graphNormal;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnGraph() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.2
		*/
		zFnGraph(zObjGraph &_graphObj, bool  _planarGraph = false, zVector _graphNormal = zVector(0,0,1) )
		{
			graphObj = &_graphObj;

			planarGraph = _planarGraph;
			graphNormal = _graphNormal;
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnGraph() {}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void from(string path, zFileTpye type,bool staticGeom = false) override
		{
			if (type == zTXT)
			{
				fromTXT(path);
				setStaticContainers();
			}
			else if (type == zJSON)
			{
				fromJSON(path);
				setStaticContainers();
			}

			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		void to(string path, zFileTpye type) override
		{
			if (type == zTXT) toTXT(path);
			else if (type == zJSON) toJSON(path);

			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		void clear() override
		{
			if (graphObj->graph.vertices != NULL)
			{
				delete[] graphObj->graph.vertices;
				graphObj->graph.vertices = NULL;

				graphObj->graph.vertexActive.clear();
				graphObj->graph.vertexPositions.clear();
				graphObj->graph.vertexColors.clear();
				graphObj->graph.vertexWeights.clear();

				graphObj->graph.positionVertex.clear();
				graphObj->graph.verticesEdge.clear();

			}

			if (graphObj->graph.edges != NULL)
			{
				delete[] graphObj->graph.edges;
				graphObj->graph.edges = NULL;

				graphObj->graph.edgeActive.clear();
				graphObj->graph.edgeColors.clear();
				graphObj->graph.edgeWeights.clear();
			}

			graphObj->graph.n_v = graphObj->graph.n_e = 0;
		}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a graph from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void create(vector<zVector>(&_positions), vector<int>(&edgeConnects), bool staticGraph = false)
		{
			graphObj->graph = zGraph(_positions, edgeConnects);

			if (staticGraph) setStaticContainers();
		}

		/*! \brief his method creates a graphfrom the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	graphNormal		- normal of the plane of the graph.
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void create(vector<zVector>(&_positions), vector<int>(&edgeConnects), zVector &graphNormal, bool staticGraph = false)
		{
					
			graphNormal.normalize();

			zVector x(1, 0, 0);
			zVector sortRef = graphNormal ^ x;

			graphObj->graph = zGraph(_positions, edgeConnects, graphNormal, sortRef);

			if (staticGraph) setStaticContainers();
		}

		/*! \brief This method creates a graph from a mesh.
		*
		*	\param		[in]	meshObj			- input mesh object.	
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void createFromMesh(zObjMesh &meshObj, bool staticGraph = false)
		{
			zFnMesh fnMesh(meshObj);

			vector<int>edgeConnects;
			vector<zVector> vertexPositions;

			fnMesh.getEdgeData(edgeConnects);
			fnMesh.getVertexPositions(vertexPositions);

			create(vertexPositions, edgeConnects);

			if (staticGraph) setStaticContainers();
		}

		

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method returns the number of vertices in the graph or mesh.
		*	\return				number of vertices.
		*	\since version 0.0.2
		*/
		int numVertices()
		{
			return graphObj->graph.n_v;
		}

		/*! \brief This method returns the number of half edges in the graph or mesh.
		*	\return				number of edges.
		*	\since version 0.0.2
		*/
		int numEdges()
		{
			return graphObj->graph.n_e;
		}

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\return		[out]		bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zVector pos, int &outVertexId)
		{
			bool out = false;;

			double factor = pow(10, 3);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			std::unordered_map<std::string, int>::const_iterator got = graphObj->graph.positionVertex.find(hashKey);


			if (got != graphObj->graph.positionVertex.end())
			{
				out = true;
				outVertexId = got->second;
			}


			return out;
		}
	
		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outEdgeId	- stores edgeId if the edge exists else it is -1.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool edgeExists(int v1, int v2, int &outEdgeId)
		{

			bool out = false;

			string hashKey = (to_string(v1) + "," + to_string(v2));
			std::unordered_map<std::string, int>::const_iterator got = graphObj->graph.verticesEdge.find(hashKey);


			if (got != graphObj->graph.verticesEdge.end())
			{
				out = true;
				outEdgeId = got->second;
			}

			return out;
		}

		/*! \brief This method gets the edges connected to input zVertex or zEdge.
		*
		*	\param		[in]	index			- index in the vertex/edge list.
		*	\param		[in]	type			- zVertexData or zEdgeData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.2
		*/
		void getConnectedEdges(int index, zHEData type, vector<int>& edgeIndicies)
		{
			edgeIndicies.clear();

			//  Vertex 
			if (type == zVertexData)
			{
				if (graphObj->graph.vertices[index].getEdge())
				{
					zEdge* start = graphObj->graph.vertices[index].getEdge();
					zEdge* e = start;

					bool exit = false;

					do
					{
						edgeIndicies.push_back(e->getEdgeId());

						//printf("\n %i %i ", e->getEdgeId(), start->getEdgeId());

						if (e->getPrev())
						{
							if (e->getPrev()->getSym()) e = e->getPrev()->getSym();
							else exit = true;
						}
						else exit = true;

					} while (e != start && !exit);
				}
			}

			//  Edge
			else if (type == zEdgeData)
			{
				vector<int> connectedEdgestoVert0;
				getConnectedEdges(graphObj->graph.edges[index].getVertex()->getVertexId(), zVertexData, connectedEdgestoVert0);

				vector<int> connectedEdgestoVert1;
				getConnectedEdges(graphObj->graph.edges[index].getSym()->getVertex()->getVertexId(), zVertexData, connectedEdgestoVert1);

				for (int i = 0; i < connectedEdgestoVert0.size(); i++)
				{
					if (connectedEdgestoVert0[i] != index) edgeIndicies.push_back(connectedEdgestoVert0[i]);
				}


				for (int i = 0; i < connectedEdgestoVert1.size(); i++)
				{
					if (connectedEdgestoVert1[i] != index) edgeIndicies.push_back(connectedEdgestoVert1[i]);
				}
			}

			else  throw std::invalid_argument(" error: invalid zHEData type");
			
		}

		/*! \brief This method gets the vertices connected to input zVertex.
		*
		*	\param		[in]	index			- index in the vertex list.
		*	\param		[in]	type			- zVertexData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.2
		*/
		void getConnectedVertices(int index, zHEData type, vector<int>& vertexIndicies)
		{
			vertexIndicies.clear();

			// Vertex
			if (type == zVertexData)
			{

				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				for (int i = 0; i < connectedEdges.size(); i++)
				{
					vertexIndicies.push_back(graphObj->graph.edges[connectedEdges[i]].getVertex()->getVertexId());
				}

			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		
		}

		/*!	\brief This method gets the vertices attached to input zEdge.
		*
		*	\param		[in]	index			- index in the edge list.
		*	\param		[in]	type			- zEdgeData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.2
		*/
		void getVertices(int index, zHEData type, vector<int>& vertexIndicies)
		{
			vertexIndicies.clear();

			//  Edge
			if (type == zEdgeData)
			{
				vertexIndicies.push_back(graphObj->graph.edges[index].getVertex()->getVertexId());
				vertexIndicies.push_back(graphObj->graph.edges[index].getSym()->getVertex()->getVertexId());
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			
		}

		/*!	\brief This method gets the vertex positions attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge container.
		*	\param		[in]	type			- zEdgeData.
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.2
		*/
		void getVertexPositions(int index, zHEData type, vector<zVector> &vertPositions)
		{
			vertPositions.clear();

			// Edge
			if (type == zEdgeData)
			{

				vector<int> eVerts;

				getVertices(index, type, eVerts);

				for (int i = 0; i < eVerts.size(); i++)
				{
					vertPositions.push_back(graphObj->graph.vertexPositions[eVerts[i]]);
				}

			}			

			else throw std::invalid_argument(" error: invalid zHEData type");

		}

		/*!	\brief This method calculate the valency of the input zVertex.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- valency of input vertex input valence.
		*	\since version 0.0.2
		*/
		int getVertexValence(int index)
		{
			int out;

			vector<int> connectedEdges;
			getConnectedEdges(index, zVertexData, connectedEdges);

			out = connectedEdges.size();

			return out;
		}

		/*!	\brief This method determines if input zVertex valency is equal to the input valence number.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	valence	- input valence value.
		*	\return				bool	- true if valency is equal to input valence.
		*	\since version 0.0.2
		*/
		bool checkVertexValency(int index, int valence = 1)
		{
			bool out = false;
			out = (getVertexValence(index) == valence) ? true : false;


			return out;
		}

		

		//--------------------------
		//--- HALF EDGE QUERY METHODS 
		//--------------------------

		/*!	\brief This method return the next edge of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zEdge	- pointer to next edge.
		*	\since version 0.0.2
		*/
		zEdge* getNext(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edges[index].getNext();
		}

		/*!	\brief This method return the next edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of next edge if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getNextIndex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (graphObj->graph.edges[index].getNext()) return graphObj->graph.edges[index].getNext()->getEdgeId();
			else return -1;
		}

		/*!	\brief This method return the previous edge of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zEdge	- pointer to previous edge.
		*	\since version 0.0.2
		*/
		zEdge* getPrev(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edges[index].getPrev();
		}

		/*!	\brief This method return the previous edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of previous edge if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getPrevIndex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (graphObj->graph.edges[index].getPrev()) return graphObj->graph.edges[index].getPrev()->getEdgeId();
			else return -1;
		}

		/*!	\brief This method return the symmetry edge of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zEdge	- pointer to symmetry edge.
		*	\since version 0.0.2
		*/
		zEdge* getSym(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edges[index].getSym();
		}

		/*!	\brief This method return the symmetry edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of symmetry edge.
		*	\since version 0.0.2
		*/
		int getSymIndex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edges[index].getSym()->getEdgeId();
		}


		/*!	\brief This method return the vertex pointed by the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zVertex	- pointer to vertex.
		*	\since version 0.0.2
		*/
		zVertex* getEndVertex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edges[index].getVertex();
		}

		/*!	\brief This method return the vertex pointed by the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of vertex if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getEndVertexIndex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (graphObj->graph.edges[index].getVertex()) return graphObj->graph.edges[index].getVertex()->getVertexId();
			else return -1;
		}

		/*!	\brief This method return the vertex pointed by the symmetry of input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zVertex	- pointer to vertex.
		*	\since version 0.0.2
		*/
		zVertex* getStartVertex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return getSym(index)->getVertex();
		}

		/*!	\brief This method return the vertex pointed by the symmetry of input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of vertex if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getStartVertexIndex(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (getSym(index)->getVertex()) return getSym(index)->getVertex()->getVertexId();
			else return -1;
		}

		/*!	\brief This method return the edge attached to the input indexed vertex.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	type	- zVertexData .
		*	\return				zEdge	- pointer to edge.
		*	\since version 0.0.2
		*/
		zEdge* getEdge(int index, zHEData type)
		{
			if (type = zVertexData)
			{
				if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return graphObj->graph.vertices[index].getEdge();
			}
			

			else throw std::invalid_argument(" error: invalid zHEData type");

		}

		/*!	\brief This method return the index of the edge attached to the input indexed vertex.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	type	- zVertexData.
		*	\return				int		- index of edge.
		*	\since version 0.0.2
		*/
		int getEdgeIndex(int index, zHEData type)
		{
			if (type = zVertexData)
			{
				if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return graphObj->graph.vertices[index].getEdge()->getEdgeId();
			}
		

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		/*! \brief This method computes the Edge colors based on the vertex colors.
		*
		*	\since version 0.0.2
		*/
		void computeEdgeColorfromVertexColor()
		{

			for (int i = 0; i < graphObj->graph.edgeActive.size(); i += 2)
			{
				if (graphObj->graph.edgeActive[i])
				{
					int v0 = graphObj->graph.edges[i].getVertex()->getVertexId();
					int v1 = graphObj->graph.edges[i + 1].getVertex()->getVertexId();

					zColor col;
					col.r = (graphObj->graph.vertexColors[v0].r + graphObj->graph.vertexColors[v1].r) * 0.5;
					col.g = (graphObj->graph.vertexColors[v0].g + graphObj->graph.vertexColors[v1].g) * 0.5;
					col.b = (graphObj->graph.vertexColors[v0].b + graphObj->graph.vertexColors[v1].b) * 0.5;
					col.a = (graphObj->graph.vertexColors[v0].a + graphObj->graph.vertexColors[v1].a) * 0.5;

					if (graphObj->graph.edgeColors.size() <= i) graphObj->graph.edgeColors.push_back(col);
					else graphObj->graph.edgeColors[i] = col;

					if (graphObj->graph.edgeColors.size() <= i + 1) graphObj->graph.edgeColors.push_back(col);
					else graphObj->graph.edgeColors[i + 1] = col;
				}
				

			}



		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.2
		*/
		void computeVertexColorfromEdgeColor()
		{
			for (int i = 0; i < graphObj->graph.vertexActive.size(); i++)
			{
				if (graphObj->graph.vertexActive[i])
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					zColor col;
					for (int j = 0; j < cEdges.size(); j++)
					{
						col.r += graphObj->graph.edgeColors[cEdges[j]].r;
						col.g += graphObj->graph.edgeColors[cEdges[j]].g;
						col.b += graphObj->graph.edgeColors[cEdges[j]].b;
					}

					col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

					graphObj->graph.vertexColors[i] = col;

				}
			}
		}

		/*! \brief This method averages the positions of vertex except for the ones on the boundary.
		*
		*	\param		[in]	numSteps	- number of times the averaging is carried out.
		*	\since version 0.0.2
		*/
		void averageVertices(int numSteps = 1)
		{
			for (int k = 0; k < numSteps; k++)
			{
				vector<zVector> tempVertPos;

				for (int i = 0; i < graphObj->graph.vertexActive.size(); i++)
				{
					tempVertPos.push_back(graphObj->graph.vertexPositions[i]);

					if (graphObj->graph.vertexActive[i])
					{
						if (!checkVertexValency(i, 1))
						{
							vector<int> cVerts;

							getConnectedVertices(i, zVertexData, cVerts);

							for (int j = 0; j < cVerts.size(); j++)
							{
								zVector p = graphObj->graph.vertexPositions[cVerts[j]];
								tempVertPos[i] += p;
							}

							tempVertPos[i] /= (cVerts.size() + 1);
						}
					}
					
				}

				// update position
				for (int i = 0; i < tempVertPos.size(); i++) graphObj->graph.vertexPositions[i] = tempVertPos[i];
			}

		}

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData .
		*	\since version 0.0.2
		*/
		void removeInactiveElements(zHEData type)
		{
			if (type == zVertexData || type == zEdgeData) removeInactive(type);
			else throw std::invalid_argument(" error: invalid zHEData type");
		}
		
		/*! \brief This method makes the graph a fixed. Computes the static edge vertex positions if true.
		*
		*	\since version 0.0.2
		*/
		void makeStatic()
		{
			setStaticContainers();
		}

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		
		/*! \brief This method sets vertex position of the input vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.2
		*/
		void setVertexPosition(int index, zVector &pos)
		{
			if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			graphObj->graph.vertexPositions[index] = pos;

		}

		/*! \brief This method sets vertex positions of all the vertices.
		*
		*	\param		[in]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void setVertexPositions(vector<zVector>& pos)
		{
			if (pos.size() != graphObj->graph.vertexPositions.size()) throw std::invalid_argument("size of position contatiner is not equal to number of graph vertices.");

			for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
			{
				graphObj->graph.vertexPositions[i] = pos[i];
			}
		}

		/*! \brief This method sets vertex color of the input vertex to the input color.
		*
		*	\param		[in]	index					- input vertex index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setVertexColor(int index, zColor col)
		{

			if (graphObj->graph.vertexColors.size() != graphObj->graph.vertexActive.size())
			{
				graphObj->graph.vertexColors.clear();
				for (int i = 0; i < graphObj->graph.vertexActive.size(); i++) graphObj->graph.vertexColors.push_back(zColor());
			}

			graphObj->graph.vertexColors[index] = col;

		}

		/*! \brief This method sets vertex color of all the vertices to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColor(zColor col, bool setEdgeColor = false)
		{
			if (graphObj->graph.vertexColors.size() != graphObj->graph.vertexActive.size())
			{
				graphObj->graph.vertexColors.clear();
				for (int i = 0; i < graphObj->graph.vertexActive.size(); i++) graphObj->graph.vertexColors.push_back(zColor(1, 0, 0, 1));
			}

			for (int i = 0; i < graphObj->graph.vertexColors.size(); i++)
			{
				graphObj->graph.vertexColors[i] = col;
			}

			if (setEdgeColor) computeEdgeColorfromVertexColor();
		}

		/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the graph.
		*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColors(vector<zColor>& col, bool setEdgeColor = false)
		{
			if (graphObj->graph.vertexColors.size() != graphObj->graph.vertexActive.size())
			{
				graphObj->graph.vertexColors.clear();
				for (int i = 0; i < graphObj->graph.vertexActive.size(); i++) graphObj->graph.vertexColors.push_back(zColor(1, 0, 0, 1));
			}

			if (col.size() != graphObj->graph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

			for (int i = 0; i < graphObj->graph.vertexColors.size(); i++)
			{
				graphObj->graph.vertexColors[i] = col[i];
			}

			if (setEdgeColor) computeEdgeColorfromVertexColor();
		}
				
		/*! \brief This method sets edge color of of the input edge and its symmetry edge to the input color.
		*
		*	\param		[in]	index					- input edge index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setEdgeColor(int index, zColor col)
		{

			if (graphObj->graph.edgeColors.size() != graphObj->graph.edgeActive.size())
			{
				graphObj->graph.edgeColors.clear();
				for (int i = 0; i < graphObj->graph.edgeActive.size(); i++) graphObj->graph.edgeColors.push_back(zColor());
			}

			graphObj->graph.edgeColors[index] = col;

			int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

			graphObj->graph.edgeColors[symEdge] = col;

		}

		/*! \brief This method sets edge color of all the edges to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColor(zColor col, bool setVertexColor = false)
		{
			if (graphObj->graph.edgeColors.size() != graphObj->graph.edgeActive.size())
			{
				for (int i = 0; i < graphObj->graph.edgeActive.size(); i++) graphObj->graph.edgeColors.push_back(zColor());
			}

			for (int i = 0; i < graphObj->graph.edgeColors.size(); i += 2)
			{
				setEdgeColor(i, col);
			}

			if (setVertexColor) computeVertexColorfromEdgeColor();

		}

		/*! \brief This method sets edge color of all the edges with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of half edges in the graph.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColors(vector<zColor>& col, bool setVertexColor)
		{
			if (col.size() != graphObj->graph.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph half edges.");

			for (int i = 0; i < graphObj->graph.edgeColors.size(); i++)
			{
				graphObj->graph.edgeColors[i] = col[i];
			}

			if (setVertexColor) computeVertexColorfromEdgeColor();
		}

		/*! \brief This method sets edge weight of of the input edge and its symmetry edge to the input weight.
		*
		*	\param		[in]	index					- input edge index.
		*	\param		[in]	wt						- input wight.
		*	\since version 0.0.2
		*/
		void setEdgeWeight(int index, double wt)
		{

			if (graphObj->graph.edgeWeights.size() != graphObj->graph.edgeActive.size())
			{
				graphObj->graph.edgeWeights.clear();
				for (int i = 0; i < graphObj->graph.edgeActive.size(); i++) graphObj->graph.edgeWeights.push_back(1);

			}

			graphObj->graph.edgeWeights[index] = wt;

			int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

			graphObj->graph.edgeWeights[symEdge] = wt;

		}

		/*! \brief This method sets edge weight of all the edges to the input weight.
		*
		*	\param		[in]	wt				- input weight.
		*	\since version 0.0.2
		*/
		void setEdgeWeight(double wt)
		{
			if (graphObj->graph.edgeWeights.size() != graphObj->graph.edgeActive.size())
			{
				graphObj->graph.edgeWeights.clear();
				for (int i = 0; i < graphObj->graph.edgeActive.size(); i++) graphObj->graph.edgeWeights.push_back(1);

			}

			for (int i = 0; i < graphObj->graph.edgeWeights.size(); i += 2)
			{
				setEdgeWeight(i, wt);
			}

		}

		/*! \brief This method sets edge weights of all the edges with the input weight contatiner.
		*
		*	\param		[in]	wt				- input weight  contatiner. The size of the contatiner should be equal to number of half edges in the mesh.
		*	\since version 0.0.2
		*/
		void setEdgeWeights(vector<double>& wt)
		{
			if (wt.size() != graphObj->graph.edgeColors.size()) throw std::invalid_argument("size of wt contatiner is not equal to number of mesh half edges.");

			for (int i = 0; i < graphObj->graph.edgeWeights.size(); i++)
			{
				graphObj->graph.edgeWeights[i] = wt[i];
			}
		}
				

		//--------------------------
		//--- GET METHODS 
		//--------------------------

		/*! \brief This method gets vertex position of the input vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zvector					- vertex position.
		*	\since version 0.0.2
		*/
		zVector getVertexPosition(int index)
		{
			if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.vertexPositions[index];

		}

		/*! \brief This method gets pointer to the vertex position at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPosition(int index)
		{
			if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &graphObj->graph.vertexPositions[index];

		}

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexPositions(vector<zVector>& pos)
		{
			pos = graphObj->graph.vertexPositions;
		}

		/*! \brief This method gets pointer to the internal vertex positions container.
		*
		*	\return				zVector*					- pointer to internal vertex position container.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPositions()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &graphObj->graph.vertexPositions[0];
		}

		/*! \brief This method gets vertex color of the input vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zColor					- vertex color.
		*	\since version 0.0.2
		*/
		zColor getVertexColor(int index)
		{
			if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.vertexColors[index];

		}

		/*! \brief This method gets pointer to the vertex color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColor(int index)
		{
			if (index > graphObj->graph.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &graphObj->graph.vertexColors[index];

		}

		/*! \brief This method gets vertex color of all the vertices.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexColors(vector<zColor>& col)
		{
			col = graphObj->graph.vertexColors;
		}

		/*! \brief This method gets pointer to the internal vertex color container.
		*
		*	\return				zColor*					- pointer to internal vertex color container.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColors()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &graphObj->graph.vertexColors[0];
		}

		/*! \brief This method gets edge color of the input edge.
		*
		*	\param		[in]	index					- input edge index.
		*	\return				zColor					- edge color.
		*	\since version 0.0.2
		*/
		zColor getEdgeColor(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return graphObj->graph.edgeColors[index];

		}

		/*! \brief This method gets pointer to the edge color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal edge color.
		*	\since version 0.0.2
		*/
		zColor* getRawEdgeColor(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &graphObj->graph.edgeColors[index];

		}

		/*! \brief This method gets edge color of all the edges.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getEdgeColors(vector<zColor>& col)
		{
			col = graphObj->graph.edgeColors;
		}

		/*! \brief This method gets pointer to the internal edge color container.
		*
		*	\return				zColor*					- pointer to internal edge color container.
		*	\since version 0.0.2
		*/
		zColor* getRawEdgeColors()
		{
			if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

			return &graphObj->graph.edgeColors[0];
		}

		/*! \brief This method computes the centers of a the input index edge or face of the mesh.
		*
		*	\param		[in]	type					- zEdgeData or zFaceData.
		*	\return				zVector					- center.
		*	\since version 0.0.2
		*/
		zVector getCenter(int index, zHEData type)
		{
			//  Edge 
			if (type == zEdgeData)
			{
				if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				vector<int> eVerts;
				getVertices(index, zEdgeData, eVerts);

				return (graphObj->graph.vertexPositions[eVerts[0]] + graphObj->graph.vertexPositions[eVerts[1]]) * 0.5;
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method computes the center the graph.
		*
		*	\return		zVector					- center .
		*	\since version 0.0.2
		*/
		zVector getCenter()
		{
			zVector out;

			for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
			{
				out += graphObj->graph.vertexPositions[i];
			}

			out /= graphObj->graph.vertexPositions.size();

			return out;

		}


		/*! \brief This method computes the centers of a all edges of the graph.
		*
		*	\param		[in]	type					- zEdgeData.
		*	\param		[out]	centers					- vector of centers of type zVector.
		*	\since version 0.0.2
		*/
		void getCenters(zHEData type, vector<zVector> &centers)
		{
			// graph Edge 
			if (type == zEdgeData)
			{
				vector<zVector> edgeCenters;

				edgeCenters.clear();

				for (int i = 0; i < graphObj->graph.edgeActive.size(); i += 2)
				{
					if (graphObj->graph.edgeActive[i])
					{
						vector<int> eVerts;
						getVertices(i, zEdgeData, eVerts);

						zVector cen = (graphObj->graph.vertexPositions[eVerts[0]] + graphObj->graph.vertexPositions[eVerts[1]]) * 0.5;

						edgeCenters.push_back(cen);
						edgeCenters.push_back(cen);
					}
					else
					{
						edgeCenters.push_back(zVector());
						edgeCenters.push_back(zVector());
					}
				}

				centers = edgeCenters;
			}
			
			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method computes the edge vector of the input edge of the graph.
		*
		*	\param		[in]	index					- edge index.
		*	\return				zVector					- edge vector.
		*	\since version 0.0.2
		*/
		zVector getEdgeVector(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			int v1 = graphObj->graph.edges[index].getVertex()->getVertexId();
			int v2 = graphObj->graph.edges[index].getSym()->getVertex()->getVertexId();

			zVector out = graphObj->graph.vertexPositions[v1] - (graphObj->graph.vertexPositions[v2]);

			return out;
		}

		/*! \brief This method computes the edge length of the input edge of the graph.
		*
		*	\param		[in]	index			- edge index.
		*	\return				double			- edge length.
		*	\since version 0.0.2
		*/
		double getEdgelength(int index)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			int v1 = graphObj->graph.edges[index].getVertex()->getVertexId();
			int v2 = graphObj->graph.edges[index].getSym()->getVertex()->getVertexId();

			double out = graphObj->graph.vertexPositions[v1].distanceTo(graphObj->graph.vertexPositions[v2]);

			return out;
		}

		/*! \brief This method computes the lengths of all the edges of a the graph.
		*
		*	\param		[out]	edgeLengths				- vector of edge lengths.
		*	\return				double					- total edge lengths.
		*	\since version 0.0.2
		*/
		double getEdgeLengths(vector<double> &edgeLengths)
		{
			double total = 0.0;

			vector<double> out;

			for (int i = 0; i < graphObj->graph.edgeActive.size(); i += 2)
			{
				if (graphObj->graph.edgeActive[i])
				{
					int v1 = graphObj->graph.edges[i].getVertex()->getVertexId();
					int v2 = graphObj->graph.edges[i].getSym()->getVertex()->getVertexId();

					zVector e = graphObj->graph.vertexPositions[v1] - graphObj->graph.vertexPositions[v2];
					double e_len = e.length();

					out.push_back(e_len);
					out.push_back(e_len);

					total += e_len;
				}
				else
				{
					out.push_back(0);
					out.push_back(0);

				}


			}

			edgeLengths = out;

			return total;
		}

		/*! \brief This method creates a duplicate of the input graph.
		*
		*	\param		[in]	planarGraph		- true if input graph is planar.
		*	\param		[in]	graphNormal		- graph normal if the input graph is planar.
		*	\return				zGraph			- duplicate graph.
		*	\since version 0.0.2
		*/
		zObjGraph getDuplicate(bool planarGraph = false, zVector graphNormal = zVector(0, 0, 1))
		{
			zObjGraph out;

			if(numVertices() != graphObj->graph.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edgeActive.size()) removeInactiveElements(zEdgeData);

			vector<zVector> positions;
			vector<int> edgeConnects;


			positions = graphObj->graph.vertexPositions;

			for (int i = 0; i < numEdges(); i += 2)
			{
				int v0 = graphObj->graph.edges[i].getVertex()->getVertexId();
				int v1 = graphObj->graph.edges[i + 1].getVertex()->getVertexId();

				edgeConnects.push_back(v1);
				edgeConnects.push_back(v0);
			}


			if (planarGraph)
			{
				graphNormal.normalize();

				zVector x(1, 0, 0);
				zVector sortRef = graphNormal ^ x;

				out.graph = zGraph(positions, edgeConnects, graphNormal, sortRef);
			}
			else out.graph = zGraph(positions, edgeConnects);

			out.graph.vertexColors = graphObj->graph.vertexColors;
			out.graph.edgeColors = graphObj->graph.edgeColors;

			return out;
		}

		/*! \brief This method returns the mesh created from planar input graph.
		*
		*	\param		[in]	inGraph					- input graph.
		*	\param		[in]	width					- offset width from the graph.
		*	\param		[in]	graphNormal				- normal of the plane of the graph.
		*	\return				zObjMesh				- output mesh object
		*	\since version 0.0.2
		*/
		zObjMesh getGraphMesh( double width = 0.5, zVector graphNormal = zVector(0, 0, 1))
		{
			zObjMesh out;
			vector<zVector>positions;
			vector<int> polyConnects;
			vector<int> polyCounts;

			if(numVertices() != graphObj->graph.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edgeActive.size()) removeInactiveElements(zEdgeData);

			positions = graphObj->graph.vertexPositions;

			vector<vector<int>> edgeVertices;
			for (int i = 0; i < numEdges(); i++)
			{
				vector<int> temp;
				int v0 = graphObj->graph.edges[i].getSym()->getVertex()->getVertexId();
				int v1 = graphObj->graph.edges[i].getVertex()->getVertexId();


				temp.push_back(v0);
				temp.push_back(v1);
				temp.push_back(-1);
				temp.push_back(-1);

				edgeVertices.push_back(temp);
			}

			for (int i = 0; i < numVertices(); i++)
			{
				vector<int> cEdges;
				getConnectedEdges(i, zVertexData, cEdges);

				if (cEdges.size() == 1)
				{

					int currentId = cEdges[0];
					int prevId = graphObj->graph.edges[currentId].getPrev()->getEdgeId();

					zVector e_current = getEdgeVector( currentId);
					e_current.normalize();

					zVector e_prev = getEdgeVector( prevId);
					e_prev.normalize();

					zVector n_current = graphNormal ^ e_current;
					n_current.normalize();

					zVector n_prev = graphNormal ^ e_prev;
					n_prev.normalize();

					zVector v = graphObj->graph.vertexPositions[i];
					double w = width * 0.5;

					edgeVertices[currentId][3] = positions.size();
					positions.push_back(v + (n_current * w));

					edgeVertices[prevId][2] = positions.size();
					positions.push_back(v + (n_prev * w));

				}

				else
				{
					for (int j = 0; j < cEdges.size(); j++)
					{
						int currentId = cEdges[j];
						int prevId = graphObj->graph.edges[currentId].getPrev()->getEdgeId();

						zVector e_current = getEdgeVector( currentId);
						e_current.normalize();

						zVector e_prev = getEdgeVector( prevId);
						e_prev.normalize();

						zVector n_current = graphNormal ^ e_current;
						n_current.normalize();

						zVector n_prev = graphNormal ^ e_prev;
						n_prev.normalize();

						zVector norm = (n_current + n_prev) * 0.5;
						norm.normalize();

						zVector v = graphObj->graph.vertexPositions[i];
						double w = width * 0.5;

						edgeVertices[currentId][3] = positions.size();

						edgeVertices[prevId][2] = positions.size();

						positions.push_back(v + (norm * w));
					}
				}

			}

			for (int i = 0; i < edgeVertices.size(); i++)
			{

				for (int j = 0; j < edgeVertices[i].size(); j++)
				{
					polyConnects.push_back(edgeVertices[i][j]);
				}

				polyCounts.push_back(edgeVertices[i].size());

			}

		


			// mesh
			if (positions.size() > 0)
			{
				out.mesh = zMesh(positions, polyCounts, polyConnects);
			}

			
			return out;;
		}

		
		//--------------------------
		//---- TOPOLOGY MODIFIER METHODS
		//--------------------------
		

		/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
		*
		*	\param		[in]	index			- index of the edge to be split.
		*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
		*	\return				int				- index of the new vertex added after splitinng the edge.
		*	\since version 0.0.2
		*/
		int splitEdge(int index, double edgeFactor = 0.5)
		{
			if (index > graphObj->graph.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!graphObj->graph.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			
			zEdge* edgetoSplit = &graphObj->graph.edges[index];
			zEdge* edgetoSplitSym = edgetoSplit->getSym();

			zEdge* e_next = edgetoSplit->getNext();
			zEdge* e_prev = edgetoSplit->getPrev();

			zEdge* es_next = edgetoSplitSym->getNext();
			zEdge* es_prev = edgetoSplitSym->getPrev();


			zVector edgeDir = getEdgeVector(index);
			double  edgeLength = edgeDir.length();
			edgeDir.normalize();


			zVector v0 =	getVertexPosition( getSym(index)->getVertex()->getVertexId());
			zVector newVertPos = v0 + edgeDir * edgeFactor * edgeLength;
					


			// check if vertex exists if not add new vertex
			int VertId;
			bool vExists = vertexExists(newVertPos, VertId);
			if (!vExists)
			{
				graphObj->graph.addVertex(newVertPos);
				VertId = graphObj->graph.vertexActive.size() - 1;
			}

			//printf("\n newVert: %1.2f %1.2f %1.2f   %s ", newVertPos.x, newVertPos.y, newVertPos.z, (vExists)?"true":"false");

			if (!vExists)
			{
				// remove from verticesEdge map
				graphObj->graph.removeFromVerticesEdge(edgetoSplit->getVertex()->getVertexId(), edgetoSplitSym->getVertex()->getVertexId());

				// add new edges
				int v1 = VertId;
				int v2 = edgetoSplit->getVertex()->getVertexId();
				bool edgesResize = graphObj->graph.addEdges(v1, v2);

				// recompute pointers if resize is true
				if (edgesResize)
				{
					edgetoSplit = &graphObj->graph.edges[index];
					edgetoSplitSym = edgetoSplit->getSym();

					e_next = edgetoSplit->getNext();
					e_prev = edgetoSplit->getPrev();

					es_next = edgetoSplitSym->getNext();
					es_prev = edgetoSplitSym->getPrev();

					//printf("\n working!");

				}

				bool v2_val1 = checkVertexValency(v2, 1);

				// update vertex pointers
				graphObj->graph.vertices[v1].setEdge(&graphObj->graph.edges[graphObj->graph.edgeActive.size() - 2]);
				graphObj->graph.vertices[v2].setEdge(&graphObj->graph.edges[graphObj->graph.edgeActive.size() - 1]);

				//// update pointers
				edgetoSplit->setVertex(&graphObj->graph.vertices[VertId]);			// current edge vertex pointer updated to new added vertex

				graphObj->graph.edges[graphObj->graph.edgeActive.size() - 1].setNext(edgetoSplitSym);		// new added edge next pointer to point to the next of current edge
				if(!v2_val1) graphObj->graph.edges[graphObj->graph.edgeActive.size() - 1].setPrev(es_prev);
				else graphObj->graph.edges[graphObj->graph.edgeActive.size() - 1].setPrev(&graphObj->graph.edges[graphObj->graph.edgeActive.size() - 2]);

				graphObj->graph.edges[graphObj->graph.edgeActive.size() - 2].setPrev(edgetoSplit);
				if (!v2_val1) graphObj->graph.edges[graphObj->graph.edgeActive.size() - 2].setNext(e_next);


				// update verticesEdge map
				graphObj->graph.addToVerticesEdge(edgetoSplitSym->getVertex()->getVertexId(), edgetoSplit->getVertex()->getVertexId(), edgetoSplit->getEdgeId());

			}

			
			return VertId;
		}

		

	};

}