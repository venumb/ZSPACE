#pragma once


#include<headers/api/object/zObjGraph.h>

#include<headers/api/functionsets/zFn.h>
#include<headers/api/functionsets/zFnMesh.h>

#include<headers/api/iterators/zItGraph.h>

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

			vector<vector<int>> edgeVerts;

			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				vector<int> verts;
				e.getVertices(verts);

				edgeVerts.push_back(verts);
			}

			graphObj->graph.setStaticEdgeVertices(edgeVerts);
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
		void removeInactive(zHEData type)
		{
			//  Vertex
			if (type == zVertexData)
			{
				zItVertex v = graphObj->graph.vertices.begin();

				while (v != graphObj->graph.vertices.end())
				{
					bool active = v->isActive();

					if (!active)
					{
						graphObj->graph.vertices.erase(v++);

						graphObj->graph.n_v--;
					}
				}
			
				graphObj->graph.indexElements(zVertexData);

				printf("\n removed inactive vertices. ");

			}

			//  Edge
			else if (type == zEdgeData || type == zHalfEdgeData)
			{

				zItHalfEdge he = graphObj->graph.halfEdges.begin();

				while (he != graphObj->graph.halfEdges.end())
				{
					bool active = he->isActive();

					if (!active)
					{
						graphObj->graph.halfEdges.erase(he++);

						graphObj->graph.n_he--;
					}
				}

				zItEdge e = graphObj->graph.edges.begin();

				while (e != graphObj->graph.edges.end())
				{
					bool active = e->isActive();

					if (!active)
					{
						graphObj->graph.edges.erase(e++);

						graphObj->graph.n_e--;
					}
				}

				printf("\n removed inactive edges. ");

				graphObj->graph.indexElements(zHalfEdgeData);
				graphObj->graph.indexElements(zEdgeData);

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
		*	\return 		bool			- true if the file was read succesfully.
		*	\since version 0.0.2
		*/
		bool fromTXT(string infilename)
		{
			vector<zVector>positions;
			vector<int>edgeConnects;


			ifstream myfile;
			myfile.open(infilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return false;

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


			if (!planarGraph) graphObj->graph.create(positions, edgeConnects);;
			if (planarGraph)
			{
				graphNormal.normalize();

				zVector x(1, 0, 0);
				zVector sortRef = graphNormal ^ x;

				graphObj->graph.create(positions, edgeConnects, graphNormal, sortRef);
			}
			printf("\n graphObj->graph: %i %i ", numVertices(), numEdges());

			return true;
		}

		/*! \brief This method imports zGraph from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		inGraph				- graph created from the JSON file.
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\return 		bool			- true if the file was read succesfully.
		*	\since version 0.0.2
		*/
		bool fromJSON(string infilename)
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
				return false;
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

			graphObj->graph.edges.clear();

			// update  graph

			graphObj->graph.clear();

			graphObj->graph.vertices.assign(graphJSON.vertices.size(), zVertex());
			graphObj->graph.halfEdges.assign(graphJSON.halfedges.size(), zHalfEdge());
			graphObj->graph.edges.assign(floor(graphJSON.halfedges.size()*0.5), zEdge());
			
			graphObj->graph.vHandles.assign(graphJSON.vertices.size(), zVertexHandle());
			graphObj->graph.eHandles.assign(floor(graphJSON.halfedges.size()*0.5), zEdgeHandle());
			graphObj->graph.heHandles.assign(graphJSON.halfedges.size(), zHalfEdgeHandle());

			int n_v = 0; 
			for (zItGraphVertex v(*graphObj); !v.end(); v.next())
			{
				v.setId(n_v);			

				if (graphJSON.vertices[n_v] != -1)
				{
					zItGraphHalfEdge e(*graphObj, graphJSON.vertices[n_v]);;
					v.setHalfEdge(e);

					graphObj->graph.vHandles[n_v].he = graphJSON.vertices[n_v];
				}



				n_v++;
			}
			graphObj->graph.setNumVertices(n_v);
		
			int n_he = 0;
			int n_e = 0;

			for (zItGraphHalfEdge he(*graphObj); !he.end(); he.next())
			{

				// Half Edge
				he.setId(n_he);

				if (graphJSON.halfedges[n_he][0] != -1)
				{
					zItGraphHalfEdge e(*graphObj, graphJSON.halfedges[n_he][0]);
					he.setPrev(e);

					graphObj->graph.heHandles[n_he].p = graphJSON.halfedges[n_he][0];
				}

				if (graphJSON.halfedges[n_he][1] != -1)
				{
					zItGraphHalfEdge e(*graphObj, graphJSON.halfedges[n_he][1]);
					he.setNext(e);

					graphObj->graph.heHandles[n_he].n = graphJSON.halfedges[n_he][1];
				}

				if (graphJSON.halfedges[n_he][2] != -1)
				{
					zItGraphVertex v(*graphObj, graphJSON.halfedges[n_he][2]);
					he.setVertex(v);

					graphObj->graph.heHandles[n_he].v = graphJSON.halfedges[n_he][2];
				}

			

				// symmetry half edges
				if (n_he % 2 == 0)
				{
					zItGraphHalfEdge e(*graphObj, n_he + 1);
					he.setSym(e);
				}
				else
				{
					zItGraphHalfEdge e(*graphObj, n_he - 1);
					he.setSym(e);
				}


				// Edge
				if (n_he % 2 == 1)
				{
					zItGraphEdge e(*graphObj, n_e);

					zItGraphHalfEdge heSym = he.getSym();

					e.setHalfEdge(heSym, 0);
					e.setHalfEdge(he, 1);								   				

					graphObj->graph.heHandles[n_he].e = n_e;

					n_e++;
				}

				n_he++;

			}

			graphObj->graph.setNumEdges(n_e);

			

			// Vertex Attributes
			graphJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
			//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

			graphObj->graph.vertexPositions.clear();			
			graphObj->graph.vertexColors.clear();
			graphObj->graph.vertexWeights.clear();
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

						graphObj->graph.vertexWeights.push_back(2.0);

						k += 5;
					}
				}
			}


			// Edge Attributes
			graphJSON.halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();
			
			graphObj->graph.edgeColors.clear();
			graphObj->graph.edgeWeights.clear();
			if (graphJSON.halfedgeAttributes.size() == 0)
			{	
				for (int i = 0; i < graphObj->graph.n_e; i++)
				{
					graphObj->graph.edgeColors.push_back(zColor());
					graphObj->graph.edgeWeights.push_back(1.0);
				}
			}
			else
			{
				for (int i = 0; i < graphJSON.halfedgeAttributes.size(); i+=2)
				{
					// color
					if (graphJSON.halfedgeAttributes[i].size() == 3)
					{
						zColor col(graphJSON.halfedgeAttributes[i][0], graphJSON.halfedgeAttributes[i][1], graphJSON.halfedgeAttributes[i][2], 1);

						graphObj->graph.edgeColors.push_back(col);
						graphObj->graph.edgeWeights.push_back(1.0);

					}
				}
			}

		

			printf("\n graph: %i %i ", numVertices(), numEdges());

			// add to maps 
			for (int i = 0; i < graphObj->graph.vertexPositions.size(); i++)
			{
				graphObj->graph.addToPositionMap(graphObj->graph.vertexPositions[i], i);
			}


			
			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				int v1 = e.getHalfEdge(0).getVertex().getId();
				int v2 = e.getHalfEdge(1).getVertex().getId();

				graphObj->graph.addToHalfEdgesMap(v1, v2, e.getHalfEdge(0).getId());
			}

			return true;

		}


		/*! \brief This method exports zGraph to a TXT file format.
		*
		*	\param [in]		inGraph				- input graph.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toTXT(string outfilename)
		{
			// remove inactive elements
			if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);


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
			for (auto &vPos : graphObj->graph.vertexPositions)
			{
				
				myfile << "\n v " << vPos.x << " " << vPos.y << " " << vPos.z;

			}

			myfile << "\n ";

			// edge connectivity
			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				int v1 = e.getHalfEdge(0).getVertex().getId();
				int v2 = e.getHalfEdge(1).getVertex().getId();

				myfile << "\n e ";

				myfile << v1 << " ";
				myfile << v2;
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
		void toJSON(string outfilename)
		{
			// remove inactive elements
			if (numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);


			// output file
			zUtilsJsonHE graphJSON;
			json j;

			// create json

			// Vertices
			for (zItGraphVertex v(*graphObj); !v.end(); v.next())
			{
				if (v.getHalfEdge().isActive()) graphJSON.vertices.push_back(v.getHalfEdge().getId());
				else graphJSON.vertices.push_back(-1);

			}

			//Edges
			for (zItGraphHalfEdge he(*graphObj); !he.end(); he.next())
			{
				vector<int> HE_edges;

				if (he.getPrev().isActive()) HE_edges.push_back(he.getPrev().getId());
				else HE_edges.push_back(-1);

				if (he.getNext().isActive()) HE_edges.push_back(he.getNext().getId());
				else HE_edges.push_back(-1);

				if (he.getVertex().isActive()) HE_edges.push_back(he.getVertex().getId());
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
		zFnGraph() 
		{
			fnType = zFnType::zGraphFn;
			graphObj = nullptr;			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_graphObj			- input graph object.
		*	\since version 0.0.2
		*/
		zFnGraph(zObjGraph &_graphObj, bool  _planarGraph = false, zVector _graphNormal = zVector(0,0,1) )
		{
			fnType = zFnType::zGraphFn;

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
			graphObj->graph.clear();				
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
			graphObj->graph.create(_positions, edgeConnects);

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

			graphObj->graph.create(_positions, edgeConnects, graphNormal, sortRef);

			if (staticGraph) setStaticContainers();
		}

		/*! \brief This method creates a graph from a mesh.
		*
		*	\param		[in]	graphObj			- input mesh object.	
		*	\param		[in]	staticGraph		- makes the graph fixed. Computes the static edge vertex positions if true.
		*	\since version 0.0.2
		*/
		void createFromMesh(zObjMesh &graphObj, bool staticGraph = false)
		{
			zFnMesh fnMesh(graphObj);

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

		/*! \brief This method returns the number of vertices in the graph.
		*	\return				number of vertices.
		*	\since version 0.0.2
		*/
		int numVertices()
		{
			return graphObj->graph.n_v;
		}

		/*! \brief This method returns the number of edges in the graph.
		*	\return				number of edges.
		*	\since version 0.0.2
		*/
		int numEdges()
		{
			return graphObj->graph.n_e;
		}

		/*! \brief This method returns the number of half edges in the graph .
		*	\return				number of half edges.
		*	\since version 0.0.2
		*/
		int numHalfEdges()
		{
			return graphObj->graph.n_he;
		}

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\return		[out]		bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zVector pos, int &outVertexId, int precisionfactor = 6)
		{		
			return graphObj->graph.vertexExists(pos, outVertexId, precisionfactor);
		}
	
		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outHalfEdge	- stores halfedgeId if the vertex exists else it is -1.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool halfEdgeExists(int v1, int v2, int &outHalfEdge)
		{			

			return graphObj->graph.halfEdgeExists(v1,v2, outHalfEdge);
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

			for(zItGraphEdge e(*graphObj); !e.end();e.next())
			{
				if (e.isActive())
				{
					int v0 = e.getHalfEdge(0).getVertex().getId();
					int v1 = e.getHalfEdge(1).getVertex().getId();

					zColor col;
					col.r = (graphObj->graph.vertexColors[v0].r + graphObj->graph.vertexColors[v1].r) * 0.5;
					col.g = (graphObj->graph.vertexColors[v0].g + graphObj->graph.vertexColors[v1].g) * 0.5;
					col.b = (graphObj->graph.vertexColors[v0].b + graphObj->graph.vertexColors[v1].b) * 0.5;
					col.a = (graphObj->graph.vertexColors[v0].a + graphObj->graph.vertexColors[v1].a) * 0.5;

					if (graphObj->graph.edgeColors.size() <= e.getId()) graphObj->graph.edgeColors.push_back(col);
					else graphObj->graph.edgeColors[e.getId()] = col;

					
				}
				

			}

		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.2
		*/
		void computeVertexColorfromEdgeColor()
		{		
			for (zItGraphVertex v(*graphObj); !v.end(); v.next())			
			{
				if (v.isActive())
				{
					vector<int> cEdges;
					v.getConnectedHalfEdges(cEdges);

					zColor col;
					for (int j = 0; j < cEdges.size(); j++)
					{
						col.r += graphObj->graph.edgeColors[cEdges[j]].r;
						col.g += graphObj->graph.edgeColors[cEdges[j]].g;
						col.b += graphObj->graph.edgeColors[cEdges[j]].b;
					}

					col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

					graphObj->graph.vertexColors[v.getId()] = col;

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

				for (zItGraphVertex v(*graphObj); !v.end(); v.next())
				{
					tempVertPos.push_back(graphObj->graph.vertexPositions[v.getId()]);

					if (v.isActive())
					{
						if (!v.checkVertexValency(1))
						{
							vector<int> cVerts;

							v.getConnectedVertices(cVerts);

							for (int j = 0; j < cVerts.size(); j++)
							{
								zVector p = graphObj->graph.vertexPositions[cVerts[j]];
								tempVertPos[v.getId()] += p;
							}

							tempVertPos[v.getId()] /= (cVerts.size() + 1);
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
			if (type == zVertexData || type == zEdgeData || type == zHalfEdgeData) removeInactive(type);
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


		/*! \brief This method sets vertex color of all the vertices to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColor(zColor col, bool setEdgeColor = false)
		{
			graphObj->graph.vertexColors.clear();
			graphObj->graph.vertexColors.assign(graphObj->graph.n_v, col);

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
			if (graphObj->graph.vertexColors.size() != graphObj->graph.vertices.size())
			{
				graphObj->graph.vertexColors.clear();
				for (int i = 0; i < graphObj->graph.vertices.size(); i++) graphObj->graph.vertexColors.push_back(zColor(1, 0, 0, 1));
			}

			if (col.size() != graphObj->graph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

			for (int i = 0; i < graphObj->graph.vertexColors.size(); i++)
			{
				graphObj->graph.vertexColors[i] = col[i];
			}

			if (setEdgeColor) computeEdgeColorfromVertexColor();
		}
				

		/*! \brief This method sets edge color of all the edges to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColor(zColor col, bool setVertexColor = false)
		{
			
			graphObj->graph.edgeColors.clear();
			graphObj->graph.edgeColors.assign(graphObj->graph.n_e, col);			

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


		/*! \brief This method sets edge weight of all the edges to the input weight.
		*
		*	\param		[in]	wt				- input weight.
		*	\since version 0.0.2
		*/
		void setEdgeWeight(double wt)
		{
			graphObj->graph.edgeWeights.clear();
			graphObj->graph.edgeWeights.assign(graphObj->graph.n_e, wt);			

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
		*	\param		[in]	type					- zEdgeData or zHalfEdgeData.
		*	\param		[out]	centers					- vector of centers of type zVector.
		*	\since version 0.0.2
		*/
		void getCenters(zHEData type, vector<zVector> &centers)
		{
			// graph Edge 
			if (type == zHalfEdgeData)
			{

				centers.clear();

				for (zItGraphHalfEdge he(*graphObj); !he.end(); he.next())
				{
					if (he.isActive())
					{
						centers.push_back(he.getCenter());
					}
					else
					{
						centers.push_back(zVector());

					}
				}

			}
			else if (type == zEdgeData)
			{

				centers.clear();

				for (zItGraphEdge e(*graphObj); !e.end(); e.next())
				{
					if (e.isActive())
					{
						centers.push_back(e.getCenter());
					}
					else
					{
						centers.push_back(zVector());

					}
				}

			}
			
			else throw std::invalid_argument(" error: invalid zHEData type");
		}


		/*! \brief This method computes the lengths of all the half edges of a the graph.
		*
		*	\param		[out]	halfEdgeLengths				- vector of halfedge lengths.
		*	\return				double						- total edge lengths.
		*	\since version 0.0.2
		*/
		double getHalfEdgeLengths(vector<double> &halfEdgeLengths)
		{
			double total = 0.0;


			halfEdgeLengths.clear();

			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				if (e.isActive())
				{
					double e_len = e.getEdgeLength();

					halfEdgeLengths.push_back(e_len);
					halfEdgeLengths.push_back(e_len);

					total += e_len;
				}
				else
				{
					halfEdgeLengths.push_back(0);
					halfEdgeLengths.push_back(0);
				}
			}

			return total;
		}

		/*! \brief This method computes the lengths of all the  edges of a the graph.
		*
		*	\param		[out]	EdgeLengths		- vector of edge lengths.
		*	\return				double				- total edge lengths.
		*	\since version 0.0.2
		*/
		double getEdgeLengths(vector<double> &edgeLengths)
		{
			double total = 0.0;


			edgeLengths.clear();

			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				if (e.isActive())
				{
					double e_len = e.getEdgeLength();
					edgeLengths.push_back(e_len);
					total += e_len;
				}
				else
				{
					edgeLengths.push_back(0);
				}
			}

			return total;
		}

		/*! \brief This method stores graph edge connectivity information in the input containers
		*
		*	\param		[out]	edgeConnects	- stores list of esdge connection with vertex ids for each edge.
		*	\since version 0.0.2
		*/
		void getEdgeData(vector<int> &edgeConnects)
		{
			edgeConnects.clear();

			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{
				edgeConnects.push_back(e.getHalfEdge(0).getVertex().getId());
				edgeConnects.push_back(e.getHalfEdge(1).getVertex().getId());
			}
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

			if(numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);

			vector<zVector> positions;
			vector<int> edgeConnects;


			positions = graphObj->graph.vertexPositions;
			getEdgeData(edgeConnects);			


			if (planarGraph)
			{
				graphNormal.normalize();

				zVector x(1, 0, 0);
				zVector sortRef = graphNormal ^ x;

				out.graph.create(positions, edgeConnects, graphNormal, sortRef);
			}
			else out.graph.create(positions, edgeConnects);

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

			if(numVertices() != graphObj->graph.vertices.size()) removeInactiveElements(zVertexData);
			if (numEdges() != graphObj->graph.edges.size()) removeInactiveElements(zEdgeData);

			positions = graphObj->graph.vertexPositions;

			vector<vector<int>> edgeVertices;
			for (zItGraphEdge e(*graphObj); !e.end(); e.next())
			{

				int v0 = e.getHalfEdge(1).getVertex().getId();
				int v1 = e.getHalfEdge(0).getVertex().getId();
		
				vector<int> temp;	
				temp.push_back(v0);
				temp.push_back(v1);
				temp.push_back(-1);
				temp.push_back(-1);

				edgeVertices.push_back(temp);
			}

			for (zItGraphVertex v(*graphObj); !v.end(); v.next())
			{
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				if (cEdges.size() == 1)
				{

					int currentId = cEdges[0].getId();
					int prevId = cEdges[0].getPrev().getId();

					zVector e_current = cEdges[0].getHalfEdgeVector();
					e_current.normalize();

					zVector e_prev = cEdges[0].getPrev().getHalfEdgeVector();
					e_prev.normalize();

					zVector n_current = graphNormal ^ e_current;
					n_current.normalize();

					zVector n_prev = graphNormal ^ e_prev;
					n_prev.normalize();

			
					double w = width * 0.5;

					edgeVertices[currentId][3] = positions.size();
					positions.push_back(v.getVertexPosition() + (n_current * w));

					edgeVertices[prevId][2] = positions.size();
					positions.push_back(v.getVertexPosition() + (n_prev * w));

				}

				else
				{
					for (int j = 0; j < cEdges.size(); j++)
					{
						int currentId = cEdges[j].getId();
						int prevId = cEdges[j].getPrev().getId();
						

						zVector e_current = cEdges[0].getHalfEdgeVector();
						e_current.normalize();

						zVector e_prev = cEdges[0].getPrev().getHalfEdgeVector();
						e_prev.normalize();

						zVector n_current = graphNormal ^ e_current;
						n_current.normalize();

						zVector n_prev = graphNormal ^ e_prev;
						n_prev.normalize();

						zVector norm = (n_current + n_prev) * 0.5;
						norm.normalize();

						
						double w = width * 0.5;

						edgeVertices[currentId][3] = positions.size();

						edgeVertices[prevId][2] = positions.size();

						positions.push_back(v.getVertexPosition() + (norm * w));
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
				out.mesh.create(positions, polyCounts, polyConnects);
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
			//if (index >= numEdges()) throw std::invalid_argument(" error: index out of bounds.");
			//zItEdge e = graphObj->graph.indexToEdge[index];
			//if (!e->isActive()) throw std::invalid_argument(" error: index out of bounds.");

			//
			//zItHalfEdge edgetoSplit = e->halfEdges[0];
			//zItHalfEdge edgetoSplitSym = edgetoSplit->sym;

			//zItHalfEdge e_next = edgetoSplit->next;
			//zItHalfEdge e_prev = edgetoSplit->prev;

			//zItHalfEdge es_next = edgetoSplitSym->next;
			//zItHalfEdge es_prev = edgetoSplitSym->prev;


			//zVector edgeDir = getHalfEdgeVector(index);
			//double  edgeLength = edgeDir.length();
			//edgeDir.normalize();


			//zVector v0 =	getVertexPosition( getSym(index)->v->index);
			//zVector newVertPos = v0 + edgeDir * edgeFactor * edgeLength;
			//		


			//// check if vertex exists if not add new vertex
			//int VertId;
			//bool vExists = vertexExists(newVertPos, VertId);
			//if (!vExists)
			//{
			//	graphObj->graph.addVertex(newVertPos);
			//	VertId =numVertices() - 1;
			//}

			////printf("\n newVert: %1.2f %1.2f %1.2f   %s ", newVertPos.x, newVertPos.y, newVertPos.z, (vExists)?"true":"false");

			//if (!vExists)
			//{
			//	// remove from verticesEdge map
			//	graphObj->graph.removeFromHalfEdgesMap(edgetoSplit->v->index, edgetoSplitSym->v->index);

			//	// add new edges
			//	int v1 = VertId;
			//	int v2 = edgetoSplit->v->index;
			//	graphObj->graph.addEdges(v1, v2);				

			//	bool v2_val1 = checkVertexValency(v2, 1);

			//	// update vertex pointers
			//	zItVertex vIter1 = graphObj->graph.indexToVertex[v1];
			//	zItVertex vIter2 = graphObj->graph.indexToVertex[v2];

			//	zItHalfEdge he1 = graphObj->graph.indexToHalfEdge[numHalfEdges() - 2];
			//	zItHalfEdge he2 = graphObj->graph.indexToHalfEdge[numHalfEdges() - 1];

			//	vIter1->e = he1;
			//	vIter2->e = he2;
			//	

			//	//// update pointers

			//	zItVertex vIter = graphObj->graph.indexToVertex[VertId];
			//	edgetoSplit->v = vIter;				// current edge vertex pointer updated to new added vertex

			//	he2->next = edgetoSplitSym;		// new added edge next pointer to point to the next of current edge
			//	
			//	if (!v2_val1)
			//	{
			//		he2->prev = es_prev;
			//		es_prev->next = he2;
			//	}
			//	else
			//	{
			//		he2->prev = he1; 
			//		he1->next = he2;
			//	}

			//	he1->prev = edgetoSplit;
			//	edgetoSplit->next = he1;

			//	
			//	if (!v2_val1)
			//	{
			//		he1->next = e_next;
			//		e_next->prev = he1;					
			//	}


			//	// update verticesEdge map
			//	graphObj->graph.addToHalfEdgesMap(edgetoSplitSym->v->index, edgetoSplit->v->index, edgetoSplit);

			//}

			//
			//return VertId;
		}

		
		//--------------------------
		//---- TRANSFORM METHODS OVERRIDES
		//--------------------------

		virtual void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) override
		{
			if (updatePositions)
			{
				zTransformationMatrix to;
				to.setTransform(inTransform, decompose);

				zTransform transMat = graphObj->transformationMatrix.getToMatrix(to);
				transformObject(transMat);

				graphObj->transformationMatrix.setTransform(inTransform);

				// update pivot values of object transformation matrix
				zVector p = graphObj->transformationMatrix.getPivot();
				p = p * transMat;
				setPivot(p);

			}
			else
			{
				graphObj->transformationMatrix.setTransform(inTransform, decompose);

				zVector p = graphObj->transformationMatrix.getO();
				setPivot(p);

			}

		}

		virtual void setScale(double3 &scale) override
		{
			// get  inverse pivot translations
			zTransform invScalemat = graphObj->transformationMatrix.asInverseScaleTransformMatrix();

			// set scale values of object transformation matrix
			graphObj->transformationMatrix.setScale(scale);

			// get new scale transformation matrix
			zTransform scaleMat = graphObj->transformationMatrix.asScaleTransformMatrix();

			// compute total transformation
			zTransform transMat = invScalemat * scaleMat;

			// transform object
			transformObject(transMat);
		}

		virtual void setRotation(double3 &rotation, bool appendRotations = false) override
		{
			// get pivot translation and inverse pivot translations
			zTransform pivotTransMat = graphObj->transformationMatrix.asPivotTranslationMatrix();
			zTransform invPivotTransMat = graphObj->transformationMatrix.asInversePivotTranslationMatrix();

			// get plane to plane transformation
			zTransformationMatrix to = graphObj->transformationMatrix;
			to.setRotation(rotation, appendRotations);
			zTransform toMat = graphObj->transformationMatrix.getToMatrix(to);

			// compute total transformation
			zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

			// transform object
			transformObject(transMat);

			// set rotation values of object transformation matrix
			graphObj->transformationMatrix.setRotation(rotation, appendRotations);;
		}


		virtual void setTranslation(zVector &translation, bool appendTranslations = false) override
		{
			// get vector as double3
			double3 t;
			translation.getComponents(t);

			// get pivot translation and inverse pivot translations
			zTransform pivotTransMat = graphObj->transformationMatrix.asPivotTranslationMatrix();
			zTransform invPivotTransMat = graphObj->transformationMatrix.asInversePivotTranslationMatrix();

			// get plane to plane transformation
			zTransformationMatrix to = graphObj->transformationMatrix;
			to.setTranslation(t, appendTranslations);
			zTransform toMat = graphObj->transformationMatrix.getToMatrix(to);

			// compute total transformation
			zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

			// transform object
			transformObject(transMat);

			// set translation values of object transformation matrix
			graphObj->transformationMatrix.setTranslation(t, appendTranslations);;

			// update pivot values of object transformation matrix
			zVector p = graphObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}

		virtual void setPivot(zVector &pivot) override
		{
			// get vector as double3
			double3 p;
			pivot.getComponents(p);

			// set pivot values of object transformation matrix
			graphObj->transformationMatrix.setPivot(p);
		}

		virtual void getTransform(zTransform &transform) override
		{
			transform = graphObj->transformationMatrix.asMatrix();
		}


	protected:

		//--------------------------
		//---- PROTECTED OVERRIDE METHODS
		//--------------------------	


		virtual void transformObject(zTransform &transform) override
		{

			if (numVertices() == 0) return;


			zVector* pos = getRawVertexPositions();

			for (int i = 0; i < numVertices(); i++)
			{

				zVector newPos = pos[i] * transform;
				pos[i] = newPos;
			}

		}

	};


#ifndef DOXYGEN_SHOULD_SKIP_THIS


	

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}