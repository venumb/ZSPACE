
#pragma once

#include <headers/core/zVector.h>
#include <headers/core/zMatrix.h>
#include <headers/core/zColor.h>

#include <headers/core/zVectorMatrixUtilities.h>
#include <headers/core/zUtilities.h>

#include <headers/geometry/zGeometryDatatypes.h>

namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryClasses
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zGraph
	*	\brief A half edge graph class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zGraph
	{

	protected:

		//--------------------------
		//---- PROTCTED ATTRIBUTES
		//--------------------------

		/*!	\brief stores number of active vertices */
		int n_v;		
		
		/*!	\brief stores number of active edges */
		int n_e;										

		/*!	\brief stores number of max vertices, which is the size of the dynamic vertices array. */
		int max_n_v;

		/*!	\brief stores number of max edges, which is the size of the dynamic vertices array. */
		int max_n_e;									


	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------


		/*!	\brief vertex container */
		zVertex *vertices;

		/*!	\brief edge container	*/
		zEdge *edges;									

		/*!	\brief container which stores vertex positions.			*/
		vector<zVector> vertexPositions;				

		/*!	\brief vertices to edgeId map. Used to check if edge exists with the haskey being the vertex sequence.	 */
		unordered_map <string, int> verticesEdge;	

		/*!	\brief position to vertexId map. Used to check if vertex exists with the haskey being the vertex position.	 */
		unordered_map <string, int> positionVertex;		


		/*!	\brief container which stores vertex colors.	*/
		vector<zColor> vertexColors;				

		/*!	\brief container which stores edge colors.	*/
		vector<zColor> edgeColors;						

		/*!	\brief container which stores vertex weights.	*/
		vector <double> vertexWeights;	

		/*!	\brief container which stores edge weights.	*/
		vector	<double> edgeWeights;					

		/*!	\brief container which stores vertex status - true if active.	*/
		vector <bool> vertexActive;	

		/*!	\brief container which stores edge status - true if active..	*/
		vector<bool> edgeActive;						


		/*!	\brief stores the start vertex ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_VertexId;	

		/*!	\brief stores the start edge ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_EdgeId;		

		/*!	\brief stores the start vertex color ID in the VBO, when attache to the zBufferObject.	*/
		int VBO_VertexColorId;							


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		
		zGraph()
		{
			n_v = n_e = 0;

			vertices = NULL;
			edges = NULL;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_positions		- vector of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- edge connection list with vertex ids for each edge
		*	\since version 0.0.1
		*/
		
		zGraph(vector<zVector>(&_positions), vector<int>(&edgeConnects))
		{

			int _num_vertices = _positions.size();
			int _num_edges = edgeConnects.size();

			// set max size
			max_n_v = _num_vertices * 4;
			max_n_e = _num_edges * 8;

			if (_num_vertices != 0) vertices = new zVertex[max_n_v];
			if (_num_edges != 0) edges = new zEdge[max_n_e];

			n_v = n_e = 0;

			// temp containers
			struct temp_numConnectedEdgesPerVerts
			{
				int vertId;
				vector<int> temp_connectedEdges;
			};

			temp_numConnectedEdgesPerVerts *cEdgesperVert = new temp_numConnectedEdgesPerVerts[_num_vertices];

			// create vertices
			for (int i = 0; i < _num_vertices; i++)
			{
				addVertex(_positions[i]);
				cEdgesperVert[i].vertId = i;

			}

			// position attribute
			vertexPositions = (_positions);

			// create edges and update connections

			for (int i = 0; i < edgeConnects.size(); i += 2)
			{

				addEdges(edgeConnects[i], edgeConnects[i + 1]);

				cEdgesperVert[edgeConnects[i]].temp_connectedEdges.push_back(n_e - 2);
				cEdgesperVert[edgeConnects[i + 1]].temp_connectedEdges.push_back(n_e - 1);

				vertices[edgeConnects[i]].setEdge(&edges[n_e - 2]);
				vertices[edgeConnects[i + 1]].setEdge(&edges[n_e - 1]);

			}

			// update pointers
			for (int i = 0; i < n_v; i++)
			{
				if (cEdgesperVert[i].temp_connectedEdges.size() > 0)
				{
					zVector cen = vertexPositions[i];
					vector<int> sorted_cEdges;
					cyclic_sortEdges(cEdgesperVert[i].temp_connectedEdges, cen, cEdgesperVert[i].temp_connectedEdges[0], sorted_cEdges);

					if (sorted_cEdges.size() > 0)
					{
						for (int j = 0; j < sorted_cEdges.size(); j++)
						{
							edges[sorted_cEdges[j]].setPrev(edges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]].getSym());
						}
					}
				}

			}


		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		
		~zGraph() {}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method gets the edges connected to input zVertex or zEdge.
		*
		*	\param		[in]	index			- index in the vertex/edge list.
		*	\param		[in]	type			- zVertexData or zEdgeData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.1
		*/
		
		void getConnectedEdges(int index, zHEData type, vector<int>& edgeIndicies)
		{
			vector<int> out;

			//  Vertex 
			if (type == zVertexData)
			{
				if (vertices[index].getEdge())
				{
					zEdge* start = vertices[index].getEdge();
					zEdge* e = start;

					bool exit = false;

					do
					{
						out.push_back(e->getEdgeId());
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
				getConnectedEdges(edges[index].getVertex()->getVertexId(), zVertexData, connectedEdgestoVert0);

				vector<int> connectedEdgestoVert1;
				getConnectedEdges(edges[index].getSym()->getVertex()->getVertexId(), zVertexData, connectedEdgestoVert1);

				for (int i = 0; i < connectedEdgestoVert0.size(); i++)
				{
					if (connectedEdgestoVert0[i] != index) out.push_back(connectedEdgestoVert0[i]);
				}


				for (int i = 0; i < connectedEdgestoVert1.size(); i++)
				{
					if (connectedEdgestoVert1[i] != index) out.push_back(connectedEdgestoVert1[i]);
				}
			}

			else  throw std::invalid_argument(" error: invalid zHEData type");

			edgeIndicies = out;
		}

		/*! \brief This method gets the vertices connected to input zVertex.
		*
		*	\param		[in]	index			- index in the vertex list.
		*	\param		[in]	type			- zVertexData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.1
		*/
		
		void getConnectedVertices(int index, zHEData type, vector<int>& vertexIndicies)
		{
			vector<int> out;

			// Graph Vertex
			if (type == zVertexData)
			{

				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				for (int i = 0; i < connectedEdges.size(); i++)
				{
					out.push_back(edges[connectedEdges[i]].getVertex()->getVertexId());
				}

			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			vertexIndicies = out;
		}

		/*!	\brief This method gets the vertices attached to input zEdge.
		*
		*	\param		[in]	index			- index in the edge list.
		*	\param		[in]	type			- zEdgeData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.1
		*/
		
		void getVertices(int index, zHEData type, vector<int>& vertexIndicies)
		{
			vector<int> out;

			// Graph Edge or  Mesh Edge
			if (type == zEdgeData)
			{
				out.push_back(edges[index].getVertex()->getVertexId());
				out.push_back(edges[index].getSym()->getVertex()->getVertexId());
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			vertexIndicies = out;
		}

		/*!	\brief This method calculate the valency of the input zVertex.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- valency of input vertex input valence.
		*	\since version 0.0.1
		*/
		
		int getVertexValence(int index, zHEData type)
		{
			int out;

			if (type == zVertexData)
			{
				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				out = connectedEdges.size();
			}			

			else throw std::invalid_argument(" error: invalid zHEData type");

			return out;
		}

		/*!	\brief This method determines if input zVertex valency is equal to the input valence number.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	valence	- input valence value.
		*	\param		[in]	type	- zVertexData.
		*	\return				bool	- true if valency is equal to input valence.
		*	\since version 0.0.1
		*/
		
		bool checkVertexValency(int index, zHEData type = zVertexData, int valence = 1)
		{
			bool out = false;

			if (type == zVertexData)
			{
				out = (getVertexValence(index, type) == valence) ? true: false;
			}
			else throw std::invalid_argument(" error: invalid zHEData type");

			return out;
		}

		//--------------------------
		//--- ATTRIBUTE METHODS 
		//--------------------------

		/*! \brief This method computes the Edge colors based on the vertex colors.
		*
		*	\since version 0.0.1
		*/
		
		void computeEdgeColorfromVertexColor()			
		{

			for (int i = 0; i < n_e; i += 2)
			{
				zColor col;
				col.r = (vertexColors[edges[i].getVertex()->getVertexId()].r + vertexColors[edges[i + 1].getVertex()->getVertexId()].r) * 0.5;
				col.g = (vertexColors[edges[i].getVertex()->getVertexId()].g + vertexColors[edges[i + 1].getVertex()->getVertexId()].g) * 0.5;
				col.b = (vertexColors[edges[i].getVertex()->getVertexId()].b + vertexColors[edges[i + 1].getVertex()->getVertexId()].b) * 0.5;
				col.a = (vertexColors[edges[i].getVertex()->getVertexId()].a + vertexColors[edges[i + 1].getVertex()->getVertexId()].a) * 0.5;

				edgeColors[i] = col;
				edgeColors[i + 1] = col;
			}



		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.1
		*/
		
		void computeVertexColorfromEdgeColor()
		{
			for (int i = 0; i < vertexActive.size(); i++)
			{
				if (vertexActive[i])
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					zColor col;
					for (int j = 0; j < cEdges.size(); j++)
					{
						col.r += edgeColors[cEdges[j]].r;
						col.g += edgeColors[cEdges[j]].g;
						col.b += edgeColors[cEdges[j]].b;
					}

					col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

					vertexColors[i] = col;

				}
			}
		}


		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		/*! \brief This method averages the positions of vertex except for the ones on the boundary.
		*
		*	\param		[in]	numSteps	- number of times the averaging is carried out.
		*	\since version 0.0.1
		*/
		
		void averageVertices(int numSteps = 1)
		{
			for (int k = 0; k < numSteps; k++)
			{
				vector<zVector> tempVertPos;

				for (int i = 0; i < n_v; i++)
				{
					tempVertPos.push_back(vertexPositions[i]);

					if (!checkVertexValency(i,zVertexData, 1))
					{
						vector<int> cVerts;

						getConnectedVertices(i, zVertexData, cVerts);

						for (int j = 0; j < cVerts.size(); j++)
						{
							zVector p = vertexPositions[cVerts[j]];
							tempVertPos[i] += p;
						}

						tempVertPos[i] /= (cVerts.size() + 1);
					}
				}

				// update position
				for (int i = 0; i < n_v; i++) vertexPositions[i] = tempVertPos[i];
			}

		}


		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\since version 0.0.1
		*/
		void addVertex(zVector &pos)
		{
			if (max_n_v - vertexActive.size() < 2)
			{
				max_n_v *= 4;
				resizeArray(max_n_v, zVertexData);
			}

			string hashKey = (to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z));
			positionVertex[hashKey] = vertexActive.size();

			vertices[vertexActive.size()] = zVertex();
			vertices[vertexActive.size()].setVertexId(vertexActive.size());

			vertexPositions.push_back(pos);

			// default Attibute values
			vertexActive.push_back(true);
			vertexColors.push_back(zColor(1, 0, 1, 1));
			vertexWeights.push_back(2.0);

			n_v++;
		}

		/*! \brief This method deletes the zGraph vertices given in the input vertex list.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\since version 0.0.1
		*/
		
		void deleteVertex(vector<int> &vertexList)
		{
			for (int i = 0; i < vertexList.size(); i++)
			{
				if (vertexActive[vertexList[i]])
				{
					// get connected edges
					vector<int> cEdges;
					getConnectedEdges(vertexList[i], zVertexData, cEdges);

					// update edge pointers
					for (int k = 0; k < cEdges.size(); k++)
					{
						if (!checkVertexValency(edges[cEdges[k]].getVertex()->getVertexId(),zVertexData, 1))
						{
							edges[cEdges[k]].getNext()->setPrev(edges[cEdges[k]].getPrev());
						}

						else
						{
							edges[cEdges[k]].getPrev()->setNext(edges[cEdges[k]].getSym()->getNext());
							vertexActive[edges[cEdges[k]].getVertex()->getVertexId()] = false;
							n_v--;
						}

						edgeActive[edges[cEdges[k]].getEdgeId()] = false;
						edgeActive[edges[cEdges[k]].getSym()->getEdgeId()] = false;
						n_e -= 2;

					}

					vertexActive[vertexList[i]] = false;
					n_v--;
				}
			}
		}

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\return		[out]		bool		- true if vertex exists else false.
		*	\since version 0.0.1
		*/
		
		bool vertexExists(zVector pos, int &outVertexId)
		{
			bool out = false;;


			string hashKey = (to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z));
			std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


			if (got != positionVertex.end())
			{
				out = true;
				outVertexId = got->second;
			}


			return out;
		}

		/*! \brief This method calculates the number of vertices in zGraph or zMesh
		*	\return				number of vertices.
		*	\since version 0.0.1
		*/
		
		int numVertices()
		{
			return n_v;
		}


		//--------------------------
		//---- EDGE METHODS
		//--------------------------

		/*! \brief This method adds an edge and its symmetry edge to the edges array.
		*
		*	\param		[in]	v1			- start zVertex of the edge.
		*	\param		[in]	v2			- end zVertex of the edge.
		*	\since version 0.0.1
		*/
		
		void addEdges(int &v1, int &v2)
		{
			if (max_n_e - edgeActive.size() < 4)
			{
				max_n_e *= 4;
				resizeArray(max_n_e, zEdgeData);

			}

			string hashKey = (to_string(v1) + "," + to_string(v2));
			verticesEdge[hashKey] = edgeActive.size();

			edges[edgeActive.size()] = zEdge();

			edges[edgeActive.size()].setEdgeId(edgeActive.size());
			edges[edgeActive.size()].setVertex(&vertices[v2]);

			n_e++;

			// default color and weights
			edgeActive.push_back(true);
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);

			// SYMMETRY edge

			string hashKey1 = (to_string(v2) + "," + to_string(v1));
			verticesEdge[hashKey1] = edgeActive.size();

			edges[edgeActive.size()] = zEdge();

			edges[edgeActive.size()].setEdgeId(edgeActive.size());
			edges[edgeActive.size()].setVertex(&vertices[v1]);

			edges[edgeActive.size()].setSym(&edges[edgeActive.size() - 1]);


			n_e++;

			// default color and weights
			edgeActive.push_back(true);
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);

		}


		/*! \brief This method calculates the number of edges in zGraph or zMesh.
		*	\return				number of edges.
		*	\since version 0.0.1
		*/
		
		int numEdges()
		{
			return n_e;
		}

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outEdgeId	- stores edgeId if the edge exists else it is -1.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.1
		*/
		
		bool edgeExists(int v1, int v2, int &outEdgeId)
		{

			bool out = false;

			string hashKey = (to_string(v1) + "," + to_string(v2));
			std::unordered_map<std::string, int>::const_iterator got = verticesEdge.find(hashKey);


			if (got != verticesEdge.end())
			{
				out = true;
				outEdgeId = got->second;
			}

			return out;
		}

		/*! \brief This method sorts edges cyclically around a given vertex using a bestfit plane.
		*
		*	\param		[in]	unsortedEdges		- vector of type zEdge holding edges to be sorted
		*	\param		[in]	center				- zVertex to which all the above edges are connected.
		*	\param		[in]	sortReferenceId		- local index in the unsorted edge which is the start reference for sorting.
		*	\param		[out]	sortedEdges			- vector of zVertex holding the sorted edges.
		*	\since version 0.0.1
		*/

		void cyclic_sortEdges(vector<int> &unSortedEdges, zVector &center, int sortReferenceIndex, vector<int> &sortedEdges)
		{

			vector<int> out;

			vector<double> angles;
			map< double, int > angle_e_Map;

			// find best fit plane
			vector<zVector> points;

			for (int i = 0; i < unSortedEdges.size(); i++)
			{
				points.push_back(vertexPositions[edges[unSortedEdges[i]].getVertex()->getVertexId()]);
			}


			zMatrixd bestPlane = getBestFitPlane(points);
			zVector norm = fromMatrixColumn(bestPlane,2);

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = fromMatrixColumn(bestPlane, 0);;
			zVector upVec = fromMatrixColumn(bestPlane, 2);;

			zVector cross = upVec ^ horz;

			if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size())
			{
				horz = zVector(vertexPositions[edges[unSortedEdges[sortReferenceIndex]].getVertex()->getVertexId()] - center);
			}


			angles.clear();

			for (int i = 0; i<unSortedEdges.size(); i++)
			{
				float angle = 0;

				if (i != sortReferenceIndex)
				{
					zVector vec1(vertexPositions[edges[unSortedEdges[i]].getVertex()->getVertexId()] - center);


					double ang = cross.angle(vec1);

					angle = horz.angle(vec1);
					if (ang > 90) angle = 360 - angle;

					//printf("\n cen: %i vert1 : %i vert2: %i andle: %1.2f angle360: %1.2f ", center.vertexId, unSortedEdges[sortReferenceIndex].v->vertexId, unSortedEdges[i].v->vertexId, angle, horz.Angle360(vec1));

				}

				//if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size()) printf("\n sortReferenceIndex : %i currentedge: %i angle : %1.2f id: %i ", unSortedEdges[sortReferenceIndex].edgeId , unSortedEdges[i].edgeId, angle, i);

				// check if same key exists
				for (int k = 0; k < angles.size(); k++)
				{
					if (angles[k] == angle) angle += 0.01;
				}

				angle_e_Map[angle] = i;
				angles.push_back(angle);
			}

			sort(angles.begin(), angles.end());

			for (int i = 0; i < angles.size(); i++)
			{
				int id = angle_e_Map.find(angles[i])->second;
				out.push_back((unSortedEdges[id]));

			}



			sortedEdges = out;

		}
		
		/*! \brief This method collapses an edge into vertex.
		*
		*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
		*	\since version 0.0.1
		*/
		
		void collapseEdges(vector<int> &edgeList)
		{

			for (int i = 0; i < edgeList.size(); i++)
			{
				int currentEdgeId = edgeList[i];
				int symEdgeId = edges[currentEdgeId].getSym()->getEdgeId();

				vector<int> eVerts;
				getVertices(edgeList[i], zEdgeData, eVerts);

				vertexPositions[eVerts[0]] = (vertexPositions[eVerts[0]] + vertexPositions[eVerts[1]])* 0.5;

				if (!checkVertexValency(eVerts[0]))
				{
					edges[currentEdgeId].getPrev()->setNext(edges[currentEdgeId].getSym()->getNext());
				}
				else if (!checkVertexValency(eVerts[1]))
				{
					edges[symEdgeId].getPrev()->setNext(edges[symEdgeId].getSym()->getNext());
				}
				else
				{
					// get connected edges
					vector<int> cEdges;
					getConnectedEdges(eVerts[1], zVertexData, cEdges);

					for (int k = 0; k < cEdges.size(); k++)
					{
						edges[cEdges[k]].getSym()->setVertex(&vertices[eVerts[0]]);
					}

					edges[currentEdgeId].getNext()->setPrev(edges[currentEdgeId].getPrev());
					edges[symEdgeId].getNext()->setPrev(edges[symEdgeId].getPrev());


				}


				edgeActive[currentEdgeId] = false;
				edgeActive[symEdgeId] = false;
				n_e -= 2;

				vertexActive[eVerts[1]] = false;
				n_v--;

			}


		}
	
		/*! \brief This method splits a set of edges of a graph in a continuous manner.
		*
		*	\param		[in]	edgeList		- indicies of the edges to be split.
		*	\param		[in]	edgeFactor		- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
		*	\param		[out]	splitVertexId	- stores indices of the new vertex per edge in the given input edgelist.
		*	\since version 0.0.1
		*/
	
		void splitEdges(vector<int> &edgeList, vector<double> &edgeFactor, vector<int> &splitVertexId)
		{
			splitVertexId.clear();

			//split Edges
			for (int i = 0; i < edgeList.size(); i++)
			{
				int currentEdgeId = edgeList[i];
				int currentNextEdgeId = edges[edgeList[i]].getNext()->getEdgeId();
				int symEdgeId = edges[edgeList[i]].getSym()->getEdgeId();
				int symPrevEdgeId = edges[symEdgeId].getPrev()->getEdgeId();


				zVector e = vertexPositions[edges[currentEdgeId].getVertex()->getVertexId()] - vertexPositions[edges[symEdgeId].getVertex()->getVertexId()];
				double eLength = e.length();
				e.normalize();

				zVector vertPos = vertexPositions[edges[symEdgeId].getVertex()->getVertexId()];
				zVector edge = e * edgeFactor[i] * eLength;
				zVector newVertPos = vertPos + edge;


				// check if vertex exists if not add new vertex
				int VertId;
				bool vExists = vertexExists(newVertPos, VertId);
				if (!vExists)
				{
					addVertex(newVertPos);
					VertId = vertexActive.size() - 1;
				}
				splitVertexId.push_back(VertId);

				if (!vExists)
				{
					// add new edges
					int v1 = VertId;
					int v2 = edges[currentEdgeId].getVertex()->getVertexId();
					addEdges(v1, v2);

					// update vertex pointers
					vertices[v1].setEdge(&edges[edgeActive.size() - 2]);
					vertices[v2].setEdge(&edges[edgeActive.size() - 1]);

					//// update pointers
					edges[currentEdgeId].setVertex(&vertices[VertId]);			// current edge vertex pointer updated to new added vertex

					edges[edgeActive.size() - 2].setNext(&edges[currentNextEdgeId]);		// new added edge next pointer to point to the next of current edge

					edges[edgeActive.size() - 2].setPrev(&edges[currentEdgeId]);			// new added edge prev pointer to point to current edge

																							// update symmetry edge pointers
					edges[edgeActive.size() - 1].setNext(&edges[symEdgeId]);
					edges[edgeActive.size() - 1].setPrev(&edges[symPrevEdgeId]);

				}
			}

		}

	
	protected:
		
		//---- PROTECTED METHODS

		/*! \brief This method resizes the array connected with the input type to the specified newSize.
		*
		*	\param		[in]	newSize			- new size of the array.
		*	\param		[in]	type			- zVertexData or zEdgeData.
		*	\since version 0.0.1
		*/
		
		void resizeArray(int newSize, zHEData type = zVertexData)
		{

			//  Vertex
			if (type == zVertexData)
			{
				zVertex *resized = new zVertex[newSize];

				for (int i = 0; i < vertexActive.size(); i++)
				{
					resized[i].setVertexId(i);

					if (vertices[i].getEdge())resized[i].setEdge(vertices[i].getEdge());
				}
				for (int i = 0; i < edgeActive.size(); i++)
				{
					if (edges[i].getVertex()) edges[i].setVertex(&resized[edges[i].getVertex()->getVertexId()]);
				}

				delete[] vertices;
				vertices = new zVertex[newSize];

				for (int i = 0; i < vertexActive.size(); i++)
				{
					vertices[i].setVertexId(i);

					if (resized[i].getEdge())vertices[i].setEdge(resized[i].getEdge());
				}
				for (int i = 0; i < edgeActive.size(); i++)
				{
					if (edges[i].getVertex()) edges[i].setVertex(&vertices[edges[i].getVertex()->getVertexId()]);
				}

				delete[] resized;

			}

			//  Edge
			else if (type == zEdgeData) {

				zEdge *resized = new zEdge[newSize];

				for (int i = 0; i <edgeActive.size(); i++)
				{
					resized[i].setEdgeId(i);

					if (edges[i].getSym()) resized[i].setSym(&resized[edges[i].getSym()->getEdgeId()]);
					if (edges[i].getNext()) resized[i].setNext(&resized[edges[i].getNext()->getEdgeId()]);
					if (edges[i].getPrev()) resized[i].setPrev(&resized[edges[i].getPrev()->getEdgeId()]);

					if (edges[i].getVertex()) resized[i].setVertex(edges[i].getVertex());
					if (edges[i].getFace()) resized[i].setFace(edges[i].getFace());

				}
				for (int i = 0; i < vertexActive.size(); i++)
				{
					if (vertices[i].getEdge()) vertices[i].setEdge(&resized[vertices[i].getEdge()->getEdgeId()]);
				}


				delete[] edges;
				edges = new zEdge[newSize];

				for (int i = 0; i <edgeActive.size(); i++)
				{
					edges[i].setEdgeId(i);

					if (resized[i].getSym()) edges[i].setSym(&resized[resized[i].getSym()->getEdgeId()]);
					if (resized[i].getNext()) edges[i].setNext(&resized[resized[i].getNext()->getEdgeId()]);
					if (resized[i].getPrev()) edges[i].setPrev(&resized[resized[i].getPrev()->getEdgeId()]);

					if (resized[i].getVertex()) edges[i].setVertex(resized[i].getVertex());
					if (resized[i].getFace()) edges[i].setFace(resized[i].getFace());

				}

				for (int i = 0; i < vertexActive.size(); i++)
				{
					if (vertices[i].getEdge()) vertices[i].setEdge(&edges[vertices[i].getEdge()->getEdgeId()]);
				}


				delete[] resized;
			}

		}


	};
}