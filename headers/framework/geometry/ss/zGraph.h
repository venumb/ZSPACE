
#pragma once

#include <headers/framework/core/zVector.h>
#include <headers/framework/core/zMatrix.h>
#include <headers/framework/core/zColor.h>

#include <headers/framework/geometry/zGeometry.h>

namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
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


	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of active vertices */
		int n_v;

		/*!	\brief stores number of active edges */
		int n_e;

		/*!	\brief stores number of max vertices, which is the size of the dynamic vertices array. */
		int max_n_v;

		/*!	\brief stores number of max edges, which is the size of the dynamic vertices array. */
		int max_n_e;

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

		
		/*!	\brief boolean indicating if the geometry is static or not. Default its false.	*/
		bool staticGeometry = false;

		/*! \brief container of edge positions . Used for display if it is a static geometry */
		vector<vector<zVector>> edgePositions;

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
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	staticGraph		- boolean indicating if the geometry is static or not. Default its false.
		*	\since version 0.0.1
		*/		
		zGraph(vector<zVector>(&_positions), vector<int>(&edgeConnects), bool staticGraph = false)
		{

			int _num_vertices = _positions.size();
			int _num_edges = edgeConnects.size();

			// set max size
			max_n_v = _num_vertices * 40;
			max_n_e = _num_edges * 80;

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

			delete[] cEdgesperVert;
			cEdgesperVert = NULL;

		}


		/*! \brief Overloaded constructor for planar graphs.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	graphNormal		- normal of the plane of the graph.
		*	\param		[in]	sortReference	- reference vector for sorting edges.
		*	\since version 0.0.1
		*/
		zGraph(vector<zVector>(&_positions), vector<int>(&edgeConnects), zVector &graphNormal , zVector &sortReference)
		{

			int _num_vertices = _positions.size();
			int _num_edges = edgeConnects.size();

			// set max size
			max_n_v = _num_vertices * 40;
			max_n_e = _num_edges * 80;

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
					cyclic_sortEdges(cEdgesperVert[i].temp_connectedEdges, cen, sortReference, graphNormal, sorted_cEdges);

					if (sorted_cEdges.size() > 0)
					{


						for (int j = 0; j < sorted_cEdges.size(); j++)
						{
							edges[sorted_cEdges[j]].setPrev(edges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]].getSym());
						}


					}
				}

			}


			delete[] cEdgesperVert;
			cEdgesperVert = NULL;

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
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\return				bool		- true if the vertices container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zVector &pos)
		{
			bool out = false;

			if (max_n_v - vertexActive.size() < 2)
			{
				max_n_v *= 40;
				resizeArray(max_n_v, zVertexData); // calls the resize in mesh 

				out = true;
			}
			
			addToPositionMap(pos, vertexActive.size());

			vertices[vertexActive.size()] = zVertex();
			vertices[vertexActive.size()].setVertexId(vertexActive.size());

			vertexPositions.push_back(pos);

			// default Attibute values
			vertexActive.push_back(true);
			vertexColors.push_back(zColor(1, 0, 0, 1));
			vertexWeights.push_back(2.0);

			n_v++;

			return out;
		}	



		/*! \brief This method sets the number of vertices in zGraph  the input value.
		*	\param		[in]		_n_v	- number of vertices.
		*	\param		[in]		setMax	- if true, sets max vertices as amultiple of _n_v.
		*	\since version 0.0.1
		*/
		void setNumVertices(int _n_v, bool setMax = true)
		{
			n_v = _n_v;

			if(setMax) max_n_v = 40 * n_v;
		}


		//--------------------------
		//---- MAP METHODS
		//--------------------------

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]		pos		- input position.
		*	\param		[in]		index	- input vertex index in the vertex position container.
		*	\since version 0.0.1
		*/		
		void addToPositionMap(zVector &pos, int index)
		{
			double factor = pow(10, 3);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			positionVertex[hashKey] = index;
		}

		/*! \brief This method removes the position given by input vector from the positionVertex Map.
		*	\param		[in]		pos		- input position.
		*	\since version 0.0.1
		*/
		void removeFromPositionMap(zVector &pos)
		{
			double factor = pow(10, 3);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string removeHashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			positionVertex.erase(removeHashKey);
		}

		/*! \brief This method adds both the half-edges given by input vertex indices to the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\param		[in]		index	- input edge index in the edge container.
		*	\since version 0.0.1
		*/
		void addToVerticesEdge(int v1, int v2, int index)
		{
			string hashKey = (to_string(v1) + "," + to_string(v2));
			verticesEdge[hashKey] = index;

			string hashKey1 = (to_string(v2) + "," + to_string(v1));
			verticesEdge[hashKey1] = index + 1;

		}

		/*! \brief This method removes both the half-edges given given by vertex input indices from the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\since version 0.0.1
		*/
		void removeFromVerticesEdge(int v1, int v2)
		{
			string removeHashKey = (to_string(v1) + "," + to_string(v2));
			verticesEdge.erase(removeHashKey);	

			string removeHashKey1 = (to_string(v2) + "," + to_string(v1));
			verticesEdge.erase(removeHashKey1);
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
			std::unordered_map<std::string, int>::const_iterator got = verticesEdge.find(hashKey);


			if (got != verticesEdge.end())
			{
				out = true;
				outEdgeId = got->second;
			}

			return out;
		}


		//--------------------------
		//---- EDGE METHODS
		//--------------------------

		/*! \brief This method adds an edge and its symmetry edge to the edges array.
		*
		*	\param		[in]	v1			- start zVertex of the edge.
		*	\param		[in]	v2			- end zVertex of the edge.
		*	\return				bool		- true if the edges container is resized.
		*	\since version 0.0.1
		*/
		bool addEdges(int &v1, int &v2)
		{
			bool out = false;

			if (max_n_e - edgeActive.size() < 4)
			{
				max_n_e *= 80;
				resizeArray(max_n_e, zEdgeData); // calls the resize in mesh 

				out = true;

			}

			addToVerticesEdge(v1, v2, edgeActive.size());

			edges[edgeActive.size()] = zEdge();

			edges[edgeActive.size()].setEdgeId(edgeActive.size());
			edges[edgeActive.size()].setVertex(&vertices[v2]);

			n_e++;

			// default color and weights
			edgeActive.push_back(true);
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);

			// SYMMETRY edge

			edges[edgeActive.size()] = zEdge();

			edges[edgeActive.size()].setEdgeId(edgeActive.size());
			edges[edgeActive.size()].setVertex(&vertices[v1]);

			edges[edgeActive.size()].setSym(&edges[edgeActive.size() - 1]);


			n_e++;

			// default color and weights
			edgeActive.push_back(true);
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);


			return out;

		}
		

		/*! \brief This method sets the number of edges to  the input value.
		*	\param		[in]		_n_e	- number of edges.
		*	\param		[in]		setMax	- if true, sets max edges as amultiple of _n_e.
		*	\since version 0.0.1
		*/	
		void setNumEdges(int _n_e, bool setMax = true)
		{
			n_e = _n_e;

			if(setMax) max_n_e = 80 * n_e;
		}


		/*! \brief This method sets the static edge positions if the graph is static.
		*	\param		[in]		_edgePositions	- input container of edgePositions.
		*	\since version 0.0.2
		*/
		void setStaticEdgePositions(vector<vector<zVector>> &_edgePositions)
		{
			if(!staticGeometry) 	throw std::invalid_argument(" error: graph not static");
			edgePositions = _edgePositions;
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

			

			zMatrixd bestPlane = coreUtils.getBestFitPlane(points);
			zVector norm = coreUtils.fromMatrixColumn(bestPlane,2);

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = coreUtils.fromMatrixColumn(bestPlane, 0);;
			zVector upVec = coreUtils.fromMatrixColumn(bestPlane, 2);;

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
				if (id > unSortedEdges.size())
				{
					id = 0;					
				}

				out.push_back((unSortedEdges[id]));

			}



			sortedEdges = out;

		}
		
		/*! \brief This method sorts edges cyclically around a given vertex using a bestfit plane.
		*
		*	\param		[in]	unsortedEdges		- vector of type zEdge holding edges to be sorted
		*	\param		[in]	center				- zVertex to which all the above edges are connected.
		*	\param		[in]	referenceDir		- reference vector to sort.
		*	\param		[in]	norm				- reference normal to sort.
		*	\param		[out]	sortedEdges			- vector of zVertex holding the sorted edges.
		*	\since version 0.0.1
		*/
		void cyclic_sortEdges(vector<int> &unSortedEdges, zVector &center, zVector& referenceDir, zVector& norm, vector<int> &sortedEdges)
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


			

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = referenceDir;;
			zVector upVec = norm;;

			zVector cross = upVec ^ horz;



			angles.clear();
					
			for (int i = 0; i < unSortedEdges.size(); i++)
			{
				float angle = 0;

				zVector vec1(vertexPositions[edges[unSortedEdges[i]].getVertex()->getVertexId()] - center);
				angle = horz.angle360(vec1, upVec);

		
				
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
				if (id > unSortedEdges.size())
				{
					id = 0;
				}
				out.push_back((unSortedEdges[id]));

			
			}



			sortedEdges = out;

		}
	
	
		
		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------

	protected:

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
				vertices = resized;
				
				//delete[] resized;
				printf("\n graph vertices resized. ");

			}

			//  Edge
			else if (type == zEdgeData) 
			{

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
				edges = resized;

				//delete[] resized;
				printf("\n graph edges resized. ");
			}

		}

		
		

	};
}