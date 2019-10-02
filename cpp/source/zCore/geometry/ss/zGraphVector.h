
#pragma once

#include <headers/framework/core/zVector.h>
#include <headers/framework/core/zMatrix.h>
#include <headers/framework/core/zColor.h>
#include <headers/framework/geometry/zGeometryTypes.h>



namespace zSpace
{

	struct connectedEdgesPerVerts
	{
		int vertId;
		vector<int> temp_connectedEdges;
	};

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

	class zGraphVector
	{


	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of vertices */
		int n_v;

		/*!	\brief stores number of edges */
		int n_e;		

		/*!	\brief vertex container */
		list<zVertexVector> vertices;

		/*!	\brief edge container	*/
		list<zEdgeVector> edges;

		/*!	\brief container which stores vertex positions.			*/
		vector<zVector> vertexPositions;				

		/*!	\brief vertices to edgeId map. Used to check if edge exists with the haskey being the vertex sequence.	 */
		unordered_map <string, zItEdge> existingEdges;	

		/*!	\brief position to vertexId map. Used to check if vertex exists with the haskey being the vertex position.	 */
		unordered_map <string, int> positionVertex;		

		/*!	\brief index to edge map. Used to get edge list iterator from index hashkey.	 */
		unordered_map< int, zItEdge > indexToEdge;
		
		/*!	\brief index to vertex map. Used to get vertex list iterator from index hashkey.	 */
		unordered_map< int, zItVertex > indexToVertex;


		/*!	\brief container which stores vertex colors.	*/
		vector<zColor> vertexColors;				

		/*!	\brief container which stores edge colors.	*/
		vector<zColor> edgeColors;						

		/*!	\brief container which stores vertex weights.	*/
		vector <double> vertexWeights;	

		/*!	\brief container which stores edge weights.	*/
		vector	<double> edgeWeights;					


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
		zGraphVector()
		{
			n_v = n_e = 0;
		}		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		
		~zGraphVector() {}
			
		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the graph from the input contatiners. 
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	staticGraph		- boolean indicating if the geometry is static or not. Default its false.
		*	\since version 0.0.1
		*/
		void create(vector<zVector>(&_positions), vector<int>(&edgeConnects), bool staticGraph = false)
		{		

			n_v = n_e = 0;		

			// temp containers
			connectedEdgesPerVerts *cEdgesperVert = new connectedEdgesPerVerts[_positions.size()];

			// create vertices
			for (int i = 0; i < _positions.size(); i++)
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
				

				zItVertex v1 = indexToVertex[edgeConnects[i]];			
				v1->e = indexToEdge[n_e - 2];

				zItVertex v2 = indexToVertex[edgeConnects[i + 1]];
				v2->e = indexToEdge[n_e - 1];

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
							zItEdge e1 = indexToEdge[sorted_cEdges[j]];

							zItEdge e2 = indexToEdge[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];							

							e1->prev = e2->sym;
							e2->sym->next = e1;

						}


					}
				}

			}




			delete[] cEdgesperVert;
			cEdgesperVert = NULL;

		}

		/*! \brief This methods creates the graph from the input contatiners for planar graphs.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	edgeConnects	- container of edge connections with vertex ids for each edge
		*	\param		[in]	graphNormal		- normal of the plane of the graph.
		*	\param		[in]	sortReference	- reference vector for sorting edges.
		*	\since version 0.0.1
		*/
		void create(vector<zVector>(&_positions), vector<int>(&edgeConnects), zVector &graphNormal, zVector &sortReference)
		{

			n_v = n_e = 0;

			// temp containers
			connectedEdgesPerVerts *cEdgesperVert = new connectedEdgesPerVerts[_positions.size()];

			// create vertices
			for (int i = 0; i < _positions.size(); i++)
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


				zItVertex v1 = indexToVertex[edgeConnects[i]];
				v1->e = indexToEdge[n_e - 2];

				zItVertex v2 = indexToVertex[edgeConnects[i + 1]];
				v2->e = indexToEdge[n_e - 1];

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
							zItEdge e1 = indexToEdge[sorted_cEdges[j]];
							zItEdge e2 = indexToEdge[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];

							e1->prev = e2->sym;
							e2->sym->next = e1;
						}


					}
				}

			}


			delete[] cEdgesperVert;
			cEdgesperVert = NULL;

		}

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
			
			addToPositionMap(pos, n_v);				
			
			zItVertex newV  = vertices.insert(vertices.end(), zVertexVector());
			newV->index = n_v;

			indexToVertex[n_v] = newV;

			n_v++;

			vertexPositions.push_back(pos);

			// default Attibute values			
			vertexColors.push_back(zColor(1, 0, 0, 1));
			vertexWeights.push_back(2.0);			

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
		}


		//--------------------------
		//---- MAP METHODS
		//--------------------------

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]		pos		- input position.
		*	\param		[in]		index	- input vertex index in the vertex position container.
		*	\since version 0.0.1
		*/		
		void addToPositionMap(zVector &pos, int index, int precisionfactor = 6)
		{
			double factor = pow(10, precisionfactor);
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
		void removeFromPositionMap(zVector &pos, int precisionfactor = 6)
		{
			double factor = pow(10, precisionfactor);
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
		void addToEdgesMap(int v1, int v2, zItEdge &eIter)
		{
		
			string e1 = (to_string(v1) + "," + to_string(v2));
			existingEdges[e1] = eIter;

			
			string e2 = (to_string(v2) + "," + to_string(v1));
			existingEdges[e2] = eIter++;

		}

		/*! \brief This method removes both the half-edges given given by vertex input indices from the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\since version 0.0.1
		*/
		void removeFromEdgesMap(int v1, int v2)
		{
			
			string e1 = (to_string(v1) + "," + to_string(v2));
			existingEdges.erase(e1);	

			string e2 = (to_string(v2) + "," + to_string(v1));
			existingEdges.erase(e2);
		}

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outEdgeId	- stores edge iterator if the edge exists else it is null.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool edgeExists(int v1, int v2, zItEdge &outEdgeId)
		{

			bool out = false;

			string e1 = (to_string(v1) + "," + to_string(v2));
			std::unordered_map<std::string, zItEdge>::const_iterator got = existingEdges.find(e1);
						

			if (got != existingEdges.end())
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
		bool addEdges(int v1, int v2)
		{
			bool out = false;

			

			zItEdge newE1 = edges.insert(edges.end(), zEdgeVector());
			newE1->index = n_e;		
			newE1->v = indexToVertex[v2];			
			indexToEdge[n_e] = newE1;				
			

			// default color and weights			
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);
			n_e++;

			// SYMMETRY edge			
			zItEdge newE2 = edges.insert(edges.end(), zEdgeVector());
			newE2->index = n_e;
			newE2->v = indexToVertex[v1];
			indexToEdge[n_e] = newE2;

			newE2->sym = newE1;
			newE1->sym = newE2;

			addToEdgesMap(v1, v2, newE1);

			// default color and weights
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);
			n_e++;


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

				zItEdge e = indexToEdge[unSortedEdges[i]];	
				points.push_back(vertexPositions[e->v->index]);
			}

			

			zMatrixd bestPlane = coreUtils.getBestFitPlane(points);
			zVector norm = coreUtils.fromMatrixColumn(bestPlane,2);

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = coreUtils.fromMatrixColumn(bestPlane, 0);;
			zVector upVec = coreUtils.fromMatrixColumn(bestPlane, 2);;

			zVector cross = upVec ^ horz;

			if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size())
			{
				zItEdge e = indexToEdge[unSortedEdges[sortReferenceIndex]];

				horz = zVector(vertexPositions[e->v->index] - center);
			}


			angles.clear();

			for (int i = 0; i<unSortedEdges.size(); i++)
			{
				float angle = 0;

				if (i != sortReferenceIndex)
				{
					zItEdge e = indexToEdge[unSortedEdges[i]];

					zVector vec1(vertexPositions[e->v->index] - center);


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

				zItEdge e = indexToEdge[unSortedEdges[i]];

				points.push_back(vertexPositions[e->v->index]);
			}


			

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = referenceDir;;
			zVector upVec = norm;;

			zVector cross = upVec ^ horz;



			angles.clear();
					
			for (int i = 0; i < unSortedEdges.size(); i++)
			{
				float angle = 0;

				zItEdge e = indexToEdge[unSortedEdges[i]];

				zVector vec1(vertexPositions[e->v->index] - center);
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

			////  Vertex
			//if (type == zVertexData)
			//{
			//	vector<zVertex> resized(newSize);

			//	for (int i = 0; i < vertexActive.size(); i++)
			//	{
			//		resized[i].setVertexId(i);

			//		if (vertices[i].getEdge())resized[i].setEdge(vertices[i].getEdge());
			//	}
			//	for (int i = 0; i < edgeActive.size(); i++)
			//	{
			//		if (edges[i].getVertex()) edges[i].setVertex(&resized[edges[i].getVertex()->getVertexId()]);
			//	}

			//	//delete[] vertices;
			//	vertices = resized;
			//	
			//	//delete[] resized;
			//	printf("\n graph vertices resized. ");

			//}

			////  Edge
			//else if (type == zEdgeData) 
			//{

			//	//zEdge *resized = new zEdge[newSize];
			//	vector<zEdge> resized(newSize);

			//	for (int i = 0; i <edgeActive.size(); i++)
			//	{
			//		resized[i].setEdgeId(i);

			//		if (edges[i].getSym()) resized[i].setSym(&resized[edges[i].getSym()->getEdgeId()]);
			//		if (edges[i].getNext()) resized[i].setNext(&resized[edges[i].getNext()->getEdgeId()]);
			//		if (edges[i].getPrev()) resized[i].setPrev(&resized[edges[i].getPrev()->getEdgeId()]);

			//		if (edges[i].getVertex()) resized[i].setVertex(edges[i].getVertex());
			//		if (edges[i].getFace()) resized[i].setFace(edges[i].getFace());

			//	}
			//	for (int i = 0; i < vertexActive.size(); i++)
			//	{
			//		if (vertices[i].getEdge()) vertices[i].setEdge(&resized[vertices[i].getEdge()->getEdgeId()]);
			//	}
			//	
			//	//delete[] edges;
			//	edges = resized;

			//	//delete[] resized;
			//	printf("\n graph edges resized. ");
			//}

		}

		
		

	};
}