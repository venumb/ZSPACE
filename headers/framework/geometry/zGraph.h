
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

	class zGraph
	{
	protected:
		


	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of active vertices */
		int n_v;

		/*!	\brief stores number of active  edges */
		int n_e;

		/*!	\brief stores number of active half edges */
		int n_he;

		/*!	\brief vertex container */
		vector<zVertex> vertices;

		/*!	\brief edge container	*/
		vector<zHalfEdge> halfEdges;

		/*!	\brief edge container	*/
		vector<zEdge> edges;

		/*!	\brief container which stores vertex positions.			*/
		vector<zVector> vertexPositions;

		/*!	\brief vertices to edgeId map. Used to check if edge exists with the haskey being the vertex sequence.	 */
		unordered_map <string, int> existingHalfEdges;

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

		/*!	\brief stores vertex handles. Used for container resizing only  */
		vector<zVertexHandle> vHandles;

		/*!	\brief stores edge handles. Used for container resizing only  */
		vector<zEdgeHandle> eHandles;

		/*!	\brief stores half edge handles. Used for container resizing only  */
		vector<zHalfEdgeHandle> heHandles;


		/*!	\brief stores the start vertex ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_VertexId;	

		/*!	\brief stores the start edge ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_EdgeId;		

		/*!	\brief stores the start vertex color ID in the VBO, when attache to the zBufferObject.	*/
		int VBO_VertexColorId;		

		
		/*!	\brief boolean indicating if the geometry is static or not. Default its false.	*/
		bool staticGeometry = false;

		/*! \brief container of edge vertices . Used for display if it is a static geometry */
		vector<vector<int>> edgeVertices;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/		
		zGraph()
		{
			n_v = n_he = n_e = 0;
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
			// clear containers
			clear();
			
			vertices.reserve(_positions.size());			
			edges.reserve(floor(edgeConnects.size() * 0.5));
			halfEdges.reserve(edgeConnects.size());
			

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

				cEdgesperVert[edgeConnects[i]].temp_connectedEdges.push_back(n_he - 2);
				cEdgesperVert[edgeConnects[i + 1]].temp_connectedEdges.push_back(n_he - 1);

				vertices[edgeConnects[i]].setHalfEdge(&halfEdges[n_he - 2]);
				vertices[edgeConnects[i + 1]].setHalfEdge(&halfEdges[n_he - 1]);


				vHandles[edgeConnects[i]].he = n_he - 2;
				vHandles[edgeConnects[i + 1]].he = n_he - 1;

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
							zHalfEdge* e1 = &halfEdges[sorted_cEdges[j]];
							zHalfEdge* e2 = &halfEdges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];

							e1->setPrev(e2->getSym());
							
							heHandles[e1->getId()].p = e2->getSym()->getId();
							heHandles[e2->getSym()->getId()].n = e1->getId();
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

			// clear containers
			clear();

			vertices.reserve(_positions.size());
			edges.reserve(floor(edgeConnects.size() * 0.5));
			halfEdges.reserve(edgeConnects.size());

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

				cEdgesperVert[edgeConnects[i]].temp_connectedEdges.push_back(n_he - 2);
				cEdgesperVert[edgeConnects[i + 1]].temp_connectedEdges.push_back(n_he - 1);


				vertices[edgeConnects[i]].setHalfEdge(&halfEdges[n_he - 2]);
				vertices[edgeConnects[i + 1]].setHalfEdge(&halfEdges[n_he - 1]);


				vHandles[edgeConnects[i]].he = n_he - 2;
				vHandles[edgeConnects[i+1]].he = n_he - 1;
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
							zHalfEdge* e1 = &halfEdges[sorted_cEdges[j]];
							zHalfEdge* e2 = &halfEdges[sorted_cEdges[(j + 1) % sorted_cEdges.size()]];

							e1->setPrev(e2->getSym());
							
							heHandles[e1->getId()].p = e2->getSym()->getId();
							heHandles[e2->getSym()->getId()].n = e1->getId();
						}


					}
				}

			}			


			delete[] cEdgesperVert;
			cEdgesperVert = NULL;

		}

		/*! \brief This methods clears all the graph containers.
		*
		*	\since version 0.0.2
		*/
		void clear()
		{
			vertices.clear();
			vertexPositions.clear();
			vertexColors.clear();
			vertexWeights.clear();			
			positionVertex.clear();

			edges.clear();
			edgeColors.clear();
			edgeWeights.clear();
			

			halfEdges.clear();
			existingHalfEdges.clear();

			vHandles.clear();
			eHandles.clear();
			heHandles.clear();

			n_v = n_e = n_he = 0;
		}

		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zVector &pos)
		{
			bool out = false;

			if (n_v == vertices.capacity())
			{
				resizeArray(zVertexData, n_v * 4);
				out = true;
			}


			addToPositionMap(pos, n_v);

			zItVertex newV = vertices.insert(vertices.end(), zVertex());
			newV->setId(n_v);


			vertexPositions.push_back(pos);
			vHandles.push_back(zVertexHandle());
			vHandles[n_v].id = n_v;

			n_v++;

			

			// default Attibute values			
			vertexColors.push_back(zColor(1, 0, 0, 1));
			vertexWeights.push_back(2.0);

			return out;
		}

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\return					bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zVector pos, int &outVertexId, int precisionfactor = 6)
		{
			bool out = false;;
			outVertexId = -1;

			double factor = pow(10, precisionfactor);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


			if (got != positionVertex.end())
			{
				out = true;
				outVertexId = got->second;
			}


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
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]		precisionfactor	- input precision factor.
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
		*	\param		[in]		pos				- input position.
		*	\param		[in]		precisionfactor	- input precision factor.
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
		void addToHalfEdgesMap(int v1, int v2, int index)
		{

			string e1 = (to_string(v1) + "," + to_string(v2));
			existingHalfEdges[e1] = index;


			string e2 = (to_string(v2) + "," + to_string(v1));
			existingHalfEdges[e2] = index +1;

		}

		/*! \brief This method removes both the half-edges given given by vertex input indices from the VerticesEdge Map.
		*	\param		[in]		v1		- input vertex index A.
		*	\param		[in]		v2		- input vertex index B.
		*	\since version 0.0.1
		*/
		void removeFromHalfEdgesMap(int v1, int v2)
		{

			string e1 = (to_string(v1) + "," + to_string(v2));
			existingHalfEdges.erase(e1);

			string e2 = (to_string(v2) + "," + to_string(v1));
			existingHalfEdges.erase(e2);
		}

		/*! \brief This method detemines if an edge already exists between input vertices.
		*
		*	\param		[in]	v1			- vertexId 1.
		*	\param		[in]	v2			- vertexId 2.
		*	\param		[out]	outEdgeId	- stores edgeId if the edge exists else it is -1.
		*	\return		[out]	bool		- true if edge exists else false.
		*	\since version 0.0.2
		*/
		bool halfEdgeExists(int v1, int v2, int &outEdgeId)
		{

			bool out = false;

			string e1 = (to_string(v1) + "," + to_string(v2));
			std::unordered_map<std::string, int>::const_iterator got = existingHalfEdges.find(e1);


			if (got != existingHalfEdges.end())
			{
				out = true;
				outEdgeId = got->second;
			}
			else outEdgeId = - 1;

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

			if (n_e == edges.capacity())
			{
				resizeArray(zEdgeData, n_e * 4);
				resizeArray(zHalfEdgeData, n_he * 8);
				out = true;
			}


			// Handles
			heHandles.push_back(zHalfEdgeHandle());
			heHandles.push_back(zHalfEdgeHandle());
			eHandles.push_back(zEdgeHandle());

			// HALF edge	
			zItHalfEdge newHE1 = halfEdges.insert(halfEdges.end(), zHalfEdge());
			newHE1->setId(n_he);
			newHE1->setVertex(&vertices[v2]);
			
			heHandles[n_he].id = n_he;
			heHandles[n_he].v = v2;
			heHandles[n_he].e = n_e;

			n_he++;

			// SYMMETRY edge			
			zItHalfEdge newHE2 = halfEdges.insert(halfEdges.end(), zHalfEdge());
			newHE2->setId(n_he);
			newHE2->setVertex(&vertices[v1]);

			heHandles[n_he].id = n_he;
			heHandles[n_he].v = v1;
			heHandles[n_he].e = n_e;

			n_he++;

			// set symmetry pointers 
			newHE2->setSym(&halfEdges[n_he - 2]);
			addToHalfEdgesMap(v1, v2, newHE1->getId());

			//EDGE
			zItEdge newE = edges.insert(edges.end(), zEdge());
			newE->setId(n_e);
			newE->setHalfEdge(&halfEdges[n_he - 2], 0);
			newE->setHalfEdge(&halfEdges[n_he - 1], 1);;


			newHE1->setEdge(&edges[n_e]);
			newHE2->setEdge(&edges[n_e]);

			eHandles[n_e].id = n_e;
			eHandles[n_e].he0 = n_he - 2;
			eHandles[n_e].he1 = n_he - 1;

			n_e++;


			// default color and weights
			edgeColors.push_back(zColor(0, 0, 0, 0));
			edgeWeights.push_back(1.0);

			return out;
		}
		

		/*! \brief This method sets the number of edges to  the input value.
		*	\param		[in]		_n_e	- number of edges.
		*	\param		[in]		setMax	- if true, sets max edges as amultiple of _n_he.
		*	\since version 0.0.1
		*/	
		void setNumEdges(int _n_e, bool setMax = true)
		{
			n_e = _n_e;
			n_he = _n_e * 2;			
		}			


		/*! \brief This method sets the static edge vertices if the graph is static.
		*	\param		[in]		_edgeVertices	- input container of edge Vertices.
		*	\since version 0.0.2
		*/
		void setStaticEdgeVertices(vector<vector<int>> &_edgeVertices)
		{
			if (!staticGeometry) 	throw std::invalid_argument(" error: mesh not static");
			edgeVertices = _edgeVertices;
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

				zHalfEdge *e = &halfEdges[unSortedEdges[i]];
				points.push_back(vertexPositions[e->getVertex()->getId()]);
			}



			zMatrixd bestPlane = coreUtils.getBestFitPlane(points);
			zVector norm = coreUtils.fromMatrixColumn(bestPlane, 2);

			// iterate through edges in list, get angle to horz, sort;

			zVector horz = coreUtils.fromMatrixColumn(bestPlane, 0);;
			zVector upVec = coreUtils.fromMatrixColumn(bestPlane, 2);;

			zVector cross = upVec ^ horz;

			if (sortReferenceIndex != -1 && sortReferenceIndex < unSortedEdges.size())
			{
				zHalfEdge *e = &halfEdges[unSortedEdges[sortReferenceIndex]];

				horz = zVector(vertexPositions[e->getVertex()->getId()] - center);
			}


			angles.clear();

			for (int i = 0; i < unSortedEdges.size(); i++)
			{
				float angle = 0;

				if (i != sortReferenceIndex)
				{
					zHalfEdge *e = &halfEdges[unSortedEdges[i]];

					zVector vec1(vertexPositions[e->getVertex()->getId()] - center);


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

				zHalfEdge *e = &halfEdges[unSortedEdges[i]];

				points.push_back(vertexPositions[e->getVertex()->getId()]);
			}




			// iterate through edges in list, get angle to horz, sort;

			zVector horz = referenceDir;;
			zVector upVec = norm;;

			zVector cross = upVec ^ horz;



			angles.clear();

			for (int i = 0; i < unSortedEdges.size(); i++)
			{
				float angle = 0;

				zHalfEdge *e = &halfEdges[unSortedEdges[i]];

				zVector vec1(vertexPositions[e->getVertex()->getId()] - center);
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
		

		/*! \brief This method assigns a unique index per graph element.
		*
		*	\param		[in]	type				- zVertexData or zEdgeData or zHalfEdgeData.
		*	\since version 0.0.1
		*/
		void indexElements(zHEData type)
		{
			if (type == zVertexData)
			{
				int n_v = 0;
				for (auto &v : vertices)
				{
					v.setId(n_v);
					n_v++;
				}
			}			

			else if (type == zEdgeData)
			{
				int n_e = 0;
				for (auto &e : edges)
				{
					e.setId(n_e);
					n_e++;
				}
			}
			
			else if (type == zHalfEdgeData)
			{
				int n_he = 0;
				for (auto &he : halfEdges)
				{
					he.setId(n_he);
					n_he++;
				}
			}
			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method resizes the array connected with the input type to the specified newSize.
		*
		*	\param		[in]	type			- zVertexData or zHalfEdgeData or zEdgeData.
		*	\param		[in]	newSize			- new size of the array.
		*	\since version 0.0.1
		*/
		void resizeArray(zHEData type, int newSize)
		{
			//  Vertex
			if (type == zVertexData)
			{
				vertices.clear();
				vertices.reserve(newSize);

				// reassign pointers
				int n_v = 0;
				for (auto &v : vHandles)
				{
					zItVertex newV = vertices.insert(vertices.end(), zVertex());
					newV->setId(n_v);
					if (v.he != -1)newV->setHalfEdge(&halfEdges[v.he]);

					n_v++;
				}

				for (int i = 0; i < heHandles.size(); i++)
				{
					if (heHandles[i].v != -1) halfEdges[i].setVertex(&vertices[heHandles[i].v]);
				}

				printf("\n graph vertices resized. ");

			}

			//  Edge
			else if (type == zHalfEdgeData)
			{

				halfEdges.clear();
				halfEdges.reserve(newSize);

				halfEdges.assign(heHandles.size(), zHalfEdge());

				// reassign pointers
				int n_he = 0;
				for (auto &he : heHandles)
				{
					halfEdges[n_he].setId(n_he);

					int sym = (n_he % 2 == 0) ? n_he + 1 : n_he - 1;

					halfEdges[n_he].setSym(&halfEdges[sym]);
					if (he.n != -1) halfEdges[n_he].setNext(&halfEdges[he.n]);
					if (he.p != -1) halfEdges[n_he].setPrev(&halfEdges[he.p]);

					if (he.v != -1) halfEdges[n_he].setVertex(&vertices[he.v]);
					if (he.e != -1) halfEdges[n_he].setEdge(&edges[he.e]);
					

					n_he++;
				}

				for (int i = 0; i < vHandles.size(); i++)
				{
					if (vHandles[i].he != -1) vertices[i].setHalfEdge(&halfEdges[vHandles[i].he]);
				}

				for (int i = 0; i < eHandles.size(); i++)
				{
					if (eHandles[i].he0 != -1) edges[i].setHalfEdge(&halfEdges[eHandles[i].he0], 0);
					if (eHandles[i].he1 != -1) edges[i].setHalfEdge(&halfEdges[eHandles[i].he1], 1);
				}			



				printf("\n graph half edges resized. ");

			}

			else if (type == zEdgeData)
			{

				edges.clear();
				edges.reserve(newSize);

				// reassign pointers
				int n_e = 0;
				for (auto &e : eHandles)
				{
					zItEdge newE = edges.insert(edges.end(), zEdge());
					newE->setId(n_e);

					if (e.he0 != -1)newE->setHalfEdge(&halfEdges[e.he0], 0);
					if (e.he1 != -1)newE->setHalfEdge(&halfEdges[e.he1], 1);

					n_e++;

				}

				for (int i = 0; i < heHandles.size(); i++)
				{
					if (heHandles[i].e != -1) halfEdges[i].setEdge(&edges[heHandles[i].e]);
				}



				printf("\n graph edges resized. ");

			}

		

			else throw std::invalid_argument(" error: invalid zHEData type");
		}
	};
}