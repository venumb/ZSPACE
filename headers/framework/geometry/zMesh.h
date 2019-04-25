#pragma once

#include <headers/framework/geometry/zGraph.h>


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

	/*! \class zMesh
	*	\brief A half edge mesh class.
	*	\since version 0.0.1
	*/
	

	/** @}*/

	/** @}*/

	class zMesh : public zGraph
	{
	protected:
		
		

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------


		/*!	\brief stores number of active faces  */
		int n_f;

		/*!	\brief stores number of max vertices, which is the size of the dynamic vertices array. */
		int max_n_f;

		/*! \brief face container			*/
		zFace * faces;								

		/*!	\brief container which stores vertex normals.	*/
		vector<zVector> vertexNormals;				
		
		/*!	\brief container which stores face normals. 	*/
		vector<zVector> faceNormals;	

		/*!	\brief container which stores face colors. 	*/
		vector<zColor> faceColors;					
		
		/*!	\brief container which stores vertex weights.	*/
		vector <bool> faceActive;		
	
		/*!	\brief stores the start face ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_FaceId;		

		/*! \brief container of face positions . Used for display if it is a static geometry */
		vector<vector<zVector>> facePositions;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zMesh()
		{
			n_v = n_e = n_f = 0;

			vertices = nullptr;
			edges = nullptr;
			faces = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	polyCounts		- container of type integer with number of vertices per polygon.
		*	\param		[in]	polyConnects	- polygon connection list with vertex ids for each face.
		*	\since version 0.0.1
		*/
		zMesh(vector<zVector>(&_positions), vector<int>(&polyCounts), vector<int>(&polyConnects))
		{
			int _num_vertices = _positions.size();
			int _num_polygons = polyCounts.size();
			int _num_edges = _num_vertices + _num_polygons;


			// set max size
			max_n_v = _num_vertices * 40;
			max_n_e = _num_edges * 80;
			max_n_f = _num_polygons * 40;


			vertices = new zVertex[max_n_v];
			edges = new zEdge[max_n_e * 2];
			faces = new zFace[max_n_f];

			n_v = n_e = n_f = 0;

			verticesEdge.clear();
			positionVertex.clear();

			// create vertices
			addVertices(_positions);			


			// create faces and edges connection
			int polyconnectsCurrentIndex = 0;
			
			vector<int> fVerts;
			for (int i = 0; i < _num_polygons; i++)
			{
				int num_faceVerts = polyCounts[i];
	
				fVerts.clear();
				for (int j = 0; j < num_faceVerts; j++)
				{
					fVerts.push_back(polyConnects[polyconnectsCurrentIndex + j]);
				}

				addPolygon(fVerts);
				polyconnectsCurrentIndex += num_faceVerts;

			}

			// update boundary pointers
			update_BoundaryEdgePointers();			

		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zMesh() 
		{
			
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

			if (max_n_v - vertexActive.size() < 2)
			{
				max_n_v *= 4;
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

		/*! \brief This method adds vertices from the input position array.
		*
		*	\param		[in]	pos			- container of zVector holding the position information of the vertex.
		*	\since version 0.0.1
		*/
		void addVertices(vector<zVector> &positions)
		{		
			zColor col(1, 0, 0, 1);
			vertexActive.clear();
			vertexActive.assign(positions.size(), true);

			vertexColors.clear();
			vertexColors.assign(positions.size(), col);

			vertexWeights.clear();
			vertexWeights.assign(positions.size(), 2.0);

			vertexPositions.clear(); 
			vertexPositions = positions;

			for (int i = 0; i < positions.size(); i++)
			{
				addToPositionMap(positions[i], i);
				vertices[i] = zVertex();
				vertices[i].setVertexId(i);				
			}

			n_v = positions.size();

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
				max_n_e *= 4;
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

		/*! \brief This method sets the static edge positions if the graph is static.
		*	\param		[in]		_edgePositions	- input container of edgePositions.
		*	\since version 0.0.2
		*/
		void setStaticEdgePositions(vector<vector<zVector>> &_edgePositions)
		{
			if (!staticGeometry) 	throw std::invalid_argument(" error: mesh not static");
			edgePositions = _edgePositions;
		}

		//--------------------------
		//---- FACE METHODS
		//--------------------------


		/*! \brief This method adds a face with null edge pointer to the faces array.
		*
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addPolygon()
		{
			bool out = false;

			if (max_n_f - faceActive.size() < 2)
			{

				max_n_f *= 40;
				resizeArray(max_n_f, zFaceData);

				out = true;
			}

			// create face
			faces[faceActive.size()] = (zFace());
			faces[faceActive.size()].setFaceId(faceActive.size());

			// update num faces
			n_f++;

			//add default faceColors 
			faceActive.push_back(true);
			faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));

			return out;

		}

		/*! \brief This method adds a face to the faces array and updates the pointers of vertices, edges and polygons of the mesh based on face vertices.
		*
		*	\param		[in]	fVertices	- array of ordered vertices that make up the polygon.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addPolygon(vector<int> &fVertices)
		{
			// add null polygon
			bool out = addPolygon();

			// get edgeIds of face
			vector<int> fEdge;

			int currentNumEdges = edgeActive.size();

			for (int i = 0; i < fVertices.size(); i++)
			{
				// check if edge exists
				int eID = -1;
				bool chkEdge = edgeExists(fVertices[i], fVertices[(i + 1) % fVertices.size()], eID);

			

				if (chkEdge)
				{
					fEdge.push_back(eID);
				}
				else
				{
					addEdges(fVertices[i], fVertices[(i + 1) % fVertices.size()]);
					fEdge.push_back(edges[edgeActive.size() - 2].getEdgeId());

				}

				//printf("\n %i %i %s %i", fVertices[i], fVertices[(i + 1) % fVertices.size()], (chkEdge) ? "true" : "false", fEdge[i]);
			}

			//update current face verts edge pointer
			for (int i = 0; i < fVertices.size(); i++)
			{
				vertices[fVertices[i]].setEdge(&edges[fEdge[i]]);
			}

			// update face, next and prev edge pointers for face Edges
			for (int i = 0; i < fEdge.size(); i++)
			{
				edges[fEdge[i]].setFace(&faces[faceActive.size() - 1]);

				edges[fEdge[i]].setNext(&edges[fEdge[(i + 1) % fEdge.size()]]);
				edges[fEdge[i]].setPrev(&edges[fEdge[(i - 1 + fEdge.size()) % fEdge.size()]]);
			}

			// update curent face edge
			faces[faceActive.size() - 1].setEdge(&edges[fEdge[0]]);

			return out;
		}			   

		/*! \brief This method sets the number of faces in zMesh  the input value.
		*	\param		[in]	_n_f	-	number of faces.
		*	\param		[in]	setMax	- if true, sets max edges as amultiple of _n_e.
		*	\since version 0.0.1
		*/		
		void setNumPolygons(int _n_f, bool setMax = true)
		{
			n_f = _n_f;
		
			if(setMax) max_n_f = 40 * n_f;
		}
		
		/*! \brief This method gets the edges of a zFace.
		*
		*	\param		[in]	index			- index in the face container.
		*	\param		[in]	type			- zFaceData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.1
		*/
		void getFaceEdges(int index, vector<int> &edgeIndicies)
		{
			vector<int> out;
			out.clear();

			if (faces[index].getEdge())
			{
				zEdge* start = faces[index].getEdge();
				zEdge* e = start;

				bool exit = false;

				do
				{
					out.push_back(e->getEdgeId());
					if (e->getNext())e = e->getNext();
					else exit = true;

				} while (e != start && !exit);
			}
	


			edgeIndicies = out;
		}


		/*!	\brief This method gets the vertices attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge/face container.
		*	\param		[in]	type			- zEdgeData or zFaceData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.1
		*/
		void getFaceVertices(int index, vector<int> &vertexIndicies)
		{
			vector<int> out;

			vector<int> faceEdges;
			getFaceEdges(index, faceEdges);

			for (int i = 0; i < faceEdges.size(); i++)
			{
				out.push_back(edges[faceEdges[i]].getSym()->getVertex()->getVertexId());
			}

			vertexIndicies = out;
		}

		/*! \brief This method sets the static edge positions if the mesh is static.
		*	\param		[in]		_edgePositions	- input container of edgePositions.
		*	\since version 0.0.2
		*/
		void setStaticFacePositions(vector<vector<zVector>> &_facePositions)
		{
			if (!staticGeometry) 	throw std::invalid_argument(" error: geometry not static");
			facePositions = _facePositions;
		}

		//--------------------------
		//---- PRIVATE METHODS
		//--------------------------

	private:

		/*! \brief This method resizes the array connected with the input type to the specified newSize.
		*
		*	\param		[in]	newSize			- new size of the array.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
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

					if (vertices[i].getEdge())resized[i].setEdge(&edges[vertices[i].getEdge()->getEdgeId()]);
				}
				for (int i = 0; i < edgeActive.size(); i++)
				{
					if (edges[i].getVertex()) edges[i].setVertex(&resized[edges[i].getVertex()->getVertexId()]);
				}
												
				delete[] vertices;		

				vertices = resized;
			
				printf("\n mesh vertices resized. ");
				
			}

			//  Edge
			else if (type == zEdgeData) {

				zEdge *resized = new zEdge[newSize];

				for (int i = 0; i < edgeActive.size(); i++)
				{
					resized[i].setEdgeId(i);

					if (edges[i].getSym()) resized[i].setSym(&resized[edges[i].getSym()->getEdgeId()]);
					if (edges[i].getNext()) resized[i].setNext(&resized[edges[i].getNext()->getEdgeId()]);
					if (edges[i].getPrev()) resized[i].setPrev(&resized[edges[i].getPrev()->getEdgeId()]);

					if (edges[i].getVertex()) resized[i].setVertex(&vertices[edges[i].getVertex()->getVertexId()]);
					if (edges[i].getFace()) resized[i].setFace(&faces[edges[i].getFace()->getFaceId()]);

				}
				
				for (int i = 0; i < vertexActive.size(); i++)
				{
					if (vertices[i].getEdge()) vertices[i].setEdge(&resized[vertices[i].getEdge()->getEdgeId()]);
				}
				
				for (int i = 0; i < faceActive.size(); i++)
				{
					if (faces[i].getEdge()) faces[i].setEdge(&resized[faces[i].getEdge()->getEdgeId()]);

				}

				delete[] edges;
				edges = resized;

				printf("\n mesh edges resized. ");

			}

			// Mesh Face
			else if (type == zFaceData)
			{

				zFace *resized = new zFace[newSize];

				for (int i = 0; i < faceActive.size(); i++)
				{
					resized[i].setFaceId(i);
					if (faces[i].getEdge()) resized[i].setEdge(&edges[faces[i].getEdge()->getEdgeId()]);

					//printf("\n %i : %i ", (resized[i].getEdge()) ? resized[i].getEdge()->getEdgeId():-1 , (faces[i].getEdge()) ? faces[i].getEdge()->getEdgeId():-1);
				}


				for (int i = 0; i < edgeActive.size(); i++)
				{
					if (edges[i].getFace())
					{
						edges[i].setFace(&resized[edges[i].getFace()->getFaceId()]);
					}
				}

				delete[] faces;
				faces = resized;	
				
				printf("\n mesh faces resized. ");
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}
			
	
		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

	public:
		
		/*! \brief This method updates the pointers for boundary Edges.
		*
		*	\param		[in]	numEdges		- number of edges in the mesh.
		*	\since version 0.0.1
		*/
		void update_BoundaryEdgePointers()
		{
			for (int i = 0; i < edgeActive.size(); i++)
			{
				if (edgeActive[i])
				{
					
					if (!edges[i].getFace())
					{
						//printf("\n %i ", i);

						zEdge* e = edges[i].getSym();
						bool exit = false;

						do
						{
							e = e->getPrev();
							e = e->getSym();

						} while (e->getFace());

						edges[i].setNext(e);
						zEdge* e1 = edges[i].getSym();

						do
						{
							e1 = e1->getNext();
							e1 = e1->getSym();

						} while (e1->getFace());

						edges[i].setPrev(e1);
					}
				}

			}
		}




	};

}
