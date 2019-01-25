#pragma once

#include <headers/geometry/zGraph.h>
#include <headers/geometry/zGraphUtilities.h>


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

	/*! \class zMesh
	*	\brief A half edge mesh class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zMesh : public zGraph
	{
	private:

		
		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief stores number of active faces  */
		int n_f;		

		/*!	\brief stores number of max vertices, which is the size of the dynamic vertices array. */
		int max_n_f;

		// triangles
		vector<int> triangle_Counts;						/*!<container which stores number of triangles per polygon face.	*/
		vector<int> triangle_Connects;						/*!<connection container with vertex ids for each triangle.	*/
		unordered_map<int, vector<int>> polygon_triangleIds;	/*!<triangleIds per polygon	*/

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

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
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	polyCounts		- container of type integer with number of vertices per polygon.
		*	\param		[in]	polyConnects	- polygon connection list with vertex ids for each face.
		*	\param		[in]	computeNormals	- computes the face and vertex normals if true.
		*	\since version 0.0.1
		*/
		zMesh(vector<zVector>(&_positions), vector<int>(&polyCounts), vector<int>(&polyConnects), bool computeNormals = true)
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

			for (int i = 0; i < _num_vertices; i++)
			{
				addVertex(_positions[i]);
			}


			// create faces and edges connection
			int polyconnectsCurrentIndex = 0;

			for (int i = 0; i < _num_polygons; i++)
			{
				int num_faceVerts = polyCounts[i];

				vector<int> fVerts;

				for (int j = 0; j < num_faceVerts; j++)
				{
					fVerts.push_back(polyConnects[polyconnectsCurrentIndex + j]);
				}

				addPolygon(fVerts);
				polyconnectsCurrentIndex += num_faceVerts;

			}

			// update boundary pointers
			update_BoundaryEdgePointers();

			// compute mesh normals
			if (computeNormals) computeMeshNormals();

		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zMesh() {}


		//--------------------------
		//---- GET-SET METHODS
		//--------------------------


		/*! \brief This method gets the edges of a zFace.
		*
		*	\param		[in]	index			- index in the face container.
		*	\param		[in]	type			- zFaceData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.1
		*/
		void getEdges(int index, zHEData type, vector<int> &edgeIndicies)
		{
			vector<int> out;
			out.clear();

			// Mesh Face 
			if (type == zFaceData)
			{
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
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			edgeIndicies = out;
		}

		/*!	\brief This method gets the vertices attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge/face container.
		*	\param		[in]	type			- zEdgeData or zFaceData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.1
		*/
		void getVertices(int index, zHEData type, vector<int> &vertexIndicies)
		{
			vector<int> out;

			// Mesh Edge
			if (type == zEdgeData)
			{
				out.push_back(edges[index].getVertex()->getVertexId());
				out.push_back(edges[index].getSym()->getVertex()->getVertexId());
			}


			// Mesh Face 
			else if (type == zFaceData)
			{

				vector<int> faceEdges;
				getEdges(index, type, faceEdges);

				for (int i = 0; i < faceEdges.size(); i++)
				{
					//out.push_back(edges[faceEdges[i]].getVertex()->getVertexId());
					out.push_back(edges[faceEdges[i]].getSym()->getVertex()->getVertexId());
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			vertexIndicies = out;
		}



		/*! \brief This method gets the faces connected to input zVertex or zFace
		*
		*	\param		[in]	index	- index in the vertex/face container.
		*	\param		[in]	type	- zVertexData or zFaceData.
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.1
		*/
		void getConnectedFaces(int index, zHEData type, vector<int> &faceIndicies)
		{
			vector<int> out;

			// Mesh Vertex
			if (type == zVertexData)
			{
				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				for (int i = 0; i < connectedEdges.size(); i++)
				{
					if (edges[connectedEdges[i]].getFace()) out.push_back(edges[connectedEdges[i]].getFace()->getFaceId());
				}
			}

			// Mesh Face
			else if (type == zFaceData)
			{
				vector<int> fEdges;
				getEdges(index, type, fEdges);



				for (int i = 0; i < fEdges.size(); i++)
				{
					vector<int> eFaces;
					getFaces(fEdges[i], zEdgeData, eFaces);

					for (int k = 0; k < eFaces.size(); k++)
					{
						if (eFaces[k] != index) out.push_back(eFaces[k]);
					}

					/*if (edges[fEdges[i]].f)
					{
					if(edges[fEdges[i]].f->faceId != index) out.push_back(edges[fEdges[i]].f->faceId);
					if(edges)
					}*/
				}


			}
			else throw std::invalid_argument(" error: invalid zHEData type");

			faceIndicies = out;
		}

		/*! \brief This method gets the faces attached to input zEdge
		*
		*	\param		[in]	index			- index in the edge list.
		*	\param		[in]	type			- zEdgeData.
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.1
		*/
		void getFaces(int &index, zHEData type, vector<int> &faceIndicies)
		{

			vector<int> out;

			// Mesh Edge
			if (type == zEdgeData)
			{
				if (edges[index].getFace()) out.push_back(edges[index].getFace()->getFaceId());
				if (edges[index].getSym()->getFace()) out.push_back(edges[index].getSym()->getFace()->getFaceId());
			}
			else throw std::invalid_argument(" error: invalid zHEData type");

			faceIndicies = out;
		}

		/*!	\brief This method determines if  input zVertex or zEdge or zFace is on the boundary.
		*
		*	\param		[in]	index	- index in the vertex/edge/face list.
		*	\param		[in]	type	- zVertexData or zEdgeData or zFaceData.
		*	\return				bool	- true if on boundary else false.
		*	\since version 0.0.1
		*/
		bool onBoundary(int index, zHEData type = zVertexData)
		{
			bool out = false;

			// zMesh Vertex
			if (type == zVertexData && index != -1)
			{

				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				for (int i = 0; i < connectedEdges.size(); i++)
				{
					if (onBoundary(connectedEdges[i], zEdgeData))
					{
						out = true;
						break;
					}
				}
			}

			// Mesh Edge 
			else if (type == zEdgeData && index != -1)
			{
				if (!edges[index].getFace()) out = true;
				//else printf("\n face: %i", edges[index].getFace()->getFaceId());
			}

			// Mesh Face 
			else if (type == zFaceData && index != -1)
			{
				vector<int> fEdges;
				getEdges(index, zFaceData, fEdges);

				for (int i = 0; i < fEdges.size(); i++)
				{
					if (onBoundary(fEdges[i], zEdgeData))
					{
						out = true;
						break;
					}
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			return out;
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


		/*! \brief This method returns the number of polygons in the mesh
		*
		*	\return		int		-	number of polygons
		*	\since version 0.0.1
		*/
		int numPolygons()
		{
			return n_f;
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
		
		//--------------------------
		//---- ATTRIBUTE METHODS
		//--------------------------

		/*! \brief This method computes the face colors based on the vertex colors.
		*
		*	\since version 0.0.1
		*/		
		void computeFaceColorfromVertexColor()
		{
			for (int i = 0; i < faceActive.size(); i++)
			{
				if (faceActive[i])
				{
					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					zColor col;
					for (int j = 0; j < fVerts.size(); j++)
					{
						col.r += vertexColors[fVerts[j]].r;
						col.g += vertexColors[fVerts[j]].g;
						col.b += vertexColors[fVerts[j]].b;
					}

					col.r /= fVerts.size(); col.g /= fVerts.size(); col.b /= fVerts.size();

					faceColors[i] = col;
				}
			}

		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.1
		*/		
		void computeVertexColorfromFaceColor()
		{

			for (int i = 0; i < vertexActive.size(); i++)
			{
				if (vertexActive[i])
				{
					vector<int> cFaces;
					getConnectedFaces(i, zVertexData, cFaces);

					zColor col;
					for (int j = 0; j < cFaces.size(); j++)
					{
						col.r += faceColors[cFaces[j]].r;
						col.g += faceColors[cFaces[j]].g;
						col.b += faceColors[cFaces[j]].b;
					}

					col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

					vertexColors[i] = col;
				}
			}

		}

		/*! \brief This method smoothens the color attributes.
		*	\param		[in]	smoothVal		- number of iterations to run the smooth operation.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.1
		*/		
		void smoothColors(int smoothVal = 1, zHEData type = zVertexData)
		{
			for (int j = 0; j < smoothVal; j++)
			{
				if (type == zVertexData)
				{
					vector<zColor> tempColors;

					for (int i = 0; i < vertexActive.size(); i++)
					{
						zColor col;
						if (vertexActive[i])
						{
							vector<int> cVerts;
							getConnectedVertices(i, zVertexData, cVerts);

							zColor currentCol = vertexColors[i];


							for (int j = 0; j < cVerts.size(); j++)
							{
								col.r += vertexColors[cVerts[j]].r;
								col.g += vertexColors[cVerts[j]].g;
								col.b += vertexColors[cVerts[j]].b;
							}

							col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

							col.r /= cVerts.size(); col.g /= cVerts.size(); col.b /= cVerts.size();


						}

						tempColors.push_back(col);

					}

					for (int i = 0; i < vertexActive.size(); i++)
					{
						if (vertexActive[i])
						{
							vertexColors[i] = (tempColors[i]);
						}
					}
				}


				if (type == zFaceData)
				{
					vector<zColor> tempColors;

					for (int i = 0; i < faceActive.size(); i++)
					{
						zColor col;
						if (faceActive[i])
						{
							vector<int> cFaces;
							getConnectedFaces(i, zFaceData, cFaces);

							zColor currentCol = faceColors[i];
							for (int j = 0; j < cFaces.size(); j++)
							{
								col.r += faceColors[cFaces[j]].r;
								col.g += faceColors[cFaces[j]].g;
								col.b += faceColors[cFaces[j]].b;
							}

							col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

							col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

						}

						tempColors.push_back(col);

					}

					for (int i = 0; i < faceActive.size(); i++)
					{
						if (faceActive[i])
						{
							faceColors[i] = (tempColors[i]);
						}
					}
				}

				else throw std::invalid_argument(" error: invalid zHEData type");
			}



		}

		/*! \brief This method computes the vertex normals based on the face normals.
		*
		*	\since version 0.0.1
		*/
		void computeVertexNormalfromFaceNormal()
		{

			vertexNormals.clear();

			for (int i = 0; i < vertexActive.size(); i++)
			{
				if (vertexActive[i])
				{
					vector<int> cFaces;
					getConnectedFaces(i, zVertexData, cFaces);

					zVector norm;

					for (int j = 0; j < cFaces.size(); j++)
					{
						norm += faceNormals[cFaces[j]];
					}

					norm /= cFaces.size();
					norm.normalize();
					vertexNormals.push_back(norm);
				}
				else vertexNormals.push_back(zVector());
			}

		}

		/*! \brief This method computes the normals assoicated with vertices and polygon faces .
		*
		*	\since version 0.0.1
		*/
		void computeMeshNormals()
		{
			faceNormals.clear();
			
			for (int i = 0; i < faceActive.size(); i++)
			{
				if (faceActive[i])
				{
					// get face vertices and correspondiing positions

					//printf("\n f %i :", i);
					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					zVector fCen; // face center

					vector<zVector> points;
					for (int i = 0; i < fVerts.size(); i++)
					{
						//printf(" %i ", fVerts[i]);
						points.push_back(vertexPositions[fVerts[i]]);

						fCen += vertexPositions[fVerts[i]];
					}

					fCen /= fVerts.size();

					zVector fNorm; // face normal

					if (fVerts.size() != 3)
					{
						for (int i = 0; i < fVerts.size(); i++)
						{
							fNorm += (points[i] - fCen) ^ (points[(i + 1) % fVerts.size()] - fCen);
						}
					}
					else
					{
						zVector cross = (points[1] - points[0]) ^ (points[fVerts.size() - 1] - points[0]);
						cross.normalize();

						fNorm = cross;

					}


					fNorm.normalize();
					faceNormals.push_back(fNorm);
				}
				else faceNormals.push_back(zVector());
			
			}


			// compute vertex normal
			computeVertexNormalfromFaceNormal();
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

		/*! \brief This method deactivates the input elements from the array connected with the input type.
		*
		*	\param		[in]	index			- index to be deactivated.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.1
		*/
		void deactivateElement(int index, zHEData type)
		{
			//  Vertex
			if (type == zVertexData)
			{
				if (index > vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				// remove from vertexPosition map
				removeFromPositionMap(vertexPositions[index]);

				// null pointers indexed vertex
				vertices[index].removeVertex();

				// disable indexed vertex

				vertexActive[index] = false;

				vertexPositions[index] = zVector(10000, 10000, 10000); // dummy position for VBO
				vertexNormals[index] = zVector(0, 0, 1); // dummy normal for VBO
				vertexColors[index] = zColor(1, 1, 1, 1); // dummy color for VBO


																 // update numVertices
				int newNumVertices = numVertices() - 1;
				setNumVertices(newNumVertices, false);
			}

			//  Edge
			else if (type == zEdgeData)
			{

				if (index > edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				int symEdge = edges[index].getSym()->getEdgeId();

				// check if the vertex attached to the edge has the pointer to the current  edge. If true update the pointer. 
				int v1 = edges[index].getVertex()->getVertexId();
				int v2 = edges[symEdge].getVertex()->getVertexId();

								
				if (vertices[v1].getEdge()->getEdgeId() == symEdge)
				{
					vector<int> cEdgesV1;
					getConnectedEdges(v1, zVertexData, cEdgesV1);

					for (int i = 0; i < cEdgesV1.size(); i++)
					{
						if (cEdgesV1[i] != symEdge) vertices[v1].setEdge(&edges[cEdgesV1[i]]);
					}
				}

				if (vertices[v2].getEdge()->getEdgeId() == index)
				{
					vector<int> cEdgesV2;
					getConnectedEdges(v2, zVertexData, cEdgesV2);

					for (int i = 0; i < cEdgesV2.size(); i++)
					{
						if (cEdgesV2[i] != index) vertices[v2].setEdge(&edges[cEdgesV2[i]]);
					}
				}

				// check if the face attached to the edge has the pointer to the current edge. if true update pointer. 

				if (edges[index].getFace())
				{

					if (edges[index].getFace()->getEdge()->getEdgeId() == index)
					{
						vector<int> cEdgesV1;
						getConnectedEdges(v1, zVertexData, cEdgesV1);

						for (int i = 0; i < cEdgesV1.size(); i++)
						{
							if (edges[cEdgesV1[i]].getFace() == edges[index].getFace()) edges[index].getFace()->setEdge(&edges[cEdgesV1[i]]);
						}

					}
					
				}

				if (edges[symEdge].getFace())
				{
					if (edges[symEdge].getFace()->getEdge()->getEdgeId() == symEdge)
					{
						vector<int> cEdgesV2;
						getConnectedEdges(v2, zVertexData, cEdgesV2);

						for (int i = 0; i < cEdgesV2.size(); i++)
						{
							if (edges[cEdgesV2[i]].getFace() == edges[symEdge].getFace()) edges[symEdge].getFace()->setEdge(&edges[cEdgesV2[i]]);
						}

					}
				
				}

				// remove edge from vertex edge map
				removeFromVerticesEdge(v1, v2);

				// make current edge pointer null
				edges[index].removeEdge();
				edges[symEdge].removeEdge();

				// deactivate edges
				edgeActive[index] = false;
				edgeActive[symEdge] = false;

				// update numEdges
				int newNumEdges = numEdges() - 2;
				setNumEdges(newNumEdges, false);
			}

			// Face
			else if (type == zFaceData)
			{
				if (index > faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			
				// make current face pointers null
				faces[index].removeFace();;

				// deactivate face
				faceActive[index] = false;

				// update numPolygons
				int newNumFaces = numPolygons() - 1;
				setNumPolygons(newNumFaces, false);
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

		}

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.1
		*/
		void removeInactiveElements(zHEData type = zVertexData)
		{
			//  Vertex
			if (type == zVertexData)
			{

				if (vertexActive.size() != numVertices())
				{
					zVertex *resized = new zVertex[max_n_v];

					int vertexActiveID = 0;
					int numOrginalVertexActive = vertexActive.size();

					for (int i = 0; i < numVertices(); i++)
					{

						while (!vertexActive[i])
						{
							vertexActive.erase(vertexActive.begin() + i);

							vertexPositions.erase(vertexPositions.begin() + i);

							vertexColors.erase(vertexColors.begin() + i);

							vertexNormals.erase(vertexNormals.begin() + i);

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
								edges[cEdges[j]].getSym()->setVertex(&resized[i]);
							}


							if (vertices[vertexActiveID].getEdge())
							{
								resized[i].setEdge(&edges[vertices[vertexActiveID].getEdge()->getEdgeId()]);

								edges[vertices[vertexActiveID].getEdge()->getEdgeId()].getSym()->setVertex(&resized[i]);
							}
						}


						vertexActiveID++;

					}

					//printf("\n m: %i %i ", numVertices(), vertexActive.size());


					delete[] vertices;

					vertices = resized;

					printf("\n removed inactive vertices. ");
				}				

			}

			//  Edge
			else if (type == zEdgeData) 
			{
				if (edgeActive.size() != numEdges())
				{
					zEdge *resized = new zEdge[max_n_e];

					int edgeActiveID = 0;
					int numOrginalEdgeActive = edgeActive.size();

					int inactiveCounter = 0;

					// clear vertices edge map
					verticesEdge.clear();


					for (int i = 0; i < numEdges(); i += 2)
					{

						while (!edgeActive[i])
						{
							edgeActive.erase(edgeActive.begin() + i);

							edgeColors.erase(edgeColors.begin() + i);

							edgeActiveID++;

						}


						resized[i].setEdgeId(i);
						resized[i + 1].setEdgeId(i + 1);

						// get connected edges and repoint their pointers
						if (edgeActiveID < numOrginalEdgeActive)
						{


							resized[i].setSym(&resized[i + 1]);

							if (edges[edgeActiveID].getNext())
							{
								resized[i].setNext(&resized[edges[edgeActiveID].getNext()->getEdgeId()]);

								edges[edgeActiveID].getNext()->setPrev(&resized[i]);
							}
							if (edges[edgeActiveID].getPrev())
							{
								resized[i].setPrev(&resized[edges[edgeActiveID].getPrev()->getEdgeId()]);

								edges[edgeActiveID].getPrev()->setNext(&resized[i]);
							}


							if (edges[edgeActiveID].getVertex())
							{
								resized[i].setVertex(&vertices[edges[edgeActiveID].getVertex()->getVertexId()]);		

								vertices[edges[edgeActiveID].getVertex()->getVertexId()].setEdge(resized[i].getSym());
							}

							if (edges[edgeActiveID].getFace())
							{
								resized[i].setFace(&faces[edges[edgeActiveID].getFace()->getFaceId()]);
								faces[edges[edgeActiveID].getFace()->getFaceId()].setEdge(&resized[i]);
							}



							//sym edge
							if (edges[edgeActiveID + 1].getNext())
							{
								resized[i + 1].setNext(&resized[edges[edgeActiveID + 1].getNext()->getEdgeId()]);

								edges[edgeActiveID + 1].getNext()->setPrev(&resized[i + 1]);

							}
							if (edges[edgeActiveID + 1].getPrev())
							{
								resized[i + 1].setPrev(&resized[edges[edgeActiveID + 1].getPrev()->getEdgeId()]);

								edges[edgeActiveID + 1].getPrev()->setNext(&resized[i + 1]);
							}

							if (edges[edgeActiveID + 1].getVertex())
							{
								resized[i + 1].setVertex(&vertices[edges[edgeActiveID + 1].getVertex()->getVertexId()]);
								vertices[edges[edgeActiveID + 1].getVertex()->getVertexId()].setEdge(resized[i + 1].getSym());
							}

							if (edges[edgeActiveID + 1].getFace())
							{
								resized[i + 1].setFace(&faces[edges[edgeActiveID + 1].getFace()->getFaceId()]);
								faces[edges[edgeActiveID + 1].getFace()->getFaceId()].setEdge(&resized[i + 1]);
							}


							// rebuild vertices edge map
							int v2 = resized[i].getVertex()->getVertexId();
							int v1 = resized[i + 1].getVertex()->getVertexId();

							addToVerticesEdge(v1, v2, i);

						}

						edgeActiveID += 2;

					}

					//printf("\n m: %i %i ", numEdges(), edgeActive.size());

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

					printf("\n removed inactive edges. ");
				}

				

			}

			// Mesh Face
			else if (type == zFaceData )
			{
				if (faceActive.size() != numPolygons())
				{
					zFace *resized = new zFace[max_n_f];

					int faceActiveID = 0;
					int numOrginalFaceActive = faceActive.size();

					for (int i = 0; i < numPolygons(); i++)
					{

						while (!faceActive[i])
						{
							faceActive.erase(faceActive.begin() + i);

							faceNormals.erase(faceNormals.begin() + i);

							faceColors.erase(faceColors.begin() + i);

							faceActiveID++;		
			
						}

						resized[i].setFaceId(i);


						// get connected edges and repoint their pointers
						if (faceActiveID < numOrginalFaceActive)
						{
							printf("\n f: %i ", faceActiveID);
							vector<int> fEdges;
							getEdges(faceActiveID, zFaceData, fEdges);

							for (int j = 0; j < fEdges.size(); j++)
							{
								edges[fEdges[j]].setFace(&resized[i]);
							}

							if (faces[faceActiveID].getEdge()) resized[i].setEdge(&edges[faces[faceActiveID].getEdge()->getEdgeId()]);

						}

						faceActiveID++;

					}

					//printf("\n m: %i %i ", numPolygons(), faceActive.size());


					delete[] faces;

					faces = resized;

					printf("\n removed inactive faces. ");
				}

				
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method cleans up the mesh to maintain manifolod topology.
		*
		*	\since version 0.0.1
		*/
		void maintainManifoldTopology()
		{

					
			//// remove edges when both half edges have null face pointers
			//for (int i = 0; i < edgeActive.size(); i += 2)
			//{
			//	if (edgeActive[i])
			//	{
			//		int symEdge = i + 1;

			//		if (onBoundary(i, zEdgeData) && onBoundary(i + 1, zEdgeData))
			//		{

			//			/*int v1 = edges[i].getVertex()->getVertexId();
			//			int v2 = edges[i + 1].getVertex()->getVertexId();

			//			if (checkVertexValency(v1, 1)) deactivateElement(v1, zVertexData);
			//			if (checkVertexValency(v2, 1)) deactivateElement(v2, zVertexData);*/

			//			deactivateElement(i, zEdgeData);
			//		}
			//	}

			//}			
			

			// remove vertices with null edge pointers. 
			for (int i = 0; i < vertexActive.size(); i += 1)
			{
				if (vertexActive[i])
				{									
					if (!vertices[i].getEdge())
					{
						deactivateElement(i, zVertexData);
					}
					else
					{
						if(!edgeActive[vertices[i].getEdge()->getEdgeId()])
							deactivateElement(i, zVertexData);
					}
				}

			}
		}

	};

}
