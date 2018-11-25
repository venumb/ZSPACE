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
			max_n_v = _num_vertices * 4;
			max_n_e = _num_edges * 8;
			max_n_f = _num_polygons * 4;


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

			string hashKey = (to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z));
			positionVertex[hashKey] = vertexActive.size();

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

				max_n_f *= 4;
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

			int currentNumEdges = n_e;

			for (int i = 0; i < fVertices.size(); i++)
			{
				// check if edge exists
				int eID;
				bool chkEdge = edgeExists(fVertices[i], fVertices[(i + 1) % fVertices.size()], eID);

				if (chkEdge)
				{
					fEdge.push_back(eID);
				}
				else
				{
					addEdges(fVertices[i], fVertices[(i + 1) % fVertices.size()]);
					fEdge.push_back(edges[n_e - 2].getEdgeId());

				}
			}

			// update curent face edge
			faces[n_f - 1].setEdge(&edges[fEdge[0]]);

			//update current face verts edge pointer
			for (int i = 0; i < fVertices.size(); i++)
			{
				vertices[fVertices[i]].setEdge(&edges[fEdge[i]]);
			}

			// update face, next and prev edge pointers for face Edges
			for (int i = 0; i < fEdge.size(); i++)
			{
				edges[fEdge[i]].setFace(&faces[n_f - 1]);

				edges[fEdge[i]].setNext(&edges[fEdge[(i + 1) % fEdge.size()]]);
				edges[fEdge[i]].setPrev(&edges[fEdge[(i - 1 + fEdge.size()) % fEdge.size()]]);
			}

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
		*	\since version 0.0.1
		*/
		
		void setNumPolygons(int _n_f)
		{
			n_f = _n_f;
			max_n_f = 2 * n_f;
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

			}

		}

		/*! \brief This method computes the normals assoicated with vertices and polygon faces .
		*
		*	\since version 0.0.1
		*/
		void computeMeshNormals()
		{
			faceNormals.clear();

			for (int i = 0; i < numPolygons(); i++)
			{
				// get face vertices and correspondiing positions

				vector<int> fVerts;
				getVertices(i, zFaceData, fVerts);
				
				vector<zVector> points;
				for (int i = 0; i < fVerts.size(); i++)
				{
					points.push_back(vertexPositions[fVerts[i]]);
				}

				zVector cross = (points[1] - points[0]) ^ (points[fVerts.size() - 1] - points[0]);
				cross.normalize();
				
				if (fVerts.size() != 3)
				{
					// compute best plane

					zMatrixd bestPlane = getBestFitPlane(points);

					zVector norm = fromMatrixColumn(bestPlane, 2);
					norm.normalize();

					printf("\n");
					for (int k = 0; k < 3; k++)
					{
						vector<double> colVals = bestPlane.getCol(k);

						printf("\n %1.2f %1.2f %1.2f ", colVals[0], colVals[1], colVals[2]);
						
					}
					printf("\n");

					// check if the cross vector and normal vector are facing the same direction i.e out of the face
					norm *= (norm * cross < 0) ? -1 : 1;

					faceNormals.push_back(norm);
				}
				else
				{
					faceNormals.push_back(cross);
				}
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

					if (vertices[i].getEdge())resized[i].setEdge(vertices[i].getEdge());
				}
				for (int i = 0; i < edgeActive.size(); i++)
				{
					if (edges[i].getVertex()) edges[i].setVertex(&resized[edges[i].getVertex()->getVertexId()]);
				}

				delete[] vertices;		

				vertices = resized;

				//delete[] resized;
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

					if (edges[i].getVertex()) resized[i].setVertex(edges[i].getVertex());
					if (edges[i].getFace()) resized[i].setFace(edges[i].getFace());

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
					if (faces[i].getEdge()) resized[i].setEdge(faces[i].getEdge());

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

				//delete[] resized;
				printf("\n mesh faces resized. ");
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method updates the pointers for boundary Edges.
		*
		*	\param		[in]	numEdges		- number of edges in the mesh.
		*	\since version 0.0.1
		*/
		
		void update_BoundaryEdgePointers()
		{
			for (int i = 0; i < numEdges(); i++)
			{
				if (!edges[i].getFace())
				{
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

		/*! \brief This method resizes and copies information in to the vertex, edge and faces arrays of the current mesh from the coresponding arrays of input mesh.
		*
		*	\param		[in]	other			- input mesh to copy arrays from.
		*	\since version 0.0.1
		*/
		void copyArraysfromMesh(zMesh &other);

	};

}
