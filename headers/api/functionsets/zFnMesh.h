#pragma once


#include<headers/api/object/zObjMesh.h>
#include<headers/api/functionsets/zFn.h>




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

	/*! \class zFnMesh
	*	\brief A mesh function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnMesh : protected zFn
	{

	private:
		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*! \brief This method sets the edge and face vertex position conatiners for static meshes.
		*
		*	\since version 0.0.2
		*/
		void setStaticContainers()
		{
			meshObj->mesh.staticGeometry = true;

			vector<vector<zVector>> edgePositions;

			for (int i = 0; i < numEdges(); i ++)
			{
				vector<zVector> vPositions;
				getVertexPositions(i, zEdgeData, vPositions);

				edgePositions.push_back(vPositions);			
			}

			meshObj->mesh.setStaticEdgePositions(edgePositions);

				
			vector<vector<zVector>> facePositions;

			for (int i = 0; i < numPolygons(); i ++)
			{
				vector<zVector> vPositions;
				getVertexPositions(i, zFaceData, vPositions);

				facePositions.push_back(vPositions);
			}

			meshObj->mesh.setStaticFacePositions(facePositions);

			
		}

	protected:
		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method deactivates the input elements from the array connected with the input type.
		*
		*	\param		[in]	index			- index to be deactivated.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.2
		*/
		void deactivate(int index, zHEData type)
		{
			//  Vertex
			if (type == zVertexData)
			{
				if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				// remove from vertexPosition map
				meshObj->mesh.removeFromPositionMap(meshObj->mesh.vertexPositions[index]);

				// null pointers indexed vertex
				meshObj->mesh.vertices[index].removeVertex();

				// disable indexed vertex

				meshObj->mesh.vertexActive[index] = false;

				meshObj->mesh.vertexPositions[index] = zVector(10000, 10000, 10000); // dummy position for VBO
				meshObj->mesh.vertexNormals[index] = zVector(0, 0, 1); // dummy normal for VBO
				meshObj->mesh.vertexColors[index] = zColor(1, 1, 1, 1); // dummy color for VBO


																 // update numVertices
				int newNumVertices = numVertices() - 1;
				meshObj->mesh.setNumVertices(newNumVertices, false);
			}

			//  Edge
			else if (type == zEdgeData)
			{

				if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				int symEdge = meshObj->mesh.edges[index].getSym()->getEdgeId();

				// check if the vertex attached to the edge has the pointer to the current  edge. If true update the pointer. 
				int v1 = meshObj->mesh.edges[index].getVertex()->getVertexId();
				int v2 = meshObj->mesh.edges[symEdge].getVertex()->getVertexId();


				if (meshObj->mesh.vertices[v1].getEdge()->getEdgeId() == symEdge)
				{
					vector<int> cEdgesV1;
					getConnectedEdges(v1, zVertexData, cEdgesV1);

					for (int i = 0; i < cEdgesV1.size(); i++)
					{
						if (cEdgesV1[i] != symEdge) meshObj->mesh.vertices[v1].setEdge(&meshObj->mesh.edges[cEdgesV1[i]]);
					}
				}

				if (meshObj->mesh.vertices[v2].getEdge()->getEdgeId() == index)
				{
					vector<int> cEdgesV2;
					getConnectedEdges(v2, zVertexData, cEdgesV2);

					for (int i = 0; i < cEdgesV2.size(); i++)
					{
						if (cEdgesV2[i] != index) meshObj->mesh.vertices[v2].setEdge(&meshObj->mesh.edges[cEdgesV2[i]]);
					}
				}

				// check if the face attached to the edge has the pointer to the current edge. if true update pointer. 

				if (meshObj->mesh.edges[index].getFace())
				{

					if (meshObj->mesh.edges[index].getFace()->getEdge()->getEdgeId() == index)
					{
						vector<int> cEdgesV1;
						getConnectedEdges(v1, zVertexData, cEdgesV1);

						for (int i = 0; i < cEdgesV1.size(); i++)
						{
							if (meshObj->mesh.edges[cEdgesV1[i]].getFace() == meshObj->mesh.edges[index].getFace()) meshObj->mesh.edges[index].getFace()->setEdge(&meshObj->mesh.edges[cEdgesV1[i]]);
						}

					}

				}

				if (meshObj->mesh.edges[symEdge].getFace())
				{
					if (meshObj->mesh.edges[symEdge].getFace()->getEdge()->getEdgeId() == symEdge)
					{
						vector<int> cEdgesV2;
						getConnectedEdges(v2, zVertexData, cEdgesV2);

						for (int i = 0; i < cEdgesV2.size(); i++)
						{
							if (meshObj->mesh.edges[cEdgesV2[i]].getFace() == meshObj->mesh.edges[symEdge].getFace()) meshObj->mesh.edges[symEdge].getFace()->setEdge(&meshObj->mesh.edges[cEdgesV2[i]]);
						}

					}

				}

				// remove edge from vertex edge map
				meshObj->mesh.removeFromVerticesEdge(v1, v2);

				// make current edge pointer null
				meshObj->mesh.edges[index].removeEdge();
				meshObj->mesh.edges[symEdge].removeEdge();

				// deactivate edges
				meshObj->mesh.edgeActive[index] = false;
				meshObj->mesh.edgeActive[symEdge] = false;

				// update numEdges
				int newNumEdges = numEdges() - 2;
				meshObj->mesh.setNumEdges(newNumEdges, false);
			}

			// Face
			else if (type == zFaceData)
			{
				if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");


				// make current face pointers null
				meshObj->mesh.faces[index].removeFace();;

				// deactivate face
				meshObj->mesh.faceActive[index] = false;

				// update numPolygons
				int newNumFaces = numPolygons() - 1;
				meshObj->mesh.setNumPolygons(newNumFaces, false);
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

		}

		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.2
		*/
		void removeInactive(zHEData type = zVertexData)
		{
			//  Vertex
			if (type == zVertexData)
			{

				if (meshObj->mesh.vertexActive.size() != numVertices())
				{
					zVertex *resized = new zVertex[meshObj->mesh.max_n_v];

					int vertexActiveID = 0;
					int numOrginalVertexActive = meshObj->mesh.vertexActive.size();

					for (int i = 0; i < numVertices(); i++)
					{

						while (!meshObj->mesh.vertexActive[i])
						{
							meshObj->mesh.vertexActive.erase(meshObj->mesh.vertexActive.begin() + i);

							meshObj->mesh.vertexPositions.erase(meshObj->mesh.vertexPositions.begin() + i);

							meshObj->mesh.vertexColors.erase(meshObj->mesh.vertexColors.begin() + i);

							meshObj->mesh.vertexNormals.erase(meshObj->mesh.vertexNormals.begin() + i);

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
								meshObj->mesh.edges[cEdges[j]].getSym()->setVertex(&resized[i]);
							}


							if (meshObj->mesh.vertices[vertexActiveID].getEdge())
							{
								resized[i].setEdge(&meshObj->mesh.edges[meshObj->mesh.vertices[vertexActiveID].getEdge()->getEdgeId()]);

								meshObj->mesh.edges[meshObj->mesh.vertices[vertexActiveID].getEdge()->getEdgeId()].getSym()->setVertex(&resized[i]);
							}
						}


						vertexActiveID++;

					}

					//printf("\n m: %i %i ", numVertices(), vertexActive.size());


					delete[] meshObj->mesh.vertices;

					meshObj->mesh.vertices = resized;

					//printf("\n removed inactive vertices. ");
				}

			}

			//  Edge
			else if (type == zEdgeData)
			{
				if (meshObj->mesh.edgeActive.size() != numEdges())
				{
					zEdge *resized = new zEdge[meshObj->mesh.max_n_e];

					int edgeActiveID = 0;
					int numOrginalEdgeActive = meshObj->mesh.edgeActive.size();

					int inactiveCounter = 0;

					// clear vertices edge map
					meshObj->mesh.verticesEdge.clear();


					for (int i = 0; i < numEdges(); i += 2)
					{

						while (!meshObj->mesh.edgeActive[i])
						{
							meshObj->mesh.edgeActive.erase(meshObj->mesh.edgeActive.begin() + i);

							meshObj->mesh.edgeColors.erase(meshObj->mesh.edgeColors.begin() + i);

							edgeActiveID++;

						}


						resized[i].setEdgeId(i);
						resized[i + 1].setEdgeId(i + 1);

						// get connected edges and repoint their pointers
						if (edgeActiveID < numOrginalEdgeActive)
						{


							resized[i].setSym(&resized[i + 1]);

							if (meshObj->mesh.edges[edgeActiveID].getNext())
							{
								resized[i].setNext(&resized[meshObj->mesh.edges[edgeActiveID].getNext()->getEdgeId()]);

								meshObj->mesh.edges[edgeActiveID].getNext()->setPrev(&resized[i]);
							}
							if (meshObj->mesh.edges[edgeActiveID].getPrev())
							{
								resized[i].setPrev(&resized[meshObj->mesh.edges[edgeActiveID].getPrev()->getEdgeId()]);

								meshObj->mesh.edges[edgeActiveID].getPrev()->setNext(&resized[i]);
							}


							if (meshObj->mesh.edges[edgeActiveID].getVertex())
							{
								resized[i].setVertex(&meshObj->mesh.vertices[meshObj->mesh.edges[edgeActiveID].getVertex()->getVertexId()]);

								meshObj->mesh.vertices[meshObj->mesh.edges[edgeActiveID].getVertex()->getVertexId()].setEdge(resized[i].getSym());
							}

							if (meshObj->mesh.edges[edgeActiveID].getFace())
							{
								resized[i].setFace(&meshObj->mesh.faces[meshObj->mesh.edges[edgeActiveID].getFace()->getFaceId()]);
								meshObj->mesh.faces[meshObj->mesh.edges[edgeActiveID].getFace()->getFaceId()].setEdge(&resized[i]);
							}



							//sym edge
							if (meshObj->mesh.edges[edgeActiveID + 1].getNext())
							{
								resized[i + 1].setNext(&resized[meshObj->mesh.edges[edgeActiveID + 1].getNext()->getEdgeId()]);

								meshObj->mesh.edges[edgeActiveID + 1].getNext()->setPrev(&resized[i + 1]);

							}
							if (meshObj->mesh.edges[edgeActiveID + 1].getPrev())
							{
								resized[i + 1].setPrev(&resized[meshObj->mesh.edges[edgeActiveID + 1].getPrev()->getEdgeId()]);

								meshObj->mesh.edges[edgeActiveID + 1].getPrev()->setNext(&resized[i + 1]);
							}

							if (meshObj->mesh.edges[edgeActiveID + 1].getVertex())
							{
								resized[i + 1].setVertex(&meshObj->mesh.vertices[meshObj->mesh.edges[edgeActiveID + 1].getVertex()->getVertexId()]);
								meshObj->mesh.vertices[meshObj->mesh.edges[edgeActiveID + 1].getVertex()->getVertexId()].setEdge(resized[i + 1].getSym());
							}

							if (meshObj->mesh.edges[edgeActiveID + 1].getFace())
							{
								resized[i + 1].setFace(&meshObj->mesh.faces[meshObj->mesh.edges[edgeActiveID + 1].getFace()->getFaceId()]);
								meshObj->mesh.faces[meshObj->mesh.edges[edgeActiveID + 1].getFace()->getFaceId()].setEdge(&resized[i + 1]);
							}


							// rebuild vertices edge map
							int v2 = resized[i].getVertex()->getVertexId();
							int v1 = resized[i + 1].getVertex()->getVertexId();

							meshObj->mesh.addToVerticesEdge(v1, v2, i);

						}

						edgeActiveID += 2;

					}

					//printf("\n m: %i %i ", numEdges(), edgeActive.size());

					for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
					{
						if (meshObj->mesh.vertices[i].getEdge()) meshObj->mesh.vertices[i].setEdge(&resized[meshObj->mesh.vertices[i].getEdge()->getEdgeId()]);
					}

					for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
					{
						if (meshObj->mesh.faces[i].getEdge()) meshObj->mesh.faces[i].setEdge(&resized[meshObj->mesh.faces[i].getEdge()->getEdgeId()]);

					}

					delete[] meshObj->mesh.edges;
					meshObj->mesh.edges = resized;

					//printf("\n removed inactive edges. ");
				}



			}

			// Mesh Face
			else if (type == zFaceData)
			{
				if (meshObj->mesh.faceActive.size() != numPolygons())
				{
					zFace *resized = new zFace[meshObj->mesh.max_n_f];

					int faceActiveID = 0;
					int numOrginalFaceActive = meshObj->mesh.faceActive.size();

					for (int i = 0; i < numPolygons(); i++)
					{

						while (!meshObj->mesh.faceActive[i])
						{
							meshObj->mesh.faceActive.erase(meshObj->mesh.faceActive.begin() + i);

							meshObj->mesh.faceNormals.erase(meshObj->mesh.faceNormals.begin() + i);

							meshObj->mesh.faceColors.erase(meshObj->mesh.faceColors.begin() + i);

							faceActiveID++;

						}

						resized[i].setFaceId(i);


						// get connected edges and repoint their pointers
						if (faceActiveID < numOrginalFaceActive)
						{
							//printf("\n f: %i ", faceActiveID);
							vector<int> fEdges;
							getEdges(faceActiveID, zFaceData, fEdges);

							for (int j = 0; j < fEdges.size(); j++)
							{
								meshObj->mesh.edges[fEdges[j]].setFace(&resized[i]);
							}

							if (meshObj->mesh.faces[faceActiveID].getEdge()) resized[i].setEdge(&meshObj->mesh.edges[meshObj->mesh.faces[faceActiveID].getEdge()->getEdgeId()]);

						}

						faceActiveID++;

					}

					//printf("\n m: %i %i ", numPolygons(), faceActive.size());


					delete[] meshObj->mesh.faces;

					meshObj->mesh.faces = resized;

					//printf("\n removed inactive faces. ");
				}


			}

			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method exports zMesh as an OBJ file.
		*
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toOBJ(string outfilename)
		{

			// remove inactive elements
			if (numVertices() != meshObj->mesh.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != meshObj->mesh.edgeActive.size()) removeInactiveElements(zEdgeData);
			if (numPolygons() != meshObj->mesh.faceActive.size()) removeInactiveElements(zFaceData);

			// output file
			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				return;

			}



			// vertex positions
			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				myfile << "\n v " << meshObj->mesh.vertexPositions[i].x << " " << meshObj->mesh.vertexPositions[i].y << " " << meshObj->mesh.vertexPositions[i].z;

			}

			// vertex nornmals
			for (int i = 0; i < meshObj->mesh.vertexNormals.size(); i++)
			{
				myfile << "\n vn " << meshObj->mesh.vertexNormals[i].x << " " << meshObj->mesh.vertexNormals[i].y << " " << meshObj->mesh.vertexNormals[i].z;

			}

			myfile << "\n";

			// face connectivity
			for (int i = 0; i < numPolygons(); i++)
			{
				vector<int> fVerts;
				getVertices(i, zFaceData, fVerts);

				myfile << "\n f ";

				for (int j = 0; j < fVerts.size(); j++)
				{
					myfile << fVerts[j] + 1;

					if (j != fVerts.size() - 1) myfile << " ";
				}

			}




			myfile.close();

			cout << endl << " OBJ exported. File:   " << outfilename.c_str() << endl;

		}

		/*! \brief This method exports zMesh to a JSON file format using JSON Modern Library.
		*
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toJSON(string outfilename)
		{

			// remove inactive elements
			if (numVertices() != meshObj->mesh.vertexActive.size()) removeInactiveElements(zVertexData);
			if (numEdges() != meshObj->mesh.edgeActive.size()) removeInactiveElements(zEdgeData);
			if (numPolygons() != meshObj->mesh.faceActive.size())removeInactiveElements(zFaceData);

			// CREATE JSON FILE
			zUtilsJsonHE meshJSON;
			json j;

			// Vertices
			for (int i = 0; i < numVertices(); i++)
			{
				if (meshObj->mesh.vertices[i].getEdge()) meshJSON.vertices.push_back(meshObj->mesh.vertices[i].getEdge()->getEdgeId());
				else meshJSON.vertices.push_back(-1);

			}

			//Edges
			for (int i = 0; i < numEdges(); i++)
			{
				vector<int> HE_edges;

				if (meshObj->mesh.edges[i].getPrev()) HE_edges.push_back(meshObj->mesh.edges[i].getPrev()->getEdgeId());
				else HE_edges.push_back(-1);

				if (meshObj->mesh.edges[i].getNext()) HE_edges.push_back(meshObj->mesh.edges[i].getNext()->getEdgeId());
				else HE_edges.push_back(-1);

				if (meshObj->mesh.edges[i].getVertex()) HE_edges.push_back(meshObj->mesh.edges[i].getVertex()->getVertexId());
				else HE_edges.push_back(-1);

				if (meshObj->mesh.edges[i].getFace()) HE_edges.push_back(meshObj->mesh.edges[i].getFace()->getFaceId());
				else HE_edges.push_back(-1);

				meshJSON.halfedges.push_back(HE_edges);
			}

			// Faces
			for (int i = 0; i < numPolygons(); i++)
			{
				if (meshObj->mesh.faces[i].getEdge()) meshJSON.faces.push_back(meshObj->mesh.faces[i].getEdge()->getEdgeId());
				else meshJSON.faces.push_back(-1);
			}

			// vertex Attributes
			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				vector<double> v_attrib;

				v_attrib.push_back(meshObj->mesh.vertexPositions[i].x);
				v_attrib.push_back(meshObj->mesh.vertexPositions[i].y);
				v_attrib.push_back(meshObj->mesh.vertexPositions[i].z);

				v_attrib.push_back(meshObj->mesh.vertexNormals[i].x);
				v_attrib.push_back(meshObj->mesh.vertexNormals[i].y);
				v_attrib.push_back(meshObj->mesh.vertexNormals[i].z);

				
				v_attrib.push_back(meshObj->mesh.vertexColors[i].r);
				v_attrib.push_back(meshObj->mesh.vertexColors[i].g);
				v_attrib.push_back(meshObj->mesh.vertexColors[i].b);
				

				meshJSON.vertexAttributes.push_back(v_attrib);
			}

			// face Attributes
			for (int i = 0; i < numPolygons(); i++)
			{
				vector<double> f_attrib;

				f_attrib.push_back(meshObj->mesh.faceNormals[i].x);
				f_attrib.push_back(meshObj->mesh.faceNormals[i].y);
				f_attrib.push_back(meshObj->mesh.faceNormals[i].z);

				f_attrib.push_back(meshObj->mesh.faceColors[i].r);
				f_attrib.push_back(meshObj->mesh.faceColors[i].g);
				f_attrib.push_back(meshObj->mesh.faceColors[i].b);

				meshJSON.faceAttributes.push_back(f_attrib);
			}



			// Json file 
			j["Vertices"] = meshJSON.vertices;
			j["Halfedges"] = meshJSON.halfedges;
			j["Faces"] = meshJSON.faces;
			j["VertexAttributes"] = meshJSON.vertexAttributes;
			j["HalfedgeAttributes"] = meshJSON.halfedgeAttributes;
			j["FaceAttributes"] = meshJSON.faceAttributes;

			// EXPORT	

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

		/*! \brief This method imports zMesh from an OBJ file.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromOBJ(string infilename)
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

				vector<string> perlineData = meshObj->mesh.coreUtils.splitString(str, " ");

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
							vector<string> faceData = meshObj->mesh.coreUtils.splitString(perlineData[i], "/");

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


			meshObj->mesh = zMesh(positions, polyCounts, polyConnects);;
			printf("\n mesh: %i %i %i", numVertices(), numEdges(), numPolygons());


			setFaceNormals(faceNormals);

		}

		/*! \brief This method imports zMesh from a JSON file format using JSON Modern Library.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromJSON(string infilename)
		{
			json j;
			zUtilsJsonHE meshJSON;

			
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

			// READ Data from JSON

			// Vertices
			meshJSON.vertices.clear();
			meshJSON.vertices = (j["Vertices"].get<vector<int>>());

			//Edges
			meshJSON.halfedges.clear();
			meshJSON.halfedges = (j["Halfedges"].get<vector<vector<int>>>());

			// Faces
			meshJSON.faces.clear();
			meshJSON.faces = (j["Faces"].get<vector<int>>());


			// update  mesh

			meshObj->mesh.vertices = new zVertex[meshJSON.vertices.size() * 2];
			meshObj->mesh.edges = new zEdge[meshJSON.halfedges.size() * 2];
			meshObj->mesh.faces = new zFace[meshJSON.faces.size() * 2];

			meshObj->mesh.setNumVertices(meshJSON.vertices.size());
			meshObj->mesh.setNumEdges(floor(meshJSON.halfedges.size()));
			meshObj->mesh.setNumPolygons(meshJSON.faces.size());

			meshObj->mesh.vertexActive.clear();
			for (int i = 0; i < meshJSON.vertices.size(); i++)
			{
				meshObj->mesh.vertices[i].setVertexId(i);
				if (meshJSON.vertices[i] != -1) meshObj->mesh.vertices[i].setEdge(&meshObj->mesh.edges[meshJSON.vertices[i]]);

				meshObj->mesh.vertexActive.push_back(true);
			}

			int k = 0;
			meshObj->mesh.edgeActive.clear();
			for (int i = 0; i < meshJSON.halfedges.size(); i++)
			{
				meshObj->mesh.edges[i].setEdgeId(i);

				if (meshJSON.halfedges[i][k] != -1) 	meshObj->mesh.edges[i].setPrev(&meshObj->mesh.edges[meshJSON.halfedges[i][k]]);
				if (meshJSON.halfedges[i][k + 1] != -1) meshObj->mesh.edges[i].setNext(&meshObj->mesh.edges[meshJSON.halfedges[i][k + 1]]);
				if (meshJSON.halfedges[i][k + 2] != -1) meshObj->mesh.edges[i].setVertex(&meshObj->mesh.vertices[meshJSON.halfedges[i][k + 2]]);
				if (meshJSON.halfedges[i][k + 3] != -1) meshObj->mesh.edges[i].setFace(&meshObj->mesh.faces[meshJSON.halfedges[i][k + 3]]);


				if (i % 2 == 0) meshObj->mesh.edges[i].setSym(&meshObj->mesh.edges[i]);
				else  meshObj->mesh.edges[i].setSym(&meshObj->mesh.edges[i - 1]);

				meshObj->mesh.edgeActive.push_back(true);
			}

			meshObj->mesh.faceActive.clear();
			for (int i = 0; i < meshJSON.faces.size(); i++)
			{
				meshObj->mesh.faces[i].setFaceId(i);
				if (meshJSON.faces[i] != -1) meshObj->mesh.faces[i].setEdge(&meshObj->mesh.edges[meshJSON.faces[i]]);

				meshObj->mesh.faceActive.push_back(true);
			}

			// Vertex Attributes
			meshJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
			//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

			meshObj->mesh.vertexPositions.clear();
			for (int i = 0; i < meshJSON.vertexAttributes.size(); i++)
			{
				for (int k = 0; k < meshJSON.vertexAttributes[i].size(); k++)
				{
					
					// position and color

					if (meshJSON.vertexAttributes[i].size() == 9)
					{
						zVector pos(meshJSON.vertexAttributes[i][k], meshJSON.vertexAttributes[i][k + 1], meshJSON.vertexAttributes[i][k + 2]);
						meshObj->mesh.vertexPositions.push_back(pos);

						zVector normal(meshJSON.vertexAttributes[i][k + 3], meshJSON.vertexAttributes[i][k + 4], meshJSON.vertexAttributes[i][k + 5]);
						meshObj->mesh.vertexNormals.push_back(pos);

						zColor col(meshJSON.vertexAttributes[i][k + 6], meshJSON.vertexAttributes[i][k + 7], meshJSON.vertexAttributes[i][k + 8], 1);
						meshObj->mesh.vertexColors.push_back(col);

						k += 8;
					}
				}
			}


			// Edge Attributes
			meshJSON.halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();


			// face Attributes
			meshJSON.faceAttributes = j["FaceAttributes"].get<vector<vector<double>>>();

			meshObj->mesh.faceColors.clear();
			meshObj->mesh.faceNormals.clear();
			for (int i = 0; i < meshJSON.faceAttributes.size(); i++)
			{
				for (int k = 0; k < meshJSON.faceAttributes[i].size(); k++)
				{
					// position
					if (meshJSON.faceAttributes[i].size() == 6)
					{
						zColor col(meshJSON.faceAttributes[i][k + 3], meshJSON.faceAttributes[i][k + 4], meshJSON.faceAttributes[i][k + 5], 1);
						meshObj->mesh.faceColors.push_back(col);


						zVector normal(meshJSON.faceAttributes[i][k], meshJSON.faceAttributes[i][k + 1], meshJSON.faceAttributes[i][k + 2]);
						meshObj->mesh.faceNormals.push_back(normal);
											   
						k += 5;
					}
				}

			}
			

			// add to maps 
			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				meshObj->mesh.addToPositionMap(meshObj->mesh.vertexPositions[i], i);
			}


			for (int i = 0; i < numEdges(); i += 2)
			{
				int v1 = meshObj->mesh.edges[i].getVertex()->getVertexId();
				int v2 = meshObj->mesh.edges[i + 1].getVertex()->getVertexId();

				meshObj->mesh.addToVerticesEdge(v1, v2, i);
			}

		}

	public:
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnMesh()
		{
			fnType = zFnType::zMeshFn;
			meshObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zFnMesh(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;
			fnType = zFnType::zMeshFn;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnMesh() {}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------
		
		void from(string path, zFileTpye type, bool staticGeom = false) override
		{
			if (type == zOBJ)
			{
				fromOBJ(path);
				if(staticGeom) setStaticContainers();
			}
			
			else if (type == zJSON)
			{
				fromJSON(path);
				if (staticGeom) setStaticContainers();
			}
			
			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		void to(string path, zFileTpye type) override
		{
			if (type == zOBJ) toOBJ(path);
			else if (type == zJSON) toJSON(path);

			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		void clear() override 
		{
			if (meshObj->mesh.vertices != NULL)
			{
				delete[] meshObj->mesh.vertices;
				meshObj->mesh.vertices = NULL;

				meshObj->mesh.vertexActive.clear();
				meshObj->mesh.vertexPositions.clear();
				meshObj->mesh.vertexNormals.clear();
				meshObj->mesh.vertexColors.clear();
				meshObj->mesh.vertexWeights.clear();

				meshObj->mesh.positionVertex.clear();
				meshObj->mesh.verticesEdge.clear();


			}

			if (meshObj->mesh.edges != NULL)
			{
				delete[] meshObj->mesh.edges;
				meshObj->mesh.edges = NULL;

				meshObj->mesh.edgeActive.clear();
				meshObj->mesh.edgeColors.clear();
				meshObj->mesh.edgeWeights.clear();

			}


			if (meshObj->mesh.faces != NULL)
			{
				delete[]meshObj->mesh.faces;
				meshObj->mesh.faces = NULL;

				meshObj->mesh.faceActive.clear();
				meshObj->mesh.faceColors.clear();
				meshObj->mesh.faceNormals.clear();
			}

			meshObj->mesh.n_v = meshObj->mesh.n_e = meshObj->mesh.n_f = 0;
		}
		
		//--------------------------
		//---- CREATE METHODS
		//--------------------------

				
		/*! \brief This method creates a mesh from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\param		[in]	polyCounts		- container of type integer with number of vertices per polygon.
		*	\param		[in]	polyConnects	- polygon connection list with vertex ids for each face.
		*	\param		[in]	staticMesh		- makes the mesh fixed. Computes the static edge and face vertex positions if true.
		*	\since version 0.0.2
		*/
		void create(vector<zVector>(&_positions), vector<int>(&polyCounts), vector<int>(&polyConnects), bool staticMesh = false)
		{
			meshObj->mesh = zMesh(_positions, polyCounts, polyConnects);

			// compute mesh normals
			computeMeshNormals();

			if (staticMesh) setStaticContainers();
			
		}
				

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		
		/*! \brief This method returns the number of vertices in the mesh.
		*	\return				number of vertices.
		*	\since version 0.0.2
		*/
		int numVertices()
		{
			return meshObj->mesh.n_v;
		}

		/*! \brief This method returns the number of half edges in the mesh.
		*	\return				number of edges.
		*	\since version 0.0.2
		*/
		int numEdges()
		{
			return meshObj->mesh.n_e;
		}

		/*! \brief This method returns the number of polygons in the mesh
		*
		*	\return		int		-	number of polygons
		*	\since version 0.0.2
		*/
		int numPolygons()
		{
			return meshObj->mesh.n_f;
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
			std::unordered_map<std::string, int>::const_iterator got = meshObj->mesh.positionVertex.find(hashKey);


			if (got != meshObj->mesh.positionVertex.end())
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
			std::unordered_map<std::string, int>::const_iterator got = meshObj->mesh.verticesEdge.find(hashKey);


			if (got != meshObj->mesh.verticesEdge.end())
			{
				out = true;
				outEdgeId = got->second;
			}

			return out;
		}

		/*! \brief This method gets the edges of a zFace.
		*
		*	\param		[in]	index			- index in the face container.
		*	\param		[in]	type			- zFaceData.
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.2
		*/
		void getEdges(int index, zHEData type, vector<int> &edgeIndicies)
		{
			
			edgeIndicies.clear();

			// Face 
			if (type == zFaceData)
			{
				if (meshObj->mesh.faces[index].getEdge())
				{
					zEdge* start = meshObj->mesh.faces[index].getEdge();
					zEdge* e = start;

					bool exit = false;

					do
					{
						edgeIndicies.push_back(e->getEdgeId());
						if (e->getNext())e = e->getNext();
						else exit = true;

					} while (e != start && !exit);
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			
		}

		/*!	\brief This method gets the vertices attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge/face container.
		*	\param		[in]	type			- zEdgeData or zFaceData.
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.2
		*/
		void getVertices(int index, zHEData type, vector<int> &vertexIndicies)
		{
			
			vertexIndicies.clear();

			// Edge
			if (type == zEdgeData)
			{
				vertexIndicies.push_back(meshObj->mesh.edges[index].getVertex()->getVertexId());
				vertexIndicies.push_back(meshObj->mesh.edges[index].getSym()->getVertex()->getVertexId());
			}


			// Face 
			else if (type == zFaceData)
			{

				vector<int> faceEdges;
				getEdges(index, type, faceEdges);

				for (int i = 0; i < faceEdges.size(); i++)
				{
					//out.push_back(edges[faceEdges[i]].getVertex()->getVertexId());
					vertexIndicies.push_back(meshObj->mesh.edges[faceEdges[i]].getSym()->getVertex()->getVertexId());
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");

			
		}

		/*!	\brief This method gets the vertex positions attached to input zEdge or zFace.
		*
		*	\param		[in]	index			- index in the edge/face container.
		*	\param		[in]	type			- zEdgeData or zFaceData.
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.2
		*/
		void getVertexPositions(int index, zHEData type, vector<zVector> &vertPositions)
		{
			vertPositions.clear();

			// Mesh Edge
			if (type == zEdgeData)
			{

				vector<int> eVerts;

				getVertices(index, type, eVerts);

				for (int i = 0; i < eVerts.size(); i++)
				{
					vertPositions.push_back(meshObj->mesh.vertexPositions[eVerts[i]]);
				}

			}


			// Mesh Face 
			else if (type == zFaceData)
			{

				vector<int> fVerts;

				getVertices(index, type, fVerts);

				for (int i = 0; i < fVerts.size(); i++)
				{
					vertPositions.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);
				}
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
						
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
				if (meshObj->mesh.vertices[index].getEdge())
				{
					zEdge* start = meshObj->mesh.vertices[index].getEdge();
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
				getConnectedEdges(meshObj->mesh.edges[index].getVertex()->getVertexId(), zVertexData, connectedEdgestoVert0);

				vector<int> connectedEdgestoVert1;
				getConnectedEdges(meshObj->mesh.edges[index].getSym()->getVertex()->getVertexId(), zVertexData, connectedEdgestoVert1);

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
					vertexIndicies.push_back(meshObj->mesh.edges[connectedEdges[i]].getVertex()->getVertexId());
				}

			}

			else throw std::invalid_argument(" error: invalid zHEData type");
						
		}

		/*! \brief This method gets the faces connected to input zVertex or zFace
		*
		*	\param		[in]	index	- index in the vertex/face container.
		*	\param		[in]	type	- zVertexData or zFaceData.
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.2
		*/
		void getConnectedFaces(int index, zHEData type, vector<int> &faceIndicies)
		{
			faceIndicies.clear();

			// Vertex
			if (type == zVertexData)
			{
				vector<int> connectedEdges;
				getConnectedEdges(index, type, connectedEdges);

				for (int i = 0; i < connectedEdges.size(); i++)
				{
					if (meshObj->mesh.edges[connectedEdges[i]].getFace()) faceIndicies.push_back(meshObj->mesh.edges[connectedEdges[i]].getFace()->getFaceId());
				}
			}

			// Face
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
						if (eFaces[k] != index) faceIndicies.push_back(eFaces[k]);
					}

					/*if (edges[fEdges[i]].f)
					{
					if(edges[fEdges[i]].f->faceId != index) out.push_back(edges[fEdges[i]].f->faceId);
					if(edges)
					}*/
				}


			}
			else throw std::invalid_argument(" error: invalid zHEData type");
						
		}

		/*! \brief This method gets the faces attached to input zEdge
		*
		*	\param		[in]	index			- index in the edge list.
		*	\param		[in]	type			- zEdgeData.
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.2
		*/
		void getFaces(int &index, zHEData type, vector<int> &faceIndicies)
		{

			faceIndicies.clear();

			// Mesh Edge
			if (type == zEdgeData)
			{
				if (meshObj->mesh.edges[index].getFace()) faceIndicies.push_back(meshObj->mesh.edges[index].getFace()->getFaceId());
				if (meshObj->mesh.edges[index].getSym()->getFace()) faceIndicies.push_back(meshObj->mesh.edges[index].getSym()->getFace()->getFaceId());
			}
			else throw std::invalid_argument(" error: invalid zHEData type");

			
		}

		/*!	\brief This method determines if  input zVertex or zEdge or zFace is on the boundary.
		*
		*	\param		[in]	index	- index in the vertex/edge/face list.
		*	\param		[in]	type	- zVertexData or zEdgeData or zFaceData.
		*	\return				bool	- true if on boundary else false.
		*	\since version 0.0.2
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
				if (!meshObj->mesh.edges[index].getFace()) out = true;
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
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getNext();
		}

		/*!	\brief This method return the next edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of next edge if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getNextIndex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if(meshObj->mesh.edges[index].getNext()) return meshObj->mesh.edges[index].getNext()->getEdgeId();
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
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getPrev();
		}

		/*!	\brief This method return the previous edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of previous edge if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getPrevIndex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (meshObj->mesh.edges[index].getPrev()) return meshObj->mesh.edges[index].getPrev()->getEdgeId();
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
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getSym();
		}

		/*!	\brief This method return the symmetry edge index of the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of symmetry edge.
		*	\since version 0.0.2
		*/
		int getSymIndex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getSym()->getEdgeId();			
		}

		/*!	\brief This method return the face attached to the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zFace	- pointer to face.
		*	\since version 0.0.2
		*/
		zFace* getFace(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getFace();
		}

		/*!	\brief This method return the face index attached to the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of face if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getFaceIndex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (meshObj->mesh.edges[index].getFace()) return meshObj->mesh.edges[index].getFace()->getFaceId();
			else return -1;
		}

		/*!	\brief This method return the vertex pointed by the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				zVertex	- pointer to vertex.
		*	\since version 0.0.2
		*/
		zVertex* getEndVertex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edges[index].getVertex();
		}

		/*!	\brief This method return the vertex pointed by the input indexed edge.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\return				int		- index of vertex if it exists , else -1.
		*	\since version 0.0.2
		*/
		int getEndVertexIndex(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (meshObj->mesh.edges[index].getVertex()) return meshObj->mesh.edges[index].getVertex()->getVertexId();
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
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

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
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (getSym(index)->getVertex()) return getSym(index)->getVertex()->getVertexId();
			else return -1;
		}

		/*!	\brief This method return the edge attached to the input indexed vertex or edge or  face.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	type	- zVertexData or zEdgeData or zFaceData.
		*	\return				zEdge	- pointer to edge.
		*	\since version 0.0.2
		*/
		zEdge* getEdge(int index, zHEData type)
		{
			if (type == zVertexData)
			{
				if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return meshObj->mesh.vertices[index].getEdge();
			}

			else if (type == zEdgeData)
			{
				if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return &meshObj->mesh.edges[index];
			}

			else if (type == zFaceData)
			{
				if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return meshObj->mesh.faces[index].getEdge();
			}

			else throw std::invalid_argument(" error: invalid zHEData type");
			
		}

		/*!	\brief This method return the index of the edge attached to the input indexed vertex or face.
		*
		*	\param		[in]	index	- index in the vertex list.
		*	\param		[in]	type	- zVertexData or zFaceData.
		*	\return				int		- index of edge.
		*	\since version 0.0.2
		*/
		int getEdgeIndex(int index, zHEData type)
		{
			if (type == zVertexData)
			{
				if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return meshObj->mesh.vertices[index].getEdge()->getEdgeId();
			}

			else if (type == zFaceData)
			{
				if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				return meshObj->mesh.faces[index].getEdge()->getEdgeId();
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

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				if (meshObj->mesh.edgeActive[i])
				{
					int v0 = meshObj->mesh.edges[i].getVertex()->getVertexId();
					int v1 = meshObj->mesh.edges[i + 1].getVertex()->getVertexId();

					zColor col;
					col.r = (meshObj->mesh.vertexColors[v0].r + meshObj->mesh.vertexColors[v1].r) * 0.5;
					col.g = (meshObj->mesh.vertexColors[v0].g + meshObj->mesh.vertexColors[v1].g) * 0.5;
					col.b = (meshObj->mesh.vertexColors[v0].b + meshObj->mesh.vertexColors[v1].b) * 0.5;
					col.a = (meshObj->mesh.vertexColors[v0].a + meshObj->mesh.vertexColors[v1].a) * 0.5;

					if (meshObj->mesh.edgeColors.size() <= i) meshObj->mesh.edgeColors.push_back(col);
					else meshObj->mesh.edgeColors[i] = col;

					if (meshObj->mesh.edgeColors.size() <= i + 1) meshObj->mesh.edgeColors.push_back(col);
					else meshObj->mesh.edgeColors[i + 1] = col;
				}


			}



		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.2
		*/
		void computeVertexColorfromEdgeColor()
		{
			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				if (meshObj->mesh.vertexActive[i])
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					zColor col;
					for (int j = 0; j < cEdges.size(); j++)
					{
						col.r += meshObj->mesh.edgeColors[cEdges[j]].r;
						col.g += meshObj->mesh.edgeColors[cEdges[j]].g;
						col.b += meshObj->mesh.edgeColors[cEdges[j]].b;
					}

					col.r /= cEdges.size(); col.g /= cEdges.size(); col.b /= cEdges.size();

					meshObj->mesh.vertexColors[i] = col;

				}
			}
		}

		/*! \brief This method computes the face colors based on the vertex colors.
		*
		*	\since version 0.0.2
		*/
		void computeFaceColorfromVertexColor()
		{
			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				if (meshObj->mesh.faceActive[i])
				{
					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					zColor col;
					for (int j = 0; j < fVerts.size(); j++)
					{
						col.r += meshObj->mesh.vertexColors[fVerts[j]].r;
						col.g += meshObj->mesh.vertexColors[fVerts[j]].g;
						col.b += meshObj->mesh.vertexColors[fVerts[j]].b;
					}

					col.r /= fVerts.size(); col.g /= fVerts.size(); col.b /= fVerts.size();

					meshObj->mesh.faceColors[i] = col;
				}
			}

		}

		/*! \brief This method computes the vertex colors based on the face colors.
		*
		*	\since version 0.0.2
		*/
		void computeVertexColorfromFaceColor()
		{

			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				if (meshObj->mesh.vertexActive[i])
				{
					vector<int> cFaces;
					getConnectedFaces(i, zVertexData, cFaces);

					zColor col;
					for (int j = 0; j < cFaces.size(); j++)
					{
						col.r += meshObj->mesh.faceColors[cFaces[j]].r;
						col.g += meshObj->mesh.faceColors[cFaces[j]].g;
						col.b += meshObj->mesh.faceColors[cFaces[j]].b;
					}

					col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

					meshObj->mesh.vertexColors[i] = col;
				}
			}

		}

		/*! \brief This method smoothens the color attributes.
		*	\param		[in]	smoothVal		- number of iterations to run the smooth operation.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData.
		*	\since version 0.0.2
		*/
		void smoothColors(int smoothVal = 1, zHEData type = zVertexData)
		{
			for (int j = 0; j < smoothVal; j++)
			{
				if (type == zVertexData)
				{
					vector<zColor> tempColors;

					for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
					{
						zColor col;
						if (meshObj->mesh.vertexActive[i])
						{
							vector<int> cVerts;
							getConnectedVertices(i, zVertexData, cVerts);

							zColor currentCol = meshObj->mesh.vertexColors[i];


							for (int j = 0; j < cVerts.size(); j++)
							{
								col.r += meshObj->mesh.vertexColors[cVerts[j]].r;
								col.g += meshObj->mesh.vertexColors[cVerts[j]].g;
								col.b += meshObj->mesh.vertexColors[cVerts[j]].b;
							}

							col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

							col.r /= cVerts.size(); col.g /= cVerts.size(); col.b /= cVerts.size();


						}

						tempColors.push_back(col);

					}

					for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
					{
						if (meshObj->mesh.vertexActive[i])
						{
							meshObj->mesh.vertexColors[i] = (tempColors[i]);
						}
					}
				}


				if (type == zFaceData)
				{
					vector<zColor> tempColors;

					for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
					{
						zColor col;
						if (meshObj->mesh.faceActive[i])
						{
							vector<int> cFaces;
							getConnectedFaces(i, zFaceData, cFaces);

							zColor currentCol = meshObj->mesh.faceColors[i];
							for (int j = 0; j < cFaces.size(); j++)
							{
								col.r += meshObj->mesh.faceColors[cFaces[j]].r;
								col.g += meshObj->mesh.faceColors[cFaces[j]].g;
								col.b += meshObj->mesh.faceColors[cFaces[j]].b;
							}

							col.r += (currentCol.r); col.g += (currentCol.g); col.b += (currentCol.b);

							col.r /= cFaces.size(); col.g /= cFaces.size(); col.b /= cFaces.size();

						}

						tempColors.push_back(col);

					}

					for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
					{
						if (meshObj->mesh.faceActive[i])
						{
							meshObj->mesh.faceColors[i] = (tempColors[i]);
						}
					}
				}

				else throw std::invalid_argument(" error: invalid zHEData type");
			}



		}

		/*! \brief This method computes the vertex normals based on the face normals.
		*
		*	\since version 0.0.2
		*/
		void computeVertexNormalfromFaceNormal()
		{

			meshObj->mesh.vertexNormals.clear();

			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				if (meshObj->mesh.vertexActive[i])
				{
					vector<int> cFaces;
					getConnectedFaces(i, zVertexData, cFaces);

					zVector norm;

					for (int j = 0; j < cFaces.size(); j++)
					{
						norm += meshObj->mesh.faceNormals[cFaces[j]];
					}

					norm /= cFaces.size();
					norm.normalize();
					meshObj->mesh.vertexNormals.push_back(norm);
				}
				else meshObj->mesh.vertexNormals.push_back(zVector());
			}

		}

		/*! \brief This method computes the normals assoicated with vertices and polygon faces .
		*
		*	\since version 0.0.2
		*/
		void computeMeshNormals()
		{
			meshObj->mesh.faceNormals.clear();

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				if (meshObj->mesh.faceActive[i])
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
						points.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);

						fCen += meshObj->mesh.vertexPositions[fVerts[i]];
					}

					fCen /= fVerts.size();

					zVector fNorm; // face normal

					if (fVerts.size() != 3)
					{
						for (int j = 0; j < fVerts.size(); j++)
						{
							fNorm += (points[j] - fCen) ^ (points[(j + 1) % fVerts.size()] - fCen);


						}

					}
					else
					{
						zVector cross = (points[1] - points[0]) ^ (points[fVerts.size() - 1] - points[0]);
						cross.normalize();

						fNorm = cross;

						//printf("\n working! %i ", i);

					}


					fNorm.normalize();
					meshObj->mesh.faceNormals.push_back(fNorm);
				}
				else meshObj->mesh.faceNormals.push_back(zVector());

			}


			// compute vertex normal
			computeVertexNormalfromFaceNormal();
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

				for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
				{
					tempVertPos.push_back(meshObj->mesh.vertexPositions[i]);

					if (meshObj->mesh.vertexActive[i])
					{
						if (!checkVertexValency(i, 1))
						{
							vector<int> cVerts;

							getConnectedVertices(i, zVertexData, cVerts);

							for (int j = 0; j < cVerts.size(); j++)
							{
								zVector p = meshObj->mesh.vertexPositions[cVerts[j]];
								tempVertPos[i] += p;
							}

							tempVertPos[i] /= (cVerts.size() + 1);
						}
					}

				}

				// update position
				for (int i = 0; i < tempVertPos.size(); i++) meshObj->mesh.vertexPositions[i] = tempVertPos[i];
			}

		}

		/*! \brief This method deactivate the input elements from the array connected with the input type.
		*
		*	\param		[in]	index			-  index in element container.
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData .
		*	\since version 0.0.2
		*/
		void deactivateElement(int index, zHEData type)
		{
			deactivate(index,type);
		}
		
		/*! \brief This method removes inactive elements from the array connected with the input type.
		*
		*	\param		[in]	type			- zVertexData or zEdgeData or zFaceData .
		*	\since version 0.0.2
		*/
		void removeInactiveElements(zHEData type)
		{
			removeInactive(type);
			
		}

		/*! \brief This method makes the mesh a static mesh. Makes the mesh fixed and computes the static edge and face vertex positions if true.
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
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			meshObj->mesh.vertexPositions[index] = pos;

		}

		/*! \brief This method sets vertex positions of all the vertices.
		*
		*	\param		[in]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void setVertexPositions(vector<zVector>& pos)
		{
			if (pos.size() != meshObj->mesh.vertexPositions.size()) throw std::invalid_argument("size of position contatiner is not equal to number of mesh vertices.");

			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				meshObj->mesh.vertexPositions[i] = pos[i];
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

			if (meshObj->mesh.vertexColors.size() != meshObj->mesh.vertexActive.size())
			{
				meshObj->mesh.vertexColors.clear();
				for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++) meshObj->mesh.vertexColors.push_back(zColor());
			}

			meshObj->mesh.vertexColors[index] = col;

		}

		/*! \brief This method sets vertex color of all the vertices to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColor(zColor col, bool setFaceColor = false)
		{
			if (meshObj->mesh.vertexColors.size() != meshObj->mesh.vertexActive.size())
			{
				meshObj->mesh.vertexColors.clear();
				for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++) meshObj->mesh.vertexColors.push_back(zColor(1, 0, 0, 1));
			}

			for (int i = 0; i < meshObj->mesh.vertexColors.size(); i++)
			{
				meshObj->mesh.vertexColors[i] = col;
			}

			if (setFaceColor) computeFaceColorfromVertexColor();
		}

		/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the mesh.
		*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColors(vector<zColor>& col, bool setFaceColor = false)
		{
			if (meshObj->mesh.vertexColors.size() != meshObj->mesh.vertexActive.size())
			{
				meshObj->mesh.vertexColors.clear();
				for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++) meshObj->mesh.vertexColors.push_back(zColor(1, 0, 0, 1));
			}

			if (col.size() != meshObj->mesh.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

			for (int i = 0; i < meshObj->mesh.vertexColors.size(); i++)
			{
				meshObj->mesh.vertexColors[i] = col[i];
			}

			if (setFaceColor) computeFaceColorfromVertexColor();
		}

		/*! \brief This method sets edge color of of the input vertex to the input color.
		*
		*	\param		[in]	index					- input vertex index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setFaceColor(int index, zColor col)
		{

			if (meshObj->mesh.faceColors.size() != meshObj->mesh.faceActive.size())
			{
				meshObj->mesh.faceColors.clear();
				for (int i = 0; i < meshObj->mesh.faceActive.size(); i++) meshObj->mesh.faceColors.push_back(zColor());
			}

			meshObj->mesh.faceColors[index] = col;

		}

		/*! \brief This method sets face color of all the faces to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the face color if true.
		*	\since version 0.0.2
		*/
		void setFaceColor(zColor col, bool setVertexColor = false)
		{
			if (meshObj->mesh.faceColors.size() != meshObj->mesh.faceActive.size())
			{
				meshObj->mesh.faceColors.clear();
				for (int i = 0; i < meshObj->mesh.faceActive.size(); i++) meshObj->mesh.faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
			}

			for (int i = 0; i < meshObj->mesh.faceColors.size(); i++)
			{
				meshObj->mesh.faceColors[i] = col;
			}

			if (setVertexColor) computeVertexColorfromFaceColor();
		}

		/*! \brief This method sets face color of all the faces to the input color contatiner.
		*
		*	\param		[in]	col				- input color contatiner. The size of the contatiner should be equal to number of faces in the mesh.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the face color if true.
		*	\since version 0.0.2
		*/
		void setFaceColors(vector<zColor>& col, bool setVertexColor = false)
		{
			if (meshObj->mesh.faceColors.size() != meshObj->mesh.faceActive.size())
			{
				meshObj->mesh.faceColors.clear();
				for (int i = 0; i < meshObj->mesh.faceActive.size(); i++) meshObj->mesh.faceColors.push_back(zColor(0.5, 0.5, 0.5, 1));
			}

			if (col.size() != meshObj->mesh.faceColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

			for (int i = 0; i < meshObj->mesh.faceColors.size(); i++)
			{
				meshObj->mesh.faceColors[i] = col[i];
			}

			if (setVertexColor) computeVertexColorfromFaceColor();
		}


		/*! \brief This method sets face normals of all the faces to the input normal.
		*
		*	\param		[in]	fNormal			- input normal.
		*	\since version 0.0.2
		*/
		void setFaceNormals(zVector &fNormal)
		{

			meshObj->mesh.faceNormals.clear();

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				meshObj->mesh.faceNormals.push_back(fNormal);
			}

			// compute normals per face based on vertex normals and store it in faceNormals
			computeVertexNormalfromFaceNormal();
		}

		/*! \brief This method sets face normals of all the faces to the input normals contatiner.
		*
		*	\param		[in]	fNormals		- input normals contatiner. The size of the contatiner should be equal to number of faces in the mesh.
		*	\since version 0.0.2
		*/
		void setFaceNormals(vector<zVector> &fNormals)
		{

			if (meshObj->mesh.faceActive.size() != fNormals.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh faces.");

			meshObj->mesh.faceNormals.clear();

			for (int i = 0; i < fNormals.size(); i++)
			{
				meshObj->mesh.faceNormals.push_back(fNormals[i]);
			}

			// compute normals per face based on vertex normals and store it in faceNormals
			computeVertexNormalfromFaceNormal();
		}

		/*! \brief This method sets edge color of of the input edge and its symmetry edge to the input color.
		*
		*	\param		[in]	index					- input edge index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setEdgeColor(int index, zColor col)
		{

			if (meshObj->mesh.edgeColors.size() != meshObj->mesh.edgeActive.size())
			{
				meshObj->mesh.edgeColors.clear();
				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++) meshObj->mesh.edgeColors.push_back(zColor());
								
			}

			meshObj->mesh.edgeColors[index] = col;

			int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

			meshObj->mesh.edgeColors[symEdge] = col;

		}

		/*! \brief This method sets edge color of all the edges to the input color.
		*
		*	\param		[in]	col				- input color.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColor(zColor col, bool setVertexColor = false)
		{
			if (meshObj->mesh.edgeColors.size() != meshObj->mesh.edgeActive.size())
			{
				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++) meshObj->mesh.edgeColors.push_back(zColor());
			}

			for (int i = 0; i < meshObj->mesh.edgeColors.size(); i += 2)
			{
				setEdgeColor( i, col);
			}

			if (setVertexColor) computeVertexColorfromEdgeColor();

		}

		/*! \brief This method sets edge color of all the edges with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of half edges in the mesh.
		*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
		*	\since version 0.0.2
		*/
		void setEdgeColors(vector<zColor>& col, bool setVertexColor)
		{
			if (col.size() != meshObj->mesh.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh half edges.");

			for (int i = 0; i < meshObj->mesh.edgeColors.size(); i++)
			{
				meshObj->mesh.edgeColors[i] = col[i];
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

			if (meshObj->mesh.edgeWeights.size() != meshObj->mesh.edgeActive.size())
			{
				meshObj->mesh.edgeWeights.clear();
				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++) meshObj->mesh.edgeWeights.push_back(1);

			}

			meshObj->mesh.edgeWeights[index] = wt;

			int symEdge = (index % 2 == 0) ? index + 1 : index - 1;

			meshObj->mesh.edgeWeights[symEdge] = wt;

		}

		/*! \brief This method sets edge weight of all the edges to the input weight.
		*
		*	\param		[in]	wt				- input weight.
		*	\since version 0.0.2
		*/
		void setEdgeWeight(double wt)
		{
			if (meshObj->mesh.edgeWeights.size() != meshObj->mesh.edgeActive.size())
			{
				meshObj->mesh.edgeWeights.clear();
				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++) meshObj->mesh.edgeWeights.push_back(1);

			}

			for (int i = 0; i < meshObj->mesh.edgeWeights.size(); i += 2)
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
			if (wt.size() != meshObj->mesh.edgeColors.size()) throw std::invalid_argument("size of wt contatiner is not equal to number of mesh half edges.");

			for (int i = 0; i < meshObj->mesh.edgeWeights.size(); i++)
			{
				meshObj->mesh.edgeWeights[i] = wt[i];
			}
		}

		
		//--------------------------
		//--- GET METHODS 
		//--------------------------

		/*! \brief This method gets vertex position at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector					- vertex position.
		*	\since version 0.0.2
		*/
		zVector getVertexPosition(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.vertexPositions[index];

		}

		/*! \brief This method gets pointer to the vertex position at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPosition(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.vertexPositions[index];

		}

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexPositions(vector<zVector>& pos)
		{
			pos = meshObj->mesh.vertexPositions;
		}

		/*! \brief This method gets pointer to the internal vertex positions container.
		*
		*	\return				zVector*					- pointer to internal vertex position container.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPositions()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.vertexPositions[0];			
		}

		/*! \brief This method gets vertex normal at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zVector				- vertex normal.
		*	\since version 0.0.2
		*/
		zVector getVertexNormal(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.vertexNormals[index];

		}

		/*! \brief This method gets pointer to the vertex normal at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zVector*			- pointer to internal vertex normal.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexNormal(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.vertexNormals[index];

		}

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	norm				- normals  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexNormals(vector<zVector>& norm)
		{
			norm = meshObj->mesh.vertexNormals;
		}

		/*! \brief This method gets pointer to the internal vertex normal container.
		*
		*	\return				zVector*					- pointer to internal vertex normal container.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexNormals()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.vertexNormals[0];
		}

		/*! \brief This method gets vertex color at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zColor					- vertex color.
		*	\since version 0.0.2
		*/
		zColor getVertexColor(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");
		
			return meshObj->mesh.vertexColors[index];

		}

		/*! \brief This method gets pointer to the vertex color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColor(int index)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.vertexColors[index];

		}

		/*! \brief This method gets vertex color of all the vertices.
		*
		*	\param		[out]	col				- color  contatiner. 
		*	\since version 0.0.2
		*/
		void getVertexColors(vector<zColor>& col)
		{
			col = meshObj->mesh.vertexColors;
		}

		/*! \brief This method gets pointer to the internal vertex color container.
		*
		*	\return				zColor*					- pointer to internal vertex color container.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColors()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.vertexColors[0];
		}

		/*! \brief This method gets edge color of the input edge.
		*
		*	\param		[in]	index					- input edge index.
		*	\return				zColor					- edge color.
		*	\since version 0.0.2
		*/
		zColor getEdgeColor(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.edgeColors[index];

		}

		/*! \brief This method gets pointer to the edge color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal edge color.
		*	\since version 0.0.2
		*/
		zColor* getRawEdgeColor(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.edgeColors[index];

		}

		/*! \brief This method gets edge color of all the edges.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getEdgeColors(vector<zColor>& col)
		{
			col = meshObj->mesh.edgeColors;
		}

		/*! \brief This method gets pointer to the internal edge color container.
		*
		*	\return				zColor*					- pointer to internal edge color container.
		*	\since version 0.0.2
		*/
		zColor* getRawEdgeColors()
		{
			if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.edgeColors[0];
		}

		/*! \brief This method gets face normal of the input face.
		*
		*	\param		[in]	index					- input face index.
		*	\return				zVector					- face normal.
		*	\since version 0.0.2
		*/
		zVector getFaceNormal(int index)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.faceNormals[index];

		}

		/*! \brief This method gets pointer to the face normal at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zVector*			- pointer to internal face normal.
		*	\since version 0.0.2
		*/
		zVector* getRawFaceNormal(int index)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.faceNormals[index];

		}

		/*! \brief This method gets face normals of all the faces.
		*
		*	\param		[out]	norm			- normals  contatiner.
		*	\since version 0.0.2
		*/
		void getFaceNormals(vector<zVector>& norm)
		{
			norm = meshObj->mesh.faceNormals;
		}

		/*! \brief This method gets pointer to the internal face normal container.
		*
		*	\return				zVector*					- pointer to internal face normal container.
		*	\since version 0.0.2
		*/
		zVector* getRawFaceNormals()
		{
			if (numPolygons() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.faceNormals[0];
		}

		/*! \brief This method gets face color of the input face.
		*
		*	\param		[in]	index					- input face index.
		*	\return				zColor					- face color.
		*	\since version 0.0.2
		*/
		zColor getFaceColor(int index)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return meshObj->mesh.faceColors[index];

		}

		/*! \brief This method gets pointer to the face color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal face color.
		*	\since version 0.0.2
		*/
		zColor* getRawFaceColor(int index)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			return &meshObj->mesh.faceColors[index];

		}

		/*! \brief This method gets face color of all the faces.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getFaceColors(vector<zColor>& col)
		{
			col = meshObj->mesh.faceColors;
		}

		/*! \brief This method gets pointer to the internal face color container.
		*
		*	\return				zColor*					- pointer to internal face color container.
		*	\since version 0.0.2
		*/
		zColor* getRawFaceColors()
		{
			if (numEdges() == 0) throw std::invalid_argument(" error: null pointer.");

			return &meshObj->mesh.faceColors[0];
		}

		/*! \brief This method computes the centers of a the input index edge or face of the mesh.
		*
		*	\param		[in]	type					- zEdgeData or zFaceData.
		*	\return				zVector					- center.
		*	\since version 0.0.2
		*/
		zVector getCenter( int index, zHEData type)
		{
			// Mesh Edge 
			if (type == zEdgeData)
			{
				if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				vector<int> eVerts;
				getVertices(index, zEdgeData, eVerts);

				return (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;
			}

			// Mesh Face 
			else if (type == zFaceData)
			{

				if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
				if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

				vector<int> fVerts;
				getVertices(index, zFaceData, fVerts);
				zVector cen;

				for (int j = 0; j < fVerts.size(); j++) cen += meshObj->mesh.vertexPositions[fVerts[j]];
				cen /= fVerts.size();

				return cen;

			}
			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method computes the center the mesh.
		*
		*	\return		zVector					- center .
		*	\since version 0.0.2
		*/
		zVector getCenter()
		{
			zVector out;

			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				out += meshObj->mesh.vertexPositions[i];
			}

			out /= meshObj->mesh.vertexPositions.size();
			
			return out;
			
		}

		/*! \brief This method computes the centers of a all edges or faces of the mesh.
		*
		*	\param		[in]	type					- zEdgeData or zFaceData.
		*	\param		[out]	centers					- vector of centers of type zVector.
		*	\since version 0.0.2
		*/
		void getCenters(zHEData type, vector<zVector> &centers)
		{
			// Mesh Edge 
			if (type == zEdgeData)
			{
				vector<zVector> edgeCenters;

				edgeCenters.clear();

				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
				{
					if (meshObj->mesh.edgeActive[i])
					{
						vector<int> eVerts;
						getVertices(i, zEdgeData, eVerts);

						zVector cen = (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;

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

			// Mesh Face 
			else if (type == zFaceData)
			{
				vector<zVector> faceCenters;
				faceCenters.clear();

				for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
				{
					if (meshObj->mesh.faceActive[i])
					{
						vector<int> fVerts;
						getVertices(i, zFaceData, fVerts);
						zVector cen;

						for (int j = 0; j < fVerts.size(); j++) cen += meshObj->mesh.vertexPositions[fVerts[j]];

						cen /= fVerts.size();
						faceCenters.push_back(cen);
					}
					else
					{
						faceCenters.push_back(zVector());
					}
				}

				centers = faceCenters;
			}
			else throw std::invalid_argument(" error: invalid zHEData type");
		}

		/*! \brief This method returns the dual mesh of the input mesh.
		*
		*	\param		[out]	dualMeshObj				- output dual mesh object.
		*	\param		[out]	inEdge_dualEdge			- container storing the corresponding dual mesh edge per input mesh edge.
		*	\param		[out]	dualEdge_inEdge			- container storing the corresponding input mesh edge per dual mesh edge.
		*	\param		[in]	excludeBoundary			- true if boundary faces are to be ignored.
		*	\param		[in]	rotate90				- true if dual mesh is to be roatated by 90. Generally used for planar mesh for 2D graphic statics application.
		*	\since version 0.0.2
		*/
		void getDualMesh(zObjMesh &dualMeshObj, vector<int> &inEdge_dualEdge, vector<int> &dualEdge_inEdge, bool excludeBoundary, bool keepExistingBoundary = false, bool rotate90 = false)
		{
			vector<zVector> positions;
			vector<int> polyConnects;
			vector<int> polyCounts;

			vector<zVector> fCenters;
			getCenters(zFaceData, fCenters);

			vector<zVector> eCenters;
			getCenters(zEdgeData, eCenters);

			positions = fCenters;

			// store map for input mesh edge to new vertex Id
			vector<int> inEdge_dualVertex;

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++)
			{
				inEdge_dualVertex.push_back(-1);


				if (!meshObj->mesh.edgeActive[i]) continue;

				if (onBoundary(i, zEdgeData))
				{
					if (!excludeBoundary)
					{
						inEdge_dualVertex[i] = positions.size();
						positions.push_back(eCenters[i]);
					}
				}
				else
				{
					inEdge_dualVertex[i] = meshObj->mesh.edges[i].getFace()->getFaceId();
				}
			}


			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				if (!meshObj->mesh.vertexActive[i]) continue;

				if (onBoundary(i, zVertexData))
				{
					if (excludeBoundary) continue;

					zEdge *e = meshObj->mesh.vertices[i].getEdge();

					zEdge *start = e;

					vector<int> tempPolyConnects;
					do
					{
						int eId = e->getEdgeId();
						int index = -1;
						bool checkRepeat = meshObj->mesh.coreUtils.checkRepeatElement(inEdge_dualVertex[eId], tempPolyConnects, index);
						if (!checkRepeat) tempPolyConnects.push_back(inEdge_dualVertex[eId]);

						if (keepExistingBoundary)
						{
							int vId = e->getVertex()->getVertexId();
							if (onBoundary(eId, zEdgeData) && vId == i)
							{
								//printf("\n working!");
								//tempPolyConnects.push_back(vId);
							}
						}


						e = e->getPrev();
						eId = e->getEdgeId();

						if (keepExistingBoundary)
						{
							int vId = e->getVertex()->getVertexId();
							if (onBoundary(eId, zEdgeData) && vId == i)
							{
								//printf("\n working2!");
								//tempPolyConnects.push_back(positions.size());
								//positions.push_back(meshObj->mesh.vertexPositions[vId]);

							}
						}

						index = -1;
						checkRepeat = meshObj->mesh.coreUtils.checkRepeatElement(inEdge_dualVertex[eId], tempPolyConnects, index);
						if (!checkRepeat) tempPolyConnects.push_back(inEdge_dualVertex[eId]);



						e = e->getSym();

					} while (e != start);


					for (int j = 0; j < tempPolyConnects.size(); j++)
					{
						polyConnects.push_back(tempPolyConnects[j]);
					}

					polyCounts.push_back(tempPolyConnects.size());
				}

				else
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					for (int j = 0; j < cEdges.size(); j++)
					{
						polyConnects.push_back(inEdge_dualVertex[cEdges[j]]);
					}

					polyCounts.push_back(cEdges.size());
				}



			}


			dualMeshObj.mesh = zMesh(positions, polyCounts, polyConnects);

			// rotate by 90

			if (rotate90)
			{
				zVector meshNorm = meshObj->mesh.vertexNormals[0];
				meshNorm.normalize();

				// bounding box
				zVector minBB, maxBB;
				meshObj->mesh.coreUtils.getBounds(dualMeshObj.mesh.vertexPositions, minBB, maxBB);

				zVector cen = (maxBB + minBB) * 0.5;

				for (int i = 0; i < dualMeshObj.mesh.vertexPositions.size(); i++)
				{
					dualMeshObj.mesh.vertexPositions[i] -= cen;
					dualMeshObj.mesh.vertexPositions[i] = dualMeshObj.mesh.vertexPositions[i].rotateAboutAxis(meshNorm, -90);
				}


			}

			// compute dualEdge_inEdge
			dualEdge_inEdge.clear();
			for (int i = 0; i < dualMeshObj.mesh.edgeActive.size(); i++)
			{
				dualEdge_inEdge.push_back(-1);
			}

			// compute inEdge to dualEdge	
			inEdge_dualEdge.clear();

			for (int i = 0; i < numEdges(); i++)
			{
				int v1 = inEdge_dualVertex[i];
				int v2 = (i % 2 == 0) ? inEdge_dualVertex[i + 1] : inEdge_dualVertex[i - 1];

				int eId = -1;
				dualMeshObj.mesh.edgeExists(v1, v2, eId);
				inEdge_dualEdge.push_back(eId);

				if (inEdge_dualEdge[i] != -1)
				{
					dualEdge_inEdge[inEdge_dualEdge[i]] = i;
				}
			}

		

		}


		/*! \brief This method returns the dual graph of the input mesh.
		*
		*	\param		[out]	dualGraphObj			- output dual graph object.
		*	\param		[out]	inEdge_dualEdge			- container storing the corresponding dual graph edge per input mesh edge.
		*	\param		[out]	dualEdge_inEdge			- container storing the corresponding input mesh edge per dual graph edge.
		*	\param		[in]	excludeBoundary			- true if boundary faces are to be ignored.
		*	\param		[in]	PlanarMesh				- true if input mesh is planar.
		*	\param		[in]	rotate90				- true if dual mesh is to be roatated by 90.
		*	\since version 0.0.2
		*/
		void getDualGraph(zObjGraph &dualGraphObj,vector<int> &inEdge_dualEdge, vector<int> &dualEdge_inEdge, bool excludeBoundary = false, bool PlanarMesh = false, bool rotate90 = false)
		{
			vector<zVector> positions;
			vector<int> edgeConnects;

			vector<zVector> fCenters;
			getCenters( zFaceData, fCenters);

			vector<zVector> eCenters;
			getCenters( zEdgeData, eCenters);
			
			positions = fCenters;

			// store map for input mesh edge to new vertex Id
			vector<int> inEdge_dualVertex;

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i++)
			{
				inEdge_dualVertex.push_back(-1);


				if (!meshObj->mesh.edgeActive[i]) continue;

				if (onBoundary(i, zEdgeData))
				{
					if (!excludeBoundary)
					{
						inEdge_dualVertex[i] = positions.size();
						positions.push_back(eCenters[i]);
					}
				}
				else
				{
					inEdge_dualVertex[i] = meshObj->mesh.edges[i].getFace()->getFaceId();
				}
			}

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				int v_0 = inEdge_dualVertex[i];
				int v_1 = inEdge_dualVertex[i + 1];

				if (v_0 != -1 && v_1 != -1)
				{
					edgeConnects.push_back(v_0);
					edgeConnects.push_back(v_1);
				}

			}

			if (PlanarMesh)
			{
				zVector graphNorm = meshObj->mesh.vertexNormals[0];
				graphNorm.normalize();

				zVector x(1, 0, 0);
				zVector sortRef = graphNorm ^ x;

				dualGraphObj.graph = zGraph(positions, edgeConnects, graphNorm, sortRef);
			}

			else dualGraphObj.graph = zGraph(positions, edgeConnects);

			// rotate by 90

			if (rotate90 && PlanarMesh)
			{
				zVector graphNorm = meshObj->mesh.vertexNormals[0];
				graphNorm.normalize();

				// bounding box
				zVector minBB, maxBB;
				meshObj->mesh.coreUtils.getBounds(dualGraphObj.graph.vertexPositions, minBB, maxBB);

				zVector cen = (maxBB + minBB) * 0.5;

				for (int i = 0; i < dualGraphObj.graph.vertexPositions.size(); i++)
				{
					dualGraphObj.graph.vertexPositions[i] -= cen;
					dualGraphObj.graph.vertexPositions[i] = dualGraphObj.graph.vertexPositions[i].rotateAboutAxis(graphNorm, -90);
				}


			}

			// compute dualEdge_inEdge
			dualEdge_inEdge.clear();
			for (int i = 0; i < dualGraphObj.graph.edgeActive.size(); i++)
			{
				dualEdge_inEdge.push_back(-1);
			}

			// compute inEdge to dualEdge	
			inEdge_dualEdge.clear();

			for (int i = 0; i < numEdges(); i++)
			{
				int v1 = inEdge_dualVertex[i];
				int v2 = (i % 2 == 0) ? inEdge_dualVertex[i + 1] : inEdge_dualVertex[i - 1];

				int eId = -1;
				dualGraphObj.graph.edgeExists(v1, v2, eId);
				inEdge_dualEdge.push_back(eId);

				if (inEdge_dualEdge[i] != -1)
				{
					dualEdge_inEdge[inEdge_dualEdge[i]] = i;
				}
			}

			
		}
		   	

		/*! \brief This method computes the input face triangulations using ear clipping algorithm.
		*
		*	\details based on  https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf & http://abitwise.blogspot.co.uk/2013/09/triangulating-concave-and-convex.html
		*	\param		[in]	faceIndex		- face index  of the face to be triangulated in the faces container.
		*	\param		[out]	numTris			- number of triangles in the input polygon.
		*	\param		[out]	tris			- index array of each triangle associated with the face.
		*	\since version 0.0.2
		*/
		void getFaceTriangles(int faceIndex, int &numTris, vector<int> &tris)
		{
			double angle_Max = 90;
			bool noEars = true; // check for if there are no ears

			vector<bool> ears;
			vector<bool> reflexVerts;

			// get face vertices

			vector<int> fVerts;
			getVertices(faceIndex, zFaceData, fVerts);
			vector<int> vertexIndices = fVerts;

			vector<zVector> points;
			for (int i = 0; i < fVerts.size(); i++)
			{
				points.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);
			}

			if (fVerts.size() < 3) throw std::invalid_argument(" error: invalid face, triangulation is not succesful.");

			// compute 
			zVector norm = meshObj->mesh.faceNormals[faceIndex];

			// compute ears

			for (int i = 0; i < vertexIndices.size(); i++)
			{
				int nextId = (i + 1) % vertexIndices.size();
				int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

				// Triangle edges - e1 and e2 defined above
				zVector v1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];
				zVector v2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];

				zVector cross = v1 ^ v2;
				double ang = v1.angle(v2);

				if (cross * norm < 0) ang *= -1;

				if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
				else reflexVerts.push_back(false);

				// calculate ears
				if (!reflexVerts[i])
				{
					bool ear = true;

					zVector p0 = meshObj->mesh.vertexPositions[fVerts[i]];
					zVector p1 = meshObj->mesh.vertexPositions[fVerts[nextId]];
					zVector p2 = meshObj->mesh.vertexPositions[fVerts[prevId]];

					bool CheckPtTri = false;

					for (int j = 0; j < fVerts.size(); j++)
					{
						if (!CheckPtTri)
						{
							if (j != i && j != nextId && j != prevId)
							{
								// vector to point to be checked
								zVector pt = meshObj->mesh.vertexPositions[fVerts[j]];

								bool Chk = meshObj->mesh.coreUtils.pointInTriangle(pt, p0, p1, p2);
								CheckPtTri = Chk;

							}
						}

					}

					if (CheckPtTri) ear = false;
					ears.push_back(ear);

					if (noEars && ear) noEars = !noEars;
				}
				else ears.push_back(false);

				//printf("\n id: %i ang: %1.2f reflex: %s ear: %s", vertexIndices[i], ang, (reflexVerts[i] == true) ? "true" : "false",(ears[i] == true)?"true":"false");
			}

			if (noEars)
			{
				for (int i = 0; i < fVerts.size(); i++)
				{
					//printf("\n %1.2f %1.2f %1.2f ", points[i].x, points[i].y, points[i].z);
				}

				throw std::invalid_argument(" error: no ears found in the face, triangulation is not succesful.");
			}

			int maxTris = fVerts.size() - 2;

			// // triangulate 

			while (numTris < maxTris - 1)
			{
				int earId = -1;
				bool earFound = false;;

				for (int i = 0; i < ears.size(); i++)
				{
					if (!earFound)
					{
						if (ears[i])
						{
							earId = i;
							earFound = !earFound;
						}
					}

				}

				if (earFound)
				{


					for (int i = -1; i <= 1; i++)
					{
						int id = (earId + i + vertexIndices.size()) % vertexIndices.size();
						tris.push_back(vertexIndices[id]);
					}
					numTris++;

					// remove vertex earid 
					vertexIndices.erase(vertexIndices.begin() + earId);

					reflexVerts.clear();
					ears.clear();

					// check for ears
					for (int i = 0; i < vertexIndices.size(); i++)
					{

						int nextId = (i + 1) % vertexIndices.size();
						int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

						// Triangle edges - e1 and e2 defined above
						zVector v1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];
						zVector v2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]] - meshObj->mesh.vertexPositions[vertexIndices[i]];

						zVector cross = v1 ^ v2;
						double ang = v1.angle(v2);

						if (cross * norm < 0) ang *= -1;

						if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
						else reflexVerts.push_back(false);

						// calculate ears
						if (!reflexVerts[i])
						{
							bool ear = true;

							zVector p0 = meshObj->mesh.vertexPositions[vertexIndices[i]];
							zVector p1 = meshObj->mesh.vertexPositions[vertexIndices[nextId]];
							zVector p2 = meshObj->mesh.vertexPositions[vertexIndices[prevId]];

							bool CheckPtTri = false;

							for (int j = 0; j < vertexIndices.size(); j++)
							{
								if (!CheckPtTri)
								{
									if (j != i && j != nextId && j != prevId)
									{
										// vector to point to be checked
										zVector pt = meshObj->mesh.vertexPositions[vertexIndices[j]];

										bool Chk = meshObj->mesh.coreUtils.pointInTriangle(pt, p0, p1, p2);
										CheckPtTri = Chk;
									}
								}

							}

							if (CheckPtTri) ear = false;
							ears.push_back(ear);

						}
						else ears.push_back(false);


						//printf("\n earId %i id: %i ang: %1.2f reflex: %s ear: %s", earId, vertexIndices[i], ang, (reflexVerts[i] == true) ? "true" : "false", (ears[i] == true) ? "true" : "false");
					}



				}
				else
				{
					for (int i = 0; i < vertexIndices.size(); i++)
					{
						//printf("\n %1.2f %1.2f %1.2f ", meshObj->mesh.vertexPositions[vertexIndices[i]].x, meshObj->mesh.vertexPositions[vertexIndices[i]].y, meshObj->mesh.vertexPositions[vertexIndices[i]].z);
					}

					throw std::invalid_argument(" error: no ears found in the face, triangulation is not succesful.");
				}

			}

			// add the last remaining triangle
			tris.push_back(vertexIndices[0]);
			tris.push_back(vertexIndices[1]);
			tris.push_back(vertexIndices[2]);
			numTris++;

		}

		/*! \brief This method computes the triangles of each face of the input mesh and stored in 2 dimensional container.
		*
		*	\param		[out]	faceTris		- container of index array of each triangle associated per face.
		*	\since version 0.0.2
		*/
		void getMeshTriangles(vector<vector<int>> &faceTris)
		{
			if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size()) computeMeshNormals();

			faceTris.clear();

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				vector<int> Tri_connects;

				if (meshObj->mesh.faceActive[i])
				{

					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					// compute polygon Triangles


					int n_Tris = 0;
					if (fVerts.size() > 0) getFaceTriangles( i, n_Tris, Tri_connects);
					else Tri_connects = fVerts;
				}


				faceTris.push_back(Tri_connects);
			}

		}

		/*! \brief This method computes the volume of the input mesh.
		*
		*	\return				double			- volume of input mesh.
		*	\since version 0.0.2
		*/
		double getMeshVolume()
		{
			double out;

			vector<vector<int>> faceTris;
			getMeshTriangles( faceTris);

			for (int i = 0; i < faceTris.size(); i++)
			{
				for (int j = 0; j < faceTris[i].size(); j += 3)
				{
					double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(meshObj->mesh.vertexPositions[faceTris[i][j + 0]], meshObj->mesh.vertexPositions[faceTris[i][j + 1]], meshObj->mesh.vertexPositions[faceTris[i][j + 2]]);

					out += vol;
				}


			}

			return out;
		}

		/*! \brief This method computes the volume of the polyhedras formed by the face vertices and the face center of the input indexed face of the mesh.
		*
		*	\param		[in]	index			- input face index.
		*	\param		[in]	faceTris		- container of index array of each triangle associated per face. It will be computed if the container is empty.
		*	\param		[in]	fCenters		- container of centers associated per face.  It will be computed if the container is empty.
		*	\param		[in]	absoluteVolumes	- will make all the volume value positive if true.
		*	\return				double			- volume of the polyhedras formed by the face vertices and the face center.
		*	\since version 0.0.2
		*/
		double getMeshFaceVolume( int index, vector<vector<int>> &faceTris, vector<zVector> &fCenters, bool absoluteVolume = true)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			if (faceTris.size() == 0 || faceTris.size() != meshObj->mesh.faceActive.size()) getMeshTriangles( faceTris);
			if (fCenters.size() == 0 || fCenters.size() != meshObj->mesh.faceActive.size()) getCenters( zFaceData, fCenters);

			double out = 0;

			zVector fCenter = fCenters[index];

			// add volume of face tris
			for (int j = 0; j < faceTris[index].size(); j += 3)
			{
				double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(meshObj->mesh.vertexPositions[faceTris[index][j + 0]], meshObj->mesh.vertexPositions[faceTris[index][j + 1]], meshObj->mesh.vertexPositions[faceTris[index][j + 2]]);

				out += vol;
			}

			// add volumes of tris formes by each pair of face edge vertices and face center

			vector<int> fVerts;
			getVertices(index, zFaceData, fVerts);

			for (int j = 0; j < fVerts.size(); j += 1)
			{
				int prevId = (j - 1 + fVerts.size()) % fVerts.size();

				double vol = meshObj->mesh.coreUtils.getSignedTriangleVolume(meshObj->mesh.vertexPositions[fVerts[j]], meshObj->mesh.vertexPositions[fVerts[prevId]], fCenter);

				out += vol;
			}

			if (absoluteVolume) out = abs(out);

			return out;

		}

		/*! \brief This method computes the volume of the polyhedras formed by the face vertices and the face center for each face of the mesh.
		*
		*	\param		[in]	faceTris		- container of index array of each triangle associated per face.  It will be computed if the container is empty.
		*	\param		[in]	fCenters		- container of centers associated per face.  It will be computed if the container is empty.
		*	\param		[out]	faceVolumes		- container of volumes of the polyhedras formed by the face vertices and the face center per face.
		*	\param		[in]	absoluteVolumes	- will make all the volume values positive if true.
		*	\since version 0.0.2
		*/
		void getMeshFaceVolumes(vector<vector<int>> &faceTris, vector<zVector> &fCenters, vector<double> &faceVolumes, bool absoluteVolumes = true)
		{
			if (faceTris.size() == 0 || faceTris.size() != meshObj->mesh.faceActive.size()) getMeshTriangles( faceTris);
			if (fCenters.size() == 0 || fCenters.size() != meshObj->mesh.faceActive.size()) getCenters( zFaceData, fCenters);

			faceVolumes.clear();

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				double vol = getMeshFaceVolume( i, faceTris, fCenters, absoluteVolumes);

				faceVolumes.push_back(vol);
			}
		}

		/*! \brief This method computes the local curvature of the mesh vertices.
		*
		*	\param		[out]	vertexCurvature		- container of vertex curvature.
		*	\since version 0.0.2
		*/
		void getPrincipalCurvature(vector<zCurvature> &vertexCurvatures)
		{
			for (int j = 0; j < numVertices(); j++)
			{

				if (meshObj->mesh.vertexActive[j])
				{
					double angleSum = 0;
					double cotangentSum = 0;
					double areaSum = 0;
					double areaSumMixed = 0;
					double edgeLengthSquare = 0;
					float gaussianCurv = 0;
					float gaussianAngle = 0;

					zCurvature curv;
					curv.k1 = 0;
					curv.k2 = 0;

					zVector meanCurvNormal;

					if (!onBoundary(j, zVertexData))
					{
						vector<int> connectedvertices;
						getConnectedVertices(j, zVertexData, connectedvertices);

						zVector pt = meshObj->mesh.vertexPositions[j];

						float multFactor = 0.125;
						for (int i = 0; i < connectedvertices.size(); i++)
						{
							int next = (i + 1) % connectedvertices.size();
							int prev = (i + connectedvertices.size() - 1) % connectedvertices.size();

							zVector pt1 = meshObj->mesh.vertexPositions[connectedvertices[i]];
							zVector pt2 = meshObj->mesh.vertexPositions[connectedvertices[next]];
							zVector pt3 = meshObj->mesh.vertexPositions[connectedvertices[prev]];

							zVector p01 = pt - pt1;
							zVector p02 = pt - pt2;
							zVector p10 = pt1 - pt;
							zVector p20 = pt2 - pt;
							zVector p12 = pt1 - pt2;
							zVector p21 = pt2 - pt1;
							zVector p31 = pt3 - pt1;
							
							zVector cr = (p10) ^ (p20);

							float ang = (p10).angle(p20);
							angleSum += ang;
							cotangentSum += (((p20)*(p10)) / cr.length());


							float e_Length = (pt1 - pt2).length();

							edgeLengthSquare += (e_Length * e_Length);

							zVector cr_alpha = (p01) ^ (p21);
							zVector cr_beta = (p01) ^ (p31);

							float coTan_alpha = (((p01)*(p21)) / cr_alpha.length());
							float coTan_beta = (((p01)*(p31)) / cr_beta.length());

							// check if triangle is obtuse
							if ((p10).angle(p20) <= 90 && (p01).angle(p21) <= 90 && (p12).angle(p02) <= 90)
							{
								areaSumMixed += (coTan_alpha + coTan_beta) * edgeLengthSquare * 0.125;
							}
							else
							{

								double triArea = (((p10) ^ (p20)).length()) / 2;

								if ((ang) <= 90) areaSumMixed += triArea * 0.25;
								else areaSumMixed += triArea * 0.5;

							}

							meanCurvNormal += ((pt - pt1)*(coTan_alpha + coTan_beta));
						}

						meanCurvNormal /= (2 * areaSumMixed);

						gaussianCurv = (360 - angleSum) / ((0.5 * areaSum) - (multFactor * cotangentSum * edgeLengthSquare));
						//outGauss.push_back(gaussianCurv);

						////// Based on Discrete Differential-Geometry Operators for Triangulated 2-Manifolds

						//gaussianCurv = (360 - angleSum) / areaSumMixed;

						double meanCurv = (meanCurvNormal.length() / 2);
						//if (meanCurv <0.001) meanCurv = 0;

						double deltaX = (meanCurv*meanCurv) - gaussianCurv;
						if (deltaX < 0) deltaX = 0;


						curv.k1 = meanCurv + sqrt(deltaX);
						curv.k2 = meanCurv - sqrt(deltaX);

					}

					vertexCurvatures.push_back(curv);
				}

				else
				{
					zCurvature curv;

					curv.k1 = -1;
					curv.k2 = -1;

					vertexCurvatures.push_back(curv);
				}
			}
		}

		/*! \brief This method computes the dihedral angle per edge of mesh.
		*
		*	\param		[out]	dihedralAngles		- vector of edge dihedralAngles.
		*	\since version 0.0.2
		*/
		void getEdgeDihedralAngles(vector<double> &dihedralAngles)
		{
			vector<double> out;

			if (meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size()) computeMeshNormals();

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				if (meshObj->mesh.edgeActive[i])
				{
					if (!onBoundary(i, zEdgeData) && !onBoundary(i + 1, zEdgeData))
					{
						// get connected face to edge
						vector<int> cFaces;
						getFaces(i, zEdgeData, cFaces);

						zVector n0 = meshObj->mesh.faceNormals[cFaces[0]];
						zVector n1 = meshObj->mesh.faceNormals[cFaces[1]];

						zVector e = meshObj->mesh.vertexPositions[meshObj->mesh.edges[i].getVertex()->getVertexId()] - meshObj->mesh.vertexPositions[meshObj->mesh.edges[i + 1].getVertex()->getVertexId()];

						double di_ang;
						di_ang = e.dihedralAngle(n0, n1);

						// per half edge
						out.push_back(di_ang);
						out.push_back(di_ang);

					}
					else
					{
						// per half edge
						out.push_back(-1);
						out.push_back(-1);
					}
				}
				else
				{
					// per half edge
					out.push_back(-2);
					out.push_back(-2);
				}


			}

			dihedralAngles = out;
		}

		/*! \brief This method computes the edge vector of the input edge of the mesh.
		*
		*	\param		[in]	index					- edge index.
		*	\return				zVector					- edge vector.
		*	\since version 0.0.2
		*/
		zVector getEdgeVector(int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			int v1 = meshObj->mesh.edges[index].getVertex()->getVertexId();
			int v2 = meshObj->mesh.edges[index].getSym()->getVertex()->getVertexId();

			zVector out = meshObj->mesh.vertexPositions[v1] - (meshObj->mesh.vertexPositions[v2]);

			return out;
		}
		
		/*! \brief This method computes the edge length of the input edge of the mesh.
		*
		*	\param		[out]	index			- edge index.
		*	\return				double			- edge length.
		*	\since version 0.0.2
		*/
		double getEdgelength( int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			int v1 = meshObj->mesh.edges[index].getVertex()->getVertexId();
			int v2 = meshObj->mesh.edges[index].getSym()->getVertex()->getVertexId();

			double out = meshObj->mesh.vertexPositions[v1].distanceTo(meshObj->mesh.vertexPositions[v2]);

			return out;
		}

		/*! \brief This method computes the lengths of all the edges of a the mesh.
		*
		*	\param		[out]	edgeLengths				- vector of edge lengths.
		*	\return				double					- total edge lengths.
		*	\since version 0.0.2
		*/
		double getEdgeLengths(vector<double> &edgeLengths)
		{
			double total = 0.0;

			vector<double> out;

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				if (meshObj->mesh.edgeActive[i])
				{
					int v1 = meshObj->mesh.edges[i].getVertex()->getVertexId();
					int v2 = meshObj->mesh.edges[i].getSym()->getVertex()->getVertexId();

					zVector e = meshObj->mesh.vertexPositions[v1] - meshObj->mesh.vertexPositions[v2];
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

		/*! \brief This method computes the edge length of the edge loop starting at the input edge of zMesh.
		*
		*	\param		[out]	index			- edge index.
		*	\return				double			- edge length.
		*	\since version 0.0.2
		*/
		double getEdgeLoopLength( int index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			bool exit = false;

			zEdge *e = &meshObj->mesh.edges[index];
			zEdge *start = &meshObj->mesh.edges[index];
			double out = 0;

			while (!exit)
			{
				out += getEdgelength(e->getEdgeId());

				int v = e->getVertex()->getVertexId();

				if (onBoundary(v, zVertexData)) exit = true;

				if (!exit) e = e->getNext()->getSym()->getNext();

				if (e == start) exit = true;
			}


			return out;
		}

		/*! \brief This method computes the area around every vertex of a mesh based on face centers.
		*
		*	\param		[in]	inMesh			- input mesh.
		*	\param		[in]	faceCenters		- vector of face centers of type zVector.
		*	\param		[in]	edgeCenters		- vector of edge centers of type zVector.
		*	\param		[out]	vertexAreas		- vector of vertex Areas.
		*	\return				double			- total area of the mesh.
		*	\since version 0.0.2
		*/
		double getVertexArea( vector<zVector> &faceCenters, vector<zVector> &edgeCenters, vector<double> &vertexAreas)
		{
			vector<double> out;

			double totalArea = 0;

			for (int i = 0; i < meshObj->mesh.vertexActive.size(); i++)
			{
				double vArea = 0;


				if (meshObj->mesh.vertexActive[i])
				{
					vector<int> cEdges;
					getConnectedEdges(i, zVertexData, cEdges);

					for (int j = 0; j < cEdges.size(); j++)
					{

						int currentEdge = cEdges[j];
						int nextEdge = cEdges[(j + 1) % cEdges.size()];

						if (!meshObj->mesh.edges[currentEdge].getFace() || !meshObj->mesh.edges[nextEdge].getSym()->getFace()) continue;

						if (meshObj->mesh.edges[currentEdge].getFace()->getFaceId() != meshObj->mesh.edges[nextEdge].getSym()->getFace()->getFaceId()) continue;

						zVector vPos = meshObj->mesh.vertexPositions[i];
						zVector fCen = faceCenters[meshObj->mesh.edges[currentEdge].getFace()->getFaceId()];
						zVector currentEdge_cen = edgeCenters[currentEdge];
						zVector nextEdge_cen = edgeCenters[nextEdge];

						double Area1 = meshObj->mesh.coreUtils.getTriangleArea(vPos, currentEdge_cen, fCen);
						vArea += (Area1);

						double Area2 = meshObj->mesh.coreUtils.getTriangleArea(vPos, nextEdge_cen, fCen);
						vArea += (Area2);

					}

				}


				out.push_back(vArea);

				totalArea += vArea;

			}

			//printf("\n totalArea : %1.4f ",  totalArea);

			vertexAreas = out;

			return totalArea;
		}

		/*! \brief This method computes the area of every face of the mesh. It works only for if the faces are planar.
		*
		*	\details	Based on http://geomalgorithms.com/a01-_area.html.
		*	\param		[in]	inMesh			- input mesh.
		*	\param		[out]	faceAreas		- vector of vertex Areas.
		*	\return				double			- total area of the mesh.
		*	\since version 0.0.2
		*/
		double getPlanarFaceAreas(vector<double> &faceAreas)
		{


			if (meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size()) computeMeshNormals();

			vector<double> out;

			double totalArea = 0;

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				double fArea = 0;

				if (meshObj->mesh.faceActive[i])
				{
					zVector fNorm = meshObj->mesh.faceNormals[i];

					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);

					for (int j = 0; j < fVerts.size(); j++)
					{
						zVector v1 = meshObj->mesh.vertexPositions[fVerts[j]];
						zVector v2 = meshObj->mesh.vertexPositions[fVerts[(j + 1) % fVerts.size()]];


						fArea += fNorm * (v1 ^ v2);
					}
				}

				fArea *= 0.5;

				out.push_back(fArea);

				totalArea += fArea;
			}

			faceAreas = out;

			return totalArea;
		}

		/*! \brief This method return the number of vertices in the face given by the input index.
		*
		*	\param		[in]	index			- index of the face.
		*	\return				int				- number of vertices in the face.
		*	\since version 0.0.2
		*/
		int getNumPolygonVertices(int index)
		{
			vector<int> fEdges;
			getEdges(index, zFaceData, fEdges);

			return fEdges.size();
		}

		/*! \brief This method stores mesh face connectivity information in the input containers
		*
		*	\param		[out]	polyConnects	- stores list of polygon connection with vertex ids for each face.
		*	\param		[out]	polyCounts		- stores number of vertices per polygon.
		*	\since version 0.0.2
		*/
		void getPolygonData(vector<int>(&polyConnects), vector<int>(&polyCounts))
		{
			polyConnects.clear();
			polyCounts.clear();

			for (int i = 0; i < meshObj->mesh.faceActive.size(); i++)
			{
				if (!meshObj->mesh.faceActive[i]) continue;

				vector<int> facevertices;
				getVertices(i, zFaceData, facevertices);

				polyCounts.push_back(facevertices.size());

				for (int j = 0; j < facevertices.size(); j++)
				{
					polyConnects.push_back(facevertices[j]);
				}
			}
		}

		/*! \brief This method stores mesh edge connectivity information in the input containers
		*
		*	\param		[out]	edgeConnects	- stores list of esdge connection with vertex ids for each edge.
		*	\since version 0.0.2
		*/
		void getEdgeData(vector<int> &edgeConnects)
		{
			edgeConnects.clear();

			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				edgeConnects.push_back(meshObj->mesh.edges[i + 1].getVertex()->getVertexId());
				edgeConnects.push_back(meshObj->mesh.edges[i].getVertex()->getVertexId());
			}
		}

		/*! \brief This method creates a duplicate of the mesh.
		*
		*	\return				zMesh			- duplicate mesh.
		*	\since version 0.0.2
		*/
		zObjMesh getDuplicate()
		{
			zObjMesh out;

			vector<zVector> positions;
			vector<int> polyConnects;
			vector<int> polyCounts;

			positions = meshObj->mesh.vertexPositions;
			getPolygonData(polyConnects, polyCounts);

			out.mesh = zMesh(positions, polyCounts, polyConnects);

			out.mesh.vertexColors = meshObj->mesh.vertexColors;
			out.mesh.edgeColors = meshObj->mesh.edgeColors;
			out.mesh.faceColors = meshObj->mesh.faceColors;

			return out;
		}

		

		//--------------------------
		//---- TRI-MESH MODIFIER METHODS
		//--------------------------
		

		/*! \brief This method triangulates the input face of the mesh.
		*
		*	\param		[in]	faceIndex		- face index  of the face to be triangulated in the faces container.
		*	\since version 0.0.2
		*/
		void faceTriangulate(int faceIndex)
		{

			if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size())computeMeshNormals();

			vector<int> fVerts;
			getVertices(faceIndex, zFaceData, fVerts);

			//printf("\n %i : nV %i ", i, fVerts.size());

			int numfaces_original = meshObj->mesh.faceActive.size();
			int numEdges_original = meshObj->mesh.edgeActive.size();

			if (fVerts.size() != 3)
			{
				// compute polygon Triangles
				int n_Tris = 0;
				vector<int> Tri_connects;
				getFaceTriangles( faceIndex, n_Tris, Tri_connects);

				//printf("\n %i numtris: %i %i ",i, n_Tris, Tri_connects.size());

				for (int j = 0; j < n_Tris; j++)
				{
					vector<int> triVerts;
					triVerts.push_back(Tri_connects[j * 3]);
					triVerts.push_back(Tri_connects[j * 3 + 1]);
					triVerts.push_back(Tri_connects[j * 3 + 2]);

					//printf("\n %i %i %i ", Tri_connects[j * 3], Tri_connects[j * 3 + 1], Tri_connects[j * 3 + 2]);

					// check if edges e01, e12 or e20
					int e01_ID, e12_ID, e20_ID;
					bool e01_Boundary = false;
					bool e12_Boundary = false;
					bool e20_Boundary = false;

					for (int k = 0; k < triVerts.size(); k++)
					{
						int e;
						bool eExists = meshObj->mesh.edgeExists(triVerts[k], triVerts[(k + 1) % triVerts.size()], e);


						if (k == 0)
						{

							if (eExists)
							{
								e01_ID = e;

								if (e01_ID < numEdges_original)
								{
									if (onBoundary(e, zEdgeData))
									{
										e01_Boundary = true;

									}
								}


							}
							else
							{
								meshObj->mesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e01_ID = meshObj->mesh.edgeActive.size() - 2;


							}
						}


						if (k == 1)
						{
							if (eExists)
							{
								e12_ID = e;

								if (e12_ID < numEdges_original)
								{
									if (onBoundary(e, zEdgeData))
									{
										e12_Boundary = true;
									}
								}

							}
							else
							{
								meshObj->mesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);

								e12_ID = meshObj->mesh.edgeActive.size() - 2;
							}
						}



						if (k == 2)
						{
							if (eExists)
							{

								e20_ID = e;

								if (e20_ID < numEdges_original)
								{
									if (onBoundary(e, zEdgeData))
									{
										e20_Boundary = true;

									}
								}

							}
							else
							{
								meshObj->mesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e20_ID = meshObj->mesh.edgeActive.size() - 2;
							}

						}



					}

					zEdge* e01 = &meshObj->mesh.edges[e01_ID];
					zEdge* e12 = &meshObj->mesh.edges[e12_ID];
					zEdge* e20 = &meshObj->mesh.edges[e20_ID];

					//printf("\n %i %i %i ", e01_ID, e12_ID, e20_ID);


					if (j > 0)
					{
						meshObj->mesh.addPolygon();
						meshObj->mesh.faces[meshObj->mesh.faceActive.size() - 1].setEdge(e01);

						if (!e01_Boundary) e01->setFace(&meshObj->mesh.faces[meshObj->mesh.faceActive.size() - 1]);
						if (!e12_Boundary) e12->setFace(&meshObj->mesh.faces[meshObj->mesh.faceActive.size() - 1]);
						if (!e20_Boundary) e20->setFace(&meshObj->mesh.faces[meshObj->mesh.faceActive.size() - 1]);
					}
					else
					{
						if (!e01_Boundary) meshObj->mesh.faces[faceIndex].setEdge(e01);
						else if (!e12_Boundary) meshObj->mesh.faces[faceIndex].setEdge(e12);
						else if (!e20_Boundary) meshObj->mesh.faces[faceIndex].setEdge(e20);


						if (!e01_Boundary) e01->setFace(&meshObj->mesh.faces[faceIndex]);
						if (!e12_Boundary) e12->setFace(&meshObj->mesh.faces[faceIndex]);
						if (!e20_Boundary) e20->setFace(&meshObj->mesh.faces[faceIndex]);
					}

					// update edge pointers
					e01->setNext(e12);
					e01->setPrev(e20);
					e12->setNext(e20);

				}
			}

		}

		/*! \brief This method triangulates the input mesh.
		*
		*	\since version 0.0.2
		*/
		void triangulate()
		{

			if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size()) computeMeshNormals();



			// iterate through faces and triangulate faces with more than 3 vetices
			int numfaces_original = meshObj->mesh.faceActive.size();

			int numEdges_original = meshObj->mesh.edgeActive.size();
			//printf("\n numfaces_before: %i ", numfaces_before);

			for (int i = 0; i < numfaces_original; i++)
			{
				if (!meshObj->mesh.faceActive[i]) continue;

				faceTriangulate(i);

			}


			computeMeshNormals();

		}

		//--------------------------
		//---- DELETE MODIFIER METHODS
		//--------------------------
	
		/*! \brief This method deletes the mesh vertex given by the input vertex index.
		*
		*	\param		[in]	index					- index of the vertex to be removed.
		*	\param		[in]	removeInactiveElems	- inactive elements in the list would be removed if true.
		*	\since version 0.0.2
		*/
		void deleteVertex(int index, bool removeInactiveElems = true)
		{
			if (index > meshObj->mesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			// check if boundary vertex
			bool boundaryVertex = (onBoundary(index, zVertexData));

			// get connected faces
			vector<int> cFaces;
			getConnectedFaces(index, zVertexData, cFaces);

			// get connected edges
			vector<int> cEdges;
			getConnectedEdges(index, zVertexData, cEdges);



			// get vertices in cyclical orders without the  the vertex to be removed. remove duplicates if any
			vector<int> outerVertices;

			vector<int> deactivateVertices;
			vector<int> deactivateEdges;

			deactivateVertices.push_back(index);

			// add to deactivate edges connected edges
			for (int i = 0; i < cEdges.size(); i++) deactivateEdges.push_back(cEdges[i]);

			// add to deactivate vertices with valence 2 and  connected edges of that vertex
			for (int i = 0; i < cEdges.size(); i++)
			{
				int v0 = meshObj->mesh.edges[cEdges[i]].getVertex()->getVertexId();

				if (!onBoundary(v0, zVertexData) && checkVertexValency(v0, 2))
				{
					deactivateVertices.push_back(v0);
					deactivateEdges.push_back(meshObj->mesh.edges[cEdges[i]].getNext()->getEdgeId());
				}
			}

			// compute new face vertices
			for (int i = 0; i < cEdges.size(); i++)
			{
				if (!meshObj->mesh.edges[cEdges[i]].getFace()) continue;

				zEdge *curEdge = &meshObj->mesh.edges[cEdges[i]];
				int v0 = curEdge->getVertex()->getVertexId();

				do
				{
					bool vertExists = false;

					for (int k = 0; k < outerVertices.size(); k++)
					{
						if (v0 == outerVertices[k])
						{
							vertExists = true;
							break;
						}
					}

					if (!vertExists)
					{
						for (int k = 0; k < deactivateVertices.size(); k++)
						{
							if (v0 == deactivateVertices[k])
							{
								vertExists = true;
								break;
							}
						}
					}

					if (!vertExists) outerVertices.push_back(v0);



					curEdge = curEdge->getNext();
					v0 = curEdge->getVertex()->getVertexId();


				} while (v0 != index);

			}


			// deactivate connected edges 
			for (int i = 0; i < deactivateEdges.size(); i++)
			{
				if (meshObj->mesh.edgeActive[deactivateEdges[i]])deactivateElement(deactivateEdges[i], zEdgeData);
			}

			// disable connected faces
			for (int i = 0; i < cFaces.size(); i++)
			{
				if (meshObj->mesh.faceActive[cFaces[i]]) deactivateElement(cFaces[i], zFaceData);
			}

			// deactivate vertex
			for (int i = 0; i < deactivateVertices.size(); i++)
			{
				if (meshObj->mesh.vertexActive[deactivateVertices[i]]) deactivateElement(deactivateVertices[i], zVertexData);
			}



			// add new face if outerVertices has more than 2 vertices

			if (outerVertices.size() > 2)
			{
				meshObj->mesh.addPolygon(outerVertices);

				if (boundaryVertex)  meshObj->mesh.update_BoundaryEdgePointers();
			}

			computeMeshNormals();

			if (removeInactiveElems)
			{
				removeInactiveElements(zVertexData);
				removeInactiveElements(zEdgeData);
				removeInactiveElements(zFaceData);
			}
		}


		/*! \brief This method deletes the mesh face given by the input face index.
		*
		*	\param		[in]	index					- index of the face to be removed.
		*	\param		[in]	removeInactiveElems	- inactive elements in the list would be removed if true.
		*	\since version 0.0.2
		*/
		void deleteFace(int index, bool removeInactiveElems = true)
		{
			if (index > meshObj->mesh.faceActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.faceActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			// check if there is only 1 polygon. If true, cant perform collapse.
			if (numPolygons() == 1)
			{
				printf("\n Can't delete on single face mesh.");
				return;
			}

			// get faces vertices
			vector<int> fVerts;
			getVertices(index, zFaceData, fVerts);



			// get face edges.
			vector<int> fEdges;
			getEdges(index, zFaceData, fEdges);

			// connected edge for each face vertex
			vector<int> fVertsValence;
			for (int i = 0; i < fVerts.size(); i++)
			{

				vector<int> cEdges;
				getConnectedEdges(fVerts[i], zVertexData, cEdges);
				fVertsValence.push_back(cEdges.size());

				// update vertex edge pointer if ther are pointing to face edges , as they will be disabled.

				for (int j = 0; j < cEdges.size(); j++)
				{
					bool chk = false;

					for (int k = 0; k < fEdges.size(); k++)
					{
						int sEdge = meshObj->mesh.edges[fEdges[k]].getSym()->getEdgeId();

						if (cEdges[j] == fEdges[k] || cEdges[j] == sEdge)
						{
							chk = true;
							break;
						}
					}

					if (!chk)
					{
						meshObj->mesh.vertices[fVerts[i]].setEdge(&meshObj->mesh.edges[cEdges[j]]);
						break;
					}
				}

			}

			// make face edges as  boundary edges, and disable them if both half edges have null face pointers.
			for (int i = 0; i < fEdges.size(); i++)
			{
				meshObj->mesh.edges[fEdges[i]].setFace(nullptr);

				int symEdge = meshObj->mesh.edges[fEdges[i]].getSym()->getEdgeId();

				if (onBoundary(fEdges[i], zEdgeData) && onBoundary(symEdge, zEdgeData))
				{
					deactivateElement(fEdges[i], zEdgeData);
				}
			}

			// get face vertices and deactivate them if all connected half edges are in active.
			for (int i = 0; i < fVerts.size(); i++)
			{
				bool removeVertex = true;
				if (fVertsValence[i] > 2) removeVertex = false;


				if (removeVertex)
				{
					deactivateElement(fVerts[i], zVertexData);
				}

			}


			// deactivate face
			deactivateElement(index, zFaceData);


			if (removeInactiveElems)
			{
				removeInactiveElements(zVertexData);
				removeInactiveElements(zEdgeData);
				removeInactiveElements(zFaceData);
			}

		}

		/*! \brief This method deletes the mesh edge given by the input face index.
		*
		*	\param		[in]	index					- index of the edge to be removed.
		*	\param		[in]	removeInactiveElements	- inactive elements in the list would be removed if true.
		*	\since version 0.0.2
		*/
		void deleteEdge( int index, bool removeInactiveElements = true);

		//--------------------------
		//---- TOPOLOGY MODIFIER METHODS
		//--------------------------

		/*! \brief This method collapses an edge into a vertex.
		*
		*	\param		[in]	index					- index of the edge to be collapsed.
		*	\param		[in]	edgeFactor				- position factor of the remaining vertex after collapse on the original egde. Needs to be between 0.0 and 1.0.
		*	\param		[in]	removeInactiveElems	- inactive elements in the list would be removed if true.
		*	\since version 0.0.2
		*/
		void collapseEdge( int index, double edgeFactor = 0.5, bool removeInactiveElems = true)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			int nFVerts = (onBoundary(index, zEdgeData)) ? 0 : getNumPolygonVertices( meshObj->mesh.edges[index].getFace()->getFaceId());

			int sEdge = meshObj->mesh.edges[index].getSym()->getEdgeId();
			int nFVerts_Sym = (onBoundary(sEdge, zEdgeData)) ? 0 : getNumPolygonVertices( meshObj->mesh.edges[sEdge].getFace()->getFaceId());

			// check if there is only 1 polygon and its a triangle. If true, cant perform collapse.
			if (numPolygons() == 1)
			{
				if (nFVerts == 3 || nFVerts_Sym == 3)
				{
					printf("\n Can't perform collapse on single trianglular face.");
					return;
				}

			}

			// get edge faces
			vector<int> eFaces;
			getFaces(index, zEdgeData, eFaces);

			if (numPolygons() == eFaces.size())
			{
				if (nFVerts == nFVerts_Sym && nFVerts_Sym == 3)
				{
					printf("\n Can't perform collapse on common edge of 2 triangular face mesh.");
					return;
				}

			}

			int v1 = meshObj->mesh.edges[index].getVertex()->getVertexId();
			int v2 = meshObj->mesh.edges[sEdge].getVertex()->getVertexId();

			int vertexRemoveID = v1;
			int vertexRetainID = v2;

			if (getVertexValence(v1) > getVertexValence(v2))
			{
				vertexRemoveID = v2;
				vertexRetainID = v1;

				edgeFactor = 1 - edgeFactor;

			}

			// set new position of retained vertex
			zVector e = meshObj->mesh.vertexPositions[vertexRemoveID] - meshObj->mesh.vertexPositions[vertexRetainID];
			double eLength = e.length();
			e.normalize();

			meshObj->mesh.vertexPositions[vertexRetainID] = meshObj->mesh.vertexPositions[vertexRetainID] + e * (edgeFactor * eLength);


			// get connected edges of vertexRemoveID
			vector<int> cEdges;
			getConnectedEdges(vertexRemoveID, zVertexData, cEdges);


			// get connected edges

			int vNext = meshObj->mesh.edges[index].getNext()->getVertex()->getVertexId();
			vector<int> cEdgesVNext;
			getConnectedEdges(vNext, zVertexData, cEdgesVNext);

			int vPrev = meshObj->mesh.edges[index].getPrev()->getVertex()->getVertexId();
			vector<int> cEdgesVPrev;
			getConnectedEdges(vPrev, zVertexData, cEdgesVPrev);

			int vNext_sEdge = meshObj->mesh.edges[sEdge].getNext()->getVertex()->getVertexId();
			vector<int> cEdgesVNext_sEdge;
			getConnectedEdges(vNext_sEdge, zVertexData, cEdgesVNext_sEdge);

			int vPrev_sEdge = meshObj->mesh.edges[sEdge].getPrev()->getVertex()->getVertexId();
			vector<int> cEdgesVPrev_sEdge;
			getConnectedEdges(vPrev_sEdge, zVertexData, cEdgesVPrev_sEdge);

			// current edge 
			if (nFVerts == 3)
			{

				// update pointers
				meshObj->mesh.edges[index].getNext()->setNext(meshObj->mesh.edges[index].getPrev()->getSym()->getNext());
				meshObj->mesh.edges[index].getNext()->setPrev(meshObj->mesh.edges[index].getPrev()->getSym()->getPrev());

				meshObj->mesh.edges[index].getPrev()->setPrev(nullptr);
				meshObj->mesh.edges[index].getPrev()->setNext(nullptr);

				meshObj->mesh.edges[index].getPrev()->getSym()->setPrev(nullptr);
				meshObj->mesh.edges[index].getPrev()->getSym()->setNext(nullptr);

				meshObj->mesh.edges[index].getNext()->setFace(meshObj->mesh.edges[index].getPrev()->getSym()->getFace());

				if (meshObj->mesh.edges[index].getPrev()->getSym()->getFace())
				{
					meshObj->mesh.edges[index].getPrev()->getSym()->getFace()->setEdge(meshObj->mesh.edges[index].getNext());
					meshObj->mesh.edges[index].getPrev()->getSym()->setFace(nullptr);
				}

				// update vertex edge pointer if pointing to prev edge

				if (meshObj->mesh.vertices[vNext].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getPrev()->getEdgeId())
				{
					for (int i = 0; i < cEdgesVNext.size(); i++)
					{
						if (cEdgesVNext[i] != meshObj->mesh.edges[index].getPrev()->getEdgeId())
						{
							meshObj->mesh.vertices[vNext].setEdge(&meshObj->mesh.edges[cEdgesVNext[i]]);
						}
					}
				}

				// update vertex edge pointer if pointing to prev edge

				if (meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getPrev()->getSym()->getEdgeId() || meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == index)
				{
					for (int i = 0; i < cEdgesVPrev.size(); i++)
					{
						if (cEdgesVPrev[i] != meshObj->mesh.edges[index].getPrev()->getSym()->getEdgeId() && cEdgesVPrev[i] != index)
						{
							meshObj->mesh.vertices[vPrev].setEdge(&meshObj->mesh.edges[cEdgesVPrev[i]]);
						}
					}
				}

				// decativate prev edge
				deactivateElement(meshObj->mesh.edges[index].getPrev()->getEdgeId(), zEdgeData);

				// decativate next and sym pointer of the next edge are same, deactivate edge
				if (meshObj->mesh.edges[index].getNext()->getNext() == meshObj->mesh.edges[index].getNext()->getSym())
				{

					deactivateElement(meshObj->mesh.edges[index].getNext()->getEdgeId(), zEdgeData);
					deactivateElement(vNext, zVertexData);
				}

				// decativate prev and sym pointer of the next edge are same, deactivate edge
				else if (meshObj->mesh.edges[index].getNext()->getPrev() == meshObj->mesh.edges[index].getNext()->getSym())
				{
					deactivateElement(meshObj->mesh.edges[index].getNext()->getVertex()->getVertexId(), zVertexData);
					deactivateElement(vNext, zVertexData);
				}

				// deactivate face pointed by collapse edge
				deactivateElement(meshObj->mesh.edges[index].getFace()->getFaceId(), zFaceData);

				meshObj->mesh.edges[index].setFace(nullptr);

				meshObj->mesh.edges[index].setNext(nullptr);
				meshObj->mesh.edges[index].setPrev(nullptr);

			}
			else
			{
				// update vertex edge pointer if pointing to current edge
				if (meshObj->mesh.vertices[vPrev].getEdge()->getEdgeId() == meshObj->mesh.edges[index].getEdgeId())
				{
					meshObj->mesh.vertices[vPrev].setEdge(meshObj->mesh.edges[index].getPrev()->getSym());
				}

				// update pointers
				meshObj->mesh.edges[index].getNext()->setPrev(meshObj->mesh.edges[index].getPrev());

				meshObj->mesh.edges[index].setNext(nullptr);
				meshObj->mesh.edges[index].setPrev(nullptr);
			}

			// symmetry edge 
			if (nFVerts_Sym == 3)
			{


				// update pointers
				meshObj->mesh.edges[sEdge].getNext()->setNext(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getNext());
				meshObj->mesh.edges[sEdge].getNext()->setPrev(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getPrev());

				meshObj->mesh.edges[sEdge].getPrev()->setPrev(nullptr);
				meshObj->mesh.edges[sEdge].getPrev()->setNext(nullptr);

				meshObj->mesh.edges[sEdge].getPrev()->getSym()->setPrev(nullptr);
				meshObj->mesh.edges[sEdge].getPrev()->getSym()->setNext(nullptr);

				meshObj->mesh.edges[sEdge].getNext()->setFace(meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace());

				if (meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace())
				{
					meshObj->mesh.edges[sEdge].getPrev()->getSym()->getFace()->setEdge(meshObj->mesh.edges[sEdge].getNext());
					meshObj->mesh.edges[sEdge].getPrev()->getSym()->setFace(nullptr);
				}

				// update vertex edge pointer if pointing to prev edge

				if (meshObj->mesh.vertices[vNext_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getPrev()->getEdgeId())
				{
					for (int i = 0; i < cEdgesVNext_sEdge.size(); i++)
					{
						if (cEdgesVNext_sEdge[i] != meshObj->mesh.edges[sEdge].getPrev()->getEdgeId())
						{
							meshObj->mesh.vertices[vNext_sEdge].setEdge(&meshObj->mesh.edges[cEdgesVNext_sEdge[i]]);
						}
					}
				}

				// update vertex edge pointer if pointing to prev edge

				if (meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getPrev()->getSym()->getEdgeId() || meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == sEdge)
				{
					for (int i = 0; i < cEdgesVPrev_sEdge.size(); i++)
					{
						if (cEdgesVPrev_sEdge[i] != meshObj->mesh.edges[sEdge].getPrev()->getSym()->getEdgeId() && cEdgesVPrev_sEdge[i] != sEdge)
						{
							meshObj->mesh.vertices[vPrev_sEdge].setEdge(&meshObj->mesh.edges[cEdgesVPrev_sEdge[i]]);
						}
					}
				}

				// decativate prev edge
				deactivateElement(meshObj->mesh.edges[sEdge].getPrev()->getEdgeId(), zEdgeData);

				// decativate next and sym pointer of the next edge are same, deactivate edge
				if (meshObj->mesh.edges[sEdge].getNext()->getNext() == meshObj->mesh.edges[sEdge].getNext()->getSym())
				{
					deactivateElement(meshObj->mesh.edges[sEdge].getNext()->getEdgeId(), zEdgeData);
					deactivateElement(vNext_sEdge, zVertexData);
				}

				// decativate prev and sym pointer of the next edge are same, deactivate edge
				else if (meshObj->mesh.edges[sEdge].getNext()->getPrev() == meshObj->mesh.edges[sEdge].getNext()->getSym())
				{
					deactivateElement(meshObj->mesh.edges[sEdge].getNext()->getEdgeId(), zEdgeData);
					deactivateElement(vNext_sEdge, zVertexData);
				}

				// deactivate face pointed by collapse edge
				deactivateElement(meshObj->mesh.edges[sEdge].getFace()->getFaceId(), zFaceData);

				meshObj->mesh.edges[sEdge].setFace(nullptr);

				meshObj->mesh.edges[sEdge].setNext(nullptr);
				meshObj->mesh.edges[sEdge].setPrev(nullptr);

			}
			else
			{
				// update vertex edge pointer if pointing to current edge
				if (meshObj->mesh.vertices[vPrev_sEdge].getEdge()->getEdgeId() == meshObj->mesh.edges[sEdge].getEdgeId())
				{
					meshObj->mesh.vertices[vPrev_sEdge].setEdge(meshObj->mesh.edges[sEdge].getPrev()->getSym());
				}

				// update pointers
				meshObj->mesh.edges[sEdge].getNext()->setPrev(meshObj->mesh.edges[sEdge].getPrev());

				meshObj->mesh.edges[sEdge].setNext(nullptr);
				meshObj->mesh.edges[sEdge].setPrev(nullptr);
			}

			// update connected edges verter pointer
			for (int i = 0; i < cEdges.size(); i++)
			{
				if (meshObj->mesh.edgeActive[cEdges[i]])
				{
					int v1 = meshObj->mesh.edges[cEdges[i]].getVertex()->getVertexId();
					int v2 = vertexRemoveID;
					meshObj->mesh.removeFromVerticesEdge(v1, v2);

					meshObj->mesh.edges[cEdges[i]].getSym()->setVertex(&meshObj->mesh.vertices[vertexRetainID]);

					meshObj->mesh.addToVerticesEdge(v1, vertexRetainID, cEdges[i]);
				}
			}


			// deactivate collapse edge
			if (meshObj->mesh.edgeActive[index])
			{
				deactivateElement(index, zEdgeData);
			}

			// deactivate vertexRemoveID
			if (meshObj->mesh.vertexActive[vertexRemoveID])
			{
				deactivateElement(vertexRemoveID, zVertexData);
			}

			// compute normals		
			computeMeshNormals();


			// remove inactive elements
			if (removeInactiveElems)
			{
				removeInactiveElements(zVertexData);
				removeInactiveElements(zEdgeData);
				removeInactiveElements(zFaceData);
			}

		}

		/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
		*
		*	\param		[in]	index			- index of the edge to be split.
		*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
		*	\param		[in]	fTriangulate	- true if contained edge faces needs to be triangulated after the edge spliting.
		*	\return				int				- index of the new vertex added after splitinng the edge.
		*	\since version 0.0.2
		*/
		int splitEdge(int index, double edgeFactor = 0.5, bool fTriangulate = false)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			vector<int> eFaces;
			getFaces(index, zEdgeData, eFaces);

			zEdge* edgetoSplit = &meshObj->mesh.edges[index];
			zEdge* edgetoSplitSym = edgetoSplit->getSym();

			zEdge* e_next = edgetoSplit->getNext();
			zEdge* e_prev = edgetoSplit->getPrev();

			zEdge* es_next = edgetoSplitSym->getNext();
			zEdge* es_prev = edgetoSplitSym->getPrev();


			zVector edgeDir = meshObj->mesh.vertexPositions[edgetoSplit->getVertex()->getVertexId()] - meshObj->mesh.vertexPositions[edgetoSplitSym->getVertex()->getVertexId()];
			double  edgeLength = edgeDir.length();
			edgeDir.normalize();

			zVector newVertPos = meshObj->mesh.vertexPositions[edgetoSplitSym->getVertex()->getVertexId()] + edgeDir * edgeFactor * edgeLength;



			// check if vertex exists if not add new vertex
			int VertId;
			bool vExists = vertexExists(newVertPos, VertId);
			if (!vExists)
			{
				meshObj->mesh.addVertex(newVertPos);
				VertId = meshObj->mesh.vertexActive.size() - 1;
			}

			//printf("\n newVert: %1.2f %1.2f %1.2f   %s ", newVertPos.x, newVertPos.y, newVertPos.z, (vExists)?"true":"false");

			if (!vExists)
			{
				// remove from verticesEdge map
				meshObj->mesh.removeFromVerticesEdge(edgetoSplit->getVertex()->getVertexId(), edgetoSplitSym->getVertex()->getVertexId());

				// add new edges
				int v1 = VertId;
				int v2 = edgetoSplit->getVertex()->getVertexId();
				bool edgesResize = meshObj->mesh.addEdges(v1, v2);

				// recompute pointers if resize is true
				if (edgesResize)
				{
					edgetoSplit = &meshObj->mesh.edges[index];
					edgetoSplitSym = edgetoSplit->getSym();

					e_next = edgetoSplit->getNext();
					e_prev = edgetoSplit->getPrev();

					es_next = edgetoSplitSym->getNext();
					es_prev = edgetoSplitSym->getPrev();

					//printf("\n working!");

				}

				// update vertex pointers
				meshObj->mesh.vertices[v1].setEdge(&meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 2]);
				meshObj->mesh.vertices[v2].setEdge(&meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 1]);

				//// update pointers
				edgetoSplit->setVertex(&meshObj->mesh.vertices[VertId]);			// current edge vertex pointer updated to new added vertex

				meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 1].setNext(edgetoSplitSym);		// new added edge next pointer to point to the next of current edge
				meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 1].setPrev(es_prev);

				if (edgetoSplitSym->getFace()) meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 1].setFace(edgetoSplitSym->getFace());

				meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 2].setPrev(edgetoSplit);
				meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 2].setNext(e_next);

				if (edgetoSplit->getFace()) meshObj->mesh.edges[meshObj->mesh.edgeActive.size() - 2].setFace(edgetoSplit->getFace());

				// update verticesEdge map
				meshObj->mesh.addToVerticesEdge(edgetoSplitSym->getVertex()->getVertexId(), edgetoSplit->getVertex()->getVertexId(), edgetoSplit->getEdgeId());

			}

			if (fTriangulate)
			{
				for (int i = 0; i < eFaces.size(); i++) faceTriangulate( eFaces[i]);
			}

			return VertId;
		}

		/*! \brief This method detaches an edge.
		*
		*	\param		[in]	index			- index of the edge to be split.
		*	\since version 0.0.2
		*/
		int detachEdge(int index);
	


		/*! \brief This method flips the edge shared bettwen two rainglua faces.
		*
		*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
		*	\since version 0.0.2
		*/
		void flipTriangleEdge(int &index)
		{
			if (index > meshObj->mesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
			if (!meshObj->mesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

			zEdge* edgetoFlip = &meshObj->mesh.edges[index];
			zEdge* edgetoFlipSym = edgetoFlip->getSym();

			if (!edgetoFlip->getFace() || !edgetoFlipSym->getFace())
			{
				throw std::invalid_argument("\n Cannot flip boundary edge %i ");
				return;
			}

			vector<int> edgetoFlip_fVerts;
			getVertices(edgetoFlip->getFace()->getFaceId(), zFaceData, edgetoFlip_fVerts);

			vector<int> edgetoFlipSym_fVerts;
			getVertices(edgetoFlipSym->getFace()->getFaceId(), zFaceData, edgetoFlipSym_fVerts);

			if (edgetoFlip_fVerts.size() != 3 || edgetoFlipSym_fVerts.size() != 3)
			{
				throw std::invalid_argument("\n Cannot flip edge not shared by two Triangles.");
				return;
			}

			zEdge* e_next = edgetoFlip->getNext();
			zEdge* e_prev = edgetoFlip->getPrev();

			zEdge* es_next = edgetoFlipSym->getNext();
			zEdge* es_prev = edgetoFlipSym->getPrev();

			// remove from verticesEdge map
			string removeHashKey = (to_string(edgetoFlip->getVertex()->getVertexId()) + "," + to_string(edgetoFlipSym->getVertex()->getVertexId()));
			meshObj->mesh.verticesEdge.erase(removeHashKey);

			string removeHashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
			meshObj->mesh.verticesEdge.erase(removeHashKey1);

			// update pointers

			if (edgetoFlip->getVertex()->getEdge() == edgetoFlipSym)edgetoFlip->getVertex()->setEdge(edgetoFlipSym->getPrev()->getSym());
			if (edgetoFlipSym->getVertex()->getEdge() == edgetoFlip) edgetoFlipSym->getVertex()->setEdge(edgetoFlip->getPrev()->getSym());

			edgetoFlip->setVertex(e_next->getVertex());
			edgetoFlipSym->setVertex(es_next->getVertex());



			edgetoFlip->setNext(e_prev);
			edgetoFlip->setPrev(es_next);

			edgetoFlipSym->setNext(es_prev);
			edgetoFlipSym->setPrev(e_next);

			e_prev->setNext(es_next);
			es_prev->setNext(e_next);

			edgetoFlip->getNext()->setFace(edgetoFlip->getFace());
			edgetoFlip->getPrev()->setFace(edgetoFlip->getFace());

			edgetoFlipSym->getNext()->setFace(edgetoFlipSym->getFace());
			edgetoFlipSym->getPrev()->setFace(edgetoFlipSym->getFace());

			edgetoFlip->getFace()->setEdge(edgetoFlip);
			edgetoFlipSym->getFace()->setEdge(edgetoFlipSym);

			// update verticesEdge map

			string hashKey = (to_string(edgetoFlip->getVertex()->getVertexId()) + "," + to_string(edgetoFlipSym->getVertex()->getVertexId()));
			meshObj->mesh.verticesEdge[hashKey] = edgetoFlipSym->getEdgeId();

			string hashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
			meshObj->mesh.verticesEdge[hashKey1] = edgetoFlip->getEdgeId();
		}


		/*! \brief This method splits a set of edges and faces of a mesh in a continuous manner.
		*
		*	\param		[in]	edgeList		- indicies of the edges to be split.
		*	\param		[in]	edgeFactor		- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
		*	\since version 0.0.2
		*/
		void splitFaces(vector<int> &edgeList, vector<double> &edgeFactor)
		{
			if (edgeFactor.size() > 0)
			{
				if (edgeList.size() != edgeFactor.size()) throw std::invalid_argument(" error: size of edgelist and edge factor dont match.");
			}

			int numOriginalVertices = meshObj->mesh.vertexActive.size();
			int numOriginalEdges = meshObj->mesh.edgeActive.size();
			int numOriginalFaces = meshObj->mesh.faceActive.size();

			for (int i = 0; i < edgeList.size(); i++)
			{
				if (edgeFactor.size() > 0) splitEdge( edgeList[i], edgeFactor[i]);
				else splitEdge( edgeList[i]);
			}

			for (int j = 0; j < edgeList.size(); j++)
			{
				for (int i = 0; i < 2; i++)
				{
					zEdge *start = (i == 0) ? &meshObj->mesh.edges[edgeList[j]] : meshObj->mesh.edges[edgeList[j]].getSym();

					zEdge *e = start;

					if (!start->getFace()) continue;

					bool exit = false;

					int v1 = start->getVertex()->getVertexId();
					int v2 = start->getVertex()->getVertexId();

					do
					{
						if (e->getNext())
						{
							e = e->getNext();
							if (e->getVertex()->getVertexId() > numOriginalVertices)
							{
								v2 = e->getVertex()->getVertexId();
								exit = true;
							}
						}
						else exit = true;

					} while (e != start && !exit);

					// add new edges and face
					if (v1 == v2) continue;

					// check if edge exists continue loop. 
					int outEdgeId;
					bool eExists = meshObj->mesh.edgeExists(v1, v2, outEdgeId);

					if (eExists) continue;

					int startEdgeId = start->getEdgeId();
					int e_EdgeId = e->getEdgeId();

					bool resizeEdges = meshObj->mesh.addEdges(v1, v2);

					if (resizeEdges)
					{
						start = &meshObj->mesh.edges[startEdgeId];
						e = &meshObj->mesh.edges[e_EdgeId];
					}

					meshObj->mesh.addPolygon(); // empty polygon

										 // update pointers
					zEdge *start_next = start->getNext();
					zEdge *e_next = e->getNext();

					start->setNext(&meshObj->mesh.edges[numEdges() - 2]);
					e_next->setPrev(&meshObj->mesh.edges[numEdges() - 2]);

					start_next->setPrev(&meshObj->mesh.edges[numEdges() - 1]);
					e->setNext(&meshObj->mesh.edges[numEdges() - 1]);

					meshObj->mesh.faces[numPolygons() - 1].setEdge(start_next);

					// edge face pointers to new face
					zEdge *newFace_E = start_next;

					do
					{
						newFace_E->setFace(&meshObj->mesh.faces[numPolygons() - 1]);

						if (newFace_E->getNext()) newFace_E = newFace_E->getNext();
						else exit = true;

					} while (newFace_E != start_next && !exit);

				}

			}


		}

		/*! \brief This method subdivides all the faces and edges of the mesh.
		*
		*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
		*	\since version 0.0.2
		*/
		void subdivideMesh(int numDivisions)
		{
			for (int j = 0; j < numDivisions; j++)
			{

				int numOriginalVertices = meshObj->mesh.vertexActive.size();

				// split edges at center
				int numOriginaledges = meshObj->mesh.edgeActive.size();

				for (int i = 0; i < numOriginaledges; i += 2)
				{
					if (meshObj->mesh.edgeActive[i]) splitEdge( i);
				}


				// get face centers
				vector<zVector> fCenters;
				getCenters( zFaceData, fCenters);


				// add faces
				int numOriginalfaces = meshObj->mesh.faceActive.size();

				for (int i = 0; i < numOriginalfaces; i++)
				{
					if (!meshObj->mesh.faceActive[i]) continue;

					vector<int> fEdges;
					getEdges(i, zFaceData, fEdges);

					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);


					// disable current face
					meshObj->mesh.faceActive[i] = false;

					int numCurrentEdges = meshObj->mesh.edgeActive.size();;

					// check if vertex exists if not add new vertex
					int VertId;
					bool vExists = vertexExists(fCenters[i], VertId);
					if (!vExists)
					{
						meshObj->mesh.addVertex(fCenters[i]);
						VertId = meshObj->mesh.vertexActive.size() - 1;
					}


					// add new faces				
					int startId = 0;
					if (meshObj->mesh.edges[fEdges[0]].getVertex()->getVertexId() < numOriginalVertices) startId = 1;

					for (int k = startId; k < fEdges.size() + startId; k += 2)
					{
						vector<int> newFVerts;

						int v1 = meshObj->mesh.edges[fEdges[k]].getVertex()->getVertexId();
						newFVerts.push_back(v1);

						int v2 = VertId; // face center
						newFVerts.push_back(v2);

						int v3 = meshObj->mesh.edges[fEdges[k]].getPrev()->getSym()->getVertex()->getVertexId();
						newFVerts.push_back(v3);

						int v4 = meshObj->mesh.edges[fEdges[k]].getPrev()->getVertex()->getVertexId();
						newFVerts.push_back(v4);

						meshObj->mesh.addPolygon(newFVerts);

					}

				}

				computeMeshNormals();

			}

		}

		/*! \brief This method applies Catmull-Clark subdivision to the mesh.
		*
		*	\details Based on https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface and https://rosettacode.org/wiki/Catmull%E2%80%93Clark_subdivision_surface.
		*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
		*	\param		[in]	smoothCorner	- corner vertex( only 2 Connected Edges) is also smothed if true.
		*	\since version 0.0.2
		*/
		void smoothMesh( int numDivisions = 1, bool smoothCorner = false)
		{
			for (int j = 0; j < numDivisions; j++)
			{


				// get face centers
				vector<zVector> fCenters;
				getCenters( zFaceData, fCenters);

				// get edge centers
				vector<zVector> tempECenters;
				vector<zVector> eCenters;
				getCenters( zEdgeData, eCenters);

				tempECenters = eCenters;

				// compute new smooth positions of the edge centers
				for (int i = 0; i < eCenters.size(); i += 2)
				{

					zVector newPos;

					if (onBoundary(i, zEdgeData) || onBoundary(i + 1, zEdgeData)) continue;

					int eId = i;

					vector<int> eVerts;
					getVertices(i, zEdgeData, eVerts);
					for (int j = 0; j < eVerts.size(); j++) newPos += meshObj->mesh.vertexPositions[eVerts[j]];


					vector<int> eFaces;
					getFaces(i, zEdgeData, eFaces);
					for (int j = 0; j < eFaces.size(); j++) newPos += fCenters[eFaces[j]];

					newPos /= (eFaces.size() + eVerts.size());

					eCenters[i] = newPos;
					eCenters[i + 1] = newPos;

				}

				// compute new smooth positions for the original vertices
				for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
				{
					if (onBoundary(i, zVertexData))
					{
						vector<int> cEdges;
						getConnectedEdges(i, zVertexData, cEdges);

						if (!smoothCorner && cEdges.size() == 2) continue;

						zVector P = meshObj->mesh.vertexPositions[i];
						int n = 1;

						zVector R;
						for (int j = 0; j < cEdges.size(); j++)
						{
							int symEdge = meshObj->mesh.edges[cEdges[j]].getSym()->getEdgeId();

							if (onBoundary(cEdges[j], zEdgeData) || onBoundary(symEdge, zEdgeData))
							{
								R += tempECenters[cEdges[j]];
								n++;
							}
						}

						meshObj->mesh.vertexPositions[i] = (P + R) / n;

					}
					else
					{
						zVector R;
						vector<int> cEdges;
						getConnectedEdges(i, zVertexData, cEdges);
						for (int j = 0; j < cEdges.size(); j++) R += tempECenters[cEdges[j]];
						R /= cEdges.size();

						zVector F;
						vector<int> cFaces;
						getConnectedFaces(i, zVertexData, cFaces);
						for (int j = 0; j < cFaces.size(); j++) F += fCenters[cFaces[j]];
						F /= cFaces.size();

						zVector P = meshObj->mesh.vertexPositions[i];
						int n = cFaces.size();

						meshObj->mesh.vertexPositions[i] = (F + (R * 2) + (P * (n - 3))) / n;
					}


				}


				int numOriginalVertices = meshObj->mesh.vertexActive.size();

				// split edges at center
				int numOriginaledges = meshObj->mesh.edgeActive.size();

				for (int i = 0; i < numOriginaledges; i += 2)
				{
					if (meshObj->mesh.edgeActive[i])
					{
						int newVert = splitEdge( i);

						meshObj->mesh.vertexPositions[newVert] = eCenters[i];
					}
				}



				// add faces
				int numOriginalfaces = meshObj->mesh.faceActive.size();

				for (int i = 0; i < numOriginalfaces; i++)
				{
					if (!meshObj->mesh.faceActive[i]) continue;

					vector<int> fEdges;
					getEdges(i, zFaceData, fEdges);

					vector<int> fVerts;
					getVertices(i, zFaceData, fVerts);


					//// disable current face

					meshObj->mesh.faceActive[i] = false;

					int numCurrentEdges = meshObj->mesh.edgeActive.size();;

					// check if vertex exists if not add new vertex
					int VertId;
					bool vExists = vertexExists(fCenters[i], VertId);
					if (!vExists)
					{
						meshObj->mesh.addVertex(fCenters[i]);
						VertId = meshObj->mesh.vertexActive.size() - 1;
					}


					// add new faces				
					int startId = 0;
					if (meshObj->mesh.edges[fEdges[0]].getVertex()->getVertexId() < numOriginalVertices) startId = 1;

					for (int k = startId; k < fEdges.size() + startId; k += 2)
					{
						vector<int> newFVerts;

						int v1 = meshObj->mesh.edges[fEdges[k]].getVertex()->getVertexId();
						newFVerts.push_back(v1);

						int v2 = VertId; // face center
						newFVerts.push_back(v2);

						int v3 = meshObj->mesh.edges[fEdges[k]].getPrev()->getSym()->getVertex()->getVertexId();
						newFVerts.push_back(v3);

						int v4 = meshObj->mesh.edges[fEdges[k]].getPrev()->getVertex()->getVertexId();
						newFVerts.push_back(v4);

						meshObj->mesh.addPolygon(newFVerts);


					}



				}


				//printMesh(inMesh);

				computeMeshNormals();

			}




		}

		/*! \brief This method returns an extruded mesh from the input mesh.
		*
		*	\param		[in]	extrudeThickness	- extrusion thickness.
		*	\param		[in]	thicknessTris		- true if the cap needs to be triangulated.
		*	\retrun				zMesh				- extruded mesh.
		*	\since version 0.0.2
		*/
		zObjMesh extrudeMesh(float extrudeThickness, bool thicknessTris = false)
		{
			if (meshObj->mesh.faceNormals.size() == 0 || meshObj->mesh.faceNormals.size() != meshObj->mesh.faceActive.size()) computeMeshNormals();

			zObjMesh out;

			vector<zVector> positions;
			vector<int> polyCounts;
			vector<int> polyConnects;



			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				positions.push_back(meshObj->mesh.vertexPositions[i]);
			}

			for (int i = 0; i < meshObj->mesh.vertexPositions.size(); i++)
			{
				positions.push_back(meshObj->mesh.vertexPositions[i] + (meshObj->mesh.vertexNormals[i] * extrudeThickness));
			}

			for (int i = 0; i < numPolygons(); i++)
			{
				vector<int> fVerts;
				getVertices(i, zFaceData, fVerts);

				for (int j = 0; j < fVerts.size(); j++)
				{
					polyConnects.push_back(fVerts[j]);
				}

				polyCounts.push_back(fVerts.size());

				for (int j = fVerts.size() - 1; j >= 0; j--)
				{
					polyConnects.push_back(fVerts[j] + meshObj->mesh.vertexPositions.size());
				}

				polyCounts.push_back(fVerts.size());

			}


			for (int i = 0; i < numEdges(); i++)
			{
				if (onBoundary(i, zEdgeData))
				{
					vector<int> eVerts;
					getVertices(i, zEdgeData, eVerts);

					if (thicknessTris)
					{
						polyConnects.push_back(eVerts[1]);
						polyConnects.push_back(eVerts[0]);
						polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());

						polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());
						polyConnects.push_back(eVerts[1] + meshObj->mesh.vertexPositions.size());
						polyConnects.push_back(eVerts[1]);

						polyCounts.push_back(3);
						polyCounts.push_back(3);
					}
					else
					{
						polyConnects.push_back(eVerts[1]);
						polyConnects.push_back(eVerts[0]);
						polyConnects.push_back(eVerts[0] + meshObj->mesh.vertexPositions.size());
						polyConnects.push_back(eVerts[1] + meshObj->mesh.vertexPositions.size());


						polyCounts.push_back(4);
					}


				}
			}

			out.mesh = zMesh(positions, polyCounts, polyConnects);

			return out;
		}
			

		/*! \brief This method returns the offset positions of a polygon of the input mesh.
		*
		*	\details	beased on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
		*	\param		[in]	faceIndex			- face index.
		*	\param		[in]	offset				- offset distance.
		*	\param		[out]	offsetPositions		- container with the offset positions.
		*	\since version 0.0.2
		*/
		void offsetMeshFace(int faceIndex, double offset, vector<zVector>& offsetPositions)
		{
			vector<zVector> out;

			vector<int> fVerts;
			getVertices(faceIndex, zFaceData, fVerts);

			for (int j = 0; j < fVerts.size(); j++)
			{
				int next = (j + 1) % fVerts.size();
				int prev = (j - 1 + fVerts.size()) % fVerts.size();


				zVector Ori = meshObj->mesh.vertexPositions[fVerts[j]];;
				zVector v1 = meshObj->mesh.vertexPositions[fVerts[prev]] - meshObj->mesh.vertexPositions[fVerts[j]];
				v1.normalize();

				zVector v2 = meshObj->mesh.vertexPositions[fVerts[next]] - meshObj->mesh.vertexPositions[fVerts[j]];
				v2.normalize();

				zVector v3 = v1;

				v1 = v1 ^ v2;
				v3 = v3 + v2;

				double cs = v3 * v2;

				zVector a1 = v2 * cs;
				zVector a2 = v3 - a1;

				double alpha = sqrt(a2.length() * a2.length());
				if (cs < 0) alpha *= -1;

				double length = offset / alpha;

				zVector mPos = meshObj->mesh.vertexPositions[fVerts[j]];
				zVector offPos = mPos + (v3 * length);

				out.push_back(offPos);

			}

			offsetPositions = out;

		}

		/*! \brief This method returns the vartiable offset positions of a polygon of the input mesh.
		*
		*	\param		[in]	inMesh					- input mesh.
		*	\param		[in]	faceIndex				- face index.
		*	\param		[in]	offsets					- offset distance from each edge of the mesh.
		*	\param		[in]	faceCenter				- center of polygon.
		*	\param		[in]	faceNormal				- normal of polygon.
		*	\param		[out]	intersectionPositions	- container with the intersection positions.
		*	\since version 0.0.2
		*/
		void offsetMeshFace_Variable( int faceIndex, vector<double>& offsets, zVector& faceCenter, zVector& faceNormal, vector<zVector>& intersectionPositions)
		{
			vector<zVector> offsetPoints;
			vector<int> fEdges;
			getEdges(faceIndex, zFaceData, fEdges);

			for (int j = 0; j < fEdges.size(); j++)
			{
				zVector p2 = meshObj->mesh.vertexPositions[meshObj->mesh.edges[fEdges[j]].getVertex()->getVertexId()];
				zVector p1 = meshObj->mesh.vertexPositions[meshObj->mesh.edges[fEdges[j]].getSym()->getVertex()->getVertexId()];

				zVector norm1 = ((p1 - p2) ^ faceNormal);
				norm1.normalize();
				if ((faceCenter - p1) * norm1 < 0) norm1 *= -1;


				offsetPoints.push_back(p1 + norm1 * offsets[j]);
				offsetPoints.push_back(p2 + norm1 * offsets[j]);

			}


			for (int j = 0; j < fEdges.size(); j++)
			{
				int prevId = (j - 1 + fEdges.size()) % fEdges.size();

				zVector a0 = offsetPoints[j * 2];
				zVector a1 = offsetPoints[j * 2 + 1];

				zVector b0 = offsetPoints[prevId * 2];
				zVector b1 = offsetPoints[prevId * 2 + 1];

				

				double uA = -1;
				double uB = -1;
				bool intersect = meshObj->mesh.coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

				if (intersect)
				{
					//printf("\n %i working!! ", j);

					zVector closestPt;

					if (uA >= uB)
					{
						zVector dir = a1 - a0;
						double len = dir.length();
						dir.normalize();

						if (uA < 0) dir *= -1;
						closestPt = a0 + dir * len * uA;
					}
					else
					{
						zVector dir = b1 - b0;
						double len = dir.length();
						dir.normalize();

						if (uB < 0) dir *= -1;

						closestPt = b0 + dir * len * uB;
					}


					intersectionPositions.push_back(closestPt);
				}

			}
		}

		

	};



}