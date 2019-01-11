#pragma once
#include <headers/geometry/zMesh.h>
#include <headers/geometry/zGraphMeshUtilities.h>
#include <headers/geometry/zMeshUtilities.h>

#include <headers/display/zPrintUtilities.h>

namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryUtilities
	*	\brief Collection of utility methods for graphs, meshes and fields.
	*  @{
	*/

	/** \addtogroup zMeshModifiers
	*	\brief Collection of mesh modifiers methods.
	*  @{
	*/



	//--------------------------
	//---- TRI-MESH MODIFIER METHODS
	//--------------------------


	/*! \brief This method triangulates the input polygon using ear clipping algorithm.
	*
	*	\details based on  https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf & http://abitwise.blogspot.co.uk/2013/09/triangulating-concave-and-convex.html
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	faceIndex		- face index  of the face to be triangulated in the faces container.
	*	\param		[out]	numTris			- number of triangles in the input polygon.
	*	\param		[out]	tris			- index array of each triangle associated with the face.
	*	\since version 0.0.1
	*/
	void polyTriangulate(zMesh &inMesh, int faceIndex, int &numTris, vector<int> &tris)
	{
		double angle_Max = 90;
		bool noEars = true; // check for if there are no ears

		vector<bool> ears;
		vector<bool> reflexVerts;

		// get face vertices

		vector<int> fVerts;
		inMesh.getVertices(faceIndex, zFaceData, fVerts);
		vector<int> vertexIndices = fVerts;

		vector<zVector> points;
		for (int i = 0; i < fVerts.size(); i++)
		{
			points.push_back(inMesh.vertexPositions[fVerts[i]]);
		}

		if (fVerts.size() < 3) throw std::invalid_argument(" error: invalid face, triangulation is not succesful.");

		// compute 
		zVector norm = inMesh.faceNormals[faceIndex];

		// compute ears

		for (int i = 0; i <vertexIndices.size(); i++)
		{
			int nextId = (i + 1) % vertexIndices.size();
			int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

			// Triangle edges - e1 and e2 defined above
			zVector v1 = inMesh.vertexPositions[vertexIndices[nextId]] - inMesh.vertexPositions[vertexIndices[i]];
			zVector v2 = inMesh.vertexPositions[vertexIndices[prevId]] - inMesh.vertexPositions[vertexIndices[i]];

			zVector cross = v1 ^ v2;
			double ang = v1.angle(v2);

			if (cross * norm < 0) ang *= -1;

			if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
			else reflexVerts.push_back(false);

			// calculate ears
			if (!reflexVerts[i])
			{
				bool ear = true;

				zVector p0 = inMesh.vertexPositions[fVerts[i]];
				zVector p1 = inMesh.vertexPositions[fVerts[nextId]];
				zVector p2 = inMesh.vertexPositions[fVerts[prevId]];

				bool CheckPtTri = false;

				for (int j = 0; j < fVerts.size(); j++)
				{
					if (!CheckPtTri)
					{
						if (j != i && j != nextId && j != prevId)
						{
							// vector to point to be checked
							zVector pt = inMesh.vertexPositions[fVerts[j]];

							bool Chk = pointInTriangle(pt, p0, p1, p2);
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
				printf("\n %1.2f %1.2f %1.2f ", points[i].x, points[i].y, points[i].z);
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
				for (int i = 0; i <vertexIndices.size(); i++)
				{

					int nextId = (i + 1) % vertexIndices.size();
					int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

					// Triangle edges - e1 and e2 defined above
					zVector v1 = inMesh.vertexPositions[vertexIndices[nextId]] - inMesh.vertexPositions[vertexIndices[i]];
					zVector v2 = inMesh.vertexPositions[vertexIndices[prevId]] - inMesh.vertexPositions[vertexIndices[i]];

					zVector cross = v1 ^ v2;
					double ang = v1.angle(v2);

					if (cross * norm < 0) ang *= -1;

					if (ang <= 0 || ang == 180) reflexVerts.push_back(true);
					else reflexVerts.push_back(false);

					// calculate ears
					if (!reflexVerts[i])
					{
						bool ear = true;

						zVector p0 = inMesh.vertexPositions[vertexIndices[i]];
						zVector p1 = inMesh.vertexPositions[vertexIndices[nextId]];
						zVector p2 = inMesh.vertexPositions[vertexIndices[prevId]];

						bool CheckPtTri = false;

						for (int j = 0; j < vertexIndices.size(); j++)
						{
							if (!CheckPtTri)
							{
								if (j != i && j != nextId && j != prevId)
								{
									// vector to point to be checked
									zVector pt = inMesh.vertexPositions[vertexIndices[j]];

									bool Chk = pointInTriangle(pt, p0, p1, p2);
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
					printf("\n %1.2f %1.2f %1.2f ", inMesh.vertexPositions[vertexIndices[i]].x, inMesh.vertexPositions[vertexIndices[i]].y, inMesh.vertexPositions[vertexIndices[i]].z);
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

	/*! \brief This method triangulates the input mesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	computeNormal	-	true if normals are not computed previously.
	*	\since version 0.0.1
	*/
	void triangulate(zMesh &inMesh)
	{

		if (inMesh.faceNormals.size() == 0 || inMesh.faceNormals.size() != inMesh.faceActive.size()) inMesh.computeMeshNormals();

		

		// iterate through faces and triangulate faces with more than 3 vetices
		int numfaces_original = inMesh.faceActive.size();

		int numEdges_original = inMesh.edgeActive.size();
		//printf("\n numfaces_before: %i ", numfaces_before);

		for (int i = 0; i < numfaces_original; i++)
		{
			if (!inMesh.faceActive[i]) continue;

			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			if (fVerts.size() != 3)
			{
				// compute polygon Triangles
				int n_Tris = 0;
				vector<int> Tri_connects;
				polyTriangulate(inMesh, i, n_Tris, Tri_connects);

				//printf("\n numtris: %i %i ", n_Tris, Tri_connects.size());

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
						bool eExists = inMesh.edgeExists(triVerts[k], triVerts[(k + 1) % triVerts.size()], e);


						if (k == 0)
						{

							if (eExists)
							{
								e01_ID = e;

								if (e01_ID < numEdges_original)
								{
									if (inMesh.onBoundary(e, zEdgeData))
									{
										e01_Boundary = true;

									}
								}


							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e01_ID = inMesh.edgeActive.size() - 2;


							}
						}


						if (k == 1)
						{
							if (eExists)
							{
								e12_ID = e;

								if (e12_ID < numEdges_original)
								{
									if (inMesh.onBoundary(e, zEdgeData))
									{
										e12_Boundary = true;
									}
								}

							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);

								e12_ID = inMesh.edgeActive.size() - 2;
							}
						}



						if (k == 2)
						{
							if (eExists)
							{

								e20_ID = e;

								if (e20_ID < numEdges_original)
								{
									if (inMesh.onBoundary(e, zEdgeData))
									{
										e20_Boundary = true;

									}
								}

							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e20_ID = inMesh.edgeActive.size() - 2;
							}

						}



					}

					zEdge* e01 = &inMesh.edges[e01_ID];
					zEdge* e12 = &inMesh.edges[e12_ID];
					zEdge* e20 = &inMesh.edges[e20_ID];

					//printf("\n %i %i %i ", e01_ID, e12_ID, e20_ID);
					

					if (j > 0)
					{
						inMesh.addPolygon();
						inMesh.faces[inMesh.faceActive.size() - 1].setEdge(e01);

						if (!e01_Boundary) e01->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
						if (!e12_Boundary) e12->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
						if (!e20_Boundary) e20->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
					}
					else
					{
						if (!e01_Boundary) inMesh.faces[i].setEdge(e01);
						else if (!e12_Boundary) inMesh.faces[i].setEdge(e12);
						else if (!e20_Boundary) inMesh.faces[i].setEdge(e20);


						if (!e01_Boundary) e01->setFace(&inMesh.faces[i]);
						if (!e12_Boundary) e12->setFace(&inMesh.faces[i]);
						if (!e20_Boundary) e20->setFace(&inMesh.faces[i]);
					}

					// update edge pointers
					e01->setNext(e12);
					e01->setPrev(e20);
					e12->setNext(e20);





				}
			}

		}


		inMesh.computeMeshNormals();

	}

	/*! \brief This method deletes the mesh vertex given by the input vertex index.
	*
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	index					- index of the vertex to be removed.
	*	\param		[in]	removeInactiveElements	- inactive elements in the list would be removed if true.
	*	\since version 0.0.1
	*/
	
	void deleteVertex(zMesh &inMesh, int index, bool removeInactiveElements = true)
	{
		if(index > inMesh.vertexActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		if (!inMesh.vertexActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		// check if boundary vertex
		bool boundaryVertex = (inMesh.onBoundary(index, zVertexData)) ;

		// get connected faces
		vector<int> cFaces;
		inMesh.getConnectedFaces(index, zVertexData, cFaces);

		// get connected edges
		vector<int> cEdges;
		inMesh.getConnectedEdges(index, zVertexData, cEdges);



		// get vertices in cyclical orders without the  the vertex to be removed. remove duplicates if any
		vector<int> outerVertices;

		for (int i = 0; i < cEdges.size();i++)
		{
			if (!inMesh.edges[cEdges[i]].getFace()) continue;

			zEdge *curEdge = &inMesh.edges[cEdges[i]];
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

				if (!vertExists) outerVertices.push_back(v0);
				
				curEdge = curEdge->getNext();
				v0 = curEdge->getVertex()->getVertexId();


			} while (v0 != index);		

		}

		// disable connected edges 



		for (int i = 0; i < cEdges.size(); i++)
		{
			int symEdge = inMesh.edges[cEdges[i]].getSym()->getEdgeId();
			
			inMesh.removeFromVerticesEdge(inMesh.edges[cEdges[i]].getVertex()->getVertexId(), inMesh.edges[cEdges[i]].getSym()->getVertex()->getVertexId());
			
			inMesh.edges[cEdges[i]] = zEdge();
			inMesh.edges[symEdge] = zEdge();

			inMesh.edgeActive[cEdges[i]] = false;
			inMesh.edgeActive[symEdge] = false;

		}
				
		int newNumEdges = inMesh.numEdges() - (2* cEdges.size());
		inMesh.setNumEdges(newNumEdges, false);
		
		// disable connected faces
		for (int i = 0; i < cFaces.size(); i++)
		{
			inMesh.faces[cFaces[i]] = zFace();

			inMesh.faceActive[cFaces[i]] = false;
			
		}

		int newNumFaces = inMesh.numPolygons() - cFaces.size();
		inMesh.setNumPolygons(newNumFaces, false);

		// remove from vertexPosition map
		inMesh.removeFromPositionMap(inMesh.vertexPositions[index]);

		// disable indexed vertex

		inMesh.vertexActive[index] = false;

		inMesh.vertexPositions[index] = zVector(10000, 10000, 10000); // dummy position for VBO
		inMesh.vertexNormals[index] = zVector(0, 0, 1); // dummy normal for VBO
		inMesh.vertexColors[index] = zColor(1, 1, 1, 1); // dummy color for VBO

		

		int newNumVertices = inMesh.numVertices() - 1;
		inMesh.setNumVertices(newNumVertices, false);
	

		// add new face if outerVertices has more than 2 vertices
		
		if (outerVertices.size() > 2)
		{
			inMesh.addPolygon(outerVertices);

			if (boundaryVertex)  inMesh.update_BoundaryEdgePointers();
		}



		inMesh.computeMeshNormals();

		if (removeInactiveElements)
		{
			inMesh.removeInactiveElements(zVertexData);			

			inMesh.removeInactiveElements(zEdgeData);
			inMesh.removeInactiveElements(zFaceData);
		}
	}
		

	/*! \brief This method collapses an edge into a vertex.
	*
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	index					- index of the edge to be collapsed.
	*	\param		[in]	edgeFactor				- position factor of the remaining vertex after collapse on the original egde. Needs to be between 0.0 and 1.0.
	*	\param		[in]	removeInactiveElements	- inactive elements in the list would be removed if true.
	*	\since version 0.0.1
	*/
	
	void collapseEdge(zMesh &inMesh, int index, double edgeFactor = 0.0,  bool removeInactiveElements = true)
	{
		if (index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		if (!inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		int nFVerts = (inMesh.onBoundary(index, zEdgeData))? getNumPolyVerts(inMesh, inMesh.edges[index].getSym()->getFace()->getFaceId()) : getNumPolyVerts(inMesh, inMesh.edges[index].getFace()->getFaceId());

		// check if there is only 1 polygon and its a triangle. If true, cant perform collapse.
		if (inMesh.numPolygons() == 1 && nFVerts == 3)
		{
			printf("\n Can't perform collapse on single trianglular face.");
			return;
		}

		int vertexRemoveID = inMesh.edges[index].getVertex()->getVertexId();
		int vertexRetainID = inMesh.edges[index].getSym()->getVertex()->getVertexId();

		printf("\n vertexRemoveID: %i vertexRetainID: %i ", vertexRemoveID, vertexRetainID);

		// check if vertexRemoveID is a boundary vertex
		bool boundaryVertex = (inMesh.onBoundary(vertexRemoveID, zVertexData));

		// set new position of retained vertex
		zVector e = inMesh.vertexPositions[vertexRemoveID] - inMesh.vertexPositions[vertexRetainID];
		double eLength = e.length();		
		e.normalize();

		inMesh.vertexPositions[vertexRetainID] = inMesh.vertexPositions[vertexRetainID] + e * (edgeFactor * eLength);

		// get edge faces
		vector<int> eFaces;
		inMesh.getFaces(index, zEdgeData, eFaces);

		// get connected faces
		vector<int> cFaces;
		inMesh.getConnectedFaces(vertexRemoveID, zVertexData, cFaces);

		// get connected edges
		vector<int> cEdges;
		inMesh.getConnectedEdges(vertexRemoveID, zVertexData, cEdges);


		vector<int> tempPolyConnects;
		vector<int> tempPolyCounts;

		for (int i = 0; i < cFaces.size(); i++)
		{

			// check if it also a edge face
			bool eFace = false; 
			for (int j = 0; j < eFaces.size(); j++)
			{
				if (eFaces[j] == cFaces[i])
				{
					eFace = true;
					break;
				}
			}

			vector<int> fVerts;
			inMesh.getVertices(cFaces[i], zFaceData, fVerts);

		

			if (eFace)
			{
				for (int j = 0; j < fVerts.size(); j++)
				{
					if (fVerts[j] != vertexRemoveID) tempPolyConnects.push_back(fVerts[j]);
				}

				

				tempPolyCounts.push_back(fVerts.size() - 1);
			}

			else
			{
				for (int j = 0; j < fVerts.size(); j++)
				{
					if (fVerts[j] == vertexRemoveID)  tempPolyConnects.push_back(vertexRetainID);
					else tempPolyConnects.push_back(fVerts[j]);
				}

				tempPolyCounts.push_back(fVerts.size() );
			}	
			

		}


		// disable connected edges 

		for (int i = 0; i < cEdges.size(); i++)
		{
			int symEdge = inMesh.edges[cEdges[i]].getSym()->getEdgeId();

			inMesh.removeFromVerticesEdge(inMesh.edges[cEdges[i]].getVertex()->getVertexId(), inMesh.edges[cEdges[i]].getSym()->getVertex()->getVertexId());

			inMesh.edges[cEdges[i]] = zEdge();
			inMesh.edges[symEdge] = zEdge();

			inMesh.edgeActive[cEdges[i]] = false;
			inMesh.edgeActive[symEdge] = false;

		}

		int newNumEdges = inMesh.numEdges() - (2 * cEdges.size());
		inMesh.setNumEdges(newNumEdges, false);

		
		// disable connected faces
		for (int i = 0; i < cFaces.size(); i++)
		{
			inMesh.faces[cFaces[i]] = zFace();

			inMesh.faceActive[cFaces[i]] = false;

		}

		int newNumFaces = inMesh.numPolygons() - cFaces.size();
		inMesh.setNumPolygons(newNumFaces, false);

		
		// remove from vertexPosition map
		inMesh.removeFromPositionMap(inMesh.vertexPositions[vertexRemoveID]);

		// disable vertexRemoveID vertex

		inMesh.vertexActive[vertexRemoveID] = false;

		inMesh.vertexPositions[vertexRemoveID] = zVector(10000, 10000, 10000); // dummy position for VBO
		inMesh.vertexNormals[vertexRemoveID] = zVector(0, 0, 1); // dummy normal for VBO
		inMesh.vertexColors[vertexRemoveID] = zColor(1, 1, 1, 1); // dummy color for VBO



		int newNumVertices = inMesh.numVertices() - 1;
		inMesh.setNumVertices(newNumVertices, false);

			

		// create faces and edges connection
		int polyconnectsCurrentIndex = 0;

		for (int i = 0; i < tempPolyCounts.size(); i++)
		{
			int num_faceVerts = tempPolyCounts[i];

			vector<int> newfVerts;

			printf("\n newfVerts : ");
			for (int j = 0; j < num_faceVerts; j++)
			{
				newfVerts.push_back(tempPolyConnects[polyconnectsCurrentIndex + j]);	

				printf(" %i ", newfVerts[j]);
			}

			if(newfVerts.size() > 2) inMesh.addPolygon(newfVerts);
			polyconnectsCurrentIndex += num_faceVerts;

		}

		if (boundaryVertex)
		{
			printf("\n boundary working! ");

			printMesh(inMesh);

			inMesh.update_BoundaryEdgePointers();
		}

		// compute normals
		inMesh.computeMeshNormals();


		// remove inactive elements
		if (removeInactiveElements)
		{
			inMesh.removeInactiveElements(zVertexData);
			inMesh.removeInactiveElements(zEdgeData);
			inMesh.removeInactiveElements(zFaceData);
		}

	}



	/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	index			- index of the edge to be split.
	*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
	*	\return				int				- index of the new vertex added after splitinng the edge.
	*	\since version 0.0.1
	*/
	int splitEdge(zMesh &inMesh, int index, double edgeFactor = 0.5)
	{
		if( index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		if ( !inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");

		zEdge* edgetoSplit = &inMesh.edges[index];
		zEdge* edgetoSplitSym = edgetoSplit->getSym();

		zEdge* e_next = edgetoSplit->getNext();
		zEdge* e_prev = edgetoSplit->getPrev();

		zEdge* es_next = edgetoSplitSym->getNext();
		zEdge* es_prev = edgetoSplitSym->getPrev();


		zVector edgeDir = inMesh.vertexPositions[edgetoSplit->getVertex()->getVertexId()] - inMesh.vertexPositions[edgetoSplitSym->getVertex()->getVertexId()];
		double  edgeLength = edgeDir.length();
		edgeDir.normalize();

		zVector newVertPos = inMesh.vertexPositions[edgetoSplitSym->getVertex()->getVertexId()] + edgeDir * edgeFactor * edgeLength;

		

		// check if vertex exists if not add new vertex
		int VertId;
		bool vExists = inMesh.vertexExists(newVertPos, VertId);
		if (!vExists)
		{
			inMesh.addVertex(newVertPos);
			VertId = inMesh.vertexActive.size() - 1;
		}

		//printf("\n newVert: %1.2f %1.2f %1.2f   %s ", newVertPos.x, newVertPos.y, newVertPos.z, (vExists)?"true":"false");

		if (!vExists)
		{
			// remove from verticesEdge map
			string removeHashKey = (to_string(edgetoSplit->getVertex()->getVertexId()) + "," + to_string(edgetoSplitSym->getVertex()->getVertexId()));
			inMesh.verticesEdge.erase(removeHashKey);

			string removeHashKey1 = (to_string(edgetoSplitSym->getVertex()->getVertexId()) + "," + to_string(edgetoSplit->getVertex()->getVertexId()));
			inMesh.verticesEdge.erase(removeHashKey1);

			// add new edges
			int v1 = VertId;
			int v2 = edgetoSplit->getVertex()->getVertexId();
			bool edgesResize = inMesh.addEdges(v1, v2);

			// recompute pointers if resize is true
			if (edgesResize)
			{
				edgetoSplit = &inMesh.edges[index];
				edgetoSplitSym = edgetoSplit->getSym();

				e_next = edgetoSplit->getNext();
				e_prev = edgetoSplit->getPrev();

				es_next = edgetoSplitSym->getNext();
				es_prev = edgetoSplitSym->getPrev();

				//printf("\n working!");

			}

			// update vertex pointers
			inMesh.vertices[v1].setEdge(&inMesh.edges[inMesh.edgeActive.size() - 2]);
			inMesh.vertices[v2].setEdge(&inMesh.edges[inMesh.edgeActive.size() - 1]);

			//// update pointers
			edgetoSplit->setVertex(&inMesh.vertices[VertId]);			// current edge vertex pointer updated to new added vertex

			inMesh.edges[inMesh.edgeActive.size() - 1].setNext(edgetoSplitSym);		// new added edge next pointer to point to the next of current edge
			inMesh.edges[inMesh.edgeActive.size() - 1].setPrev(es_prev);
			
			if(edgetoSplitSym->getFace()) inMesh.edges[inMesh.edgeActive.size() - 1].setFace(edgetoSplitSym->getFace());

			inMesh.edges[inMesh.edgeActive.size() - 2].setPrev(edgetoSplit);
			inMesh.edges[inMesh.edgeActive.size() - 2].setNext(e_next);
			
			if (edgetoSplit->getFace()) inMesh.edges[inMesh.edgeActive.size() - 2].setFace(edgetoSplit->getFace());

			// update verticesEdge map

			string hashKey = (to_string(edgetoSplit->getVertex()->getVertexId()) + "," + to_string(edgetoSplitSym->getVertex()->getVertexId()));
			inMesh.verticesEdge[hashKey] = edgetoSplitSym->getEdgeId();

			string hashKey1 = (to_string(edgetoSplitSym->getVertex()->getVertexId()) + "," + to_string(edgetoSplit->getVertex()->getVertexId()));
			inMesh.verticesEdge[hashKey1] = edgetoSplit->getEdgeId();

		}

		return VertId;
	}


	/*! \brief This method flips the edge shared bettwen two rainglua faces.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
	*	\since version 0.0.1
	*/
	void flipTriangleEdge(zMesh &inMesh, int &index)
	{
		if (index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		if (!inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");
		
		zEdge* edgetoFlip = &inMesh.edges[index];
		zEdge* edgetoFlipSym = edgetoFlip->getSym();

		if (!edgetoFlip->getFace() || !edgetoFlipSym->getFace())
		{
			throw std::invalid_argument("\n Cannot flip boundary edge %i ");
			return;
		}

		vector<int> edgetoFlip_fVerts;
		inMesh.getVertices(edgetoFlip->getFace()->getFaceId(), zFaceData, edgetoFlip_fVerts);

		vector<int> edgetoFlipSym_fVerts;
		inMesh.getVertices(edgetoFlipSym->getFace()->getFaceId(), zFaceData, edgetoFlipSym_fVerts);

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
		inMesh.verticesEdge.erase(removeHashKey);

		string removeHashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
		inMesh.verticesEdge.erase(removeHashKey1);

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
		inMesh.verticesEdge[hashKey] = edgetoFlipSym->getEdgeId();

		string hashKey1 = (to_string(edgetoFlipSym->getVertex()->getVertexId()) + "," + to_string(edgetoFlip->getVertex()->getVertexId()));
		inMesh.verticesEdge[hashKey1] = edgetoFlip->getEdgeId();
	}

	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	index					- index of the edge to be removed.
	*	\param		[in]	removeInactiveElements	- inactive elements in the list would be removed if true.
	*	\since version 0.0.1
	*/
	void deleteEdge(zMesh &inMesh, int index, bool removeInactiveElements = true)
	{

	}
	


	/*! \brief This method splits a set of edges and faces of a mesh in a continuous manner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	edgeList		- indicies of the edges to be split.
	*	\param		[in]	edgeFactor		- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
	*	\since version 0.0.1
	*/	
	void splitFaces(zMesh &inMesh, vector<int> &edgeList, vector<double> &edgeFactor)
	{
		if (edgeFactor.size() > 0)
		{
			if(edgeList.size() != edgeFactor.size()) throw std::invalid_argument(" error: size of edgelist and edge factor dont match.");
		}
		
		int numOriginalVertices = inMesh.vertexActive.size();
		int numOriginalEdges = inMesh.edgeActive.size();
		int numOriginalFaces = inMesh.faceActive.size();

		for (int i = 0; i < edgeList.size(); i ++ )
		{
			if(edgeFactor.size() > 0) splitEdge(inMesh, edgeList[i], edgeFactor[i]);
			else splitEdge(inMesh, edgeList[i]);
		}

		for (int j = 0; j <  edgeList.size(); j++)
		{
			for (int i = 0; i < 2; i++)
			{
				zEdge *start = (i ==0) ? &inMesh.edges[edgeList[j]] : inMesh.edges[edgeList[j]].getSym();

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
				bool eExists = inMesh.edgeExists(v1, v2, outEdgeId);

				if (eExists) continue;

				int startEdgeId = start->getEdgeId();
				int e_EdgeId = e->getEdgeId();

				bool resizeEdges = inMesh.addEdges(v1, v2);

				if (resizeEdges)
				{
					start = &inMesh.edges[startEdgeId];
					e = &inMesh.edges[e_EdgeId];
				}

				inMesh.addPolygon(); // empty polygon

									 // update pointers
				zEdge *start_next = start->getNext();
				zEdge *e_next = e->getNext();

				start->setNext(&inMesh.edges[inMesh.numEdges() - 2]);
				e_next->setPrev(&inMesh.edges[inMesh.numEdges() - 2]);

				start_next->setPrev(&inMesh.edges[inMesh.numEdges() - 1]);
				e->setNext(&inMesh.edges[inMesh.numEdges() - 1]);

				inMesh.faces[inMesh.numPolygons() - 1].setEdge(start_next);

				// edge face pointers to new face
				zEdge *newFace_E = start_next;

				do
				{
					newFace_E->setFace(&inMesh.faces[inMesh.numPolygons() - 1]);

					if (newFace_E->getNext()) newFace_E = newFace_E->getNext();
					else exit = true;

				} while (newFace_E != start_next && !exit);

			}

		}


	}
	

	/*! \brief This method subdivides all the faces and edges of the mesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*	\since version 0.0.1
	*/	
	void subdivideMesh(zMesh &inMesh, int numDivisions)
	{
		for (int j = 0; j < numDivisions; j++)
		{

			int numOriginalVertices = inMesh.vertexActive.size();

			// split edges at center
			int numOriginaledges = inMesh.edgeActive.size();

			for (int i = 0; i < numOriginaledges; i += 2)
			{
				if(inMesh.edgeActive[i]) splitEdge(inMesh, i);
			}

			
			// get face centers
			vector<zVector> fCenters;
			getCenters(inMesh, zFaceData, fCenters);					
	
		
			// add faces
			int numOriginalfaces = inMesh.faceActive.size();
			
			for (int i = 0; i < numOriginalfaces; i++)
			{
				if (!inMesh.faceActive[i]) continue;
		
				vector<int> fEdges; 
				inMesh.getEdges(i,zFaceData, fEdges);

				vector<int> fVerts;
				inMesh.getVertices(i, zFaceData, fVerts);


				// disable current face
				inMesh.faceActive[i] = false;	

				int numCurrentEdges = inMesh.edgeActive.size();;

				// check if vertex exists if not add new vertex
				int VertId;
				bool vExists = inMesh.vertexExists(fCenters[i], VertId);
				if (!vExists)
				{
					inMesh.addVertex(fCenters[i]);
					VertId = inMesh.vertexActive.size() - 1;
				}


				// add new faces				
				int startId = 0; 
				if (inMesh.edges[fEdges[0]].getVertex()->getVertexId() < numOriginalVertices) startId = 1;

				for (int k = startId; k < fEdges.size() + startId; k += 2)
				{
					vector<int> newFVerts;

					int v1 = inMesh.edges[fEdges[k]].getVertex()->getVertexId();
					newFVerts.push_back(v1);

					int v2 = VertId; // face center
					newFVerts.push_back(v2);

					int v3 = inMesh.edges[fEdges[k]].getPrev()->getSym()->getVertex()->getVertexId();
					newFVerts.push_back(v3);

					int v4 = inMesh.edges[fEdges[k]].getPrev()->getVertex()->getVertexId();
					newFVerts.push_back(v4);
					
					inMesh.addPolygon(newFVerts);
					
				}
				
			}

			inMesh.computeMeshNormals();

		}
		
	}

	/*! \brief This method applies Catmull-Clark subdivision to the mesh.
	*
	*	\details Based on https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface and https://rosettacode.org/wiki/Catmull%E2%80%93Clark_subdivision_surface.
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*	\param		[in]	smoothCorner	- corner vertex( only 2 Connected Edges) is also smothed if true.
	*	\since version 0.0.1
	*/
	void smoothMesh(zMesh &inMesh, int numDivisions = 1, bool smoothCorner = false)
	{
		for (int j = 0; j < numDivisions; j++)
		{

			// get face centers
			vector<zVector> fCenters;
			getCenters(inMesh, zFaceData, fCenters);

			// get edge centers
			vector<zVector> tempECenters;
			vector<zVector> eCenters;
			getCenters(inMesh, zEdgeData, eCenters);

			tempECenters = eCenters;

			// compute new smooth positions of the edge centers
			for (int i = 0; i < eCenters.size(); i += 2)
			{

				zVector newPos;

				if (inMesh.onBoundary(i, zEdgeData) || inMesh.onBoundary(i + 1, zEdgeData)) continue;

				int eId = i;

				vector<int> eVerts;
				inMesh.getVertices(i, zEdgeData, eVerts);
				for (int j = 0; j < eVerts.size(); j++) newPos += inMesh.vertexPositions[eVerts[j]];


				vector<int> eFaces;
				inMesh.getFaces(i, zEdgeData, eFaces);
				for (int j = 0; j < eFaces.size(); j++) newPos += fCenters[eFaces[j]];

				newPos /= (eFaces.size() + eVerts.size());

				eCenters[i] = newPos;
				eCenters[i + 1] = newPos;

			}

			// compute new smooth positions for the original vertices
			for (int i = 0; i < inMesh.vertexPositions.size(); i++)
			{
				if (inMesh.onBoundary(i, zVertexData))
				{
					vector<int> cEdges;
					inMesh.getConnectedEdges(i, zVertexData, cEdges);

					if (!smoothCorner && cEdges.size() == 2) continue;

					zVector P = inMesh.vertexPositions[i];
					int n = 1;

					zVector R;
					for (int j = 0; j < cEdges.size(); j++)
					{
						int symEdge = inMesh.edges[cEdges[j]].getSym()->getEdgeId();

						if (inMesh.onBoundary(cEdges[j], zEdgeData) || inMesh.onBoundary(symEdge, zEdgeData))
						{
							R += tempECenters[cEdges[j]];
							n++;
						}
					}

					inMesh.vertexPositions[i] = (P + R) / n;

				}
				else
				{
					zVector R;
					vector<int> cEdges;
					inMesh.getConnectedEdges(i, zVertexData, cEdges);
					for (int j = 0; j < cEdges.size(); j++) R += tempECenters[cEdges[j]];
					R /= cEdges.size();

					zVector F;
					vector<int> cFaces;
					inMesh.getConnectedFaces(i, zVertexData, cFaces);
					for (int j = 0; j < cFaces.size(); j++) F += fCenters[cFaces[j]];
					F /= cFaces.size();

					zVector P = inMesh.vertexPositions[i];
					int n = cFaces.size();

					inMesh.vertexPositions[i] = (F + (R * 2) + (P * (n - 3))) / n;
				}


			}


			int numOriginalVertices = inMesh.vertexActive.size();

			// split edges at center
			int numOriginaledges = inMesh.edgeActive.size();

			for (int i = 0; i < numOriginaledges; i += 2)
			{
				if (inMesh.edgeActive[i])
				{
					int newVert = splitEdge(inMesh, i);

					inMesh.vertexPositions[newVert] = eCenters[i];
				}
			}



			// add faces
			int numOriginalfaces = inMesh.faceActive.size();

			for (int i = 0; i < numOriginalfaces; i++)
			{
				if (!inMesh.faceActive[i]) continue;

				vector<int> fEdges;
				inMesh.getEdges(i, zFaceData, fEdges);

				vector<int> fVerts;
				inMesh.getVertices(i, zFaceData, fVerts);


				// disable current face
				inMesh.faceActive[i] = false;

				int numCurrentEdges = inMesh.edgeActive.size();;

				// check if vertex exists if not add new vertex
				int VertId;
				bool vExists = inMesh.vertexExists(fCenters[i], VertId);
				if (!vExists)
				{
					inMesh.addVertex(fCenters[i]);
					VertId = inMesh.vertexActive.size() - 1;
				}


				// add new faces				
				int startId = 0;
				if (inMesh.edges[fEdges[0]].getVertex()->getVertexId() < numOriginalVertices) startId = 1;

				for (int k = startId; k < fEdges.size() + startId; k += 2)
				{
					vector<int> newFVerts;

					int v1 = inMesh.edges[fEdges[k]].getVertex()->getVertexId();
					newFVerts.push_back(v1);

					int v2 = VertId; // face center
					newFVerts.push_back(v2);

					int v3 = inMesh.edges[fEdges[k]].getPrev()->getSym()->getVertex()->getVertexId();
					newFVerts.push_back(v3);

					int v4 = inMesh.edges[fEdges[k]].getPrev()->getVertex()->getVertexId();
					newFVerts.push_back(v4);

					inMesh.addPolygon(newFVerts);

				}

			}

			inMesh.computeMeshNormals();

		}

	}


	//--------------------------
	//---- REMESH METHODS
	//--------------------------


	/*! \brief This method splits an edge longer than the given input value at its midpoint and  triangulates the mesh. the adjacent triangles are split into 2-4 triangles.
	*
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	maxEdgeLength	- maximum edge length.
	*	\since version 0.0.1
	*/
	void splitLongEdges(zMesh &inMesh, double maxEdgeLength = 0.5);

	/*! \brief This method collapses an edge shorter than the given minimum edge length value if the collapsing doesnt produce adjacent edges longer than the maximum edge length.
	*
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf
	*	\param		[in]	inMesh				- input mesh.
	*	\param		[in]	minEdgeLength		- minimum edge length.
	*	\param		[in]	maxEdgeLength		- maximum edge length.
	*	\since version 0.0.1
	*/
	void collapseShortEdges(zMesh &inMesh, double minEdgeLength = 0.1, double maxEdgeLength = 0.5);

	/*! \brief This method equalizes the vertex valences by flipping edges of the input triangulated mesh. Target valence for interior vertex is 4 and boundary vertex is 6.
	*
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf
	*	\param		[in]	inMesh				- input mesh.
	*	\since version 0.0.1
	*/
	void equalizeValences(zMesh &inMesh);

	/*! \brief This method applies an iterative smoothing to the mesh by  moving the vertex but constrained to its tangent plane.
	*
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf
	*	\param		[in]	inMesh				- input mesh.
	*	\since version 0.0.1
	*/
	void tangentialRelaxation(zMesh &inMesh);






	/** @}*/

	/** @}*/

	/** @}*/
}