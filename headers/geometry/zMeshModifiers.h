#pragma once
#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>



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

	/** \addtogroup zMeshModifiers
	*	\brief Collection of mesh modifiers methods.
	*  @{
	*/


	//--------------------------
	//---- MODIFIER METHODS
	//--------------------------


	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	pos			- zVector holding the position information of the vertex.
	*	\since version 0.0.1
	*/
	
	void deleteVertex(zMesh &inMesh, vector<int> &vertexList)
	{
		for (int i = 0; i < vertexList.size(); i++)
		{
			// get connected edges
			//vector<int> cEdges;
			//inMesh.getConnectedEdges(vertexList[i], zVertexData, cEdges);

			// check if it is a boundary vertex
			if (inMesh.onBoundary(vertexList[i], zVertexData))
			{

			}
			else
			{

				// get connected edges
				vector<int> cFaces;
				inMesh.getConnectedFaces(vertexList[i], zVertexData, cFaces);
							
				int minFaceId = zMin(cFaces);

				vector<int> temp_cFaceEdges;

				for (int k = 0; k < cFaces.size(); k++)
				{

					vector<int> fEdges;
					inMesh.getEdges(cFaces[k], zFaceData, fEdges);

					for (int j = 0; j < fEdges.size(); j++)
					{
						temp_cFaceEdges.push_back(fEdges[j]);
					}
					
				}

				
				for (int k = 0; k < temp_cFaceEdges.size(); k++)
				{

					// update edge pointers

					if (inMesh.edges[temp_cFaceEdges[k]].getVertex()->getVertexId() == vertexList[i])
					{
						// remove from verticesEdge map
						inMesh.removeFromVerticesEdge(inMesh.edges[temp_cFaceEdges[k]].getVertex()->getVertexId(), inMesh.edges[temp_cFaceEdges[k]].getSym()->getVertex()->getVertexId());

						inMesh.edges[temp_cFaceEdges[k]].getPrev()->setNext(inMesh.edges[temp_cFaceEdges[k]].getSym()->getNext());

						// vertex edge pointer update
						inMesh.edges[temp_cFaceEdges[k]].getPrev()->getVertex()->setEdge(inMesh.edges[temp_cFaceEdges[k]].getSym()->getNext());


						inMesh.edgeActive[inMesh.edges[temp_cFaceEdges[k]].getEdgeId()] = false;
						inMesh.edgeActive[inMesh.edges[temp_cFaceEdges[k]].getSym()->getEdgeId()] = false;

						int newNumEdges = inMesh.numEdges() - 2;
						inMesh.setNumEdges(newNumEdges, false);
					}

					// update face containers

					int faceId = inMesh.edges[temp_cFaceEdges[k]].getFace()->getFaceId();
				
					if (faceId != minFaceId && inMesh.faceActive[faceId] )
					{

						//printf("\n %i : faceId : %i , minFaceId %i ", temp_cFaceEdges[k], faceId, minFaceId);

						inMesh.faceActive[faceId] = false;
						int newNumFaces = inMesh.numPolygons() - 1;
						inMesh.setNumPolygons(newNumFaces, false);

						
					}

					if (inMesh.edgeActive[temp_cFaceEdges[k]])
					{
						inMesh.edges[temp_cFaceEdges[k]].setFace(&inMesh.faces[minFaceId]);

						inMesh.faces[minFaceId].setEdge(&inMesh.edges[temp_cFaceEdges[k]]);
					}
					
					
				}
			}

			// update vertex containers
			inMesh.vertexActive[vertexList[i]] = false;

			inMesh.vertexPositions[vertexList[i]] = zVector(10000, 10000, 10000); // dummy position for VBO
			inMesh.vertexNormals[vertexList[i]] = zVector(0, 0, 1); // dummy normal for VBO
			inMesh.vertexColors[vertexList[i]] = zColor(1, 1, 1,1); // dummy color for VBO

			// remove from vertexPosition map
			inMesh.removeFromPositionMap(inMesh.vertexPositions[vertexList[i]]);

			int newNumVertices = inMesh.numVertices() - 1;
			inMesh.setNumVertices(newNumVertices, false);			

		}

		inMesh.computeMeshNormals();
	}
		

	/*! \brief This method collapses an edge into a vertex.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	index	- index of the edge to be collapsed.
	*	\since version 0.0.1
	*/
	
	void collapseEdge(zMesh &inMesh, int index)
	{
		if (index > inMesh.edgeActive.size()) throw std::invalid_argument(" error: index out of bounds.");
		if (!inMesh.edgeActive[index]) throw std::invalid_argument(" error: index out of bounds.");



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
	void flipTriangleEdges(zMesh &inMesh, vector<int> &edgeList);

	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	pos			- zVector holding the position information of the vertex.
	*	\since version 0.0.1
	*/
	void deleteEdge(zMesh &inMesh, vector<int> &edgeList);

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

			if (ang <= 0 || ang == 180 ) reflexVerts.push_back(true);
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
				for(int i = 0; i <vertexIndices.size(); i++)
				{

					int nextId = (i + 1) % vertexIndices.size();
					int prevId = (i - 1 + vertexIndices.size()) % vertexIndices.size();

					// Triangle edges - e1 and e2 defined above
					zVector v1 = inMesh.vertexPositions[vertexIndices[nextId]] - inMesh.vertexPositions[vertexIndices[i]];
					zVector v2 = inMesh.vertexPositions[vertexIndices[prevId]] - inMesh.vertexPositions[vertexIndices[i]];

					zVector cross = v1 ^ v2;
					double ang = v1.angle(v2);

					if (cross * norm < 0) ang *= -1;

					if(ang <= 0 || ang == 180 ) reflexVerts.push_back(true);
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
	
		if (inMesh.faceNormals.size() == 0 || inMesh.faceNormals.size() != inMesh.numPolygons()) inMesh.computeMeshNormals();


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
				polyTriangulate(inMesh,i, n_Tris, Tri_connects);

				//printf("\n numtris: %i %i ", n_Tris, Tri_connects.size());

				for (int j = 0; j < n_Tris; j++)
				{

					vector<int> triVerts;
					triVerts.push_back(Tri_connects[j * 3]);
					triVerts.push_back(Tri_connects[j * 3 + 1]);
					triVerts.push_back(Tri_connects[j * 3 + 2]);

					

					// check if edges e01, e12 or e20
					int e01_ID, e12_ID, e20_ID;
					bool e01_Boundary = false;
					bool e12_Boundary = false;
					bool e20_Boundary = false;

					for (int k = 0; k < triVerts.size(); k++)
					{
						int e;
						bool eExists = inMesh.edgeExists(triVerts[k], triVerts[(k + 1) % triVerts.size()],  e);
					

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
								e12_ID =e;

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

			
					// add face if the current face edge points to the edges of the current triangle

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
	
	void subDivideMesh(zMesh &inMesh, int numDivisions)
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

				int numCurrentEdges = inMesh.edgeActive.size();;

				// check if vertex exists if not add new vertex
				int VertId;
				bool vExists = inMesh.vertexExists(fCenters[i], VertId);
				if (!vExists)
				{
					inMesh.addVertex(fCenters[i]);
					VertId = inMesh.vertexActive.size() - 1;
				}

				// add new edges
				int startId = 0; 
				if (inMesh.edges[fEdges[0]].getVertex()->getVertexId() < numOriginalVertices) startId = 1;

				for (int k = startId; k < fEdges.size() + startId; k += 2)
				{
					int v1 = inMesh.edges[fEdges[k]].getVertex()->getVertexId();
					int v2 = VertId; // face center
					
					bool edgesResize = inMesh.addEdges(v1, v2);



					if (k == startId) inMesh.vertices[v2].setEdge(&inMesh.edges[inMesh.numEdges() - 1]);
					
				}

				// update pointers
				if (startId == 0)
				{
					for (int k = 0; k < fEdges.size(); k += 2)
					{
						int prevId = ((k - 1 + fEdges.size()) % fEdges.size());

						inMesh.edges[fEdges[k]].setNext(&inMesh.edges[numCurrentEdges + k]);

						inMesh.edges[numCurrentEdges + k].setNext(&inMesh.edges[numCurrentEdges + prevId]);

						inMesh.edges[numCurrentEdges + prevId].setNext(inMesh.edges[fEdges[k]].getPrev());


						if (k > 0)
						{
							// add face
							bool facesResize = inMesh.addPolygon();							
							inMesh.faces[inMesh.numPolygons() - 1].setEdge(&inMesh.edges[fEdges[k]]);

							// update edge face pointers
							inMesh.edges[fEdges[k]].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[numCurrentEdges + k].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[numCurrentEdges + prevId].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[fEdges[k]].getPrev()->setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
												
						}
						else
						{
							// update edge face pointers
							inMesh.edges[numCurrentEdges + k].setFace(&inMesh.faces[i]);
							inMesh.edges[numCurrentEdges + prevId].setFace(&inMesh.faces[i]);
						}

					}
				}
				else
				{
					for (int k = 0; k < fEdges.size() ; k += 2)
					{
						int nextId = ((k + 1) % fEdges.size());

						int prevId = ((k - 1 + fEdges.size()) % fEdges.size());

						inMesh.edges[fEdges[nextId]].setNext(&inMesh.edges[numCurrentEdges + k]);

						inMesh.edges[numCurrentEdges + k].setNext(&inMesh.edges[numCurrentEdges + prevId]);

						inMesh.edges[numCurrentEdges + prevId].setNext(&inMesh.edges[fEdges[k]]);


						if (k > 0)
						{
							// add face
							bool facesResize = inMesh.addPolygon();
							inMesh.faces[inMesh.numPolygons() - 1].setEdge(&inMesh.edges[fEdges[nextId]]);

							// update edge face pointers
							inMesh.edges[fEdges[k]].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[numCurrentEdges + k].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[numCurrentEdges + prevId].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
							inMesh.edges[fEdges[nextId]].setFace(&inMesh.faces[inMesh.numPolygons() - 1]);
						}
						else
						{
							// update edge face pointers
							inMesh.faces[i].setEdge(&inMesh.edges[fEdges[nextId]]);

							inMesh.edges[numCurrentEdges + k].setFace(&inMesh.faces[i]);
							inMesh.edges[numCurrentEdges + prevId].setFace(&inMesh.faces[i]);
						}

					}
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