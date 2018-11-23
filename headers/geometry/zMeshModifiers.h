#pragma once
#include <headers/geometry/zMesh.h>


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


	/*! \brief This method triangulates the input polygon using ear clipping algorithm based on : https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	faceIndex		- face index  of the face to be triangulated in the face array.
	*	\param		[out]	numTris			- number of triangles in the input polygon.
	*	\param		[out]	tris			- index array of each triangle associated with the face.
	*	\since version 0.0.1
	*/
	void polyTriangulate(zMesh &inMesh, int &faceIndex, int &numTris, vector<int> &tris, bool computeNormal = true)
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


		// compute 
	
	

	}

	/*! \brief This method triangulates the input mesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	computeNormal	-	true if normals are not computed previously.
	*	\since version 0.0.1
	*/
	
	void triangulate(zMesh &inMesh, bool computeNormal = false)
	{
	
		// iterate through faces and triangulate faces with more than 3 vetices
		int numfaces_before = inMesh.faceActive.size();
		//printf("\n numfaces_before: %i ", numfaces_before);

		for (int i = 0; i < numfaces_before; i++)
		{
			if (!inMesh.faceActive[i]) continue;

			vector<int> fVerts;
			inMesh.getVertices(i, zFaceData, fVerts);

			if (fVerts.size() != 3)
			{
				// compute polygon Triangles
				int n_Tris = 0;
				vector<int> Tri_connects;
				polyTriangulate(inMesh,i, n_Tris, Tri_connects, computeNormal);

				//printf("\n numtris: %i %i ", n_Tris, Tri_connects.size());

				for (int j = 0; j < n_Tris; j++)
				{

					vector<int> triVerts;
					triVerts.push_back(Tri_connects[j * 3]);
					triVerts.push_back(Tri_connects[j * 3 + 1]);
					triVerts.push_back(Tri_connects[j * 3 + 2]);

					// check if edges e01, e12 or e20
					zEdge* e01 = NULL;
					zEdge* e12 = NULL;
					zEdge* e20 = NULL;

					for (int k = 0; k < triVerts.size(); k++)
					{
						int e;
						bool eExists = inMesh.edgeExists(triVerts[k], triVerts[(k + 1) % triVerts.size()], e);

						//printf("\n k: %i e : %i %i %s",k, triVerts[k], triVerts[(k + 1) % triVerts.size()], (eExists)?"true": "false");

						if (k == 0)
						{

							if (eExists)
							{
								e01 = &inMesh.edges[e];

							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e01 = &inMesh.edges[inMesh.edgeActive.size() - 2];


							}
						}


						if (k == 1)
						{
							if (eExists)
							{
								e12 = &inMesh.edges[e];
							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e12 = &inMesh.edges[inMesh.edgeActive.size() - 2];
							}
						}



						if (k == 2)
						{
							if (eExists)
							{
								e20 = &inMesh.edges[e];
							}
							else
							{
								inMesh.addEdges(triVerts[k], triVerts[(k + 1) % triVerts.size()]);
								e20 = &inMesh.edges[inMesh.edgeActive.size() - 2];
							}

						}



					}

			
					// add face if the current face edge points to the edges of the current triangle

					if (e01 != NULL && e12 != NULL && e20 != NULL)
					{
						if (inMesh.faces[i].getEdge() != e01 && inMesh.faces[i].getEdge() != e12 && inMesh.faces[i].getEdge() != e20)
						{
							inMesh.addPolygon();
							inMesh.faces[inMesh.faceActive.size() - 1].setEdge(e01);

							e01->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
							e12->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
							e20->setFace(&inMesh.faces[inMesh.faceActive.size() - 1]);
						}
						else
						{
							e01->setFace(&inMesh.faces[i]);
							e12->setFace(&inMesh.faces[i]);
							e20->setFace(&inMesh.faces[i]);
						}

						// update edge pointers
						e01->setNext(e12);
						e01->setPrev(e20);
						e12->setNext(e20);
					}


				}
			}

		}


	}


	/*! \brief This method splits a set of edges and faces of a mesh in a continuous manner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	edgeList	- indicies of the edges to be split.
	*	\param		[in]	edgeFactor	- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
	*	\since version 0.0.1
	*/
	void splitFaces(zMesh &inMesh, vector<int> &edgeList, vector<double> &edgeFactor);

	/*! \brief This method subdivides all the faces and edges of the mesh.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*	\since version 0.0.1
	*/
	void subDivideMesh(zMesh &inMesh, int numDivisions);

	/*! \brief This method subdivides the face and contained edges of the mesh at the given input index.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*	\since version 0.0.1
	*/
	void subDivideFace(zMesh &inMesh, int &index, int numDivisions);


	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	pos			- zVector holding the position information of the vertex.
	*	\since version 0.0.1
	*/
	void deleteVertex(zMesh &inMesh, vector<int> &vertexList);

	/*! \brief This method collapses all the edges in the input edge list.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
	*	\since version 0.0.1
	*/
	void collapseEdges(zMesh &inMesh, vector<int> &edgeList);

	/*! \brief This method collapses an edge into a vertex.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	index	- index of the edge to be collapsed.
	*	\since version 0.0.1
	*/
	void collapseEdge(zMesh &inMesh, int index);


	/*! \brief This method splits a set of edges of a mesh in a continuous manner.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	edgeList		- indicies of the edges to be split.
	*	\param		[in]	edgeFactor		- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
	*	\param		[out]	splitVertexId	- stores indices of the new vertex per edge in the given input edgelist.
	*	\since version 0.0.1
	*/
	void splitEdges(zMesh &inMesh, vector<int> &edgeList, vector<double> &edgeFactor, vector<int> &splitVertexId);

	/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[in]	index			- index of the edge to be split.
	*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
	*	\return				int				- index of the new vertex added after splitinng the edge.
	*	\since version 0.0.1
	*/
	int splitEdge(zMesh &inMesh, int index, double edgeFactor = 0.5);


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