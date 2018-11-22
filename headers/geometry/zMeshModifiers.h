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

	/*! \brief This method triangulates the input mesh.
	*
	*	\param		[in]	computeNormal	-	true if normals are not computed previously.
	*	\since version 0.0.1
	*/
	void triangulate(bool computeNormal = false);

	/*! \brief This method triangulates the input polygon using ear clipping algorithm based on : https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
	*
	*	\param		[in]	polyIndex			- polygon index  of the face to be triangulated in the face array.
	*	\param		[out]	numTris				- number of triangles in the input polygon.
	*	\param		[out]	tris				- index array of each triangle associated with the face.
	*/
	void polyTriangulate(int &polyIndex, int &numTris, vector<int> &tris, bool computeNormal = true);

	/*! \brief This method splits a set of edges and faces of a mesh in a continuous manner.
	*
	*	\param		[in]	edgeList	- indicies of the edges to be split.
	*	\param		[in]	edgeFactor	- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
	*/
	void splitFaces(vector<int> &edgeList, vector<double> &edgeFactor);

	/*! \brief This method subdivides all the faces and edges of the mesh.
	*
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*/
	void subDivideMesh(int numDivisions);

	/*! \brief This method subdivides the face and contained edges of the mesh at the given input index.
	*
	*	\param		[in]	numDivisions	- number of subdivision to be done on the mesh.
	*/
	void subDivideFace(int &index, int numDivisions);


	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	pos			- zVector holding the position information of the vertex.
	*/
	void deleteVertex(vector<int> &vertexList);

	/*! \brief This method collapses all the edges in the input edge list.
	*
	*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
	*/
	void collapseEdges(vector<int> &edgeList);

	/*! \brief This method collapses an edge into a vertex.
	*
	*	\param		[in]	index	- index of the edge to be collapsed.
	*/
	void collapseEdge(int index);


	/*! \brief This method splits a set of edges of a mesh in a continuous manner.
	*
	*	\param		[in]	edgeList		- indicies of the edges to be split.
	*	\param		[in]	edgeFactor		- array of factors in the range [0,1] that represent how far along each edge must the split be done. This array must have the same number of elements as the edgeList array.
	*	\param		[out]	splitVertexId	- stores indices of the new vertex per edge in the given input edgelist.
	*/
	void splitEdges(vector<int> &edgeList, vector<double> &edgeFactor, vector<int> &splitVertexId);

	/*! \brief This method splits an edge and inserts a vertex along the edge at the input factor.
	*
	*	\param		[in]	index			- index of the edge to be split.
	*	\param		[in]	edgeFactor		- factor in the range [0,1] that represent how far along each edge must the split be done.
	*	\return				int				- index of the new vertex added after splitinng the edge.
	*/
	int splitEdge(int index, double edgeFactor = 0.5);


	/*! \brief This method flips the edge shared bettwen two rainglua faces.
	*
	*	\param		[in]	edgeList	- indicies of the edges to be collapsed.
	*/
	void flipTriangleEdges(vector<int> &edgeList);

	/*! \brief This method deletes the zMesh vertices given in the input vertex list.
	*
	*	\param		[in]	pos			- zVector holding the position information of the vertex.
	*/
	void deleteEdge(vector<int> &edgeList);



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