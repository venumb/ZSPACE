// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#ifndef ZSPACE_TS_STATICS_POLYTOPAL_H
#define ZSPACE_TS_STATICS_POLYTOPAL_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;


#include <depends/alglib/cpp/src/ap.h>
#include <depends/alglib/cpp/src/linalg.h>
#include <depends/alglib/cpp/src/optimization.h>
using namespace alglib;

namespace zSpace
{	   
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z3DGS
	*	\brief tool sets for 3D graphic statics.
	*  @{
	*/

	/*! \class zTsPolytopal
	*	\brief A toolset for 3D graphics and poytopal meshes.
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf) and Freeform Developable Spatial Structures (https://www.ingentaconnect.com/content/iass/piass/2016/00002016/00000003/art00010 )
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsPolytopal
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- FORCE DIAGRAM ATTRIBUTES
		//--------------------------
		

		//--------------------------
		//---- FORM DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief container of  form particle objects  */
		vector<zObjParticle> formParticlesObj;

		/*!	\brief container of form particle function set  */
		vector<zFnParticle> fnFormParticles;

		/*!	\brief container storing the target for form edges.  */
		zVectorArray targetEdges_form;

		//--------------------------
		//---- POLYTOPAL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to polytopal Object  */
		zObjMeshPointerArray polytopalObjs;

		int smoothSubDivs;


	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to form Object  */
		zObjGraph *formObj;

		/*!	\brief pointer container to force Object  */
		zObjMeshPointerArray forceObjs;

		/*!	\brief form function set  */
		zFnGraph fnForm;

		/*!	\brief container of force function set  */
		vector<zFnMesh> fnForces;

		/*!	\brief container of polytopals function set  */
		vector <zFnMesh> fnPolytopals;

		/*!	\brief force volume mesh faceID to form graph vertexID map.  */
		unordered_map <string, int> forceVolumeFace_formGraphVertex;

		/*!	\brief form graph vertexID to volume meshID map.  */
		zIntArray formGraphVertex_forceVolumeMesh;

		/*!	\brief form graph vertexID to local volume mesh faceID map.*/
		zIntArray formGraphVertex_forceVolumeFace;

		/*!	\brief graph vertex offset container.*/
		vector<double> formGraphVertex_Offsets;

		/*!	\brief container of facecenter per force volume  */
		vector<vector<zVector>> force_fCenters;

		//--------------------------
		//---- ALGEBRAIC ATTRIBUTES
		//--------------------------

		int primal_n_v = 0;
		int primal_n_f = 0;
		int primal_n_e = 0;

		int primal_n_v_i = 0; // primal vertex internal	
		int primal_n_e_i = 0; // primal edges internal		
		int primal_n_f_i = 0; // primal faces internal


		unordered_map <string, int> volumeVertex_PrimalVertex;  // map of volume-vertex hashkey to primal vertex

		unordered_map <string, int> volumeEdge_PrimalEdge;  // map of volume-edge hashkey to primal edge
		unordered_map <string, int> primalVertices_PrimalEdge;  // map of primal vertex-vertex hashkey to primal edge

		unordered_map <string, int> volumeFace_PrimalFace; // map of volume-face hashkey to primal face

		vector< zIntArray > primalFace_VolumeFace; // stores the one volume face combination per primal face
		vector< zIntArray > primalEdge_VolumeEdge; // stores the one volume edge combination per primal edge

		vector< zIntArray > primalEdge_PrimalVertices; // stores theprimal vertices per primal edge

		zIntArray primal_internalFaceIndex; // -1 for GFP or SSP faces 
		zIntArray internalFaceIndex_primalFace;

		zIntArray primal_internalEdgeIndex; // -1 for GFP or SSP edges 
		zIntArray internalEdgeIndex_primalEdge;

		zIntArray primal_internalVertexIndex; // -1 for GFP or SSP vertex 
		zIntArray internalVertexIndex_primalVertex;

		zPointArray primalVertexPositions;
		zPointArray primalFaceCenters;
		zPointArray primalFaceNormals;
		zDoubleArray primalFaceAreas;

		vector<zIntArray> primalFace_Volumes;

		zBoolArray GFP_SSP_Face;
		zBoolArray GFP_SSP_Edge;
		zBoolArray GFP_SSP_Vertex;

		vector<zIntArray> primalVertex_ConnectedPrimalFaces;
		zIntArray Y_vertex;
		
		zPointArray userSection_points;
		double userSection_edgeLength;
		double userSection_numEdge;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsPolytopal();


		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_forceObjs			- input container of force objects.
		*	\param		[in]	_polytopalObjs		- input container of polytopal objects.
		*	\since version 0.0.2
		*/
		zTsPolytopal(zObjGraph &_formObj, zObjMeshArray &_forceObjs, zObjMeshArray  &_polytopalObjs);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsPolytopal();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param [in]		filePaths		- input container of file paths.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.2
		*/
		void createForceFromFiles(zStringArray filePaths, zFileTpye type);

		/*! \brief This method creates the center line graph based on the input volume meshes.
		*
		*	\param		[in]	edgeCol						- input edge color for the form.
		*	\param		[in]	includeBoundary				- external force vectors included  as edges of the graph if true.
		*	\param		[in]	boundaryEdgelength			- length of external force vectors included  as edges of the graph.
		*	\since version 0.0.2
		*/
		void createFormFromForce( zColor edgeCol = zColor(0.75, 0.5, 0, 1), bool includeBoundary = false , double boundaryEdgelength = 1.0);

		/*! \brief This method creates the polytopal mesh from the force volume meshes and form graph.
		*
		*	\param		[in]	offset					- input offset value for the cross section of node.
		*	\param		[in]	param					- input param value for straight beam sections.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void createPolytopalsFromForce(double offset, double param = 0.0, int subdivs = 0);

		/*! \brief This method creates the polytopal mesh from the force volume meshes and form graph and user profile parameters.
		*
		*	\param		[in]	user_nEdges				- input number of edges in user profile
		*	\param		[in]	user_edgeLength			- input edge lengths of user profile
		*	\param		[in]	offset					- input offset value for the cross section of node.
		*	\param		[in]	param					- input param value for straight beam sections.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void createPolytopalsFromForce_profile(int user_nEdges, double user_edgeLength, double offset, double param = 0.0, int subdivs = 0);

		/*! \brief This method creates the user section points transformed to  all the internal faces.
		*
		*	\param		[in]	user_nEdges				- input number of edges of the profile.
		*	\param		[out]	user_edgeLength			- input length of each edge in the profile.
		*	\since version 0.0.3
		*/
		void createSectionPoints(int user_nEdges, double user_edgeLength);
		
		//--------------------------
		//----3D GS ITERATIVE 
		//--------------------------
		
		/*! \brief This method updates the form diagram to find equilibrium shape..
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations.
		*	\param		[in]	angleTolerance				- tolerance for angle.
		*	\param		[in]	colorEdges					- colors edges based on angle deviation if true.
		*	\param		[in]	printInfo					- prints angle deviation info if true.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool equilibrium(bool &computeTargets, double minmax_Edge, zDomainFloat &deviations,  double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = EPS, bool colorEdges = false, bool printInfo = false);
		
		//--------------------------
		//----3D GS ALGEBRAIC
		//--------------------------

		/*! \brief This method computes global indicies of faces, edges and vertices of the primal, required to build the connectivity matrix.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[in]	GFP_SSP_Index				- cell index of the Global force polyhedra (primal force diagram)  or Self Stressed Polyhedra (primal form diagram). If -1 , the external faces of the primal is considered as GFP or SSP.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		void getPrimal_GlobalElementIndicies(zDiagramType type, int GFP_SSP_Index = -1, int precisionFac = 6);
			
		/*! \brief This method computes dual using the algebraic method.
		*
		*	\details based on https://psl.design.upenn.edu/wp-content/uploads/2019/03/a3dgs.pdf , https://psl.design.upenn.edu/wp-content/uploads/2018/07/IASS18_AGS_MA_.pdf
		*	\param		[in]	threshold					- input value to determine when values are to be considered nonzero.
		*	\param		[in]	includeBoundary				- external force vectors included  as edges of the graph if true.
		*	\param		[in]	boundaryEdgelength			- length of external force vectors included  as edges of the graph.
		*	\since version 0.0.3
		*/
		void getDual(double threshold, bool includeBoundary = false, double boundaryEdgelength = 1.0);

		//--------------------------
		//----POLYTOPAL
		//--------------------------

		/*! \brief This method closes the polytopal meshes.
		*
		*	\since version 0.0.2
		*/
		void closePolytopals();

		/*! \brief This method computes polytopal mesh based on the force and form diagram.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	offset					- input offset distance from face or volume center.
		*	\param		[in]	param					- input parametrised offset distance from face or volume center.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopal(int forceIndex, double offset, double param = 0.0, int subdivs = 0);

		/*! \brief This method computes polytopal mesh based on the force, form diagram and user profile.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	user_nEdges				- input number of edges in user profile
		*	\param		[in]	user_edgeLength			- input edge lengths of user profile
		*	\param		[in]	offset					- input offset value for the cross section of node.
		*	\param		[in]	param					- input param value for straight beam sections.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*	\warning currently works only with triangluar faces.
		*/
		void getPolytopal_Profile(int forceIndex, int user_nEdges, double user_edgeLength, double offset, double param = 0.0, int subdivs = 0);

		/*! \brief This method computes polytopal beam mesh based on the force, form diagram and user profile.
		*
		*	\param		[in]	user_nEdges				- input number of edges in user profile
		*	\param		[in]	param					- input param value for straight beam sections.
		*	\since version 0.0.2
		*	\warning currently works only with triangluar faces.
		*/
		void getPolytopal_Beams_Profile(int user_nEdges, double user_edgeLength, double param = 0.0);

		//--------------------------
		//--- SET METHODS 
		//--------------------------


		/*! \brief This method sets form vertex offsets of all the vertices to the input value.
		*
		*	\param		[in]	offset			- offset value.
		*	\since version 0.0.2
		*/
		void setVertexOffset(double offset);

		/*! \brief This method sets form vertex offsets of all the vertices with the input container of values.
		*
		*	\param		[in]	offsets			- container of offsets values.
		*	\since version 0.0.2
		*/
		void setVertexOffsets(zDoubleArray &offsets);

		/*! \brief This method computes the dual graph edge weights based on the primal volume mesh face areas.
		*
		*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf)
		*	\param		[in]	weightDomain				- weight domain of the edge.
		*	\since version 0.0.2
		*/
		void setDualEdgeWeightsfromPrimal(zDomainFloat weightDomain = zDomainFloat(2.0, 10.0));

		//--------------------------
		//---- UTILITY METHOD
		//--------------------------

		/*! \brief This method combines the individual polytopal mesh to one mesh object.
		*
		*	\since version 0.0.3
		*/
		zObjMesh getPolytopals();

		/*! \brief This method gets the cut planes required for FREP per edge vertex.
		*
		*	\since version 0.0.3
		*/
		void getBisectorPlanes(zVector3DArray &planes);

		/*! \brief This method exports cut planes required for FREP per edge vertex in a JSON file format.
		*
		*	\since version 0.0.3
		*/
		void exportBisectorPlanes(string path);

	protected:
		
		//--------------------------
		//---- ITERATIVE METHOD UTILITIES
		//--------------------------

		/*! \brief This method computes the face centers of the input force volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.2
		*/
		void computeForcesFaceCenters();

		/*! \brief This method computes the targets per edge of the form.
		*
		*	\since version 0.0.2
		*/
		void computeFormTargets();

		/*! \brief This method if the form mesh edges and corresponding target edge are parallel.
		*
		*	\param		[out]	minDeviation						- stores minimum deviation.
		*	\param		[out]	maxDeviation						- stores maximum deviation.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel or within tolerance.
		*	\since version 0.0.2
		*/
		bool checkParallelity(zDomainFloat & deviation, double angleTolerance, bool colorEdges, bool printInfo);

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations.
		*	\since version 0.0.2
		*/
		void updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000);

		//--------------------------
		//---- ALGEBRAIC METHOD UTILITIES
		//--------------------------

		/*! \brief This method gets the edge-vertex connectivity matrix of the primal. It corresponds to the face-cell connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_EdgeVertexMatrix(zDiagramType type, zSparseMatrix &out);


		/*! \brief This method gets the edge-face connectivity matrix of the primal. It corresponds to the face-edge connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_EdgeFaceMatrix(zDiagramType type, zSparseMatrix &out);

		/*! \brief This method gets the face-cell connectivity matrix of the primal. It corresponds to the edge-vertex connectivity matrix of the dual.
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool getPrimal_FaceCellMatrix(zDiagramType type, zSparseMatrix &out);

		/*! \brief This method gets the equlibrium matrix of the primal. 
		*
		*	\param		[in]	type						- input type of primal - zForceDiagram / zFormDiagram.
		*	\param		[out]	out							- output connectivity matrix.
		*	\return				bool						- true if the matrix is computed.
		*	\since version 0.0.3
		*/
		bool get_EquilibriumMatrix(zDiagramType type, MatrixXd &out);

		/*! \brief This method gets the force densities using the Linear Programing Approach method.
		*
		*	\details based on http://www.alglib.net/translator/man/manual.cpp.html#unit_minbleic
		*	\param		[out]	q							- output vector of force densities.
		*	\since version 0.0.3
		*/
		void getDual_ForceDensities_LPA(VectorXd &q);

#ifndef USING_CLR
		/*! \brief This method gets the force densities using the Moore–Penrose inverse method.
		*
		*	\param		[out]	q							- output vector of force densities.
		*	\param		[in]	threshold					- input threshold value to determine when singular values are to be considered nonzero.
		*	\since version 0.0.3
		*/		
		void getDual_ForceDensities_MPI(VectorXd &q, double threshold = 0.5);

		/*! \brief This method gets the force densities using the Reduced Row Echelon Form method.
		*
		*	\param		[out]	q							- output vector of force densities.
		*	\param		[in]	threshold					- input threshold value to determine when matrix values are to be considered nonzero.
		*	\since version 0.0.3
		*/
		void getDual_ForceDensities_RREF(VectorXd &q, double threshold = 0.5);

		/*! \brief This method gets the B matrix required for computing force densities using RREF method.
		*
		*	\param		[in]	A							- input equilibrium matrix.
		*	\param		[in]	threshold					- input threshold value to determine when matrix values are to be considered nonzero.
		*	\param		[out]	rank						- output rank of equilibrium matrix.
		*	\param		[out]	B							- output B matrix.
		*	\param		[out]	indEdges					- output booelan container indicating independent edges (true).
		*	\return				bool						- true if the matrix is computed.
		*	\note B is of size rank * f-rank	and stores -1.0 * RREF values.
		*	\since version 0.0.3
		*/		
		void get_BMatrix(mat &A, double threshold, int &rank, mat &B, zBoolArray &indEdges);
#endif

		/*! \brief This method gets the load path value for the input force densities.
		*
		*	\details based on http://block.arch.ethz.ch/brg/files/2019_LIEW_SMO_optimising-loadpath-ompression-networks_1550490861.pdf
		*	\param		[in]	q							- input vector of force densities.
		*	\return				double						- load path value.	
		*	\since version 0.0.3
		*/
		double getDual_Loadpath(VectorXd &q);

		//--------------------------
		//---- POLYTOPAL METHOD UTILITIES
		//--------------------------

		/*! \brief This method creates the polytopal mesh based on the input force volume mesh and form graph.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopal(int forceIndex, int subdivs = 0);

		/*! \brief This method remeshes the input mesh to have rulings in ony one direction.
		*
		*	\param		[in]	inFnMesh				- input mesh function set.
		*	\param		[in]	SUBDIVS					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopalRulingRemesh(int index, zObjMesh &inMeshObj, int SUBDIVS);

		/*! \brief This method computes the ruling intersetions.
		*
		*	\param		[in]	v0						- input vertex index 0.
		*	\param		[in]	v1						- input vertex index 1.
		*	\param		[out]	closestPt				- stores closest point if there is intersection bettwen the two ruling edges.
		*	\return				bool					- true if there is a intersection else false.
		*	\since version 0.0.2
		*/
		bool computeRulingIntersection(int polytopalIndex, zItMeshVertex &v0, zItMeshVertex &v1, zPoint &closestPt);

		/*! \brief This method closes the input polytopal mesh.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\since version 0.0.2
		*/
		void getClosePolytopalMesh(int forceIndex);

		/*! \brief This method computes the Y vertex per face of the primal. It is require to compute the frame at each face center.
		*
		*	\param		[out]	Y_vertex				- output container storing indicies of vertex per face.
		*	\since version 0.0.3
		*/
		void get_YVertex(zIntArray &Y_vertex);

		/*! \brief This method computes the user section points centered on world origin.
		*
		*	\param		[in]	face_numEdges			- input number of edges of the face.
		*	\param		[out]	sectionPoints			- output container storing user section points.
		*	\since version 0.0.3
		*/
		void getUserSection(int face_numEdges, zPointArray &sectionPoints);
		

	};


#ifndef DOXYGEN_SHOULD_SKIP_THIS
		
	
	//--------------------------
	//---- REFERENCES FOR MATRIX DECOMPOSITION
	//--------------------------
	//https://scicomp.stackexchange.com/questions/2510/null-space-of-a-rectangular-dense-matrix
	//https://math.stackexchange.com/questions/1720647/is-there-no-difference-between-upper-triangular-matrix-and-echelon-matrixrow-ec
	//http://sites.millersville.edu/bikenaga/linear-algebra/matrix-subspaces/matrix-subspaces.html
	//https://inside.mines.edu/~whoff/courses/EENG512/lectures/17-SVD.pdf
	//http://pillowlab.princeton.edu/teaching/statneuro2018/slides/notes03a_SVDandLinSys.pdf
	//https://math.stackexchange.com/questions/1521264/kernels-and-reduced-row-echelon-form-explanation
	//https://math.stackexchange.com/questions/487629/column-space-and-svd
	//http://flintlib.org/flint-2.4.2.pdf
	//https://math.stackexchange.com/questions/19948/pseudoinverse-matrix-and-svd
	//https://medium.com/@ringlayer/regular-inverse-pseudo-inverse-matrix-calculation-using-singular-value-decomposition-914cedb28e20
	//https://www.johndcook.com/blog/2018/05/05/svd/
	//https://stackoverflow.com/questions/19947772/non-trival-solution-for-ax-0-using-eigen-c
	//https://math.stackexchange.com/questions/206904/how-do-i-solve-ax-0
	//https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c03_p47-84.pdf?source=post_page
	//http://people.maths.ox.ac.uk/nanda/linalg/notes/Lecture%2011.pdf

	//http://www.nealen.com/projects/mls/asapmls.pdf

#endif

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/statics/zTsPolytopal.cpp>
#endif

#endif