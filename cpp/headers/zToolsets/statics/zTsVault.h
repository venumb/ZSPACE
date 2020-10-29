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

#ifndef ZSPACE_TS_STATICS_VAULT_H
#define ZSPACE_TS_STATICS_VAULT_H


#pragma once

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

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

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \class zTsVault
	*	\brief A formfinding tool set class using both iterative and linear algebra methods.
	*	\tparam				T			- Type to work with zObjMesh or zObjGraph.
	*	\tparam				U			- Type to work with zFnMesh or zFnGraph.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	template<typename T, typename U>
	class ZSPACE_TOOLS zTsVault
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- RESULT ATTRIBUTES
		//--------------------------
			

		/*!	\brief container of  result particle objects  */
		vector<zObjParticle> resultParticlesObj;

		/*!	\brief container of result particle function set  */
		vector<zFnParticle> fnResultParticles;

		/*!	\brief container storing the update weights of the result diagram.  */
		vector<float> resultVWeights;

		/*!	\brief container storing the vertex mass of the result diagram.  */
		vector<float> resultVThickness;

		/*!	\brief container storing the vertex mass of the result diagram.  */
		vector<float> resultVMass;

		//--------------------------
		//---- FORM DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief container of  form particle objects  */
		vector<zObjParticle> formParticlesObj;

		/*!	\brief container of form particle function set  */
		vector<zFnParticle> fnFormParticles;

		/*!	\brief container storing the update weights of the form diagram.  */
		vector<float> formVWeights;

		//--------------------------
		//---- FORCE DIAGRAM ATTRIBUTES
		//--------------------------
			

		/*!	\brief container of  force particle objects  */
		vector<zObjParticle> forceParticlesObj;

		/*!	\brief container of force particle function set  */
		vector<zFnParticle> fnForceParticles;

		/*!	\brief container storing the update weights of the force diagram.  */
		vector<float> forceVWeights;

		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*!	\brief container of force densities  */
		vector<float> forceDensities;

		/*!	\brief container of indicies of fixed vertices  */
		vector<int> fixedVertices;		

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> fixedVerticesBoolean;

		/*!	\brief container storing the corresponding force edge per form edge.  */
		vector<int> forceEdge_formEdge;

		/*!	\brief container storing the corresponding form edge per force edge.  */
		vector<int> formEdge_forceEdge;

		/*!	\brief container storing the form edges in tension.  */
		vector<bool> form_tensionEdges;

		/*!	\brief container storing the force edges in tension.  */
		vector<bool> force_tensionEdges;

		/*!	\brief container storing the force edges in tension.  */
		vector<bool> result_tensionEdges;

		/*!	\brief container storing the horizontal equilibrium target for form edges.  */
		vector<zVector> targetEdges_form;

		/*!	\brief container storing the horizontal equilibrium target for force edges.  */
		vector<zVector> targetEdges_force;

		//--------------------------
		//---- FDM CONSTRAINT SOLVE ATTRIBUTES
		//--------------------------

		/*!	\brief container storing the update weights of plan constraint of the result diagram.  */
		zIntArray result_vertex_PlanConstraints;
		zBoolArray result_vertex_PlanConstraintsBoolean;

		// 0 - no pleats, 1 - valey, 2 - ridge
		zFloatArray result_vertex_ValleyRidge;

		zFloatArray result_vertex_PleatDepth;

		zPointArray originalPoints;

		MatrixXd X, X_orig;
		zIntArray freeVertices;

		zIntArray forcedensityEdgeMap;
		int numFreeEdges;

		zSparseMatrix C;

		VectorXd qInitial, qCurrent;

		VectorXd Pn;

		zIntPairArray vertexSymmetryPairs;
		zIntPairArray edgeSymmetryPairs;

	public:
		
		/*!	\brief color domain - min for tension edges and max for compression edges.  */
		zDomainColor elementColorDomain = zDomainColor(zColor(0.5, 0, 0.2, 1), zColor(0, 0.2, 0.5, 1));		

		/*!	\brief pointer to result Object  */
		T *resultObj;

		/*!	\brief pointer to form Object  */
		T *formObj;

		/*!	\brief pointer to force Object  */
		zObjMesh *forceObj;

		/*!	\brief result function set  */
		U fnResult;

		/*!	\brief form function set  */
		U fnForm;

		/*!	\brief force function set  */
		zFnMesh fnForce;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsVault();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_resultObj			- input result object.
		*	\since version 0.0.2
		*/
		zTsVault(T &_resultObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_resultObj			- input result object.
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_forceObj			- input force object.
		*	\since version 0.0.2
		*/
		zTsVault(T &_resultObj, T &_formObj, zObjMesh &_forceObj);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsVault();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the result geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createResultfromFile(string path, zFileTpye type);

		/*! \brief This method creates the force geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createForcefromFile(string path, zFileTpye type);

		/*! \brief This method creates the form geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*	\warning works only with meshes.
		*/
		void createFormfromFile(string path, zFileTpye type);

		/*! \brief This method computes the form diagram from the result.
		*
		*	\since version 0.0.2
		*/
		void createFormFromResult();

		/*! \brief This method computes the form diagram from the force diagram. Works only when form is to a be a graph.
		*
		*	\param		[in]	excludeBoundary			- true if boundary faces are to be ignored.
		*	\param		[in]	PlanarMesh				- true if input mesh is planar.
		*	\param		[in]	rotate90				- true if form diagram is to be roatated by 90.
		*	\since version 0.0.2
		*	\warning works only with graphs.
		*/
		void createFormFromForce(bool excludeBoundary = false, bool PlanarForceMesh = false, bool rotate90 = false);

		/*! \brief This method computes the force diagram from the form diagram. Works only when form is a mesh.
		*
		*	\param		[in]	rotate90				- true if force diagram is to be roatated by 90.
		*	\since version 0.0.2
		*	\warning works only with meshes.
		*/
		void createForceFromForm(bool rotate90);

		/*! \brief This method computes the result from the form.
		*
		*	\since version 0.0.2
		*/
		void createResultFromForm();			

		//--------------------------
		//---- FDM METHODS
		//--------------------------		

		/*! \brief This method computes the result based on the force density method.
		*
		*	\details Based on Schek, H-J. "The force density method for form finding and computation of general networks." Computer methods in applied mechanics and engineering 3.1 (1974): 115-134. (https://www.sciencedirect.com/science/article/pii/0045782574900450)
			and Linkwitz, K. (2014). Force density method. Shell Structures for Architecture: Form Finding and Optimization, Routledge, Oxon and New York, 59-71.
		*	\return				bool								- true if solution is found.
		*	\since version 0.0.2
		*/
		bool forceDensityMethod();

		//--------------------------
		//---- BEST FIT FDM AND CONSTRAINT METHODS
		//--------------------------

		double fdm_constraintsolve(bool &computeQInitial, float alpha, float tolerance , float qLB, float qUB );

		void getSymmetryPairs(zIntPairArray &vPairs, zIntPairArray &ePairs,  zPoint &p_center, zVector &p_norm);

		void boundForceDensities(zIntArray &fdMap, VectorXd &fDensities,float qLB = -10000.0, float qUB = 10000.0);

		void boundGradientForceDensities(VectorXd &grad_fDensities, VectorXd &current_fDensities, float qLB = -10000.0, float qUB = 10000.0);

		bool checkObjectiveAchieved(MatrixXd &currentX, MatrixXd &prevX, float tolerance);

		void updateEquilibriumPositions(VectorXd &fDensities);

		void perturbPleatPositions(MatrixXd &origX);

		void getBestFitForceDensities(VectorXd &bestfit_fDensities);

		void getBestFitForceDensities(zFloatArray &bestfit_fDensities);		

		void getResidual_Gradient(VectorXd & current_fDensities, zVectorArray &targets, VectorXd &residual, VectorXd &gradient_fDensities);

		void getResiduals(float alpha, VectorXd & current_fDensities, VectorXd &residual, VectorXd &residualU, VectorXd &residualC);

		void getGradients(VectorXd & current_fDensities, VectorXd &residualU, VectorXd &residualC, VectorXd &gradPos, VectorXd &gradFDensities);

		void getfreeVertices(zIntArray &freeVerts);

		int getNumFreeEdges(zIntArray &fdMap);

		void getPositionMatrix(MatrixXd &X);

		void getLoadVector(VectorXd &Pn);

		void setConstraint_plan(const zIntArray &vertex_PlanWeights = zIntArray());

		void setConstraint_pleats(zFloatArray &vertex_ValleyRidge, zFloatArray &vertex_PleatDepth);
		
		void getConstraint_Planarity(zVectorArray &targets, float planarityTolerance = EPS);

		//--------------------------
		//---- TNA METHODS
		//--------------------------

		/*! \brief This method computes the horizontal equilibrium of the form and force diagram.
		*
		*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
		*	\param		[in]	computeTargets						- true if the targets for horizontal equiibrium have to be computed.
		*	\param		[in]	formWeight							- weight of form mesh update. To be between 0 and 1.
		*	\param		[in]	dT									- integration timestep.
		*	\param		[in]	type								- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations						- number of iterations to run.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	minMax_formEdge						- minimum value of the target edge for form mesh.
		*	\param		[in]	minMax_forceEdge					- minimum value of the target edge for force mesh.
		*	\param		[in]	printInfo							- prints information of minimum and maximum deviation if true.
		*	\param		[in]	colorEdges							- colors diagram edges based on deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel.
		*	\since version 0.0.2
		*/
		bool equilibriumHorizontal(bool &computeTargets, float formWeight, float dT, zIntergrationType type, int numIterations = 1, float angleTolerance = 0.001, float minMax_formEdge = 0.1, float minMax_forceEdge = 0.1, bool colorEdges = false, bool printInfo = false);

		/*! \brief This method computes the vertical equilibrium of the result diagram using linear algebra method.
		*
		*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
		*	\param		[in]	computeForceDensitities				- true if the force densitities from the diagrams have to be computed.
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\since version 0.0.2
		*/
		bool equilibriumVertical(bool &computeForceDensitities, float forceDiagramScale);

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the element color domain. (min - tensionColor , max - compressionColor)
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\since version 0.0.2
		*/
		void setElementColorDomain(zDomainColor &colDomain);
		
		/*! \brief This method sets anchor points of the geometry.
		*
		*	\param		[in]	type						- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_constraintVertices			- container of vertex indices to be fixed. If empty the boundary(mesh) or valence 1(graph) vertices are fixed.
		*	\since version 0.0.2
		*/
		void setConstraints(zDiagramType type = zResultDiagram, const vector<int> &_constraintVertices = vector<int>());

		/*! \brief This method appends the vertex indicies in the container as anchor points of the geometry.
		*
		*	\param		[in]	_constraintVertices			- container of vertex indices to be fixed.
		*	\since version 0.0.2
		*/
		void appendConstraints(vector<int> &_constraintVertices);

		/*! \brief This method sets force density of all the edges to the input value.
		*
		*	\param		[in]	fDensity			- force density value.
		*	\since version 0.0.2
		*/
		void setForceDensity(float fDensity);

		/*! \brief This method sets force density of the edges with the input container of values.
		*
		*	\param		[in]	fDensities			- container of force density values.
		*	\since version 0.0.2
		*/
		void setForceDensities(vector<float> &fDensities);

		/*! \brief This method sets the force densities edges based on form and force diagrams.
		*
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\since version 0.0.2
		*/
		void setForceDensitiesFromDiagrams(float forceDiagramScale , bool negate = false);

		/*! \brief This method sets tension edges of the form diagram.
		*
		*	\param		[in]	type					- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_tensionEdges			- container of edge indices to be made tension. If empty all edges are non tension.
		*	\since version 0.0.2
		*/
		void setTensionEdges(zDiagramType type = zResultDiagram, const vector<int> &_tensionEdges = vector<int>());

		/*! \brief This method appends the edges in the container as tension edge of the geometry.
		*
		*	\param		[in]	type					- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_tensionEdges			- container of edge indices to be made tension.
		*	\since version 0.0.2
		*/
		void appendTensionEdges(zDiagramType type, vector<int> &_tensionEdges);

		/*! \brief This method sets the force tension edges based on form tension edges.
		*
		*	\since version 0.0.2
		*/
		void setForceTensionEdgesfromForm();

		/*! \brief This method sets the result tension edges based on form tension edges.
		*
		*	\since version 0.0.2
		*/
		void setResultTensionEdgesfromForceDensities();
		
		/*! \brief This method sets the result tension edges based on form tension edges.
		*
		*	\since version 0.0.2
		*/
		void setResultTensionEdgesfromForm();

		/*! \brief This method sets the form tension edges based on result tension edges.
		*
		*	\since version 0.0.2
		*/
		void setFormTensionEdgesfromResult();

		/*! \brief This method sets the form tension edges based on force tension edges.
		*
		*	\since version 0.0.2
		*/
		void setFormTensionEdgesfromForce();		

		/*! \brief This method sets the compression or tension colors to the edges of the input diagram.
		*
		*	\param		[in]	type					- diagram type . works with zFormDiagram/ zForceDiagram/ zResultDiagram.
		*	\since version 0.0.2
		*/
		void setElementColorDomain(zDiagramType type);	

		/*! \brief This method sets the vertex update weights for each vertex of the input diagram type.
		*
		*	\param		[in]	type							- zFormDiagram or zForceDiagram or zResultDiagram.
		*	\since version 0.0.2
		*/
		void setVertexWeights(zDiagramType type, const vector<float> &vWeights = vector<float>());

		/*! \brief This method sets the result vertex update weights for each vertex of the input diagram type based on the constraints.
		*
		*	\since version 0.0.2
		*/
		void setVertexWeightsfromConstraints(zDiagramType type = zResultDiagram);

		/*! \brief This method sets thickness of all the result vertices to the input value.
		*
		*	\param		[in]	thickness			- thickness value.
		*	\since version 0.0.2
		*/
		void setVertexThickness(float thickness);

		/*! \brief This method sets thickness of all the result vertices to the input container of values.
		*
		*	\param		[in]	thickness			- container of thickness values.
		*	\since version 0.0.2
		*/
		void setVertexThickness(vector<float> &thickness);

		/*! \brief This method sets vertex mass of all the result vertices to the input value.
		*
		*	\param		[in]	mass			- mass value.
		*	\since version 0.0.2
		*/
		void setVertexMass(float mass);

		/*! \brief This method sets vertex mass of all the result vertices to the input container of values.
		*
		*	\param		[in]	mass			- container of mass values.
		*	\since version 0.0.2
		*/
		void setVertexMass(vector<float> &mass);

		/*! \brief This method sets vertex mass of all the result vertices based on vertex tributary area. Works only on mesh result diagram.
		*
		*	\since version 0.0.2
		*	\warning works only for meshes.
		*/
		void setVertexMassfromVertexArea();		

		//--------------------------
		//--- GET METHODS 
		//--------------------------		

		/*! \brief This method gets the corresponding force diagram halfedge for the input form diagram indexed halfedge.
		*
		*	\param		[in]	formEdgeindex			- form diagram halfedge index.
		*	\param		[out]	outForceEdge			- force diagram halfedge iterator if edge exists.
		*	\return				bool					- true if correponding force halfedge exists, else false.
		*	\since version 0.0.2
		*/
		bool getCorrespondingForceEdge(int formEdgeindex, zItMeshHalfEdge &outForceEdge);

		/*! \brief This method gets the corresponding form diagram halfedge for the input force diagram indexed halfedge.
		*
		*	\param		[in]	forceEdgeindex			- force diagram halfedge index.
		*	\param		[out]	outFormEdge				- form diagram halfedge iterator if edge exists.
		*	\return				bool					- true correponding form halfedge exists, else false.
		*	\since version 0.0.2
		*/
		bool getCorrespondingFormEdge(int forceEdgeindex, zItMeshHalfEdge &outFormEdge);

		/*! \brief This method gets the horizontal equilibrium target vectors for the input diagram type.
		*
		*	\param		[in]	type					- diagram type . works with zFormDiagram and zForceDiagram.
		*	\return				targets					- container of tearget vectors per half edge.
		*	\since version 0.0.2
		*/
		void getHorizontalTargetEdges(zDiagramType type, vector<zVector> &targets);

		/*! \brief This method gets the force vectors for the result diagram using gradient descent.
		*
		*	\param		[out]	forces					- container of forces per vertex.
		*	\since version 0.0.2
		*/
		void getForces_GradientDescent(vector<zVector> &forces);

		zIntArray getConstraints();
	
		//--------------------------
		//---- UTILITY METHODS 
		//--------------------------

		void translateForceDiagram(float value);

	protected:		

		/*! \brief This method computes the Edge Node Matrix for the input mesh.
		*
		*	\param		[in]	numCols								- number of columns in the out matrix.
		*	\return				zSparseMatrix						- edge node matrix.
		*	\since version 0.0.2
		*/		
		zSparseMatrix getEdgeNodeMatrix(int numCols);

		/*! \brief This method computes the sub Matrix of a sparse matrix.
		*
		*	\param		[in]	C									- input sparse matrix.
		*	\param		[in]	nodes								- container of integers.
		*	\return				zSparseMatrix								- sub matrix.
		*	\since version 0.0.2
		*/
		zSparseMatrix subMatrix(zSparseMatrix &C, vector<int> &nodes);

		/*! \brief This method computes the sub Matrix of a matrix.
		*
		*	\param		[in]	C									- input sparse matrix.
		*	\param		[in]	nodes								- container of integers.
		*	\return				MatrixXd							- sub matrix.
		*	\since version 0.0.2
		*/
		MatrixXd subMatrix(MatrixXd &X, vector<int> &nodes);

		/*! \brief This method computes the targets for the form and force edges.
		*
		*	\param		[in]	formWeight							- weight of form diagram update. To be between 0 and 1.
		*	\since version 0.0.2
		*/
		void getHorizontalTargets(float formWeight);

		/*! \brief This method if the form mesh edges and corresponding force mesh edge are parallel.
		*
		*	\param		[out]	deviation							- deviation domain.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel or within tolerance.
		*	\since version 0.0.2
		*/
		bool checkHorizontalParallelity(zDomainFloat &deviation, float angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false);

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updateFormDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations = 1);

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updateForceDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations = 1);


		void getPleatDataJSON(string infilename);
	};
	   
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \typedef zTsMeshVault
	*	\brief A vault object for meshes.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsVault<zObjMesh, zFnMesh> zTsMeshVault;

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \typedef zTsGraphVault
	*	\brief A vault object for graphs.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsVault<zObjGraph, zFnGraph> zTsGraphVault;


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/statics/zTsVault.cpp>
#endif

#endif
