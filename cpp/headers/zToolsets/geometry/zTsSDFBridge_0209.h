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

#ifndef ZSPACE_TS_GEOMETRY_SDFBRIDGE_H
#define ZSPACE_TS_GEOMETRY_SDFBRIDGE_H

#pragma once

#include <headers/zCore/base/zExtern.h>

#include <headers/zInterface/functionsets/zFnMesh.h>
#include <headers/zInterface/functionsets/zFnGraph.h>
#include <headers/zInterface/functionsets/zFnParticle.h>

#include <headers/zInterface/functionsets/zFnMeshField.h>

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

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \struct zGlobalVertex
	*	\brief A struct for to hold global vertex attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zGlobalVertex
	{
		zPoint pos;
		
		/*!	\brief container of conincedent vertex indicies  */
		zIntArray coincidentVertices;

	};
	

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \struct zBlock
	*	\brief A struct for to hold block attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zBlock
	{
		/*!	\brief block index  */
		int id = -1;

		/*!	\brief length of block medial graph  */
		float mg_Length = 0;

		/*!	\brief container of block face indicies  */
		zIntArray faces;

		///*!	\brief container of block indicies which make up a macro block  */
		//zIntArray macroBlocks;

		///*!	\brief container of block indicies which make up a macro block  */
		//zIntArray macroBlock_BoundaryFaces;

		//zIntArray macroBlock_sideFaces;

		///*!	\brief container of int pairs for holding plane face Id with correponding edge id intersecting it  */
		//zIntPairArray sectionPlaneFace_GuideSmoothEdge;

		///*!	\brief container of intersection points of plane and mesh  */
		//zPointArray intersectionPoints;

		///*!	\brief container of section graph objects  */
		//zObjGraphArray o_sectionGraphs;

		///*!	\brief container of contour graph objects  */
		//zObjGraphArray o_contourGraphs;
		//
		///*!	\brief medial graph of the block  */
		//zGraph medialGraph;

		///*!	\brief container of section frames  */
		//vector<zTransform> sectionFrames;

	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \struct zBlock
	*	\brief A struct for to hold block attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zMacroBlock
	{
		/*!	\brief block index  */
		int id = -1;

		/*!	\brief length of block medial graph  */
		float mg_Length = 0;
				
		

		/*!	\brief container of block indicies which make up a macro block  */
		vector< zBlock> leftBlocks;
		vector< zBlock> rightBlocks;

		/*!	\brief container of block indicies which make up a macro block  */
		zIntArray left_BoundaryFaces;
		zIntArray right_BoundaryFaces;

		zIntArray left_sideFaces;
		zIntArray right_sideFaces;

		/*!	\brief container of int pairs for holding plane face Id with correponding edge id intersecting it  */
		zIntPairArray sectionPlaneFace_GuideSmoothEdge;

		/*!	\brief container of intersection points of plane and mesh  */
		zPointArray intersectionPoints;

		/*!	\brief container of section graph objects  */
		zObjGraphArray o_sectionGraphs;

		/*!	\brief container of contour graph objects  */
		zObjGraphArray o_contourGraphs;

		/*!	\brief medial graph of the block  */
		zGraph medialGraph;

		/*!	\brief container of section frames  */
		vector<zTransform> sectionFrames;

	};
	
	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	
	/*! \class zTsSDFBridge
	*	\brief A toolset for 3D graphics and poytopal meshes.
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf) and Freeform Developable Spatial Structures (https://www.ingentaconnect.com/content/iass/piass/2016/00002016/00000003/art00010 )
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsSDFBridge
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to input guide mesh Object  */
		zObjMesh *o_guideMesh;

		/*!	\brief pointer to input guide mesh Object  */
		zObjMesh o_guideSmoothMesh;

		/*!	\brief cut plane mesh Object  */
		zObjMesh o_planeMesh;

		/*!	\brief map between a coincident plane vertex and global vertex  */
		zIntArray planeVertex_globalVertex;

		/*!	\brief map between a guide edge to global vertex  */
		vector<zIntArray> guideVertex_globalVertex;

		/*!	\brief map between a guide edge to global vertex  */
		vector<zIntArray> guideEdge_globalVertex;

		/*!	\brief map between a guide face to global vertex  */
		vector<zIntArray> guideFace_globalVertex;

		/*!	\brief map between a guide vertex to plane face  */
		zIntArray guideVertex_planeFace;

		/*!	\brief map between a guide half edge to plane face  */
		zInt2DArray guideHalfEdge_planeFace;

		/*!	\brief container of global vertices  */
		vector<zGlobalVertex> globalVertices;

		/*!	\brief container of blocks  */
		//vector<zBlock> blocks;

		vector<zMacroBlock> macroblocks;
				
		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*!	\brief container of indicies of fixed vertices  */
		vector<int> fixedVertices;

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> fixedVerticesBoolean;


		/*!	\brief container of indicies of fixed vertices  */
		vector<int> printMedialVertices;

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> printMedialVerticesBoolean;

		/*!	\brief container of booleans of faces excluded in the cut plane creation */
		vector<bool> excludeFacesBoolean;

		/*!	\brief container storing the equilibrium target for guide half edges.  */
		vector<zVector> targetHalfEdges_guide;

		/*!	\brief container storing the equilibrium target for cut plane normals.  */
		vector<zVector> targetNormals_cutplane;

		/*!	\brief container of indicies of medial edges of guide mesh  */
		zIntPairArray guide_MedialEdges;

		/*!	\brief container of booleans of medial edges of guide mesh */
		zBoolArray guide_MedialEdgesBoolean;

		/*!	\brief container of indicies of medial edges of smooth guide mesh  */
		zIntPairArray guideSmooth_MedialEdges;

		/*!	\brief container of booleans of medial edges of smooth guide mesh  */
		zBoolArray guideSmooth_MedialEdgesBoolean;


		/*!	\brief container of indicies of Print medial edges of guide mesh  */
		zIntPairArray guide_PrintMedialEdges;

		/*!	\brief container of booleans of Print medial edges of guide mesh */
		zBoolArray guide_PrintMedialEdgesBoolean;

		/*!	\brief container of indicies of Print medial edges of smooth guide mesh  */
		zIntPairArray guideSmooth_PrintMedialEdges;

		/*!	\brief container of booleans of Print medial edges of smooth guide mesh  */
		zBoolArray guideSmooth_PrintMedialEdgesBoolean;

		//--------------------------
		//---- GUIDE MESH ATTRIBUTES
		//--------------------------

		/*!	\brief container of  guide particle objects  */
		vector<zObjParticle> o_guideParticles;

		/*!	\brief container of guide particle function set  */
		vector<zFnParticle> fnGuideParticles;

		zBoolArray skeletalGraphVertex;

		/*!	\brief container storing the update weights of the guide mesh.  */
		zFloatArray guideVWeights;

		//--------------------------
		//---- PLANE MESH ATTRIBUTES
		//--------------------------

		/*!	\brief container of  plane particle objects  */
		vector<zObjParticle> o_planeParticles;

		/*!	\brief container of plane particle function set  */
		vector<zFnParticle> fnPlaneParticles;

		/*!	\brief container storing the update weights of the plane mesh.  */
		zFloatArray planeVWeights;

		zIntArray planeFace_targetPair;

		zColorArray planeFace_colors;

		//--------------------------
		//---- SDF ATTRIBUTES
		//--------------------------

		zObjMeshScalarField o_field;
		zObjGraph o_isoContour;

		zObjMesh o_isoMesh;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.4
		*/
		zTsSDFBridge();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.4
		*/
		~zTsSDFBridge();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param		[in]	width					- input plane width.
		*	\since version 0.0.4
		*/
		void createCutPlaneMesh(double width);

		void createSplitMesh(double width);

		/*! \brief This method creates the smooth mesh of guide mesh.
		*
		*	\param		[in]	subdiv			- input smooth subdivisions.
		*	\since version 0.0.4
		*/
		void createSmoothGuideMesh(int subdiv = 2);
	
		/*! \brief This method creates the smooth mesh of guide mesh from file.
		*
		*	\param		[in]	path			- input file path
		*	\param		[in]	fileType		- input file type
		*	\since version 0.0.4
		*/
		void createSmoothGuideMeshfromFile(string path, zFileTpye fileType);

		/*! \brief This method creates the dual mesh of guide mesh.
		*
		*	\param		[in]	width			- input plane width.
		*	\since version 0.0.4
		*/
		void createDualPlaneMesh(double width);

		/*! \brief This method creates the block section graphs from the input files.
		*
		* 	\param		[in]	fileDir			- input file directory.
		*	\param		[in]	type			- input file type : zJSON.
		*	\since version 0.0.4
		*/
		void createBlockSectionGraphsfromFiles(string fileDir , zFileTpye type);


		/*! \brief This method creates the filed mesh.
		*
		*	\param		[in]	bb			- input domain of bounds.
		*	\param		[in]	res			- input resolution of field in both X and Y .
		*	\since version 0.0.4
		*/
		void createFieldMesh(zDomain<zPoint> &bb,  int res);

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------

		/*! \brief This method updates the form diagram to find equilibrium shape..
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	guideWeight					- weight of guide mesh update. To be between 0 and 1.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations.
		*	\param		[in]	angleTolerance				- tolerance for angle.
		*	\param		[in]	colorEdges					- colors edges based on angle deviation if true.
		*	\param		[in]	printInfo					- prints angle deviation info if true.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.4
		*/
		bool equilibrium(bool& computeTargets, float guideWeight, double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = EPS, bool colorEdges = false, bool printInfo = false);


		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the guide mesh object.
		*
		*	\param		[in]	_o_guideMesh			- input guide mesh object.
		*	\since version 0.0.4
		*/
		void setGuideMesh(zObjMesh& _o_guideMesh);
				
		/*! \brief This method sets anchor points of the geometry.
		*
		*	\param		[in]	_fixedVertices			- container of vertex indices to be fixed. If empty the boundary(mesh) or valence 1(graph) vertices are fixed.
		*	\since version 0.0.4
		*/
		void setConstraints( const vector<int>& _fixedVertices = vector<int>());

		/*! \brief This method sets anchor points of the geometry.
		*
		*	\param		[in]	_fixedVertices			- container of vertex indices to be fixed. If empty the boundary(mesh) or valence 1(graph) vertices are fixed.
		*	\since version 0.0.4
		*/
		void setPrintMedials(const vector<int>& _fixedVertices = vector<int>());

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets pointer to the internal combined voxel mesh object.
		*
		*	\return				zObjMesh*					- pointer to internal cutplane meshobject.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawCutPlaneMesh();
		
		/*! \brief This method gets pointer to the internal combined voxel mesh object.
		*
		*	\return				zObjMesh*					- pointer to internal smooth guideMesh object.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawSmoothGuideMesh();

		/*! \brief This method computes the targets for equilibrium.
		*
		*	\param		[in]	guideWeight					- weight of guideMesh update. To be between 0 and 1.
		*	\since version 0.0.2
		*/
		int getCorrespondingPlaneFace(int guideEdgeId);

		/*! \brief This method computes the block intersection points of the block with smooth guide mesh.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zPointArray			    - intersections if they exist.
		*	\since version 0.0.2
		*/
		zPointArray getBlockIntersectionPoints(int blockId);

		/*! \brief This method computes the block intersection points of the block with smooth guide mesh.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zPointArray			    - intersections if they exist.
		*	\since version 0.0.2
		*/
		vector<zTransform> getBlockFrames(int blockId);

		zObjGraphPointerArray getBlockGraphs(int blockId, int &numGraphs);

		zObjGraphPointerArray getBlockContourGraphs(int blockId, int& numGraphs);

		/*! \brief This method gets pointer to the internal field object.
		*
		*	\return				zObjMeshScalarField*					- pointer to internal field object.
		*	\since version 0.0.4
		*/
		zObjMeshScalarField* getRawFieldMesh();

		/*! \brief This method gets pointer to the internal field object.
		*
		*	\return				zObjMeshScalarField*					- pointer to internal field object.
		*	\since version 0.0.4
		*/
		zObjMesh* getRawIsocontour();

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------

		/*! \brief This method computes the targets for equilibrium.
		*
		*	\param		[in]	guideWeight					- weight of guideMesh update. To be between 0 and 1.
		*	\since version 0.0.4
		*/
		void computeEquilibriumTargets(float guideWeight);

		/*! \brief This method computes the medial edges from contraint points.
		*
		*	\since version 0.0.4
		*/
		void computeMedialEdgesfromConstraints();

		/*! \brief This method computes the medial edges from contraint points.
		*
		*	\since version 0.0.4
		*/
		void computePrintMedialEdgesfromMedialVertices();

		/*! \brief This method computes the medial edges from contraint points.
		*
		* 	\param		[in]	printLayerDepth				- input print layer depth.
		*	\since version 0.0.4
		*/
		void computeBlocks(float printLayerDepth =  0.1);

		/*! \brief This method computes the block mesh.
		*
		* 	\param		[in]	blockId				- input block index.
		*	\since version 0.0.4
		*/
		void computeBlockMesh(int blockId);

		/*! \brief This method computes the SDF for the blocks.
		*
		*	\since version 0.0.4
		*/
		void computeSDF();

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method add the convex hulls which make up the macro block.
		*
		*	\param		[in]	mBlock						- input macro block.
		*	\param		[in]	guideMesh_halfedge			- input guideMesh halfedge.
		*	\since version 0.0.4
		*/
		void addConvexBlocks(zMacroBlock &mBlock,zItMeshHalfEdge& guideMesh_halfedge);
		
		
		/*! \brief This method updates the guide mesh.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updateGuideMesh(float minmax_Edge, float dT, zIntergrationType type, int numIterations = 1);

		/*! \brief This method updates the plane mesh.
		*
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updatePlaneMesh( float dT, zIntergrationType type, int numIterations = 1);

		/*! \brief This method if the form mesh edges and corresponding force mesh edge are parallel.
		*
		*	\param		[out]	deviation							- deviation domain.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel or within tolerance.
		*	\since version 0.0.4
		*/
		bool checkDeviation(zDomainFloat& deviation, float angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false);

		/*! \brief This method updates the plane mesh.
		*
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.4
		*/
		bool planarisePlaneMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1);

		/*! \brief This method compute the block faces.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computeBlockFaces_fromVertex(zBlock &_block, zItMeshVertex& guideMesh_vertex);

		void computeBlockFaces_fromEdge(zBlock& _block, zItMeshEdge& guideMesh_edge);

		/*! \brief This method compute the block indices which make up a macro block.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computeMacroBlockIndices(zBlock& _block, zItMeshEdge& guideMesh_edge);

		void  computeMacroBlock();

		/*! \brief This method compute the block medial graph.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computeBlockIntersections(zBlock& _block, zItMeshVertex& guideMesh_vertex);

		void computeBlockIntersections(zBlock& _block, zItMeshEdge& guideMesh_edge);

		/*! \brief This method compute the block frames.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	printLayerDepth				- input print layer depth.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computeBlockFrames(zBlock& _block, float printLayerDepth);

		/*! \brief This method compute the block SDF for the balustrade.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	graphId						- input index of section graph.
		*	\since version 0.0.4
		*/
		void computeBalustradeSDF(zBlock& _block, int graphId);

		/*! \brief This method compute the transform from input Vectors.
		*
		*	\param		[in]	O							- input origin point.
		*	\param		[in]	X							- input X axis vector.
		* 	\param		[in]	Y							- input Y axis vector.
		*	\param		[in]	Z							- input Z axis vector.
		*	\return				zTransform					- output transform.
		*	\since version 0.0.4
		*/
		zTransform setTransformFromVectors(zPoint &O, zVector &X , zVector& Y, zVector& Z);
	

		void blockPlanesToTXT(string dir, string filename);

		void blockSidePlanesToTXT(string dir, string filename);

		void blockSectionPlanesToTXT(string dir, string filename);

		void blockSectionsFromJSON(string dir, string filename);

		void blockContoursToJSON(string dir, string filename);

		void blockContoursToIncr3D(string dir, string filename, float layerWidth = 0.025);

		void addBlocktoMacro(zBlock& _blockA, zBlock& _blockB);

		void toBRGJSON(string path, zPointArray& points, zVectorArray& normals);
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsSDFBridge.cpp>
#endif

#endif