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

	/*! \struct zConvexBlock
	*	\brief A struct for to hold convex hull block attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zConvexBlock
	{
		/*!	\brief block index  */
		int id = -1;

		/*!	\brief length of block medial graph  */
		float mg_Length = 0;

		/*!	\brief container of block face indicies  */
		zIntArray faces;
	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \struct zConvexBlock
	*	\brief A struct for to hold block attributes.
	*	\details
	*	\since version 0.0.4
	*/

	/** @}*/

	/** @}*/

	struct zPrintBlock
	{
		/*!	\brief block index  */
		int id = -1;

		/*!	\brief length of medial graph length  */
		float mg_Length = 0;

		/*!	\brief angle between start-end plane of of left convex block faces  */
		float left_planeAngle;

		/*!	\brief angle between start-end plane of of left convex block faces  */
		float right_planeAngle;

		/*!	\brief color of the block from guide mesh  */
		zColor color;
				
		/*!	\brief guideMesh interior vertex of the block  */
		int guideMesh_interiorVertex;

		/*!	\brief guideMesh interior vertex position of the block  */
		zPoint guideMesh_interiorVertexPosition;
				
		/*!	\brief guideMesh interior vertex of the block  */
		int guideMesh_right_railInteriorVertex;

		/*!	\brief guideMesh interior vertex of the block  */
		int guideMesh_left_railInteriorVertex;

		/*!	\brief container of left convex block indicies which make up the print block  */
		vector< zConvexBlock> leftBlocks;

		/*!	\brief container of right convex block indicies which make up the print block  */
		vector< zConvexBlock> rightBlocks;

		/*!	\brief container of left block start-end face indicies  */
		zIntArray left_BoundaryFaces;

		/*!	\brief container of right block start-end face indicies  */
		zIntArray right_BoundaryFaces;

		/*!	\brief container of left block side trim face indicies  */
		zIntArray left_sideFaces;

		/*!	\brief container of right  block side trim face indicies  */
		zIntArray right_sideFaces;

		/*!	\brief container of int pairs for holding plane face Id with correponding edge id intersecting it  for left convex block */
		zIntPairArray right_sectionPlaneFace_GuideSmoothEdge;
		
		/*!	\brief container of int pairs for holding plane face Id with correponding edge id intersecting it  for right convex block */
		zIntPairArray left_sectionPlaneFace_GuideSmoothEdge;

		/*!	\brief container of intersection points of plane and start-end faces  */
		zPointArray intersectionPoints;

		/*!	\brief container of intersection points of plane and start-end faces  */
		zPointArray printStartEndPoints;

		/*!	\brief container of section graph objects  */
		zObjGraphArray o_sectionGraphs;

		/*!	\brief container of section graph objects  */
		zObjGraphArray o_sectionGuideGraphs;

		/*!	\brief container of contour graph objects  */
		zObjGraphArray o_trimGraphs;

		/*!	\brief container of contour graph objects  */
		zObjGraphArray o_contourGraphs;

		/*!	\brief container of raft graph objects  */
		zObjGraphArray o_raftGraphs;

		/*!	\brief medial graph of the block  */
		zGraph medialGraph;

		/*!	\brief container of section frames for both right and left blocks  */
		vector<zTransform> sectionFrames;

		/*!	\brief container of section trims faces index (array index) */
		zIntPairArray sectionTrimFaces;

		/*!	\brief container of section graphs start vertex  */
		zIntPairArray sectionGraphs_startEndVertex;

		zPointArray startPos;
		zPointArray endPos;

		float minLayerHeight  = 0;

		float maxLayerHeight = 0 ;

		float totalLength = 0;

		bool footingBlock = false;
	
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

		/*!	\brief pointer to input thickened guide mesh Object  */
		zObjMesh *o_guideThickMesh;

		/*!	\brief pointer to input thickened guide mesh Object  */
		zObjMesh* o_guideSmoothThickMesh;

		/*!	\brief pointer to input guide mesh Object  */
		zObjMesh* o_guideMesh;

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
		//vector<zConvexBlock> blocks;

	
				
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

		/*!	\brief container of indicies of fixed vertices  */
		vector<int> printRailVertices;

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> printRailVerticesBoolean;


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

		/*!	\brief container of indicies of Print rail edges of guide mesh  */
		zIntPairArray guide_PrintRailEdges;

		/*!	\brief container of booleans of Print rail edges of guide mesh */
		zBoolArray guide_PrintRailEdgesBoolean;

		/*!	\brief container of indicies of Print rail edges of smooth guide mesh  */
		zIntPairArray guideSmooth_PrintRailEdges;

		/*!	\brief container of booleans of Print rail edges of smooth guide mesh  */
		zBoolArray guideSmooth_PrintRailEdgesBoolean;

		//--------------------------
		//---- GUIDE MESH ATTRIBUTES
		//--------------------------

		/*!	\brief container of  guide particle objects  */
		vector<zObjParticle> o_guideParticles;

		/*!	\brief container of guide particle function set  */
		vector<zFnParticle> fnGuideParticles;

		zPointArray orig_GuideMeshPositions;
		
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

		zVectorArray targetNormals;
		zPointArray targetCenters;

		zColorArray planeFace_colors;

		zBoolArray globalFixedVertices;

		//--------------------------
		//---- SMOOTH MESH ATTRIBUTES
		//--------------------------

		/*!	\brief container of  plane particle objects  */
		vector<zObjParticle> o_smoothMeshParticles;

		/*!	\brief container of plane particle function set  */
		vector<zFnParticle> fnSmoothMeshParticles;

		//--------------------------
		//---- SDF ATTRIBUTES
		//--------------------------

		zObjMeshScalarField o_field;
		zObjGraph o_isoContour;

		zObjMesh o_isoMesh;

		//--------------------------
		//---- COLOR ATTRIBUTES
		//--------------------------
		
		zColor red, yellow, green, cyan, blue, magenta, grey , orange;

		zColorArray blockColors;

	public:

		vector<zPrintBlock> printBlocks;

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
		void createEdgeBeamMesh(double width, const vector<int>& _constraintVertices = vector<int>());

		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param		[in]	width					- input plane width.
		*	\since version 0.0.4
		*/
		void createSplitMesh(double width, bool useThickMeshPoints = false);

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
		*	\param		[in]	resX		- input resolution of field in X.
		*	\param		[in]	resY		- input resolution of field in Y.
		*	\since version 0.0.4
		*/
		void createFieldMesh(zDomain<zPoint> &bb,  int resX , int resY);

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the guide mesh object.
		*
		*	\param		[in]	_o_guideMesh			- input guide mesh object.
		*	\since version 0.0.4
		*/
		void setGuideMesh(zObjMesh& _o_guideMesh);

		/*! \brief This method sets the guide mesh object.
		*
		*	\param		[in]	_o_guideThickMesh			- input thickened guide mesh object.
		*	\since version 0.0.4
		*/
		void setThickGuideMesh(zObjMesh& _o_guideThickMesh);

		/*! \brief This method sets the guide mesh object.
		*
		*	\param		[in]	_o_guideThickMesh			- input thickened guide mesh object.
		*	\since version 0.0.4
		*/
		void setSmoothThickGuideMesh(zObjMesh& _o_guideSmoothThickMesh);
				
		/*! \brief This method sets convex blocks medial start points of the geometry.
		*
		*	\param		[in]	_fixedVertices			- container of vertex indices
		*	\since version 0.0.4
		*/
		void setConvexMedials( const vector<int>& _fixedVertices = vector<int>());

		/*! \brief This method sets print blocks medial start points of the geometry.
		*
		*	\param		[in]	_fixedVertices			- container of vertex indices
		*	\since version 0.0.4
		*/
		void setPrintMedials(const vector<int>& _fixedVertices = vector<int>());

		/*! \brief This method sets print blocks medial start points of the geometry.
		*
		*	\param		[in]	_fixedVertices			- container of vertex indices
		*	\since version 0.0.4
		*/
		void setPrintRails(const vector<int>& _fixedVertices = vector<int>());

		void resetPlaneColors();

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the number of print blocks.
		*
		*	\return				int					- number of print blocks.
		*	\since version 0.0.4
		*/
		int numPrintBlocks();

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
		
		/*! \brief This method gets the block interior point of the block.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zPointArray			    - interior points.
		*	\since version 0.0.2
		*/
		zPoint getBlockInteriorPoint(int blockId);

		/*! \brief This method gets the block intersection points of the block with smooth guide mesh.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zPointArray			    - container of intersection points if they exist.
		*	\since version 0.0.2
		*/
		zPointArray getBlockIntersectionPoints(int blockId);

		/*! \brief This method gets the block intersection points of the block with smooth guide mesh.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zPointArray			    - container of intersection points if they exist.
		*	\since version 0.0.2
		*/
		zPointArray getBlockPrintStartEndPoints(int blockId);

		/*! \brief This method gets the block frames.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				vector<zTransform>	    - cantainer of transforms if they exist.
		*	\since version 0.0.2
		*/
		vector<zTransform> getBlockFrames(int blockId);

		/*! \brief This method gets the block section graphs
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zObjGraphPointerArray	-  pointer conatiner of graphs if they exist.
		*	\since version 0.0.2
		*/
		zObjGraphPointerArray getBlockGraphs(int blockId, int &numGraphs , zPointArray &startPoints,  zPointArray &endPoints);

		/*! \brief This method gets the block section graphs
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zObjGraphPointerArray	-  pointer conatiner of graphs if they exist.
		*	\since version 0.0.2
		*/
		zObjGraphPointerArray getBlockRaftGraphs(int blockId, int& numGraphs);


		/*! \brief This method gets the block section guide graphs
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zObjGraphPointerArray	-  pointer conatiner of graphs if they exist.
		*	\since version 0.0.2
		*/
		zObjGraphPointerArray getBlockGuideGraphs(int blockId, int& numGraphs);

		/*! \brief This method gets the block SDF contour graphs
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				zObjGraphPointerArray	- pointer conatiner of graphs if they exist.
		*	\since version 0.0.2
		*/
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

		/*! \brief This method gets pointer to the internal field object.
		*
		*	\param		[in]	blockId					- input block index.
		*	\return				bool					- true if boundary block else false.
		*	\since version 0.0.4
		*/
		bool onBoundaryBlock(int blockId);

		/*! \brief This method computes the medial edges from contraint points.
		*
		*	\since version 0.0.4
		*/
		void computeMedialEdgesfromConstraints(const vector<int>& pattern);

		/*! \brief This method computes the medial edges from contraint points.
		*
		*	\since version 0.0.4
		*/
		void computePrintMedialEdgesfromMedialVertices();

		/*! \brief This method computes the medial edges from contraint points.
		*
		*	\since version 0.0.4
		*/
		void computePrintRailEdgesfromRailVertices();

		/*! \brief This method computes the.
		*
		* 	\param		[in]	printLayerDepth				- input print layer depth.
		*	\since version 0.0.4
		*/
		void computePrintBlocks(int blockId, float printLayerDepth, float printLayerWidth , zDomainFloat neopreneOffset, float raftLayerWidth, const zIntArray &flipBlockIds = zIntArray(), bool compBLOCKS = true, bool compFrames = true, bool compSDF = true);

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
		void computeSDF(zPrintBlock& _block, float printWidth, float neopreneOffset, float raftWidth);

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method planarises the plane mesh.
		*
		*	\param		[in]	dT							- input integration timestep.
		*	\param		[in]	type						- input integration type - zEuler or zRK4.
		* 	\param		[in]	tolerance					- input deviation tolerance.
		*	\param		[in]	numIterations				- input number of iterations to run.
		* 	\param		[in]	printInfo					- input boolean indicating print of information of deviation.
		*  	\param		[in]	minEdgeConstraint			- input boolean indicating minimum edge length constraint.
		*  	\param		[in]	minEdgeLen					- input minimum edge length constraint.
		* 	\return				bool						- output boolean , true if  deviation below tolerance.
		*	\since version 0.0.4
		*/
		bool planarisePlaneMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1, bool printInfo =false, bool minEdgeConstraint = false, float minEdgeLen = 0.05);

		/*! \brief This method aligns the normals of the face pairs of the plane mesh.
		*
		*	\param		[in]	dT							- input integration timestep.
		*	\param		[in]	type						- input integration type - zEuler or zRK4.
		* 	\param		[in]	tolerance					- input deviation tolerance.
		*	\param		[in]	numIterations				- input number of iterations to run.
		* 	\param		[in]	printInfo					- input boolean indicating print of information of deviation.
		*  	\param		[in]	minEdgeConstraint			- input boolean indicating minimum edge length constraint.
		*  	\param		[in]	minEdgeLen					- input minimum edge length constraint.
		* 	\return				bool						- output boolean , true if  deviation below tolerance.
		*	\since version 0.0.4
		*/
		bool alignFacePairs(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1, bool printInfo = false);


		/*! \brief This method aligns the normal of the blanes to input targets.
		*
		*	\param		[in]	dT							- input integration timestep.
		*	\param		[in]	type						- input integration type - zEuler or zRK4.
		* 	\param		[in]	tolerance					- input deviation tolerance.
		*	\param		[in]	numIterations				- input number of iterations to run.
		* 	\param		[in]	printInfo					- input boolean indicating print of information of deviation.
		* 	\return				bool						- output boolean , true if  deviation below tolerance.
		*	\since version 0.0.4
		*/
		bool alignToBRGTargets(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1, bool printInfo = false);

		/*! \brief This method updates the smooth mesh.
		*
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.4
		*/
		bool updateSmoothMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1, bool printInfo = false);
	
		/*! \brief This method updates the guide mesh.
		*
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.4
		*/
		bool updateGuideMesh(zDomainFloat& deviation, float dT, zIntergrationType type, float tolerance = EPS, int numIterations = 1);


		/*! \brief This method computes the boundary planes of the print block.
		*
		*	\param		[in]	mBlock						- input macro block.
		*	\since version 0.0.4
		*/
		void addPrintBlockBoundary(zPrintBlock& mBlock);

		/*! \brief This method adds the convex hulls which make up the macro block.
		*
		*	\param		[in]	mBlock						- input macro block.
		*	\param		[in]	guideMesh_halfedge			- input guideMesh halfedge.
		*	\since version 0.0.4
		*/
		void addConvexBlocks(zPrintBlock &mBlock,zItMeshHalfEdge& guideMesh_halfedge, bool boundaryBlock);	
				
		/*! \brief This method compute the convex block faces.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh edge.
		*	\since version 0.0.4
		*/
		void addConvexBlockFaces_fromEdge(zConvexBlock& _block, zItMeshEdge& guideMesh_edge);

		/*! \brief This method compute the block medial graph.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computePrintBlockIntersections(zPrintBlock& _block, const zIntArray& FlipBlockIds);

		/*! \brief This method compute the block medial graph.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computePrintBlockRailIntersections(zPrintBlock& _block, int &railInteriorVertex, zPoint& startPlaneOrigin, zVector& startPlaneNorm, zPoint& endPlaneOrigin, zVector& endPlaneNorm, const zIntArray& FlipBlockIds, zItMeshHalfEdgeArray& railIntersectionHE, zPointArray& railIntersectionPoints);

		/*! \brief This method compute the block frames.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	printLayerDepth				- input print layer depth.
		*	\param		[in]	guideMesh_vertex			- input guide mesh vertex.
		*	\since version 0.0.4
		*/
		void computePrintBlockFrames(zPrintBlock& _block, float printLayerDepth , float neopreneOffset_start , float neopreneOffset_end , const zIntArray& FlipBlockIds, bool useRail);

		/*! \brief This method compute the block frames.
		*
		*	\param		[in]	_block						- input block.
		*	\since version 0.0.4
		*/
		void computePrintBlockSections_Internal(zPrintBlock& _block);	

		/*! \brief This method compute the block frames.
		*
		*	\param		[in]	_block						- input block.
		*	\since version 0.0.4
		*/
		void computePrintBlockSections_Boundary(zPrintBlock& _block);

		/*! \brief This method compute the block frames.
		*
		*	\param		[in]	_block						- input block.
		*	\since version 0.0.4
		*/
		bool checkPrintLayerHeights(zPrintBlock& _block);

		/*! \brief This method compute the block frames for thickned mesh.
		*
		*	\param		[in]	_block						- input block.
		*	\since version 0.0.4
		*/
		void computePrintBlock_bounds(zPrintBlock& _block);

		/*! \brief This method compute the legth of  medial graph of the block
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	leftBlock					- input booealn indicating if its a left block or a right
		*	\since version 0.0.4
		*/
		void computePrintBlockRailInteriorVertex(zPrintBlock& _block, zItMeshHalfEdge& guideMesh_halfedge, bool boundaryBlock);

		/*! \brief This method compute the legth of  medial graph of the block
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	leftBlock					- input booealn indicating if its a left block or a right
		*	\since version 0.0.4
		*/
		void computePrintBlockLength(zPrintBlock& _block, bool leftBlock);

		/*! \brief This method compute the legth of  medial graph of the block
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	leftBlock					- input booealn indicating if its a left block or a right
		*	\since version 0.0.4
		*/
		float computePrintBlockRailLength(zPrintBlock& _block,zPoint &startPlaneOrigin, zVector& startPlaneNorm, zPoint& endPlaneOrigin, zVector& endPlaneNorm, zColor& blockColor, bool leftBlock, const zIntArray& FlipBlockIds, zItMeshHalfEdgeArray& railIntersectionHE, zPointArray& railIntersectionPoints);

		/*! \brief This method compute the block SDF for the balustrade.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	graphId						- input index of section graph.
		*	\since version 0.0.4
		*/
		void computeBlockSDF_Internal(zPrintBlock& _block, int graphId, int printLayerId, bool rightSide,  float printWidth = 0.020, float neopreneOffset = 0.005, bool addRaft = false, int raftId = 0, float raftWidth = 0.030 , bool exportBMP = false);

		/*! \brief This method compute the block SDF for the balustrade.
		*
		*	\param		[in]	_block						- input block.
		*	\param		[in]	graphId						- input index of section graph.
		*	\since version 0.0.4
		*/
		void computeBlockSDF_Boundary(zPrintBlock& _block, int graphId, float printWidth = 0.020, float neopreneOffset = 0.005, bool addRaft = false, int raftId = 0, float raftWidth = 0.030);

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
	
		void polyTopBottomEdges(zObjGraph& inPoly, zItGraphHalfEdgeArray& topHE, zItGraphHalfEdgeArray& bottomHE, float& topLength, float& bottomLength, zPoint startPlaneOrigin, zVector startPlaneNorm, float hardVertexTolerance = 120 );
	
		

		//--------------------------
		//---- IO METHODS
		//--------------------------

		void blockPlanesToTXT(string dir, string filename);

		void blockSidePlanesToTXT(string dir, string filename);

		void blockSectionPlanesToTXT(string dir, string filename);

		void blockSectionsFromJSON(string dir, string filename);

		void blockConvexToJSON(zPrintBlock& _block,string dir, string filename);

		void blockContoursToJSON(int blockId, string dir, string filename, float printLayerWidth, float raftLayerWidth);

		void blockContoursToIncr3D(int blockId, string dir, string filename, float layerWidth = 0.025);

		void toBRGJSON(string path, zPointArray& points, zVectorArray& normals, zPointArray& vThickness);

		bool fromBRGJSON(string path, zPointArray& points, zVectorArray& normals, zPointArray& vThickness);
	
		
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/geometry/zTsSDFBridge.cpp>
#endif

#endif