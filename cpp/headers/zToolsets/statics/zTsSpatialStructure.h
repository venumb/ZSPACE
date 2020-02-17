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

#ifndef ZSPACE_TS_STATICS_SPATIALSTRUCTURE_H
#define ZSPACE_TS_STATICS_SPATIALSTRUCTURE_H

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

	/** \addtogroup z3DGS
	*	\brief tool sets for 3D graphic statics.
	*  @{
	*/

	/*! \class zTsSpatialStructures
	*	\brief A toolset for creating spatial strctures from volume meshes.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsSpatialStructures
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- CELL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer container to volume Object  */
		vector<zObjMesh*> volumeObjs;

		//--------------------------
		//---- FORM DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to form Object  */
		zObjGraph *formObj;

		/*!	\brief container of  form particle objects  */
		vector<zObjParticle> formParticlesObj;

		/*!	\brief container of form particle function set  */
		vector<zFnParticle> fnFormParticles;

		/*!	\brief container storing the target for form edges.  */
		vector<zVector> targetEdges_form;

		//--------------------------
		//---- POLYTOPAL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to polytopal Object  */
		vector<zObjMesh*> polytopalObjs;

		int smoothSubDivs;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnGraph fnForm;

		/*!	\brief container of force function set  */
		vector<zFnMesh> fnVolumes;

		/*!	\brief container of polytopals function set  */
		vector <zFnMesh> fnPolytopals;

		/*!	\brief force volume mesh faceID to form graph vertexID map.  */
		unordered_map <string, int> volumeFace_formGraphVertex;

		/*!	\brief form graph vertexID to volume meshID map.  */
		vector<int> formGraphVertex_volumeMesh;

		/*!	\brief form graph vertexID to local volume mesh faceID map.*/
		vector<int> formGraphVertex_volumeFace;

		/*!	\brief graph vertex offset container.*/
		vector<double> formGraphVertex_Offsets;

		/*!	\brief container of facecenter per force volume  */
		vector<vector<zVector>> volume_fCenters;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsSpatialStructures();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_volumeObjs			- input container of volume objects.
		*	\param		[in]	_polytopalObjs		- input container of polytopal objects.
		*	\since version 0.0.2
		*/
		zTsSpatialStructures(zObjGraph &_formObj, vector<zObjMesh> &_volumeObjs, vector<zObjMesh>  &_polytopalObjs);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsSpatialStructures();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------				

		/*! \brief This method creates the volume geometry from the input files.
		*
		*	\param [in]		directory		- input directory path.
		*	\param [in]		filename		- input filename.
		*	\param [in]		numFiles		- input number of files in the directory.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.1
		*/
		void createVolumeFromFile(string directory, string filename, int numFiles, zFileTpye type);

		/*! \brief This method creates the center line graph based on the input volume meshes.
		*
		*	\param		[in]	offsets						- input offsets value container.
		*	\param		[in]	precisionFac				- precision factor of the points for checking.
		*	\since version 0.0.1
		*/
		void createFormFromVolume(double offset, int precisionFac = 3, zColor edgeCol = zColor(0.75, 0.5, 0, 1));

		/*! \brief This method creates the polytopal mesh from the volume meshes and form graph.
		*
		*	\since version 0.0.2
		*/
		void createPolytopalsFromVolume();

		//--------------------------
		//----UPDATE METHOD
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
		bool equilibrium(bool computeTargets, double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false);	

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
		void setVertexOffsets(vector<double> &offsets);

		/*! \brief This method computes the form graph edge weights based on the volume mesh face areas.
		*
		*	\param		[in]	weightDomain				- weight domain of the edge.		
		*	\since version 0.0.1
		*/
		void setFormEdgeWeightsfromVolume(zDomainFloat weightDomain = zDomainFloat(2.0, 10.0));

	protected:		

		void extrudeConnectionFaces(int volumeIndex);

		/*! \brief This method creates the polytopal mesh based on the input force volume mesh and form graph.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopal(int volumeIndex);

		/*! \brief This method computes the face centers of the input volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.1
		*/
		void computeVolumesFaceCenters();

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
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	tolerance					- tolerance for force.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000);

	};
	
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/statics/zTsSpatialStructure.cpp>
#endif

#endif