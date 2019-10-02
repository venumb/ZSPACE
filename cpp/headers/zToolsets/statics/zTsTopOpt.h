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

#ifndef ZSPACE_TS_STATICS_TOPOPT_H
#define ZSPACE_TS_STATICS_TOPOPT_H

#pragma once
#include <headers/zInterface/functionsets/zFnMesh.h>

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

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop of topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptMaterial
	*	\brief A struct to hold material information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	/** @}*/
	   
	struct zTopOptMaterial
	{
		/*!	\brief Thickness  */
		double thickness; 

		/*!	\brief Young's Modulus  */
		double E;

		/*!	\brief Shear Modulus  */
		double G;

		/*!	\brief Poisson's Ratio  */
		double NU;

		/*!	\brief Mass Density  */
		double RHO;

		/*!	\brief Stress Limits in Tension  */
		double ST;

		/*!	\brief Stress Limits in Compression  */
		double SC;

		/*!	\brief Stress Limits in Shear  */
		double SS;
						
	};

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop of topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptLoads
	*	\brief A struct to hold load information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zTopOptLoads
	{
		/*!	\brief Magnitude  */
		double magnitude;

		/*!	\brief Load Directions  */
		zVector dir;

		/*!	\brief vertex indicies where the load is applied  */
		vector<int> indicies;		
	};


	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop of topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptPattern
	*	\brief A struct to hold pattern grouping information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zTopOptPattern
	{
		

		/*!	\brief type  */
		int type;

		/*!	\brief achor  */
		zVector anchor;

		/*!	\brief node1  */
		zVector node1;

		/*!	\brief node2  */
		zVector node2;		
	};


	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop of topology optimization with Altair Hypermesh.
	*  @{
	*/
	
	/*! \class zTsTopOpt
	*	\brief A tool set class for interOp to Altair Hypermesh.
	*	\since version 0.0.2
	*/
		
	/** @}*/

	/** @}*/

	/** @}*/

	class ZSPACE_TOOLS zTsTopOpt
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to result Object  */
		zObjMesh *meshObj;

		/*!	\brief material  */
		zTopOptMaterial mat;

		/*!	\brief container of loads  */
		vector<zTopOptLoads> loads;

		/*!	\brief container of pattern groupings  */
		zTopOptPattern pattern;

		/*!	\brief container of booleans indicating if a face is design(true) or non design space(false)  */
		vector<bool> designSpace_Boolean;
							
		/*!	\brief container of booleans indicating if a vertex is SPC(true) or not(false)  */
		vector<bool> SPC_Boolean;		

	public:

		/*!	\brief color domain.  */
		zDomainColor elementColorDomain = zDomainColor(zColor(0.5, 0, 0.2, 1), zColor(0, 0.2, 0.5, 1));

		/*!	\brief result function set  */
		zFnMesh fnMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsTopOpt();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsTopOpt(zObjMesh &_meshObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsTopOpt();

			   
		//--------------------------
		//---- TO METHOD
		//--------------------------

		/*! \brief This method exports the topOpt as a JSON file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file - zJSON.
		*	\since version 0.0.2
		*/
		void to(string path, zFileTpye type);

		//--------------------------
		//---- CREATE
		//--------------------------

		/*! \brief This method creates the mesh from a file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createFromFile(string path, zFileTpye type);

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the single point constraints.
		*
		*	\param [in]		_SPC			- input container of constraint vertices. If the container is empty , then all vertices will non constrained.
		*	\since version 0.0.2
		*/
		void setSinglePointConstraints(const vector<int> &_SPC = vector<int>());

		/*! \brief This method sets the non design space.
		*
		*	\param [in]		_SPC			- input container of constraint vertices. If the container is empty , then all faces will be set as design.
		*	\since version 0.0.2
		*/
		void setNonDesignSpace(const vector<int> &_NonDesignSpace = vector<int>());

		/*! \brief This method sets the material.
		*
		*	\param [in]		material			- input material.
		*	\since version 0.0.2
		*/
		void setMaterial(zTopOptMaterial &material);


		/*! \brief This method sets the pattern grouping.
		*
		*	\param [in]		_pattern		- input pattern.
		*	\since version 0.0.2
		*/
		void setPatternGrouping(zTopOptPattern &_pattern);

		/*! \brief This method sets the pattern grouping.
		*
		*	\param [in]		type			- input type. ( 0 - no groupings, 1 - 1 plane symmetry, 2 - 2 plane symmetry)
		*	\param [in]		anchor			- input anchor point.
		*	\param [in]		n1				- input first node point.
		*	\param [in]		n2				- input second node point.
		*	\since version 0.0.2
		*/
		void setPatternGrouping(int type = 0, const zVector &anchor = zVector(), const zVector &n1 = zVector(), const zVector &n2 = zVector());

		//--------------------------
		//--- LOAD METHODS 
		//--------------------------

		/*! \brief This method adds a load conditions.
		*
		*	\param [in]		_magnitude		- input load magnitude.
		*	\param [in]		_dir			- input load direction.
		*	\param [in]		vIndices		- input container of vertices to which the load is applied.
		*	\since version 0.0.2
		*/
		void addLoad(double _magnitude, zVector &_dir, vector<int>& vIndices);

		/*! \brief This method removes a existing load conditions.
		*
		*	\param [in]		index			- input load index to be removed.
		*	\since version 0.0.2
		*/
		void removeLoad(int index);

		/*! \brief This method removes all existing load conditions.
		*
		*	\since version 0.0.2
		*/
		void removeLoads();
		

		//--------------------------
		//---- PROTECTED FACTORY METHODS
		//--------------------------

	protected:
		/*! \brief This method exports zMesh to a JSON file format using JSON Modern Library.
		*
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toJSON(string outfilename);

	};

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zToolsets/statics/zTsTopOpt.cpp>
#endif

#endif