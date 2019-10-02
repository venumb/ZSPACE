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

#ifndef ZSPACE_OBJ_MESH_H
#define ZSPACE_OBJ_MESH_H

#pragma once

#include <headers/zInterface/objects/zObj.h>
#include <headers/zCore/geometry/zMesh.h>

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjMesh
	*	\brief The mesh object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	class ZSPACE_API zObjMesh : public zObj
	{
	protected:
		/*! \brief boolean for displaying the mesh vertices */
		bool showVertices;

		/*! \brief boolean for displaying the mesh edges */
		bool showEdges;

		/*! \brief boolean for displaying the mesh faces */
		bool showFaces;

		/*! \brief boolean for displaying the mesh dihedral edges */
		bool showDihedralEdges;

		/*! \brief boolean for displaying the mesh vertex normals */
		bool showVertexNormals;

		/*! \brief boolean for displaying the mesh face normals */
		bool showFaceNormals;

		/*! \brief container for storing dihderal angles */
		zDoubleArray edge_dihedralAngles;

		/*! \brief container for storing face centers */
		zPointArray faceCenters;

		/*! \brief dihedral angle threshold */
		double dihedralAngleThreshold;

		/*! \brief normals display scale */
		double normalScale;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief mesh */
		zMesh mesh;

	

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjMesh();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjMesh();

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show vertices, edges and face booleans.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\param		[in]	_showFaces				- input show faces booelan.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showVerts, bool _showEdges, bool _showFaces);

		/*! \brief This method sets show vertices boolean.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\since version 0.0.2
		*/
		void setShowVertices(bool _showVerts);

		/*! \brief This method sets show edges boolean.
		*
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowEdges(bool _showEdges);

		/*! \brief This method sets show faces boolean.
		*
		*	\param		[in]	_showFaces				- input show faces booelan.
		*	\since version 0.0.2
		*/
		void setShowFaces(bool _showFaces);

		/*! \brief This method sets show dihedral edges boolean.
		*
		*	\param		[in]	_showDihedralEdges			- input show faces booelan.
		*	\param		[in]	_angles						- input container of edge dihedral angles.
		*	\param		[in]	_threshold					- input angle threshold.
		*	\since version 0.0.2
		*/
		void setShowDihedralEdges(bool _showDihedralEdges, zDoubleArray &_angles, double _threshold);

		/*! \brief This method sets show vertex normals boolean.
		*
		*	\param		[in]	_showVertexNormal			- input show vertex normals booelan.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.2
		*/
		void setShowVertexNormals(bool _showVertexNormal, double _normalScale);

		/*! \brief This method sets show face normals boolean.
		*
		*	\param		[in]	_showFaceNormal				- input show face normals booelan.
		*	\param		[in]	_faceCenters				- input container of face centers.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.2
		*/
		void setShowFaceNormals(bool _showFaceNormal, vector<zVector> &_faceCenters, double _normalScale);

		//--------------------------
		//---- GET METHODS
		//--------------------------
		
		/*! \brief This method gets the vertex VBO Index .
		*
		*	\return			int				- vertex VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexID();

		/*! \brief This method gets the edge VBO Index .
		*
		*	\return			int				- edge VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_EdgeID();

		/*! \brief This method gets the face VBO Index .
		*
		*	\return			int				- face VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_FaceID();

		/*! \brief This method gets the vertex color VBO Index .
		*
		*	\return			int				- vertex color VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexColorID();		

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override;

		void getBounds(zPoint &minBB, zPoint &maxBB) override;

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.2
		*/
		void drawMesh();

		/*! \brief This method displays the dihedral edges of a mesh above the input angle threshold.
		*
		*	\since version 0.0.2
		*/
		void drawMesh_DihedralEdges();

		/*! \brief This method displays the vertex normals of a mesh.
		*
		*	\param		[in]	dispScale				- display scale of the normal.
		*	\since version 0.0.2
		*/
		void drawMesh_VertexNormals();

		/*! \brief This method displays the face normals of a mesh.
		*
		*	\since version 0.0.2
		*/
		void drawMesh_FaceNormals();

		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends mesh to the buffer.
		*
		*	\param		[in]	edge_dihedralAngles	- input container of edge dihedral angles.
		*	\param		[in]	DihedralEdges		- true if only edges above the dihedral angle threshold are stored.
		*	\param		[in]	angleThreshold		- angle threshold for the edge dihedral angles.
		*	\since version 0.0.1
		*/
		void appendToBuffer(zDoubleArray edge_dihedralAngles = zDoubleArray(), bool DihedralEdges = false, double angleThreshold = 45);

	};


	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zBase
	*	\brief  The base classes, enumerators ,defintions of the library.
	*  @{
	*/

	/** \addtogroup zTypeDefs
	*	\brief  The type defintions of the library.
	*  @{
	*/

	/** \addtogroup Container
	*	\brief  The container typedef of the library.
	*  @{
	*/

	/*! \typedef zObjMeshArray
	*	\brief A vector of zObjMesh.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjMesh> zObjMeshArray;

	/*! \typedef zObjMeshPointerArray
	*	\brief A vector of zObjMesh pointers.
	*
	*	\since version 0.0.4
	*/
	typedef vector<zObjMesh*> zObjMeshPointerArray;

	/** @}*/
	/** @}*/
	/** @}*/
	/** @}*/

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/objects/zObjMesh.cpp>
#endif

#endif