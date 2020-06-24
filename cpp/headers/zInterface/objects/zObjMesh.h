// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
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
		bool displayVertices;

		/*! \brief boolean for displaying the mesh edges */
		bool displayEdges;

		/*! \brief boolean for displaying the mesh faces */
		bool displayFaces;

		/*! \brief boolean for displaying the mesh dihedral edges */
		bool displayDihedralEdges;

		/*! \brief boolean for displaying the mesh vertex normals */
		bool displayVertexNormals;

		/*! \brief boolean for displaying the mesh face normals */
		bool displayFaceNormals;

		/*! \brief container for storing dihderal angles */
		zDoubleArray edge_dihedralAngles;

		/*! \brief container for storing face centers */
		zPointArray faceCenters;

		/*! \brief dihedral angle threshold */
		double dihedralAngleThreshold;

		/*! \brief normals display scale */
		double normalScale;

		/*! \brief boolean for displaying the mesh ids */
		bool displayVertexIds, displayEdgeIds, displayFaceIds;

		/*! \brief container for storing edge centers */
		zPointArray edgeCenters;


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

		/*! \brief This method sets display vertices, edges and face booleans.
		*
		*	\param		[in]	_displayVerts				- input display vertices booelan.
		*	\param		[in]	_displayEdges				- input display edges booelan.
		*	\param		[in]	_displayFaces				- input display faces booelan.
		*	\since version 0.0.4
		*/
		void setDisplayElements(bool _displayVerts, bool _displayEdges, bool _displayFaces);

		/*! \brief This method sets display vertexIds, edgeIds and faceIds booleans.
		*
		*	\param		[in]	_displayVertIds				- input display vertexIds booelan.
		*	\param		[in]	_displayEdgeIds				- input display edgeIds booelan.
		*	\param		[in]	_displayFaceIds				- input display faceIds booelan.
		*	\since version 0.0.4
		*/
		void setDisplayElementIds(bool _displayVertIds, bool _displayEdgeIds, bool _displayFaceIds);

		/*! \brief This method sets display vertices boolean.
		*
		*	\param		[in]	_displayVerts				- input display vertices booelan.
		*	\since version 0.0.4
		*/
		void setDisplayVertices(bool _displayVerts);

		/*! \brief This method sets display edges boolean.
		*
		*	\param		[in]	_displayEdges				- input display edges booelan.
		*	\since version 0.0.4
		*/
		void setDisplayEdges(bool _displayEdges);

		/*! \brief This method sets display faces boolean.
		*
		*	\param		[in]	_displayFaces				- input display faces booelan.
		*	\since version 0.0.4
		*/
		void setDisplayFaces(bool _displayFaces);

		/*! \brief This method sets display dihedral edges boolean.
		*
		*	\param		[in]	_displayDihedralEdges			- input display faces booelan.
		*	\param		[in]	_threshold					- input angle threshold.
		*	\since version 0.0.4
		*/
		void setDisplayDihedralEdges(bool _displayDihedralEdges, double _threshold); // compute in seperate method

		/*! \brief This method sets display vertex normals boolean.
		*
		*	\param		[in]	_displayVertexNormal			- input display vertex normals booelan.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.4
		*/
		void setDisplayVertexNormals(bool _displayVertexNormal, double _normalScale);

		/*! \brief This method sets display face normals boolean.
		*
		*	\param		[in]	_displayFaceNormal				- input display face normals booelan.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.4
		*/
		void setDisplayFaceNormals(bool _displayFaceNormal, double _normalScale);

		/*! \brief This method sets face center container.
		*
		*	\param		[in]	_faceCenters				- input face centers container.
		*	\since version 0.0.4
		*/
		void setFaceCenters(zPointArray &_faceCenters);

		/*! \brief This method sets edge centers container.
		*
		*	\param		[in]	_edgeCenters				- input edge center conatiner.
		*	\since version 0.0.4
		*/
		void setEdgeCenters(zPointArray &_edgeCenters);

		/*! \brief This method sets dihedral angle container.
		*
		*	\param		[in]	_dihedralAngles				- input dihedral angle container.
		*	\since version 0.0.4
		*/
		void setDihedralAngles(zDoubleArray &_edge_dihedralAngles);

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
		
		void getBounds(zPoint &minBB, zPoint &maxBB) override;

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) 
		// Do Nothing
#else
		void draw() override;
#endif

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

		protected:
		//--------------------------
		//---- PROTECTED DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.4
		*/
		void drawMesh();

		/*! \brief This method displays the dihedral edges of a mesh above the input angle threshold.
		*
		*	\since version 0.0.4
		*/
		void drawMesh_DihedralEdges();

		/*! \brief This method displays the vertex normals of a mesh.
		*
		*	\param		[in]	dispScale				- display scale of the normal.
		*	\since version 0.0.4
		*/
		void drawMesh_VertexNormals();

		/*! \brief This method displays the face normals of a mesh.
		*
		*	\since version 0.0.4
		*/
		void drawMesh_FaceNormals();
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