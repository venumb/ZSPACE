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


#include<headers/zInterface/objects/zObjMeshField.h>

namespace zSpace
{

	//---- CONSTRUCTOR
	
	template<typename T>
	ZSPACE_INLINE zObjMeshField<T>::zObjMeshField()
	{

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayVertices = false;
		displayEdges = true;
		displayFaces = true;

		displayDihedralEdges = false;
		displayVertexNormals = false;
		displayFaceNormals = false;

		dihedralAngleThreshold = 45;

		normalScale = 1.0;
	}

	//---- DESTRUCTOR

	template<typename T>
	ZSPACE_INLINE zObjMeshField<T>::~zObjMeshField() {}

	//---- OVERRIDE METHODS

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
	
	template<typename T>
	ZSPACE_INLINE void zObjMeshField<T>::draw()
	{
		zObjMesh::draw();
	}

#endif

	template<typename T>
	ZSPACE_INLINE void zObjMeshField<T>::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		zObjMesh::getBounds(minBB, maxBB);
	}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zObjMeshField<zVector>;

	template class zObjMeshField<zScalar>;

#endif
}