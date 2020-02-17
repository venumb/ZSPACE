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


#include<headers/zInterface/objects/zObjPointCloud.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjPointCloud::zObjPointCloud()
	{

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayVertices = false;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjPointCloud::~zObjPointCloud() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjPointCloud::setDisplayElements(bool _displayVerts)
	{
		displayVertices = _displayVerts;
	}

	//---- OVERRIDE METHODS
	
	ZSPACE_INLINE void zObjPointCloud::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		coreUtils.getBounds(pCloud.vertexPositions, minBB, maxBB);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zObjPointCloud::draw()
	{
		if (displayObject)
		{
			drawPointCloud();
		}

		if (displayObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}

	}

	//---- DISPLAY BUFFER METHODS

	ZSPACE_INLINE void zObjPointCloud::appendToBuffer()
	{
		displayObject = displayVertices = false;

		// Vertex Attributes
		zVector*_dummynormals = nullptr;

		pCloud.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(&pCloud.vertexPositions[0], _dummynormals, pCloud.vertexPositions.size());
		pCloud.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(&pCloud.vertexColors[0], pCloud.vertexColors.size());
	}

	//---- PROTECTED DISPLAY METHODS

	ZSPACE_INLINE void zObjPointCloud::drawPointCloud()
	{

		if (displayVertices)
		{

			displayUtils->drawPoints(&pCloud.vertexPositions[0], &pCloud.vertexColors[0], &pCloud.vertexWeights[0], pCloud.vertexPositions.size());

		}

	}

#endif // !ZSPACE_UNREAL_INTEROP

}