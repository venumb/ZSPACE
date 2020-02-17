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


#include<headers/zInterface/objects/zObjSpatialBin.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjSpatialBin::zObjSpatialBin()
	{
#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayBounds = false;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjSpatialBin::~zObjSpatialBin() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjSpatialBin::setDisplayBounds(bool _displayBounds)
	{
		displayBounds = _displayBounds;
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zObjSpatialBin::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		minBB = field.minBB;
		maxBB = field.maxBB;
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else

	ZSPACE_INLINE void zObjSpatialBin::draw()
	{

		if (displayObject)
		{
			drawBins();
		}

		if (displayBounds)
		{
			drawBounds();
		}

		if (displayObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}
	}

	//---- PROTECTED DISPLAY METHODS

	ZSPACE_INLINE void zObjSpatialBin::drawBins()
	{
		glLineWidth(1);

		zVector unit(field.unit_X, field.unit_Y, field.unit_Z);

		for (int i = 0; i < pCloud.n_v; i++)
		{
			zVector bottom = pCloud.vertexPositions[i];
			zVector top = pCloud.vertexPositions[i] + unit;

			if (bins[i].contains())
			{
				displayUtils->drawCube(bottom, top, zColor(0, 0, 0, 1));
			}
		}
	}

	ZSPACE_INLINE void zObjSpatialBin::drawBounds()
	{
		glLineWidth(1);
		displayUtils->drawCube(field.minBB, field.maxBB, zColor(0, 1, 0, 1));
	}

#endif //!ZSPACE_UNREAL_INTEROP

}