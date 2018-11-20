#pragma once

#include<headers/core/zVector.h>

namespace zSpace
{
	
	/*! \brief This method computes the distances in X,Y,Z for the input bounds.
	*
	*	\param		[in]	minBB			- lower bounds as zVector.
	*	\param		[in]	maxBB			- upper bounds as zVector
	*	\param		[out]	Dims			- distances in X,Y,Z axis in local frame
	*/
	
	zVector getDimsFromBounds(zVector &minBB, zVector &maxBB)
	{
		zVector out;

		out.x = abs(maxBB.x - minBB.x);
		out.y = abs(maxBB.y - minBB.y);
		out.z = abs(maxBB.z - minBB.z);

		return out;
	}


}
