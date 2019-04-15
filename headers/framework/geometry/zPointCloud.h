#pragma once

#include <headers/framework/core/zVector.h>

namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/*! \class zPointCloud
	*	\brief A point cloud class.
	*	\since version 0.0.2
	*/

	/** @}*/

	class zPointCloud 
	{
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief container which stores point positions. 	*/
		vector<zVector> points;

		/*!	\brief container which stores point colors. 	*/
		vector<zColor> pointColors;

		/*!	\brief container which stores point weights. 	*/
		vector<double> pointWeights;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zPointCloud(){}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pts			- input container of points.
		*	\since version 0.0.2
		*/
		zPointCloud(vector<zVector> &_pts) 
		{
			points = _pts;

			for (int i = 0; i < _pts.size(); i++)
			{
				pointColors.push_back(zColor(1, 0, 0, 1));
				pointWeights.push_back(1.0);
			}
		}
		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zPointCloud() {}


	};
}