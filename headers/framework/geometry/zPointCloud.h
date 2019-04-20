#pragma once

#include <headers/framework/core/zVector.h>

namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zPointCloud
	*	\brief A point cloud class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zPointCloud 
	{
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief container which stores vertex positions. 	*/
		vector<zVector> vertexPositions;

		/*!	\brief container which stores vertex colors. 	*/
		vector<zColor> vertexColors;

		/*!	\brief container which stores vertex weights. 	*/
		vector<double> vertexWeights;

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
			vertexPositions = _pts;

			for (int i = 0; i < _pts.size(); i++)
			{
				vertexColors.push_back(zColor(1, 0, 0, 1));
				vertexWeights.push_back(1.0);
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