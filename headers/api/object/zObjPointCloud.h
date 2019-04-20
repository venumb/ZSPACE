#pragma once

#include <headers/api/object/zObj.h>
#include <headers/framework/geometry/zPointCloud.h>

namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjPointCloud
	*	\brief The point cloud object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zObjPointCloud :public zObj
	{
	private:
		/*! \brief boolean for displaying the points color */
		bool showColors;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief point cloud */
		zPointCloud pCloud;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjPointCloud()
		{
			displayUtils = nullptr;

			showColors = false;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjPointCloud() {}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show points boolean.
		*
		*	\param		[in]	_showCols				- input show colors booelan.
		*	\since version 0.0.2
		*/
		void setShowColors(bool _showCols)
		{
			showColors = _showCols;
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			if (showObject)
			{
				drawPointCloud();

				
			}

		}
		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.2
		*/
		void drawPointCloud()
		{
			for (int i = 0; i < pCloud.vertexPositions.size(); i++)
			{
				
					zColor col;
					double wt = 1;

					if (pCloud.vertexColors.size() > i)  col = pCloud.vertexColors[i];
					if (pCloud.vertexWeights.size() > i) wt = pCloud.vertexWeights[i];

					displayUtils->drawPoint(pCloud.vertexPositions[i], col, wt);

			}
		}

	};
}