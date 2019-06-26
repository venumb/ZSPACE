#pragma once
#pragma once

#include <headers/api/object/zObjPointCloud.h>
#include <headers/framework/field/zField3D.h>
#include <headers/framework/field/zBin.h>

#include <vector>
using namespace std;

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

	/*! \class zObjSpatialBin
	*	\brief The spatial binning class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/	

	class zObjSpatialBin : public zObjPointCloud
	{
	protected:
		/*! \brief boolean for displaying the bins */
		bool showBounds;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField3D<double> field;	

		/*!	\brief bins			*/
		vector<zBin> bins;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zObjSpatialBin()
		{
			displayUtils = nullptr;	

			showBounds = false;
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.3
		*/
		~zObjSpatialBin() {}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show bins booleans.
		*
		*	\param		[in]	_showBounds				- input show bounds boolean.
		*	\since version 0.0.3
		*/
		void setShowBounds(bool _showBounds)
		{
			showBounds = _showBounds;
		}

	
		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			
			if (showObject)
			{
				drawBins();
			}

			if (showBounds)
			{
				drawBounds();
			}

			if (showObjectTransform)
			{
				displayUtils->drawTransform(transformationMatrix);
			}
		}		


		void getBounds(zVector &minBB, zVector &maxBB) override
		{
			minBB = field.minBB;
			maxBB = field.maxBB;
		}

		/*! \brief This method displays the point cloud.
		*
		*	\since version 0.0.3
		*/
		void drawBins()
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

		/*! \brief This method displays the bounds of bins.
		*
		*	\since version 0.0.3
		*/
		void drawBounds()
		{
			glLineWidth(1);
			displayUtils->drawCube(field.minBB, field.maxBB, zColor(0, 1, 0, 1));
		}
	};




}

