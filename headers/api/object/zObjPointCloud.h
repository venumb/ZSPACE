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

		/*! \brief boolean for displaying the vertices */
		bool showVertices;

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

			showVertices = false;

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

		/*! \brief This method sets show vertices booleans.
		*
		*	\param		[in]	_showVerts				- input show vertices boolean.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showVerts)
		{
			showVertices = _showVerts;		
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

			if (showVertices)
			{

				displayUtils->drawPoints(pCloud.vertexPositions, pCloud.vertexColors, pCloud.vertexWeights);

			}

		}


		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends pointcloud to the buffer.
		*
		*	\since version 0.0.1
		*/
		void appendToBuffer()
		{
			showObject =  showVertices =  false;			

			// Vertex Attributes
			vector<zVector>_dummynormals;

			pCloud.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(pCloud.vertexPositions, _dummynormals);
			pCloud.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(pCloud.vertexColors);
		}

	};
}