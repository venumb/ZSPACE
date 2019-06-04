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

			if (showObjectTransform)
			{
				displayUtils->drawTransform(transformationMatrix);
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

				displayUtils->drawPoints(&pCloud.vertexPositions[0], &pCloud.vertexColors[0], &pCloud.vertexWeights[0], pCloud.vertexPositions.size());

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
			zVector*_dummynormals = nullptr;

			pCloud.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(&pCloud.vertexPositions[0], _dummynormals, pCloud.vertexPositions.size());
			pCloud.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(&pCloud.vertexColors[0], pCloud.vertexColors.size());
		}

	};
}