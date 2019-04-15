#pragma once
#include <headers/api/object/zObject.h>

namespace zSpace
{

	/** \addtogroup zViewer
	*	\brief The viewer related class and function set classes of the library.
	*  @{
	*/

	/*! \class zModel
	*	\brief The model class.
	*	\since version 0.0.2
	*/

	/** @}*/

	class zModel 
	{
	private:
		//--------------------------
		//----  PRIVATE ATTRIBUTES
		//--------------------------
				

		/*! \brief boolean for displaying the buffer point object colors */
		bool showBufPointColors;
		
		/*! \brief boolean for displaying the buffer object as points */
		bool showBufPoints;

		/*! \brief boolean for displaying the buffer line object colors */
		bool showBufLinesColors;

		/*! \brief boolean for displaying the buffer object as lines */
		bool showBufLines;

		/*! \brief boolean for displaying the buffer triangle object colors */
		bool showBufTrisColors;

		/*! \brief boolean for displaying the buffer object as triangles */
		bool showBufTris;

		/*! \brief boolean for displaying the buffer Quads object colors */
		bool showBufQuadsColors;

		/*! \brief boolean for displaying the buffer object as quads */
		bool showBufQuads;


	public:
		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief display utilities object	*/
		zUtilsCore coreUtils;

		/*! \brief display utilities object	*/
		zUtilsDisplay displayUtils;

		/*! \brief container of scene objects	*/
		vector<zObject *> sceneObjects;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zModel()
		{
			
			showBufPointColors = false;
			showBufPoints = false;

			showBufLinesColors = false;
			showBufLines = false;
			
			showBufTrisColors = false;
			showBufTris = false;

			showBufQuadsColors = false;
			showBufQuads = false;
		}

		/*! \brief Overloaded constructor.
		*
		*	\since version 0.0.2
		*/
		zModel(int _buffersize)
		{
			displayUtils.bufferObj = zObjBuffer(_buffersize);

			showBufPointColors = false;
			showBufPoints = false;

			showBufLinesColors = false;
			showBufLines = false;

			showBufTrisColors = false;
			showBufTris = false;

			showBufQuadsColors = false;
			showBufQuads = false;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zModel() {}

		//--------------------------
		//---- OBJECT METHODS
		//--------------------------

		/*! \brief This method adds the input object to the scene.
		*
		*	\param		[in]	obj				- input object.
		*	\since version 0.0.2
		*/
		void addObject(zObject &obj)
		{
			sceneObjects.push_back(&obj);

			obj.setUtils(displayUtils,coreUtils);


		}


		/*! \brief This method adds the input container of objects to the scene.
		*
		*	\param		[in]	objs				- input  contatiner of objects.
		*	\since version 0.0.2
		*/
		void addObjects(vector<zObject> &objs)
		{
			for (int i = 0; i < objs.size(); i++)
			{
				sceneObjects.push_back(&objs[i]);

				objs[i].setUtils(displayUtils, coreUtils);
			}


		}
		//--------------------------
		//---- DRAW METHODS
		//--------------------------

		/*! \brief This method draws the scene objects.
		*
		*	\since version 0.0.2
		*/
		void draw()
		{
			for (auto obj : sceneObjects)
			{
				obj->draw();
			}

			// buffer display

			if (showBufPoints) displayUtils.drawPointsFromBuffer(showBufPointColors);

			if (showBufLines) displayUtils.drawLinesFromBuffer(showBufLinesColors);

			if (showBufTris) displayUtils.drawTrianglesFromBuffer(showBufTrisColors);

			if (showBufQuads) displayUtils.drawQuadsFromBuffer(showBufQuadsColors);

		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show buffer points boolean.
		*
		*	\param		[in]	_showBufPoints				- input show buffer points boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufPoints(bool _showBufPoints, bool showColors = false)
		{
			showBufPoints = _showBufPoints;

			showBufPointColors = showColors;
		}

		/*! \brief This method sets show buffer lines boolean.
		*
		*	\param		[in]	_showBufLines				- input show buffer line boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufLines(bool _showBufLines, bool showColors = false)
		{
			showBufLines = _showBufLines;

			showBufLinesColors = showColors;
		}

		/*! \brief This method sets show buffer triangles boolean.
		*
		*	\param		[in]	_showBufTris				- input show buffer tris boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufTris(bool _showBufTris, bool showColors = false)
		{
			showBufTris = _showBufTris;

			showBufTrisColors = showColors;
		}

		/*! \brief This method sets show buffer quads boolean.
		*
		*	\param		[in]	_showBufQuads				- input show buffer quads boolean.
		*	\param		[in]	showColors					- true if color needs to be displayed.
		*	\since version 0.0.2
		*/
		void setShowBufQuads(bool _showBufQuads, bool showColors = false)
		{
			showBufQuads = _showBufQuads;

			showBufQuadsColors = showColors;
		}

	};
}