#pragma once
#pragma once

#include <headers/api/object/zObject.h>
#include <headers/framework/geometry/zGraph.h>

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

	/*! \class zObjGraph
	*	\brief The graph object class.
	*	\since version 0.0.2
	*/

	/** @}*/
	
	/** @}*/

	class zObjGraph : public zObject
	{
	private:
		/*! \brief boolean for displaying the mesh vertices */
		bool showVertices;

		/*! \brief boolean for displaying the mesh edges */
		bool showEdges;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief graph */
		zGraph graph;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjGraph()
		{
			displayUtils = nullptr;

			showVertices = false;
			showEdges = true;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_displayUtils			- input display utilities.
		*	\since version 0.0.2
		*/
		zObjGraph(zUtilsDisplay &_displayUtils)
		{
			displayUtils = &_displayUtils;

			showVertices = false;
			showEdges = true;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjGraph() {}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show vertices, edges and face booleans.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showVerts, bool _showEdges)
		{
			showVertices = _showVerts;
			showEdges = _showEdges;			
		}

		/*! \brief This method sets show vertices boolean.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\since version 0.0.2
		*/
		void setShowVertices(bool _showVerts)
		{
			showVertices = _showVerts;
		}

		/*! \brief This method sets show edges boolean.
		*
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowEdges(bool _showEdges)
		{
			showEdges = _showEdges;
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			if (showObject)
			{
				drawGraph();
			}
		}

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zGraph.
		*
		*	\since version 0.0.2
		*/
		inline void drawGraph()
		{

			//draw vertex
			if (showVertices)
			{
				for (int i = 0; i < graph.vertexActive.size(); i++)
				{
					if (graph.vertexActive[i])
					{
						zColor col;
						double wt = 1;

						if (graph.vertexColors.size() > i)  col = graph.vertexColors[i];
						if (graph.vertexWeights.size() > i) wt = graph.vertexWeights[i];

						displayUtils->drawPoint(graph.vertexPositions[i], col, wt);
					}
				}
			}

			//draw edges
			if (showEdges)
			{

				for (int i = 0; i < graph.edgeActive.size(); i += 2)
				{
					if (graph.edgeActive[i])
					{
						if (graph.edges[i].getVertex() && graph.edges[i + 1].getVertex())
						{
							zColor col;
							double wt = 1;

							if (graph.edgeColors.size() > i)  col = graph.edgeColors[i];
							if (graph.edgeWeights.size() > i) wt = graph.edgeWeights[i];

							int v1 = graph.edges[i].getVertex()->getVertexId();
							int v2 = graph.edges[i + 1].getVertex()->getVertexId();

							displayUtils->drawLine(graph.vertexPositions[v1], graph.vertexPositions[v2], col, wt);
						}
					}

				}

			}
		}


		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends graph to the buffer.
		*
		*	\since version 0.0.1
		*/
		void appendGraph()
		{
			// Edge Indicies
			vector<int> _edgeIndicies;

			for (int i = 0; i < graph.edgeActive.size(); i++)
			{
				_edgeIndicies.push_back(graph.edges[i].getVertex()->getVertexId() + displayUtils->bufferObj.nVertices);
			}

			graph.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);


			// Vertex Attributes
			vector<zVector>_dummynormals;

			graph.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(graph.vertexPositions, _dummynormals);
			graph.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(graph.vertexColors);
		}

	};




}

