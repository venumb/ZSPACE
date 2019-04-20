#pragma once
#pragma once

#include <headers/api/object/zObj.h>
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

	class zObjGraph : public zObj
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

				displayUtils->drawPoints(graph.vertexPositions, graph.vertexColors, graph.vertexWeights);

			}


			//draw edges
			if (showEdges)
			{
				if (graph.staticGeometry)
				{
					displayUtils->drawEdges(graph.edgePositions, graph.edgeColors, graph.edgeWeights);
				}

				else
				{
					vector<vector<zVector>> edgePositions;

					for (int i = 0; i < graph.n_e; i++)
					{

						if (graph.edges[i].getVertex() && graph.edges[i + 1].getVertex())
						{
							int v1 = graph.edges[i].getVertex()->getVertexId();
							int v2 = graph.edges[i + 1].getVertex()->getVertexId();

							vector<zVector> vPositions;
							vPositions.push_back(graph.vertexPositions[v1]);
							vPositions.push_back(graph.vertexPositions[v2]);

							edgePositions.push_back(vPositions);

						}

					}

					displayUtils->drawEdges(edgePositions, graph.edgeColors, graph.edgeWeights);

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

