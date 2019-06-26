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
		/*! \brief boolean for displaying the vertices */
		bool showVertices;

		/*! \brief boolean for displaying the edges */
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

			if (showObjectTransform)
			{
				displayUtils->drawTransform(transformationMatrix);
			}
		}

		void getBounds(zVector &minBB, zVector &maxBB) override
		{
			coreUtils->getBounds(graph.vertexPositions, minBB, maxBB);
		}

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zGraph.
		*
		*	\since version 0.0.2
		*/
		void drawGraph()
		{

			//draw vertex
			if (showVertices)
			{

				displayUtils->drawVertices(graph.vHandles, &graph.vertexPositions[0], &graph.vertexColors[0], &graph.vertexWeights[0]);

			}


			//draw edges
			if (showEdges)
			{
				if (graph.staticGeometry)
				{
					displayUtils->drawEdges(graph.eHandles, graph.edgeVertices, &graph.vertexPositions[0], &graph.edgeColors[0], &graph.edgeWeights[0]);
				}

				else
				{
					vector<vector<int>> edgeVertices;
					edgeVertices.assign(graph.edges.size(), vector<int>(2) = { -1,-1 });

					for (auto &e : graph.edges)
					{

						if (graph.eHandles[e.getId()].id != -1)
						{
							vector<int> eVerts;

							edgeVertices[e.getId()][0] = e.getHalfEdge(0)->getVertex()->getId();
							edgeVertices[e.getId()][1] = e.getHalfEdge(1)->getVertex()->getId();
						}


					}

					displayUtils->drawEdges(graph.eHandles, edgeVertices, &graph.vertexPositions[0], &graph.edgeColors[0], &graph.edgeWeights[0]);
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
		void appendToBuffer()
		{
			showObject = showEdges = showVertices =  false;

			// Edge Indicies
			vector<int> _edgeIndicies;

			//for (int i = 0; i < graph.edgeActive.size(); i++)
			for (auto &e : graph.halfEdges)
			{
				_edgeIndicies.push_back(e.getVertex()->getId() + displayUtils->bufferObj.nVertices);
			}

			graph.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);


			// Vertex Attributes
			zVector* _dummynormals = nullptr;

			graph.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(&graph.vertexPositions[0], _dummynormals, graph.vertexPositions.size());
			graph.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(&graph.vertexColors[0], graph.vertexColors.size());
		}

	};




}

