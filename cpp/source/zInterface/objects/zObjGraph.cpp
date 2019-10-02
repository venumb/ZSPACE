// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//


#include<headers/zInterface/objects/zObjGraph.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjGraph::zObjGraph()
	{
		displayUtils = nullptr;

		showVertices = false;
		showEdges = true;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjGraph::~zObjGraph() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjGraph::setShowElements(bool _showVerts, bool _showEdges)
	{
		showVertices = _showVerts;
		showEdges = _showEdges;
	}

	ZSPACE_INLINE void zObjGraph::setShowVertices(bool _showVerts)
	{
		showVertices = _showVerts;
	}

	ZSPACE_INLINE void zObjGraph::setShowEdges(bool _showEdges)
	{
		showEdges = _showEdges;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zObjGraph::getVBO_VertexID()
	{
		return graph.VBO_VertexId;
	}

	ZSPACE_INLINE int zObjGraph::getVBO_EdgeID()
	{
		return graph.VBO_EdgeId;
	}

	ZSPACE_INLINE int zObjGraph::getVBO_VertexColorID()
	{
		return graph.VBO_VertexColorId;
	}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE void zObjGraph::draw()
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

	ZSPACE_INLINE void zObjGraph::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		coreUtils->getBounds(graph.vertexPositions, minBB, maxBB);
	}

	//---- DISPLAY METHODS

	ZSPACE_INLINE void zObjGraph::drawGraph()
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
				vector<zIntArray> edgeVertices;
				edgeVertices.assign(graph.edges.size(), zIntArray(2) = { -1,-1 });

				for (auto &e : graph.edges)
				{

					if (graph.eHandles[e.getId()].id != -1)
					{
						zIntArray eVerts;
						edgeVertices[e.getId()][0] = e.getHalfEdge(0)->getVertex()->getId();
						edgeVertices[e.getId()][1] = e.getHalfEdge(1)->getVertex()->getId();

					}


				}

				displayUtils->drawEdges(graph.eHandles, edgeVertices, &graph.vertexPositions[0], &graph.edgeColors[0], &graph.edgeWeights[0]);
			}

		}
	}

	//---- DISPLAY BUFFER METHODS

	ZSPACE_INLINE void zObjGraph::appendToBuffer()
	{
		showObject = showEdges = showVertices = false;

		// Edge Indicies
		zIntArray _edgeIndicies;

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

}