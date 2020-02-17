// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zInterface/objects/zObjGraph.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zObjGraph::zObjGraph()
	{

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
		// Do Nothing
#else
		displayUtils = nullptr;
#endif

		displayVertices = false;
		displayEdges = true;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zObjGraph::~zObjGraph() {}

	//---- SET METHODS

	ZSPACE_INLINE void zObjGraph::setDisplayElements(bool _displayVertices, bool _displayEdges)
	{
		displayVertices = _displayVertices;
		displayEdges = _displayEdges;
	}

	ZSPACE_INLINE void zObjGraph::setDisplayElementIds(bool _displayVertexIds, bool _displayEdgeIds)
	{
		displayVertexIds = _displayVertexIds;
		displayEdgeIds = _displayEdgeIds;
	}

	ZSPACE_INLINE void zObjGraph::setDisplayVertices(bool _displayVertices)
	{
		displayVertices = _displayVertices;
	}

	ZSPACE_INLINE void zObjGraph::setDisplayEdges(bool _displayEdges)
	{
		displayEdges = _displayEdges;
	}

	ZSPACE_INLINE void zObjGraph::setEdgeCenters(zPointArray &_edgeCenters)
	{
		edgeCenters = _edgeCenters;
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

	ZSPACE_INLINE void zObjGraph::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		coreUtils.getBounds(graph.vertexPositions, minBB, maxBB);
	}

#if defined (ZSPACE_UNREAL_INTEROP) || defined (ZSPACE_MAYA_INTEROP) || defined (ZSPACE_RHINO_INTEROP)
	// Do Nothing
#else
	   
	ZSPACE_INLINE void zObjGraph::draw()
	{
		if (displayObject)
		{
			drawGraph();
		}

		if (displayObjectTransform)
		{
			displayUtils->drawTransform(transformationMatrix);
		}
	}

	//---- DISPLAY BUFFER METHODS

	ZSPACE_INLINE void zObjGraph::appendToBuffer()
	{
		displayObject = displayEdges = displayVertices = false;

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
	
	//---- PROTECTED DISPLAY METHODS

	ZSPACE_INLINE void zObjGraph::drawGraph()
	{
		// draw vertex
		if (displayVertices)
		{
			displayUtils->drawVertices(graph.vHandles, &graph.vertexPositions[0], &graph.vertexColors[0], &graph.vertexWeights[0]);
		}

		// draw vertex Ids
		if (displayVertexIds)
		{
			zColor col(0.8, 0, 0, 1);
			displayUtils->drawVertexIds(graph.n_v, &graph.vertexPositions[0], col);
		}

		// draw edges
		if (displayEdges)
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

		// draw edge Ids
		if (displayEdgeIds)
		{
			if (edgeCenters.size() != graph.n_e) throw std::invalid_argument(" error: edge centers are not computed.");

			zColor col(0, 0.8, 0, 1);
			displayUtils->drawEdgeIds(graph.n_e, &edgeCenters[0], col);
		}
	}

#endif // !ZSPACE_UNREAL_INTEROP
}