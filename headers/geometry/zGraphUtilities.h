#pragma once

#include <headers/geometry/zGraph.h>


namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief  The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryUtilities
	*	\brief Collection of utility methods for graphs, meshes and fields.
	*  @{
	*/

	/** \addtogroup zGraphUtilities
	*	\brief Collection of utility methods for graphs.
	*  @{
	*/

	/*! \brief This method returns the total edge length of the graph
	*
	*	\param		[in]	inGraph	- input graph.
	*	\since version 0.0.1
	*/
	inline double totalEdgeLength(zGraph &inGraph)
	{
		double out = 0.0;

		for (int i = 0; i < inGraph.edgeActive.size(); i += 2)
		{
			if (inGraph.edgeActive[i])
			{
				int v1 = inGraph.edges[i].getVertex()->getVertexId();
				int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

				zVector posV2 = inGraph.vertexPositions[v2];
				double dist = inGraph.vertexPositions[v1].distanceTo(posV2);
				//printf("\n %1.2f ", dist);

				out += dist;
			}
		}

		return out;
	}
	
	/*! \brief This method sets vertex color of all the vertices to the input color. 
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.	
	*	\since version 0.0.1
	*/	
	inline void setVertexColor(zGraph &inGraph, zColor col, bool setEdgeColor = false)
	{
		if (inGraph.vertexColors.size() != inGraph.vertexActive.size())
		{
			inGraph.vertexColors.clear();
			for (int i = 0; i < inGraph.vertexActive.size(); i++) inGraph.vertexColors.push_back(zColor(1, 0, 0, 1));
		}

		for (int i = 0; i < inGraph.vertexColors.size(); i++)
		{
			inGraph.vertexColors[i] = col;
		}

		if (setEdgeColor) inGraph.computeEdgeColorfromVertexColor();

	}

	/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the graph.
	*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/
	inline void setVertexColors(zGraph &inGraph, vector<zColor>& col, bool setEdgeColor = false)
	{
		if (inGraph.vertexColors.size() != inGraph.vertexActive.size())
		{
			inGraph.vertexColors.clear();
			for (int i = 0; i < inGraph.vertexActive.size(); i++) inGraph.vertexColors.push_back(zColor(1, 0, 0, 1));
		}

		if (col.size() != inGraph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < inGraph.vertexColors.size(); i++)
		{
			inGraph.vertexColors[i] = col[i];
		}

		if (setEdgeColor) inGraph.computeEdgeColorfromVertexColor();
	}

	/*! \brief This method sets edge color of all the edges to the input color.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/
	inline void setEdgeColor(zGraph & inGraph, zColor col, bool setVertexColor = false)
	{
		if (inGraph.edgeColors.size() != inGraph.edgeActive.size())
		{
			inGraph.edgeColors.clear();
			for (int i = 0; i < inGraph.edgeActive.size(); i++) inGraph.edgeColors.push_back(zColor());
		}

		for (int i = 0; i < inGraph.edgeActive.size(); i++)
		{
			 inGraph.edgeColors[i] = col;
		}

		if (setVertexColor) inGraph.computeVertexColorfromEdgeColor();

	}

	/*! \brief This method sets edge color of all the edges with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of edges in the graph.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/
	inline void setEdgeColors(zGraph & inGraph, vector<zColor>& col, bool setVertexColor = false)
	{
		if (inGraph.edgeColors.size() != inGraph.edgeActive.size())
		{
			inGraph.edgeColors.clear();
			for (int i = 0; i < inGraph.edgeActive.size(); i++) inGraph.edgeColors.push_back(zColor());
		}

		if (col.size() != inGraph.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph edges.");

		for (int i = 0; i < inGraph.edgeColors.size(); i++)
		{
			inGraph.edgeColors[i] = col[i];
		}

		if (setVertexColor) inGraph.computeVertexColorfromEdgeColor();
	}
	

	


	/** @}*/
	
	/** @}*/

	/** @}*/
}