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
	inline void setVertexColor(zGraph &ingraph, zColor col, bool setEdgeColor)
	{

		for (int i = 0; i < ingraph.vertexColors.size(); i++)
		{
			ingraph.vertexColors[i] = col;
		}

		if (setEdgeColor) ingraph.computeEdgeColorfromVertexColor();

	}

	/*! \brief This method sets vertex color of all the vertices with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of vertices in the graph.
	*	\param		[in]	setEdgeColor	- edge color is computed based on the vertex color if true.
	*	\since version 0.0.1
	*/
	inline void setVertexColors(zGraph &ingraph, vector<zColor>& col, bool setEdgeColor)
	{
		if (col.size() != ingraph.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph vertices.");

		for (int i = 0; i < ingraph.vertexColors.size(); i++)
		{
			ingraph.vertexColors[i] = col[i];
		}

		if (setEdgeColor) ingraph.computeEdgeColorfromVertexColor();
	}

	/*! \brief This method sets edge color of all the edges to the input color.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/
	inline void setEdgeColor(zGraph & ingraph, zColor col, bool setVertexColor)
	{
		for (int i = 0; i < ingraph.edgeActive.size(); i++)
		{
			if (i >= ingraph.edgeColors.size()) ingraph.edgeColors.push_back(col);
			else ingraph.edgeColors[i] = col;
		}

		if (setVertexColor) ingraph.computeVertexColorfromEdgeColor();

	}

	/*! \brief This method sets edge color of all the edges with the input color contatiner.
	*
	*	\param		[in]	inGraph			- input graph.
	*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of edges in the graph.
	*	\param		[in]	setVertexColor	- vertex color is computed based on the edge color if true.
	*	\since version 0.0.1
	*/
	inline void setEdgeColors(zGraph & ingraph, vector<zColor>& col, bool setVertexColor)
	{
		if (col.size() != ingraph.edgeColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of graph edges.");

		for (int i = 0; i < ingraph.edgeColors.size(); i++)
		{
			ingraph.edgeColors[i] = col[i];
		}

		if (setVertexColor) ingraph.computeVertexColorfromEdgeColor();
	}
	

	


	/** @}*/
	
	/** @}*/

	/** @}*/
}