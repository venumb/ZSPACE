#pragma once

#include<headers/api/iterators/zIt.h>
#include<headers/api/object/zObjGraph.h>

namespace zSpace
{
	class zItGraphVertex;
	class zItGraphEdge;
	class zItGraphHalfEdge;


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphVertex
	*	\brief The graph vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItGraphVertex : public zIt
	{
	protected:

		zItVertex iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphVertex()
		{
			graphObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphVertex(zObjGraph &_meshObj)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.vertices.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\param		[in]	_index				- input index in graph vertex list.
		*	\since version 0.0.3
		*/
		zItGraphVertex(zObjGraph &_meshObj, int _index)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.vertices.begin();

			if (_index < 0 && _index >= graphObj->graph.vertices.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = graphObj->graph.vertices.begin();
		}

		virtual void next() override
		{
			iter++;
		}

		virtual void prev() override
		{
			iter--;
		}

		virtual bool end() override
		{
			return (iter == graphObj->graph.vertices.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = graphObj->graph.vertices.begin();
		}


		virtual int size() override
		{

			return graphObj->graph.vertices.size();
		}

		//--------------------------
		//---- TOPOLOGY QUERY METHODS
		//--------------------------

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge iterator.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<zItGraphHalfEdge>& halfedges);


		/*! \brief This method gets the indicies of halfedges connected to the iterator.
		*
		*	\param		[out]	halfedgeIndicies	- vector of halfedge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<int>& halfedgeIndicies);

		/*! \brief This method gets the edges connected to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge iterator.
		*	\since version 0.0.3
		*/
		void getConnectedEdges(vector<zItGraphEdge>& edges);


		/*! \brief This method gets the indicies of edges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedEdges(vector<int>& edgeIndicies);


		/*! \brief This method gets the vertices connected to the iterator.
		*
		*	\param		[out]	verticies	- vector of verticies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(vector<zItGraphVertex>& verticies);


		/*! \brief This method gets the indicies of vertices connected to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(vector<int>& vertexIndicies);
			   		 
		/*!	\brief This method calculate the valency of the vertex.
		*
		*	\return				int		- valency of input vertex input valence.
		*	\since version 0.0.3
		*/
		int getVertexValence();


		/*!	\brief This method determines if vertex valency is equal to the input valence number.
		*
		*	\param		[in]	valence	- input valence value.
		*	\return				bool	- true if valency is equal to input valence.
		*	\since version 0.0.3
		*/
		bool checkVertexValency(int valence = 1);


		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId()
		{
			return iter->getId();
		}

		/*! \brief This method gets the half edge attached to the vertex.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getHalfEdge();

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItVertex		- raw iterator
		*	\since version 0.0.3
		*/
		zItVertex  getRawIter()
		{
			return iter;
		}

		/*! \brief This method gets position of the vertex.
		*
		*	\return				zVector					- vertex position.
		*	\since version 0.0.3
		*/
		zVector getVertexPosition()
		{
			return graphObj->graph.vertexPositions[getId()];
		}

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zVector* getRawVertexPosition()
		{
			return &graphObj->graph.vertexPositions[getId()];
		}

		
		/*! \brief This method gets color of the vertex.
		*
		*	\return				zColor					- vertex color.
		*	\since version 0.0.3
		*/
		zColor getVertexColor()
		{
			return graphObj->graph.vertexColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the vertex.
		*
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.3
		*/
		zColor* getRawVertexColor()
		{
			return &graphObj->graph.vertexColors[getId()];
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id)
		{
			iter->setId(_id);
		}

		/*! \brief This method sets the vertex half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItGraphHalfEdge &he);

		/*! \brief This method sets position of the vertex.
		*
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.3
		*/
		void setVertexPosition(zVector &pos)
		{
			graphObj->graph.vertexPositions[getId()] = pos;
		}

		/*! \brief This method sets color of the vertex.
		*
		*	\param		[in]	col						- vertex color.
		*	\since version 0.0.3
		*/
		void setVertexColor(zColor col)
		{
			graphObj->graph.vertexColors[getId()] = col;
		}

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive()
		{
			return iter->isActive();
		}


		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItGraphVertex &other)
		{
			return (getId() == other.getId());
		}


		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphVertex &other)
		{
			return (getId() != other.getId());
		}



	};


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphHalfEdge
	*	\brief The graph half edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItGraphHalfEdge : public zIt
	{
	protected:

		zItHalfEdge iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge()
		{
			graphObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge(zObjGraph &_meshObj)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.halfEdges.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\param		[in]	_index				- input index in graph halfedge list.
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge(zObjGraph &_meshObj, int _index)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.halfEdges.begin();

			if (_index < 0 && _index >= graphObj->graph.halfEdges.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = graphObj->graph.halfEdges.begin();
		}

		virtual void next() override
		{
			iter++;
		}

		virtual void prev() override
		{
			iter--;
		}

		virtual bool end() override
		{
			return (iter == graphObj->graph.halfEdges.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = graphObj->graph.halfEdges.begin();
		}


		virtual int size() override
		{

			return graphObj->graph.halfEdges.size();
		}


		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the vertex pointed by the symmetry of input iterator.
		*
		*	\return				zItGraphVertex	- iterator to vertex.
		*	\since version 0.0.3
		*/
		zItGraphVertex getStartVertex()
		{
			if (!isActive()) throw std::invalid_argument(" error: out of bounds.");

			return getSym().getVertex();
		}


		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(vector<zItGraphVertex> &verticies)
		{
			verticies.push_back(getVertex());
			verticies.push_back(getSym().getVertex());
		}

		/*!	\brief This method gets the indicies of the vertices attached to halfedge.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(vector<int> &vertexIndicies)
		{
			vertexIndicies.push_back(getVertex().getId());
			vertexIndicies.push_back(getSym().getVertex().getId());
		}

		/*!	\brief TThis method gets the vertex positions attached to the iterator and its symmetry.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions)
		{
			vector<int> eVerts;

			getVertices(eVerts);

			for (int i = 0; i < eVerts.size(); i++)
			{
				vertPositions.push_back(graphObj->graph.vertexPositions[eVerts[i]]);
			}
		}

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<zItGraphHalfEdge>& edgeIndicies)
		{
			zItGraphVertex v1 = getVertex();
			vector<zItGraphHalfEdge> connectedEdgestoVert0;
			v1.getConnectedHalfEdges(connectedEdgestoVert0);

			zItGraphVertex v2 = getSym().getVertex();
			vector<zItGraphHalfEdge> connectedEdgestoVert1;
			v2.getConnectedHalfEdges(connectedEdgestoVert1);

			for (int i = 0; i < connectedEdgestoVert0.size(); i++)
			{
				if (connectedEdgestoVert0[i].getId() != getId()) edgeIndicies.push_back(connectedEdgestoVert0[i]);
			}


			for (int i = 0; i < connectedEdgestoVert1.size(); i++)
			{
				if (connectedEdgestoVert1[i].getId() != getId()) edgeIndicies.push_back(connectedEdgestoVert1[i]);
			}
		}

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<int>& edgeIndicies)
		{
			zItGraphVertex v1 = getVertex();
			vector<int> connectedEdgestoVert0;
			v1.getConnectedHalfEdges(connectedEdgestoVert0);

			zItGraphVertex v2 = getSym().getVertex();
			vector<int> connectedEdgestoVert1;
			v2.getConnectedHalfEdges(connectedEdgestoVert1);

			for (int i = 0; i < connectedEdgestoVert0.size(); i++)
			{
				if (connectedEdgestoVert0[i] != getId()) edgeIndicies.push_back(connectedEdgestoVert0[i]);
			}


			for (int i = 0; i < connectedEdgestoVert1.size(); i++)
			{
				if (connectedEdgestoVert1[i] != getId()) edgeIndicies.push_back(connectedEdgestoVert1[i]);
			}
		}
		

		/*! \brief This method checks if the half edge is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary()
		{
			return !iter->getFace();
		}

		/*! \brief This method computes the centers of a the half edge.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter()
		{
			vector<int> eVerts;
			getVertices(eVerts);

			return (graphObj->graph.vertexPositions[eVerts[0]] + graphObj->graph.vertexPositions[eVerts[1]]) * 0.5;
		}

		/*! \brief This method gets the vector of the half edge.
		*
		*	\return				zVector					- edge vector.
		*	\since version 0.0.3
		*/
		zVector getHalfEdgeVector()
		{

			int v1 = getVertex().getId();
			int v2 = getSym().getVertex().getId();

			zVector out = graphObj->graph.vertexPositions[v1] - (graphObj->graph.vertexPositions[v2]);

			return out;
		}

		/*! \brief This method computes the edge length of the half edge.
		*
		*	\return				double			- half edge length.
		*	\since version 0.0.3
		*/
		double getHalfEdgelength()
		{
			return getHalfEdgeVector().length();
		}


		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId()
		{
			return iter->getId();
		}

		/*! \brief This method gets the symmetry half edge attached to the halfedge.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getSym()
		{
			return zItGraphHalfEdge(*graphObj, iter->getSym()->getId());
		}

		/*! \brief This method gets the next half edge attached to the halfedge.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getNext()
		{
			return zItGraphHalfEdge(*graphObj, iter->getNext()->getId());
		}

		/*! \brief This method gets the prev half edge attached to the halfedge.
		*
		*	\return			zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getPrev()
		{
			return zItGraphHalfEdge(*graphObj, iter->getPrev()->getId());
		}

		/*! \brief This method gets the vertex attached to the halfedge.
		*
		*	\return			zItGraphVertex		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphVertex getVertex();


		/*! \brief This method gets the edge attached to the halfedge.
		*
		*	\return			zItGraphEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphEdge getEdge();

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItHalfEdge		- raw iterator
		*	\since version 0.0.3
		*/
		zItHalfEdge  getRawIter()
		{
			return iter;
		}

		/*! \brief This method gets color of the halfedge.
		*
		*	\return				zColor					- halfedge color.
		*	\since version 0.0.3
		*/
		zColor getHalfEdgeColor()
		{
			return graphObj->graph.edgeColors[iter->getEdge()->getId()];
		}

		/*! \brief This method gets pointer to the color of the halfedge.
		*
		*	\return				zColor*					- halfedge color.
		*	\since version 0.0.3
		*/
		zColor* getRawHalfEdgeColor()
		{
			return &graphObj->graph.edgeColors[iter->getEdge()->getId()];
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id)
		{
			iter->setId(_id);
		}

		/*! \brief This method sets the symmetry half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setSym(zItGraphHalfEdge &he);

		/*! \brief This method sets the next half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setNext(zItGraphHalfEdge &he);

		/*! \brief This method sets the previous half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setPrev(zItGraphHalfEdge &he);

		/*! \brief This method sets the half edge vertex to the input vertex.
		*
		*	\param	[in]	v		- input vertex iterator
		*	\since version 0.0.3
		*/
		void setVertex(zItGraphVertex &v);

		/*! \brief This method sets the halfedge edge to the input edge.
		*
		*	\param	[in]	e		- input edge iterator
		*	\since version 0.0.3
		*/
		void setEdge(zItGraphEdge &e);


		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method checks if the half edge is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive()
		{
			return iter->isActive();
		}


		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItGraphHalfEdge &other)
		{
			return (getId() == other.getId());
		}

		/*! \brief This operator checks for non equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphHalfEdge &other)
		{
			return (getId() != other.getId());
		}
	};


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zGraphIterators
	*	\brief The graph iterator classes of the library.
	*  @{
	*/

	/*! \class zItGraphEdge
	*	\brief The graph  edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItGraphEdge : public zIt
	{
	protected:

		zItEdge iter;

		/*!	\brief pointer to a graph object  */
		zObjGraph *graphObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.3
		*/
		zItGraphEdge()
		{
			graphObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\since version 0.0.3
		*/
		zItGraphEdge(zObjGraph &_meshObj)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.edges.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input graph object.
		*	\param		[in]	_index				- input index in graph edge list.
		*	\since version 0.0.3
		*/
		zItGraphEdge(zObjGraph &_meshObj, int _index)
		{
			graphObj = &_meshObj;

			iter = graphObj->graph.edges.begin();

			if (_index < 0 && _index >= graphObj->graph.edges.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = graphObj->graph.edges.begin();
		}

		virtual void next() override
		{
			iter++;
		}

		virtual void prev() override
		{
			iter--;
		}

		virtual bool end() override
		{
			return (iter == graphObj->graph.edges.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = graphObj->graph.edges.begin();
		}

		virtual int size() override
		{

			return graphObj->graph.edges.size();
		}

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	verticies	- vector of vertex iterators.
		*	\since version 0.0.3
		*/
		void getVertices(vector<zItGraphVertex> &verticies)
		{
			verticies.push_back(getHalfEdge(0).getVertex());
			verticies.push_back(getHalfEdge(1).getVertex());
		}

		/*!	\brief This method gets the indicies of the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(vector<int> &vertexIndicies)
		{
			vertexIndicies.push_back(getHalfEdge(0).getVertex().getId());
			vertexIndicies.push_back(getHalfEdge(1).getVertex().getId());
		}

		/*!	\brief TThis method gets the vertex positions attached to the iterator.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions)
		{
			vector<int> eVerts;

			getVertices(eVerts);

			for (int i = 0; i < eVerts.size(); i++)
			{
				vertPositions.push_back(graphObj->graph.vertexPositions[eVerts[i]]);
			}
		}
		


		/*! \brief This method computes the centers of a the edge.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter()
		{
			vector<int> eVerts;
			getVertices(eVerts);

			return (graphObj->graph.vertexPositions[eVerts[0]] + graphObj->graph.vertexPositions[eVerts[1]]) * 0.5;
		}

		/*! \brief This method gets the vector of the edge.
		*
		*	\return				zVector					- edge vector.
		*	\since version 0.0.3
		*/
		zVector getEdgeVector()
		{

			int v1 = getHalfEdge(0).getVertex().getId();
			int v2 = getHalfEdge(1).getVertex().getId();

			zVector out = graphObj->graph.vertexPositions[v1] - (graphObj->graph.vertexPositions[v2]);

			return out;
		}

		/*! \brief This method gets the edge length of the edge.
		*
		*	\return				double			- edge length.
		*	\since version 0.0.3
		*/
		double getEdgeLength()
		{
			return getEdgeVector().length();
		}

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the index of the iterator.
		*
		*	\return			int		- iterator index
		*	\since version 0.0.3
		*/
		int getId()
		{
			return iter->getId();
		}

		/*! \brief This method gets the half edge attached to the edge.
		*
		*	\param		[in]	_index				- input index ( 0 or 1).
		*	\return				zItGraphHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItGraphHalfEdge getHalfEdge(int _index);

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItEdge		- raw iterator
		*	\since version 0.0.3
		*/
		zItEdge  getRawIter()
		{
			return iter;
		}

		/*! \brief This method gets color of the edge.
		*
		*	\return				zColor					- edge color.
		*	\since version 0.0.3
		*/
		zColor getEdgeColor()
		{
			return graphObj->graph.edgeColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the edge.
		*
		*	\return				zColor*					- edge color.
		*	\since version 0.0.3
		*/
		zColor* getRawEdgeColor()
		{
			return &graphObj->graph.edgeColors[getId()];
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the index of the iterator.
		*
		*	\param	[in]	_id		- input index
		*	\since version 0.0.3
		*/
		void setId(int _id)
		{
			iter->setId(_id);
		}

		/*! \brief This method sets the edge half edge to the input half edge.
		*
		*	\param		[in]	he		- input half edge iterator
		*	\param		[in]	_index	- input index ( 0 or 1).
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItGraphHalfEdge &he, int _index);
		
		/*! \brief This method sets color of the edge.
		*
		*	\param		[in]	col						- input color.
		*	\since version 0.0.3
		*/
		void setEdgeColor(zColor col)
		{
			graphObj->graph.edgeColors[getId()] = col;
		}

		/*! \brief This method sets display weight of the edge.
		*
		*	\param		[in]	wt						- input weight.
		*	\since version 0.0.3
		*/
		void setEdgeWeight(double wt)
		{
			graphObj->graph.edgeWeights[getId()] = wt;
		}

		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method gets if the vertex is active.
		*
		*	\return			bool		- true if active else false.
		*	\since version 0.0.3
		*/
		bool isActive()
		{
			return iter->isActive();
		}

		//--------------------------
		//---- OPERATOR METHODS
		//--------------------------

		/*! \brief This operator checks for equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItGraphEdge &other)
		{
			return (getId() == other.getId());
		}

		/*! \brief This operator checks for non equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItGraphEdge &other)
		{
			return (getId() != other.getId());
		}



	};
	



	//--------------------------
	//--- INLINE VERTEX METHODS 
	//--------------------------

	//getHalfEdge
	inline zItGraphHalfEdge zSpace::zItGraphVertex::getHalfEdge()
	{
		return zItGraphHalfEdge(*graphObj, iter->getHalfEdge()->getId());
	}

	// setHalfEdge 
	inline void zSpace::zItGraphVertex::setHalfEdge(zItGraphHalfEdge &he)
	{
		iter->setHalfEdge(&graphObj->graph.halfEdges[he.getId()]);
	}

	// getConnectedHalfEdges 
	inline void zSpace::zItGraphVertex::getConnectedHalfEdges(vector<zItGraphHalfEdge>& halfedges)
	{
		if (!getHalfEdge().isActive()) return;

		zItGraphHalfEdge start = getHalfEdge();
		zItGraphHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedges.push_back(e);
			e = e.getPrev().getSym();

		} while (e != start);
	}

	// getConnectedHalfEdges 
	inline void zSpace::zItGraphVertex::getConnectedHalfEdges(vector<int>& halfedgeIndicies)
	{
		if (!getHalfEdge().isActive()) return;


		zItGraphHalfEdge start = getHalfEdge();
		zItGraphHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedgeIndicies.push_back(e.getId());
			e = e.getPrev().getSym();

		} while (e != start);
	}

	// getConnectedEdges 
	inline void zSpace::zItGraphVertex::getConnectedEdges(vector<zItGraphEdge>& edges)
	{
		vector<zItGraphHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edges.push_back(he.getEdge());
		}
	}

	// getConnectedEdges 
	inline void zSpace::zItGraphVertex::getConnectedEdges(vector<int>& edgeIndicies)
	{
		vector<zItGraphHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edgeIndicies.push_back(he.getEdge().getId());
		}
	}

	//getConnectedVertices
	inline void zSpace::zItGraphVertex::getConnectedVertices(vector<zItGraphVertex>& verticies)
	{
		vector<zItGraphHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			verticies.push_back(he.getVertex());
		}
	}

	//getConnectedVertices
	inline void zSpace::zItGraphVertex::getConnectedVertices(vector<int>& vertexIndicies)
	{
		vector<zItGraphHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			vertexIndicies.push_back(he.getVertex().getId());
		}
	}
	


	//getVertexValence
	inline int zSpace::zItGraphVertex::getVertexValence()
	{
		int out;

		vector<int> cHEdges;
		getConnectedHalfEdges(cHEdges);

		out = cHEdges.size();

		return out;
	}

	//checkVertexValency
	inline bool zSpace::zItGraphVertex::checkVertexValency(int valence)
	{
		bool out = false;
		out = (getVertexValence() == valence) ? true : false;

		return out;
	}

	//--------------------------
	//--- INLINE HALF EDGE METHODS 
	//--------------------------

	//getVertex
	inline zItGraphVertex zSpace::zItGraphHalfEdge::getVertex()
	{
		return zItGraphVertex(*graphObj, iter->getVertex()->getId());
	}
	

	//getEdge
	inline zItGraphEdge zSpace::zItGraphHalfEdge::getEdge()
	{
		return zItGraphEdge(*graphObj, iter->getEdge()->getId());
	}

	//setSym
	inline void zItGraphHalfEdge::setSym(zItGraphHalfEdge & he)
	{
		iter->setSym(&graphObj->graph.halfEdges[he.getId()]);
	}

	//setNext
	inline void zItGraphHalfEdge::setNext(zItGraphHalfEdge & he)
	{
		iter->setNext(&graphObj->graph.halfEdges[he.getId()]);
	}

	//setPrev
	inline void zItGraphHalfEdge::setPrev(zItGraphHalfEdge & he)
	{
		iter->setPrev(&graphObj->graph.halfEdges[he.getId()]);
	}

	//setVertex
	inline void zItGraphHalfEdge::setVertex(zItGraphVertex & v)
	{
		iter->setVertex(&graphObj->graph.vertices[v.getId()]);
	}

	//setEdge
	inline void zItGraphHalfEdge::setEdge(zItGraphEdge & e)
	{
		iter->setEdge(&graphObj->graph.edges[e.getId()]);
	}


	//--------------------------
	//--- INLINE EDGE METHODS 
	//--------------------------

	//getHalfEdge
	zItGraphHalfEdge zSpace::zItGraphEdge::getHalfEdge(int _index)
	{
		return zItGraphHalfEdge(*graphObj, iter->getHalfEdge(_index)->getId());
	}

	//setHalfEdge
	inline void zItGraphEdge::setHalfEdge(zItGraphHalfEdge & he, int _index)
	{
		iter->setHalfEdge(&graphObj->graph.halfEdges[he.getId()], _index);
	}





}