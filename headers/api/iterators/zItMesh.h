#pragma once

#include<headers/api/iterators/zIt.h>
#include<headers/api/object/zObjMesh.h>

namespace zSpace
{
	class zItMeshVertex;
	class zItMeshEdge;
	class zItMeshHalfEdge;
	class zItMeshFace;
	
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zIterators
	*	\brief The iterators classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshVertex
	*	\brief The mesh vertex iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItMeshVertex : public zIt
	{
	protected:

		zItVertex iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshVertex()
		{
			meshObj = nullptr;
		}
	
		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshVertex(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.vertices.begin();			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh vertex list.
		*	\since version 0.0.3
		*/
		zItMeshVertex(zObjMesh &_meshObj, int _index)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.vertices.begin();

			if (_index < 0 && _index >= meshObj->mesh.vertices.size()) throw std::invalid_argument(" error: index out of bounds"); 
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = meshObj->mesh.vertices.begin();
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
			return (iter == meshObj->mesh.vertices.end()) ? true : false;
		}


		virtual void reset() override 
		{
			iter = meshObj->mesh.vertices.begin();
		}		
		

		virtual int size() override
		{		

			return meshObj->mesh.vertices.size();
		}

		virtual void deactivate() override 
		{
			meshObj->mesh.vHandles[iter->getId()] = zVertexHandle();
			iter->reset();

			
		}

		//--------------------------
		//---- TOPOLOGY QUERY METHODS
		//--------------------------
	
		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge iterator.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<zItMeshHalfEdge>& halfedges);
		

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
		void getConnectedEdges(vector<zItMeshEdge>& edges);


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
		void getConnectedVertices(vector<zItMeshVertex>& verticies);
		

		/*! \brief This method gets the indicies of vertices connected to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getConnectedVertices(vector<int>& vertexIndicies);		

		/*! \brief This method gets the faces connected to the iterator.
		*
		*	\param		[out]	faces	- vector of faces.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(vector<zItMeshFace>& faces);

		/*! \brief This method gets the indicies of faces connected to the iterator.
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(vector<int>& faceIndicies);

		/*!	\brief This method determines if  the element is on the boundary.
		*
		*	\return				bool	- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary();
		

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
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge();	

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
			return meshObj->mesh.vertexPositions[getId()];
		}

		/*! \brief This method gets pointer to the position of the vertex.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.3
		*/
		zVector* getRawVertexPosition()
		{
			return &meshObj->mesh.vertexPositions[getId()];
		}

		/*! \brief This method gets normal of the vertex.
		*
		*	\return				zVector					- vertex normal.
		*	\since version 0.0.3
		*/
		zVector getVertexNormal()
		{
			return meshObj->mesh.vertexNormals[getId()];
		}

		/*! \brief This method gets pointer to the normal of the vertex.
		*
		*	\return				zVector*				- pointer to internal vertex normal.
		*	\since version 0.0.3
		*/
		zVector* getRawVertexNormal()
		{
			return &meshObj->mesh.vertexNormals[getId()];
		}

		/*! \brief This method gets color of the vertex.
		*
		*	\return				zColor					- vertex color.
		*	\since version 0.0.3
		*/
		zColor getVertexColor()
		{
			return meshObj->mesh.vertexColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the vertex.
		*
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.3
		*/
		zColor* getRawVertexColor()
		{
			return &meshObj->mesh.vertexColors[getId()];
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
		void setHalfEdge(zItMeshHalfEdge &he);

		/*! \brief This method sets position of the vertex.
		*
		*	\param		[in]	pos						- vertex position.
		*	\since version 0.0.3
		*/
		void setVertexPosition(zVector &pos)
		{
			meshObj->mesh.vertexPositions[getId()] = pos;
		}

		/*! \brief This method sets color of the vertex.
		*
		*	\param		[in]	col						- vertex color.
		*	\since version 0.0.3
		*/
		void setVertexColor(zColor &col)
		{
			meshObj->mesh.vertexColors[getId()] = col;
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
		bool operator==(zItMeshVertex &other)
		{
			return (getId() == other.getId());
		}


		/*! \brief This operator checks for non equality of two vertex iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshVertex &other)
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

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshHalfEdge
	*	\brief The mesh half edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItMeshHalfEdge : public zIt
	{
	protected:

		zItHalfEdge iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge()
		{
			meshObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.halfEdges.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh halfedge list.
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge(zObjMesh &_meshObj, int _index)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.halfEdges.begin();

			if (_index < 0 && _index >= meshObj->mesh.halfEdges.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = meshObj->mesh.halfEdges.begin();
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
			return (iter == meshObj->mesh.halfEdges.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = meshObj->mesh.halfEdges.begin();
		}
	

		virtual int size() override
		{

			return meshObj->mesh.halfEdges.size();
		}

		virtual void deactivate() override
		{
			meshObj->mesh.heHandles[iter->getId()] = zHalfEdgeHandle();
			iter->reset();
		
		}

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the vertex pointed by the symmetry of input iterator.
		*	
		*	\return				zItMeshVertex	- iterator to vertex.
		*	\since version 0.0.3
		*/
		zItMeshVertex getStartVertex()
		{
			if (!isActive()) throw std::invalid_argument(" error: out of bounds.");

			return getSym().getVertex();
		}
		

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(vector<zItMeshVertex> &verticies)
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
				vertPositions.push_back(meshObj->mesh.vertexPositions[eVerts[i]]);
			}
		}

		/*! \brief This method gets the halfedges connected to the iterator.
		*
		*	\param		[out]	edgeIndicies	- vector of edge indicies.
		*	\since version 0.0.3
		*/
		void getConnectedHalfEdges(vector<zItMeshHalfEdge>& edgeIndicies)
		{
			zItMeshVertex v1 = getVertex();
			vector<zItMeshHalfEdge> connectedEdgestoVert0;
			v1.getConnectedHalfEdges(connectedEdgestoVert0);

			zItMeshVertex v2 = getSym().getVertex();
			vector<zItMeshHalfEdge> connectedEdgestoVert1;
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
			zItMeshVertex v1 = getVertex();
			vector<int> connectedEdgestoVert0;
			v1.getConnectedHalfEdges(connectedEdgestoVert0);

			zItMeshVertex v2 = getSym().getVertex();
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

		/*! \brief This method gets the faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(vector<zItMeshFace> &faceIndicies);		

		/*! \brief This method gets the indicies of faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(vector<int> &faceIndicies);		

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

			return (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;
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

			zVector out = meshObj->mesh.vertexPositions[v1] - (meshObj->mesh.vertexPositions[v2]);

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
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getSym()
		{
			return zItMeshHalfEdge(*meshObj, iter->getSym()->getId());
		}

		/*! \brief This method gets the next half edge attached to the halfedge.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getNext()
		{
			return zItMeshHalfEdge(*meshObj, iter->getNext()->getId());
		}

		/*! \brief This method gets the prev half edge attached to the halfedge.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getPrev()
		{
			return zItMeshHalfEdge(*meshObj, iter->getPrev()->getId());
		}

		/*! \brief This method gets the vertex attached to the halfedge.
		*
		*	\return			zItMeshVertex		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshVertex getVertex();
		
		
		/*! \brief This method gets the face attached to the halfedge.
		*
		*	\return			zItMeshFace		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshFace getFace();
	
		/*! \brief This method gets the edge attached to the halfedge.
		*
		*	\return			zItMeshEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshEdge getEdge();
		
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
			return meshObj->mesh.edgeColors[iter->getEdge()->getId()];
		}

		/*! \brief This method gets pointer to the color of the halfedge.
		*
		*	\return				zColor*					- halfedge color.
		*	\since version 0.0.3
		*/
		zColor* getRawHalfEdgeColor()
		{
			return &meshObj->mesh.edgeColors[iter->getEdge()->getId()];
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
		void setSym(zItMeshHalfEdge &he);

		/*! \brief This method sets the next half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setNext(zItMeshHalfEdge &he);

		/*! \brief This method sets the previous half edge to the input half edge.
		*
		*	\param	[in]	he		- input half edge iterator
		*	\since version 0.0.3
		*/
		void setPrev(zItMeshHalfEdge &he);

		/*! \brief This method sets the half edge vertex to the input vertex.
		*
		*	\param	[in]	v		- input vertex iterator
		*	\since version 0.0.3
		*/
		void setVertex(zItMeshVertex &v);

		/*! \brief This method sets the halfedge edge to the input edge.
		*
		*	\param	[in]	e		- input edge iterator
		*	\since version 0.0.3
		*/
		void setEdge(zItMeshEdge &e);

		/*! \brief This method sets the halfedge face to the input edge.
		*
		*	\param	[in]	f		- input face iterator
		*	\since version 0.0.3
		*/
		void setFace(zItMeshFace &f);

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
		bool operator==(zItMeshHalfEdge &other)
		{
			return (getId() == other.getId());
		}

		/*! \brief This operator checks for non equality of two halfedge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshHalfEdge &other)
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

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshEdge
	*	\brief The mesh  edge iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItMeshEdge : public zIt
	{
	protected:

		zItEdge iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshEdge()
		{
			meshObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshEdge(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.edges.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh edge list.
		*	\since version 0.0.3
		*/
		zItMeshEdge(zObjMesh &_meshObj, int _index)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.edges.begin();

			if (_index < 0 && _index >= meshObj->mesh.edges.size()) throw std::invalid_argument(" error: index out of bounds");
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = meshObj->mesh.edges.begin();
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
			return (iter == meshObj->mesh.edges.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = meshObj->mesh.edges.begin();
		}		

		virtual int size() override
		{

			return meshObj->mesh.edges.size();
		}

		virtual void deactivate() override
		{
			meshObj->mesh.eHandles[iter->getId()] = zEdgeHandle();
			iter->reset();

			
		}

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	verticies	- vector of vertex iterators.
		*	\since version 0.0.3
		*/
		void getVertices(vector<zItMeshVertex> &verticies)
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
				vertPositions.push_back(meshObj->mesh.vertexPositions[eVerts[i]]);
			}
		}

		/*! \brief This method gets the faces attached to the iterator
		*
		*	\param		[out]	faces	- vector of face iterators.
		*	\since version 0.0.3
		*/
		void getFaces(vector<zItMeshFace> &faces);
		

		/*! \brief This method gets the indicies of faces attached to the iterator and its symmetry
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getFaces(vector<int> &faceIndicies);
		

		/*! \brief This method checks if the edge is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary()
		{
			return (getHalfEdge(0).onBoundary() || getHalfEdge(1).onBoundary());
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

			return (meshObj->mesh.vertexPositions[eVerts[0]] + meshObj->mesh.vertexPositions[eVerts[1]]) * 0.5;
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

			zVector out = meshObj->mesh.vertexPositions[v1] - (meshObj->mesh.vertexPositions[v2]);

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
		*	\return				zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge(int _index);

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
			return meshObj->mesh.edgeColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the edge.
		*
		*	\return				zColor*					- edge color.
		*	\since version 0.0.3
		*/
		zColor* getRawEdgeColor()
		{
			return &meshObj->mesh.edgeColors[getId()];
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
		void setHalfEdge(zItMeshHalfEdge &he, int _index);

		/*! \brief This method sets color of the edge.
		*
		*	\param		[in]	col						- input color.
		*	\since version 0.0.3
		*/
		void setEdgeColor(zColor col)
		{
			meshObj->mesh.edgeColors[getId()] = col;
		}

		/*! \brief This method sets display weight of the edge.
		*
		*	\param		[in]	wt						- input weight.
		*	\since version 0.0.3
		*/
		void setEdgeWeight(double wt)
		{
			meshObj->mesh.edgeWeights[getId()] = wt;
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
		bool operator==(zItMeshEdge &other)
		{
			return (getId() == other.getId());
		}

		/*! \brief This operator checks for non equality of two edge iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshEdge &other)
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

	/** \addtogroup zMeshIterators
	*	\brief The mesh iterator classes of the library.
	*  @{
	*/

	/*! \class zItMeshFace
	*	\brief The mesh  face iterator class.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	/** @}*/
	class zItMeshFace : public zIt
	{
	protected:

		zItFace iter;

		/*!	\brief pointer to a mesh object  */
		zObjMesh *meshObj;


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*		
		*	\since version 0.0.3
		*/
		zItMeshFace()
		{
			meshObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.3
		*/
		zItMeshFace(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.faces.begin();
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\param		[in]	_index				- input index in mesh face list.
		*	\since version 0.0.3
		*/
		zItMeshFace(zObjMesh &_meshObj, int _index)
		{
			meshObj = &_meshObj;

			iter = meshObj->mesh.faces.begin();

			if (_index < 0 && _index >= meshObj->mesh.faces.size()) throw std::invalid_argument(" error: index out of bounds"); 
			advance(iter, _index);
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		virtual void begin() override
		{
			iter = meshObj->mesh.faces.begin();
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
			return (iter == meshObj->mesh.faces.end()) ? true : false;
		}


		virtual void reset() override
		{
			iter = meshObj->mesh.faces.begin();
		}		

		virtual int size() override
		{

			return meshObj->mesh.faces.size();
		}

		virtual void deactivate() override
		{
			meshObj->mesh.fHandles[iter->getId()] = zFaceHandle();
			iter->reset();

			
		}

		//--------------------------
		//--- TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the half edges attached to the iterator.
		*
		*	\param		[out]	halfedges	- vector of halfedge interators.
		*	\since version 0.0.3
		*/
		void getHalfEdges(vector<zItMeshHalfEdge> &halfedges)
		{
			halfedges.clear();


			if (!getHalfEdge().onBoundary())
			{
				zItMeshHalfEdge start = getHalfEdge();
				zItMeshHalfEdge e = getHalfEdge();

				bool exit = false;

				do
				{
					halfedges.push_back(e);
					e = e.getNext();

				} while (e != start);
			}
			
		}

		/*! \brief This method gets the indicies of half edges attached to the iterator.
		*
		*	\param		[out]	halfedgeIndicies	- vector of halfedge indicies.
		*	\since version 0.0.3
		*/
		void getHalfEdges(vector<int> &halfedgeIndicies)
		{
			halfedgeIndicies.clear();		


			if (!getHalfEdge().onBoundary())
			{
				zItMeshHalfEdge start = getHalfEdge();
				zItMeshHalfEdge e = getHalfEdge();

				bool exit = false;

				do
				{
					halfedgeIndicies.push_back(e.getId());
					e = e.getNext();					

				} while (e != start);
			}			
		}

		/*!	\brief This method gets the the vertices attached to the iterator.
		*
		*	\param		[out]	verticies	- vector of vertex iterators.
		*	\since version 0.0.3
		*/
		void getVertices(vector<zItMeshVertex> &verticies)
		{
			vector<zItMeshHalfEdge> faceHEdges;
			getHalfEdges( faceHEdges);

			for (auto &he : faceHEdges)
			{
				verticies.push_back(he.getSym().getVertex());
			}
		}

		/*!	\brief This method gets the indicies of the vertices attached to the iterator.
		*
		*	\param		[out]	vertexIndicies	- vector of vertex indicies.
		*	\since version 0.0.3
		*/
		void getVertices(vector<int> &vertexIndicies)
		{
			vector<zItMeshHalfEdge> faceHEdges;
			getHalfEdges(faceHEdges);
						
			for (auto &he : faceHEdges)
			{
				vertexIndicies.push_back(he.getSym().getVertex().getId());
			}
		}

		/*!	\brief TThis method gets the vertex positions attached to the iterator.
		*
		*	\param		[out]	vertPositions	- vector of vertex positions.
		*	\since version 0.0.3
		*/
		void getVertexPositions(vector<zVector> &vertPositions)
		{
			vector<int> fVerts;

			getVertices(fVerts);

			for (int i = 0; i < fVerts.size(); i++)
			{
				vertPositions.push_back(meshObj->mesh.vertexPositions[fVerts[i]]);
			}
		}


		/*! \brief This method gets the faces connected to the iterator.
		*
		*	\param		[out]	faces	- vector of faces.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(vector<zItMeshFace>& faces)
		{
			vector<zItMeshHalfEdge> cHEdges;
			getHalfEdges(cHEdges);


			for (auto &he : cHEdges)
			{
				vector<zItMeshFace> eFaces;
				he.getFaces(eFaces);

				for (int k = 0; k < eFaces.size(); k++)
				{
					if (eFaces[k].getId() != getId()) faces.push_back(eFaces[k]);
				}
			}
		}

		/*! \brief This method gets the indicies of faces connected to the iterator.
		*
		*	\param		[out]	faceIndicies	- vector of face indicies.
		*	\since version 0.0.3
		*/
		void getConnectedFaces(vector<int>& faceIndicies)
		{
			vector<zItMeshHalfEdge> cHEdges;
			getHalfEdges(cHEdges);


			for (auto &he : cHEdges)
			{
				vector<int> eFaces;
				he.getFaces(eFaces);

				for (int k = 0; k < eFaces.size(); k++)
				{
					if (eFaces[k] != getId()) faceIndicies.push_back(eFaces[k]);
				}
			}
		}

		/*! \brief This method checks if the face is on boundary.
		*
		*	\return			bool		- true if on boundary else false.
		*	\since version 0.0.3
		*/
		bool onBoundary()
		{
			bool out = false;

			vector<zItMeshHalfEdge> cHEdges;
			getHalfEdges(cHEdges);


			for (auto &he : cHEdges)
			{
				if (he.onBoundary())
				{
					out = true;
					break;
				}
			}

			return out;
		}

		/*! \brief This method computes the center of a the face.
		*
		*	\return				zVector					- center.
		*	\since version 0.0.3
		*/
		zVector getCenter()
		{
			vector<int> fVerts;
			getVertices(fVerts);
			zVector cen;

			for (int j = 0; j < fVerts.size(); j++) cen += meshObj->mesh.vertexPositions[fVerts[j]];
			cen /= fVerts.size();

			return cen;
		}

		/*! \brief This method gets the number of vertices in the face.
		*	
		*	\return				int				- number of vertices in the face.
		*	\since version 0.0.3
		*/
		int getNumVertices()
		{
			vector<int> fEdges;
			getHalfEdges(fEdges);

			return fEdges.size();
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

		/*! \brief This method gets the half edge attached to the face.
		*
		*	\return			zItMeshHalfEdge		- half edge iterator
		*	\since version 0.0.3
		*/
		zItMeshHalfEdge getHalfEdge();	

		/*! \brief This method gets the raw internal iterator.
		*
		*	\return			zItFace		- raw iterator
		*	\since version 0.0.3
		*/
		zItFace  getRawIter()
		{
			return iter;
		}



		/*! \brief This method gets the offset positions of the face.
		*
		*	\details	beased on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
		*	\param		[in]	offset				- offset distance.
		*	\param		[out]	offsetPositions		- container with the offset positions.
		*	\since version 0.0.3
		*/
		void getOffsetFacePositions(double offset, vector<zVector>& offsetPositions)
		{
			vector<zVector> out;

			vector<int> fVerts;
			getVertices(fVerts);

			for (int j = 0; j < fVerts.size(); j++)
			{
				int next = (j + 1) % fVerts.size();
				int prev = (j - 1 + fVerts.size()) % fVerts.size();


				zVector Ori = meshObj->mesh.vertexPositions[fVerts[j]];;
				zVector v1 = meshObj->mesh.vertexPositions[fVerts[prev]] - meshObj->mesh.vertexPositions[fVerts[j]];
				v1.normalize();

				zVector v2 = meshObj->mesh.vertexPositions[fVerts[next]] - meshObj->mesh.vertexPositions[fVerts[j]];
				v2.normalize();

				zVector v3 = v1;

				v1 = v1 ^ v2;
				v3 = v3 + v2;

				double cs = v3 * v2;

				zVector a1 = v2 * cs;
				zVector a2 = v3 - a1;

				double alpha = sqrt(a2.length() * a2.length());
				if (cs < 0) alpha *= -1;

				double length = offset / alpha;

				zVector mPos = meshObj->mesh.vertexPositions[fVerts[j]];
				zVector offPos = mPos + (v3 * length);

				out.push_back(offPos);

			}

			offsetPositions = out;

		}

		/*! \brief This method returns the vartiable offset positions of the face.
		*
		*	\param		[in]	offsets					- offset distance from each edge of the mesh.
		*	\param		[in]	faceCenter				- center of polygon.
		*	\param		[in]	faceNormal				- normal of polygon.
		*	\param		[out]	intersectionPositions	- container with the intersection positions.
		*	\since version 0.0.3
		*/
		void getOffsetFacePositions_Variable(vector<double>& offsets, zVector& faceCenter, zVector& faceNormal, vector<zVector>& intersectionPositions)
		{
			vector<zVector> offsetPoints;
			vector<int> fEdges;
			getHalfEdges(fEdges);

			for (int j = 0; j < fEdges.size(); j++)
			{
				zItMeshHalfEdge he(*meshObj, fEdges[j]);


				zVector p2 = meshObj->mesh.vertexPositions[he.getVertex().getId()];
				zVector p1 = meshObj->mesh.vertexPositions[he.getSym().getVertex().getId()];

				zVector norm1 = ((p1 - p2) ^ faceNormal);
				norm1.normalize();
				if ((faceCenter - p1) * norm1 < 0) norm1 *= -1;


				offsetPoints.push_back(p1 + norm1 * offsets[j]);
				offsetPoints.push_back(p2 + norm1 * offsets[j]);

			}


			for (int j = 0; j < fEdges.size(); j++)
			{
				int prevId = (j - 1 + fEdges.size()) % fEdges.size();

				zVector a0 = offsetPoints[j * 2];
				zVector a1 = offsetPoints[j * 2 + 1];

				zVector b0 = offsetPoints[prevId * 2];
				zVector b1 = offsetPoints[prevId * 2 + 1];



				double uA = -1;
				double uB = -1;
				bool intersect = meshObj->mesh.coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

				if (intersect)
				{
					//printf("\n %i working!! ", j);

					zVector closestPt;

					if (uA >= uB)
					{
						zVector dir = a1 - a0;
						double len = dir.length();
						dir.normalize();

						if (uA < 0) dir *= -1;
						closestPt = a0 + dir * len * uA;
					}
					else
					{
						zVector dir = b1 - b0;
						double len = dir.length();
						dir.normalize();

						if (uB < 0) dir *= -1;

						closestPt = b0 + dir * len * uB;
					}


					intersectionPositions.push_back(closestPt);
				}

			}
		}

		/*! \brief This method gets normal of the face.
		*
		*	\return				zVector					- face normal.
		*	\since version 0.0.3
		*/
		zVector getFaceNormal()
		{
			return meshObj->mesh.faceNormals[getId()];
		}

		/*! \brief This method gets pointer to the normal of the face.
		*
		*	\return				zVector*			- pointer to internal face normal.
		*	\since version 0.0.3
		*/
		zVector* getRawFaceNormal()
		{
			return &meshObj->mesh.faceNormals[getId()];

		}

		/*! \brief This method gets color of the face.
		*
		*	\return				zColor					- face color.
		*	\since version 0.0.3
		*/
		zColor getFaceColor()
		{
			return meshObj->mesh.faceColors[getId()];
		}

		/*! \brief This method gets pointer to the color of the face.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal face color.
		*	\since version 0.0.3
		*/
		zColor* getRawFaceColor()
		{
			return &meshObj->mesh.faceColors[getId()];
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
		*	\since version 0.0.3
		*/
		void setHalfEdge(zItMeshHalfEdge &he);

		/*! \brief This method sets color of the face.
		*
		*	\param		[in]	col						- input color.
		*	\since version 0.0.3
		*/
		void setFaceColor(zColor col)
		{
			meshObj->mesh.faceColors[getId()] = col;
		}
			   
		//--------------------------
		//---- UTILITY METHODS
		//--------------------------

		/*! \brief This method gets if the face is active.
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

		/*! \brief This operator checks for equality of two face iterators.
		*
		*	\param		[in]	other	- input iterator against which the equality is checked.
		*	\return				bool	- true if equal.
		*	\since version 0.0.3
		*/
		bool operator==(zItMeshFace &other)
		{
			return (getId() == other.getId());
		}

		/*! \brief This operator checks for non equality of two face iterators.
		*
		*	\param		[in]	other	- input iterator against which the non-equality is checked.
		*	\return				bool	- true if not equal.
		*	\since version 0.0.3
		*/
		bool operator!=(zItMeshFace &other)
		{
			return (getId() != other.getId());
		}
	};



	//--------------------------
	//--- INLINE VERTEX METHODS 
	//--------------------------
	
	//getHalfEdge
	inline zItMeshHalfEdge zSpace::zItMeshVertex::getHalfEdge()
	{
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge()->getId());
	}

	// setHalfEdge 
	inline void zSpace::zItMeshVertex::setHalfEdge(zItMeshHalfEdge &he)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int heId = he.getId();

		meshObj->mesh.vHandles[id].he = heId;
	}

	// getConnectedHalfEdges 
	inline void zSpace::zItMeshVertex::getConnectedHalfEdges(vector<zItMeshHalfEdge>& halfedges)
	{
		if (!this->iter->getHalfEdge()) return;

		if (!getHalfEdge().isActive()) return;

		zItMeshHalfEdge start = getHalfEdge();
		zItMeshHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedges.push_back(e);
			e = e.getPrev().getSym();

		} while (e != start);
	}

	// getConnectedHalfEdges 
	inline void zSpace::zItMeshVertex::getConnectedHalfEdges(vector<int>& halfedgeIndicies)
	{
		if (!this->iter->getHalfEdge()) return;

		if (!getHalfEdge().isActive()) return;


		zItMeshHalfEdge start = getHalfEdge();
		zItMeshHalfEdge e = getHalfEdge();

		bool exit = false;

		do
		{
			halfedgeIndicies.push_back(e.getId());
			e = e.getPrev().getSym();

		} while (e != start);
	}


	// getConnectedEdges 
	inline void zSpace::zItMeshVertex::getConnectedEdges(vector<zItMeshEdge>& edges)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edges.push_back(he.getEdge());
		}
	}

	// getConnectedEdges 
	inline void zSpace::zItMeshVertex::getConnectedEdges(vector<int>& edgeIndicies)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			edgeIndicies.push_back(he.getEdge().getId());
		}
	}

	//getConnectedVertices
	inline void zSpace::zItMeshVertex::getConnectedVertices(vector<zItMeshVertex>& verticies)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			verticies.push_back(he.getVertex());
		}
	}

	//getConnectedVertices
	inline void zSpace::zItMeshVertex::getConnectedVertices(vector<int>& vertexIndicies)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			vertexIndicies.push_back(he.getVertex().getId());
		}
	}

	//getConnectedFaces
	inline void zSpace::zItMeshVertex::getConnectedFaces(vector<zItMeshFace>& faces)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			if (!he.onBoundary()) faces.push_back(he.getFace());
		}
	}

	//getConnectedFaces
	inline void zSpace::zItMeshVertex::getConnectedFaces(vector<int>& faceIndicies)
	{
		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);


		for (auto &he : cHEdges)
		{
			if (!he.onBoundary()) faceIndicies.push_back(he.getFace().getId());
		}
	}

	//onBoundary
	inline bool zSpace::zItMeshVertex::onBoundary()
	{
		bool out = false;

		vector<zItMeshHalfEdge> cHEdges;
		getConnectedHalfEdges(cHEdges);

		for (auto &he : cHEdges)
		{
			if (he.onBoundary())
			{
				out = true;
				break;
			}

		}

		return out;
	}

	//getVertexValence
	inline int zSpace::zItMeshVertex::getVertexValence()
	{
		int out;

		vector<int> cHEdges;
		getConnectedHalfEdges(cHEdges);

		out = cHEdges.size();

		return out;
	}

	//checkVertexValency
	inline bool zSpace::zItMeshVertex::checkVertexValency(int valence)
	{
		bool out = false;
		out = (getVertexValence() == valence) ? true : false;

		return out;
	}

	//--------------------------
	//--- INLINE HALF EDGE METHODS 
	//--------------------------

	//getVertex
	inline zItMeshVertex zSpace::zItMeshHalfEdge::getVertex()
	{
		return zItMeshVertex(*meshObj, iter->getVertex()->getId());
	}

	//getFace
	inline zItMeshFace zSpace::zItMeshHalfEdge::getFace()
	{
		return zItMeshFace(*meshObj, iter->getFace()->getId());
	}

	//getEdge
	inline zItMeshEdge zSpace::zItMeshHalfEdge::getEdge()
	{
		return zItMeshEdge(*meshObj, iter->getEdge()->getId());
	}

	//setSym
	inline void zItMeshHalfEdge::setSym(zItMeshHalfEdge & he)
	{
		iter->setSym(&meshObj->mesh.halfEdges[he.getId()]);				
	}

	//setNext
	inline void zItMeshHalfEdge::setNext(zItMeshHalfEdge & he)
	{
		iter->setNext(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int nextId = he.getId();

		meshObj->mesh.heHandles[id].n = nextId;
		meshObj->mesh.heHandles[nextId].p = id;
	}

	//setPrev
	inline void zItMeshHalfEdge::setPrev(zItMeshHalfEdge & he)
	{
		iter->setPrev(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int prevId = he.getId();

		meshObj->mesh.heHandles[id].p = prevId;
		meshObj->mesh.heHandles[prevId].n = id;
	}

	//setVertex
	inline void zItMeshHalfEdge::setVertex(zItMeshVertex & v)
	{
		iter->setVertex(&meshObj->mesh.vertices[v.getId()]);

		int id = getId();
		int vId = v.getId();

		meshObj->mesh.heHandles[id].v = vId;
		
	}

	//setEdge
	inline void zItMeshHalfEdge::setEdge(zItMeshEdge & e)
	{
		iter->setEdge(&meshObj->mesh.edges[e.getId()]);

		int id = getId();
		int eId = e.getId();

		meshObj->mesh.heHandles[id].e = eId;
	
	}

	//setFace
	inline void zItMeshHalfEdge::setFace(zItMeshFace & f)
	{
		iter->setFace(&meshObj->mesh.faces[f.getId()]);

		int id = getId();
		int fId = f.getId();

		meshObj->mesh.heHandles[id].f = fId;
	}

	//getFaces
	inline void zItMeshHalfEdge::getFaces(vector<zItMeshFace> &faceIndicies)
	{
		this->getEdge().getFaces(faceIndicies);
	}

	//getFaces
	inline void zItMeshHalfEdge::getFaces(vector<int> &faceIndicies)
	{
		this->getEdge().getFaces(faceIndicies);
	}

	//--------------------------
	//--- INLINE EDGE METHODS 
	//--------------------------
	
	//getHalfEdge
	zItMeshHalfEdge zSpace::zItMeshEdge::getHalfEdge(int _index)
	{
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge(_index)->getId());
	}

	//setHalfEdge
	inline void zItMeshEdge::setHalfEdge(zItMeshHalfEdge & he, int _index)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()], _index);

		int id = getId();
		int heId = he.getId();

		if (_index == 0) meshObj->mesh.eHandles[id].he0 = heId;
		if (_index == 1) meshObj->mesh.eHandles[id].he1 = heId;
	}

	//getFaces
	inline void zItMeshEdge::getFaces(vector<zItMeshFace> &faces)
	{
		if (!getHalfEdge(0).onBoundary()) faces.push_back(getHalfEdge(0).getFace());
		if (!getHalfEdge(1).onBoundary()) faces.push_back(getHalfEdge(1).getFace());
	}

	//getFaces
	inline void zItMeshEdge::getFaces(vector<int> &faceIndicies)
	{
		if (!getHalfEdge(0).onBoundary()) faceIndicies.push_back(getHalfEdge(0).getFace().getId());
		if (!getHalfEdge(1).onBoundary()) faceIndicies.push_back(getHalfEdge(1).getFace().getId());
	}
	
	//--------------------------
	//--- INLINE FACE METHODS 
	//--------------------------

	inline zItMeshHalfEdge zSpace::zItMeshFace::getHalfEdge()
	{
		return zItMeshHalfEdge(*meshObj, iter->getHalfEdge()->getId());
	}

	inline void zItMeshFace::setHalfEdge(zItMeshHalfEdge & he)
	{
		iter->setHalfEdge(&meshObj->mesh.halfEdges[he.getId()]);

		int id = getId();
		int heId = he.getId();

		meshObj->mesh.fHandles[id].he = heId;
	}
}