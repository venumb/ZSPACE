#pragma once


namespace zSpace
{
	
	
	
	class  zEdge;
	class  zHalfEdge;
	class  zVertex;
	class  zFace;

	typedef std::vector<zVertex>::iterator			zItVertex;
	typedef std::vector<zHalfEdge>::iterator		zItHalfEdge;
	typedef std::vector<zEdge>::iterator			zItEdge;
	typedef std::vector<zFace>::iterator			zItFace;

	typedef std::vector<zVertex>::const_iterator		zCItVertex;
	typedef std::vector<zHalfEdge>::const_iterator		zCItHalfEdge;
	typedef std::vector<zEdge>::const_iterator			zCItEdge;
	typedef std::vector<zFace>::const_iterator			zCItFace;

	
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
	*  @{
	*/

	/*! \class zEdge
	*	\brief An edge struct to  hold edge information of a half-edge data structure.
	*	\since version 0.0.3
	*/

	/** @}*/

	/** @}*/

	class  zEdge
	{
	protected:
		int index;

		/*!	\brief iterator	to 2 half edges of the edge. */
		zHalfEdge* he[2];

	public:		
	
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		zEdge()
		{
			index = -1;
			he[0] = nullptr;
			he[1] = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/

		~zEdge()
		{
			/*he[0] = nullptr;
			he[1] = nullptr;

			delete he[0];
			delete he[1];*/

			
		}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the edgeId of current zEdge.
		*	\return				edgeId.
		*	\since version 0.0.1
		*/

		int getId()
		{
			return this->index;
		}

		/*! \brief This method sets the edgeId of current zEdge to the the input value.
		*	\param		[in]	edgeId.
		*	\since version 0.0.1
		*/

		void setId(int _edgeId)
		{
			this->index = _edgeId;
		}

		zHalfEdge* getHalfEdge(int _index)
		{
			return this->he[_index];
		}

		void setHalfEdge(zHalfEdge* _he , int _index)
		{
			this->he[_index] = _he;
		}

		/*! \brief This method checks if the current element is active.
		*	\since version 0.0.1
		*/
		bool isActive() const
		{
			return index >= 0;
		}
		
		/*! \brief This method makes the pointers of the current zEdge to null.
		*	\since version 0.0.1
		*/
		void reset()
		{
			he[0] = nullptr;
			he[1] = nullptr;
			index = -1;
		}

	};

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
	*  @{
	*/

	/*! \class zHalfEdge
	*	\brief An half edge struct to  hold half edge information of a half-edge data structure.
	*	\since version 0.0.2
	*/

	/** @}*/ 

	/** @}*/

	class  zHalfEdge
	{
	protected:
		/*!	\brief iterator	in vertex list. */
		zVertex* v;

		/*!	\brief iterator in face list. */
		zFace* f;

		/*!	\brief iterator in edge list. */
		zEdge* e;

		/*!	\brief iterator to previous  halfedge */
		zHalfEdge* prev;

		/*!	\brief iterator to next halfedge */
		zHalfEdge* next;

		/*!	\brief iterator to symmerty/twin half edge.			*/
		zHalfEdge* sym;

		/*!	\brief index of half edge.			*/
		int index;


	public:	

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		zHalfEdge()
		{
			index = -1;
			v = nullptr;
			f = nullptr;
			e = nullptr;

			prev = nullptr;
			next = nullptr;
			sym = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zHalfEdge()
		{
			/*v = nullptr;
			f = nullptr;
			e = nullptr;

			prev = nullptr;
			next = nullptr;
			sym = nullptr;

			delete v;
			delete f;
			delete e;

			delete prev;
			delete next;
			delete sym;*/
		}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the edgeId of current zEdge.
		*	\return				edgeId.
		*	\since version 0.0.1
		*/

		int getId()
		{
			return this->index;
		}

		/*! \brief This method sets the edgeId of current zEdge to the the input value.
		*	\param		[in]	edgeId.
		*	\since version 0.0.1
		*/

		void setId(int _edgeId)
		{
			this->index = _edgeId;
		}

		/*! \brief This method returns the symmetry edge of current zEdge.
		*	\return				symmetry edge of type zEdge.
		*	\since version 0.0.1
		*/

		zHalfEdge* getSym()
		{
			return this->sym;
		}

		/*! \brief This method sets the symmetry edge of current zEdge to the the input edge
		*	\param		[in]	symmetry edge of type zEdge.
		*	\since version 0.0.1
		*/

		void setSym(zHalfEdge* _sym)
		{
			this->sym = _sym;
			_sym->sym = this;
		}

		/*! \brief This method returns the previous edge of current zEdge.
		*	\return				previous edge of type zEdge.
		*	\since version 0.0.1
		*/

		zHalfEdge* getPrev()
		{
			return this->prev;
		}

		/*! \brief This method sets the previous edge of current zEdge to the the input edge
		*	\param		[in]	_prev		- previous edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setPrev(zHalfEdge* _prev)
		{
			this->prev = _prev;
			if (this->getPrev()) _prev->next = this;
		}

		/*! \brief This method returns the next edge of current zEdge.
		*	\return				next edge of type zEdge.
		*	\since version 0.0.1
		*/
		zHalfEdge* getNext()
		{
			return this->next;
		}

		/*! \brief This method sets the next edge of current zEdge to the the input edge
		*	\param		[in]	_next		- next edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setNext(zHalfEdge* _next)
		{
			this->next = _next;
			if (this->getNext()) _next->prev = this;
		}

		/*! \brief This method returns the vertex pointed to by the current zEdge.
		*	\return				vertex of type zVertex.
		*	\since version 0.0.1
		*/
		zVertex* getVertex()
		{
			return this->v;
		}

		/*! \brief This method sets the vertex pointed to by the current zEdge to the the input zVertex.
		*	\param		[in]	vertex of type zVertex.
		*	\since version 0.0.1
		*/
		void setVertex(zVertex* _v)
		{
			this->v = _v;
		}

		/*! \brief This method returns the face pointed to by the current zEdge.
		*	\return				face of type zface.
		*	\since version 0.0.1
		*/
		zFace* getFace()
		{
			return this->f;
		}

		/*! \brief This method sets the face pointed to by the current zEdge to the the input zFace.
		*	\param		[in]	face of type zface.
		*	\since version 0.0.1
		*/
		void setFace(zFace* _f)
		{
			this->f = _f;
		}

		/*! \brief This method returns the edge pointed to by the current half edge.
		*	\return				face of type zface.
		*	\since version 0.0.1
		*/
		zEdge* getEdge()
		{
			return this->e;
		}

		/*! \brief This method sets the edge pointed to by the current halfedge to the the input edge.
		*	\param		[in]	face of type zface.
		*	\since version 0.0.1
		*/
		void setEdge(zEdge* _e)
		{
			this->e = _e;
		}

		/*! \brief This method makes the pointer of the current and symmetry zEdge to null.
		*	\since version 0.0.1
		*/
		void reset()
		{
			if (this->getNext()) this->getNext()->setPrev(this->getSym()->getPrev());
			if (this->getPrev()) this->getPrev()->setNext(this->getSym()->getNext());


			this->next = nullptr;
			this->prev = nullptr;
			this->v = nullptr;
			this->f = nullptr;

			if (this->getSym())
			{
				this->getSym()->next = nullptr;
				this->getSym()->prev = nullptr;
				this->getSym()->v = nullptr;
				this->getSym()->f = nullptr;
			}

			index = -1;

		}

		/*! \brief This method checks if the current element is active.
		*	\since version 0.0.1
		*/
		bool isActive() const
		{
			return index >= 0;
		}

	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
	*  @{
	*/

	/*! \class zVertex
	*	\brief A vertex struct to  hold vertex information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class  zVertex
	{
	protected:
		int index;

		/*!	\brief pointer to zHalfEdge starting at the current zVertex.		*/
		zHalfEdge* he;

	public:		
			

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		zVertex()
		{
			index = -1;
			he = nullptr;			
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zVertex()
		{
			/*he = nullptr;
			delete he;*/
		}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the vertexId of current zVertex.
		*	\return				vertexId.
		*	\since version 0.0.1
		*/

		int getId()
		{
			return this->index;
		}

		/*! \brief This method sets the vertexId of current zVertex to the the input value.
		*	\param		[in]	vertexId.
		*	\since version 0.0.1
		*/

		void setId(int _vertexId)
		{
			this->index = _vertexId;
		}

		/*! \brief This method returns the associated edge of current zVertex.
		*	\return				associated edge of type zEdge.
		*	\since version 0.0.1
		*/

		zHalfEdge* getHalfEdge()
		{
			return this->he;
		}

		/*! \brief This method sets the associated edge of current zVertex to the the input edge
		*	\param		[in]	edge of type zEdge.
		*	\since version 0.0.1
		*/

		void setHalfEdge(zHalfEdge* _he)
		{
			this->he = _he;
		}

		/*! \brief This method makes the pointers of the current zVertex to null.
		*	\since version 0.0.1
		*/
		void reset()
		{
			this->he = nullptr;
			index = -1;
		}

		/*! \brief This method checks if the current element is active.
		*	\since version 0.0.1
		*/
		bool isActive() const
		{
			return index >= 0;
		}
	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
	*  @{
	*/

	/*! \class zFace
	*	\brief A face struct to  hold polygonal information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class  zFace 
	{
	protected:
		
		/*!	\brief stores unique ID per face. 	*/
		int index;

		/*!	\brief pointer to one of the zHalfEdge contained in the polygon.		*/
		zHalfEdge* he;

	public:	
		
		
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/
		zFace()
		{
			index = -1;
			he = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/
		~zFace()
		{
			/*he = nullptr;
			delete he;*/
		}

		//---- GET-SET METHODS

		/*! \brief This method returns the faceId of current zFace.
		*	\return				faceId.
		*	\since version 0.0.1
		*/

		int getId()
		{
			return this->index;
		}


		/*! \brief This method sets the faceId of current zFace to the the input value.
		*	\param		[in]	faceId.
		*	\since version 0.0.1
		*/

		void setId(int _faceId)
		{
			this->index = _faceId;
		}

		/*! \brief This method returns the associated edge of current zFace.
		*	\return				associated edge of type zEdge.
		*	\since version 0.0.1
		*/
		zHalfEdge* getHalfEdge()
		{
			return this->he;
		}

		/*! \brief This method sets the associated edge of current zFace to the the input edge
		*	\param		[in]	associated edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setHalfEdge(zHalfEdge* _he)
		{
			this->he = _he;
		}

		/*! \brief This method makes the pointers of the current zFace to null.
		*	\since version 0.0.1
		*/
		void reset()
		{
			this->he = nullptr;
			index = -1;
		}

		/*! \brief This method checks if the current element is active.
		*	\since version 0.0.1
		*/
		bool isActive() const 
		{ 
			return index >= 0; 
		}

	};

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometryHandles
	*	\brief The geometry handle classes of the library.
	*  @{
	*/

	/*! \struct zCurvature
	*	\brief A curvature struct defined to  hold defined to hold curvature information of a half-edge geometry. 
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/
	struct zCurvature
	{
		double k1;		/*!< stores principal curvature 1		*/
		double k2;		/*!< stores principal curvature 2		*/

	};


}
