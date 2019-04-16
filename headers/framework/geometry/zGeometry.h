#pragma once

#include <headers/framework/core/zVector.h>

namespace zSpace
{
	// Class Vertex , Edge and Face
	class  zEdge;
	class  zVertex;
	class  zFace;

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zEdge
	*	\brief An edge class to  hold edge information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class  zEdge
	{

	private:
		/*!	\brief stores edgeId. It is unique per edge. 	*/
		int edgeId;		

		/*!	\brief pointer to zVertex	in vertex list.		*/
		zVertex *v;	

		/*!	\brief pointer to zFace in face list.		*/
		zFace *f;		

		/*!	\brief pointer to previous zEdge in a polygon.			*/
		zEdge *prev;	

		/*!	\brief pointer to next zEdge in a polygon.			*/
		zEdge *next;

		/*!	\brief pointer to symmerty/twin zEdge  in edge list.			*/
		zEdge *sym;		

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/

		zEdge()
		{
			this->edgeId = -2;
			this->sym = nullptr;
			this->next = nullptr;
			this->prev = nullptr;
			this->v = nullptr;
			this->f = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/

		~zEdge() 
		{
			
		}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the edgeId of current zEdge.
		*	\return				edgeId.
		*	\since version 0.0.1
		*/

		int getEdgeId()
		{
			return this->edgeId;
		}

		/*! \brief This method sets the edgeId of current zEdge to the the input value.
		*	\param		[in]	edgeId.
		*	\since version 0.0.1
		*/

		void setEdgeId(int _edgeId)
		{
			this->edgeId = _edgeId;
		}

		/*! \brief This method returns the symmetry edge of current zEdge.
		*	\return				symmetry edge of type zEdge.
		*	\since version 0.0.1
		*/

		zEdge* getSym()
		{
			return this->sym;
		}

		/*! \brief This method sets the symmetry edge of current zEdge to the the input edge
		*	\param		[in]	symmetry edge of type zEdge.
		*	\since version 0.0.1
		*/

		void setSym(zEdge* _sym)
		{
			this->sym = _sym;
			_sym->sym = this;
		}

		/*! \brief This method returns the previous edge of current zEdge.
		*	\return				previous edge of type zEdge.
		*	\since version 0.0.1
		*/

		zEdge* getPrev()
		{
			return this->prev;
		}

		/*! \brief This method sets the previous edge of current zEdge to the the input edge
		*	\param		[in]	_prev		- previous edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setPrev(zEdge* _prev)
		{
			this->prev = _prev;
			if(this->getPrev()) _prev->next = this;
		}

		/*! \brief This method returns the next edge of current zEdge.
		*	\return				next edge of type zEdge.
		*	\since version 0.0.1
		*/
		zEdge* getNext()
		{
			return this->next;
		}

		/*! \brief This method sets the next edge of current zEdge to the the input edge
		*	\param		[in]	_next		- next edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setNext(zEdge* _next)
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

		/*! \brief This method makes the pointer of the current and symmetry zEdge to null.
		*	\since version 0.0.1
		*/
		void removeEdge()
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

		}


	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zVertex
	*	\brief A vertex class to  hold vertex information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class  zVertex
	{


	private:
		
		/*!	\brief stores vertexId. It is unique per vertex.		*/
		int vertexId;	

		/*!	\brief pointer to zEdge starting at the current zVertex.		*/
		zEdge *e;		

	public:
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/

		zVertex()
		{
			this->vertexId = -2;
			this->e = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/

		~zVertex() 
		{
			
		}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method returns the vertexId of current zVertex.
		*	\return				vertexId.
		*	\since version 0.0.1
		*/

		int getVertexId()
		{
			return this->vertexId;
		}

		/*! \brief This method sets the vertexId of current zVertex to the the input value.
		*	\param		[in]	vertexId.
		*	\since version 0.0.1
		*/

		void setVertexId(int _vertexId)
		{
			this->vertexId = _vertexId;
		}

		/*! \brief This method returns the associated edge of current zVertex.
		*	\return				associated edge of type zEdge.
		*	\since version 0.0.1
		*/

		zEdge* getEdge()
		{
			return this->e;
		}

		/*! \brief This method sets the associated edge of current zVertex to the the input edge
		*	\param		[in]	edge of type zEdge.
		*	\since version 0.0.1
		*/

		void setEdge(zEdge* _e)
		{
			this->e = _e;	
		}

		/*! \brief This method makes the pointers of the current zVertex to null.
		*	\since version 0.0.1
		*/
		void removeVertex()
		{
			this->e = nullptr;
		}

	};


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zFace
	*	\brief A face class to  hold polygonal information of a half-edge data structure.
	*	\since version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class  zFace
	{

	private:
		/*!	\brief stores faceId. It is unique per polygon.		*/
		int faceId;	

		/*!	\brief pointer to one of the zEdge contained in the polygon.		*/
		zEdge *e;		

	public:
		/*! \brief Default constructor.
		*	\since version 0.0.1
		*/

		zFace()
		{
			this->faceId = -2;
			this->e = nullptr;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/

		~zFace() 
		{
			
		}

		//---- GET-SET METHODS

		/*! \brief This method returns the faceId of current zFace.
		*	\return				faceId.
		*	\since version 0.0.1
		*/

		int getFaceId()
		{
			return this->faceId;
		}


		/*! \brief This method sets the faceId of current zFace to the the input value.
		*	\param		[in]	faceId.
		*	\since version 0.0.1
		*/

		void setFaceId(int _faceId)
		{
			this->faceId = _faceId;
		}

		/*! \brief This method returns the associated edge of current zFace.
		*	\return				associated edge of type zEdge.
		*	\since version 0.0.1
		*/
		zEdge* getEdge()
		{
			return this->e;
		}

		/*! \brief This method sets the associated edge of current zFace to the the input edge
		*	\param		[in]	associated edge of type zEdge.
		*	\since version 0.0.1
		*/
		void setEdge(zEdge* _e)
		{
			this->e = _e;			
		}

		/*! \brief This method makes the pointers of the current zFace to null.
		*	\since version 0.0.1
		*/
		void removeFace()
		{
			this->e = nullptr;
		}

	};

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
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
