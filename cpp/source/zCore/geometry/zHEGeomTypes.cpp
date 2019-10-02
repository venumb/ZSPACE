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


#include<headers/zCore/geometry/zHEGeomTypes.h>

//---- ZEDGE ------------------------------------------------------------------------------

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zEdge::zEdge()
	{
		index = -1;
		he[0] = nullptr;
		he[1] = nullptr;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zEdge::~zEdge() {}

	//---- GET-SET METHODS

	ZSPACE_INLINE int zEdge::getId()
	{
		return this->index;
	}

	ZSPACE_INLINE void zEdge::setId(int _edgeId)
	{
		this->index = _edgeId;
	}

	ZSPACE_INLINE zHalfEdge* zEdge::getHalfEdge(int _index)
	{
		return this->he[_index];
	}

	ZSPACE_INLINE void zEdge::setHalfEdge(zHalfEdge* _he, int _index)
	{
		this->he[_index] = _he;
	}

	//---- METHODS

	ZSPACE_INLINE bool zEdge::isActive() const
	{
		return index >= 0;
	}

	ZSPACE_INLINE void zEdge::reset()
	{
		he[0] = nullptr;
		he[1] = nullptr;
		index = -1;
	}
}

//---- ZHALFEDGE ------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zHalfEdge::zHalfEdge()
	{
		index = -1;
		v = nullptr;
		f = nullptr;
		e = nullptr;

		prev = nullptr;
		next = nullptr;
		sym = nullptr;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zHalfEdge::~zHalfEdge(){}

	//---- GET-SET METHODS

	ZSPACE_INLINE int zHalfEdge::getId()
	{
		return this->index;
	}

	ZSPACE_INLINE void zHalfEdge::setId(int _edgeId)
	{
		this->index = _edgeId;
	}

	ZSPACE_INLINE zHalfEdge* zHalfEdge::getSym()
	{
		return this->sym;
	}

	ZSPACE_INLINE void zHalfEdge::setSym(zHalfEdge* _sym)
	{
		this->sym = _sym;
		_sym->sym = this;
	}

	ZSPACE_INLINE zHalfEdge* zHalfEdge::getPrev()
	{
		return this->prev;
	}

	ZSPACE_INLINE void zHalfEdge::setPrev(zHalfEdge* _prev)
	{
		this->prev = _prev;
		if (this->getPrev()) _prev->next = this;
	}

	ZSPACE_INLINE zHalfEdge* zHalfEdge::getNext()
	{
		return this->next;
	}

	ZSPACE_INLINE void zHalfEdge::setNext(zHalfEdge* _next)
	{
		this->next = _next;
		if (this->getNext()) _next->prev = this;
	}

	ZSPACE_INLINE zVertex* zHalfEdge::getVertex()
	{
		return this->v;
	}

	ZSPACE_INLINE void zHalfEdge::setVertex(zVertex* _v)
	{
		this->v = _v;
	}

	ZSPACE_INLINE zFace* zHalfEdge::getFace()
	{
		return this->f;
	}

	ZSPACE_INLINE void zHalfEdge::setFace(zFace* _f)
	{
		this->f = _f;
	}

	ZSPACE_INLINE zEdge* zHalfEdge::getEdge()
	{
		return this->e;
	}

	ZSPACE_INLINE void zHalfEdge::setEdge(zEdge* _e)
	{
		this->e = _e;
	}

	//---- METHODS

	ZSPACE_INLINE void zHalfEdge::reset()
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

	ZSPACE_INLINE bool zHalfEdge::isActive() const
	{
		return index >= 0;
	}

}

//---- ZVERTEX ---------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zVertex::zVertex()
	{
		index = -1;
		he = nullptr;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zVertex::~zVertex() {}

	//---- GET-SET METHODS

	ZSPACE_INLINE int zVertex::getId()
	{
		return this->index;
	}

	ZSPACE_INLINE void zVertex::setId(int _vertexId)
	{
		this->index = _vertexId;
	}

	ZSPACE_INLINE zHalfEdge* zVertex::getHalfEdge()
	{
		return this->he;
	}

	ZSPACE_INLINE void zVertex::setHalfEdge(zHalfEdge* _he)
	{
		this->he = _he;
	}

	//---- METHODS

	ZSPACE_INLINE void zVertex::reset()
	{
		this->he = nullptr;
		index = -1;
	}

	ZSPACE_INLINE bool zVertex::isActive() const
	{
		return index >= 0;
	}

}

//---- ZFACE -----------------------------------------------------------------------------------

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFace::zFace()
	{
		index = -1;
		he = nullptr;
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFace::~zFace(){}

	//---- GET-SET METHODS

	ZSPACE_INLINE int zFace::getId()
	{
		return this->index;
	}

	ZSPACE_INLINE void zFace::setId(int _faceId)
	{
		this->index = _faceId;
	}

	ZSPACE_INLINE zHalfEdge* zFace::getHalfEdge()
	{
		return this->he;
	}

	ZSPACE_INLINE void zFace::setHalfEdge(zHalfEdge* _he)
	{
		this->he = _he;
	}

	//---- METHODS

	ZSPACE_INLINE void zFace::reset()
	{
		this->he = nullptr;
		index = -1;
	}

	ZSPACE_INLINE bool zFace::isActive() const
	{
		return index >= 0;
	}
}