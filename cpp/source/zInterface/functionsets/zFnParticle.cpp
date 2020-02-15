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


#include<headers/zInterface/functionsets/zFnParticle.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zFnParticle::zFnParticle()
	{
		fnType = zFnType::zMeshFn;

		particleObj = nullptr;

	}

	ZSPACE_INLINE zFnParticle::zFnParticle(zObjParticle &_particleObj)
	{
		particleObj = &_particleObj;
		fnType = zFnType::zParticleFn;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zFnParticle::~zFnParticle() {}

	//---- OVERRIDE METHODS

	ZSPACE_INLINE zFnType zFnParticle::getType()
	{
		return zFnType::zParticleFn;
	}

	ZSPACE_INLINE void zFnParticle::from(string path, zFileTpye type, bool staticGeom)
	{
		
	}

	ZSPACE_INLINE void zFnParticle::to(string path, zFileTpye type)
	{
		
	}

	ZSPACE_INLINE void zFnParticle::getBounds(zPoint &minBB, zPoint &maxBB)
	{
		
	}

	ZSPACE_INLINE void zFnParticle::clear()
	{

		
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zFnParticle::create(zPoint &_p, bool _fixed , double _m , zVector _v , zVector _f )
	{

		if (particleObj == NULL)
		{
			zObjParticle pObj;
			pObj.particle = zParticle(_p, _fixed, _m, _v, _f);

			particleObj = &pObj;
		}
		else particleObj->particle = zParticle(_p, _fixed, _m, _v, _f);
	}

	//---- SET METHODS

	ZSPACE_INLINE void zFnParticle::setMass(double mass)
	{
		particleObj->particle.m = mass;
	}

	ZSPACE_INLINE void zFnParticle::setPosition(zVector *pos)
	{
		particleObj->particle.s.p = pos;
	}

	ZSPACE_INLINE void zFnParticle::setVelocity(zVector &vel)
	{
		particleObj->particle.s.v = vel;
	}

	ZSPACE_INLINE void zFnParticle::setState(zParticleState &state)
	{
		particleObj->particle.s = state;
	}

	ZSPACE_INLINE void zFnParticle::setForce(zVector &force)
	{
		particleObj->particle.f = force;
	}

	ZSPACE_INLINE void zFnParticle::setFixed(bool _fixed)
	{
		particleObj->particle.fixed = _fixed;
	}

	//---- GET METHODS

	ZSPACE_INLINE double zFnParticle::getMass()
	{
		return particleObj->particle.m;
	}

	ZSPACE_INLINE zPoint zFnParticle::getPosition()
	{
		return *particleObj->particle.s.p;
	}

	ZSPACE_INLINE zPoint* zFnParticle::getRawPosition()
	{
		return particleObj->particle.s.p;
	}

	ZSPACE_INLINE zVector zFnParticle::getVelocity()
	{
		return particleObj->particle.s.v;
	}

	ZSPACE_INLINE zParticleState zFnParticle::getState()
	{
		return particleObj->particle.s;
	}

	ZSPACE_INLINE zVector zFnParticle::getForce()
	{
		return particleObj->particle.f;
	}

	ZSPACE_INLINE bool zFnParticle::getFixed()
	{
		return particleObj->particle.fixed;
	}

	//---- FORCE METHODS

	ZSPACE_INLINE void zFnParticle::addForce(zVector &force)
	{
		particleObj->particle.f += force;
	}

	ZSPACE_INLINE void zFnParticle::clearForce()
	{
		particleObj->particle.f = zVector();
	}

	ZSPACE_INLINE void zFnParticle::addForces(vector<zVector> &forces)
	{
		for (int i = 0; i < forces.size(); i++) particleObj->particle.f += forces[i];
	}

	//---- INTEGRATION METHODS

	ZSPACE_INLINE zParticleDerivative zFnParticle::getDerivatives(zParticleDerivative &deriv, double dT)
	{
		zParticleDerivative out;

		//acceleration
		out.dV = (particleObj->particle.f / particleObj->particle.m) * dT;

		//veclocity
		out.dP = deriv.dV + (out.dV * dT);

		return out;
	}

	ZSPACE_INLINE void zFnParticle::integrateForces(double dT, zIntergrationType type)
	{

		if (getFixed())
		{
			return;
		}

		// Semi-Implicit Euler Integrate
		if (type == zEuler)
		{
			particleObj->particle.derivative = getDerivatives(particleObj->particle.derivative, dT);
		}

		// Renge Kutta 4th Derivative Integrate
		else if (type == zRK4)
		{

			zParticleDerivative a, b, c, d, temp;

			a = getDerivatives(particleObj->particle.derivative, dT) * 0.5;

			temp = particleObj->particle.derivative + a;
			b = getDerivatives(temp, dT) * 0.5;

			temp = particleObj->particle.derivative + b;
			c = getDerivatives(temp, dT);

			temp = particleObj->particle.derivative + c;
			d = getDerivatives(temp, dT);

			particleObj->particle.derivative = a * (0.167) + b * (0.334) + c * (0.334) + d * (0.167);

		}

		// pixel Integrate
		else if (type == zPixel)
		{

		}

		else throw std::invalid_argument(" error: invalid zIntergrationType ");

	}

	ZSPACE_INLINE zVector zFnParticle::getUpdatePosition(double dT, zIntergrationType type)
	{
		integrateForces(dT, type);
		return (*particleObj->particle.s.p + particleObj->particle.derivative.dP);
	}

	ZSPACE_INLINE void zFnParticle::updateParticle(bool clearForce, bool clearVelocity, bool clearDerivatives)
	{
		if (getFixed()) return;

		*particleObj->particle.s.p += particleObj->particle.derivative.dP;
		particleObj->particle.s.v += particleObj->particle.derivative.dV;

		if (clearForce) particleObj->particle.f = zVector();
		if (clearVelocity) particleObj->particle.s.v = zVector();
		if (clearDerivatives) particleObj->particle.derivative = zParticleDerivative();
	}

	//---- TRANSFORM METHODS OVERRIDES

	ZSPACE_INLINE void zFnParticle::setTransform(zTransform &inTransform, bool decompose, bool updatePositions)
	{
		if (updatePositions)
		{
			zTransformationMatrix to;
			to.setTransform(inTransform, decompose);

			zTransform transMat = particleObj->transformationMatrix.getToMatrix(to);
			transformObject(transMat);

			particleObj->transformationMatrix.setTransform(inTransform);

			// update pivot values of object transformation matrix
			zVector p = particleObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}
		else
		{
			particleObj->transformationMatrix.setTransform(inTransform, decompose);

			zVector p = particleObj->transformationMatrix.getO();
			setPivot(p);

		}

	}

	ZSPACE_INLINE void zFnParticle::setScale(zFloat4 &scale)
	{
		// get  inverse pivot translations
		zTransform invScalemat = particleObj->transformationMatrix.asInverseScaleTransformMatrix();

		// set scale values of object transformation matrix
		particleObj->transformationMatrix.setScale(scale);

		// get new scale transformation matrix
		zTransform scaleMat = particleObj->transformationMatrix.asScaleTransformMatrix();

		// compute total transformation
		zTransform transMat = invScalemat * scaleMat;

		// transform object
		transformObject(transMat);
	}

	ZSPACE_INLINE void zFnParticle::setRotation(zFloat4 &rotation, bool appendRotations)
	{
		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = particleObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = particleObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = particleObj->transformationMatrix;
		to.setRotation(rotation, appendRotations);
		zTransform toMat = particleObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set rotation values of object transformation matrix
		particleObj->transformationMatrix.setRotation(rotation, appendRotations);;
	}

	ZSPACE_INLINE void zFnParticle::setTranslation(zVector &translation, bool appendTranslations)
	{
		// get vector as zDouble3
		zFloat4 t;
		translation.getComponents(t);

		// get pivot translation and inverse pivot translations
		zTransform pivotTransMat = particleObj->transformationMatrix.asPivotTranslationMatrix();
		zTransform invPivotTransMat = particleObj->transformationMatrix.asInversePivotTranslationMatrix();

		// get plane to plane transformation
		zTransformationMatrix to = particleObj->transformationMatrix;
		to.setTranslation(t, appendTranslations);
		zTransform toMat = particleObj->transformationMatrix.getToMatrix(to);

		// compute total transformation
		zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

		// transform object
		transformObject(transMat);

		// set translation values of object transformation matrix
		particleObj->transformationMatrix.setTranslation(t, appendTranslations);;

		// update pivot values of object transformation matrix
		zVector p = particleObj->transformationMatrix.getPivot();
		p = p * transMat;
		setPivot(p);

	}

	ZSPACE_INLINE void zFnParticle::setPivot(zVector &pivot)
	{
		// get vector as zDouble3
		zFloat4 p;
		pivot.getComponents(p);

		// set pivot values of object transformation matrix
		particleObj->transformationMatrix.setPivot(p);
	}

	ZSPACE_INLINE void zFnParticle::getTransform(zTransform &transform)
	{
		transform = particleObj->transformationMatrix.asMatrix();
	}

	//---- PROTECTED TRANSFORM  METHODS

	ZSPACE_INLINE void zFnParticle::transformObject(zTransform &transform)
	{	

		zVector* pos = getRawPosition();
		
		zVector newPos = pos[0] * transform;
		pos[0] = newPos;	

	}

}