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


#include<headers/zInterOp/interOp/zIONurbsCurve.h>

#if defined(ZSPACE_MAYA_INTEROP)  && defined(ZSPACE_RHINO_INTEROP)

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zIONurbsCurve::zIONurbsCurve() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zIONurbsCurve::~zIONurbsCurve() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE bool zIONurbsCurve::toZSpaceCurves(MObjectArray &maya_curves, bool onlyCorners, zObjGraph &zspace_graphObj)
	{
		
		zMayaFnGraph tempFn(zspace_graphObj);
		tempFn.fromMayaCurves(maya_curves, onlyCorners);

		return true;
	}

	ZSPACE_INLINE bool zIONurbsCurve::toZSpaceCurves(ON_ClassArray<ON_NurbsCurve> &rhino_curves, bool onlyCorners, zObjGraph &zspace_graphObj)
	{
		zRhinoFnGraph tempFn(zspace_graphObj);
		tempFn.fromRhinoCurves(rhino_curves, onlyCorners);

		return true;
	}

	ZSPACE_INLINE bool zIONurbsCurve::toRhinoCurves(zObjGraph &zspace_graphObj, ON_ClassArray<ON_NurbsCurve> &rhino_curves)
	{
		zRhinoFnGraph tempFn(zspace_graphObj);
		tempFn.toRhinoCurves(rhino_curves);

		return true;
	}

	ZSPACE_INLINE bool zIONurbsCurve::toRhinoCurve(MObject &maya_curve, ON_NurbsCurve &rhino_curve)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_curve.hasFn(fn);

		if (fn != MFn::kNurbsCurve) return false;

		MFnNurbsCurve maya_fnNurbsCurve(maya_curve);

		// vertices
		ON_3dPointArray pts;

		MPointArray pos;
		maya_fnNurbsCurve.getCVs(pos);

		for (int i = 0; i < maya_fnNurbsCurve.numCVs(); i++)
		{
			pts.Append(ON_3fPoint(pos[i].x, pos[i].y, pos[i].z));
		}

		int dimension = 3;
		bool bIsRat = false;
		int order = maya_fnNurbsCurve.degree() + 1;
		int cv_count = maya_fnNurbsCurve.numCVs();

		rhino_curve = ON_NurbsCurve(dimension, bIsRat, order, cv_count);

		//Set CV points
		rhino_curve.ReserveCVCapacity(cv_count);
		for (int i = 0; i < pts.Count(); i++)
		{
			rhino_curve.SetCV(i, pts[i]);
		}

		//Set Knots
		rhino_curve.ReserveKnotCapacity(order + cv_count - 2);
		ON_MakeClampedUniformKnotVector(order, cv_count, rhino_curve.m_knot);

		return true;
	}

	ZSPACE_INLINE bool zIONurbsCurve::toMayaCurves(zObjGraph &zspace_graphObj, MObjectArray &maya_curves)
	{
		zMayaFnGraph tempFn(zspace_graphObj);
		tempFn.toMayaCurves(maya_curves);

		return true;
	}

	ZSPACE_INLINE bool zIONurbsCurve::toMayaCurve(ON_NurbsCurve &rhino_curve, MObject &maya_curve)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_curve.hasFn(fn);

		if (fn != MFn::kNurbsCurve) return false;

		int numVertices = rhino_curve.CVCount();

		MPointArray pts;
		MDoubleArray knots;

		for (int i = 0; i < rhino_curve.CVCount(); i++)
		{
			ON_3dPoint p;
			rhino_curve.GetCV(i, p);

			pts.append(MPoint(p.x, p.y, p.z));
		}

		const double *k = rhino_curve.Knot();

		for (int i = 0; i < rhino_curve.KnotCount(); i++)
		{
			knots.append(k[i]);
		}

		MFnNurbsCurve fn_operateCurve(maya_curve);

		unsigned int deg = 1;
		
		if (rhino_curve.IsClosed())
		{
			fn_operateCurve.create(pts, knots, rhino_curve.Degree(), MFnNurbsCurve::kClosed, false, false);
		}
		else
		{
			fn_operateCurve.create(pts, knots, rhino_curve.Degree(), MFnNurbsCurve::kOpen, false, false);
		}
		
		fn_operateCurve.updateCurve();

		return true;
	}

}

#endif