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


#include<headers/zInterOp/interOp/zIONurbsSurface.h>

#if defined(ZSPACE_MAYA_INTEROP)  && defined(ZSPACE_RHINO_INTEROP)

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zIONurbsSurface::zIONurbsSurface() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zIONurbsSurface::~zIONurbsSurface() {}

	//---- INTEROP METHODS

	ZSPACE_INLINE bool zIONurbsSurface::toRhinoSurface(MObject &maya_surface, ON_NurbsSurface &rhino_surface)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_surface.hasFn(fn);

		if (fn != MFn::kNurbsSurface) return false;

		MFnNurbsSurface maya_fnNurbs(maya_surface);

		// vertices
		ON_3dPointArray pts;

		MPointArray pos;
		maya_fnNurbs.getCVs(pos);

		for (int i = 0; i < maya_fnNurbs.numCVsInU(); i++)
		{
			for (int j = 0; j < maya_fnNurbs.numCVsInV(); j++)
			{
				int id = i * maya_fnNurbs.numCVsInV() + j;
				pts.Append(ON_3fPoint(pos[id].x, pos[id].y, pos[id].z));
			}				
		}

		const int bIsRational = false;
		const int dim = 3;
		int u_degree = maya_fnNurbs.degreeU();
		int v_degree = maya_fnNurbs.degreeV();
		int u_cv_count = maya_fnNurbs.numCVsInU();
		int v_cv_count = maya_fnNurbs.numCVsInV();

		MDoubleArray u_Knot;
		maya_fnNurbs.getKnotsInU(u_Knot);

		MDoubleArray v_Knot;
		maya_fnNurbs.getKnotsInV(v_Knot);		

		rhino_surface = ON_NurbsSurface(dim, bIsRational, u_degree + 1, v_degree + 1, u_cv_count, v_cv_count);

		for (int i = 0; i < rhino_surface.KnotCount(0); i++) rhino_surface.SetKnot(0, i, u_Knot[i]);

		for (int j = 0; j < rhino_surface.KnotCount(1); j++) rhino_surface.SetKnot(1, j, v_Knot[j]);

		for (int i = 0; i < rhino_surface.CVCount(0); i++) 
		{
			for (int j = 0; j < rhino_surface.CVCount(1); j++)
			{
				rhino_surface.SetCV(i, j, pts[i * rhino_surface.CVCount(1) + j]);
			}
		}

		return true;
	}

	ZSPACE_INLINE bool zIONurbsSurface::toMayaSurface(ON_NurbsSurface &rhino_surface, MObject &maya_surface)
	{
		MFn::Type fn = MFn::kInvalid;
		maya_surface.hasFn(fn);

		if (fn != MFn::kNurbsSurface) return false;

		

		MPointArray pts;
		MDoubleArray uKnots, vKnots;

		for (int i = 0; i < rhino_surface.CVCount(0); i++)
		{
			for (int j = 0; j < rhino_surface.CVCount(1); j++)
			{

				ON_3dPoint p;
				rhino_surface.GetCV(i,j, p);
				pts.append(MPoint(p.x, p.y, p.z));
			}

			
		}

		const double *uK = rhino_surface.Knot(0);
		for (int i = 0; i < rhino_surface.KnotCount(0); i++) uKnots.append(uK[i]);
		
		const double *vK = rhino_surface.Knot(1);
		for (int i = 0; i < rhino_surface.KnotCount(1); i++) vKnots.append(vK[i]);

		MFnNurbsSurface fn_operateSurface(maya_surface);		

		if (rhino_surface.IsClosed(0) && rhino_surface.IsClosed(1))
		{
			if(rhino_surface.IsClosed(1))
				fn_operateSurface.create(pts,uKnots,vKnots,rhino_surface.Degree(0), rhino_surface.Degree(1), MFnNurbsSurface::kClosed, MFnNurbsSurface::kClosed, rhino_surface.IsRational());
			else 
				fn_operateSurface.create(pts, uKnots, vKnots, rhino_surface.Degree(0), rhino_surface.Degree(1), MFnNurbsSurface::kClosed, MFnNurbsSurface::kOpen, rhino_surface.IsRational());

		}
		else
		{
			if (rhino_surface.IsClosed(1))
				fn_operateSurface.create(pts, uKnots, vKnots, rhino_surface.Degree(0), rhino_surface.Degree(1), MFnNurbsSurface::kOpen, MFnNurbsSurface::kClosed, rhino_surface.IsRational());
			else
				fn_operateSurface.create(pts, uKnots, vKnots, rhino_surface.Degree(0), rhino_surface.Degree(1), MFnNurbsSurface::kOpen, MFnNurbsSurface::kOpen, rhino_surface.IsRational());
		}

		fn_operateSurface.updateSurface();

		return true;
	}

}

#endif