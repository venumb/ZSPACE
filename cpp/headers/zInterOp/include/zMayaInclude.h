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

#if defined(ZSPACE_MAYA_INTEROP) 

#ifndef ZSPACE_MAYA_INCLUDE_H
#define ZSPACE_MAYA_INCLUDE_H

#pragma once

#include <maya/MPxLocatorNode.h> 
#include <maya/MString.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MVector.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/M3dView.h>
#include <maya/MDistance.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MIntArray.h>
#include <maya/MFnDagNode.h>
#include <maya/MIntArray.h>
#include <maya/MFnGenericAttribute.h>
#include <maya/MDataHandle.h>
#include <maya/MDataBlock.h>
#include <maya/MFnNurbsSurface.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MDagPath.h>
#include <maya/MPlugArray.h>
#include <maya/MFnDagNode.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMesh.h>
#include <maya/MVectorArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MTime.h>
#include <maya/MFnNurbsCurveData.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnMeshData.h>
#include <maya/MTimer.h>
#include <maya/MQuaternion.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnCamera.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnFluid.h>
#include <maya/MDGModifier.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MMatrixArray.h>
#include <maya/MFnMatrixData.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MFnComponentListData.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MFnSet.h>
#include <maya/MFnTransform.h>

#include <maya/MPxCommand.h>
#include <maya/MAnimControl.h>
#include <maya/MArrayDataBuilder.h>

#include <maya/MEventMessage.h>
#include <maya/MStreamUtils.h>

#include<maya/MFnLambertShader.h>
#include<maya/MFnBlinnShader.h>
#include<maya/MFnPhongShader.h>
#include<maya/MItDependencyNodes.h>

#include <maya/MFnIntArrayData.h>
#include <maya/MObjectArray.h>
#include <maya/MFnNurbsCurveData.h>
#include <maya/MItMeshEdge.h>
#include <maya/MItMeshVertex.h>
#include <maya/MFloatVector.h>
#include <maya/MItMeshPolygon.h>

#include <maya/MFnNurbsSurfaceData.h>

#include <maya/MFileIO.h>
#include <maya/MLibrary.h>
#include <maya/MIOStream.h>

// Viewport 2.0
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MViewport2Renderer.h>



#endif

#endif