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


#include <headers/zConfigurator/unit/zCfUnit.h>

//---- zCfUnit ------------------------------------------------------------------------------

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zCfUnit::zCfUnit() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zCfUnit::~zCfUnit() {}

	//---- CREATE METHODS

	ZSPACE_INLINE void zCfUnit::createCombinedVoxelMesh()
	{

		zPointArray positions;
		zIntArray pCounts, pConnects;

		zColorArray faceColors;

		for (auto voxelAttrib : voxelAttributes)
		{	
			int vCount = positions.size();

			for (zItMeshVertex v(*o_voxels[voxelAttrib.index]); !v.end(); v++)
			{
				positions.push_back(v.getPosition());
			}

			for (zItMeshFace f(*o_voxels[voxelAttrib.index]); !f.end(); f++)
			{
				zIntArray fVerts;
				f.getVertices(fVerts);

				for (auto& fV : fVerts) pConnects.push_back(vCount + fV);
				pCounts.push_back(fVerts.size());

				faceColors.push_back(f.getColor());
			}
		}

		zFnMesh fnMesh(combinedVoxelObj);
		fnMesh.create(positions, pCounts, pConnects);
			
		fnMesh.setFaceColors(faceColors);
	}

	ZSPACE_INLINE void zCfUnit::createSpacePlanMesh(int xSubdiv, int ySubdiv)
	{
		zPointArray positions;
		zIntArray pCounts, pConnects;

		zFloat4 s;
		unitTransform->getScale(s);

		zVector X = unitTransform->getX();
		zVector Y = unitTransform->getY();
		zVector Z = unitTransform->getZ();


		int xNum = ceil(s[0] / voxelDims->x);
		int yNum = ceil(s[1] / voxelDims->y);
		int zNum = ceil(s[2] / voxelDims->z);

		float xGridDiv = voxelDims->x / (xSubdiv + 1);
		float yGridDiv = voxelDims->y / (ySubdiv + 1);

		for (int i = 0; i < zNum; i++)
		{
			for (int j = 0; j < yNum; j++)
			{
				for (int k = 0; k < xNum; k++)
				{
					int vCount = positions.size();

					zVector ori = unitTransform->getTranslation();

					ori += X * k * voxelDims->x;
					ori += Y * j * voxelDims->y;
					ori += Z * i * voxelDims->z;

					// points
					for (int l = 0; l < ySubdiv + 2; l++)
					{
						for (int m = 0; m < xSubdiv + 2; m++)
						{
							zPoint p = ori;
							p += X * m * xGridDiv;
							p += Y * l * yGridDiv;
							positions.push_back(p);
							
							if (m == 0)
							{
											
								// add extra point for the wall panel and strcture center line
								//p += X * (wallPanel_width + column_primary_width + (beam_width * 0.5)); 
								p += X * (column_primary_width + (beam_width * 0.5));
								positions.push_back(p);
							}

							if (m == xSubdiv)
							{
								// add extra point for the wall panel and strcture center line											
								p += X * xGridDiv;								
								//p -= X * (wallPanel_width + column_primary_width + (beam_width * 0.5));
								p -= X * (column_primary_width + (beam_width * 0.5));

								positions.push_back(p);
								
							}

						}
						
					}

					// pConnects					
					for (int l = 0; l < ySubdiv + 1; l++)
					{
						// 2 additional points added
						for (int m = 0; m < xSubdiv + 1 + 2; m++)
						{
							int id0 = vCount + m + l * (xSubdiv + 4);

							pConnects.push_back(id0);
							
							pConnects.push_back(id0 + 1);
							
							pConnects.push_back(id0 + xSubdiv + 4 + 1);
							
							pConnects.push_back(id0 + xSubdiv + 4);

							pCounts.push_back(4);
						}
					}
				}
			}
		}
	

		zFnMesh fnMesh(unitSpacePlan);
		fnMesh.create(positions, pCounts, pConnects);

	}

	ZSPACE_INLINE void zCfUnit::createPrimaryColumns()
	{
		primaryColumns.clear();
		primaryColumns.assign(voxelAttributes.size() * 4, zCfColumn());

		int n_columns = 0;	

		for (auto& voxelAttrib : voxelAttributes)
		{
			zTransformationMatrix t = *unitTransform;		

			
			zItMeshFace f(*o_voxels[voxelAttrib.index],0);

			zItMeshVertexArray fVerts;
			f.getVertices(fVerts);
			
			
			for (int i =0; i< fVerts.size(); i++)
			{
				// back voxel , outer vertices
				if (voxelAttrib.location == VL_Entrance && fVerts[i].getId() % 4 == 2) { n_columns++; continue; }
				if (voxelAttrib.location == VL_Entrance && fVerts[i].getId() % 4 == 3) { n_columns++; continue; }

				// front voxel , start vertices
				if (voxelAttrib.location == VL_Terrace && fVerts[i].getId() % 4 == 0) { n_columns++; continue; }
				if (voxelAttrib.location == VL_Terrace && fVerts[i].getId() % 4 == 1) { n_columns++; continue; }

				// set transform rotation
				zFloat4 rot;
				rot[0] = 0; rot[1] = 0; rot[2] = 0; rot[3] = 1;

				if (fVerts[i].getId() % 4 == 1) { rot[1] = 180; }
				if (fVerts[i].getId() % 4 == 3) { rot[2] = 180; }
				if (fVerts[i].getId() % 4 == 2) { rot[1] = 180;  rot[2] = 180; }

				t.setRotation(rot);
				
				// set transform translation
				zPoint p = fVerts[i].getPosition();					

				zVector X = t.getX();
				//p += X * (wallPanel_width + column_primary_width + (beam_width * 0.5));
				

				zFloat4 trans;
				trans[0] = p.x;
				trans[1] = p.y;
				trans[2] = p.z;
				trans[3] = 1;

				t.setTranslation(trans);

				// set transform scale
				zFloat4 scale;
				scale[0] = column_primary_length;
				scale[1] = column_primary_width;
				scale[2] = voxelDims->z;
				scale[3] = 1;
				t.setScale(scale);

				//create column
				primaryColumns[n_columns].createSplitColumn(t, beam_width * 0.5, beam_width * 0.5);
				n_columns++;
			}
		}
	}

	ZSPACE_INLINE void zCfUnit::createWalls()
	{
		walls.clear();
		walls.assign(voxelAttributes.size() * 4, zCfWall());

		int n_Walls = 0;
		zTransformationMatrix t = *unitTransform;

		
		for (auto& voxelAttrib : voxelAttributes)
		{	

			// bottom face
			zItMeshFace f(*o_voxels[voxelAttrib.index], 0);

			zItMeshVertexArray fVerts;
			f.getVertices(fVerts);

			for (int i = 0; i < fVerts.size(); i++)
			{
				// set transform rotation
				zFloat4 rot;
				rot[0] = 0; rot[1] = 0; rot[2] = 0; rot[3] = 1;

				if (fVerts[i].getId() % 4 == 1) { rot[1] = 180; }
				if (fVerts[i].getId() % 4 == 3) { rot[2] = 180; }
				if (fVerts[i].getId() % 4 == 2) { rot[1] = 180;  rot[2] = 180; }

				t.setRotation(rot);

				// get face geometry type
				string faceGeomType = voxelAttrib.faceGeomTypes[0];
				if (fVerts[i].getId() % 4 == 0) faceGeomType = voxelAttrib.faceGeomTypes[1];
				if (fVerts[i].getId() % 4 == 1) faceGeomType = voxelAttrib.faceGeomTypes[2];
				if (fVerts[i].getId() % 4 == 2) faceGeomType = voxelAttrib.faceGeomTypes[3];
				if (fVerts[i].getId() % 4 == 3) faceGeomType = voxelAttrib.faceGeomTypes[4];

				// set transform translation
				zPoint p = fVerts[i].getPosition();

				zVector X = t.getX();
				//p += X * (wallPanel_width + column_primary_width + (beam_width * 0.5));


				zFloat4 trans;
				trans[0] = p.x;
				trans[1] = p.y;
				trans[2] = p.z;
				trans[3] = 1;

				t.setTranslation(trans);

				// set transform scale
				zFloat4 scale;
				scale[0] = (fVerts[i].getId() % 4 == 0 || fVerts[i].getId() % 4 == 2) ? voxelDims->x  : voxelDims->y ;
				scale[1] = wallPanel_width;
				scale[2] = voxelDims->z - floor_height - ceiling_height;
				scale[3] = 1;
				t.setScale(scale);

				//create column
				zFloat2 cOffsets = { 0,0 };
				walls[n_Walls].createGuideGraphs(t, beam_width * 0.5, beam_width * 0.5, scale[0] / 8, cOffsets);
				n_Walls++;
			}
		}
	}

	//---- SET METHODS

	ZSPACE_INLINE void zCfUnit::setVoxels(zObjMeshArray& _o_voxels)
	{
		for (int i = 0; i < _o_voxels.size(); i++)
		{
			o_voxels.push_back(&_o_voxels[i]);
		}
	}

	ZSPACE_INLINE void zCfUnit::setCenterlineGraph(zObjGraph& _o_centerline)
	{
		o_centerline = &_o_centerline;
	}

	ZSPACE_INLINE void zCfUnit::setVoxelAttributes(vector<zCfVoxelAttributes>& _voxelATTRIB)
	{
		voxelAttributes = _voxelATTRIB;
	}

	ZSPACE_INLINE void zCfUnit::setVoxelInteriorProgram(zBoolArray& _voxelInteriorProgram)
	{
		voxelInteriorProgram = &_voxelInteriorProgram;
	}

	ZSPACE_INLINE void zCfUnit::setUnitTransform(zTransformationMatrix& _unitTransform)
	{
		unitTransform = &_unitTransform;
	}

	ZSPACE_INLINE void zCfUnit::setVoxelDimensions(zVector& _voxelDims)
	{
		voxelDims = &_voxelDims;
	}

	ZSPACE_INLINE void zCfUnit::setUnitString(string& _unitString)
	{
		computeUnitAttributes(_unitString);
	}

	ZSPACE_INLINE void zCfUnit::setDatabase(zDatabase& _db)
	{
		db = &_db;
	}

	ZSPACE_INLINE void zCfUnit::setFaceColorsFromGeometryTypes()
	{
		for (auto& voxelAttrib : voxelAttributes)
		{
			int fId = 0;		
			
			for (auto& faceGeomType : voxelAttrib.faceGeomTypes)
			{
				zItMeshFace f(*o_voxels[voxelAttrib.index], fId);
				zColor fColor(1,1,1,1);
								
				//ceiling h: 0 to 60, interior v = 1, exterior v = 0.2,  for void s = 0.0, v = 1.0
				if (faceGeomType == VFG_Interior_Ceiling_Arched){	fColor.h = 0; fColor.v = 1.0;  fColor.s = 1.0;	}
				if (faceGeomType == VFG_Nosing_Ceiling_Arched) { fColor.h = 0; fColor.v = 0.5; fColor.s = 1.0;	}

				//wall h: 180 to 240, interior v = 1, exterior v = 0.2 , for void s = 0.0, v = 1.0
				if (faceGeomType == VFG_Interior_Wall_Void) { fColor.h = 240; fColor.v = 1.0;  fColor.s = 0.0; }
				if (faceGeomType == VFG_Nosing_Wall_Void) { fColor.h = 240; fColor.v = 0.5; fColor.s = 0.0;	}

				if (faceGeomType == VFG_Interior_Wall_LongSolid) { fColor.h = 240; fColor.v = 1.0; fColor.s = 1.0; }
				if (faceGeomType == VFG_Nosing_Wall_LongSolid) { fColor.h = 240; fColor.v = 0.5; fColor.s = 1.0;}

				if (faceGeomType == VFG_Interior_Wall_ShortSolid) { fColor.h = 230; fColor.v = 1.0; fColor.s = 1.0; }
				if (faceGeomType == VFG_Nosing_Wall_ShortSolid) { fColor.h = 230; fColor.v = 0.5; fColor.s = 1.0; }

				if (faceGeomType == VFG_Nosing_Wall_LipSolid) { fColor.h = 220; fColor.v = 0.5; fColor.s = 1.0; }
				
				if (faceGeomType == VFG_Interior_Wall_LongGlazed) { fColor.h = 190; fColor.v = 1.0; fColor.s = 1.0; }
				
				if (faceGeomType == VFG_Nosing_Wall_ShortCurveGlazed) { fColor.h = 180; fColor.v = 0.5; fColor.s = 1.0; }

				//floor h: 90 to 150, interior v = 1, exterior v = 0.2,  for void s = 0.0, v = 1.0
				if (faceGeomType == VFG_Interior_Floor_Flat) { fColor.h = 90; fColor.v = 1.0;  fColor.s = 1.0; }
				if (faceGeomType == VFG_Nosing_Floor_Flat) { fColor.h = 90; fColor.v = 0.5; fColor.s = 1.0; }
							

				//-----
				fColor.toRGB();
				f.setColor(fColor);
			

				fId++;
			}


			
			
		}
	}

	//---- GET METHODS

	ZSPACE_INLINE zObjMesh* zCfUnit::getRawCombinedVoxelMesh()
	{
		return &combinedVoxelObj;
	}

	ZSPACE_INLINE zObjMesh* zCfUnit::getRawSpacePlanMesh()
	{
		return &unitSpacePlan;
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zCfUnit::computeUnitAttributes(string& _unitString)
	{
		zStringArray data = coreUtils.splitString(_unitString, "_");

		unitID = data[0];
		unitType = data[1];
		unitSpacePlanType = data[2];

		computeUnitFaceGeometryTypes();
	}

	ZSPACE_INLINE void zCfUnit::computeUnitFaceGeometryTypes()
	{
		int xNum, yNum, zNum;
		int voxelCounter = 0;

		//base
		if (unitType == Unit_Base)
		{
			xNum = 1;
			yNum = 4;
			zNum = 1;
						
			for (int i = 0; i < zNum; i++)
			{
				for (int j = 0; j < yNum; j++)
				{
					for (int k = 0; k < xNum; k++)
					{
						voxelAttributes[voxelCounter].faceGeomTypes.clear();
						voxelAttributes[voxelCounter].faceGeomTypes.assign(6, string());

						// entrance
						if (j == 3)
						{
							voxelAttributes[voxelCounter].faceGeomTypes[0] = VFG_Nosing_Floor_Flat;
							voxelAttributes[voxelCounter].faceGeomTypes[1] = VFG_Nosing_Wall_ShortSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[2] = VFG_Nosing_Wall_LipSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[3] = VFG_Nosing_Wall_Void;
							voxelAttributes[voxelCounter].faceGeomTypes[4] = VFG_Nosing_Wall_LipSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[5] = VFG_Nosing_Ceiling_Arched;
						}

						// interior
						if (j == 2)
						{
							voxelAttributes[voxelCounter].faceGeomTypes[0] = VFG_Interior_Floor_Flat;
							voxelAttributes[voxelCounter].faceGeomTypes[1] = VFG_Interior_Wall_Void;
							voxelAttributes[voxelCounter].faceGeomTypes[2] = VFG_Interior_Wall_LongSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[3] = VFG_Interior_Wall_ShortSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[4] = VFG_Interior_Wall_LongSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[5] = VFG_Interior_Ceiling_Arched;
						}

						if (j == 1)
						{
							voxelAttributes[voxelCounter].faceGeomTypes[0] = VFG_Interior_Floor_Flat;
							voxelAttributes[voxelCounter].faceGeomTypes[1] = VFG_Interior_Wall_Void;
							voxelAttributes[voxelCounter].faceGeomTypes[2] = VFG_Interior_Wall_LongSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[3] = VFG_Interior_Wall_Void;
							voxelAttributes[voxelCounter].faceGeomTypes[4] = VFG_Interior_Wall_LongGlazed;
							voxelAttributes[voxelCounter].faceGeomTypes[5] = VFG_Interior_Ceiling_Arched; 
						}

						if (j == 0)
						{
							voxelAttributes[voxelCounter].faceGeomTypes[0] = VFG_Nosing_Floor_Flat;
							voxelAttributes[voxelCounter].faceGeomTypes[1] = VFG_Nosing_Wall_Void;
							voxelAttributes[voxelCounter].faceGeomTypes[2] = VFG_Nosing_Wall_LipSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[3] = VFG_Nosing_Wall_ShortCurveGlazed;
							voxelAttributes[voxelCounter].faceGeomTypes[4] = VFG_Nosing_Wall_LipSolid;
							voxelAttributes[voxelCounter].faceGeomTypes[5] = VFG_Nosing_Ceiling_Arched; 
						}

						voxelCounter++;
					};

				}
			}
		}
		

		//wide
		if (unitType == Unit_Wide)
		{

		}

		//tall
		if (unitType == Unit_Tall)
		{

		}
		
		//side
		if (unitType == Unit_Side)
		{

		}
	}

}