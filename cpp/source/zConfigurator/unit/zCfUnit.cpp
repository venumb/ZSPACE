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

		for (auto voxelID : voxelIDS)
		{	
			int vCount = positions.size();

			for (zItMeshVertex v(*voxelObjs[voxelID]); !v.end(); v++)
			{
				positions.push_back(v.getPosition());
			}

			for (zItMeshFace f(*voxelObjs[voxelID]); !f.end(); f++)
			{
				zIntArray fVerts;
				f.getVertices(fVerts);

				for (auto& fV : fVerts) pConnects.push_back(vCount + fV);
				pCounts.push_back(fVerts.size());
			}
		}

		zFnMesh fnMesh(combinedVoxelObj);
		fnMesh.create(positions, pCounts, pConnects);
			

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
								p += X * (wallPanel_width + column_primary_width + (beam_width * 0.5));
								positions.push_back(p);
							}

							if (m == xSubdiv)
							{
								// add extra point for the wall panel and strcture center line											
								p += X * xGridDiv;								
								p -= X * (wallPanel_width + column_primary_width + (beam_width * 0.5));

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
		primaryColumns.assign(voxelIDS.size() * 4, zCfColumn());

		int n_columns = 0;

	

		for (auto& voxelID : voxelIDS)
		{
			zTransformationMatrix t = *unitTransform;			

			printf("\n voxId %i ", voxelID);
			zItMeshFace f(*voxelObjs[voxelID],0);

			zItMeshVertexArray fVerts;
			f.getVertices(fVerts);

			
			
			for (int i =0; i< fVerts.size(); i++)
			{
				// set transform rotation
				zFloat4 rot;
				rot[0] = 0; rot[1] = 0; rot[2] = 0; rot[3] = 1;

				if (i == 1) { rot[1] = 180; }
				if (i == 2) { rot[2] = 180; }
				if (i == 3) { rot[1] = 180;  rot[2] = 180; }

				t.setRotation(rot);
				

				// set transform translation
				zPoint p = fVerts[i].getPosition();				

				printf("\n %i: %1.2f %1.2f %1.2f ",i, p.x, p.y, p.z);

				zVector X = t.getX();
				p += X * (wallPanel_width + column_primary_width + (beam_width * 0.5));

				zFloat4 trans;
				trans[0] = fVerts[i].getPosition().x;
				trans[1] = fVerts[i].getPosition().y;
				trans[2] = fVerts[i].getPosition().z;
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

	//---- SET METHODS

	ZSPACE_INLINE void zCfUnit::setVoxels(zObjMeshArray& _voxelObjs)
	{
		for (int i = 0; i < _voxelObjs.size(); i++)
		{
			voxelObjs.push_back(&_voxelObjs[i]);
		}
	}

	ZSPACE_INLINE void zCfUnit::setCenterlineGraph(zObjGraph& _centerLineObj)
	{
		centerlineObj = &_centerLineObj;
	}

	ZSPACE_INLINE void zCfUnit::setVoxelIds(zIntArray& _voxelIDS)
	{
		voxelIDS = _voxelIDS;
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
	}

}