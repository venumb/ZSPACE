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


#include <headers/zConfigurator/base/zCfVoxels.h>

//---- zCfVoxels ------------------------------------------------------------------------------

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zCfVoxels::zCfVoxels() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zCfVoxels::~zCfVoxels() {}

	//---- SET METHODS

	ZSPACE_INLINE void zCfVoxels::setGlobalSetoutAnchor(zPoint _anchor)
	{
		globalSetoutAnchor = _anchor;
	}

	ZSPACE_INLINE void zCfVoxels::setDatabase(char* DatabaseFileName)
	{
		db = new zDatabase(DatabaseFileName);
	}

	//---- CREATE METHODS

	ZSPACE_INLINE void zCfVoxels::createVoxelsFromFiles(zStringArray filePaths, zFileTpye type)
	{
		n_voxels = filePaths.size();
		o_voxels.clear();
		o_voxels.assign(n_voxels, zObjMesh());

		if (type == zJSON)
		{
			for (int i = 0; i < filePaths.size(); i++)
			{
				if (i < n_voxels)
				{
					zFnMesh fnMesh(o_voxels[i]);
					fnMesh.from(filePaths[i], type);
				}

			}
		}

		else if (type == zOBJ)
		{
			for (int i = 0; i < filePaths.size(); i++)
			{
				if (i < n_voxels)
				{
					zFnMesh fnMesh(o_voxels[i]);
					fnMesh.from(filePaths[i], type);
				}
			}
		}

		else throw std::invalid_argument(" error: invalid zFileTpye type");

	}

	ZSPACE_INLINE void zCfVoxels::createVoxelsFromTransforms(vector<zTransformationMatrix> &unitTransforms, zStringArray& unitAttributes, zVector& _voxelDims)
	{
		voxelDims = _voxelDims;
		
		n_voxels = computeNumVoxelfromTransforms(unitTransforms);
		o_voxels.clear();
		o_voxels.assign(n_voxels, zObjMesh());

		voxelInteriorProgram.clear();
		voxelInteriorProgram.assign(n_voxels, false);

		units.assign(unitTransforms.size(), zCfUnit());
		
		//-----
		
		int vCounter = 0; 
		int unitCounter = 0;

		// create voxels
		for (auto& t : unitTransforms)
		{
			// compute voxel number from scale
			zFloat4 s;
			t.getScale(s);

			zVector X = t.getX();
			zVector Y = t.getY();
			zVector Z = t.getZ();
			

			int xNum = ceil(s[0] / voxelDims.x);
			int yNum = ceil(s[1] / voxelDims.y);
			int zNum = ceil(s[2] / voxelDims.z);

			// create voxel mesh
			for (int i = 0; i < zNum; i++)
			{
				for (int j = 0; j < yNum; j++)
				{
					for (int k = 0; k < xNum; k++)
					{
						zVector ori = t.getTranslation();
						
						ori += X * k * voxelDims.x;
						ori += Y * j * voxelDims.y;
						ori += Z * i * voxelDims.z;


						// points
						zPointArray pts; 

						pts.push_back(ori);
						pts.push_back(ori + X * voxelDims.x);						
						pts.push_back(ori + Y * voxelDims.y);
						pts.push_back(ori + X * voxelDims.x + Y * voxelDims.y);

						pts.push_back(pts[0] + Z * voxelDims.z);
						pts.push_back(pts[1] + Z * voxelDims.z);
						pts.push_back(pts[2] + Z * voxelDims.z);
						pts.push_back(pts[3] + Z * voxelDims.z);

						// polyconnects
						zIntArray pConnects;
						zIntArray pCounts;

						pConnects.push_back(0); pConnects.push_back(2); pConnects.push_back(3);  pConnects.push_back(1);
						pConnects.push_back(0); pConnects.push_back(1); pConnects.push_back(5);  pConnects.push_back(4);
						pConnects.push_back(1); pConnects.push_back(3); pConnects.push_back(7);  pConnects.push_back(5);
						pConnects.push_back(3); pConnects.push_back(2); pConnects.push_back(6);  pConnects.push_back(7);
						pConnects.push_back(2); pConnects.push_back(0); pConnects.push_back(4);  pConnects.push_back(6);
						pConnects.push_back(4); pConnects.push_back(5); pConnects.push_back(7);  pConnects.push_back(6);

						pCounts.push_back(4); pCounts.push_back(4); pCounts.push_back(4);
						pCounts.push_back(4); pCounts.push_back(4); pCounts.push_back(4);

						// create Voxels
						
						if (vCounter < n_voxels)
						{
							zFnMesh fnVoxel(o_voxels[vCounter]);
							fnVoxel.create(pts, pCounts, pConnects);				
							
							vCounter++;
						}
						
						else throw std::invalid_argument(" error: voxel out of bounds");


						// assign  interior / exterior program

						if (k == 0 || k == xNum - 1) voxelInteriorProgram[vCounter - 1] = false;
						else voxelInteriorProgram[vCounter - 1] = true;
						
					}
				}
			}
		}


		// create units
		vCounter = 0;
		for (auto& t : unitTransforms)
		{
			// compute voxel number from scale
			zFloat4 s;
			t.getScale(s);

			zVector X = t.getX();
			zVector Y = t.getY();
			zVector Z = t.getZ();


			int xNum = ceil(s[0] / voxelDims.x);
			int yNum = ceil(s[1] / voxelDims.y);
			int zNum = ceil(s[2] / voxelDims.z);

			
			vector<zCfVoxelAttributes> unit_VoxelAttribs;
			for (int i = 0; i < zNum; i++)
			{
				for (int j = 0; j < yNum; j++)
				{
					for (int k = 0; k < xNum; k++)
					{
						unit_VoxelAttribs.push_back(zCfVoxelAttributes());

						int id = unit_VoxelAttribs.size() - 1;

						unit_VoxelAttribs[id].index = vCounter;

						// default middle,interior
						unit_VoxelAttribs[id].location = VL_Interior;
						unit_VoxelAttribs[id].occupancy = 1.0;

						// front,terrace
						if( j == 0 ) 
						{
							unit_VoxelAttribs[id].location = VL_Terrace;
							unit_VoxelAttribs[id].occupancy = 0.75;
						}

						// back,entrance
						if (j == yNum -1)
						{
							unit_VoxelAttribs[id].location = VL_Entrance;
							unit_VoxelAttribs[id].occupancy = 0.25;
						}

						vCounter += 1;
					}
				}
							
			}


			// set unit attibutes
			units[unitCounter].setUnitTransform(t);
			units[unitCounter].setVoxelDimensions(voxelDims);
			units[unitCounter].setCenterlineGraph(o_centerline);
			units[unitCounter].setVoxels(o_voxels);
			units[unitCounter].setVoxelAttributes(unit_VoxelAttribs);
			units[unitCounter].setUnitString(unitAttributes[unitCounter]);
			units[unitCounter].setDatabase(*db);
			units[unitCounter].setVoxelInteriorProgram(voxelInteriorProgram);
			units[unitCounter].setFaceColorsFromGeometryTypes();

			//-- create methods
			units[unitCounter].createCombinedVoxelMesh();
			units[unitCounter].createSpacePlanMesh();
			units[unitCounter].createPrimaryColumns();
			units[unitCounter].createWalls();

			unitCounter++;
		}


	}

	ZSPACE_INLINE void zCfVoxels::createCenterLineFromVoxels(zColor edgeCol )
	{

		zPointArray positions;
		zIntArray  edgeConnects;

		zSparseMatrix C_fc;
		getGlobal_FaceCellMatrix( C_fc);
		//cout << "\n C_fc :" << endl << C_fc << endl;


		for (int j = 0; j < n_voxels; j++)
		{
			zFnMesh fnVoxel(o_voxels[j]);
			positions.push_back(fnVoxel.getCenter());
		}

		edgeConnects.assign(global_n_f_i * 2, -1);

		for (int j = 0; j < C_fc.rows(); j++)
		{
			int edgeId = j;

			for (int i = 0; i < C_fc.cols(); i++)
			{
				int vertexId = i;

				// head of the edge
				if (C_fc.coeff(edgeId, vertexId) == 1)
				{
					edgeConnects[edgeId * 2 + 1] = vertexId;
				}

				// tail of the edge
				if (C_fc.coeff(edgeId, vertexId) == -1)
				{
					edgeConnects[edgeId * 2] = vertexId;
				}
			}
		}

		zFnGraph fnCenterline(o_centerline);
		fnCenterline.clear();
		fnCenterline.create(positions, edgeConnects);
		fnCenterline.setEdgeColor(edgeCol);
		
	}

	//--- TOPOLOGY QUERY METHODS 
	
	ZSPACE_INLINE int zCfVoxels::numVoxels()
	{
		return n_voxels;
	}

	//---- GET METHODS

	ZSPACE_INLINE zObjGraph* zCfVoxels::getRawGraph()
	{
		return &o_centerline;
	}

	ZSPACE_INLINE zObjMesh* zCfVoxels::getRawVoxel(int id)
	{
		if (id  >= n_voxels) throw std::invalid_argument(" error: null pointer.");
		
		return &o_voxels[id];
	}

	//---- UTILITY METHODS

	ZSPACE_INLINE void zCfVoxels::computeGlobalElementIndicies( int precisionFac)
	{
		global_n_f = 0;
		global_n_f_i = 0;

		globalFace_VoxelFace.clear();

		Global_BoundaryFace.clear();
		global_internalFaceIndex.clear();
	
		internalFaceIndex_globalFace.clear();

		unordered_map <string, int> positionVertex;
		unordered_map <string, int> faceCenterpositionVertex;



		// face map
		for (int j = 0; j < n_voxels; j++)
		{
			zFnMesh fnVoxel(o_voxels[j]);

			zPointArray fCenters;

			fnVoxel.getCenters(zFaceData, fCenters);


			for (int i = 0; i < fCenters.size(); i++)
			{
				int globalFaceId = -1;
				bool chkExists = coreUtils.vertexExists(faceCenterpositionVertex, fCenters[i], precisionFac, globalFaceId);


				if (!chkExists)
				{
					coreUtils.addToPositionMap(faceCenterpositionVertex, fCenters[i], global_n_f, precisionFac);

					zIntArray  volumeFace = { j,i };
					globalFace_VoxelFace.push_back(volumeFace);

					globalFaceId = global_n_f;


					// GFP or SSP are external faces
					Global_BoundaryFace.push_back(true);


					global_n_f++;
				}
				else
				{
					// GFP or SSP are external faces
					Global_BoundaryFace[globalFaceId] = false;
				}

				string hashKey_volFace = (to_string(j) + "," + to_string(i));
				voxelFace_GlobalFace[hashKey_volFace] = globalFaceId;
			}
		}


		// compute internal face index
		for (int i = 0; i < Global_BoundaryFace.size(); i++)
		{
			if (!Global_BoundaryFace[i])
			{
				global_internalFaceIndex.push_back(global_n_f_i);
				internalFaceIndex_globalFace.push_back(i);
				global_n_f_i++;
			}
			else global_internalFaceIndex.push_back(-1);
		}		

	}
		
	ZSPACE_INLINE bool zCfVoxels::getGlobal_FaceCellMatrix(zSparseMatrix& out)
	{
		
			if (global_n_f_i == 0) return false;

			int global_n_vol = n_voxels;

			out = zSparseMatrix(global_n_f_i, global_n_vol);
			out.setZero();

			vector<zTriplet> coefs;

			for (int j = 0; j < n_voxels; j++)
			{
				int volId = j;


				for (zItMeshFace f(o_voxels[volId]); !f.end(); f++)
				{
					int faceId = f.getId();

					string hashKey_f = (to_string(volId) + "," + to_string(faceId));
					std::unordered_map<std::string, int>::const_iterator gotFace = voxelFace_GlobalFace.find(hashKey_f);

					if (gotFace != voxelFace_GlobalFace.end())
					{
						int p_f = gotFace->second;

						if (Global_BoundaryFace[p_f]) continue;

						int p_volId = globalFace_VoxelFace[p_f][0];
						int p_faceId = globalFace_VoxelFace[p_f][1];

						if (volId == p_volId && faceId == p_faceId && !Global_BoundaryFace[p_f])
						{
							coefs.push_back(zTriplet(global_internalFaceIndex[p_f], volId, 1.0));
						}
						else coefs.push_back(zTriplet(global_internalFaceIndex[p_f], volId, -1.0));
					}

				}

			}

			out.setFromTriplets(coefs.begin(), coefs.end());

			return true;		

	}

	ZSPACE_INLINE void zCfVoxels::computeVoxelSetoutFace(zVector fNorm)
	{
		voxelSetoutFaceId.clear();

		for (int j = 0; j < n_voxels; j++)
		{			
			for (zItMeshFace f(o_voxels[j]); !f.end(); f++)
			{
				if (f.getNormal() == fNorm)
				{
					voxelSetoutFaceId.push_back(f.getId());

					printf("\n vox: %i face : %i ", j, f.getId());
				}
			}
		}
	}

	ZSPACE_INLINE void zCfVoxels::computeVoxelHistory()
	{
		voxelHistory.clear();

		voxelHistory.assign(n_voxels, zStringArray());

		
		for (int j = 0; j < n_voxels; j++)
		{
			string cmd;

			zItMeshFace f(o_voxels[j], voxelSetoutFaceId[j]);

			// 0. Id
			voxelHistory[j].push_back(CMD_INDEX);
			cmd += " voxel_";
			cmd += to_string(j);

			voxelHistory[j].push_back(cmd);
			cmd.clear();

			// 1. Origin Point
			voxelHistory[j].push_back(CMD_ORIGIN);

			cmd += to_string(globalSetoutAnchor.x);
			cmd += " ";
			cmd += to_string(globalSetoutAnchor.y);
			cmd += " ";
			cmd += to_string(globalSetoutAnchor.z);
			
			voxelHistory[j].push_back(cmd);
			cmd.clear();

			// 2. Translation
			voxelHistory[j].push_back(CMD_TRANSLATE);

			zVector trans = f.getCenter() - globalSetoutAnchor;

			cmd += to_string(trans.x);
			cmd += " ";
			cmd += to_string(trans.y);
			cmd += " ";
			cmd += to_string(trans.z);

			voxelHistory[j].push_back(cmd);
			cmd.clear();

			// 3. Create Quad
			voxelHistory[j].push_back(CMD_QUADCENTERED);

			zPointArray fVerts;
			f.getVertexPositions(fVerts);

			for (auto& p : fVerts)
			{
				zVector qv = p - f.getCenter();

				cmd += to_string(qv.x);
				cmd += " ";
				cmd += to_string(qv.y);
				cmd += " ";
				cmd += to_string(qv.z);
				cmd += " ";
			}
			
			voxelHistory[j].push_back(cmd);
			cmd.clear();

			// 4. Extrude 
			voxelHistory[j].push_back(CMD_EXTRUDE);

			zVector minBB, maxBB;
			zFnMesh fnVoxel(o_voxels[j]);
			fnVoxel.getBounds(minBB, maxBB);

			zVector norm(0, 0, 1);
			cmd += to_string(norm.x);
			cmd += " ";
			cmd += to_string(norm.y);
			cmd += " ";
			cmd += to_string(norm.z);
			cmd += " ";
			cmd += to_string(maxBB.z - minBB.z);

			voxelHistory[j].push_back(cmd);
			cmd.clear();

			
		}
	}

	ZSPACE_INLINE int zCfVoxels::computeNumVoxelfromTransforms(vector<zTransformationMatrix>& transforms)
	{
		int vCounter = 0;

		for (auto& t : transforms)
		{
			zFloat4 s;
			t.getScale(s);

			zVector X = t.getX();
			zVector Y = t.getY();
			zVector Z = t.getZ();


			int xNum = ceil(s[0] / voxelDims.x);
			int yNum = ceil(s[1] / voxelDims.y);
			int zNum = ceil(s[2] / voxelDims.z);

			vCounter += (xNum * yNum * zNum);
		}

		return vCounter;
	}

}