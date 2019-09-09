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


#include<headers/zToolsets/statics/zTsTopOpt.h>

namespace zSpace
{


	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsTopOpt::zTsTopOpt()
	{

		meshObj = nullptr;
	}

	ZSPACE_INLINE zTsTopOpt::zTsTopOpt(zObjMesh &_meshObj)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);

		SPC_Boolean.assign(fnMesh.numVertices(), false);

		designSpace_Boolean.assign(fnMesh.numPolygons(), false);

		fnMesh.setFaceColor(zColor(0, 1, 0.5, 1));
	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsTopOpt::~zTsTopOpt() {}

	//---- TO METHOD

	ZSPACE_INLINE void zTsTopOpt::to(string path, zFileTpye type)
	{
		if (type == zJSON)
		{
			toJSON(path);

		}
		else throw std::invalid_argument(" error: invalid zFileTpye type");
	}

	//---- CREATE

	ZSPACE_INLINE void zTsTopOpt::createFromFile(string path, zFileTpye type)
	{
		fnMesh.from(path, type, true);
	}

	//--- SET METHODS 

	ZSPACE_INLINE void zTsTopOpt::setSinglePointConstraints(const vector<int> &_SPC)
	{

		if (SPC_Boolean.size() != fnMesh.numVertices())
		{
			SPC_Boolean.clear();
			SPC_Boolean.assign(fnMesh.numVertices(), false);

			fnMesh.setVertexColor(zColor(0.2, 0.2, 0.2, 1));
		}

		for (int i = 0; i < _SPC.size(); i++)
		{
			SPC_Boolean[_SPC[i]] = true;

			zItMeshVertex v(*meshObj, _SPC[i]);
			v.setColor(zColor(1, 0, 0, 1));
		}

		if (_SPC.size() == 0) std::fill(SPC_Boolean.begin(), SPC_Boolean.end(), false);
	}

	ZSPACE_INLINE void zTsTopOpt::setNonDesignSpace(const vector<int> &_NonDesignSpace )
	{
		if (designSpace_Boolean.size() != fnMesh.numPolygons())
		{
			designSpace_Boolean.clear();
			designSpace_Boolean.assign(fnMesh.numPolygons(), true);

			fnMesh.setFaceColor(zColor(0, 1, 0.5, 1));
		}

		for (int i = 0; i < _NonDesignSpace.size(); i++)
		{
			designSpace_Boolean[_NonDesignSpace[i]] = false;

			zItMeshFace f(*meshObj);
			f.setColor(zColor(1, 0, 0.5, 1));
		}

		if (_NonDesignSpace.size() == 0) std::fill(designSpace_Boolean.begin(), designSpace_Boolean.end(), true);
	}

	ZSPACE_INLINE void zTsTopOpt::setMaterial(zTopOptMaterial &material)
	{
		mat = material;
	}

	ZSPACE_INLINE void zTsTopOpt::setPatternGrouping(zTopOptPattern &_pattern)
	{
		pattern = _pattern;
	}

	ZSPACE_INLINE void zTsTopOpt::setPatternGrouping(int type, const zVector &anchor, const zVector &n1, const zVector &n2 )
	{
		pattern.type = type;
		pattern.anchor = anchor;
		pattern.node1 = n1;
		pattern.node2 = n2;
	}

	//--- LOAD METHODS 

	ZSPACE_INLINE void zTsTopOpt::addLoad(double _magnitude, zVector &_dir, vector<int>& vIndices)
	{
		zTopOptLoads load;
		load.magnitude = _magnitude;
		load.dir = _dir;
		load.indicies = vIndices;

		loads.push_back(load);

	}

	ZSPACE_INLINE void zTsTopOpt::removeLoad(int index)
	{
		if (index < 0 || index >= loads.size()) throw std::invalid_argument("input index out of bounds.");

		loads.erase(loads.begin() + index);
	}

	ZSPACE_INLINE void zTsTopOpt::removeLoads()
	{
		loads.clear();
	}

	//---- PROTECTED FACTORY METHODS

	ZSPACE_INLINE void zTsTopOpt::toJSON(string outfilename)
	{
		zUtilsJsonTopOpt meshJSON;
		json j;

		// edges

		//faces
		for (zItMeshFace f(*meshObj); !f.end(); f++)
		{
			vector<int> fVerts;
			f.getVertices(fVerts);

			meshJSON.faces.push_back(fVerts);
		}

		// vertex Attributes
		zVector* vPos = fnMesh.getRawVertexPositions();
		zVector* vNorm = fnMesh.getRawVertexNormals();

		for (int i = 0; i < fnMesh.numVertices(); i++)
		{
			vector<double> v_attrib;

			v_attrib.push_back(vPos[i].x);
			v_attrib.push_back(vPos[i].y);
			v_attrib.push_back(vPos[i].z);

			v_attrib.push_back(vNorm[i].x);
			v_attrib.push_back(vNorm[i].y);
			v_attrib.push_back(vNorm[i].z);

			meshJSON.vertexAttributes.push_back(v_attrib);
		}

		// face Attributes
		zVector* fNorm = fnMesh.getRawFaceNormals();

		for (int i = 0; i < fnMesh.numPolygons(); i++)
		{
			vector<double> f_attrib;

			f_attrib.push_back(fNorm[i].x);
			f_attrib.push_back(fNorm[i].y);
			f_attrib.push_back(fNorm[i].z);

			meshJSON.faceAttributes.push_back(f_attrib);
		}

		// SPC
		meshJSON.SPC = SPC_Boolean;

		// Design Space
		meshJSON.designSpace = designSpace_Boolean;

		// loads
		for (int i = 0; i < loads.size(); i++)
		{
			vector<double> load_attrib;

			load_attrib.push_back(loads[i].magnitude);

			load_attrib.push_back(loads[i].dir.x);
			load_attrib.push_back(loads[i].dir.y);
			load_attrib.push_back(loads[i].dir.z);

			meshJSON.loads.push_back(load_attrib);

			meshJSON.loadPoints.push_back(loads[i].indicies);
		}

		// PatternGrouping
		vector<double> type;
		type.push_back(pattern.type);
		meshJSON.patternGrouping.push_back(type);

		vector<double> anchor;
		anchor.push_back(pattern.anchor.x);
		anchor.push_back(pattern.anchor.y);
		anchor.push_back(pattern.anchor.z);
		meshJSON.patternGrouping.push_back(anchor);

		vector<double> node1;
		node1.push_back(pattern.node1.x);
		node1.push_back(pattern.node1.y);
		node1.push_back(pattern.node1.z);
		meshJSON.patternGrouping.push_back(node1);

		vector<double> node2;
		node2.push_back(pattern.node2.x);
		node2.push_back(pattern.node2.y);
		node2.push_back(pattern.node2.z);
		meshJSON.patternGrouping.push_back(node2);


		// Material
		meshJSON.material.push_back(mat.thickness);
		meshJSON.material.push_back(mat.E);
		meshJSON.material.push_back(mat.G);
		meshJSON.material.push_back(mat.NU);
		meshJSON.material.push_back(mat.RHO);
		meshJSON.material.push_back(mat.ST);
		meshJSON.material.push_back(mat.SC);
		meshJSON.material.push_back(mat.SS);


		// Json file 
		j["Edges"] = meshJSON.edges;
		j["Faces"] = meshJSON.faces;
		j["VertexAttributes"] = meshJSON.vertexAttributes;
		j["FaceAttributes"] = meshJSON.faceAttributes;
		j["SPC"] = meshJSON.SPC;
		j["DesignSpace"] = meshJSON.designSpace;
		j["Loads"] = meshJSON.loads;
		j["LoadPoints"] = meshJSON.loadPoints;
		j["PatternGrouping"] = meshJSON.patternGrouping;
		j["Material"] = meshJSON.material;

		// EXPORT	

		ofstream myfile;
		myfile.open(outfilename.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << outfilename.c_str() << endl;
			return;
		}

		//myfile.precision(16);
		myfile << j.dump(1);
		myfile.close();
	}

}