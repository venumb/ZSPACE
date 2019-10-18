// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling@zaha-hadid.com>
//


#include<headers/zToolsets/geometry/zTsMesh2Pix.h>


namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsMesh2Pix::zTsMesh2Pix() {}

	ZSPACE_INLINE zTsMesh2Pix::zTsMesh2Pix(zObjMesh &_meshObj, int _maxVerts, int _maxEdges, int _maxFaces)
	{
		meshObj = &_meshObj;
		fnMesh = zFnMesh(_meshObj);

		maxVertices = _maxVerts;
		maxEdges = _maxEdges;
		maxFaces = _maxFaces;

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsMesh2Pix::~zTsMesh2Pix() {}

	//---- GENERATE DATA METHODS
	
	ZSPACE_INLINE void zTsMesh2Pix::generatePrintSupport2Pix(string directory, string filename, double angle_threshold, bool train, int numIters, bool perturbPositions, zVector perturbVal)
	{
		vector<MatrixXd> outMat_A;
		vector<MatrixXd> outMat_B;

		vector<MatrixXd> outMat;

		// make folders
		string trainDir = directory + "/train/";
		string testDir = directory + "/test/";


		int numTrainFiles = 0;
		numTrainFiles = coreUtils.getNumfiles_Type(trainDir, zPNG);

		int numTestFiles = 0;
		numTestFiles = coreUtils.getNumfiles_Type(testDir, zPNG);

		if (numTrainFiles == 0) _mkdir(trainDir.c_str());
		if (numTestFiles == 0) _mkdir(testDir.c_str());

		if (!perturbPositions)
		{
			outMat_A.clear();


			// get Matrix from vertex normals
			zDomainDouble outDomain_A(0.05, 0.45);
			getMatrixFromNormals(zVertexVertex, outDomain_A, outMat_A);

			// get edge length data 
			zDoubleArray heLength;
			zIntPairArray hedgeVertexPair;
			zDomainDouble outDomain(0.5, 0.9);

			for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
			{
				heLength.push_back(he.getLength());

				zIntPair vertPair;
				vertPair.first = he.getStartVertex().getId();
				vertPair.second = he.getVertex().getId();

				hedgeVertexPair.push_back(vertPair);
			}

			getMatrixFromContainer(zVertexVertex, fnMesh.numVertices(), heLength, hedgeVertexPair, outDomain, outMat_A);

			// support matrix 
			outMat_B.clear();

			zDomainDouble outDomain_B(0.0, 0.9);

			zBoolArray supports;
			getVertexSupport(angle_threshold, supports);

			getMatrixFromContainer(zVertexVertex, supports, outDomain_B, outMat_B);

			// combine matrix

			getCombinedMatrix(outMat_B, outMat_A, outMat);
			string path3 = (train) ? directory + "/train/" + filename + ".png" : directory + "/test/" + filename + ".png";
			coreUtils.matrixToPNG(outMat, path3);
			
		}

		else
		{
			// to generate different random number every time the program runs
			srand(time(NULL));

			vector<double> randNumberX;
			vector<double> randNumberY;
			vector<double> randNumberZ;
						
			for (int i = 0; i < numIters * fnMesh.numVertices(); i++)
			{
				double x = coreUtils.randomNumber_double(perturbVal.x * -1, perturbVal.x);
				double y = coreUtils.randomNumber_double(perturbVal.y * -1, perturbVal.y);
				double z = coreUtils.randomNumber_double(perturbVal.z * -1, perturbVal.z);

				randNumberX.push_back(x);
				randNumberY.push_back(y);
				randNumberZ.push_back(z);
			}			

			for (int j = 0; j < numIters; j++)
			{
				zPointArray originalPoints;
				fnMesh.getVertexPositions(originalPoints);

				// translate vertices
				zPoint* vertPos = fnMesh.getRawVertexPositions();

				for (int i = 0; i < fnMesh.numVertices(); i++)
				{
					int id = j * fnMesh.numVertices() + i;

					vertPos[i] += zVector(randNumberX[id], randNumberY[id], 0);
				}

				fnMesh.computeMeshNormals();

				// export

				bool train = ((j + 1) > floor((float) numIters* 0.8)) ? false : true;
				int id = (train) ? numTrainFiles++ : numTestFiles++;					

				string tmp_fileName = (train) ? filename + "_train_" + to_string(id) : filename + "_test_" + to_string(id);
				generatePrintSupport2Pix(directory, tmp_fileName, angle_threshold, train);

				// reset mesh positions				
				fnMesh.setVertexPositions(originalPoints);
			}

		}


	}
	
	ZSPACE_INLINE void zTsMesh2Pix::predictPrintSupport2Pix(string directory, string filename, bool genPix)
	{
		vector<MatrixXd> outMat;	

		// generate prediction image
		if (genPix)
		{
			string predictDir = directory + "/predict/";
			int numPredictFiles = 0;
			numPredictFiles = coreUtils.getNumfiles_Type(predictDir, zPNG);
			if (numPredictFiles == 0) _mkdir(predictDir.c_str());

			vector<MatrixXd> outMat_A;
			vector<MatrixXd> outMat_B;

			outMat_A.clear();

			// get Matrix from vertex normals
			zDomainDouble outDomain_A(0.05, 0.45);
			getMatrixFromNormals(zVertexVertex, outDomain_A, outMat_A);

			// get edge length data 
			zDoubleArray heLength;
			zIntPairArray hedgeVertexPair;
			zDomainDouble outDomain(0.5, 0.9);

			for (zItMeshHalfEdge he(*meshObj); !he.end(); he++)
			{
				heLength.push_back(he.getLength());

				zIntPair vertPair;
				vertPair.first = he.getStartVertex().getId();
				vertPair.second = he.getVertex().getId();

				hedgeVertexPair.push_back(vertPair);
			}

			getMatrixFromContainer(zVertexVertex, fnMesh.numVertices(), heLength, hedgeVertexPair, outDomain, outMat_A);

			// support matrix 
			outMat_B.clear();

			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			MatrixXd R(n_v, n_v);
			MatrixXd G(n_v, n_v);
			MatrixXd B(n_v, n_v);

			R.setConstant(0.95);
			G.setConstant(0.95);
			B.setConstant(0.95);

			outMat_B.push_back(R);
			outMat_B.push_back(G);
			outMat_B.push_back(B);


			// combine matrix

			getCombinedMatrix(outMat_B, outMat_A, outMat);
			string path3 = directory + "/predict/" + filename + ".png";
			coreUtils.matrixToPNG(outMat, path3);
			
		}

		else
		{
			string predictFile = directory + filename + ".png";
			coreUtils.matrixFromPNG(outMat, predictFile);

			// color vertex color
			zColor * vCols = fnMesh.getRawVertexColors();
			
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				vCols[i].r = outMat[0](i, i);
				vCols[i].g = outMat[1](i, i);
				vCols[i].b = outMat[2](i, i);
				vCols[i].a = outMat[3](i, i);
			}
		}

		

	}

	//---- PRIVATE GET METHODS

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromNormals(zConnectivityType type, zDomainDouble &outDomain, vector<MatrixXd> &normMat)
	{
		if (type == zVertexVertex)
		{		

			// get vertex normals diagonal matrix
			zVectorArray norms;
			fnMesh.getVertexNormals(norms);
			getMatrixFromContainer(type, norms, outDomain, normMat);
			
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}
	
	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, zVectorArray &data, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexVertex)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			if (outMat.size() == 0)
			{			

				MatrixXd R(n_v, n_v);
				MatrixXd G(n_v, n_v);
				MatrixXd B(n_v, n_v);

				R.setConstant(0.95);
				G.setConstant(0.95);
				B.setConstant(0.95);

				outMat.push_back(R);
				outMat.push_back(G);
				outMat.push_back(B);
			}


			for (int i = 0; i < data.size(); i++)
			{
				outMat[0](i, i) = coreUtils.ofMap(data[i].x, -1.0, 1.0, outDomain.min, outDomain.max);
				outMat[1](i, i) = coreUtils.ofMap(data[i].y, -1.0, 1.0, outDomain.min, outDomain.max);
				outMat[2](i, i) = coreUtils.ofMap(data[i].z, -1.0, 1.0, outDomain.min, outDomain.max);	

				//printf("\n %i : %1.2f %1.2f %1.2f ",i, outMat[0](i, i) * 255, outMat[1](i, i) * 255, outMat[2](i, i) * 255);
			}

			
		}
				

		else throw std::invalid_argument(" error: invalid zConnectivityType");

	}

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, zBoolArray &data, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexVertex)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();		

			if (outMat.size() == 0)
			{
				MatrixXd R(n_v, n_v);
				MatrixXd G(n_v, n_v);
				MatrixXd B(n_v, n_v);

				R.setConstant(0.95);
				G.setConstant(0.95);
				B.setConstant(0.95);

				outMat.push_back(R);
				outMat.push_back(G);
				outMat.push_back(B);
			}


			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				for (int j = 0; j < fnMesh.numVertices(); j++)
				{
					if (i == j)
					{
						if (data[i])
						{							
							outMat[0](i, j) = 1.0;
							outMat[1](i, j) = 0.0;
							outMat[2](i, j) = 0.0;
						}

						else
						{
							outMat[0](i, j) = 0.0;
							outMat[1](i, j) = 1.0;
							outMat[2](i, j) = 0.0;
						}
					}
					
					else
					{
						outMat[0](i, j) = 0.0;
						outMat[1](i, j) = 0.0;
						outMat[2](i, j) = 1.0;
					}
				}
				
								
			}

			
		}

		

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}

	ZSPACE_INLINE void zTsMesh2Pix::getMatrixFromContainer(zConnectivityType type, int numVerts, zDoubleArray &data, zIntPairArray &dataPair, zDomainDouble &outDomain, vector<MatrixXd> &outMat)
	{
		if (type == zVertexVertex)
		{
			int n_v = (maxVertices != -1) ? maxVertices : fnMesh.numVertices();

			if (outMat.size() == 0)
			{
				MatrixXd R(n_v, n_v);
				MatrixXd G(n_v, n_v);
				MatrixXd B(n_v, n_v);

				R.setConstant(0.95);
				G.setConstant(0.95);
				B.setConstant(0.95);

				outMat.push_back(R);
				outMat.push_back(G);
				outMat.push_back(B);
			}
				

			zDomainDouble inDomain;

			inDomain.min = coreUtils.zMin(data);
			inDomain.max = coreUtils.zMax(data);

			if (inDomain.min == inDomain.max) inDomain.min = 0.0;

			for (int i = 0; i < numVerts; i++)
			{
				for (int j = 0; j < numVerts; j++)
				{
					if (i != j)
					{
						outMat[0](i, j) = 0.00;
						outMat[1](i, j) = 0.00;
						outMat[2](i, j) = 0.00;
					}
				}
			}

			for (int k = 0; k < dataPair.size(); k++)
			{
				int  i = dataPair[k].first;
				int  j = dataPair[k].second;

				outMat[0](i, j) = coreUtils.ofMap(data[k], inDomain, outDomain);
				outMat[1](i, j) = outMat[0](i, j);
				outMat[2](i, j) = outMat[0](i, j);
			}	

			
			
		}

		else throw std::invalid_argument(" error: invalid zConnectivityType");
	}

	ZSPACE_INLINE void zTsMesh2Pix::getVertexSupport(double angle_threshold, zBoolArray &support)
	{
		int numVerts_lowPoly = fnMesh.numVertices();
		
		support.assign(numVerts_lowPoly, false);

		// get Duplicate
		zObjMesh smoothMesh;
		fnMesh.getDuplicate(smoothMesh);
		
		// get smooth mesh

		zFnMesh fnSmoothMesh(smoothMesh);
		fnSmoothMesh.smoothMesh(3);
		
		// get bounds
		zVector minBB, maxBB;
		fnSmoothMesh.getBounds(minBB, maxBB);

		// compute nearest lowest vertex
		zVector* positions = fnSmoothMesh.getRawVertexPositions();

		for (zItMeshVertex vIt(*meshObj); !vIt.end(); vIt++)
		{

			if (vIt.getId() >= numVerts_lowPoly) continue;

			zIntArray cVerts;
			vIt.getConnectedVertices(cVerts);

			int lowestId;
			double val = 10e10;
			for (auto vId : cVerts)
			{
				double zVal = positions[vId].z;

				if (zVal < val)
				{
					lowestId = vId;
					val = zVal;
				}
			}

			zVector lowestV = positions[lowestId];

			zVector vec = vIt.getPosition() - lowestV;
			zVector unitz = zVector(0, 0, 1);

			double ang = vec.angle(unitz);

			// compute support
			
			if (positions[vIt.getId()].z > minBB.z)
					(ang > (angle_threshold)) ? support[vIt.getId()] = true : support[vIt.getId()] = false;
						
		}
	}

	ZSPACE_INLINE void zTsMesh2Pix::getCombinedMatrix(vector<MatrixXd>& mat1, vector<MatrixXd>& mat2, vector<MatrixXd>& out)
	{
		if (mat1.size() == 0) throw std::invalid_argument(" error: mat1 container size is 0. ");
		if (mat2.size() == 0) throw std::invalid_argument(" error: mat2 container size is 0. ");
		if (mat1.size() != mat2.size()) throw std::invalid_argument(" error: sizes of the matrix container dont match. ");
		if (mat1[0].cols() != mat2[0].cols() ) throw std::invalid_argument(" error: rows of the matrix dont match. ");

		int nRows = mat1[0].rows() + mat2[0].rows();
		int nCols = mat1[0].cols()  ;

		out.clear();

		for (int i = 0; i < mat1.size(); i++)
		{
			MatrixXd temp(nRows, nCols);

			temp << mat1[i], mat2[i];

			out.push_back(temp);
		}

	}

}