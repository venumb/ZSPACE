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


#include<headers/zToolsets/statics/zTsVault.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	template<typename T, typename U>
	ZSPACE_INLINE zTsVault<T,U>::zTsVault() {}

	//---- graph specilization for zTsVault overloaded constructor 
	template<>
	ZSPACE_INLINE  zTsVault<zObjGraph, zFnGraph>::zTsVault(zObjGraph & _resultObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnGraph(_resultObj);
	}

	//---- mesh specilization for zTsVault overloaded constructor 
	template<>
	ZSPACE_INLINE  zTsVault<zObjMesh, zFnMesh>::zTsVault(zObjMesh & _resultObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnMesh(_resultObj);
	}

	//---- graph specilization for zTsVault overloaded constructor 
	template<>
	ZSPACE_INLINE  zTsVault<zObjGraph, zFnGraph>::zTsVault(zObjGraph & _resultObj, zObjGraph &_formObj, zObjMesh &_forceObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnGraph(_resultObj);

		formObj = &_formObj;
		fnForm = zFnGraph(_formObj);

		forceObj = &_forceObj;
		fnForce = zFnMesh(_forceObj);
		
	}

	//---- mesh specilization for zTsVault overloaded constructor 
	template<>
	ZSPACE_INLINE  zTsVault<zObjMesh, zFnMesh>::zTsVault(zObjMesh & _resultObj, zObjMesh &_formObj, zObjMesh &_forceObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnMesh(_resultObj);

		formObj = &_formObj;
		fnForm = zFnMesh(_formObj);

		forceObj = &_forceObj;
		fnForce = zFnMesh(_forceObj);

	}

	//---- DESTRUCTOR

	template<typename T, typename U>
	ZSPACE_INLINE zTsVault<T, U>::~zTsVault() {}

	//---- CREATE METHODS

	//---- graph specilization for createResultfromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createResultfromFile(string path, zFileTpye type)
	{
		fnResult.from(path, type);

		setTensionEdges(zResultDiagram);
		setelementColorDomain(zResultDiagram);


		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createResultfromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createResultfromFile(string path, zFileTpye type)
	{
		fnResult.from(path, type);

		setTensionEdges(zResultDiagram);
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- graph specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setelementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---- mesh specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setelementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---- mesh specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createFormfromFile(string path, zFileTpye type)
	{
		fnForm.from(path, type);

		setTensionEdges(zFormDiagram);
		setelementColorDomain(zFormDiagram);


		setVertexWeights(zFormDiagram);
	}

	//---- graph specilization for createFormFromResult
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createFormFromResult()
	{
		*formObj = fnResult.getDuplicate();

		zVector* pos = fnForm.getRawVertexPositions();
		zColor* col = fnForm.getRawVertexColors();

		zColor col1(1, 1, 1, 1);

		for (int i = 0; i < fnForm.numVertices(); i++)
		{

			pos[i].z = 0;
			col[i] = col1;
		}

		zColor col2(0, 0, 0, 1);

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			col[fixedVertices[i]] = col2;

		}

		fnForm.setEdgeColor(elementColorDomain.max);

		setVertexWeights(zFormDiagram);

		setFormTensionEdgesfromResult();
		setelementColorDomain(zFormDiagram);
	}

	//---- mesh specilization for createFormFromResult
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createFormFromResult()
	{
		std::clock_t start;
		start = std::clock();


		fnResult.getDuplicate(*formObj);

		zVector* pos = fnForm.getRawVertexPositions();
		zColor* col = fnForm.getRawVertexColors();

		zColor col1(1, 1, 1, 1);

		for (int i = 0; i < fnForm.numVertices(); i++)
		{

			pos[i].z = 0;
			col[i] = col1;
		}


		zColor col2(0, 0, 0, 1);

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			col[fixedVertices[i]] = col2;

		}

		fnForm.setEdgeColor(elementColorDomain.max);


		setVertexWeights(zFormDiagram);
		setFormTensionEdgesfromResult();
		setelementColorDomain(zFormDiagram);
	}

	//---- graph specilization for createFormFromForce
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createFormFromForce(bool excludeBoundary, bool PlanarForceMesh, bool rotate90)
	{
		fnForce.getDualGraph(*formObj, forceEdge_formEdge, formEdge_forceEdge, excludeBoundary, PlanarForceMesh, rotate90);

		fnForm.setEdgeColor(elementColorDomain.max);

		setVertexWeights(zFormDiagram);


		setFormTensionEdgesfromForce();
		setelementColorDomain(zFormDiagram);
	}

	//---- mesh specilization for createForceFromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createForceFromForm(bool rotate90)
	{

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;


		int n_v = fnForm.numVertices();
		int n_e = fnForm.numHalfEdges();
		int n_f = fnForm.numPolygons();


		vector<zVector> faceCenters;
		fnForm.getCenters(zFaceData, faceCenters);
		positions = faceCenters;

		vector<int> formHalfEdge_forceVertex;

		for (zItMeshHalfEdge he(*formObj); !he.end(); he++)
		{
			formHalfEdge_forceVertex.push_back(-1);

			if (!he.onBoundary())
			{
				formHalfEdge_forceVertex[he.getId()] = he.getFace().getId();
			}
		}

		for (zItMeshHalfEdge he(*formObj); !he.end(); he++)
		{
			int i = he.getId();
			if (formHalfEdge_forceVertex[i] == -1)
			{

				// check if both the  vertices of the boundary edge arent fixed

				bool edgeVertsFixed = false;

				
				vector<int> eVerts;
				he.getVertices(eVerts);

				int v1 = eVerts[0];
				int v2 = eVerts[1];


				if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2])
				{
					edgeVertsFixed = true;

				}

				// walk on boundary and find all vertices between two fixed vertices
				if (!edgeVertsFixed)
				{
					vector<int>  boundaryEdges;
					vector<int>  boundaryVertices;

					zItMeshHalfEdge start = he;
					zItMeshHalfEdge e = he;

					bool exit = false;


					// walk prev
					do
					{

						if (fixedVerticesBoolean[e.getSym().getVertex().getId()])
						{

							exit = true;
							boundaryVertices.push_back(e.getSym().getVertex().getId());
						}

						boundaryEdges.push_back(e.getId());
						boundaryVertices.push_back(e.getVertex().getId());

						if (e.getPrev().isActive())
						{
							e = e.getPrev();
						}
						else exit = true;

					} while (e != start && !exit);


					// walk next 
					// checking if the prev walk as completed the full edge loop
					if (e != start)
					{
						bool exit = false;
						e = start;
						do
						{
							if (fixedVerticesBoolean[e.getVertex().getId()])
							{
								exit = true;

							}

							if (exit) continue;

							if (e.getNext().isActive())
							{
								e = e.getNext();
								boundaryVertices.push_back(e.getVertex().getId());
								boundaryEdges.push_back(e.getId());

								if (fixedVerticesBoolean[e.getVertex().getId()])
								{
									exit = true;


								}

							}
							else exit = true;


						} while (e != start && !exit);
					}


					if (boundaryEdges.size() > 1)
					{

						int vertId = positions.size();
						zVector newPos;
						for (int j = 0; j < boundaryEdges.size(); j++)
						{
							formHalfEdge_forceVertex[boundaryEdges[j]] = vertId;
						}



						for (int j = 0; j < boundaryVertices.size(); j++)
						{
							zItMeshVertex v(*formObj, boundaryVertices[j]);
							zVector pos = v.getPosition();
							newPos += pos;
						}



						newPos /= boundaryVertices.size();
						positions.push_back(newPos);
					}

				}


			}
		}

		for (zItMeshVertex v(*formObj); !v.end(); v++)
		{
			int i = v.getId();

			if (!fixedVerticesBoolean[i])
			{
				vector<int> cEdges;
				v.getConnectedHalfEdges(cEdges);

				if (cEdges.size() > 2)
				{
					for (int j = 0; j < cEdges.size(); j++)
					{
						int vertId = formHalfEdge_forceVertex[cEdges[j]];

						polyConnects.push_back(vertId);
					}

					polyCounts.push_back(cEdges.size());
				}

			}


		}

		forceObj->mesh.create(positions, polyCounts, polyConnects);

		//printf("\n forceMesh: %i %i %i", out.numVertices(), out.numEdges(), out.numPolygons());

		if (rotate90)
		{
			// bounding box
			zVector minBB, maxBB;
			vector<zVector> vertPositions;
			fnForce.getVertexPositions(vertPositions);

			coreUtils.getBounds(vertPositions, minBB, maxBB);

			zVector cen = (maxBB + minBB) * 0.5;

			zVector rotAxis(0, 0, 1);

			for (int i = 0; i < vertPositions.size(); i++)
			{
				vertPositions[i] -= cen;
				vertPositions[i] = vertPositions[i].rotateAboutAxis(rotAxis, -90);
			}


			fnForce.setVertexPositions(vertPositions);

			//printf("\n working!");
		}


		// compute forceEdge_formEdge
		forceEdge_formEdge.clear();
		for (int i = 0; i < fnForce.numHalfEdges(); i++)
		{
			forceEdge_formEdge.push_back(-1);
		}

		// compute form edge to force edge	
		formEdge_forceEdge.clear();

		for (zItMeshHalfEdge e_form(*formObj); !e_form.end(); e_form++)
		{
			int i = e_form.getId();

			int v1 = formHalfEdge_forceVertex[i];
			int v2 = (i % 2 == 0) ? formHalfEdge_forceVertex[i + 1] : formHalfEdge_forceVertex[i - 1];

			zItMeshHalfEdge e_force;
			bool chk = fnForce.halfEdgeExists(v1, v2, e_force);

			int eId = -1;

			if (chk)
			{
				zVector e_Form_vec = e_form.getVector();
				e_Form_vec.normalize();

				zVector e_Force_vec = e_force.getVector();
				e_Force_vec.normalize();

				eId = e_force.getId();

				// for tension edge point to the edge in the opposite direction
				//int symId = fnForm.getSymIndex(i);
				//if (fnForm.onBoundary(i, zHalfEdgeData) || fnForm.onBoundary(symId, zHalfEdgeData))
				//{
				//if (form_tensionEdges[i])
				//{
				//	if (e_Form_vec*e_Force_vec > 0) eId = e_force.getSym().getId();
				//}
				////for compression edge point to the edge in the same direction
				//else
				//{
				//	if (e_Form_vec*e_Force_vec < 0) eId = e_force.getSym().getId();
				//}
				//}			



			}

			formEdge_forceEdge.push_back(eId);

			if (formEdge_forceEdge[i] != -1)
			{
				forceEdge_formEdge[formEdge_forceEdge[i]] = i;

			}
		}




		setVertexWeights(zForceDiagram);

		setForceTensionEdgesfromForm();
		setelementColorDomain(zForceDiagram);
	}

	//---- graph specilization for createResultFromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createResultFromForm()
	{
		*resultObj = fnForm.getDuplicate();

		setResultTensionEdgesfromForm();
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createResultFromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createResultFromForm()
	{
		fnForm.getDuplicate(*resultObj);

		setResultTensionEdgesfromForm();
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- FDM METHODS

	//---- graph specilization for ForceDensityMethod
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjGraph, zFnGraph>::forceDensityMethod()
	{
		zHEData type = zVertexData;
		bool positiveDensities = true;

		//int n_e = fnResult.numHalfEdges();
		int n_v = fnResult.numVertices();

		int numEdges = fnResult.numEdges();;

		for (zItGraphEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd X(fnResult.numVertices(), 3);


		for (zItGraphVertex v(*resultObj); !v.end(); v++)
		{
			zVector pos = v.getPosition();
			int i = v.getId();

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};

		// EDGE NODE MATRIX
		zSparseMatrix C = getEdgeNodeMatrix(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (zItGraphEdge e(*resultObj); !e.end(); e++)
		{

			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				int i = e.getId();
				q[FD_EdgesCounter] = forceDensities[i] ;

				if (result_tensionEdges[e.getHalfEdge(0).getId()])
				{
					positiveDensities = false;

					q[FD_EdgesCounter] *= -1;
				}
				FD_EdgesCounter++;
			}

		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		zDiagonalMatrix Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < resultVMass.size(); i++)
		{
			p[i] = resultVMass[i] * resultVThickness[i] * resultVWeights[i];
		}


		MatrixXd P(fnResult.numVertices(), 3);
		P.setConstant(0.0);
		P.col(2) = p.col(0);

		// SUB MATRICES
		vector<int> freeVertices;

		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVertices.push_back(j);
				//printf("\n j: %i ", j);
			}

		}

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

		//cout << "Cn: \n" << Cn << endl;
		//cout << "Cf: \n" << Cf << endl;

		zSparseMatrix Cn_transpose;
		Cn_transpose = Cn.transpose();

		//CHOLESKY DECOMPOSITION

		zSparseMatrix Dn = Cn_transpose * Q * Cn;
		zSparseMatrix Df = Cn_transpose * Q * Cf;

		MatrixXd  B = Pn - Df * Xf;

		// solve
		MatrixXd Xn;

		if (positiveDensities)
		{
			SimplicialLLT< zSparseMatrix > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
				return false;


			Xn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
				return false;

			//cout << endl << Xn;
		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Xn = denseDn.ldlt().solve(B);

			// convergence error check.
			double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			cout << endl << relative_error << " FDM - negative" << endl;

		}


		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			zItGraphVertex v(*resultObj, id);

			zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
			v.setPosition(pos);
		}

		return true;
	}

	//---- mesh specilization for ForceDensityMethod
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjMesh, zFnMesh>::forceDensityMethod()
	{
		zHEData type = zVertexData;
		bool positiveDensities = true;

		//int n_e = fnResult.numHalfEdges();
		int n_v = fnResult.numVertices();

		int numEdges = fnResult.numEdges();;

		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd X(fnResult.numVertices(), 3);


		for (zItMeshVertex v(*resultObj); !v.end(); v++)
		{
			zVector pos = v.getPosition();
			int i = v.getId();

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};

		// EDGE NODE MATRIX
		zSparseMatrix C = getEdgeNodeMatrix(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{

			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				int i = e.getId();
				q[FD_EdgesCounter] = forceDensities[i];

				if (result_tensionEdges[e.getHalfEdge(0).getId()])
				{
					positiveDensities = false;
					q[FD_EdgesCounter] *= -1;
				}
				FD_EdgesCounter++;
			}

		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		zDiagonalMatrix Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < resultVMass.size(); i++)
		{
			p[i] = resultVMass[i] * resultVThickness[i] * resultVWeights[i];
		}


		MatrixXd P(fnResult.numVertices(), 3);
		P.setConstant(0.0);
		P.col(2) = p.col(0);

		// SUB MATRICES
		vector<int> freeVertices;

		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVertices.push_back(j);
				//printf("\n j: %i ", j);
			}

		}

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

		//cout << "Cn: \n" << Cn << endl;
		//cout << "Cf: \n" << Cf << endl;

		zSparseMatrix Cn_transpose;
		Cn_transpose = Cn.transpose();

		//CHOLESKY DECOMPOSITION

		zSparseMatrix Dn = Cn_transpose * Q * Cn;
		zSparseMatrix Df = Cn_transpose * Q * Cf;

		MatrixXd  B = Pn - Df * Xf;

		// solve
		MatrixXd Xn;

		if (positiveDensities)
		{
			SimplicialLLT< zSparseMatrix > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
				return false;


			Xn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
				return false;

			//cout << endl << Xn;
		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Xn = denseDn.ldlt().solve(B);

			// convergence error check.
			double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			cout << endl << relative_error << " FDM - negative" << endl;

		}


		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			zItMeshVertex v(*resultObj, id);

			zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
			v.setPosition(pos);
		}

		return true;
	}

	//---- TNA METHODS

	template<typename T, typename U>
	ZSPACE_INLINE bool zTsVault<T, U>::equilibriumHorizontal(bool &computeTargets, float formWeight, float dT, zIntergrationType type, int numIterations , float angleTolerance, float minMax_formEdge , float minMax_forceEdge , bool colorEdges, bool printInfo )
	{
		// compute horizontal equilibrium targets
		if (computeTargets)
		{
			getHorizontalTargets(formWeight);

			if (formVWeights.size() == 0) setVertexWeights(zDiagramType::zFormDiagram);
			if (forceVWeights.size() == 0) setVertexWeights(zDiagramType::zForceDiagram);
			
			computeTargets = !computeTargets;
		}

		// update diagrams

		if (formWeight != 1.0)
		{
			updateFormDiagram(minMax_formEdge, dT, type, numIterations);
		}

		if (formWeight != 0.0)
		{
			updateForceDiagram(minMax_forceEdge, dT, type, numIterations);
		}

		// check deviations
		zDomainFloat dev;		
		bool out = checkHorizontalParallelity(dev, angleTolerance, colorEdges, printInfo);

		if (out)
		{

			setelementColorDomain(zFormDiagram);
			setelementColorDomain(zForceDiagram);
		}

		return out;
	}

	//---- graph specilization for verticalEquilibrium using Linear Algebra
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjGraph, zFnGraph>::equilibriumVertical(bool &computeForceDensitities, float forceDiagramScale)
	{
		if (computeForceDensitities)
		{
			setForceDensitiesFromDiagrams(forceDiagramScale);

			computeForceDensitities = false;
		}

		zHEData type = zVertexData;

		bool positiveDensities = true;

		//int n_e = fnResult.numHalfEdges();
		int n_v = fnResult.numVertices();

		int numEdges = fnResult.numEdges();

		for (zItGraphEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd Xz(fnResult.numVertices(), 1);
		zVector* pos = fnResult.getRawVertexPositions();

		zVector* posForm = fnForm.getRawVertexPositions();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			Xz(i, 0) = pos[i].z;

			// set x and y to form diagram
			pos[i].x = posForm[i].x;
			pos[i].y = posForm[i].y;

		};

		// EDGE NODE MATRIX
		zSparseMatrix C = getEdgeNodeMatrix(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (zItGraphEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				int i = e.getId();
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}
		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		zDiagonalMatrix Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			p[i] = (resultVMass[i] * resultVThickness[i] * resultVWeights[i] * forceDiagramScale);
		}


		MatrixXd P(fnResult.numVertices(), 1);
		P.col(0) = p.col(0);

		// SUB MATRICES
		vector<int> freeVertices;

		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVertices.push_back(j);
				//printf("\n j: %i ", j);
			}

		}

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(Xz, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

		//cout << "Cn: \n" << Cn << endl;
		//cout << "Cf: \n" << Cf << endl;

		zSparseMatrix Cn_transpose;
		Cn_transpose = Cn.transpose();

		//CHOLESKY DECOMPOSITION

		zSparseMatrix Dn = Cn_transpose * Q * Cn;
		zSparseMatrix Df = Cn_transpose * Q * Cf;

		MatrixXd  B = Pn - Df * Xf;

		// solve
		MatrixXd Xn;

		if (positiveDensities)
		{
			SimplicialLLT< zSparseMatrix > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
				return false;


			Xn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
				return false;

			//cout << endl << Xn;
		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Xn = denseDn.ldlt().solve(B);

			// convergence error check.
			double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			cout << endl << relative_error << " FDM - negative" << endl;

		}


		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			pos[id].z = Xn(i, 0);

		}

		return true;
	}

	//---- mesh specilization for verticalEquilibrium using Linear Algebra
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjMesh, zFnMesh>::equilibriumVertical(bool &computeForceDensitities, float forceDiagramScale)
	{
		if (computeForceDensitities)
		{
			setForceDensitiesFromDiagrams(forceDiagramScale);

			computeForceDensitities = false;
		}		

		zHEData type = zVertexData;

		bool positiveDensities = true;

		//int n_e = fnResult.numHalfEdges();
		int n_v = fnResult.numVertices();

		int numEdges = fnResult.numEdges();

		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd Xz(fnResult.numVertices(), 1);
		zVector* pos = fnResult.getRawVertexPositions();

		zVector* posForm = fnForm.getRawVertexPositions();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			Xz(i, 0) = pos[i].z;

			// set x and y to form diagram
			pos[i].x = posForm[i].x;
			pos[i].y = posForm[i].y;

		};

		// EDGE NODE MATRIX
		zSparseMatrix C = getEdgeNodeMatrix(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				int i = e.getId();
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}
		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		zDiagonalMatrix Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			p[i] = (resultVMass[i] * resultVThickness[i] * resultVWeights[i] * forceDiagramScale);
		}


		MatrixXd P(fnResult.numVertices(), 1);
		P.col(0) = p.col(0);

		// SUB MATRICES
		vector<int> freeVertices;

		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVertices.push_back(j);
				//printf("\n j: %i ", j);
			}

		}

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(Xz, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

		//cout << "Cn: \n" << Cn << endl;
		//cout << "Cf: \n" << Cf << endl;

		zSparseMatrix Cn_transpose;
		Cn_transpose = Cn.transpose();

		//CHOLESKY DECOMPOSITION

		zSparseMatrix Dn = Cn_transpose * Q * Cn;
		zSparseMatrix Df = Cn_transpose * Q * Cf;

		MatrixXd  B = Pn - Df * Xf;

		// solve
		MatrixXd Xn;

		if (positiveDensities)
		{
			SimplicialLLT< zSparseMatrix > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
				return false;


			Xn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
				return false;

			//cout << endl << Xn;
		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Xn = denseDn.ldlt().solve(B);

			// convergence error check.
			double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			cout << endl << relative_error << " FDM - negative" << endl;

		}


		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			pos[id].z = Xn(i, 0);

		}

		return true;
	}

	//--- SET METHODS 

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::setElementColorDomain(zDomainColor &colDomain)
	{
		elementColorDomain = colDomain;
	}

	//---- graph specilization for setFixedVertices
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setConstraints(zDiagramType type, const vector<int>& _fixedVertices)
	{
		if (type == zResultDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (zItGraphVertex v(*resultObj); !v.end(); v++)
				{
					if (v.checkValency(1))
					{
						fixedVertices.push_back(v.getId());
					}
				}
			}
			else
			{
				fixedVertices = _fixedVertices;
			}

			fixedVerticesBoolean.clear();

			for (int i = 0; i < fnResult.numVertices(); i++)
			{
				fixedVerticesBoolean.push_back(false);
			}

			for (int i = 0; i < fixedVertices.size(); i++)
			{
				zItGraphVertex v(*resultObj, fixedVertices[i]);
				zColor col;
				v.setColor(col);
				fixedVerticesBoolean[fixedVertices[i]] = true;
			}

		}

		else if (type == zFormDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (zItGraphVertex v(*formObj); !v.end(); v++)
				{
					if (v.checkValency(1)) fixedVertices.push_back(v.getId());
				}

			}
			else
			{
				fixedVertices = _fixedVertices;
			}

			fixedVerticesBoolean.clear();


			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				fixedVerticesBoolean.push_back(false);
			}



			for (int i = 0; i < fixedVertices.size(); i++)
			{
				zItGraphVertex v(*formObj, fixedVertices[i]);
				zColor col;
				v.setColor(col);

				fixedVerticesBoolean[fixedVertices[i]] = true;
			}
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	//---- mesh specilization for setFixedVertices
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setConstraints(zDiagramType type, const vector<int>& _fixedVertices)
	{


		if (type == zResultDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (zItMeshVertex v(*resultObj); !v.end(); v++)
				{
					if (v.onBoundary())
					{
						fixedVertices.push_back(v.getId());
					}
				}
			}
			else
			{
				fixedVertices = _fixedVertices;
			}

			fixedVerticesBoolean.clear();

			for (int i = 0; i < fnResult.numVertices(); i++)
			{
				fixedVerticesBoolean.push_back(false);
			}

			fnResult.setVertexColor(zColor(1, 1, 1, 1));

			for (int i = 0; i < fixedVertices.size(); i++)
			{
				zItMeshVertex v(*resultObj, fixedVertices[i]);
				zColor col;
				v.setColor(col);
				fixedVerticesBoolean[fixedVertices[i]] = true;
			}

			printf("\n fixed: %i ", fixedVertices.size());

		}

		else if (type == zFormDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (zItMeshVertex v(*formObj); !v.end(); v++)
				{
					if (v.onBoundary()) fixedVertices.push_back(v.getId());
				}

			}
			else
			{
				fixedVertices = _fixedVertices;
			}

			fixedVerticesBoolean.clear();


			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				fixedVerticesBoolean.push_back(false);
			}


			fnForm.setVertexColor(zColor(1, 1, 1, 1));

			for (int i = 0; i < fixedVertices.size(); i++)
			{
				zItMeshVertex v(*formObj, fixedVertices[i]);
				zColor col;
				v.setColor(col);

				fixedVerticesBoolean[fixedVertices[i]] = true;
			}

			printf("\n fixed: %i ", fixedVertices.size());
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::appendConstraints(vector<int> &_constraintVertices)
	{
		for (int i = 0; i < _constraintVertices.size(); i++)
		{
			int id;
			bool checkRepeat = coreUtils.checkRepeatElement(_constraintVertices[i], fixedVertices, id);
			if (!checkRepeat) fixedVertices.push_back(_constraintVertices[i]);
		}

		for (int i = 0; i < _constraintVertices.size(); i++)
		{
			if (_constraintVertices[i] >= 0 && _constraintVertices[i] < fixedVerticesBoolean.size()) fixedVerticesBoolean[_constraintVertices[i]] = true;
		}
	}

	//---- graph specilization for setForceDensity
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setForceDensity(float fDensity)
	{
		forceDensities.clear();

		for (int i = 0; i < fnResult.numEdges(); i++)
		{
			forceDensities.push_back(fDensity);
		}
	}

	//---- mesh specilization for setForceDensity
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setForceDensity(float fDensity)
	{
		forceDensities.clear();

		for (int i = 0; i < fnResult.numEdges(); i++)
		{
			forceDensities.push_back(fDensity);
		}
	}

	//---- graph specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setForceDensities(vector<float> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of graph half edges.");

		forceDensities = fDensities;
	}

	//---- mesh specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setForceDensities(vector<float> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of mesh edges.");

		forceDensities = fDensities;
	}

	//---- graph specilization for setForceDensitiesFromDiagrams
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setForceDensitiesFromDiagrams(float forceDiagramScale, bool negate)
	{
		forceDensities.clear();

		for (zItGraphEdge e(*formObj); !e.end(); e++)
		{
			// form edge

			zItGraphHalfEdge e_form = e.getHalfEdge(0);
			int eId_form = e_form.getId();;
			zVector e_form_vec = e_form.getVector();
			double e_form_len = e_form_vec.length();

			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge

				int eId_force = formEdge_forceEdge[eId_form];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				zVector e_force_vec = e_force.getVector();
				double e_force_len = e_force_vec.length();

				double forceDensity = ((e_force_len / e_form_len));

				e_form_vec.normalize();
				e_force_vec.normalize();
				forceDensity *= (e_form_vec * e_force_vec);


				forceDensities.push_back(forceDensity / forceDiagramScale);


				if (negate)
				{
					forceDensities[forceDensities.size() - 1] *= -1;
				}

			}

			else
			{
				forceDensities.push_back(0);

			}
		}


	}

	//---- mesh specilization for setForceDensitiesFromDiagrams
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setForceDensitiesFromDiagrams(float forceDiagramScale, bool negate)
	{
		forceDensities.clear();

		for (zItMeshEdge e(*formObj); !e.end(); e++)
		{
			// form edge

			zItMeshHalfEdge e_form = e.getHalfEdge(0);
			int eId_form = e_form.getId();;
			zVector e_form_vec = e_form.getVector();
			double e_form_len = e_form_vec.length();

			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge

				int eId_force = formEdge_forceEdge[eId_form];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				zVector e_force_vec = e_force.getVector();
				double e_force_len = e_force_vec.length();

				double forceDensity = ((e_force_len / e_form_len));

				e_form_vec.normalize();
				e_force_vec.normalize();
				forceDensity *= (e_form_vec * e_force_vec);


				forceDensities.push_back(forceDensity / forceDiagramScale);


				if (negate)
				{
					forceDensities[forceDensities.size() - 1] *= -1;
				}

			}

			else
			{
				forceDensities.push_back(0);

			}
		}


	}

	//---- graph specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setTensionEdges(zDiagramType type, const vector<int>& _tensionEdges)
	{

		if (type == zResultDiagram)
		{
			result_tensionEdges.clear();

			for (int i = 0; i < fnResult.numHalfEdges(); i++) result_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnResult.numHalfEdges())
				{
					result_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zFormDiagram)
		{
			form_tensionEdges.clear();

			for (int i = 0; i < fnForm.numHalfEdges(); i++) form_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForm.numHalfEdges())
				{
					form_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zForceDiagram)
		{
			force_tensionEdges.clear();

			for (int i = 0; i < fnForce.numHalfEdges(); i++) force_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForce.numHalfEdges())
				{
					force_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else throw std::invalid_argument(" invalid diagram type.");
	}

	//---- mesh specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setTensionEdges(zDiagramType type, const vector<int>& _tensionEdges)
	{

		if (type == zResultDiagram)
		{
			result_tensionEdges.clear();

			for (int i = 0; i < fnResult.numHalfEdges(); i++) result_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnResult.numHalfEdges())
				{
					result_tensionEdges[_tensionEdges[i]] = true;

				}
			}

			setelementColorDomain(type);
		}

		else if (type == zFormDiagram)
		{
			form_tensionEdges.clear();

			for (int i = 0; i < fnForm.numHalfEdges(); i++) form_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForm.numHalfEdges())
				{
					form_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zForceDiagram)
		{
			force_tensionEdges.clear();

			for (int i = 0; i < fnForce.numHalfEdges(); i++) force_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForce.numHalfEdges())
				{
					force_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::appendTensionEdges(zDiagramType type, vector<int> &_tensionEdges)
	{
		if (type == zFormDiagram)
		{
			for (int i = 0; i < _tensionEdges.size(); i++)
			{

				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < form_tensionEdges.size()) form_tensionEdges[_tensionEdges[i]] = true;
			}
		}
		else if (type == zResultDiagram)
		{
			for (int i = 0; i < _tensionEdges.size(); i++)
			{

				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < result_tensionEdges.size())result_tensionEdges[_tensionEdges[i]] = true;
			}
		}

		else if (type == zForceDiagram)
		{
			for (int i = 0; i < _tensionEdges.size(); i++)
			{

				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < force_tensionEdges.size())force_tensionEdges[_tensionEdges[i]] = true;
			}
		}

		else throw std::invalid_argument(" invalid diagram type.");
	}

	//---- graph specilization for setForceTensionEdgesfromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setForceTensionEdgesfromForm()
	{
		force_tensionEdges.clear();

		for (int i = 0; i < fnForce.numHalfEdges(); i++)
		{
			force_tensionEdges.push_back(false);
		}

		for (zItGraphHalfEdge he(*formObj); !he.end(); he++)
		{
			int i = he.getId();
			if (formEdge_forceEdge[i] != -1 && form_tensionEdges[i])
			{
				force_tensionEdges[formEdge_forceEdge[i]] = true;;

				zItMeshHalfEdge e_force(*forceObj, formEdge_forceEdge[i]);

				int symEdge = e_force.getSym().getId();
				force_tensionEdges[symEdge] = true;;

			}
		}

		setelementColorDomain(zForceDiagram);
	}

	//---- mesh specilization for setForceTensionEdgesfromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setForceTensionEdgesfromForm()
	{
		force_tensionEdges.clear();

		for (int i = 0; i < fnForce.numHalfEdges(); i++)
		{
			force_tensionEdges.push_back(false);
		}

		for (zItMeshHalfEdge he(*formObj); !he.end(); he++)
		{
			int i = he.getId();
			if (formEdge_forceEdge[i] != -1 && form_tensionEdges[i])
			{
				force_tensionEdges[formEdge_forceEdge[i]] = true;;

				zItMeshHalfEdge e_force(*forceObj, formEdge_forceEdge[i]);

				int symEdge = e_force.getSym().getId();
				force_tensionEdges[symEdge] = true;;

			}
		}

		setelementColorDomain(zForceDiagram);
	}

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::setResultTensionEdgesfromForm()
	{
		result_tensionEdges.clear();
		result_tensionEdges = form_tensionEdges;


		setelementColorDomain(zResultDiagram);
	}

	template<typename T, typename U>
	ZSPACE_INLINE void  zTsVault<T, U>::setFormTensionEdgesfromResult()
	{
		form_tensionEdges.clear();
		form_tensionEdges = result_tensionEdges;

		setelementColorDomain(zFormDiagram);

	}

	//---- graph specilization for setFormTensionEdgesfromForce
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setFormTensionEdgesfromForce()
	{
		form_tensionEdges.clear();

		for (int i = 0; i < fnForm.numHalfEdges(); i++)
		{
			form_tensionEdges.push_back(false);
		}

		for (zItMeshHalfEdge he(*forceObj); !he.end(); he++)
		{
			int i = he.getId();
			if (forceEdge_formEdge[i] != -1 && force_tensionEdges[i])
			{
				form_tensionEdges[forceEdge_formEdge[i]] = true;;

				zItGraphHalfEdge e_form(*formObj, forceEdge_formEdge[i]);

				int symEdge = e_form.getSym().getId();
				form_tensionEdges[symEdge] = true;;

			}
		}

		setelementColorDomain(zFormDiagram);

	}

	//---- mesh specilization for setFormTensionEdgesfromForce
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setFormTensionEdgesfromForce()
	{
		form_tensionEdges.clear();

		for (int i = 0; i < fnForm.numHalfEdges(); i++)
		{
			form_tensionEdges.push_back(false);
		}

		for (zItMeshHalfEdge he(*forceObj); !he.end(); he++)
		{
			int i = he.getId();
			if (forceEdge_formEdge[i] != -1 && force_tensionEdges[i])
			{
				form_tensionEdges[forceEdge_formEdge[i]] = true;;

				zItMeshHalfEdge e_form(*formObj, forceEdge_formEdge[i]);

				int symEdge = e_form.getSym().getId();
				form_tensionEdges[symEdge] = true;;

			}
		}

		setelementColorDomain(zFormDiagram);

	}

	//---- graph specilization for setVertexWeights
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setelementColorDomain(zDiagramType type)
	{
		if (type == zFormDiagram)
		{
			for (zItGraphHalfEdge he(*formObj); !he.end(); he++)
			{
				int i = he.getId();
				if (form_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else  he.getEdge().setColor(elementColorDomain.max);
			}
		}
		else if (type == zResultDiagram)
		{
			for (zItGraphHalfEdge he(*resultObj); !he.end(); he++)
			{
				int i = he.getId();
				if (result_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else he.getEdge().setColor(elementColorDomain.max);
			}
		}
		else if (type == zForceDiagram)
		{
			for (zItMeshHalfEdge he(*forceObj); !he.end(); he++)
			{
				int i = he.getId();
				if (force_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else he.getEdge().setColor(elementColorDomain.max);
			}


		}
		else throw std::invalid_argument(" invalid diagram type.");
	}

	//---- mesh specilization for setVertexWeights	
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setelementColorDomain(zDiagramType type)
	{
		if (type == zFormDiagram)
		{
			for (zItMeshHalfEdge he(*formObj); !he.end(); he++)
			{
				int i = he.getId();
				if (form_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else  he.getEdge().setColor(elementColorDomain.max);
			}
		}
		else if (type == zResultDiagram)
		{
			for (zItMeshHalfEdge he(*resultObj); !he.end(); he++)
			{
				int i = he.getId();
				if (result_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else he.getEdge().setColor(elementColorDomain.max);
			}
		}
		else if (type == zForceDiagram)
		{
			for (zItMeshHalfEdge he(*forceObj); !he.end(); he++)
			{
				int i = he.getId();
				if (force_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else he.getEdge().setColor(elementColorDomain.max);
			}


		}
		else throw std::invalid_argument(" invalid diagram type.");
	}

	//---- graph specilization for setVertexWeights
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setVertexWeights(zDiagramType type, const vector<float>& vWeights)
	{
		if (type == zFormDiagram)
		{
			if (vWeights.size() == 0)
			{
				for (int i = 0; i < fnForm.numVertices(); i++) formVWeights.push_back(1.0);
			}
			else
			{
				if (vWeights.size() != fnForm.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of form vertices.");

				formVWeights = vWeights;
			}

		}

		else if (type == zForceDiagram)
		{
			if (vWeights.size() == 0)
			{
				for (int i = 0; i < fnForce.numVertices(); i++) forceVWeights.push_back(1.0);
			}
			else
			{
				if (vWeights.size() != fnForce.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of force vertices.");

				forceVWeights = vWeights;
			}

		}

		else if (type == zResultDiagram)
		{
			if (vWeights.size() == 0)
			{
				for (int i = 0; i < fnResult.numVertices(); i++) resultVWeights.push_back(1.0);
			}
			else
			{
				if (vWeights.size() != fnResult.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of result vertices.");

				resultVWeights = vWeights;
			}

		}

		else throw std::invalid_argument(" error: invalid zDiagramType type");
	}

	//---- mesh specilization for setVertexWeights
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexWeights(zDiagramType type, const vector<float>& vWeights)
	{
		if (type == zFormDiagram)
		{
			if (vWeights.size() == 0)
			{
				formVWeights.clear();
				formVWeights.assign(fnForm.numVertices(), 1.0);
			}
			else
			{
				if (vWeights.size() != fnForm.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of form vertices.");

				formVWeights = vWeights;
			}

		}

		else if (type == zForceDiagram)
		{
			if (vWeights.size() == 0)
			{
				forceVWeights.clear();
				forceVWeights.assign(fnForce.numVertices(), 1.0);

			}
			else
			{
				if (vWeights.size() != fnForce.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of force vertices.");

				forceVWeights = vWeights;
			}

		}

		else if (type == zResultDiagram)
		{
			if (vWeights.size() == 0)
			{

				resultVWeights.clear();
				resultVWeights.assign(fnResult.numVertices(), 1.0);

			}
			else
			{
				if (vWeights.size() != fnResult.numVertices()) throw std::invalid_argument("size of loads contatiner is not equal to number of result vertices.");

				resultVWeights = vWeights;
			}

		}

		else throw std::invalid_argument(" error: invalid zDiagramType type");
	}

	template<typename T, typename U>
	void  zTsVault<T, U>::setVertexWeightsfromConstraints(zDiagramType type)
	{
		if (fixedVertices.size() == 0)
		{
			setConstraints(type);
		}

		if (type == zResultDiagram)
		{
			resultVWeights.clear();
			resultVWeights.assign(fixedVerticesBoolean.size(), 1.0);

			for (int i = 0; i < fixedVerticesBoolean.size(); i++)
			{
				if (fixedVerticesBoolean[i]) resultVWeights[i] = (0.0);
			}
		}

		if (type == zFormDiagram)
		{
			formVWeights.clear();
			formVWeights.assign(fixedVerticesBoolean.size(), 1.0);

			for (int i = 0; i < fixedVerticesBoolean.size(); i++)
			{
				if (fixedVerticesBoolean[i]) formVWeights[i] = (0.0);

			}
		}

	}

	//---- graph specilization for setVertexThickness
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setVertexThickness(float thickness)
	{
		resultVThickness.clear();

		resultVThickness.assign(fnResult.numVertices(), thickness);


	}

	//---- mesh specilization for setVertexThickness
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexThickness(float thickness)
	{
		resultVThickness.clear();
		resultVThickness.assign(fnResult.numVertices(), thickness);


	}

	//---- graph specilization for setVertexThickness
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setVertexThickness(vector<float> &thickness)
	{
		if (thickness.size() != fnResult.numVertices()) throw std::invalid_argument("size of thickness contatiner is not equal to number of mesh vertices.");

		resultVThickness = thickness;

	}

	//---- mesh specilization for setVertexThickness
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexThickness(vector<float> &thickness)
	{
		if (thickness.size() != fnResult.numVertices()) throw std::invalid_argument("size of thickness contatiner is not equal to number of mesh vertices.");

		resultVThickness = thickness;

	}

	//---- graph specilization for setVertexMass
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setVertexMass(float mass)
	{
		resultVMass.clear();

		resultVMass.assign(fnResult.numVertices(), mass);


	}

	//---- mesh specilization for setVertexMass
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexMass(float mass)
	{
		resultVMass.clear();
		resultVMass.assign(fnResult.numVertices(), mass);
	}

	//---- graph specilization for setVertexMass
	template<>
	ZSPACE_INLINE void zTsVault<zObjGraph, zFnGraph>::setVertexMass(vector<float> &mass)
	{
		if (mass.size() != fnResult.numVertices()) throw std::invalid_argument("size of mass contatiner is not equal to number of mesh vertices.");

		resultVMass = mass;
	}

	//---- mesh specilization for setVertexMass
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexMass(vector<float> &mass)
	{
		if (mass.size() != fnResult.numVertices()) throw std::invalid_argument("size of mass contatiner is not equal to number of mesh vertices.");

		resultVMass = mass;
	}

	//---- mesh specilization for setVertexMass
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setVertexMassfromVertexArea()
	{
		resultVMass.clear();

		vector<double> vAreas;

		vector<zVector> fCenters;
		fnResult.getCenters(zFaceData, fCenters);

		vector<zVector> eCenters;
		fnResult.getCenters(zHalfEdgeData, eCenters);

		fnResult.getVertexAreas(fCenters, eCenters, resultVMass);

	}

	//--- GET METHODS 

	template<typename T, typename U>
	ZSPACE_INLINE bool zTsVault<T, U>::getCorrespondingForceEdge(int formEdgeindex, zItMeshHalfEdge &outForceEdge)
	{
		if (formEdgeindex > formEdge_forceEdge.size()) throw std::invalid_argument(" error: index out of bounds.");
		if (formEdge_forceEdge[formEdgeindex] != -1) outForceEdge = zItMeshHalfEdge(*forceObj, formEdge_forceEdge[formEdgeindex]);

		return (formEdge_forceEdge[formEdgeindex] == -1) ? false : true;
	}

	template<typename T, typename U>
	ZSPACE_INLINE bool zTsVault<T, U>::getCorrespondingFormEdge(int forceEdgeindex, zItMeshHalfEdge &outFormEdge)
	{
		if (forceEdgeindex > forceEdge_formEdge.size()) throw std::invalid_argument(" error: index out of bounds.");

		if (forceEdge_formEdge[forceEdgeindex] != -1) outFormEdge = zItMeshHalfEdge(*forceObj, forceEdge_formEdge[forceEdgeindex]);

		return (forceEdge_formEdge[forceEdgeindex] == -1) ? false : true;
	}

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::getHorizontalTargetEdges(zDiagramType type, vector<zVector> &targets)
	{
		if (type == zFormDiagram) targets = targetEdges_form;

		else if (type == zForceDiagram) targets = targetEdges_force;

		else throw std::invalid_argument(" invalid diagram type.");
	}

	//---- graph specilization for getForces_GradientDescent
	template<>
	ZSPACE_INLINE  void zTsVault < zObjGraph, zFnGraph>::getForces_GradientDescent(vector<zVector>& forces)
	{
		zVector* pos = fnResult.getRawVertexPositions();

		forces.clear();
		forces.assign(fnResult.numVertices(), zVector());

		for (zItGraphVertex v(*resultObj); !v.end(); v++)
		{
			int i = v.getId();
			if (fixedVerticesBoolean[i]) continue;

			vector<int> cVerts;
			v.getConnectedVertices(cVerts);


			// get lowest positions

			zVector lowPosition = v.getPosition();
			int lowId = i;
			for (int j = 0; j < cVerts.size(); j++)
			{
				if (pos[cVerts[j]].z < lowPosition.z)
				{
					lowPosition = pos[cVerts[j]];
					lowId = cVerts[j];
				}
			}

			if (lowId != i)
			{
				forces[i] = lowPosition - pos[i];
				forces[i].normalize();
			}


		}

	}

	//---- mesh specilization for getForces_GradientDescent
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::getForces_GradientDescent(vector<zVector>& forces)
	{
		zVector* pos = fnResult.getRawVertexPositions();

		forces.clear();
		forces.assign(fnResult.numVertices(), zVector());

		for (zItMeshVertex v(*resultObj); !v.end(); v++)
		{
			int i = v.getId();
			if (fixedVerticesBoolean[i]) continue;

			vector<zItMeshFace> cFaces;
			v.getConnectedFaces(cFaces);


			vector<int> positionIndicies;
			for (auto &f : cFaces)
			{
				vector<int> fVerts;
				f.getVertices(fVerts);

				for (int k = 0; k < fVerts.size(); k++) positionIndicies.push_back(fVerts[k]);
			}


			// get lowest positions

			zVector lowPosition = v.getPosition();
			int lowId = i;
			for (int j = 0; j < positionIndicies.size(); j++)
			{
				if (pos[positionIndicies[j]].z < lowPosition.z)
				{
					lowPosition = pos[positionIndicies[j]];
					lowId = positionIndicies[j];
				}
			}

			if (lowId != i)
			{
				forces[i] = lowPosition - pos[i];
				forces[i].normalize();

			}


		}

	}

	//---- UTILITY METHODS 

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::translateForceDiagram(float value)
	{
		// bounding box
		zVector minBB, maxBB;
		vector<zVector> vertPositions;
		fnForce.getVertexPositions(vertPositions);

		coreUtils.getBounds(vertPositions, minBB, maxBB);

		zVector dir;
		dir.x = (maxBB.x - minBB.x);


		zVector rotAxis(0, 0, 1);

		for (int i = 0; i < vertPositions.size(); i++)
		{

			vertPositions[i] += dir * value;
		}

		fnForce.setVertexPositions(vertPositions);
	}

	//---- graph specilization for getEdgeNodeMatrix 
	template<>
	ZSPACE_INLINE  zSparseMatrix zTsVault<zObjGraph, zFnGraph>::getEdgeNodeMatrix(int numCols)
	{
		int n_v = fnResult.numVertices();
		zSparseMatrix out(numCols, n_v);
		out.setZero();

		int coefsCounter = 0;

		vector<zTriplet> coefs; // 1 for from vertex and -1 for to vertex

		for (zItGraphEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				coefs.push_back(zTriplet(coefsCounter, v1, 1));
				coefs.push_back(zTriplet(coefsCounter, v2, -1));

				coefsCounter++;
			}

		}

		out.setFromTriplets(coefs.begin(), coefs.end());

		return out;

	}

	//---- mesh specilization for getEdgeNodeMatrix 
	template<>
	ZSPACE_INLINE zSparseMatrix zTsVault<zObjMesh, zFnMesh>::getEdgeNodeMatrix(int numCols)
	{
		int n_v = fnResult.numVertices();

		zSparseMatrix out(numCols, n_v);
		out.setZero();

		int coefsCounter = 0;

		vector<zTriplet> coefs; // 1 for from vertex and -1 for to vertex

		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{


			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				coefs.push_back(zTriplet(coefsCounter, v1, 1));
				coefs.push_back(zTriplet(coefsCounter, v2, -1));

				coefsCounter++;
			}

		}

		out.setFromTriplets(coefs.begin(), coefs.end());

		return out;

	}
	
	template<typename T, typename U>
	ZSPACE_INLINE zSparseMatrix  zTsVault<T, U>::subMatrix(zSparseMatrix &C, vector<int> &nodes)
	{
		zSparseMatrix C_sub(C.rows(), nodes.size());
		for (int i = 0; i < nodes.size(); i++)C_sub.col(i) = C.col(nodes[i]);
		return C_sub;
	}

	template<typename T, typename U>
	ZSPACE_INLINE MatrixXd zTsVault<T, U>::subMatrix(MatrixXd &X, vector<int> &nodes)
	{
		MatrixXd X_sub(nodes.size(), X.cols());
		for (int i = 0; i < nodes.size(); i++)X_sub.row(i) = X.row(nodes[i]);
		return X_sub;
	}

	//---- graph specilization for getHorizontalTargets
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::getHorizontalTargets(float formWeight)
	{
		targetEdges_form.clear();
		targetEdges_force.clear();

		targetEdges_force.assign(fnForce.numHalfEdges(), zVector());

		for (zItGraphHalfEdge e_form(*formObj); !e_form.end(); e_form++)
		{
			int i = e_form.getId();

			//form edge
			int eId_form = e_form.getId();
			zVector e_form_vec = e_form.getVector();
			e_form_vec.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				if (form_tensionEdges[eId_form])
				{
					eId_force = e_force.getSym().getId();
					e_force = e_force.getSym();
				}

				zVector e_force_vec = e_force.getVector();
				e_force_vec.normalize();

				// target edge 
				zVector e_target = (e_form_vec *formWeight) + (e_force_vec * (1 - formWeight));
				e_target.normalize();

				targetEdges_form.push_back(e_target);
				targetEdges_force[eId_force] = e_target;
			}

			else
			{
				// target edge 
				zVector e_target = (e_form_vec * 1);
				targetEdges_form.push_back(e_target);
			}


		}
	}

	//---- mesh specilization for getHorizontalTargets
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::getHorizontalTargets(float formWeight)
	{
		targetEdges_form.clear();
		targetEdges_force.clear();

		targetEdges_force.assign(fnForce.numHalfEdges(), zVector());

		for (zItMeshHalfEdge e_form(*formObj); !e_form.end(); e_form++)
		{
			int i = e_form.getId();

			//form edge
			int eId_form = e_form.getId();
			zVector e_form_vec = e_form.getVector();
			e_form_vec.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				if (form_tensionEdges[eId_form])
				{
					eId_force = e_force.getSym().getId();
					e_force = e_force.getSym();
				}

				zVector e_force_vec = e_force.getVector();
				e_force_vec.normalize();

				// target edge 
				zVector e_target = (e_form_vec *formWeight) + (e_force_vec * (1 - formWeight));
				e_target.normalize();

				targetEdges_form.push_back(e_target);
				targetEdges_force[eId_force] = e_target;


			}

			else
			{
				// target edge 
				zVector e_target = (e_form_vec * 1);
				targetEdges_form.push_back(e_target);
			}


		}
	}

	//---- graph specilization for checkHorizontalParallelity
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjGraph, zFnGraph>::checkHorizontalParallelity(zDomainFloat &deviation, float angleTolerance, bool colorEdges, bool printInfo)
	{
		bool out = true;
		vector<double> deviations;
		deviation.min = 10000;
		deviation.max = -10000;

		for (zItGraphHalfEdge e_form(*formObj); !e_form.end(); e_form++)
		{


			//form edge
			int eId_form = e_form.getId();;
			zVector e_form_vec = e_form.getVector();
			e_form_vec.normalize();


			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[eId_form];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				zVector e_force_vec = e_force.getVector();
				e_force_vec.normalize();

				// angle

				double a_i = e_form_vec.angle(e_force_vec);
				if ((form_tensionEdges[eId_form]))a_i = 180 - a_i;



				deviations.push_back(a_i);

				if (a_i > angleTolerance)
				{
					out = false;
				}


				//printf("\n %i %i %i %i ", v1_form, v2_form, v1_force, v2_force);
				//printf("\n e: %i angle :  %1.2f ", i, a_i);

				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;

			}
			else
			{
				deviations.push_back(-1);
			}
		}

		if (printInfo)
		{
			printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
		}

		if (colorEdges)
		{
			zDomainColor colDomain(zColor(180, 1, 1), zColor(0, 1, 1));

			for (zItGraphHalfEdge e_form(*formObj); !e_form.end(); e_form++)
			{
				int i = e_form.getId();
				if (deviations[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

					if (deviations[i] < angleTolerance) col = zColor();

					e_form.getEdge().setColor(col);

					int eId_force = formEdge_forceEdge[i];

					zItMeshHalfEdge eForce(*forceObj, eId_force);
					eForce.getEdge().setColor(col);

				}

			}

		}


		return out;
	}

	//---- mesh specilization for checkHorizontalParallelity
	template<>
	ZSPACE_INLINE  bool zTsVault<zObjMesh, zFnMesh>::checkHorizontalParallelity(zDomainFloat &deviation, float angleTolerance, bool colorEdges, bool printInfo)
	{
		bool out = true;
		vector<double> deviations;
		deviation.min = 10000;
		deviation.max = -10000;
		
		for (zItMeshHalfEdge e_form(*formObj); !e_form.end(); e_form++)
		{


			//form edge
			int eId_form = e_form.getId();;
			zVector e_form_vec = e_form.getVector();
			e_form_vec.normalize();


			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[eId_form];
				zItMeshHalfEdge e_force(*forceObj, eId_force);

				zVector e_force_vec = e_force.getVector();
				e_force_vec.normalize();

				// angle

				double a_i = e_form_vec.angle(e_force_vec);
				if ((form_tensionEdges[eId_form]))a_i = 180 - a_i;

				deviations.push_back(a_i);

				if (a_i > angleTolerance)
				{
					out = false;
				}


				//printf("\n %i %i %i %i ", v1_form, v2_form, v1_force, v2_force);
				//printf("\n e: %i angle :  %1.2f ", i, a_i);

				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;

			}
			else
			{
				deviations.push_back(-1);
			}
		}
		

		if (printInfo)
		{			
			printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
		}

		if (colorEdges)
		{
			zDomainColor colDomain(zColor(180, 1, 1), zColor(0, 1, 1));

			for (zItMeshHalfEdge e_form(*formObj); !e_form.end(); e_form++)
			{
				int i = e_form.getId();
				if (deviations[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

					if (deviations[i] < angleTolerance) col = zColor();

					e_form.getEdge().setColor(col);

					int eId_force = formEdge_forceEdge[i];

					zItMeshHalfEdge eForce(*forceObj, eId_force);
					eForce.getEdge().setColor(col);

				}

			}

		}


		return out;
	}

	//---- graph specilization for updateFormDiagram
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::updateFormDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations)
	{


		if (fnFormParticles.size() != fnForm.numVertices())
		{
			fnFormParticles.clear();
			formParticlesObj.clear();


			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(formObj->graph.vertexPositions[i], fixed);
				formParticlesObj.push_back(p);

			}

			for (int i = 0; i < formParticlesObj.size(); i++)
			{
				fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
			}
		}

		vector<zVector> v_residual;
		v_residual.assign(fnForm.numVertices(), zVector());

		vector<double> edgelengths;
		fnForm.getEdgeLengths(edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = coreUtils.zMin(edgelengths);
		maxEdgeLength = coreUtils.zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;


		zVector* positions = fnForm.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItGraphVertex v(*formObj); !v.end(); v++)
			{
				int i = v.getId();
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zVector v_i = positions[v.getId()];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (auto &he : cEdges)
				{

					zVector v_j = positions[he.getVertex().getId()];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_form[he.getSym().getId()];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force				
				v_residual[i] = (b_i - v_i);

				zVector forceV = v_residual[i] * formVWeights[i];
				fnFormParticles[i].addForce(forceV);
			}

			// update positions
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				fnFormParticles[i].integrateForces(dT, type);
				fnFormParticles[i].updateParticle(true);
			}
		}
	}
	
	//---- mesh specilization for updateFormDiagram
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::updateFormDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations)
	{

		if (fnFormParticles.size() != fnForm.numVertices())
		{
			fnFormParticles.clear();
			formParticlesObj.clear();


			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(formObj->mesh.vertexPositions[i], fixed);
				formParticlesObj.push_back(p);

			}

			for (int i = 0; i < formParticlesObj.size(); i++)
			{
				fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
			}
		}

		vector<zVector> v_residual;
		v_residual.assign(fnForm.numVertices(), zVector());

		vector<double> edgelengths;
		fnForm.getEdgeLengths(edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = coreUtils.zMin(edgelengths);
		maxEdgeLength = coreUtils.zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;


		zVector* positions = fnForm.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItMeshVertex v(*formObj); !v.end(); v++)
			{
				int i = v.getId();
				vector<zItMeshHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				zVector v_i = positions[v.getId()];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (auto &he : cEdges)
				{

					zVector v_j = positions[he.getVertex().getId()];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_form[he.getSym().getId()];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force				
				v_residual[i] = (b_i - v_i);

				zVector forceV = v_residual[i] * formVWeights[i];
				fnFormParticles[i].addForce(forceV);
			}

			// update positions
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				fnFormParticles[i].integrateForces(dT, type);
				fnFormParticles[i].updateParticle(true);
			}
		}

	}

	//---- graph specilization for updateForceDiagram
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::updateForceDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations)
	{
		if (fnForceParticles.size() != fnForce.numVertices())
		{
			fnForceParticles.clear();
			forceParticlesObj.clear();


			for (int i = 0; i < fnForce.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(forceObj->mesh.vertexPositions[i], fixed);
				forceParticlesObj.push_back(p);

			}

			for (int i = 0; i < forceParticlesObj.size(); i++)
			{
				fnForceParticles.push_back(zFnParticle(forceParticlesObj[i]));
			}
		}

		vector<zVector> v_residual;
		v_residual.assign(fnForce.numVertices(), zVector());

		vector<double> edgelengths;
		fnForce.getEdgeLengths(edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = coreUtils.zMin(edgelengths);
		maxEdgeLength = coreUtils.zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;



		zVector* positions = fnForce.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItMeshVertex v(*forceObj); !v.end(); v++)
			{
				int i = v.getId();
				vector<zItMeshHalfEdge> cEdges;
				cEdges.clear();
				v.getConnectedHalfEdges(cEdges);

				zVector v_i = positions[v.getId()];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (auto &he : cEdges)
				{
					zVector v_j = positions[he.getVertex().getId()];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_force[he.getSym().getId()];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				v_residual[i] = b_i - v_i;

				zVector forceV = v_residual[i] * forceVWeights[i];
				fnForceParticles[i].addForce(forceV);
			}

			// update positions
			for (int i = 0; i < fnForceParticles.size(); i++)
			{
				fnForceParticles[i].integrateForces(dT, type);
				fnForceParticles[i].updateParticle(true);
			}
		}
	}

	//---- mesh specilization for updateForceDiagram
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::updateForceDiagram(float minmax_Edge, float dT, zIntergrationType type, int numIterations)
	{
		if (fnForceParticles.size() != fnForce.numVertices())
		{
			fnForceParticles.clear();
			forceParticlesObj.clear();


			for (int i = 0; i < fnForce.numVertices(); i++)
			{
				bool fixed = false;


				zObjParticle p;
				p.particle = zParticle(forceObj->mesh.vertexPositions[i], fixed);
				forceParticlesObj.push_back(p);

			}

			for (int i = 0; i < forceParticlesObj.size(); i++)
			{
				fnForceParticles.push_back(zFnParticle(forceParticlesObj[i]));
			}
		}

		vector<zVector> v_residual;
		v_residual.assign(fnForce.numVertices(), zVector());

		vector<double> edgelengths;
		fnForce.getEdgeLengths(edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = coreUtils.zMin(edgelengths);
		maxEdgeLength = coreUtils.zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minmax_Edge;



		zVector* positions = fnForce.getRawVertexPositions();

		for (int k = 0; k < numIterations; k++)
		{
			for (zItMeshVertex v(*forceObj); !v.end(); v++)
			{
				int i = v.getId();
				vector<zItMeshHalfEdge> cEdges;
				cEdges.clear();
				v.getConnectedHalfEdges(cEdges);

				zVector v_i = positions[v.getId()];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (auto &he : cEdges)
				{
					zVector v_j = positions[he.getVertex().getId()];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_force[he.getSym().getId()];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				v_residual[i] = b_i - v_i;

				zVector forceV = v_residual[i] * forceVWeights[i];
				fnForceParticles[i].addForce(forceV);
			}

			// update positions
			for (int i = 0; i < fnForceParticles.size(); i++)
			{
				fnForceParticles[i].integrateForces(dT, type);
				fnForceParticles[i].updateParticle(true);
			}
		}
	}


#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zTsVault<zObjMesh, zFnMesh>;

	template class zTsVault<zObjGraph, zFnGraph>;

#endif

}