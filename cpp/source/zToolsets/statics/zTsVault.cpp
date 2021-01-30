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
		setElementColorDomain(zResultDiagram);


		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createResultfromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createResultfromFile(string path, zFileTpye type)
	{
		fnResult.from(path, type);

		// add crease data
		if (type == zJSON)
		{
			getPleatDataJSON(path);
		}
		

		setTensionEdges(zResultDiagram);
		setElementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- graph specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setElementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---- mesh specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setElementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---- mesh specilization for createForcefromFile
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createFormfromFile(string path, zFileTpye type)
	{
		fnForm.from(path, type);

		setTensionEdges(zFormDiagram);
		setElementColorDomain(zFormDiagram);


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
		setElementColorDomain(zFormDiagram);
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
		setElementColorDomain(zFormDiagram);
	}

	//---- graph specilization for createFormFromForce
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createFormFromForce(bool excludeBoundary, bool PlanarForceMesh, bool rotate90)
	{
		fnForce.getDualGraph(*formObj, forceEdge_formEdge, formEdge_forceEdge, excludeBoundary, PlanarForceMesh, rotate90);

		fnForm.setEdgeColor(elementColorDomain.max);

		setVertexWeights(zFormDiagram);


		setFormTensionEdgesfromForce();
		setElementColorDomain(zFormDiagram);
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
		setElementColorDomain(zForceDiagram);
	}

	//---- graph specilization for createResultFromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::createResultFromForm()
	{
		*resultObj = fnForm.getDuplicate();

		setResultTensionEdgesfromForm();
		setElementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createResultFromForm
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::createResultFromForm()
	{
		fnForm.getDuplicate(*resultObj);

		setResultTensionEdgesfromForm();
		setElementColorDomain(zResultDiagram);

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

				if (q[FD_EdgesCounter] < 0)positiveDensities = false;

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

	//---- BEST FIT FDM AND CONSTRAINT METHODS

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

				if(q[FD_EdgesCounter] < 0)positiveDensities = false;

				FD_EdgesCounter++;
			}

		}


		//printf("\n Force Densities: \n");
		//cout << endl << q << endl;

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

		//cout << "Pn: \n" << Pn << endl;

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

	//---- mesh specilization for fdm_constraintsolve
	template<>
	ZSPACE_INLINE double zTsVault<zObjMesh, zFnMesh>::fdm_constraintsolve(bool & computeQInitial, float alpha, float tolerance, float qLB, float qUB)
	{
		// 0 - compute q intial
		if (computeQInitial)
		{
			numFreeEdges = getNumFreeEdges(forcedensityEdgeMap);
			getfreeVertices(freeVertices);
			getPositionMatrix(X_orig);

			X = X_orig;

			C = getEdgeNodeMatrix(numFreeEdges);

			getLoadVector(Pn);

			getBestFitForceDensities(qInitial);	

			zPoint p_center;
			zVector p_norm(1, 0, 0);
			getSymmetryPairs(vertexSymmetryPairs, edgeSymmetryPairs, p_center, p_norm);

			boundForceDensities(forcedensityEdgeMap,qInitial, qLB, qUB);
			qCurrent = qInitial;

			
		}

		else getLoadVector(Pn);;

		// 1 - Move vertices to user specified pleat depth values
		//perturbPleatPositions(X_orig);
		//getPositionMatrix(X);

		// 2 - Compute design target vectors
		zVectorArray targets;
		//getConstraint_Planarity(targets);


		// 3 - Compute residual and gradient q
		VectorXd qGradient, posGradient, residual, residualU, residualC;
		//getResidual_Gradient(qCurrent, targets, residual, qGradient);

		getResiduals(alpha, qCurrent, residual, residualU, residualC);
		getGradients(qCurrent, residualU, residualC, posGradient, qGradient);

		if (computeQInitial)
		{
			/*float d0 = residual.norm();
			int n_v = fnResult.numVertices() - fixedVertices.size();
			int counter = 0;
			for (zItMeshVertex v(*resultObj); !v.end(); v++)
			{
				int i = v.getId();
				if (fixedVerticesBoolean[i]) continue;

				if (v.onBoundary())
				{
					printf("\n initial %i : %1.4f %1.4f %1.4f ", i, residual[counter], residual[n_v + counter], residual[(n_v * 2) + counter]);

					zVector posG(residual[counter], residual[n_v + counter], residual[(n_v * 2) + counter]);
					
					zPoint p = v.getPosition();
					p += posG;

					v.setPosition(p);
					
				}
				counter++;
			}
			printf("\n residual0 : %1.4f  ", d0);

			getPositionMatrix(X);

			getBestFitForceDensities(qInitial);
			boundForceDensities(forcedensityEdgeMap, qInitial, qLB, qUB);
			qCurrent = qInitial;

			getResiduals(alpha, qCurrent, residual, residualU, residualC);
			getGradients(qCurrent, residualU, residualC, posGradient, qGradient);*/

			computeQInitial = !computeQInitial;
		}

		//boundGradientForceDensities(qGradient, qCurrent, qLB, qUB);

		// 4 - update qCurrent
		qCurrent += qGradient;
		boundForceDensities(forcedensityEdgeMap, qCurrent, qLB, qUB);
		//cout << endl << "qCurrent \n " << qCurrent << endl;

		// 5 - update positions
		MatrixXd Xtemp = X;
		//updateEquilibriumPositions(qCurrent);

		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			zItMeshVertex v(*resultObj, id);

			zVector posG(posGradient(i +  0 * freeVertices.size()), posGradient(i + 1 * freeVertices.size()), posGradient(i + 2 * freeVertices.size()));
			//posG *= 0.1;

			zPoint p = v.getPosition();
			p += posG;
			
			v.setPosition(p);
		}

		getPositionMatrix(X);

		int qID = 0;
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			zIntArray eVerts;
			e.getVertices(eVerts);

			if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]])
			{
				e.setColor(zColor(0,0,0,1));
				continue;
			}

			
			if (qCurrent[qID] > 0)  e.setColor(elementColorDomain.max);
			else  e.setColor(elementColorDomain.min);

			qID++;
		}

		// 6 - calculate objective function

		// 7 - check if update below threshold
		//bool out = checkObjectiveAchieved(X, Xtemp, tolerance);	

		float d = residual.norm();
		int n_v = fnResult.numVertices() - fixedVertices.size();
		int counter = 0;
		for (zItMeshVertex v(*resultObj); !v.end(); v++)
		{
			int i = v.getId();
			if (fixedVerticesBoolean[i]) continue;

			//if(v.onBoundary()) printf("\n %i : %1.4f %1.4f %1.4f ", i, residual[counter], residual[n_v + counter], residual[(n_v * 2) + counter]);
			
			counter++;
		}
		printf("\n residual : %1.4f  ", d);

		return d ;
	}

	//---- mesh specilization for getSymmetryPairs
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getSymmetryPairs(zIntPairArray &vPairs, zIntPairArray & ePairs, zPoint &p_center, zVector &p_norm)
	{
	
		vPairs.clear();
		vPairs.assign(fnResult.numVertices(), zIntPair());

		ePairs.clear();
		ePairs.assign(fnResult.numEdges(), zIntPair());

		zPoint* pos = fnResult.getRawVertexPositions();

		// vertex pairs
		for (int i = 0; i < fnResult.numVertices();i++)
		{

			vPairs[i] = std::make_pair(i, -1);

			zPoint pos0 = pos[i];
			float d0 = coreUtils.minDist_Point_Plane(pos0, p_center, p_norm);
						

			if (abs(d0) < EPS)
			{
				vPairs[i] = std::make_pair(i, i);
				continue; // lying on symmetry plan
			}

			for (int j = 0; j < fnResult.numVertices(); j++)
			{
				if (i == j) continue;
			
				zPoint pos1 = pos[j];
				float d1 = coreUtils.minDist_Point_Plane(pos1, p_center, p_norm);

				if (d0 == (d1 * -1))
				{
					if (vPairs[i].second != -1)
					{
						zPoint pos2 = pos[vPairs[i].second];

						if (pos1.distanceTo(pos0) < pos2.distanceTo(pos0))
						{
							vPairs[i] = std::make_pair(i, j);
						}

					}
					else
					{
						vPairs[i] = std::make_pair(i, j);
						
					}					
					
				}

			}

		}

		// edge pairs
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			zIntArray eVerts;
			e.getVertices(eVerts);

			if (vPairs[eVerts[0]].second != -1 && vPairs[eVerts[1]].second != -1)
			{

				int v0 = vPairs[eVerts[0]].second;
				int v1 = vPairs[eVerts[1]].second;

				int he;
				bool chk = fnResult.halfEdgeExists(v0, v1, he);

				if (chk)
				{
					int eSymId = floor(he * 0.5);
					ePairs[e.getId()] = std::make_pair(e.getId(), eSymId);

					//printf("\n %i %i %i | %i %i %i", e.getId(), eVerts[0], eVerts[1], eSymId, v0, v1);
				}

			}


		}
		
	}

	//---- mesh specilization for fdm_constraintsolve

	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::boundForceDensities(zIntArray &fdMap, VectorXd &fDensities, float qLB, float qUB)
	{
		int id = 0;
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			zIntArray eVerts;
			e.getVertices(eVerts);

			int fdId0 = fdMap[e.getId()];

			if (fdId0 == -1) continue;

			

			/*if (edgeSymmetryPairs.size() > 0)
			{
				if (edgeSymmetryPairs[e.getId()].second != -1)
				{
					int fdId1 = fdMap[edgeSymmetryPairs[e.getId()].second];
					
					if (fDensities[fdId1] != fDensities[fdId0])
					{
						fDensities[fdId0] = fDensities[fdId1];
					}

				}

			}*/

			//if (fDensities[id] <= qLB) fDensities[id] = qLB;
			//if (fDensities[id] >= qUB) fDensities[id] = qUB;
			

			if (fDensities[id] >= -qLB) fDensities[id] = -qLB;
			if (fDensities[id] <= -qUB) fDensities[id] = -qUB;

			//if (!e.onBoundary())
			//{
			//	if (fDensities[id] < qLB) fDensities[id] = qLB;
			//	if (fDensities[id] > qUB) fDensities[id] = qUB;

			//	if (fDensities[id] < 0 && abs(fDensities[id]) < EPS) fDensities[id] = -EPS;
			//	if (fDensities[id] > 0 && abs(fDensities[id]) < EPS) fDensities[id] = EPS;
			//}

			//else
			//{
			//	//if (fDensities[id] < qLB) fDensities[id] = qLB;
			//	//if (fDensities[id] > qUB) fDensities[id] = qUB;

			//	if (fDensities[id] >= 0) fDensities[id] = -qLB;
			//	////if (fDensities[id] > qLB) fDensities[id] = qLB;

			//	////if (fDensities[id] < -qLB || abs(fDensities[id]) < EPS) fDensities[id] = -qLB;
			//	////if (fDensities[id] > qUB) fDensities[id] = qUB;

			//	if (fDensities[id] < 0 && abs(fDensities[id]) < EPS) fDensities[id] = -EPS;
			//	if (fDensities[id] > 0 && abs(fDensities[id]) < EPS) fDensities[id] = EPS;
			//}

			id++;
		}
	}

	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::boundGradientForceDensities(VectorXd & grad_fDensities, VectorXd & current_fDensities, float qLB, float qUB)
	{

		/*for (int i = 0; i < current_fDensities.rows(); i++)
		{
			if (current_fDensities[i] < qLB) grad_fDensities[i] = qLB - current_fDensities[i];
			if (current_fDensities[i] > qUB) grad_fDensities[i] = qUB - current_fDensities[i];
		}*/

		int id = 0;
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			zIntArray eVerts;
			e.getVertices(eVerts);

			if (fixedVerticesBoolean[eVerts[0]] && fixedVerticesBoolean[eVerts[1]]) continue;

			if (!e.onBoundary())
			{
				if (current_fDensities[id] < qLB) grad_fDensities[id] = qLB - current_fDensities[id];
				if (current_fDensities[id] > qUB) grad_fDensities[id] = qUB - current_fDensities[id];
			}

			/*else
			{
				if (current_fDensities[id] < -qUB)  grad_fDensities[id] = -qUB - current_fDensities[id]; 
				if (current_fDensities[id] > qLB)  grad_fDensities[id] = qLB - current_fDensities[id];
			}*/

			id++;
		}
	}

	//---- mesh specilization for checkObjectiveAchieved
	template<>
	ZSPACE_INLINE bool zTsVault<zObjMesh, zFnMesh>::checkObjectiveAchieved(MatrixXd & currentX, MatrixXd & prevX, float tolerance)
	{
		bool out = true;
		
		for (int i = 0; i < freeVertices.size(); i++)
		{
			zVector pos(currentX(i, 0), currentX(i, 1), currentX(i, 2));
			zVector posPrev(prevX(i, 0), prevX(i, 1), prevX(i, 2));

			if (pos.distanceTo(posPrev) > tolerance)
			{
				out = false;
				break;
			}
		}


		return out;
	}

	//---- mesh specilization for updatePositions
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::updateEquilibriumPositions(VectorXd & q)
	{
		zHEData type = zVertexData;
		bool positiveDensities = true;

		int n_v = fnResult.numVertices();
		int numEdges = numFreeEdges;;
	

		// POSITION MATRIX
		

		// FORCE DENSITY VECTOR
		
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
		
		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);		

		MatrixXd Pn_mat = subMatrix(P, freeVertices);

		zSparseMatrix Cn_transpose;
		Cn_transpose = Cn.transpose();

		//CHOLESKY DECOMPOSITION

		zSparseMatrix Dn = Cn_transpose * Q * Cn;
		zSparseMatrix Df = Cn_transpose * Q * Cf;

		MatrixXd  B = Pn_mat - Df * Xf;

		// solve
		MatrixXd Xn;


		MatrixXd denseDn;
		denseDn = MatrixXd(Dn);
		Xn = denseDn.ldlt().solve(B);

		// convergence error check.
		double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
		cout << endl << relative_error << " FDM - negative" << endl;

	

		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			zItMeshVertex v(*resultObj, id);

			zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
			v.setPosition(pos);			
		}				

	}

	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::perturbPleatPositions( MatrixXd &origX)
	{
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			zItMeshVertex v(*resultObj, id);


			zItMeshHalfEdgeArray cHEdges;
			v.getConnectedHalfEdges(cHEdges);


			zVector p = v.getPosition();
			//cout << "\n p: " << p;

			// compute normal
			zVector n;
			zVector c;
			for (int j = 0; j < cHEdges.size(); j++)
			{	

				zPoint p2 = cHEdges[j].getSym().getVertex().getPosition();
				zVector e1 = p2 - p;
				
				zPoint p3 = cHEdges[(j + 1) % cHEdges.size()].getSym().getVertex().getPosition();

				zVector e2 = p3 - p;
				zVector cross = e1 ^ e2;

				n += cross;
				c += p2;
			}
			n /= cHEdges.size();
			n.normalize();
			c /= cHEdges.size();

			

			float offset = (result_vertex_PleatDepth[id]*2) - c.distanceTo(p);

			p += (n * offset);

			//if (result_vertex_ValleyRidge[id] == 1) // valley
			//{				
			//	p += (n * offset);
			//}
			//else if (result_vertex_ValleyRidge[id] == 2) // ridge
			//{
			//	p += (n * -1 * offset);
			//}

			//zVector pOrig(origX(id, 0), origX(id, 1), origX(id, 2));

			//float offset = result_vertex_PleatDepth[id] -  (p.z - pOrig.z);
			////if (id == 25) cout << " \n offset " << offset;

			//p.z += offset;

			v.setPosition(p);
		}
	}


	//---- mesh specilization for computeBestFitForceDensities
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getBestFitForceDensities(VectorXd &bestfit_fDensities)
	{
		int n_v = fnResult.numVertices();

		int numEdges = numFreeEdges;;		

		// POSITION MATRIX
		
		
		// SUB MATRICES	

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);

		zSparseMatrix Cnt;
		Cnt = Cn.transpose();		

		VectorXd u, v, w;
		u = C * X.col(0);
		v = C * X.col(1);
		w = C * X.col(2);

		zDiagonalMatrix U = u.asDiagonal();
		zDiagonalMatrix V = v.asDiagonal();
		zDiagonalMatrix W = w.asDiagonal();

		MatrixXd A(3 * freeVertices.size(), numEdges);
		A << MatrixXd(Cnt*U), MatrixXd(Cnt*V), MatrixXd(Cnt*W);

		//cout << "P: \n" << Pn << endl;
		//printf("\n %i %i ", freeVert_Counter, freeVertices.size());

		//clock_t cpu_startTime, cpu_endTime;

		//double cpu_ElapseTime = 0;
		//cpu_startTime = clock();

		MatrixXd At = A.transpose();
		MatrixXd LHS = At * A;
		MatrixXd RHS = At * (Pn * 1);

		bestfit_fDensities = LHS.ldlt().solve(RHS);
		bestfit_fDensities *= -1;

		//cpu_endTime = clock();
		//cpu_ElapseTime = ((cpu_endTime - cpu_startTime));

		//printf("\n cpu %1.8f ms \n", cpu_ElapseTime);

		//cout << "\n initial q \n " << bestfit_fDensities << endl;

	}

	//---- mesh specilization for computeBestFitForceDensities
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getBestFitForceDensities(zFloatArray &bestfit_fDensities)
	{
		numFreeEdges = getNumFreeEdges(forcedensityEdgeMap);
		getfreeVertices(freeVertices);

		int n_v = fnResult.numVertices();

		int numEdges = numFreeEdges;;

		/*for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}*/

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

		// SUB MATRICES
		//vector<int> freeVertices;

		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVertices.push_back(j);
			}

		}

		zSparseMatrix Cn = subMatrix(C, freeVertices);
		zSparseMatrix Cf = subMatrix(C, fixedVertices);

		zSparseMatrix Cnt;
		Cnt = Cn.transpose();

		MatrixXd Xn = subMatrix(X, freeVertices);

		VectorXd u, v, w;
		u = C * X.col(0);
		v = C * X.col(1);
		w = C * X.col(2);

		zDiagonalMatrix U = u.asDiagonal();
		zDiagonalMatrix V = v.asDiagonal();
		zDiagonalMatrix W = w.asDiagonal();

		MatrixXd A(3 * freeVertices.size(), numEdges);
		A << MatrixXd(Cnt*U), MatrixXd(Cnt*V), MatrixXd(Cnt*W);	

		
		getLoadVector(Pn);

		
		//printf("\n  Pn: %i %i  ", Pn.rows(), Pn.cols());

		//cout << "P: \n" << Pn << endl;
			//printf("\n %i %i ", freeVert_Counter, freeVertices.size());
		

		MatrixXd LHS = A.transpose() * A;
		MatrixXd RHS = A.transpose() * Pn   ;
		VectorXd del_q = LHS.ldlt().solve(RHS);

		printf("\n bestFit Q \n");
		cout << del_q << endl <<endl;

		bestfit_fDensities.clear();

		int counter = 0;
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) 
				bestfit_fDensities.push_back(0.0);
			else
			{
				bestfit_fDensities.push_back(del_q(counter));
				counter++;
			}

			//printf("\n %1.4f ", bestfit_fDensities[e.getId()]);

		}

		

	}


	//---- mesh specilization for getResidual_Gradient
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getResidual_Gradient(VectorXd & current_fDensities, zVectorArray &targets, VectorXd & residual, VectorXd & gradient_fDensities)
	{

		bool positiveDensities = true;

		int n_v = fnResult.numVertices();
		int numEdges = numFreeEdges;;
		
		
		// SUB MATRICES
		zSparseMatrix Cn = subMatrix(C, freeVertices);

		zSparseMatrix Cnt;
		Cnt = Cn.transpose();

		VectorXd u, v, w;
		u = C * X.col(0);
		v = C * X.col(1);
		w = C * X.col(2);

		zDiagonalMatrix U = u.asDiagonal();
		zDiagonalMatrix V = v.asDiagonal();
		zDiagonalMatrix W = w.asDiagonal();

		MatrixXd A(3 * freeVertices.size(), numEdges);
		A << MatrixXd(Cnt*U), MatrixXd(Cnt*V), MatrixXd(Cnt*W);

		

		// FORCE DENSITY VECTOR

		// residual
		residual = (A * current_fDensities) - Pn ;

		//VectorXd g = residual;
		////cout << "\n residual r \n " << residual << endl;
		//
		//if (targets.size() > 0)
		//{
		//	VectorXd Tn(freeVertices.size() * 3);
		//	int freeVert_Counter = 0;

		//	for (int j = 0; j < resultVMass.size(); j++)
		//	{
		//		if (!fixedVerticesBoolean[j])
		//		{
		//			Tn[freeVert_Counter + (0 * freeVertices.size())] = targets[j].x;
		//			Tn[freeVert_Counter + (1 * freeVertices.size())] = targets[j].y;
		//			Tn[freeVert_Counter + (2 * freeVertices.size())] = targets[j].z;
		//			freeVert_Counter++;
		//		}
		//	}

		//	g += Tn;
		//}

		// gradient q
		MatrixXd LHS = A.transpose() * A;
		MatrixXd RHS = A.transpose() * residual;
		gradient_fDensities = LHS.ldlt().solve(RHS);
		
		//cout << "\n gradient q \n " << gradient_fDensities << endl;

	}

	//---- mesh specilization for getResidual_Gradient
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getResiduals(float alpha,VectorXd & current_fDensities, VectorXd &residual, VectorXd & residualU, VectorXd & residualC)
	{

		bool positiveDensities = true;

		int n_v = fnResult.numVertices();
		int numEdges = numFreeEdges;;


		// SUB MATRICES
		zSparseMatrix Cn = subMatrix(C, freeVertices);

		zSparseMatrix Cnt;
		Cnt = Cn.transpose();

		VectorXd u, v, w;
		u = C * X.col(0);
		v = C * X.col(1);
		w = C * X.col(2);

		zDiagonalMatrix U = u.asDiagonal();
		zDiagonalMatrix V = v.asDiagonal();
		zDiagonalMatrix W = w.asDiagonal();

		MatrixXd A(3 * freeVertices.size(), numEdges);
		A << MatrixXd(Cnt*U), MatrixXd(Cnt*V), MatrixXd(Cnt*W);
			   
		// FORCE DENSITY VECTOR

		// residual
		residual = (A * current_fDensities) - (Pn);

		residualU = VectorXd(residual.rows());
		residualC = VectorXd(residual.rows());


		int id = 0;
		for (int j = 0; j < n_v; j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				zItMeshVertex v(*resultObj, j);

				if (!v.onBoundary()) // free node 
				{
					/* DNIPRO
					residualU(id + 0 * freeVertices.size()) = alpha * residual(id + 0 * freeVertices.size());
					residualU(id + 1 * freeVertices.size()) = alpha * residual(id + 1 * freeVertices.size());
					residualU(id + 2 * freeVertices.size()) = alpha * residual(id + 2 * freeVertices.size());

					residualC(id + 0 * freeVertices.size()) = (1 - alpha) * residual(id + 0 * freeVertices.size());
					residualC(id + 1 * freeVertices.size()) = (1 - alpha) * residual(id + 1 * freeVertices.size());
					residualC(id + 2 * freeVertices.size()) = (1 - alpha) * residual(id + 2 * freeVertices.size());*/
				
					residualU(id + 0 * freeVertices.size()) = 1.0 * residual(id + 0 * freeVertices.size());
					residualU(id + 1 * freeVertices.size()) = 1.0 * residual(id + 1 * freeVertices.size());
					residualU(id + 2 * freeVertices.size()) = 1.0 * residual(id + 2 * freeVertices.size());

					residualC(id + 0 * freeVertices.size()) = 0.0 * residual(id + 0 * freeVertices.size());
					residualC(id + 1 * freeVertices.size()) = 0.0 * residual(id + 1 * freeVertices.size());
					residualC(id + 2 * freeVertices.size()) = 0.0 * residual(id + 2 * freeVertices.size());
				
				}
				else // free node constraint to positions
				{
					residualU(id + 0 * freeVertices.size()) = 0.0 * residual(id + 0 * freeVertices.size());
					residualU(id + 1 * freeVertices.size()) = 0.0 * residual(id + 1 * freeVertices.size());
					residualU(id + 2 * freeVertices.size()) = 1.0 * residual(id + 2 * freeVertices.size());

					residualC(id + 0 * freeVertices.size()) = 1.0 * residual(id + 0 * freeVertices.size());
					residualC(id + 1 * freeVertices.size()) = 1.0 * residual(id + 1 * freeVertices.size());
					residualC(id + 2 * freeVertices.size()) = 0.00 * residual(id + 2 * freeVertices.size());
				}

				id++;
			}
		}

	}

	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getGradients(VectorXd & current_fDensities, VectorXd & residualU, VectorXd & residualC, VectorXd & gradPos, VectorXd & gradFDensities)
	{
		int n_v = fnResult.numVertices();
		int numEdges = numFreeEdges;;


		// SUB MATRICES
		zSparseMatrix Cn = subMatrix(C, freeVertices);

		zSparseMatrix CTabs =  Cn.transpose().cwiseAbs();
		
		zSparseMatrix Cnt;
		Cnt = Cn.transpose();

		VectorXd u, v, w;
		u = C * X.col(0);
		v = C * X.col(1);
		w = C * X.col(2);

		zDiagonalMatrix U = u.asDiagonal();
		zDiagonalMatrix V = v.asDiagonal();
		zDiagonalMatrix W = w.asDiagonal();

		VectorXd l(u.rows());

		for (int i = 0; i < u.rows(); i++)l(i) = (u(i) * u(i)) + (v(i) * v(i)) + (w(i) * w(i));;		
				
		zDiagonalMatrix L = l.asDiagonal();
	

		VectorXd current_fDensities_abs = current_fDensities.cwiseAbs();

		printf("\n CTabs %i %i ", CTabs.rows(), CTabs.cols());
		printf("\n L %i %i ", L.rows(), L.cols());
		printf("\n current_fDensities %i %i ", current_fDensities.rows(), current_fDensities.cols());

		VectorXd k = (CTabs * L * current_fDensities);
		VectorXd denom = (CTabs * L * current_fDensities_abs);
			
		for (int i = 0; i < k.rows(); i++)
		{
			k(i) /= (denom(i));
			k(i) *= -1;
		}
		

		gradPos = VectorXd(3 * freeVertices.size());
		
		int freeVert_Counter = 0;
		for (int j = 0; j < n_v; j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				gradPos[freeVert_Counter + (0 * freeVertices.size())] = residualU[freeVert_Counter + (0 * freeVertices.size())] * k[freeVert_Counter];
				gradPos[freeVert_Counter + (1 * freeVertices.size())] = residualU[freeVert_Counter + (1 * freeVertices.size())] * k[freeVert_Counter];
				gradPos[freeVert_Counter + (2 * freeVertices.size())] = residualU[freeVert_Counter + (2 * freeVertices.size())] * k[freeVert_Counter];


				zItMeshVertex v(*resultObj, j);

				/*if (!v.onBoundary())
				{
					printf("\n %1.6f |  %1.6f %1.6f %1.6f ", k[freeVert_Counter],
													gradPos[freeVert_Counter + (0 * freeVertices.size())],
													gradPos[freeVert_Counter + (1 * freeVertices.size())],
													gradPos[freeVert_Counter + (2 * freeVertices.size())]);
				}*/

				freeVert_Counter++;
			}
		}

		MatrixXd A(3 * freeVertices.size(), numEdges);
		A << MatrixXd(Cnt*U), MatrixXd(Cnt*V), MatrixXd(Cnt*W);

		MatrixXd LHS = A.transpose() * A;
		MatrixXd RHS = A.transpose() * residualC;
		gradFDensities = LHS.ldlt().solve(RHS);

		gradFDensities *= -1;
	}

	//---- getfreeVertices
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getfreeVertices(zIntArray & freeVerts)
	{
		freeVerts.clear();
		for (int j = 0; j < fixedVerticesBoolean.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				freeVerts.push_back(j);
			}

		}
	}

	//---- getNumFreeEdges
	template<>
	ZSPACE_INLINE int zTsVault<zObjMesh, zFnMesh>::getNumFreeEdges(zIntArray &fdMap)
	{
		int numEdges = fnResult.numEdges();;
		fdMap.clear();

		fdMap.assign(numEdges, -1);
		int fdCounter = 0;
		for (zItMeshEdge e(*resultObj); !e.end(); e++)
		{
			vector<int> eVerts;
			e.getVertices(eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;
			else
			{
				fdMap[e.getId()] = fdCounter;
				fdCounter++;
			}

		}

		return numEdges;
	}

	//---- mesh specilization for getPositionMatrix
	template<>
	ZSPACE_INLINE void zTsVault<zObjGraph, zFnGraph>::getPositionMatrix(MatrixXd & X)
	{
		X = MatrixXd(fnResult.numVertices(), 3);

		for (zItGraphVertex v(*resultObj); !v.end(); v++)
		{
			zVector pos = v.getPosition();
			int i = v.getId();

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};
	}

	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getPositionMatrix(MatrixXd & X)
	{
		 X = MatrixXd(fnResult.numVertices(), 3);

		for (zItMeshVertex v(*resultObj); !v.end(); v++)
		{
			zVector pos = v.getPosition();
			int i = v.getId();

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};
	}

	//---- getLoadVector
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getLoadVector(VectorXd & Pn)
	{
		//setVertexMassfromVertexArea();

		 Pn = VectorXd(freeVertices.size() * 3);


		int freeVert_Counter = 0;
		for (int j = 0; j < resultVMass.size(); j++)
		{
			if (!fixedVerticesBoolean[j])
			{
				Pn[freeVert_Counter + (0 * freeVertices.size())] = 0;
				Pn[freeVert_Counter + (1 * freeVertices.size())] = 0;
				Pn[freeVert_Counter + (2 * freeVertices.size())] = resultVMass[j] * resultVThickness[j] * resultVWeights[j];
				freeVert_Counter++;
			}
		}
	}

	//---- mesh specilization for setConstraint_Planarity
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getConstraint_Planarity(zVectorArray &targets, float planarityTolerance)
	{

		if (targets.size() == 0)
		{
			targets.assign(fnResult.numVertices(), zVector());
		}

		zInt2DArray fTris;
		zPointArray fCenters;
		zDoubleArray fVolumes;
		fnResult.getMeshFaceVolumes(fTris, fCenters, fVolumes);
		
		for (zItMeshFace f(*resultObj); !f.end(); f++)
		{
			int i = f.getId();

			if (f.isActive() && fVolumes[i] > planarityTolerance)
			{
				vector<int> fVerts;
				f.getVertices(fVerts);

				vector<zVector> fVertPositions;
				f.getVertexPositions(fVertPositions);

				zVector fNormal = f.getNormal();

				for (int j = 0; j < fVertPositions.size(); j++)
				{
					if (fixedVerticesBoolean[fVerts[j]]) continue;

					double dist = coreUtils.minDist_Point_Plane(fVertPositions[j], fCenters[i], fNormal);

					zVector pForce = fNormal * dist * -1.0;
					
					targets[fVerts[j]] += pForce;
				}
			}

		}
	}

	//---- mesh specilization for setConstraint_plan
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::setConstraint_plan(const zIntArray & vertex_PlanWeights)
	{		
		
		if (vertex_PlanWeights.size() == 0)
		{
			result_vertex_PlanConstraints.clear();

		}
		else
		{
			result_vertex_PlanConstraints = vertex_PlanWeights;
		}

		result_vertex_PlanConstraintsBoolean.clear();
		result_vertex_PlanConstraintsBoolean.assign(fnResult.numVertices(), false);

		for (int i = 0; i < result_vertex_PlanConstraints.size(); i++)
		{			
			result_vertex_PlanConstraintsBoolean[result_vertex_PlanConstraints[i]] = true;
		}
	}

	//---- mesh specilization for setConstraint_pleats
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::setConstraint_pleats(zFloatArray & vertex_ValleyRidge, zFloatArray & vertex_PleatDepth)
	{
		result_vertex_ValleyRidge = vertex_ValleyRidge;
		result_vertex_PleatDepth = vertex_PleatDepth;
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

			setElementColorDomain(zFormDiagram);
			setElementColorDomain(zForceDiagram);
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
				fixedVertices.clear();
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

		setResultTensionEdgesfromForceDensities();
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

		setResultTensionEdgesfromForceDensities();
	}

	//---- graph specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setForceDensities(vector<float> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of graph half edges.");

		forceDensities = fDensities;

		setResultTensionEdgesfromForceDensities();
	}

	//---- mesh specilization for setForceDensities
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setForceDensities(vector<float> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of mesh edges.");

		forceDensities = fDensities;

		//setResultTensionEdgesfromForceDensities();
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


		//setResultTensionEdgesfromForceDensities();

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

		//setResultTensionEdgesfromForceDensities();
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

			setElementColorDomain(type);
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

			setElementColorDomain(type);
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

			setElementColorDomain(type);
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

			setElementColorDomain(type);
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

			setElementColorDomain(type);
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

			setElementColorDomain(type);
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

		setElementColorDomain(zForceDiagram);
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

		setElementColorDomain(zForceDiagram);
	}

	template<typename T, typename U>
	void zTsVault<T, U>::setResultTensionEdgesfromForceDensities()
	{
		result_tensionEdges.clear();
	
		for (int i = 0; i < fnResult.numHalfEdges(); i++)
		{
			result_tensionEdges.push_back(false);
		}


		for (int i = 0; i < forceDensities.size(); i++)
		{
			if (forceDensities[i] < 0)
			{
				result_tensionEdges[i * 2 + 0] = true;
				result_tensionEdges[i * 2 + 1] = true;

				forceDensities[i] = abs(forceDensities[i]);
			}
		}

		setElementColorDomain(zResultDiagram);
	}

	template<typename T, typename U>
	ZSPACE_INLINE void zTsVault<T, U>::setResultTensionEdgesfromForm()
	{
		result_tensionEdges.clear();
		result_tensionEdges = form_tensionEdges;


		setElementColorDomain(zResultDiagram);
	}

	template<typename T, typename U>
	ZSPACE_INLINE void  zTsVault<T, U>::setFormTensionEdgesfromResult()
	{
		form_tensionEdges.clear();
		form_tensionEdges = result_tensionEdges;

		setElementColorDomain(zFormDiagram);

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

		setElementColorDomain(zFormDiagram);

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

		setElementColorDomain(zFormDiagram);

	}

	//---- graph specilization for setElementColorDomain
	template<>
	ZSPACE_INLINE  void zTsVault<zObjGraph, zFnGraph>::setElementColorDomain(zDiagramType type)
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

	//---- mesh specilization for setElementColorDomain	
	template<>
	ZSPACE_INLINE  void zTsVault<zObjMesh, zFnMesh>::setElementColorDomain(zDiagramType type)
	{
		if (type == zFormDiagram)
		{
			for (zItMeshHalfEdge he(*formObj); !he.end(); he++)
			{
				int i = he.getId();
								
				if (form_tensionEdges[i]) he.getEdge().setColor(elementColorDomain.min);
				else  he.getEdge().setColor(elementColorDomain.max);

				if (fixedVerticesBoolean.size() > 0)
				{
					if (fixedVerticesBoolean[he.getVertex().getId()] && fixedVerticesBoolean[he.getStartVertex().getId()]) 
						he.getEdge().setColor(zColor(0,0,0,1));
				}
			}
		}
		else if (type == zResultDiagram)
		{
			for (zItMeshHalfEdge he(*resultObj); !he.end(); he++)
			{
				int i = he.getId();
			
				if (fixedVerticesBoolean.size() > 0)
				{
					if (fixedVerticesBoolean[he.getVertex().getId()] && fixedVerticesBoolean[he.getStartVertex().getId()]) continue;
				}

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


	//---- getConstraints
	template<typename T, typename U>
	zIntArray zTsVault<T, U>::getConstraints()
	{
		return fixedVertices;
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
	ZSPACE_INLINE  zSparseMatrix zTsVault<zObjGraph, zFnGraph>::getEdgeNodeMatrix(int numRows)
	{
		int n_v = fnResult.numVertices();
		zSparseMatrix out(numRows, n_v);
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
	ZSPACE_INLINE zSparseMatrix zTsVault<zObjMesh, zFnMesh>::getEdgeNodeMatrix(int numRows)
	{
		int n_v = fnResult.numVertices();

		zSparseMatrix out(numRows, n_v);
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


	//---- mesh specilization for getPleatDataJSON
	template<>
	ZSPACE_INLINE void zTsVault<zObjMesh, zFnMesh>::getPleatDataJSON(string infilename)
	{
		json j;
		zUtilsJsonHE meshJSON;


		ifstream in_myfile;
		in_myfile.open(infilename.c_str());

		int lineCnt = 0;

		if (in_myfile.fail())
		{
			cout << " error in opening file  " << infilename.c_str() << endl;

		}

		in_myfile >> j;
		in_myfile.close();

		// READ Data from JSON

		// Vertices
		/*meshJSON.pleatData.clear();

		bool checkCreaseData = (j.find("PleatDepth") != j.end()) ? true : false;
		result_vertex_PleatDepth.clear();

		if (checkCreaseData)
		{
			meshJSON.pleatData = (j["PleatDepth"].get<vector<double>>());

			for (int i = 0; i < meshJSON.pleatData.size(); i++)
			{

				result_vertex_PleatDepth.push_back(meshJSON.pleatData[i]);

			}
		}
*/

		meshJSON.vertexAttributes.clear();

		meshJSON.vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();

		for (int i = 0; i < meshJSON.vertexAttributes.size(); i++)
		{
			for (int k = 0; k < meshJSON.vertexAttributes[i].size(); k++)
			{

				// pleat depths
				if (meshJSON.vertexAttributes[i].size() == 8)
				{
					result_vertex_PleatDepth.push_back(meshJSON.vertexAttributes[i][k + 6]);

					k += 7;
				}
			}

		}
			
		
		//for (auto d : result_vertex_PleatDepth) printf("\n %1.2f ", d);

	}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
	// explicit instantiation
	template class zTsVault<zObjMesh, zFnMesh>;

	template class zTsVault<zObjGraph, zFnGraph>;

#endif

}