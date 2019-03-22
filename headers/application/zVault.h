#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>
#include <headers/geometry/zGraphMeshUtilities.h>
#include <headers/geometry/zMeshModifiers.h>

#include <headers/dynamics/zParticle.h>

#include <headers/IO/zExchange.h>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> Diag;

namespace zSpace
{
	/** \addtogroup zApplication
	*	\brief Collection of general applications.
	*  @{
	*/

	/** \addtogroup zVault
	*	\brief Collection of methods for form-finding compressin or tension nets .
	*  @{
	*/

	/** \addtogroup zVault_Utilities
	*	\brief Collection of utility methods used in form finding.
	*  @{
	*/

	//--------------------------
	//---- MESH UTILITES
	//--------------------------
	
	/*! \brief This method computes the Edge Node Matrix for the input mesh.
	*
	*	\tparam				T									- Type to work with SpMat and MatrixXd.
	*	\param		[in]	inMesh								- input mesh.
	*	\param		[in]	fixedVerticesBoolean				- container of booelans for fixed vertices.
	*	\param		[in]	numEdges							- num of edges in the input mesh
	*	\return				T									- edge node matrix.
	*	\since version 0.0.1
	*/
	template <class T>
	T getEdgeNodeMatrix(zMesh &inMesh, vector<bool>& fixedVerticesBoolean, int &numEdges);

	/*! \brief This method computes the Edge Node Matrix for the input graph.
	*
	*	\tparam				T									- Type to work with SpMat and MatrixXd.
	*	\param		[in]	inGraph								- input graph.
	*	\param		[in]	fixedVerticesBoolean				- container of booelans for fixed vertices.
	*	\param		[in]	numEdges							- num of edges in the input graph
	*	\return				T									- edge node matrix.
	*	\since version 0.0.1
	*/
	template <class T>
	T getEdgeNodeMatrix(zGraph &inGraph, vector<bool>& fixedVerticesBoolean, int &numEdges);

	/*! \brief This method computes the sub Matrix of a sparse matrix.
	*
	*	\param		[in]	C									- input sparse matrix.
	*	\param		[in]	nodes								- container of integers.
	*	\return				SpMat								- sub matrix.
	*	\since version 0.0.1
	*/
	SpMat subMatrix( SpMat &C, vector<int> &nodes)
	{
		SpMat C_sub(C.rows(), nodes.size());
		for (int i = 0; i < nodes.size(); i++)C_sub.col(i) = C.col(nodes[i]);
		return C_sub;
	}

	/*! \brief This method computes the sub Matrix of a sparse matrix.
	*
	*	\param		[in]	C									- input sparse matrix.
	*	\param		[in]	nodes								- container of integers.
	*	\return				MatrixXd							- sub matrix.
	*	\since version 0.0.1
	*/
	MatrixXd subMatrix( MatrixXd &X, vector<int> &nodes)
	{
		MatrixXd X_sub(nodes.size(), X.cols());
		for (int i = 0; i < nodes.size(); i++)X_sub.row(i) = X.row(nodes[i]);
		return X_sub;
	}


	/** @}*/

	/** \addtogroup zVault_FDM
	*	\brief Collection of methods for form finding using force density method.
	*	\details Based on Schek, H-J. "The force density method for form finding and computation of general networks." Computer methods in applied mechanics and engineering 3.1 (1974): 115-134. (https://www.sciencedirect.com/science/article/pii/0045782574900450)
	and Linkwitz, K. (2014). Force density method. Shell Structures for Architecture: Form Finding and Optimization, Routledge, Oxon and New York, 59-71.
	*  @{
	*/

	/*! \brief This method computes the form graph/mesh based on the force density method.
	*	
	*	\tparam				T									- Type to work with zGraph and zMesh.
	*	\param		[in]	inHEdataStructure					- input force volume meshes container.
	*	\param		[in]	forceDensities						- input force densities.
	*	\param		[in]	fixedVertices						- input container of fixed vertices.
	*	\param		[in]	load								- input container of loads.
	*	\since version 0.0.1
	*/
	template<class T>
	bool ForceDensityMethod(T &inHEdataStructure, vector<double>& forceDensities, vector<int>& fixedVertices, vector<double> &load);
	

	/** @}*/

	/** \addtogroup zVault_TNA
	*	\brief Collection of methods for form finding using thrust network analysis ( 2D Graphic Statics).
	*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
	*  @{
	*/

	//--------------------------
	//---- FORM AND FORCE DIAGRAM 
	//--------------------------

	/*! \brief This method computes the form mesh based on the thrust netwrok analysis.
	*
	*	\param		[in]	inMesh								- input mesh.
	*	\param		[in]	fixedVertices						- input container of fixed vertices.
	*	\param		[in]	computeEdgeColor					- edge color is computed from vertex color if true.
	*	\return				zMesh								- form mesh.
	*	\since version 0.0.1
	*/
	zMesh createFormMesh(zMesh &inMesh, vector<int> &fixedVertices, bool computeEdgeColor =false)
	{
		zMesh out = duplicateMesh(inMesh);

		for (int i = 0; i < out.vertexPositions.size(); i++)
		{
			out.vertexPositions[i].z = 0;
		}

		for (int i = 0; i < out.numVertices(); i++)
		{
			zColor col;
			out.vertexColors[i] = col;
		}

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			zColor col(1, 1, 1, 1);
			out.vertexColors[fixedVertices[i]] = col;
		}

		if(computeEdgeColor) out.computeEdgeColorfromVertexColor();

		return out;

	}

	/*! \brief This method computes the force mesh based on the thrust netwrok analysis.
	*
	*	\param		[in]	formMesh							- input form mesh.
	*	\param		[in]	fixedVertices						- input container of fixed vertices.
	*	\param		[in]	formEdge_forceEdge					- container storing the corresponding force mesh edge per form mesh edge.
	*	\param		[in]	forceEdge_formEdge					- container storing the corresponding form mesh edge  per force mesh edge.
	*	\param		[in]	rotate90							- rotates force mesh by 90 if true.
	*	\return				zMesh								- force mesh.
	*	\since version 0.0.1
	*/
	zMesh createForceMesh(zMesh & formMesh, vector<bool>& fixedVertices, vector<int> &formEdge_forceEdge, vector<int> &forceEdge_formEdge, bool rotate90 = false)
	{
		zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;


		int n_v = formMesh.numVertices();
		int n_e = formMesh.numEdges();
		int n_f = formMesh.numPolygons();
		

		vector<zVector> faceCenters;
		getCenters(formMesh, zFaceData, faceCenters);
		positions = faceCenters;

		vector<int> formHalfEdge_forceVertex;

		for (int i = 0; i < n_e; i++)
		{
			formHalfEdge_forceVertex.push_back(-1);

			if (formMesh.edges[i].getFace())
			{
				formHalfEdge_forceVertex[i] = formMesh.edges[i].getFace()->getFaceId();
			}
		}

		for (int i = 0; i < n_e; i++)
		{

			if (formHalfEdge_forceVertex[i] == -1)
			{

				// check if both the  vertices of the boundary edge arent fixed

				bool edgeVertsFixed = false;
				if (fixedVertices[formMesh.edges[i].getVertex()->getVertexId()] && fixedVertices[formMesh.edges[i].getSym()->getVertex()->getVertexId()])
					edgeVertsFixed = true;


				// walk on boundary and find all vertices between two fixed vertices
				if (!edgeVertsFixed)
				{
					vector<int>  boundaryEdges;
					vector<int>  boundaryVertices;

					zEdge* start = &formMesh.edges[i];
					zEdge* e = start;

					bool exit = false;


					// walk prev
					do
					{

						if (fixedVertices[e->getSym()->getVertex()->getVertexId()])
						{
							exit = true;
							boundaryVertices.push_back(e->getSym()->getVertex()->getVertexId());
						}

						boundaryEdges.push_back(e->getEdgeId());
						boundaryVertices.push_back(e->getVertex()->getVertexId());

						if (e->getPrev())
						{
							e = e->getPrev();


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
							if (fixedVertices[e->getVertex()->getVertexId()])
							{
								exit = true;
							}

							if (exit) continue;

							if (e->getNext())
							{
								e = e->getNext();
								boundaryVertices.push_back(e->getVertex()->getVertexId());
								boundaryEdges.push_back(e->getEdgeId());

								if (fixedVertices[e->getVertex()->getVertexId()])
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
							newPos += formMesh.vertexPositions[boundaryVertices[j]];						
						}

						newPos /= boundaryVertices.size();
						positions.push_back(newPos);
					}

				}


			}
		}


		for (int i = 0; i < n_v; i++)
		{

			if (!fixedVertices[i])
			{
				vector<int> cEdges;
				formMesh.getConnectedEdges(i, zVertexData, cEdges);

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

	

		out = zMesh(positions, polyCounts, polyConnects);;
		//printf("\n forceMesh: %i %i %i", out.numVertices(), out.numEdges(), out.numPolygons());


		if (rotate90)
		{
			// bounding box
			zVector minBB, maxBB;
			getBounds(out.vertexPositions,minBB, maxBB);

			zVector cen = (maxBB + minBB) * 0.5;
			zVector rotAxis(cen.x, cen.y, 1);

			for (int i = 0; i < out.vertexPositions.size(); i++)
			{
				out.vertexPositions[i].rotateAboutAxis(rotAxis, 90);
			}


		}


		// compute forceEdge_formEdge
		forceEdge_formEdge.clear();
		for (int i = 0; i < out.numEdges(); i++)
		{
			forceEdge_formEdge.push_back(-1);
		}
		
		// compute form edge to force edge	
		formEdge_forceEdge.clear();

		for (int i = 0; i < n_e; i++)
		{
			int v1 = formHalfEdge_forceVertex[i];
			int v2 = (i % 2 == 0) ? formHalfEdge_forceVertex[i + 1] : formHalfEdge_forceVertex[i - 1];

			int eId = -1;
			out.edgeExists(v1, v2, eId);
			formEdge_forceEdge.push_back(eId);	

			if (formEdge_forceEdge[i] != -1)
			{
				forceEdge_formEdge[formEdge_forceEdge[i]] = i;
			}
		}


	

		return out;
	}

	//--------------------------
	//---- HORIZONTAL EQUILIBRIUM
	//--------------------------

	/*! \brief This method checks if the form mesh edges and corresponding force mesh edge are parallel.
	*
	*	\param		[in]	formMesh							- input form mesh.
	*	\param		[in]	forceMesh							- input force mesh.
	*	\param		[in]	form_tensionEdges					- container storing the edges which are in tension.
	*	\param		[in]	formEdge_ForceEdge					- container storing the corresponding force mesh edge  per form mesh edge.
	*	\param		[out]	minDeviation						- stores minimum deviation.
	*	\param		[out]	maxDeviation						- stores maximum deviation.
	*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
	*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
	*	\since version 0.0.1
	*/
	bool checkParallelity(zMesh &formMesh, zMesh &forceMesh, vector<bool>& form_tensionEdges, vector<int> &formEdge_ForceEdge, double &minDeviation, double &maxDeviation, double angleTolerance = 0.001, bool printInfo = false)
	{
		bool out = true;

		minDeviation = 10000;
		maxDeviation = -10000;

		for (int i = 0; i < formMesh.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			int v1_form = formMesh.edges[eId_form].getVertex()->getVertexId();
			int v2_form = formMesh.edges[eId_form].getSym()->getVertex()->getVertexId();

			zVector e_form = formMesh.vertexPositions[v1_form] - formMesh.vertexPositions[v2_form];
			e_form.normalize();


			if (formEdge_ForceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_ForceEdge[i];
				int v1_force = forceMesh.edges[eId_force].getVertex()->getVertexId();
				int v2_force = forceMesh.edges[eId_force].getSym()->getVertex()->getVertexId();

				zVector e_force = forceMesh.vertexPositions[v1_force] - forceMesh.vertexPositions[v2_force];
				e_force.normalize();

				// angle

				double a_i = e_form.angle(e_force);
				if ((form_tensionEdges[eId_form]))a_i = 180 - a_i;

				if (a_i > angleTolerance)
				{
					out = false;
					if (!printInfo) break;
				}

				if (printInfo)
				{
					//printf("\n %i %i %i %i ", v1_form, v2_form, v1_force, v2_force);
					//printf("\n e: %i angle :  %1.2f ", i, a_i);

					if (a_i < minDeviation) minDeviation = a_i;
					if (a_i > maxDeviation) maxDeviation = a_i;
				}

			}
		}

		if (printInfo)
		{
			printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, minDeviation, maxDeviation);
		}



		return out;
	}


	/*! \brief This method computes the  if the form mesh edges and corresponding force mesh edge are parallel.
	*
	*	\param		[in]	formMesh							- input form mesh.
	*	\param		[in]	forceMesh							- input force mesh.
	*	\param		[in]	form_tensionEdges					- container storing the edges which are in tension.
	*	\param		[in]	formEdge_ForceEdge					- container storing the corresponding force mesh edge  per form mesh edge.
	*	\param		[in]	formWeight							- weight of form mesh update. To be between 0 and 1.
	*	\param		[out]	targetEdges_form					- container of target edge for form mesh edges.
	*	\param		[out]	targetEdges_force					- container of target edge for force mesh edges.
	*	\since version 0.0.1
	*/
	void HorizontalEquilibrium_targets(zMesh &formMesh, zMesh &forceMesh, vector<bool>& form_tensionEdges, vector<int> &formEdge_ForceEdge, double formWeight, vector<zVector> &targetEdges_form, vector<zVector> &targetEdges_force)
	{
		targetEdges_form.clear();
		targetEdges_force.clear();

		for (int i = 0; i < forceMesh.numEdges(); i++)
		{
			targetEdges_force.push_back(zVector());
		}

		//formMesh.printMeshInfo();
		//forceMesh.printMeshInfo();

		for (int i = 0; i < formMesh.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			int v1_form = formMesh.edges[eId_form].getVertex()->getVertexId();
			int v2_form = formMesh.edges[eId_form].getSym()->getVertex()->getVertexId();

			zVector e_form = formMesh.vertexPositions[v1_form] - formMesh.vertexPositions[v2_form];
			e_form.normalize();


			if (formEdge_ForceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_ForceEdge[i];
				int v1_force = forceMesh.edges[eId_force].getVertex()->getVertexId();
				int v2_force = forceMesh.edges[eId_force].getSym()->getVertex()->getVertexId();
							   
				zVector e_force = forceMesh.vertexPositions[v1_force] - forceMesh.vertexPositions[v2_force];
				e_force.normalize();

				// target edge 
				zVector e_target = (e_form *formWeight) + (e_force * (1 - formWeight));
				e_target.normalize();

				if (form_tensionEdges[eId_form]) e_target *= -1;

				targetEdges_form.push_back(e_target);
				targetEdges_force[eId_force] = e_target;				
			}

			else
			{
				// target edge 
				zVector e_target = (e_form * 1);
				targetEdges_form.push_back(e_target);
			}


		}
	}

	/*! \brief This method updates the input diagram - form / force mesh.
	*
	*	\param		[in]	diagramMesh							- input form / force mesh to be updated.
	*	\param		[in]	targetEdges							- container of target edges.
	*	\param		[in]	vWeights							- container of vertex weights.
	*	\param		[out]	minMax_diagramEdge					- minimum value of the target edge.
	*	\since version 0.0.1
	*/
	void updateDiagramMesh(zMesh & diagramMesh, vector<zParticle> &diagramParticles,  vector<zVector>& targetEdges, vector<double>& vWeights, double minMax_diagramEdge)
	{
		if (diagramParticles.size() != diagramMesh.vertexActive.size()) fromMESH(diagramParticles, diagramMesh);

		vector<zVector> v_residual;

		vector<double> edgelengths;
		getEdgeLengths(diagramMesh, edgelengths);

		double minEdgeLength, maxEdgeLength;
		minEdgeLength = zMin(edgelengths);
		maxEdgeLength = zMax(edgelengths);

		minEdgeLength = maxEdgeLength * minMax_diagramEdge;

		for (int i = 0; i < diagramMesh.numVertices(); i++)
		{
			vector<int> cEdges;
			diagramMesh.getConnectedEdges(i, zVertexData, cEdges);
			zVector v_i = diagramMesh.vertexPositions[i];

			// compute barycenter per vertex
			zVector b_i;
			for (int j = 0; j < cEdges.size(); j++)
			{

				zVector v_j = diagramMesh.vertexPositions[diagramMesh.edges[cEdges[j]].getVertex()->getVertexId()];

				zVector e_ij = v_i - v_j;
				double len_e_ij = e_ij.length();

				if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
				if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;				

				zVector t_ij = targetEdges[diagramMesh.edges[cEdges[j]].getSym()->getEdgeId()];
				t_ij.normalize();
				
				b_i += (v_j + (t_ij * len_e_ij));

			}

			b_i /= cEdges.size();

			// compute residue force
			zVector r_i = b_i - v_i;
			v_residual.push_back(r_i);

			zVector forceV = v_residual[i] * vWeights[i];
			diagramParticles[i].addForce(forceV);
		}

		// update positions
		for (int i = 0; i < diagramParticles.size(); i++)
		{
			diagramParticles[i].updateParticle(true);			
		}
	}

	/*! \brief This method computes the horizontal equilibrium of the form and force mesh.
	*
	*	\param		[in]	formMesh							- input form mesh.
	*	\param		[in]	forceMesh							- input force mesh.
	*	\param		[in]	formParticles						- container storing the particles of form mesh.
	*	\param		[in]	forceParticles						- container storing the particles of force mesh.
	*	\param		[in]	targetEdges_form					- container of target edges for form mesh edges.
	*	\param		[in]	targetEdges_force					- container of target edges for force mesh edges.
	*	\param		[in]	formWeight							- weight of form mesh update. To be between 0 and 1.
	*	\param		[in]	form_vWeights						- container of vertex weights for form mesh.
	*	\param		[in]	force_vWeights						- container of vertex weights for force mesh.
	*	\param		[in]	minMax_formEdge						- minimum value of the target edge for form mesh.
	*	\param		[in]	minMax_forceEdge					- minimum value of the target edge for force mesh.
	*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
	*	\since version 0.0.1
	*/
	void HorizontalEquilibrium(zMesh &formMesh, zMesh &forceMesh, vector<zParticle>& formParticles, vector<zParticle>& forceParticles, vector<int> &formEdge_ForceEdge, vector<zVector> &targetEdges_form, vector<zVector> &targetEdges_force, vector<double> &form_vWeights, vector<double> &force_vWeights, double formWeight, double minMax_formEdge, double minMax_forceEdge, double angleTolerance)
	{

		if (formWeight != 1)
		{
			updateDiagramMesh(formMesh, formParticles, targetEdges_form, form_vWeights, minMax_formEdge);
		}

		if (formWeight != 0)
		{
			updateDiagramMesh(forceMesh, forceParticles, targetEdges_force, force_vWeights, minMax_forceEdge);
		}
	}


	/** @}*/

	/** @}*/

	/** @}*/

}


#ifndef DOXYGEN_SHOULD_SKIP_THIS

//--------------------------
//---- TEMPLATE SPECIALIZATION DEFINITIONS 
//--------------------------

//---------------//

//---- SpMat specilization for getEdgeNodeMatrix using mesh
template <>
inline SpMat zSpace::getEdgeNodeMatrix(zMesh &inMesh, vector<bool>& fixedVerticesBoolean, int &numEdges)
{
	int n_e = inMesh.numEdges();
	int n_v = inMesh.numVertices();


	SpMat out(numEdges, n_v);
	out.setZero();

	int coefsCounter = 0;

	vector<T> coefs; // 1 for from vertex and -1 for to vertex
						
	for (int i = 0; i < n_e; i += 2)
	{


		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
		{
			coefs.push_back(T(coefsCounter, v1, 1));
			coefs.push_back(T(coefsCounter, v2, -1));

			coefsCounter++;
		}

	}

	out.setFromTriplets(coefs.begin(), coefs.end());


	return out;
}

//---- MatrixXd specilization for getEdgeNodeMatrix using mesh
template <>
inline MatrixXd zSpace::getEdgeNodeMatrix(zMesh &inMesh, vector<bool>& fixedVerticesBoolean, int &numEdges)
{
	int n_e = inMesh.numEdges();
	int n_v = inMesh.numVertices();


	MatrixXd out(numEdges, n_v);
	out.setZero();

	int outEdgesCounter = 0;

	for (int i = 0; i < n_e; i += 2)
	{
		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
		{
			out(outEdgesCounter, v1) = 1;
			out(outEdgesCounter, v2) = -1;
			outEdgesCounter++;
		}



	}


	return out;
}


//---- SpMat specilization for getEdgeNodeMatrix using graph
template <>
inline SpMat zSpace::getEdgeNodeMatrix(zGraph &inGraph, vector<bool>& fixedVerticesBoolean, int &numEdges)
{
	int n_e = inGraph.numEdges();
	int n_v = inGraph.numVertices();


	SpMat out(numEdges, n_v);
	out.setZero();

	int coefsCounter = 0;

	vector<T> coefs; // 1 for from vertex and -1 for to vertex

	for (int i = 0; i < n_e; i += 2)
	{


		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
		{
			coefs.push_back(T(coefsCounter, v1, 1));
			coefs.push_back(T(coefsCounter, v2, -1));

			coefsCounter++;
		}

	}

	out.setFromTriplets(coefs.begin(), coefs.end());


	return out;
}

//---- MatrixXd specilization for getEdgeNodeMatrix using graph
template <>
inline MatrixXd zSpace::getEdgeNodeMatrix(zGraph &inGraph, vector<bool>& fixedVerticesBoolean, int &numEdges)
{
	int n_e = inGraph.numEdges();
	int n_v = inGraph.numVertices();


	MatrixXd out(numEdges, n_v);
	out.setZero();

	int outEdgesCounter = 0;

	for (int i = 0; i < n_e; i += 2)
	{
		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
		{
			out(outEdgesCounter, v1) = 1;
			out(outEdgesCounter, v2) = -1;
			outEdgesCounter++;
		}



	}


	return out;
}

//---------------//

//---- graph specilization for ForceDensityMethod
template <>
inline bool zSpace::ForceDensityMethod(zGraph &inGraph, vector<double>& forceDensities, vector<int>& fixedVertices, vector<double> &load)
{
	zHEData type = zVertexData;


	int n_e = inGraph.numEdges();
	int n_v = inGraph.numVertices();
	int numEdges = floor(n_e*0.5);

	// FIXED VERTS BOOLEAN
	vector<bool> fixedVertsBoolean;
	for (int i = 0; i < n_v; i++) fixedVertsBoolean.push_back(false);
	for (int i = 0; i < fixedVertices.size(); i++) fixedVertsBoolean[fixedVertices[i]] = true;

	for (int i = 0; i < n_e; i += 2)
	{
		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();

		if (fixedVertsBoolean[v1] && fixedVertsBoolean[v2]) numEdges--;

	}

	// POSITION MATRIX
	MatrixXd X(inGraph.numVertices(), 3);
	for (int i = 0; i < inGraph.numVertices(); i++)
	{
		X(i, 0) = inGraph.vertexPositions[i].x;
		X(i, 1) = inGraph.vertexPositions[i].y;
		X(i, 2) = inGraph.vertexPositions[i].z;
	};

	// EDGE NODE MATRIX
	SpMat C = getEdgeNodeMatrix<SpMat>(inGraph, fixedVertsBoolean, numEdges);

	// FORCE DENSITY VECTOR
	VectorXd q(numEdges);
	int FD_EdgesCounter = 0;

	for (int i = 0; i < n_e; i += 2)
	{

		int v1 = inGraph.edges[i].getVertex()->getVertexId();
		int v2 = inGraph.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVertsBoolean[v1] || !fixedVertsBoolean[v2])
		{
			q[FD_EdgesCounter] = forceDensities[i];
			FD_EdgesCounter++;
		}

	}


	//printf("\n Force Densities: \n");
	//cout << endl << q;

	Diag Q = q.asDiagonal();

	// LOAD VECTOR
	VectorXd p(inGraph.numVertices());
	//p.setConstant(load);
	for (int i = 0; i < inGraph.numVertices(); i++)
	{
		p[i] = load[i];
	}


	MatrixXd P(inGraph.numVertices(), 3);
	P.setConstant(0.0);
	P.col(2) = p.col(0);

	// SUB MATRICES
	vector<int> freeVertices;

	for (int j = 0; j < fixedVertsBoolean.size(); j++)
	{
		if (!fixedVertsBoolean[j])
		{
			freeVertices.push_back(j);
			//printf("\n j: %i ", j);
		}

	}

	SpMat Cn = subMatrix( C, freeVertices);
	SpMat Cf = subMatrix( C, fixedVertices);
	MatrixXd Xf = subMatrix( X, fixedVertices);
	MatrixXd Pn = subMatrix( P, freeVertices);

	//cout << "Cn: \n" << Cn << endl;
	//cout << "Cf: \n" << Cf << endl;

	SpMat Cn_transpose;
	Cn_transpose = Cn.transpose();

	//CHOLESKY DECOMPOSITION

	SpMat Dn = Cn_transpose * Q * Cn;
	SpMat Df = Cn_transpose * Q * Cf;

	MatrixXd  B = Pn - Df * Xf;

	// solve
	MatrixXd Xn;

	SimplicialLLT< SpMat > solver; // sparse cholesky solver
	solver.compute(Dn); // compute cholesky factors

	if (solver.info() != Eigen::Success)
		return false;


	Xn = solver.solve(B); // solve AX = B ;
	if (solver.info() != Eigen::Success)
		return false;

	cout << endl << Xn;
	// POSITIONS OF NON FIXED VERTICES
	for (int i = 0; i < freeVertices.size(); i++)
	{
		int id = freeVertices[i];

		zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
		inGraph.vertexPositions[id] = pos;

	}

	return true;
}


//---- mesh specilization for ForceDensityMethod
template <>
inline bool zSpace::ForceDensityMethod(zMesh &inMesh, vector<double>& forceDensities, vector<int>& fixedVertices, vector<double> &load)
{
	zHEData type = zVertexData;


	int n_e = inMesh.numEdges();
	int n_v = inMesh.numVertices();
	int numEdges = floor(n_e*0.5);

	// FIXED VERTS BOOLEAN
	vector<bool> fixedVertsBoolean;
	for (int i = 0; i < n_v; i++) fixedVertsBoolean.push_back(false);
	for (int i = 0; i < fixedVertices.size(); i++) fixedVertsBoolean[fixedVertices[i]] = true;

	for (int i = 0; i < n_e; i += 2)
	{
		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();

		if (fixedVertsBoolean[v1] && fixedVertsBoolean[v2]) numEdges--;

	}

	// POSITION MATRIX
	MatrixXd X(inMesh.numVertices(), 3);
	for (int i = 0; i < inMesh.numVertices(); i++)
	{
		X(i, 0) = inMesh.vertexPositions[i].x;
		X(i, 1) = inMesh.vertexPositions[i].y;
		X(i, 2) = inMesh.vertexPositions[i].z;
	};

	// EDGE NODE MATRIX
	SpMat C = getEdgeNodeMatrix<SpMat>(inMesh, fixedVertsBoolean, numEdges);

	// FORCE DENSITY VECTOR
	VectorXd q(numEdges);
	int FD_EdgesCounter = 0;

	for (int i = 0; i < n_e; i += 2)
	{

		int v1 = inMesh.edges[i].getVertex()->getVertexId();
		int v2 = inMesh.edges[i + 1].getVertex()->getVertexId();


		if (!fixedVertsBoolean[v1] || !fixedVertsBoolean[v2])
		{
			q[FD_EdgesCounter] = forceDensities[i];
			FD_EdgesCounter++;
		}

	}


	//printf("\n Force Densities: \n");
	//cout << endl << q;

	Diag Q = q.asDiagonal();

	// LOAD VECTOR
	VectorXd p(inMesh.numVertices());
	//p.setConstant(load);
	for (int i = 0; i < inMesh.numVertices(); i++)
	{
		p[i] = load[i];
	}


	MatrixXd P(inMesh.numVertices(), 3);
	P.setConstant(0.0);
	P.col(2) = p.col(0);

	// SUB MATRICES
	vector<int> freeVertices;

	for (int j = 0; j < fixedVertsBoolean.size(); j++)
	{
		if (!fixedVertsBoolean[j])
		{
			freeVertices.push_back(j);
			//printf("\n j: %i ", j);
		}

	}

	SpMat Cn = subMatrix( C, freeVertices);
	SpMat Cf = subMatrix( C, fixedVertices);
	MatrixXd Xf = subMatrix( X, fixedVertices);
	MatrixXd Pn = subMatrix( P, freeVertices);

	//cout << "Cn: \n" << Cn << endl;
	//cout << "Cf: \n" << Cf << endl;

	SpMat Cn_transpose;
	Cn_transpose = Cn.transpose();

	//CHOLESKY DECOMPOSITION

	SpMat Dn = Cn_transpose * Q * Cn;
	SpMat Df = Cn_transpose * Q * Cf;

	MatrixXd  B = Pn - Df * Xf;

	// solve
	MatrixXd Xn;

	SimplicialLLT< SpMat > solver; // sparse cholesky solver
	solver.compute(Dn); // compute cholesky factors

	if (solver.info() != Eigen::Success)
		return false;


	Xn = solver.solve(B); // solve AX = B ;
	if (solver.info() != Eigen::Success)
		return false;

	cout << endl << Xn;
	// POSITIONS OF NON FIXED VERTICES
	for (int i = 0; i < freeVertices.size(); i++)
	{
		int id = freeVertices[i];

		zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
		inMesh.vertexPositions[id] = pos;

	}

	return true;
}


#endif /* DOXYGEN_SHOULD_SKIP_THIS */