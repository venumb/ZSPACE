#pragma once

#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>
#include <headers/api/functionsets/zFnParticle.h>


typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> Tri;
typedef DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> Diag;

namespace zSpace
{



	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \class zTsVault
	*	\brief A formfinding tool set class using both iterative and linear algebra methods.
	*	\tparam				T			- Type to work with zObjMesh or zObjGraph.
	*	\tparam				U			- Type to work with zFnMesh or zFnGraph.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	template<typename T, typename U>
	class zTsVault
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- RESULT ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to result Object  */
		T *resultObj;

		/*!	\brief container of  result particle objects  */
		vector<zObjParticle> resultParticlesObj;

		/*!	\brief container of result particle function set  */
		vector<zFnParticle> fnResultParticles;

		/*!	\brief container storing the update weights of the result diagram.  */
		vector<double> resultVWeights;

		/*!	\brief container storing the vertex mass of the result diagram.  */
		vector<double> resultVThickness;

		/*!	\brief container storing the vertex mass of the result diagram.  */
		vector<double> resultVMass;

		//--------------------------
		//---- FORM DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to form Object  */
		T *formObj;

		/*!	\brief container of  form particle objects  */
		vector<zObjParticle> formParticlesObj;

		/*!	\brief container of form particle function set  */
		vector<zFnParticle> fnFormParticles;

		/*!	\brief container storing the update weights of the form diagram.  */
		vector<double> formVWeights;

		//--------------------------
		//---- FORCE DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to force Object  */
		zObjMesh *forceObj;

		/*!	\brief container of  force particle objects  */
		vector<zObjParticle> forceParticlesObj;

		/*!	\brief container of force particle function set  */
		vector<zFnParticle> fnForceParticles;

		/*!	\brief container storing the update weights of the force diagram.  */
		vector<double> forceVWeights;

		//--------------------------
		//---- ATTRIBUTES
		//--------------------------

		/*!	\brief container of force densities  */
		vector<double> forceDensities;

		/*!	\brief container of indicies of fixed vertices  */
		vector<int> fixedVertices;		

		/*!	\brief container of booleans of fixed vertices  */
		vector<bool> fixedVerticesBoolean;

		/*!	\brief container storing the corresponding force edge per form edge.  */
		vector<int> forceEdge_formEdge;

		/*!	\brief container storing the corresponding form edge per force edge.  */
		vector<int> formEdge_forceEdge;

		/*!	\brief container storing the form edges in tension.  */
		vector<bool> form_tensionEdges;

		/*!	\brief container storing the force edges in tension.  */
		vector<bool> force_tensionEdges;

		/*!	\brief container storing the force edges in tension.  */
		vector<bool> result_tensionEdges;

		/*!	\brief container storing the horizontal equilibrium target for form edges.  */
		vector<zVector> targetEdges_form;

		/*!	\brief container storing the horizontal equilibrium target for force edges.  */
		vector<zVector> targetEdges_force;

	public:
		
		/*!	\brief color domain - min for tension edges and max for compression edges.  */
		zDomainColor elementColorDomain = zDomainColor(zColor(0.5, 0, 0.2, 1), zColor(0, 0.2, 0.5, 1));		

		/*!	\brief result function set  */
		U fnResult;

		/*!	\brief form function set  */
		U fnForm;

		/*!	\brief force function set  */
		zFnMesh fnForce;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsVault() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_resultObj			- input result object.
		*	\since version 0.0.2
		*/
		zTsVault(T &_resultObj);

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_resultObj			- input result object.
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_forceObj			- input force object.
		*	\since version 0.0.2
		*/
		zTsVault(T &_resultObj, T &_formObj, zObjMesh &_forceObj);


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsVault() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the result geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createResultfromFile(string path, zFileTpye type);

		/*! \brief This method creates the force geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createForcefromFile(string path, zFileTpye type);

		/*! \brief This method creates the form geometry from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createFormfromFile(string path, zFileTpye type);

		/*! \brief This method computes the form diagram from the result.
		*
		*	\since version 0.0.2
		*/
		void createFormFromResult();

		/*! \brief This method computes the form diagram from the force diagram. Works only when form is to a be a graph.
		*
		*	\param		[in]	excludeBoundary			- true if boundary faces are to be ignored.
		*	\param		[in]	PlanarMesh				- true if input mesh is planar.
		*	\param		[in]	rotate90				- true if dual mesh is to be roatated by 90.
		*	\since version 0.0.2
		*/
		void createFormFromForce(bool excludeBoundary = false, bool PlanarForceMesh = false, bool rotate90 = false);

		/*! \brief This method computes the force diagram from the form diagram. Works only when form is a mesh.
		*
		*	\since version 0.0.2
		*/
		void createForceFromForm(bool rotate90);

		/*! \brief This method computes the result from the form.
		*
		*	\since version 0.0.2
		*/
		void createResultFromForm();
				

		//--------------------------
		//---- FDM METHODS
		//--------------------------

		

		/*! \brief This method computes the result based on the force density method.
		*
		*	\details Based on Schek, H-J. "The force density method for form finding and computation of general networks." Computer methods in applied mechanics and engineering 3.1 (1974): 115-134. (https://www.sciencedirect.com/science/article/pii/0045782574900450)
			and Linkwitz, K. (2014). Force density method. Shell Structures for Architecture: Form Finding and Optimization, Routledge, Oxon and New York, 59-71.
		*	\return				bool								- true if solution is found.
		*	\since version 0.0.2
		*/
		bool forceDensityMethod();

	

		//--------------------------
		//---- TNA METHODS
		//--------------------------
				

		/*! \brief This method computes the horizontal equilibrium of the form and force diagram.
		*
		*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
		*	\param		[in]	computeTargets						- true if the targets fro horizontal equiibrium have to be computed.
		*	\param		[in]	formWeight							- weight of form mesh update. To be between 0 and 1.
		*	\param		[in]	dT									- integration timestep.
		*	\param		[in]	type								- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations						- number of iterations to run.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	minMax_formEdge						- minimum value of the target edge for form mesh.
		*	\param		[in]	minMax_forceEdge					- minimum value of the target edge for force mesh.
		*	\param		[in]	printInfo							- prints information of minimum and maximum deviation if true.
		*	\param		[in]	colorEdges							- colors diagram edges based on deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel.
		*	\since version 0.0.2
		*/
		bool equilibriumHorizontal(bool &computeTargets, double formWeight, double dT, zIntergrationType type, int numIterations = 1, double angleTolerance = 0.001, double minMax_formEdge = 0.1, double minMax_forceEdge = 0.1, bool colorEdges = false, bool printInfo = false)
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
			zDomainDouble dev;
			bool out = checkHorizontalParallelity(dev, angleTolerance, colorEdges, printInfo);

			if (out) 
			{
				
				setelementColorDomain(zFormDiagram);
				setelementColorDomain(zForceDiagram);
			}

			return out;
		}

		/*! \brief This method computes the vertical equilibrium of the result diagram using iterative method.
		*
		*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
		*	\param		[in]	dT									- integration timestep.
		*	\param		[in]	type								- integration type - zEuler or zRK4.
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\param		[in]	tolerance							-  tolerance for force check.		
		*	\return				bool								- true if the all the forces  per vertex add up to zero or below tolerance.
		*	\since version 0.0.2
		*/
		bool equilibriumVertical(double dT, zIntergrationType type, double forceDiagramScale, double tolerance = 0.001)
		{
			bool out = updateResultDiagram(forceDiagramScale, dT, type);

			return out;
		}

		/*! \brief This method computes the vertical equilibrium of the result diagram using linear algebra method.
		*
		*	\details Based on Block, Philippe, and John Ochsendorf. "Thrust network analysis: A new methodology for three-dimensional equilibrium." Journal of the International Association for shell and spatial structures 48.3 (2007): 167-173.
		*	\param		[in]	computeForceDensitities				- true if the force densitities from the diagrams have to be computed.
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\since version 0.0.2
		*/
		bool equilibriumVertical(bool &computeForceDensitities, double forceDiagramScale);

		/** @}*/

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/*! \brief This method sets the element color domain. (min - tensionColor , max - compressionColor)
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\since version 0.0.2
		*/
		void setElementColorDomain(zDomainColor &colDomain)
		{
			elementColorDomain = colDomain;
		}
		
		/*! \brief This method sets anchor points of the geometry.
		*
		*	\param		[in]	type						- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_constraintVertices			- container of vertex indices to be fixed. If empty the boundary(mesh) or valence 1(graph) vertices are fixed.
		*	\since version 0.0.2
		*/
		void setConstraints(zDiagramType type = zResultDiagram, const vector<int> &_constraintVertices = vector<int>());

		/*! \brief This method appends the vertex indicies in the container as anchor points of the geometry.
		*
		*	\param		[in]	_constraintVertices			- container of vertex indices to be fixed.
		*	\since version 0.0.2
		*/
		void appendConstraints(vector<int> &_constraintVertices)
		{
			for (int i = 0; i < _constraintVertices.size(); i++)
			{
				bool checkRepeat = coreUtils.checkRepeatElement(_constraintVertices[i], fixedVertices);
				if (!checkRepeat) fixedVertices.push_back(_constraintVertices[i]);
			}

			for (int i = 0; i < _constraintVertices.size(); i++)
			{
				if (_constraintVertices[i] >= 0 && _constraintVertices[i] < fixedVerticesBoolean.size()) fixedVerticesBoolean[_constraintVertices[i]] = true;
			}
		}

		/*! \brief This method sets force density of all the edges to the input value.
		*
		*	\param		[in]	fDensity			- force density value.
		*	\since version 0.0.2
		*/
		void setForceDensity(double fDensity);

		/*! \brief This method sets force density of the edges with the input container of values.
		*
		*	\param		[in]	fDensities			- container of force density values.
		*	\since version 0.0.2
		*/
		void setForceDensities(vector<double> &fDensities);

		/*! \brief This method sets the force densities edges based on form and force diagrams.
		*
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\since version 0.0.2
		*/
		void setForceDensitiesFromDiagrams(double forceDiagramScale);	


		/*! \brief This method sets tension edges of the form diagram.
		*
		*	\param		[in]	type					- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_tensionEdges			- container of edge indices to be made tension. If empty all edges are non tension.
		*	\since version 0.0.2
		*/
		void setTensionEdges(zDiagramType type = zResultDiagram, const vector<int> &_tensionEdges = vector<int>());

		/*! \brief This method appends the edges in the container as tension edge of the geometry.
		*
		*	\param		[in]	type					- diagram type - work with  zResultDiagram or zFormDiagram.
		*	\param		[in]	_tensionEdges			- container of edge indices to be made tension.
		*	\since version 0.0.2
		*/
		void appendTensionEdges(zDiagramType type , vector<int> &_tensionEdges)
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

		/*! \brief This method sets the force tension edges based on form tension edges.
		*
		*	\since version 0.0.2
		*/
		void setForceTensionEdgesfromForm()
		{
			force_tensionEdges.clear();

			for (int i = 0; i < fnForce.numEdges(); i++)
			{
				force_tensionEdges.push_back(false);
			}

			for (int i = 0; i < fnForm.numEdges(); i++)
			{
				if (formEdge_forceEdge[i] != -1 && form_tensionEdges[i])
				{
					force_tensionEdges[formEdge_forceEdge[i]] = true;;

					int symEdge = fnForce.getSymIndex(formEdge_forceEdge[i]);
					force_tensionEdges[symEdge] = true;;
					
				}
			}

			setelementColorDomain(zForceDiagram);
		}

		/*! \brief This method sets the result tension edges based on form tension edges.
		*
		*	\since version 0.0.2
		*/
		void setResultTensionEdgesfromForm()
		{
			result_tensionEdges.clear();
			result_tensionEdges = form_tensionEdges;

			
			setelementColorDomain(zResultDiagram);
		}

		/*! \brief This method sets the form tension edges based on result tension edges.
		*
		*	\since version 0.0.2
		*/
		void setFormTensionEdgesfromResult()
		{
			form_tensionEdges.clear();
			form_tensionEdges = result_tensionEdges ;

			setelementColorDomain(zFormDiagram);
			
		}

		/*! \brief This method sets the form tension edges based on force tension edges.
		*
		*	\since version 0.0.2
		*/
		void setFormTensionEdgesfromForce()
		{
			form_tensionEdges.clear();

			for (int i = 0; i < fnForm.numEdges(); i++)
			{
				form_tensionEdges.push_back(false);
			}

			for (int i = 0; i < fnForce.numEdges(); i++)
			{
				if (forceEdge_formEdge[i] != -1 && force_tensionEdges[i])
				{
					form_tensionEdges[forceEdge_formEdge[i]] = true;;

					int symEdge = fnForm.getSymIndex(forceEdge_formEdge[i]);
					form_tensionEdges[symEdge] = true;;

				}
			}

			setelementColorDomain(zFormDiagram);

		}

		/*! \brief This method sets the compression or tension colors to the edges of the input diagram.
		*
		*	\param		[in]	type					- diagram type . works with zFormDiagram/ zForceDiagram/ zResultDiagram.
		*	\since version 0.0.2
		*/
		void setelementColorDomain(zDiagramType type)
		{
			if (type == zFormDiagram)
			{
				for (int i = 0; i < form_tensionEdges.size(); i++)
				{
					if (form_tensionEdges[i]) fnForm.setEdgeColor(i, elementColorDomain.min);
					else fnForm.setEdgeColor(i, elementColorDomain.max);
				}
			}
			else if (type == zResultDiagram)
			{
				for (int i = 0; i < result_tensionEdges.size(); i++)
				{
					if (result_tensionEdges[i]) fnResult.setEdgeColor(i, elementColorDomain.min);
					else fnResult.setEdgeColor(i, elementColorDomain.max);
				}
			}
			else if (type == zForceDiagram)
			{
				for (int i = 0; i < force_tensionEdges.size(); i++)
				{
					if (force_tensionEdges[i]) fnForce.setEdgeColor(i, elementColorDomain.min);
					else fnForce.setEdgeColor(i, elementColorDomain.max);
				}
				

			}
			else throw std::invalid_argument(" invalid diagram type.");
		}
				

		/*! \brief This method sets the vertex update weights for each vertex of the input diagram type.
		*
		*	\param		[in]	type							- zFormDiagram or zForceDiagram or zResultDiagram.
		*	\since version 0.0.2
		*/
		void setVertexWeights(zDiagramType type, const vector<double> &vWeights = vector<double>());

		/*! \brief This method sets the result vertex update weights for each vertex of the input diagram type based on the constraints.
		*
		*	\since version 0.0.2
		*/
		void setVertexWeightsfromConstraints(zDiagramType type = zResultDiagram)
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

		/*! \brief This method sets thickness of all the result vertices to the input value.
		*
		*	\param		[in]	thickness			- thickness value.
		*	\since version 0.0.2
		*/
		void setVertexThickness(double thickness);

		/*! \brief This method sets thickness of all the result vertices to the input container of values.
		*
		*	\param		[in]	thickness			- container of thickness values.
		*	\since version 0.0.2
		*/
		void setVertexThickness(vector<double> &thickness);

		/*! \brief This method sets vertex mass of all the result vertices to the input value.
		*
		*	\param		[in]	mass			- mass value.
		*	\since version 0.0.2
		*/
		void setVertexMass(double mass);

		/*! \brief This method sets vertex mass of all the result vertices to the input container of values.
		*
		*	\param		[in]	mass			- container of mass values.
		*	\since version 0.0.2
		*/
		void setVertexMass(vector<double> &mass);

		/*! \brief This method sets vertex mass of all the result vertices based on vertex tributary area. Works only on mesh result diagram.
		*
		*	\since version 0.0.2
		*/
		void setVertexMassfromVertexArea();

		

		//--------------------------
		//--- GET METHODS 
		//--------------------------
		

		/*! \brief This method gets the corresponding force diagram edge for the input form diagram indexed edge.
		*
		*	\param		[in]	formEdgeindex			- form diagram edge index.
		*	\return				int						- correponding force edge index if it exists, else -1.
		*	\since version 0.0.2
		*/
		int getCorrespondingForceEdge(int formEdgeindex)
		{
			if (formEdgeindex > formEdge_forceEdge.size()) throw std::invalid_argument(" error: index out of bounds.");

			return formEdge_forceEdge[formEdgeindex];
		}

		/*! \brief This method gets the corresponding form diagram edge for the input force diagram indexed edge.
		*
		*	\param		[in]	forceEdgeindex			- force diagram edge index.
		*	\return				int						- correponding form edge index if it exists, else -1.
		*	\since version 0.0.2
		*/
		int getCorrespondingFormEdge(int forceEdgeindex)
		{
			if (forceEdgeindex > forceEdge_formEdge.size()) throw std::invalid_argument(" error: index out of bounds.");

			return forceEdge_formEdge[forceEdgeindex];
		}

		/*! \brief This method gets the horizontal equilibrium target vectors for the input diagram type.
		*
		*	\param		[in]	type					- diagram type . works with zFormDiagram and zForceDiagram.
		*	\return				targets					- container of tearget vectors per half edge.
		*	\since version 0.0.2
		*/
		void getHorizontalTargetEdges(zDiagramType type, vector<zVector> &targets)
		{
			if (type == zFormDiagram) targets = targetEdges_form;
			
			else if (type == zForceDiagram) targets = targetEdges_force;

			else throw std::invalid_argument(" invalid diagram type.");
		}
		
		/*! \brief This method gets the force vectors for the result diagram.
		*
		*	\param		[out]	forces					- container of forces per vertex.
		*	\since version 0.0.2
		*/
		void getForces(vector<zVector> &forces);

		/*! \brief This method gets the force vectors for the result diagram using gradient descent.
		*
		*	\param		[out]	forces					- container of forces per vertex.
		*	\since version 0.0.2
		*/
		void getForces_GradientDescent(vector<zVector> &forces);
	
	
		//--------------------------
		//---- UTILITY METHODS 
		//--------------------------

		
		void translateForceDiagram( double value)
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

	protected:
		

		/*! \brief This method computes the Edge Node Matrix for the input mesh.
		*
		*	\tparam				V									- Type to work with SpMat and MatrixXd.
		*	\param		[in]	numCols								- number of columns in the out matrix.
		*	\return				V									- edge node matrix.
		*	\since version 0.0.2
		*/
		template <class V>
		V getEdgeNodeMatrix(int numCols);


		/*! \brief This method computes the sub Matrix of a sparse matrix.
		*
		*	\param		[in]	C									- input sparse matrix.
		*	\param		[in]	nodes								- container of integers.
		*	\return				SpMat								- sub matrix.
		*	\since version 0.0.2
		*/
		SpMat subMatrix(SpMat &C, vector<int> &nodes)
		{
			SpMat C_sub(C.rows(), nodes.size());
			for (int i = 0; i < nodes.size(); i++)C_sub.col(i) = C.col(nodes[i]);
			return C_sub;
		}

		/*! \brief This method computes the sub Matrix of a matrix.
		*
		*	\param		[in]	C									- input sparse matrix.
		*	\param		[in]	nodes								- container of integers.
		*	\return				MatrixXd							- sub matrix.
		*	\since version 0.0.2
		*/
		MatrixXd subMatrix(MatrixXd &X, vector<int> &nodes)
		{
			MatrixXd X_sub(nodes.size(), X.cols());
			for (int i = 0; i < nodes.size(); i++)X_sub.row(i) = X.row(nodes[i]);
			return X_sub;
		}

		/*! \brief This method computes the targets for the form and force edges.
		*
		*	\param		[in]	formWeight							- weight of form diagram update. To be between 0 and 1.
		*	\since version 0.0.2
		*/
		void getHorizontalTargets(double formWeight);

		/*! \brief This method if the form mesh edges and corresponding force mesh edge are parallel.
		*
		*	\param		[out]	deviation							- deviation domain.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel or within tolerance.
		*	\since version 0.0.2
		*/
		bool checkHorizontalParallelity(zDomainDouble &deviation, double angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false);

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1);

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations to run.
		*	\since version 0.0.2
		*/
		void updateForceDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1)
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

			vector<int> cEdges;
			
			zVector* positions = fnForce.getRawVertexPositions();

			for (int k = 0; k < numIterations; k++)
			{
				for (int i = 0; i < fnForce.numVertices(); i++)
				{
					cEdges.clear();
					fnForce.getConnectedEdges(i, zVertexData, cEdges);

					zVector v_i = positions[i];

					// compute barycenter per vertex
					zVector b_i(0, 0, 0);
					for (int j = 0; j < cEdges.size(); j++)
					{

						zVector v_j = positions[fnForce.getEndVertexIndex(cEdges[j])];

						zVector e_ij = v_i - v_j;
						double len_e_ij = e_ij.length();

						if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
						if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

						zVector t_ij = targetEdges_force[fnForce.getSymIndex(cEdges[j])];
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

		/*! \brief This method add the axial force to the  result diagram.
		*
		*	\param		[in]	forceDiagramScale					- scale of force diagram.
		*	\since version 0.0.2
		*/
		void addAxialForces(double forceDiagramScale);

		/*! \brief This method add the dead load force to the  result diagram.
		*
		*	\since version 0.0.2
		*/
		void addDeadLoadForces();

		/*! \brief This method updates the result diagram.
		*
		*	\param		[in]	forceDiagramScale			- scale of force diagram.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	angleTolerance				- tolerance for force.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool updateResultDiagram(double forceDiagramScale, double dT, zIntergrationType type, double tolerance = 0.001);
			   		

	};
	   

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \typedef zTsMeshVault
	*	\brief A vault object for meshes.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsVault<zObjMesh, zFnMesh> zTsMeshVault;

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup z2DGS
	*	\brief tool sets for 2D graphic statics.
	*  @{
	*/

	/*! \typedef zTsGraphVault
	*	\brief A vault object for graphs.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	typedef zTsVault<zObjGraph, zFnGraph> zTsGraphVault;
	

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- graph specilization for zTsVault overloaded constructor 
	template<>
	inline zTsVault<zObjGraph, zFnGraph>::zTsVault(zObjGraph & _resultObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnGraph(_resultObj);
	}

	//---- mesh specilization for zTsVault overloaded constructor 
	template<>
	inline zTsVault<zObjGraph, zFnGraph>::zTsVault(zObjGraph & _resultObj, zObjGraph &_formObj, zObjMesh &_forceObj)
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
	inline zTsVault<zObjMesh, zFnMesh>::zTsVault(zObjMesh & _resultObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnMesh(_resultObj);
	}

	//---- mesh specilization for zTsVault overloaded constructor 
	template<>
	inline zTsVault<zObjMesh, zFnMesh>::zTsVault(zObjMesh & _resultObj, zObjMesh &_formObj, zObjMesh &_forceObj)
	{
		resultObj = &_resultObj;
		fnResult = zFnMesh(_resultObj);

		formObj = &_formObj;
		fnForm = zFnMesh(_formObj);

		forceObj = &_forceObj;
		fnForce = zFnMesh(_forceObj);

	}

	//---------------//

	//---- SpMat specilization for getEdgeNodeMatrix using graph
	template<>
	template<>
	inline SpMat zTsVault<zObjGraph, zFnGraph>::getEdgeNodeMatrix(int numCols)
	{
		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();



		SpMat out(numCols, n_v);
		out.setZero();

		int coefsCounter = 0;

		vector<Tri> coefs; // 1 for from vertex and -1 for to vertex

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				coefs.push_back(Tri(coefsCounter, v1, 1));
				coefs.push_back(Tri(coefsCounter, v2, -1));

				coefsCounter++;
			}

		}

		out.setFromTriplets(coefs.begin(), coefs.end());

		return out;

	}

	//---- SpMat specilization for getEdgeNodeMatrix using mesh
	template<>
	template<>
	inline SpMat zTsVault<zObjMesh, zFnMesh>::getEdgeNodeMatrix(int numCols)
	{
		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();



		SpMat out(numCols, n_v);
		out.setZero();

		int coefsCounter = 0;

		vector<Tri> coefs; // 1 for from vertex and -1 for to vertex

		for (int i = 0; i < n_e; i += 2)
		{


			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				coefs.push_back(Tri(coefsCounter, v1, 1));
				coefs.push_back(Tri(coefsCounter, v2, -1));

				coefsCounter++;
			}

		}

		out.setFromTriplets(coefs.begin(), coefs.end());

		return out;

	}

	//---------------//

	//---- MatrixXd specilization for getEdgeNodeMatrix using graph
	template<>
	template<>
	inline MatrixXd zTsVault<zObjGraph, zFnGraph>::getEdgeNodeMatrix(int numCols)
	{
		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();


		MatrixXd out(numCols, n_v);
		out.setZero();

		int outEdgesCounter = 0;

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				out(outEdgesCounter, v1) = 1;
				out(outEdgesCounter, v2) = -1;
				outEdgesCounter++;
			}

		}


		return out;

	}

	//---- MatrixXd specilization for getEdgeNodeMatrix using mesh
	template<>
	template<>
	inline MatrixXd zTsVault<zObjMesh, zFnMesh>::getEdgeNodeMatrix(int numCols)
	{
		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e * 0.5);

		MatrixXd out(numCols, n_v);
		out.setZero();

		int outEdgesCounter = 0;

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];


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

	//---- graph specilization for setFixedVertices
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setConstraints(zDiagramType type, const vector<int>& _fixedVertices)
	{
		if (type == zResultDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();
				
				for (int i = 0; i < fnResult.numVertices(); i++)
				{
					if (fnResult.checkVertexValency(i, 1))
					{
						fixedVertices.push_back(i);
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

				fnResult.setVertexColor(fixedVertices[i], zColor());
				fixedVerticesBoolean[fixedVertices[i]] = true;
			}

		}

		else if (type == zFormDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (int i = 0; i < fnForm.numVertices(); i++)
				{
					if (fnForm.checkVertexValency(i, 1)) fixedVertices.push_back(i);
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
				fnForm.setVertexColor(fixedVertices[i], zColor());

				fixedVerticesBoolean[fixedVertices[i]] = true;
			}
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}


	//---- mesh specilization for setFixedVertices
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setConstraints(zDiagramType type, const vector<int>& _fixedVertices)
	{


		if (type == zResultDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (int i = 0; i < fnResult.numVertices(); i++)
				{
					if (fnResult.onBoundary(i, zVertexData))
					{
						fixedVertices.push_back(i);
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

				fnResult.setVertexColor(fixedVertices[i], zColor());
				fixedVerticesBoolean[fixedVertices[i]] = true;			
			}

		}

		else if (type == zFormDiagram)
		{
			if (_fixedVertices.size() == 0)
			{
				fixedVertices.clear();

				for (int i = 0; i < fnForm.numVertices(); i++)
				{
					if (fnForm.onBoundary(i, zVertexData)) fixedVertices.push_back(i);
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
				fnForm.setVertexColor(fixedVertices[i], zColor());

				fixedVerticesBoolean[fixedVertices[i]] = true;
			}
		}

		else throw std::invalid_argument(" invalid diagram type.");

	}

	//---------------//

	//---- graph specilization for setForceDensity
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setForceDensity(double fDensity)
	{
		forceDensities.clear();

		for (int i = 0; i < fnResult.numEdges(); i++)
		{
			forceDensities.push_back(fDensity);
		}
	}

	//---- mesh specilization for setForceDensity
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setForceDensity(double fDensity)
	{
		forceDensities.clear();

		for (int i = 0; i < fnResult.numEdges(); i++)
		{
			forceDensities.push_back(fDensity);
		}
	}

	//---------------//

	//---- graph specilization for setForceDensities
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setForceDensities(vector<double> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of graph half edges.");

		forceDensities = fDensities;
	}

	
	//---- mesh specilization for setForceDensities
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setForceDensities(vector<double> &fDensities)
	{
		if (fDensities.size() != fnResult.numEdges()) throw std::invalid_argument("size of fDensities contatiner is not equal to number of mesh half edges.");

		forceDensities = fDensities;
	}

	//---------------//

	//---- graph specilization for setForceDensitiesFromDiagrams
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setForceDensitiesFromDiagrams(double forceDiagramScale)
	{
		forceDensities.clear();

		for (int i = 0; i < fnForm.numEdges(); i += 2)
		{
			// form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			double e_form_len = e_form.length();

			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[eId_form];
				zVector e_force = fnForce.getEdgeVector(eId_force);
				double e_force_len = e_force.length();

				double forceDensity = ((e_force_len / e_form_len));

				e_form.normalize();
				e_force.normalize();
				forceDensity *= (e_form * e_force);


				forceDensities.push_back(forceDensity / forceDiagramScale);
				forceDensities.push_back(forceDensity / forceDiagramScale);

			}

			else
			{
				forceDensities.push_back(0);
				forceDensities.push_back(0);
			}
		}


	}

	//---- mesh specilization for setForceDensitiesFromDiagrams
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setForceDensitiesFromDiagrams(double forceDiagramScale)
	{
		forceDensities.clear();

		for (int i = 0; i < fnForm.numEdges(); i += 2)
		{
			// form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			double e_form_len = e_form.length();

			if (formEdge_forceEdge[eId_form] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[eId_form];
				zVector e_force = fnForce.getEdgeVector(eId_force);
				double e_force_len = e_force.length();

				double forceDensity = ((e_force_len / e_form_len));

				e_form.normalize();
				e_force.normalize();
				forceDensity *= (e_form * e_force);


				forceDensities.push_back(forceDensity / forceDiagramScale);
				forceDensities.push_back(forceDensity / forceDiagramScale);

			}

			else
			{
				forceDensities.push_back(0);
				forceDensities.push_back(0);
			}
		}


	}

	//---------------//

	//---- graph specilization for setForceDensities
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setTensionEdges(zDiagramType type, const vector<int>& _tensionEdges)
	{

		if (type == zResultDiagram)
		{
			result_tensionEdges.clear();

			for (int i = 0; i < fnResult.numEdges(); i++) result_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnResult.numEdges())
				{
					result_tensionEdges[_tensionEdges[i]] = true;					
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zFormDiagram)
		{
			form_tensionEdges.clear();

			for (int i = 0; i < fnForm.numEdges(); i++) form_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForm.numEdges())
				{
					form_tensionEdges[_tensionEdges[i]] = true;					
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zForceDiagram)
		{
			force_tensionEdges.clear();

			for (int i = 0; i < fnForce.numEdges(); i++) force_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForce.numEdges())
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
	inline void zTsVault<zObjMesh, zFnMesh>::setTensionEdges(zDiagramType type, const vector<int>& _tensionEdges)
	{		

		if (type == zResultDiagram)
		{
			result_tensionEdges.clear();

			for (int i = 0; i < fnResult.numEdges(); i++) result_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnResult.numEdges())
				{
					result_tensionEdges[_tensionEdges[i]] = true;
					
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zFormDiagram) 
		{
			form_tensionEdges.clear();

			for (int i = 0; i < fnForm.numEdges(); i++) form_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForm.numEdges())
				{
					form_tensionEdges[_tensionEdges[i]] = true;					
				}
			}

			setelementColorDomain(type);
		}

		else if (type == zForceDiagram)
		{
			force_tensionEdges.clear();

			for (int i = 0; i < fnForce.numEdges(); i++) force_tensionEdges.push_back(false);

			for (int i = 0; i < _tensionEdges.size(); i++)
			{
				if (_tensionEdges[i] >= 0 && _tensionEdges[i] < fnForce.numEdges())
				{
					force_tensionEdges[_tensionEdges[i]] = true;
				}
			}

			setelementColorDomain(type);
		}

		else throw std::invalid_argument(" invalid diagram type.");
		
	}



	//---------------//

	//---- graph specilization for setVertexWeights
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setVertexWeights(zDiagramType type, const vector<double>& vWeights)
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
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexWeights(zDiagramType type, const vector<double>& vWeights)
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

	//---------------//

	//---- graph specilization for setVertexThickness
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setVertexThickness(double thickness)
	{
		resultVThickness.clear();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			resultVThickness.push_back(thickness);
		}

	}

	//---- mesh specilization for setVertexThickness
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexThickness(double thickness)
	{
		resultVThickness.clear();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			resultVThickness.push_back(thickness);
		}

	}

	//---------------//

	//---- graph specilization for setVertexThickness
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setVertexThickness(vector<double> &thickness)
	{
		if (thickness.size() != fnResult.numVertices()) throw std::invalid_argument("size of thickness contatiner is not equal to number of mesh vertices.");

		resultVThickness = thickness;

	}

	//---- mesh specilization for setVertexThickness
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexThickness(vector<double> &thickness)
	{
		if (thickness.size() != fnResult.numVertices()) throw std::invalid_argument("size of thickness contatiner is not equal to number of mesh vertices.");

		resultVThickness = thickness;

	}

	//---------------//

	//---- graph specilization for setVertexMass
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setVertexMass(double mass)
	{
		resultVMass.clear();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			resultVMass.push_back(mass);
		}
	}

	//---- mesh specilization for setVertexMass
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexMass(double mass)
	{
		resultVMass.clear();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			resultVMass.push_back(mass);
		}
	}

	//---------------//

	//---- graph specilization for setVertexMass
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::setVertexMass(vector<double> &mass)
	{
		if (mass.size() != fnResult.numVertices()) throw std::invalid_argument("size of mass contatiner is not equal to number of mesh vertices.");

		resultVMass = mass;
	}

	//---- mesh specilization for setVertexMass
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexMass(vector<double> &mass)
	{
		if (mass.size() != fnResult.numVertices()) throw std::invalid_argument("size of mass contatiner is not equal to number of mesh vertices.");

		resultVMass = mass;
	}


	//---------------//

	//---- mesh specilization for setVertexMass
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::setVertexMassfromVertexArea()
	{
		resultVMass.clear();

		vector<double> vAreas;

		vector<zVector> fCenters;
		fnResult.getCenters(zFaceData, fCenters);

		vector<zVector> eCenters;
		fnResult.getCenters(zEdgeData, eCenters);

		fnResult.getVertexArea(fCenters, eCenters, resultVMass);

	}

	
	

	//---------------//

	//---- graph specilization for ForceDensityMethod
	template<>
	inline bool zTsVault<zObjGraph, zFnGraph>::forceDensityMethod()
	{
		zHEData type = zVertexData;
		bool positiveDensities = true;

		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e*0.5);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd X(fnResult.numVertices(), 3);
		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			zVector pos = fnResult.getVertexPosition(i);

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};

		// EDGE NODE MATRIX
		SpMat C = getEdgeNodeMatrix<SpMat>(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (int i = 0; i < n_e; i += 2)
		{

			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				q[FD_EdgesCounter] = forceDensities[i];

				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}

		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		Diag Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < resultVMass.size(); i++)
		{
			p[i] = resultVMass[i];
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

		SpMat Cn = subMatrix(C, freeVertices);
		SpMat Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

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

		if (positiveDensities)
		{
			SimplicialLLT< SpMat > solver; // sparse cholesky solver
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

			zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
			fnResult.setVertexPosition(id, pos);
		}

		return true;
	}


	//---- mesh specilization for ForceDensityMethod
	template<>
	inline bool zTsVault<zObjMesh, zFnMesh>::forceDensityMethod()
	{
		zHEData type = zVertexData;
		bool positiveDensities = true;

		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e*0.5);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd X(fnResult.numVertices(), 3);
		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			zVector pos = fnResult.getVertexPosition(i);

			X(i, 0) = pos.x;
			X(i, 1) = pos.y;
			X(i, 2) = pos.z;
		};

		// EDGE NODE MATRIX
		SpMat C = getEdgeNodeMatrix<SpMat>(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (int i = 0; i < n_e; i += 2)
		{

			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}

		}


		//printf("\n Force Densities: \n");
		//cout << endl << q <<endl;

		Diag Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < resultVMass.size(); i++)
		{
			p[i] = resultVMass[i];
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

		SpMat Cn = subMatrix(C, freeVertices);
		SpMat Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

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

		if (positiveDensities)
		{
			SimplicialLLT< SpMat > solver; // sparse cholesky solver
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
			cout <<endl << relative_error << " FDM - negative" << endl;

		}

		// POSITIONS OF NON FIXED VERTICES
		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];

			zVector pos(Xn(i, 0), Xn(i, 1), Xn(i, 2));
			fnResult.setVertexPosition(id, pos);
		}

		return true;
	}

	//---------------//
		

	//---- graph specilization for createResultfromFile
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::createResultfromFile(string path, zFileTpye type)
	{
		fnResult.from(path, type);

		setTensionEdges(zResultDiagram);
		setelementColorDomain(zResultDiagram);
		

		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createResultfromFile
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::createResultfromFile(string path, zFileTpye type)
	{
		fnResult.from(path, type);

		setTensionEdges(zResultDiagram);
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---------------//

	//---- graph specilization for createForcefromFile
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setelementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---- mesh specilization for createForcefromFile
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::createForcefromFile(string path, zFileTpye type)
	{
		fnForce.from(path, type);

		setTensionEdges(zForceDiagram);
		setelementColorDomain(zForceDiagram);

		setVertexWeights(zForceDiagram);
	}

	//---------------//

	//---- mesh specilization for createForcefromFile
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::createFormfromFile(string path, zFileTpye type)
	{
		fnForm.from(path, type);

		setTensionEdges(zFormDiagram);
		setelementColorDomain(zFormDiagram);


		setVertexWeights(zFormDiagram);
	}

	//---------------//

	//---- graph specilization for createFormFromResult
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::createFormFromResult()
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
	inline void zTsVault<zObjMesh, zFnMesh>::createFormFromResult()
	{
		std::clock_t start;
		start = std::clock();

	
		*formObj = fnResult.getDuplicate();

		zVector* pos = fnForm.getRawVertexPositions();
		zColor* col = fnForm.getRawVertexColors();

		zColor col1(1,1,1,1);

		for (int i = 0; i < fnForm.numVertices(); i++)
		{

			pos[i].z = 0;			
			col[i] = col1;	
		}
			

		zColor col2(0,0,0,1);

		for (int i = 0; i < fixedVertices.size(); i++)
		{
			col[fixedVertices[i]] = col2;
			
		}

		fnForm.setEdgeColor(elementColorDomain.max);

		
		setVertexWeights(zFormDiagram);
		setFormTensionEdgesfromResult();
		setelementColorDomain(zFormDiagram);
	}

	//---------------//

	//---- graph specilization for createFormFromForce
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::createFormFromForce(bool excludeBoundary, bool PlanarForceMesh, bool rotate90)
	{
		fnForce.getDualGraph(*formObj,forceEdge_formEdge, formEdge_forceEdge, excludeBoundary, PlanarForceMesh, rotate90);

		fnForm.setEdgeColor(elementColorDomain.max);

		setVertexWeights(zFormDiagram);
		

		setFormTensionEdgesfromForce();
		setelementColorDomain(zFormDiagram);
	}

	//---------------//

	//---- mesh specilization for createForceFromForm
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::createForceFromForm(bool rotate90)
	{


		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;


		int n_v = fnForm.numVertices();
		int n_e = fnForm.numEdges();
		int n_f = fnForm.numPolygons();


		vector<zVector> faceCenters;
		fnForm.getCenters(zFaceData, faceCenters);
		positions = faceCenters;

		vector<int> formHalfEdge_forceVertex;


		for (int i = 0; i < n_e; i++)
		{
			formHalfEdge_forceVertex.push_back(-1);

			if (fnForm.getFace(i))
			{
				formHalfEdge_forceVertex[i] = formObj->mesh.edges[i].getFace()->getFaceId();

				formHalfEdge_forceVertex[i] = fnForm.getFaceIndex(i);

			}

		}






		for (int i = 0; i < n_e; i++)
		{

			if (formHalfEdge_forceVertex[i] == -1)
			{

				// check if both the  vertices of the boundary edge arent fixed

				bool edgeVertsFixed = false;

				vector<int> eVerts;
				fnForm.getVertices(i, zEdgeData, eVerts);

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

					zEdge* start = fnForm.getEdge(i, zEdgeData);
					zEdge* e = start;

					bool exit = false;


					// walk prev
					do
					{

						if (fixedVerticesBoolean[e->getSym()->getVertex()->getVertexId()])
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
							if (fixedVerticesBoolean[e->getVertex()->getVertexId()])
							{
								exit = true;

							}

							if (exit) continue;

							if (e->getNext())
							{
								e = e->getNext();
								boundaryVertices.push_back(e->getVertex()->getVertexId());
								boundaryEdges.push_back(e->getEdgeId());

								if (fixedVerticesBoolean[e->getVertex()->getVertexId()])
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
							zVector pos = fnForm.getVertexPosition(boundaryVertices[j]);
							newPos += pos;
						}



						newPos /= boundaryVertices.size();
						positions.push_back(newPos);
					}

				}


			}
		}


		for (int i = 0; i < n_v; i++)
		{

			if (!fixedVerticesBoolean[i])
			{
				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

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

		forceObj->mesh = zMesh(positions, polyCounts, polyConnects);

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
		for (int i = 0; i < fnForce.numEdges(); i++)
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
			fnForce.edgeExists(v1, v2, eId);

			if (eId != -1)
			{
				zVector eForm = fnForm.getEdgeVector(i);
				eForm.normalize(); 

				zVector eForce = fnForce.getEdgeVector(eId);
				eForce.normalize();

				// for tension edge point to the edge in the opposite direction
				int symId = fnForm.getSymIndex(i);
				//if (fnForm.onBoundary(i, zEdgeData) || fnForm.onBoundary(symId, zEdgeData))
				//{
					if (form_tensionEdges[i])
					{
						if (eForm*eForce > 0) eId = fnForce.getSymIndex(eId);
					}
					// for compression edge point to the edge in the same direction
					else
					{
						if (eForm*eForce < 0) eId = fnForce.getSymIndex(eId);
					}
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

	//---------------//

	//---- graph specilization for createForceFromForm
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::createResultFromForm()
	{
		*resultObj = fnForm.getDuplicate();

		setResultTensionEdgesfromForm();
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---- mesh specilization for createForceFromForm
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::createResultFromForm()
	{
		*resultObj = fnForm.getDuplicate();

		setResultTensionEdgesfromForm();
		setelementColorDomain(zResultDiagram);

		setVertexWeights(zResultDiagram);
	}

	//---------------//

	//---- graph specilization for getHorizontalTargets
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::getHorizontalTargets(double formWeight)
	{
		targetEdges_form.clear();
		targetEdges_force.clear();

		for (int i = 0; i < fnForce.numEdges(); i++)
		{
			targetEdges_force.push_back(zVector());
		}

		for (int i = 0; i < fnForm.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			e_form.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				if (form_tensionEdges[eId_form]) eId_force = fnForce.getSymIndex(eId_force);

				zVector e_force = fnForce.getEdgeVector(eId_force);
				e_force.normalize();

				// target edge 
				zVector e_target = (e_form *formWeight) + (e_force * (1 - formWeight));
				e_target.normalize();

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

	//---- mesh specilization for getHorizontalTargets
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::getHorizontalTargets(double formWeight)
	{
		targetEdges_form.clear();
		targetEdges_force.clear();

		for (int i = 0; i < fnForce.numEdges(); i++)
		{
			targetEdges_force.push_back(zVector());
		}

		for (int i = 0; i < fnForm.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			e_form.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				if (form_tensionEdges[eId_form]) eId_force = fnForce.getSymIndex(eId_force);

				zVector e_force = fnForce.getEdgeVector(eId_force);
				e_force.normalize();

				// target edge 
				zVector e_target = (e_form *formWeight) + (e_force * (1 - formWeight));
				e_target.normalize();

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

	//---------------//

	//---- graph specilization for checkHorizontalParallelity
	template<>
	inline bool zTsVault<zObjGraph, zFnGraph>::checkHorizontalParallelity(zDomainDouble &deviation, double angleTolerance, bool colorEdges, bool printInfo)
	{
		bool out = true;
		vector<double> deviations;
		deviation.min = 10000;
		deviation.max = -10000;

		for (int i = 0; i < fnForm.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			e_form.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				zVector e_force = fnForce.getEdgeVector(eId_force);
				e_force.normalize();

				// angle

				double a_i = e_form.angle(e_force);
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

			for (int i = 0; i < fnForm.numEdges(); i++)
			{
				if (deviations[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

					if (deviations[i] < angleTolerance) col = zColor();

					fnForm.setEdgeColor(i, col);

					int eId_force = formEdge_forceEdge[i];
					fnForce.setEdgeColor(eId_force, col);

				}

			}

		}


		return out;
	}

	//---- mesh specilization for checkHorizontalParallelity
	template<>
	inline bool zTsVault<zObjMesh, zFnMesh>::checkHorizontalParallelity(zDomainDouble &deviation, double angleTolerance, bool colorEdges, bool printInfo)
	{
		bool out = true;
		vector<double> deviations;
		deviation.min = 10000;
		deviation.max = -10000;

		for (int i = 0; i < fnForm.numEdges(); i++)
		{
			//form edge
			int eId_form = i;
			zVector e_form = fnForm.getEdgeVector(eId_form);
			e_form.normalize();


			if (formEdge_forceEdge[i] != -1)
			{
				// force edge
				int eId_force = formEdge_forceEdge[i];
				zVector e_force = fnForce.getEdgeVector(eId_force);
				e_force.normalize();

				// angle

				double a_i = e_form.angle(e_force);
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

			for (int i = 0; i < fnForm.numEdges(); i++)
			{
				if (deviations[i] != -1)
				{
					zColor col = coreUtils.blendColor(deviations[i], deviation, colDomain, zHSV);

					if (deviations[i] < angleTolerance) col = zColor();

					fnForm.setEdgeColor(i, col);

					int eId_force = formEdge_forceEdge[i];
					fnForce.setEdgeColor(eId_force, col);

				}

			}

		}


		return out;
	}

	//---------------//

	//---- graph specilization for updateFormDiagram
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type,int numIterations)
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

		vector<int> cEdges;
		zVector v_i, v_j, e_ij, t_ij, b_i, r_i;
		zVector forceV;


		for (int k = 0; k < numIterations; k++)
		{
			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				cEdges.clear();
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				v_i = fnForm.getVertexPosition(i);

				// compute barycenter per vertex
				b_i = zVector(0, 0, 0);
				for (int j = 0; j < cEdges.size(); j++)
				{

					v_j = fnForm.getVertexPosition(fnForm.getEndVertexIndex(cEdges[j]));

					e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					t_ij = targetEdges_form[fnForm.getSymIndex(cEdges[j])];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				r_i = b_i - v_i;
				v_residual[i] = (r_i);

				forceV = v_residual[i] * formVWeights[i];
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
	inline void zTsVault<zObjMesh, zFnMesh>::updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations)
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
			vector<int> cEdges;

			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				cEdges.clear();
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				zVector v_i = positions[i];

				// compute barycenter per vertex
				zVector b_i(0, 0, 0);
				for (int j = 0; j < cEdges.size(); j++)
				{

					zVector v_j = positions[fnForm.getEndVertexIndex(cEdges[j])];

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					zVector t_ij = targetEdges_form[fnForm.getSymIndex(cEdges[j])];
					t_ij.normalize();

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				v_residual[i] = b_i - v_i;
				

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

	//---------------//

	//---- graph specilization for addAxialForces
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::addAxialForces(double forceDiagramScale)
	{

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			vector<int> cEdges;
			fnResult.getConnectedEdges(i, zVertexData, cEdges);

			zVector aForce;

			for (int j = 0; j < cEdges.size(); j++)
			{
				// result edge
				int eId_resut = cEdges[j];
				zVector e_result = fnResult.getEdgeVector(eId_resut);
				double e_result_len = e_result.length();
				e_result.normalize();

				// form edge
				int eId_form = cEdges[j];
				zVector e_form = fnForm.getEdgeVector(eId_form);
				double e_form_len = e_form.length();

				if (formEdge_forceEdge[eId_form] != -1)
				{
					// force edge
					int eId_force = formEdge_forceEdge[eId_form];
					zVector e_force = fnForce.getEdgeVector(eId_force);
					double e_force_len = e_force.length();

					double forceDensity = ((e_force_len / e_form_len));

					e_form.normalize();
					e_force.normalize();
					forceDensity *= (e_form * e_force);

					//printf("\n %i | forceDensity %1.2f ", i, forceDensity);

					aForce += (e_result * e_result_len * forceDensity * forceDiagramScale);
				}
			}

			aForce *= resultVWeights[i];

			//printf("\n aForce %i | %1.2f %1.2f %1.2f ", i, aForce.x, aForce.y, aForce.z);
			fnResultParticles[i].addForce(aForce);
		}

	}

	//---- mesh specilization for addAxialForces
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::addAxialForces(double forceDiagramScale)
	{
		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			vector<int> cEdges;
			fnResult.getConnectedEdges(i, zVertexData, cEdges);

			zVector aForce;

			for (int j = 0; j < cEdges.size(); j++)
			{
				// result edge
				int eId_resut = cEdges[j];
				zVector e_result = fnResult.getEdgeVector(eId_resut);
				double e_result_len = e_result.length();
				e_result.normalize();

				// form edge
				int eId_form = cEdges[j];
				zVector e_form = fnForm.getEdgeVector(eId_form);
				double e_form_len = e_form.length();

				if (formEdge_forceEdge[eId_form] != -1)
				{
					// force edge
					int eId_force = formEdge_forceEdge[eId_form];
					zVector e_force = fnForce.getEdgeVector(eId_force);
					double e_force_len = e_force.length();

					double forceDensity = ((e_force_len / e_form_len));

					e_form.normalize();
					e_force.normalize();
					forceDensity *= (e_form * e_force);


					aForce += (e_result * e_result_len * forceDensity * forceDiagramScale);
				}
			}

			aForce *= resultVWeights[i];

			//printf("\n aForce %i | %1.2f %1.2f %1.2f ", i, aForce.x, aForce.y, aForce.z);
			fnResultParticles[i].addForce(aForce);
		}

	}

	//---------------//

	//---- graph specilization for addDeadLoadForces
	template<>
	inline void zTsVault<zObjGraph, zFnGraph>::addDeadLoadForces()
	{


		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			zVector dForce(0, 0, 1);

			dForce *= (resultVMass[i] * resultVThickness[i] * resultVWeights[i]);

			//printf("\n dForce %i | %1.2f %1.2f %1.2f ", i, dForce.x, dForce.y, dForce.z);
			fnResultParticles[i].addForce(dForce);
		}
	}

	//---- mesh specilization for addDeadLoadForces
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::addDeadLoadForces()
	{

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			zVector dForce(0, 0, 1);

			dForce *= (resultVMass[i] * resultVThickness[i] * resultVWeights[i]);

			fnResultParticles[i].addForce(dForce);
		}
	}

	//---------------//

	//---- graph specilization for addDeadLoadForces
	template<>
	inline bool zTsVault<zObjGraph, zFnGraph>::updateResultDiagram(double forceDiagramScale, double dT, zIntergrationType type, double tolerance)
	{
		if (fnResultParticles.size() != fnResult.numVertices())
		{
			fnResultParticles.clear();
			resultParticlesObj.clear();


			for (int i = 0; i < fnResult.numVertices(); i++)
			{
				bool fixed = false;

				zObjParticle p;
				p.particle = zParticle(resultObj->graph.vertexPositions[i], fixed);
				resultParticlesObj.push_back(p);

			}

			for (int i = 0; i < resultParticlesObj.size(); i++)
			{
				fnResultParticles.push_back(zFnParticle(resultParticlesObj[i]));
			}
		}

		// compute forces
		addAxialForces(forceDiagramScale);
		addDeadLoadForces();

		bool out = true;
		// update positions
		for (int i = 0; i < fnResultParticles.size(); i++)
		{
			if (fnResultParticles[i].getForce().length() > tolerance) out = false;

			fnResultParticles[i].integrateForces(dT, type);
			fnResultParticles[i].updateParticle(true);
		}

		return out;
	}

	//---- mesh specilization for addDeadLoadForces
	template<>
	inline bool zTsVault<zObjMesh, zFnMesh>::updateResultDiagram(double forceDiagramScale, double dT, zIntergrationType type, double tolerance)
	{


		if (fnResultParticles.size() != fnResult.numVertices())
		{
			fnResultParticles.clear();
			resultParticlesObj.clear();


			for (int i = 0; i < fnResult.numVertices(); i++)
			{
				bool fixed = false;

				zObjParticle p;
				p.particle = zParticle(resultObj->mesh.vertexPositions[i], fixed);
				resultParticlesObj.push_back(p);

			}

			for (int i = 0; i < resultParticlesObj.size(); i++)
			{
				fnResultParticles.push_back(zFnParticle(resultParticlesObj[i]));
			}
		}

		// compute forces
		addAxialForces(forceDiagramScale);
		addDeadLoadForces();


		bool out = true;
		// update positions
		for (int i = 0; i < fnResultParticles.size(); i++)
		{
			if (fnResultParticles[i].getForce().length() > tolerance) out = false;

			fnResultParticles[i].integrateForces(dT, type);
			fnResultParticles[i].updateParticle(true);
		}

		return out;
	}

	//---------------//

	//---- graph specilization for verticalEquilibrium using Linear Algebra
	template<>
	inline bool zTsVault<zObjGraph, zFnGraph>::equilibriumVertical(bool &computeForceDensitities, double forceDiagramScale)
	{
		if (computeForceDensitities)
		{
			setForceDensitiesFromDiagrams(forceDiagramScale);

			computeForceDensitities = false;
		}

		zHEData type = zVertexData;
		
		bool positiveDensities = true;

		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e*0.5);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd Xz(fnResult.numVertices(), 1);
		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			zVector pos = fnResult.getVertexPosition(i);

			Xz(i, 0) = pos.z;

		};

		// EDGE NODE MATRIX
		SpMat C = getEdgeNodeMatrix<SpMat>(numEdges);

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;


		for (int i = 0; i < fnResult.numEdges(); i+= 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}

			//// form edge
			//int eId_form = i;
			//zVector e_form = fnForm.getEdgeVector(eId_form);
			//double e_form_len = e_form.length();

			//if (formEdge_forceEdge[eId_form] != -1)
			//{
			//	// force edge
			//	int eId_force = formEdge_forceEdge[eId_form];
			//	zVector e_force = fnForce.getEdgeVector(eId_force);
			//	double e_force_len = e_force.length();

			//	double forceDensity = ((e_force_len / e_form_len));

			//	e_form.normalize();
			//	e_force.normalize();
			//	forceDensity *= (e_form * e_force);

			//	if (forceDensity < 0)positiveDensities = false;

			//	q[FD_EdgesCounter] = (forceDensity / forceDiagramScale);
			//	FD_EdgesCounter++;
			//}
		}	


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		Diag Q = q.asDiagonal();

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

		SpMat Cn = subMatrix(C, freeVertices);
		SpMat Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(Xz, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

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

		if (positiveDensities)
		{
			SimplicialLLT< SpMat > solver; // sparse cholesky solver
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

			zVector pos =	fnResult.getVertexPosition(id);
			pos.z = Xn(i, 0);
			
			fnResult.setVertexPosition(id, pos);
		}

		return true;
	}

	//---- mesh specilization for verticalEquilibrium using Linear Algebra
	template<>
	inline bool zTsVault<zObjMesh, zFnMesh>::equilibriumVertical(bool &computeForceDensitities, double forceDiagramScale)
	{
		if (computeForceDensitities)
		{
			setForceDensitiesFromDiagrams(forceDiagramScale);

			computeForceDensitities = false;
		}

		zHEData type = zVertexData;

		bool positiveDensities = true;

		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e*0.5);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd Xz(fnResult.numVertices(), 1);
		zVector* pos = fnResult.getRawVertexPositions();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			Xz(i, 0) = pos[i].z;

		};

		//printf("\n xZ: \n");
		//cout << endl << Xz;

		// EDGE NODE MATRIX
		SpMat C = getEdgeNodeMatrix<SpMat>(numEdges);

		//printf("\n edgeNode: \n");
		//cout << endl << C;

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (int i = 0; i < fnResult.numEdges(); i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}

			//// form edge
			//int eId_form = i;
			//zVector e_form = fnForm.getEdgeVector(eId_form);
			//double e_form_len = e_form.length();

			//if (formEdge_forceEdge[eId_form] != -1)
			//{
			//	// force edge
			//	int eId_force = formEdge_forceEdge[eId_form];
			//	zVector e_force = fnForce.getEdgeVector(eId_force);
			//	double e_force_len = e_force.length();

			//	double forceDensity = ((e_force_len / e_form_len));

			//	e_form.normalize();
			//	e_force.normalize();
			//	forceDensity *= (e_form * e_force);

			//	if (forceDensity < 0) positiveDensities = false;

			//	q[FD_EdgesCounter] = (forceDensity /forceDiagramScale);
			//	FD_EdgesCounter++;
			//}
		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		Diag Q = q.asDiagonal();

		// LOAD VECTOR
		VectorXd p(fnResult.numVertices());

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			p[i] = (resultVMass[i] * resultVThickness[i] * resultVWeights[i] * forceDiagramScale);
		}

		//printf("\n Load: \n");
		//cout << endl << p;

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

		SpMat Cn = subMatrix(C, freeVertices);
		SpMat Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(Xz, fixedVertices);
		MatrixXd Pn = subMatrix(P, freeVertices);

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

		if (positiveDensities)
		{
			

			SimplicialLLT< SpMat > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
			{
				return false;
			}
				
		

			Xn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
			{				
				return false;
			}

		
		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Xn = denseDn.ldlt().solve(B);

			// convergence error check.
			double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			cout <<endl << relative_error << " FDM - negative" << endl;

		}
				

		// POSITIONS OF NON FIXED VERTICES
		
		

		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];
			pos[id].z = Xn(i, 0);		
					
		}			


		return true;
	}

	//---------------//

	//---- mesh specilization for getForces
	template<>
	inline void zTsVault<zObjMesh, zFnMesh>::getForces(vector<zVector>& forces)
	{
		zHEData type = zVertexData;

		bool positiveDensities = true;

		int n_e = fnResult.numEdges();
		int n_v = fnResult.numVertices();

		int numEdges = floor(n_e*0.5);

		for (int i = 0; i < n_e; i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (fixedVerticesBoolean[v1] && fixedVerticesBoolean[v2]) numEdges--;

		}

		// POSITION MATRIX
		MatrixXd X(fnResult.numVertices(), 3);
		zVector* pos = fnResult.getRawVertexPositions();

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			X(i, 0) = pos[i].x;
			X(i, 1) = pos[i].y;
			X(i, 2) = pos[i].z;

		};

		// EDGE NODE MATRIX
		SpMat C = getEdgeNodeMatrix<SpMat>(numEdges);

		//printf("\n edgeNode: \n");
		//cout << endl << C;

		// FORCE DENSITY VECTOR
		VectorXd q(numEdges);
		int FD_EdgesCounter = 0;

		for (int i = 0; i < fnResult.numEdges(); i += 2)
		{
			vector<int> eVerts;
			fnResult.getVertices(i, zEdgeData, eVerts);

			int v1 = eVerts[0];
			int v2 = eVerts[1];

			if (!fixedVerticesBoolean[v1] || !fixedVerticesBoolean[v2])
			{
				q[FD_EdgesCounter] = forceDensities[i];
				if (forceDensities[i] < 0) positiveDensities = false;
				FD_EdgesCounter++;
			}			
		}


		//printf("\n Force Densities: \n");
		//cout << endl << q;

		Diag Q = q.asDiagonal();

		
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

		SpMat Cn = subMatrix(C, freeVertices);
		SpMat Cf = subMatrix(C, fixedVertices);
		MatrixXd Xf = subMatrix(X, fixedVertices);	
		MatrixXd Xn = subMatrix(X, freeVertices);
		

		SpMat Cn_transpose;
		Cn_transpose = Cn.transpose();
	
		
		//CHOLESKY DECOMPOSITION

		SpMat Dn = Cn_transpose * Q * Cn;
		SpMat Df = Cn_transpose * Q * Cf;
		
		MatrixXd  B =Xn - Df * Xf ;
			   			
		
		// solve
		MatrixXd Fn;

		
		if (positiveDensities)
		{


			SimplicialLLT< SpMat > solver; // sparse cholesky solver
			solver.compute(Dn); // compute cholesky factors

			if (solver.info() != Eigen::Success)
			{
				return;
			}


			Fn = solver.solve(B); // solve AX = B ;
			if (solver.info() != Eigen::Success)
			{
				return ;
			}


		}
		else
		{
			MatrixXd denseDn;
			denseDn = MatrixXd(Dn);
			Fn = denseDn.ldlt().solve(B);

			// convergence error check.
			//double relative_error = (denseDn*Xn - B).norm() / B.norm(); // norm() is L2 norm
			//cout << endl << relative_error << " FDM - negative" << endl;

		}

		printf("\n Fn:");
		cout << endl << Fn << endl;


		//// OUT
		forces.clear();
		forces.assign(fnResult.numVertices(), zVector());

		for (int i = 0; i < freeVertices.size(); i++)
		{
			int id = freeVertices[i];

			forces[id].x = Fn(i, 0)  * -1 * (pos[id].z / abs(pos[id].z));
			forces[id].y = Fn(i, 1)  * -1 * (pos[id].z / abs(pos[id].z));
			forces[id].z = Fn(i, 2)  * -1 * (pos[id].z / abs(pos[id].z));
		}

	}

	//---------------//

	//---- graph specilization for getForces_GradientDescent
	template<>
	inline void zTsVault < zObjGraph, zFnGraph>::getForces_GradientDescent(vector<zVector>& forces)
	{
		zVector* pos = fnResult.getRawVertexPositions();

		forces.clear();
		forces.assign(fnResult.numVertices(), zVector());

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			if (fixedVerticesBoolean[i]) continue;

			vector<int> cVerts;
			fnResult.getConnectedVertices(i, zVertexData, cVerts);			


			// get lowest positions

			zVector lowPosition = fnResult.getVertexPosition(i);
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
	inline void zTsVault<zObjMesh, zFnMesh>::getForces_GradientDescent(vector<zVector>& forces)
	{
		zVector* pos = fnResult.getRawVertexPositions();

		forces.clear(); 
		forces.assign(fnResult.numVertices(), zVector());

		for (int i = 0; i < fnResult.numVertices(); i++)
		{
			if (fixedVerticesBoolean[i]) continue;

			vector<int> cFaces; 
			fnResult.getConnectedFaces(i, zVertexData, cFaces);

		
			vector<int> positionIndicies;
			for (int j = 0; j < cFaces.size(); j++)
			{
				vector<int> fVerts;
				fnResult.getVertices(cFaces[j], zFaceData, fVerts);

				for (int k = 0; k < fVerts.size(); k++) positionIndicies.push_back(fVerts[k]);
			}
					

			// get lowest positions
			
			zVector lowPosition = fnResult.getVertexPosition(i);
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

	//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */






}



