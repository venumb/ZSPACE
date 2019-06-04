#pragma once

#include <headers/api/functionsets/zFnMesh.h>

#include <depends/spectra/include/Spectra/SymEigsShiftSolver.h>
#include <depends/spectra/include/Spectra/MatOp/SparseSymShiftSolve.h>
using namespace Spectra;

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

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsMeshSpectral
	*	\brief A mesh spectral processing tool set class on triangular meshes.
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/


	class zTsSpectral
	{
	protected:

		/*! \brief core utilities object */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;

		/*!	\brief container of area per vertex of the mesh object  */
		vector<double> vertexArea;

		/*!	\brief container storing vertex types.  */
		vector< zSpectralVertexType> vertexType;

		/*!	\brief container storing ring neighbour indices per vertex.  */
		vector< vector<int>> vertexRing;

		/*!	\brief matrix to store mesh laplacian weights  */
		zSparseMatrix meshLaplacian;

		/*!	\brief container storing eigen function values.  */
		vector<double> eigenFunctionValues;

		/*!	\brief color type - zHSV/ zRGB  */
		zColorType colorType;

		/*!	\brief color domain.  */
		zDomainColor colorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

		/*!	\brief eigen values domain.  */
		zDomainDouble eigenDomain = zDomainDouble(0.0, 1.0);

		

		/*!	\brief number of eigen vectors required.  */
		int n_Eigens;
		
	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief mesh function set  */
		zFnMesh fnMesh;	

		/*!	\brief Eigen vectors matrix  */		
		MatrixXd eigenVectors;

		/*!	\brief Eigen values vector  */
		VectorXd eigenValues;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsSpectral() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsSpectral(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;
			fnMesh = zFnMesh(_meshObj);

			n_Eigens = fnMesh.numVertices() - 1;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsSpectral() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates the robot from the input file.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be imported.
		*	\since version 0.0.2
		*/
		void createMeshfromFile(string path, zFileTpye type)
		{
			fnMesh.from(path, type, false);		
			
			n_Eigens = fnMesh.numVertices() - 1;
		}

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------
		
		
		/*! \brief This method computes the Eigen function operator.
		*
		*	\details Based on http://mgarland.org/files/papers/ssq.pdf
		*	\param		[in]	frequency		- input frequency value.		
		*	\param		[in]	computeEigenVectors	- cmputes eigen vectors if true.
		*	\since version 0.0.2
		*/
		double computeEigenFunction(double &frequency, bool &computeEigenVectors)
		{
			int n_v = fnMesh.numVertices();

			if (meshLaplacian.cols() != n_v)
			{
				std::clock_t start;
				start = std::clock();

				meshLaplacian = fnMesh.getTopologicalLaplacian();

				printf("\n meshLaplacian: r %i  c %i ", meshLaplacian.rows(), meshLaplacian.cols());

				double t_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "\n mesh Laplacian compute: " << t_duration << " seconds";					
			}

			if (computeEigenVectors)
			{
				std::clock_t start;
				start = std::clock();

				//using spectra
				SparseSymShiftSolve<double> op(meshLaplacian);
				SymEigsShiftSolver< double, LARGEST_MAGN, SparseSymShiftSolve<double> > eigs(&op, n_Eigens, n_Eigens + 1, 0.0);

				// Initialize and compute
				eigs.init();
				int nconv = eigs.compute();

				double t_duration2 = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				std::cout << "\n Eigen solve : " << t_duration2 << " seconds";

				// Retrieve results
				if (eigs.info() == SUCCESSFUL)
				{
					eigenVectors = eigs.eigenvectors(nconv);
					eigenValues = eigs.eigenvalues();
				}
				else
				{
					cout << "\n Eigen convergence unsuccessful ";
					return -1.0;
				}

				printf("\n Eigen num converge %i : eigenVectors r %i  c %i ", nconv, eigenVectors.rows(), eigenVectors.cols());

				computeEigenVectors = !computeEigenVectors;
			}


			if (eigenVectors.rows() != n_v) return -1.0;

			int val = (int) frequency;
			if (val >= n_Eigens) val = (int)frequency % n_Eigens;			

			// compute derivatives
			zDomainDouble inDomain;			
			computeMinMax_Eigen(val, inDomain);


			// compute eigen function operator
			if (eigenFunctionValues.size() == 0 || eigenFunctionValues.size() != n_v)
			{
				eigenFunctionValues.clear(); 
				eigenFunctionValues.assign(n_v, 0);
			}

			for (int i = 0; i < n_v; i++)
			{
				double matVal = eigenVectors.col(val).row(i).value();			

				double EigenFunctionRegular = matVal;

				eigenFunctionValues[i] = coreUtils.ofMap(matVal, inDomain, eigenDomain);
			}

			setVertexColorFromEigen(true);

			return eigenValues[val];
		}

		/*! \brief This method computes the vertex type.
		*
		*	\param		[in]	frequency		- input frequency value.
		*	\param		[in]	useVertexArea	- uses vertex area for laplcaian weight calculation if true.
		*	\since version 0.0.2
		*/
		void computeVertexType()
		{
			int n_v = fnMesh.numVertices();

			// compute vertex neighbours if not computed already
			if (vertexRing.size() == 0 || vertexRing.size() != n_v)
			{
				vertexRing.clear();

				for (zItMeshVertex v(*meshObj); !v.end(); v.next())
				{
					vector<int> cVertices;
					v.getConnectedVertices(cVertices);

					vertexRing.push_back(cVertices);
				}
			}

			// set size of vertex type to number of vertices
			if (vertexType.size() == 0 || vertexType.size() != n_v)
			{
				vertexType.clear(); 
				vertexType.assign(n_v, zSpectralVertexType::zRegular);
			}

			// compute vertex type
			for (int i = 0; i < n_v; i++)
			{
				int n_rv = vertexRing[i].size();

				int minCounter = 0;
				int maxCounter = 0;

				vector<int> minId;

				for (int j = 0; j < n_rv; j++)
				{
					int otherId = vertexRing[i][j];
					
					if (eigenFunctionValues[i] > eigenFunctionValues[otherId]) maxCounter++;
					else
					{
						minId.push_back(j);
						minCounter++;
					}
				}

				vertexType[i] = zSpectralVertexType::zRegular;
				if (maxCounter == n_rv) vertexType[i] = zSpectralVertexType::zMaxima;; 
				if (minCounter == n_rv) vertexType[i] = zSpectralVertexType::zMinima;;

				// Check for  Saddle
				bool chkChain = true;

				if (maxCounter != n_rv && minCounter != n_rv)
				{
					int disContuinityCounter = 0;

					for (int j = 0; j < minCounter; j++)
					{

						int k = (j + 1) % minCounter;

						int chk = (minId[j] + 1) % n_rv;

						if (chk != minId[k]) disContuinityCounter++;

					}

					if (disContuinityCounter > 1)
					{
						chkChain = false;
					}

				}

				if (!chkChain) vertexType[i] = zSpectralVertexType::zSaddle;;; 
			}
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the number of eigens to be requested. The value is capped to number of vertices in the mesh , if the input value is higher.
		*
		*	\param		[in]	_numEigen		- input number of eigens. 
		*	\since version 0.0.2
		*/
		void setNumEigens( int _numEigen)
		{
			n_Eigens = _numEigen;

			if (_numEigen >= fnMesh.numVertices()) n_Eigens = fnMesh.numVertices() - 1;
		}

		/*! \brief This method sets the color domain.
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\param		[in]	colType			- input color type.
		*	\since version 0.0.2
		*/
		void setColorDomain(zDomainColor &colDomain, zColorType colType)
		{
			colorDomain = colDomain;
			colorType = colType;
		}

		/*! \brief This method sets vertex color of all the vertices based on the eigen function.
		*
		*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColorFromEigen(bool setFaceColor = false)
		{
			zColor* cols = fnMesh.getRawVertexColors();			

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{				
				cols[i] = coreUtils.blendColor(eigenFunctionValues[i], eigenDomain, colorDomain, colorType);				
			}

			if (setFaceColor) fnMesh.computeFaceColorfromVertexColor();
		}


		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method gets the number of eigens.
		*
		*	\return			int		-  number of eigens.
		*	\since version 0.0.2
		*/
		int numEigens()
		{
			return n_Eigens;
		}

		/*! \brief This method gets the current function values.
		*
		*	\param		[out]	_eigenFunctionValues	- contatiner of eigen function values per vertex.
		*	\since version 0.0.2
		*/
		void getEigenFunctionValues(vector<double> &_eigenFunctionValues)
		{
			_eigenFunctionValues = eigenFunctionValues;
		}

		/*! \brief This method gets the pointer to the internal function value container.
		*
		*	\return		double*		- pointer to internal contatiner of eigen function values per vertex.
		*	\since version 0.0.2
		*/
		double* getRawEigenFunctionValues()
		{
			return &eigenFunctionValues[0];
		}

	protected:

		
		//--------------------------
		//---- PROTECTED UTILITY METHODS
		//--------------------------
			   
		/*! \brief This method computes the min and max of the eigen function at the input column index.
		*
		*	\param		[in]	colIndex				- input column index.
		*	\param		[out]	EigenDomain				- output eigen value domain.
		*	\since version 0.0.2
		*/
		void computeMinMax_Eigen(int &colIndex , zDomainDouble &inDomain)
		{
			inDomain.min = 10000;
			inDomain.max = -10000;

			
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double matVal = eigenVectors.col(colIndex).row(i).value();			

				if (matVal < inDomain.min) inDomain.min = matVal;
				if (matVal > inDomain.max) inDomain.max = matVal;
			}
			
		}

	};

}