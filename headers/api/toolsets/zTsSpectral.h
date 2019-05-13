#pragma once

#include <headers/api/functionsets/zFnMesh.h>

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
		MatrixXd meshLaplacian;

		/*!	\brief container storing eigen function values.  */
		vector<double> eigenFunctionValues;

		/*!	\brief color domain.  */
		zDomainColor colorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

		/*!	\brief eigen values domain.  */
		zDomainDouble eigenDomain = zDomainDouble(0.0, 1.0);

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief mesh function set  */
		zFnMesh fnMesh;	

		/*!	\brief eigen solver  */
		SelfAdjointEigenSolver<MatrixXd> eigensolver;		


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
			
		}

		//--------------------------
		//---- COMPUTE METHODS
		//--------------------------
		
		
		/*! \brief This method computes the Eigen function operator.
		*
		*	\details Based on http://mgarland.org/files/papers/ssq.pdf
		*	\param		[in]	frequency		- input frequency value.		
		*	\param		[in]	useVertexArea	- uses vertex area for laplcaian weight calculation if true.
		*	\since version 0.0.2
		*/
		void computeEigenFunction(double &frequency)
		{
			int n_v = fnMesh.numVertices();


			if (meshLaplacian.cols() != n_v)
			{
				meshLaplacian = fnMesh.getTopologicalLaplacian();

				eigensolver.compute(meshLaplacian);
				if (eigensolver.info() != Success) abort();
			}


			int val = (int) frequency;
			if (val >= n_v) val = (int)frequency % n_v;

			

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
				
				double matVal = eigensolver.eigenvectors().col(val).row(i).value();

				double EigenFunctionRegular = matVal;

				eigenFunctionValues[i] = coreUtils.ofMap(matVal, inDomain, eigenDomain);
			}

			setVertexColorFromEigen(true);
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

				for (int i = 0; i < n_v; i++)
				{
					vector<int> cVertices;
					fnMesh.getConnectedVertices(i, zVertexData, cVertices);

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

		/*! \brief This method sets the color domain.
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\since version 0.0.2
		*/
		void setColorDomain(zDomainColor &colDomain)
		{
			colorDomain = colDomain;
		}

		/*! \brief This method sets vertex color of all the vertices based on the eigen function.
		*
		*	\param		[in]	setFaceColor	- face color is computed based on the vertex color if true.
		*	\since version 0.0.2
		*/
		void setVertexColorFromEigen(bool setFaceColor = false)
		{
			zColor* cols = fnMesh.getRawVertexColors();

			colorDomain.min.toHSV(); colorDomain.max.toHSV();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				
				cols[i] = coreUtils.blendColor(eigenFunctionValues[i], eigenDomain, colorDomain, zHSV);
				
			}

			if (setFaceColor) fnMesh.computeFaceColorfromVertexColor();
		}


		//--------------------------
		//---- GET METHODS
		//--------------------------

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
				double matVal = eigensolver.eigenvectors().col(colIndex).row(i).value();

				if (matVal < inDomain.min) inDomain.min = matVal;
				if (matVal > inDomain.max) inDomain.max = matVal;
			}
			
		}

	};

}