#pragma once


#include<headers/core/zDefinitions.h>
#include<headers/core/zVector.h>
#include<headers/core/zMatrix.h>


#include<depends/Eigen/Core>
#include<depends/Eigen/Dense>
#include<depends/Eigen/Sparse>
#include<depends/Eigen/Eigen>
#include<depends/Eigen/Sparse>
using namespace Eigen;


namespace zSpace
{
	/** \addtogroup zVectorMatrixUtilities
	*	\brief Collection of utility methods using vector and matricies.
	*  @{
	*/


	//--------------------------
	//---- VECTOR METHODS 
	//--------------------------

	/*! \brief This method a zVector from the input matrix row.
	*
	*	\param		[in]		zMatrixd	- input matrix. number of columns need to be 3 or 4.
	*	\param		[in]		rowIndex	- row index to be extracted.
	*	\return					zVector		- zVector of the row matrix.
	*	\since version 0.0.1
	*/
	zVector fromMatrixRow(zMatrixd &inMatrix, int rowIndex = 0)
	{
		if (inMatrix.getNumCols() < 3 || inMatrix.getNumCols() > 4) throw std::invalid_argument("cannot convert matrix row to vector.");

		vector<double> rVals = inMatrix.getRow(rowIndex);

		return zVector(rVals[0], rVals[1], rVals[2]);
	}

	/*! \brief This method returns extracts a zVector from the input matrix column.
	*
	*	\param		[in]		zMatrixd	- input matrix. number of rows need to be 3 or 4.
	*	\param		[in]		rowIndex	- row index to be extracted.
	*	\return					zVector		- zVector of the column matrix.
	*	\since version 0.0.1
	*/

	zVector fromMatrixColumn(zMatrixd &inMatrix, int colIndex)
	{
		if (inMatrix.getNumRows() < 3 || inMatrix.getNumRows() > 4) throw std::invalid_argument("cannot convert matrix column to vector.");

		vector<double> cVals = inMatrix.getCol(colIndex);
		return zVector(cVals[0], cVals[1], cVals[2]);
	}

	/*! \brief This method returns the bounds of the input list points.
	*
	*	\param  	[in]	inGraph	- input graph.
	*	\param  	[out]	minBB	- stores zVector of bounding box minimum.
	*	\param		[out]	maxBB	- stores zVector of bounding box maximum.
	*/
	void getBounds(vector<zVector> &inPoints, zVector &minBB, zVector &maxBB)
	{
		minBB = zVector(10000, 10000, 10000);
		maxBB = zVector(-10000, -10000, -10000);

		for (int i = 0; i < inPoints.size(); i++)
		{
			if (inPoints[i].x < minBB.x) minBB.x = inPoints[i].x;
			if (inPoints[i].y < minBB.y) minBB.y = inPoints[i].y;
			if (inPoints[i].y < minBB.z) minBB.z = inPoints[i].z;

			if (inPoints[i].x > maxBB.x) maxBB.x = inPoints[i].x;
			if (inPoints[i].y > maxBB.y) maxBB.y = inPoints[i].y;
			if (inPoints[i].z > maxBB.z) maxBB.z = inPoints[i].z;
		}
	}

	/*! \brief This method computes the distances in X,Y,Z for the input bounds.
	*
	*	\param		[in]	minBB			- lower bounds as zVector.
	*	\param		[in]	maxBB			- upper bounds as zVector
	*	\param		[out]	Dims			- distances in X,Y,Z axis in local frame
	*	\since version 0.0.1
	*/

	zVector getDimsFromBounds(zVector &minBB, zVector &maxBB)
	{
		zVector out;

		out.x = abs(maxBB.x - minBB.x);
		out.y = abs(maxBB.y - minBB.y);
		out.z = abs(maxBB.z - minBB.z);

		return out;
	}

	/*! \brief This method  returns the intersection of two planes which is  line.
	*
	*	\details Based on http://paulbourke.net/geometry/pointlineplane/
	*	\param		[in]		nA	- input normal of plane A.
	*	\param		[in]		nB	- input normal of plane B.
	*	\param		[in]		pA		- input point on plane A.
	*	\param		[in]		pB		- input point on plane B.
	*	\param		[out]		outPt1	- intersection point 1.
	*	\param		[out]		outPt2	- intersection point 2.
	*	\return					bool	- true if the planes intersect.
	*	\since version 0.0.1
	*/

	bool plane_planeIntersection(zVector &nA, zVector &nB, zVector &pA, zVector &pB, zVector &outP1, zVector &outP2)
	{
		{
			bool out = false;

			if ((nA^nB).length() > 0)
			{
				double detr = ((nA*nA)*(nB*nB)) - ((nA*nB) * (nA*nB));

				double d1 = nA * pA;
				double d2 = nB * pB;

				double c1 = (d1*(nB*nB)) - (d2*(nA*nB));
				c1 /= detr;

				double c2 = (d2*(nA*nA)) - (d1*(nA*nB));
				c2 /= detr;


			}


			return out;
		}
	}

	/*! \brief This method  returns the closest points of two lines.
	*
	*	\details Based on http://paulbourke.net/geometry/pointlineplane/
	*	\param		[in]		a0		- first input point on line A.
	*	\param		[in]		a1		- second input point on line A.
	*	\param		[in]		b0		- first input point on line B.
	*	\param		[in]		b1		- second input point on line B.
	*	\param		[out]		uA		- line parameter for closest point on A .
	*	\param		[out]		uB		- line parameter for closest point on B .
	*	\return					bool	- true if the planes intersect.
	*	\since version 0.0.1
	*/

	bool line_lineClosestPoints(zVector &a0, zVector &a1, zVector &b0, zVector &b1, double &uA, double &uB)
	{
		bool out = false;

		zVector u = a1 - a0;
		zVector v = b1 - b0;
		zVector w = a0 - b0;

		/*printf("\n u  : %1.2f %1.2f %1.2f ", u.x, u.y, u.z);
		printf("\n v  : %1.2f %1.2f %1.2f ", v.x, v.y, v.z);
		printf("\n w  : %1.2f %1.2f %1.2f ", w.x, w.y, w.z);*/

		double uu = u * u;
		double uv = u * v;
		double vv = v * v;
		double uw = u * w;
		double vw = v * w;

		/*	printf("\n uu  : %1.4f  ", uu);
		printf("\n uv  : %1.4f  ", uv);
		printf("\n vv  : %1.4f  ", vv);
		printf("\n uw  : %1.4f  ", uw);
		printf("\n vw  : %1.4f  ", vw);*/

		double denom = 1 / (uu*vv - uv * uv);
		//printf("\n denom: %1.4f ", denom);

		if (denom != 0)
		{
			uA = (uv*vw - vv * uw) * denom;
			uB = (uu*vw - uv * uw) * denom;

			//printf("\n tA: %1.4f tB: %1.4f ", tA, tB);

			out = true;
		}

		return out;
	}

	/*! \brief This method  returns the intersection of two lines which is  point.
	*
	*	\details Based on http://paulbourke.net/geometry/pointlineplane/
	*	\param		[in]		p1				- first input point on line A.
	*	\param		[in]		p2				- second input point on line A.
	*	\param		[in]		planeNorm		- plane normal.
	*	\param		[in]		p3				- point in plane.
	*	\param		[out]		intersectionPt	- intersection point .
	*	\return					bool			- true if the line and plane intersect.
	*	\since version 0.0.1
	*/

	bool line_PlaneIntersection(zVector &p1, zVector &p2, zVector &planeNorm, zVector &p3, zVector &intersectionPt)
	{
		bool out = false;

		//printf("\n p1: %1.2f  %1.2f  %1.2f  ", p1.x, p1.y, p1.z);
		//printf("\n p2: %1.2f  %1.2f  %1.2f  ", p2.x, p2.y, p2.z);
		//printf("\n p3: %1.2f  %1.2f  %1.2f  ", p3.x, p3.y, p3.z);
		//printf("\n n: %1.2f  %1.2f  %1.2f  ", planeNorm.x, planeNorm.y, planeNorm.z);

		zVector p31 = p3 - p1;
		zVector p21 = p2 - p1;

		double denom = (planeNorm * (p21));

		//printf("\n denom: %1.2f " , denom);

		if (denom != 0)
		{
			double u = (planeNorm * (p31)) / denom;

			//printf("\n u : %1.2f ", u);

			if (u >= 0 && u <= 1)
			{
				out = true;
				double lenP21 = p21.length();
				p21.normalize();

				intersectionPt = (p21 * lenP21 * u);
				intersectionPt += p1;
			}
		}


		return out;
	}

	/*! \brief This method returns the area of triagle defined by the two input zVectors.
	*
	*	\param		[in]		p1				- first input point of triangle.
	*	\param		[in]		p2				- second input point of triangle.
	*	\param		[in]		p3				- second input point of triangle.
	*	\return					double			- area of triangle defirned by the vectors.
	*	\since version 0.0.1
	*/

	double triangleArea(zVector &v1, zVector &v2, zVector &v3)
	{
		double area = 0;

		zVector e12 = v2 - v1;
		zVector e13 = v3 - v1;

		area = ((e12^e13).length() * 0.5);

		return area;
	}

	//--------------------------
	//---- zMATRIX METHODS
	//--------------------------

	
	/*! \brief This method computes the tranformation to the world space of the input 4x4 matrix.
	*
	*	\tparam				T			- Type to work with standard c++ numerical datatypes.
	*	\param		[in]	inMatrix	- input zMatrix to be transformed.
	*	\return 			zMatrix		- world transformation matrix.
	*/
	
	template <typename T>
	zMatrix<T> toWorldMatrix(zMatrix<T> &inMatrix)
	{
		if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
		if (inMatrix.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");

		zMatrix<T> outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));


		outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
		outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
		outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = Cen.x; outMatrix(1, 3) = Cen.y; outMatrix(2, 3) = Cen.z;

		return outMatrix;
	}

	/*! \brief This method computes the tranformation to the local space of the input 4x4 matrix.
	*
	*	\tparam				T			- Type to work with standard c++ numerical datatypes.
	*	\param		[in]	inMatrix	- input 4X4 zMatrix to be transformed.
	*	\return 			zMatrix		- world transformation matrix.
	*/
	
	template <typename T>
	zMatrix<T> toLocalMatrix(zMatrix<T> &inMatrix)
	{
		if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
		if (inMatrix.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");
		
		zMatrix<T> outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));

		zVector orig(0, 0, 0);
		zVector d = Cen - orig;

		outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
		outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
		outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);

		

		return outMatrix;
	}

	/*! \brief This method computes the tranformation from one 4X4 matrix to another.
	*
	*	\tparam				T			- Type to work with standard c++ numerical datatypes.
	*	\param		[in]	from		- input 4X4 zMatrix.
	*	\param		[in]	to			- input 4X4 zMatrix.
	*	\return 			zMatrix		- transformation matrix.
	*/

	template <typename T>
	zMatrix<T> PlanetoPlane(zMatrix<T> &from, zMatrix<T> &to)
	{
		if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
		if (inMatrix.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");

		zMatrix<T> world = toWorldMatrix(to);
		zMatrix<T> local = toLocalMatrix(from);

		zMatrix<T> out = world * local;

		return out;
	}

	
	/*! \brief This method computes the tranformation to change the baseis from one 4X4 matrix to another.
	*
	*	\tparam				T			- Type to work with standard c++ numerical datatypes.
	*	\param		[in]	from		- input 4X4 zMatrix.
	*	\param		[in]	to			- input 4X4 zMatrix.
	*	\return 			zMatrix		- transformation matrix.
	*/
	
	template <typename T>
	zMatrix<T> ChangeBasis(zMatrix<T> &from, zMatrix<T> &to)
	{
		return toLocalMatrix(to) * toWorldMatrix(from);
	}

	/*! \brief This method computes the input target as per the input new basis.
	*
	*	\param [in]		infilename			- input file name including the directory path.
	*/

	/*! \brief This method computes the input target as per the input new basis.
	*
	*	\tparam				T			- Type to work with standard c++ numerical datatypes.
	*	\param		[in]	target		- input 4X4 zMatrix.
	*	\param		[in]	newBasis	- input 4X4 zMatrix.
	*	\return 			zMatrix		- new target matrix.
	*/
	
	template <typename T>
	zMatrix<T> target_newBasis(zMatrix<T> &target, zMatrix<T> &newBasis)
	{		
		zMatrix<T> C = newBasis;
		zMatrix<T> C_inverse;
		
		bool chkInvr = C.inverseMatrix(C_inverse);

		if(!chkInvr) throw std::invalid_argument("input Matrix is singular and doesnt have an inverse matrix.");

		Matrix4f targ_newbasis;
		targ_newbasis.setIdentity();

		targ_newbasis = C_inverse * target;

		return targ_newbasis;	

	}



	//--------------------------
	//---- MATRIX METHODS USING EIGEN
	//--------------------------


	/*! \brief This method return a 4X4 matrix of the best fit plane for the given points using Principal Component Analysis
	*
	*	\param		[in]	points		- input points.
	*	\return 			zMatrixd	- Best fit plane as a 4X4 matrix.
	*/

	zMatrixd getBestFitPlane(vector<zVector>& points)
	{

		zMatrixd out;

		// compute average point
		zVector averagePt;

		for (int i = 0; i < points.size(); i++)
		{
			averagePt += points[i];

		}

		averagePt /= points.size();
	
		// compute covariance matrix 
		SelfAdjointEigenSolver<Matrix3f> eigensolver;
		Matrix3f covarianceMat;

		for (int i = 0; i <3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
		
				float val = 0;
				for (int k = 0; k < points.size(); k++)
				{
					val += (points[k].getComponent(i) - averagePt.getComponent(i)) * (points[k].getComponent(j) - averagePt.getComponent(j));
				}

				if (val > 0.001) val /= (points.size() - 1);
				else val = 0.00;

				covarianceMat(i, j) = val;
			}

		}

		eigensolver.compute(covarianceMat);
		if (eigensolver.info() != Success) abort();

		vector<double> X = { eigensolver.eigenvectors().col(2)(0), eigensolver.eigenvectors().col(2)(1), eigensolver.eigenvectors().col(2)(2), 1 };
		vector<double> Y = { eigensolver.eigenvectors().col(1)(0), eigensolver.eigenvectors().col(2)(1), eigensolver.eigenvectors().col(2)(2), 1 };
		vector<double> Z = { eigensolver.eigenvectors().col(0)(0), eigensolver.eigenvectors().col(2)(1), eigensolver.eigenvectors().col(2)(2), 1 };
		vector<double> O = { averagePt.x, averagePt.y, averagePt.z, 1 };
		
		out.setCol(0, X);
		out.setCol(1, Y);
		out.setCol(2, Z);
		out.setCol(3, O);

		return out;
	}


	/*! \brief This method computes the bounding box for the given points using PCA
	*
	*	\param		[in]	points			- input points.
	*	\param		[out]	minBB			- lower bounds as zVector
	*	\param		[out]	maxBB			- upper bounds as zVector
	*/
	
	void boundingboxPCA(vector<zVector> points, zVector &minBB, zVector &maxBB, zVector &minBB_local, zVector &maxBB_local)
	{
		zMatrixd bPlane_Mat = getBestFitPlane(points);

		// translate points to local frame
		zMatrixd bPlane_Mat_local = toLocalMatrix(bPlane_Mat);

		for (int i = 0; i < points.size(); i++)
		{
			zVector new_pos = points[i] * bPlane_Mat_local;
			points[i] = new_pos;
		}

		// compute boundings in local frame
		
		getBounds(points, minBB_local, maxBB_local);


		// translate points to world frame
		zMatrixd bPlane_Mat_world = toWorldMatrix(bPlane_Mat);

		for (int i = 0; i < points.size(); i++)
		{
			zVector new_pos = points[i] * bPlane_Mat_world;
			points[i] = new_pos;

		}

		minBB = minBB_local * bPlane_Mat_world;
		maxBB = maxBB_local * bPlane_Mat_world;	

	}

	


	/** @}*/ 

}
