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

#ifndef ZSPACE_UTILS_CORE_H
#define ZSPACE_UTILS_CORE_H

#pragma once

#include<headers/zCore/base/zEnumerators.h>
#include<headers/zCore/base/zDefinitions.h>
#include <headers/zCore/base/zVector.h>
#include <headers/zCore/base/zColor.h>
#include <headers/zCore/base/zMatrix.h>
#include <headers/zCore/base/zDate.h>
#include <headers/zCore/base/zTransformationMatrix.h>
#include <headers/zCore/base/zQuaternion.h>
#include <headers/zCore/base/zDomain.h>
#include <headers/zCore/base/zTypeDef.h>
#include <headers/zCore/base/zExtern.h>

#include <string.h>
#include <vector>
#include <algorithm> 
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <stdio.h>
#include <time.h> 
#include <ctype.h>
#include <numeric>
#include <map>
#include <unordered_map>
#include <iostream>

using namespace std;

#include<headers/zCore/utilities/zUtilsBMP.h>
#include<headers/zCore/utilities/zUtilsPointerMethods.h>

#ifndef __CUDACC__	

	#include <depends/tooJPEG/toojpeg.h>
	#include <depends/lodePNG/lodepng.h>


#ifndef QUICKHULL_H
	#define _CRT_SECURE_NO_WARNINGS
#endif 

#endif

#if defined(QUICKHULL_H)  || defined(ZSPACE_UNREAL_INTEROP)
	// All defined OK so do nothing
#else
	#define QUICKHULL_IMPLEMENTATION
#include <depends/quickhull/quickhull.h>

#include <direct.h>

#endif


namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \class zUtilsCore
	*	\brief A core utility class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class ZSPACE_CORE zUtilsCore
	{
	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zUtilsCore();

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE ~zUtilsCore();

		//--------------------------
		//---- WINDOWS UTILITY  METHODS
		//--------------------------

		/*! \brief This method returns the number of files in the input folder path.
		*
		*	\param		[in]	dirPath			- input directory path.
		*	\return				int				- number of files in the input folder.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST int getNumfiles(string dirPath);

		/*! \brief This method returns the number of files in the input directory of the input file type.
		*
		*	\param		[in]	dirPath			- input directory path.
		*	\param		[in]	type			- input zFileTpye.
		*	\return				int				- number of files in the input folder.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST int getNumfiles_Type(string dirPath, zFileTpye type = zJSON);
		
			
		/*! \brief This method gets all the files on the input file type in the input directory sorted by time of creation.
		*
		*	\param		[out]	fpaths			- container of file paths.
		*	\param		[in]	dirPath			- input directory path.
		*	\param		[in]	type			- input zFileTpye.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST void getFilesFromDirectory(zStringArray &fpaths, string dirPath, zFileTpye type = zJSON);

		//--------------------------
		//---- STRING METHODS
		//--------------------------

		/*! \brief This method splits the input string based on the input delimiter.
		*
		*	\param		[in]	str				- input string to be split.
		*	\param		[in]	delimiter		- input delimiter.
		*	\return				vector<string>	- list of elements of split string.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE_HOST vector<string> splitString(const string& str, const string& delimiter);
		

		//--------------------------
		//---- NUMERICAL METHODS
		//--------------------------

		/*! \brief This method maps the input value from the input domain to output domain.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	value			- input value to be mapped.
		*	\param		[in]	inputMin		- input domain minimum.
		*	\param		[in]	inputMax		- input domain maximum.
		*	\param		[in]	outputMin		- output domain minimum.
		*	\param		[in]	outputMax		- output domain maximum.
		*	\return				double			- mapped value.
		*	\since version 0.0.1
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T ofMap(T value, T inputMin, T inputMax, T outputMin, T outputMax);

		/*! \brief This method maps the input value from the input domain to output domain.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	value			- input value to be mapped.
		*	\param		[in]	inDomain		- input domain.
		*	\param		[in]	outDomain		- output domain.
		*	\return				double			- mapped value.
		*	\since version 0.0.2
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T ofMap(T value, zDomain<T> inDomain, zDomain<T> outDomain);

		/*! \brief This method clamps the input value to the input domain.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	value			- input value to be mapped.
		*	\param		[in]	inputMin		- input domain minimum.
		*	\param		[in]	inputMax		- input domain maximum.
		*	\return				double			- clamped value.
		*	\since version 0.0.1
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T ofClamp(T value, T inputMin, T inputMax);

		/*! \brief This method returns the minimum of the two input values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val0			- input value 1.
		*	\param		[in]	val1			- input value 2.
		*	\return 			double			- minimum value
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T zMin(T val0, T val1);

		/*! \brief This method returns the minimum of the input container values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- container of values.
		*	\return 			double			- minimum value
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE_HOST T zMin(vector<T> &vals);

		/*! \brief This method returns the maximum of the two input values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val0			- input value 1.
		*	\param		[in]	val1			- input value 2.
		*	\return 			double			- maximum value
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T zMax(T val0, T val1);

		/*! \brief This method returns the maximun of the input container values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- container of values.
		*	\return 			double			- maximum value
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE_HOST T zMax(vector<T> &vals);

		/*! \brief This method returns a random number in the input domain.
		*
		*	\param		[in]	min		- domain minimum value.
		*	\param		[in]	max		- domain maximum value.
		*	\return				int		- random number between the 2 input values.
		*/
		ZSPACE_CUDA_CALLABLE int randomNumber(int min, int max);

		/*! \brief This method returns a random number in the input domain.
		*
		*	\param		[in]	min			- domain minimum value.
		*	\param		[in]	max			- domain maximum value.
		*	\return				double		- random number between the 2 input values.
		*/
		ZSPACE_CUDA_CALLABLE double randomNumber_double(double min, double max);

		/*! \brief This method returns the factorised value to the input precision.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes except integer.
		*	\param		[in]	inputValue		- input value.
		*	\param		[in]	precision		- precision or number of digits after the decimal point.
		*	\return				T				- factorised value.
		*	\since version 0.0.1
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE T factorise(T inputValue, int precision = 3);

		/*! \brief This method returns the factorised vector to the input precision.
		*
		*	\param		[in]	inputValue		- input vector.
		*	\param		[in]	precision		- precision or number of digits after the decimal point.
		*	\return				zVector				- factorised vector.
		*	\since version 0.0.1
		*/		
		ZSPACE_CUDA_CALLABLE zVector factoriseVector(zVector inputValue, int precision = 3);

		/*! \brief This method checks if the input value is repeated in input container.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	inVal			- input value.
		*	\param		[in]	values			- input container of values to be checked against.
		*	\param		[in]	precision		- precision or number of digits after the decimal point.
		*	\param		[out]	index			- index of the first repeat element.
		*	\return				bool			- true if there is a repeat element.
		*	\since version 0.0.1
		*/
		template <typename T>
		ZSPACE_CUDA_CALLABLE_HOST bool checkRepeatElement(T &inVal, vector<T> values, int &index, int precision = 3);

		/*! \brief This method checks if the input vector is repeated in input container.
		*
		*	\param		[in]	inVal			- input vector.
		*	\param		[in]	values			- input container of vectors to be checked against.
		*	\param		[in]	precision		- precision or number of digits after the decimal point.
		*	\param		[out]	index			- index of the first repeat element.
		*	\return				bool			- true if there is a repeat element.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE_HOST bool checkRepeatVector(zVector &inVec, vector<zVector> values, int &index, int precision = 3);

		//--------------------------
		//---- MAP METHODS 
		//--------------------------

		/*! \brief This method checks if the input hashkey exists in the map.
		*
		*	\param		[in]	hashKey		- input string to check.
		*	\param		[in]	map			- input map.
		*	\param		[out]	outVal		- index of the string in the map if it exists.
		*	\return				bool		- true if the string exists in the map.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE bool existsInMap(string hashKey, unordered_map<string, int> map, int &outVal);

		/*! \brief This method checks if the input position exists in the map.
		*
		*	\param		[in]	positionVertex	- input position vertex map.
		*	\param		[in]	pos				- input position.
		*	\param		[in]	precisionFac	- precision factor of the points to check.
		*	\param		[out]	outVertexId		- index of the position in the map if it exists.
		*	\return				bool			- true if the position exists in the map.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE_HOST bool vertexExists(unordered_map<string, int>& positionVertex, zVector & pos, int precisionFac, int & outVertexId);

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]	positionVertex		- input position vertex map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]	precisionFac		- precision factor of the points to check.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE_HOST void addToPositionMap(unordered_map<string, int>& positionVertex, zVector &pos, int index, int precisionFac);
			   
		//--------------------------
		//---- VECTOR METHODS 
		//--------------------------

		/*! \brief This method returns the minimum of the input container of zVectors.
		*
		*	\param		[in]	vals			- container of zVectors.
		*	\return 			zVectors		- vector with minimum length
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST zVector zMin(vector<zVector> &vals);

		/*! \brief This method returns the maximum of the input container of zVectors.
		*
		*	\param		[in]	vals			- container of zVectors.
		*	\return 			zVectors		- vector with maximum length
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST zVector zMax(vector<zVector> &vals);

		/*! \brief This method maps the input value from the input domain to output domain.
		*
		*	\param		[in]	value			- input vector to be mapped.
		*	\param		[in]	inputMin		- input domain minimum.
		*	\param		[in]	inputMax		- input domain maximum.
		*	\param		[in]	outputMin		- output vector domain minimum.
		*	\param		[in]	outputMax		- output vector domain maximum.
		*	\return				double			- mapped vector.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector ofMap(float value, float inputMin, float inputMax, zVector outputMin, zVector outputMax);

		/*! \brief This method maps the input value from the input domain to output domain.
		*
		*	\param		[in]	value			- input vector to be mapped.
		*	\param		[in]	inputDomain		- input domain of double.
		*	\param		[in]	outDomain		- output domain of vectors.
		*	\return				double			- mapped vector.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector ofMap(float value, zDomainFloat inputDomain, zDomainVector outDomain);

		/*! \brief This method a zVector from the input matrix row.
		*
		*	\param		[in]		zMatrixd	- input matrix. number of columns need to be 3 or 4.
		*	\param		[in]		rowIndex	- row index to be extracted.
		*	\return					zVector		- zVector of the row matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector fromMatrix4Row(zMatrix4 &inMatrix, int rowIndex = 0);

		/*! \brief This method returns extracts a zVector from the input matrix column.
		*
		*	\param		[in]		zMatrixd	- input matrix. number of rows need to be 3 or 4.
		*	\param		[in]		rowIndex	- row index to be extracted.
		*	\return					zVector		- zVector of the column matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector fromMatrix4Column(zMatrix4 &inMatrix, int colIndex);

		/*! \brief This method returns the factorised vector to the input precision.
		*
		*	\param		[in]		inVector		- input vector.
		*	\param		[in]		precision		- precision or number of digits after the decimal point.
		*	\return					zVector			- factorised vector.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector factorise(zVector &inVector, int precision = 3);

		/*! \brief This method scales the input point cloud by the input scale factor.
		*
		*	\param		[in]	inPoints			- input point cloud.
		*	\param		[out]	scaleFac			- scale factor.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST void scalePointCloud(vector<zVector> &inPoints, double scaleFac);

		/*! \brief This method computes the center of the input point cloud.
		*
		*	\param		[in]	inPoints			- input point cloud.
		*	\param		[out]	center				- center of point cloud.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST void getCenter_PointCloud(vector<zVector> &inPoints, zVector &center);

		/*! \brief This method return index of the closest point in the input container to the input position.
		*
		*	\param		[in]	pos				- input position.
		*	\param		[in]	inPositions		- input container of positions.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST int getClosest_PointCloud(zVector &pos, vector<zVector> inPositions);

		/*! \brief This method returns the bounds of the input list points.
		*
		*	\param  	[in]	inPoints	- input container of positions.
		*	\param  	[out]	minBB		- stores zVector of bounding box minimum.
		*	\param		[out]	maxBB		- stores zVector of bounding box maximum.
		*/
		ZSPACE_CUDA_CALLABLE_HOST void getBounds(vector<zVector> &inPoints, zVector &minBB, zVector &maxBB);

		/*! \brief This method returns the bounds of the input list points.
		*
		*	\param  	[in]	inPoints	- input pointer to container of positions.
		*	\param  	[in]	numPoints	- number of points in the container.
		*	\param  	[out]	minBB		- stores zVector of bounding box minimum.
		*	\param		[out]	maxBB		- stores zVector of bounding box maximum.
		*/
		ZSPACE_CUDA_CALLABLE void getBounds(zVector* inPoints, int numPoints, zVector &minBB, zVector &maxBB);

		/*! \brief This method computes the distances in X,Y,Z for the input bounds.
		*
		*	\param		[in]	minBB			- lower bounds as zVector.
		*	\param		[in]	maxBB			- upper bounds as zVector
		*	\param		[out]	Dims			- distances in X,Y,Z axis in local frame
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zVector getDimsFromBounds(zVector &minBB, zVector &maxBB);

		/*! \brief This method checks if the input position is inside the input bounds.
		*
		*	\param		[in]	inPoint			- input point.
		*	\param		[in]	minBB			- lower bounds as zVector.
		*	\param		[in]	maxBB			- upper bounds as zVector
		*	\return				bool			- true if input position is inside the bounds.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool pointInBounds(zVector &inPoint, zVector &minBB, zVector &maxBB);

		/*! \brief This method computes the inverse distance weights of the input positions container on the input point .
		*
		*	\details	bsaed on http://www.gitta.info/ContiSpatVar/en/html/Interpolatio_learningObject2.xhtml.
		*	\param		[in]	inPos			- input position.
		*	\param		[in]	positions		- input container of positions.
		*	\param		[in]	power			- power of the distance.
		*	\param		[out]	weights			- influence Weights between 0 and 1.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST void getDistanceWeights(zPoint& inPos, zPointArray positions, double power, zDoubleArray &weights);
		
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
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool plane_planeIntersection(zVector &nA, zVector &nB, zVector &pA, zVector &pB, zVector &outP1, zVector &outP2);

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
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool line_lineClosestPoints(zVector &a0, zVector &a1, zVector &b0, zVector &b1, double &uA, double &uB);

		/*! \brief This method  returns the closest points of two lines.
		*
		*	\details Based on http://paulbourke.net/geometry/pointlineplane/ and http://paulbourke.net/geometry/pointlineplane/lineline.c
		*	\param		[in]		p1		- first input point on line A.
		*	\param		[in]		p2		- second input point on line A.
		*	\param		[in]		p3		- first input point on line B.
		*	\param		[in]		p4		- second input point on line B.
		*	\param		[out]		uA		- line parameter for closest point on A .
		*	\param		[out]		uB		- line parameter for closest point on B .
		*	\param		[out]		pA		- closest point on A .
		*	\param		[out]		pA		- closest point on B .
		*	\return					bool	- true if the lines intersect.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool line_lineClosestPoints(zVector &p1, zVector &p2, zVector &p3, zVector &p4, double &uA, double &uB, zVector &pA, zVector &pB);

		/*! \brief This method  returns the intersection of two lines which is  point.
		*
		*	\details Based on http://paulbourke.net/geometry/pointlineplane/
		*	\param		[in]		p1				- first input point on line A.
		*	\param		[in]		p2				- second input point on line A.
		*	\param		[in]		planeNorm		- plane normal.
		*	\param		[in]		p3				- point in plane.
		*	\param		[out]		intersectionPt	- intersection point .
		*	\return					bool			- true if the line and plane intersect.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool line_PlaneIntersection(zVector &p1, zVector &p2, zVector &planeNorm, zVector &p3, zVector &intersectionPt);


		/*! \brief This method  returns the intersection of two lines which is  point.
		*
		*	\details Based on https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d/42752998#42752998
		*	\param		[in]		a				- first point on triangle.
		*	\param		[in]		b				- second point on triangle.
		*	\param		[in]		c				- third point on triangle.
		*	\param		[in]		d				- direction of ray.
		*	\param		[in]		o				- origin of ray.
		*	\param		[out]		intersectionPt	- intersection point, if it exists.
		*	\return					bool			- true if the line and plane intersect.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE bool ray_triangleIntersection( zPoint &a, zPoint &b, zPoint &c, zVector & d, zPoint & o , zPoint &intersectionPt);


		/*! \brief This method returns the area of triagle defined by the three input zVectors.
		*
		*	\param		[in]		p1				- first input point of triangle.
		*	\param		[in]		p2				- second input point of triangle.
		*	\param		[in]		p3				- second input point of triangle.
		*	\return					double			- area of triangle defirned by the vectors.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double getTriangleArea(zVector &v1, zVector &v2, zVector &v3);

		/*! \brief This method returns the signed volume of the tetrahedron formed by the three input zVectors and the origin.
		*
		*	\details Based on http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf and http://www.dillonbhuff.com/?p=345
		*	\param		[in]		p1				- first input point of triangle.
		*	\param		[in]		p2				- second input point of triangle.
		*	\param		[in]		p3				- second input point of triangle.
		*	\return					double			- volume of tetrahedron formed by the three input vectors  and the origin.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double getSignedTriangleVolume(zVector &v1, zVector &v2, zVector &v3);

		/*! \brief This method checks if the given input points liess within the input triangle.
		*
		*	\details based on http://blackpawn.com/texts/pointinpoly/default.html
		*	\param		[in]	pt			- zVector holding the position information of the point to be checked.
		*	\param		[in]	t0,t1,t2	- zVector holding the position information for the 3 points of the triangle.
		*	\return				bool		- true if point is inside the input triangle.
		*/
		ZSPACE_CUDA_CALLABLE bool pointInTriangle(zVector &pt, zVector &t0, zVector &t1, zVector &t2);

		/*! \brief This method computes the minimum distance between a point and edge and the closest Point on the edge.
		*
		*	\details based on http://paulbourke.net/geometry/pointlineplane/
		*	\param	[in]	pt			- point
		*	\param	[in]	e0			- start point of edge.
		*	\param	[in]	e1			- end point of edge.
		*	\param	[out]	closest_Pt	- closest point on edge to the input point.
		*	\return			minDist		- distance to closest point.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double minDist_Edge_Point(zVector & pt, zVector & e0, zVector & e1, zVector & closest_Pt);

		/*! \brief This method computes the minimum distance between a point and a plane.
		*
		*	\details based on http://paulbourke.net/geometry/pointlineplane/
		*	\param	[in]	pA			- point
		*	\param	[in]	pB			- point on the plane.
		*	\param	[in]	norm		- normal of the plane.
		*	\return			minDist		- minimum distance to plane.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE double minDist_Point_Plane(zVector & pA, zVector & pB, zVector & norm);

		/*! \brief This method gets the bary-center of the input positions based on the input weights.
		*
		*	\param		[in]	inPositions		- input container of positions.
		*	\param		[in]	weights			- input container of weights.
		*	\return				zVector			- bary-center.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE_HOST zVector getBaryCenter(zPointArray &inPositions, zDoubleArray& weights);

		/*! \brief This method converts spherical coordinates to cartesian coordinates.
		*
		*	\param		[in]	azimuth			- input azimuth coordinate.
		*	\param		[in]	altitude		- input altitude coordinate.
		*	\param		[in]	radius			- input radius.
		*	\return				zVector			- output cartesian point.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zPoint sphericalToCartesian(double azimuth, double altitude, double radius);

		/*! \brief This method converts cartesian coordinates to spherical coordinates.
		*
		*	\param		[in]	inVec			- input cartesioan point.
		*	\param		[in]	radius			- input radius.
		*	\param		[out]	azimuth			- output azimuth coordinate.
		*	\param		[out]	altitude		- output altitude coordinate.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void cartesianToSpherical(zPoint &inVec, double &radius, double &azimuth, double &altitude);

		//--------------------------
		//---- 4x4 zMATRIX  TRANSFORMATION METHODS
		//--------------------------

		/*! \brief This method inputs the vector values at the input index of the 4X4 tranformation matrix.
		*
		*	\param		[in]	inMatrix	- input zMatrix .
		*	\param		[in]	inVec		- input vector.
		*	\param		[in]	index		- column index.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void setColfromVector(zMatrix4 &inMatrix, zVector &inVec, int index);

		/*! \brief This method inputs the vector values at the input index of the 4X4 tranformation matrix.
		*
		*	\param		[in]	inMatrix	- input zMatrix .
		*	\param		[in]	inVec		- input vector.
		*	\param		[in]	index		- column index.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void setRowfromVector(zMatrix4 &inMatrix, zVector &inVec, int index);

		/*! \brief This method returns the 4X4 tranformation matrix to change the origin to the input vector.
		*
		*	\param		[out]	inMatrix	- input zMatrix .
		*	\param		[in]	X			- input X Axis as a vector.
		*	\param		[in]	Y			- input Y Axis as a vector.
		*	\param		[in]	Z			- input Z Axis as a vector.
		*	\param		[in]	O			- input origin as a vector.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void setTransformfromVectors(zMatrix4 &inMatrix, zVector &X, zVector &Y, zVector &Z, zVector &O);

		/*! \brief This method computes the tranformation to the world space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 toWorldMatrix(zMatrix4 &inMatrix);

		/*! \brief This method computes the tranformation to the local space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input 4X4 zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 toLocalMatrix(zMatrix4 &inMatrix);

		/*! \brief This method computes the tranformation from one 4X4 matrix to another.
		*
		*	\param		[in]	from		- input 4X4 zMatrix.
		*	\param		[in]	to			- input 4X4 zMatrix.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 PlanetoPlane(zMatrix4 &from, zMatrix4 &to);

		/*! \brief This method computes the tranformation to change the baseis from one 4X4 matrix to another.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	from		- input 4X4 zMatrix.
		*	\param		[in]	to			- input 4X4 zMatrix.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE zMatrix4 ChangeBasis(zMatrix4 &from, zMatrix4 &to);

		/*! \brief This method computes the input target as per the input new basis.
		*
		*	\param [in]		infilename			- input file name including the directory path.
		*	\since version 0.0.2
		*/


		//--------------------------
		//---- VECTOR METHODS GEOMETRY
		//--------------------------

		/*! \brief This method computes the points on a ellipse centered around world origin for input radius, and number of points.
		*
		*	\param		[in]		radius		- radius.
		*	\param		[in]		numPoints	- number of points in the the ellipse.
		*	\param		[out]		Pts			- points on circle.
		*	\param		[in]		localPlane	- orientation plane, by default a world plane.
		*	\param		[out]		xFactor		- the factor of scaling in x direction. For a circle both xFactor and yFactor need to be equal.
		*	\param		[out]		yFactor		- the factor of scaling in y direction. For a circle both xFactor and yFactor need to be equal.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void getEllipse(double radius, int numPoints, zPointArray &Pts, zMatrix4 localPlane = zMatrix4(), double xFactor = 1.0, double yFactor = 1.0);

		/*! \brief This method computes the points on a rectangle for input dimensions centers around the world origin.
		*
		*	\param		[in]		dims			- dimensions of rectangle.
		*	\param		[out]		rectanglePts	- points on rectangle.
		*	\param		[in]		localPlane		- orientation plane, by default a world plane.
		*	\since version 0.0.2
		*/
		ZSPACE_CUDA_CALLABLE void getRectangle(zVector dims, zPointArray &rectanglePts, zMatrix4 localPlane = zMatrix4());

		//--------------------------
		//---- COLOR  METHODS
		//--------------------------
			
		/*! \brief This method returns the average color of the two input colors.
		*
		*	\param		[in]	c1		- input color 1.
		*	\param		[in]	c2		- input color 2.
		*	\param		[in]	type	- color type - zRGB / zHSV.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor averageColor(zColor c1, zColor c2, zColorType type);

		/*! \brief This method returns the average color of the input color container.
		*
		*	\param		[in]	c1		- input color container.
		*	\param		[in]	c2		- input color 2.
		*	\param		[in]	type	- color type - zRGB / zHSV.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor averageColor(zColorArray &c1, zColorType type);

		/*! \brief This method returns the blend color based on the input value, domain and the color domains.
		*
		*	\param		[in]	value			- input value to be mapped.
		*	\param		[in]	inputMin		- input domain minimum.
		*	\param		[in]	inputMax		- input domain maximum.
		*	\param		[in]	cMin			- input color domain minimum.
		*	\param		[in]	cMax			- input color domain maximum.
		*	\param		[in]	type			- color type - zRGB / zHSV.
		*	\return				zColor			- output color.
		*	\since version 0.0.1
		*/
		ZSPACE_CUDA_CALLABLE zColor blendColor(float inputValue, zDomainFloat inDomain, zDomainColor outDomain, zColorType type);
		
#ifndef __CUDACC__

		//--------------------------
		//---- IMAGE  METHODS
		//--------------------------

		/*! \brief This method writes a BMP from the input matrix.
		*
		*	\param		[in]	mat				- input container of matrices.
		*	\param		[in]	path			- input path where to write the image to.
		*	\since version 0.0.4
		*/
		void matrixToBMP(vector<MatrixXf> &matrices, string path);

		/*! \brief This method writes a JPEG from the input matrix.
		*
		*	\param		[in]	mat				- input container of matrices.
		*	\param		[in]	path			- input path where to write the image to.
		*	\since version 0.0.4
		*/
		void matrixToJPEG(vector<MatrixXf> &matrices, string path);

		/*! \brief This method writes a PNG from the input matrix.
		*
		*	\param		[in]	mat				- input container of matrices.
		*	\param		[in]	path			- input path where to write the image to.
		*	\since version 0.0.4
		*/
		void matrixToPNG(vector<MatrixXf> &matrices, string path);

		/*! \brief This method read a PNG values to the output matrix.
		*
		*	\param		[out]	mat				- input container of matrices.
		*	\param		[in]	path			- input path where to write the image to.
		*	\since version 0.0.4
		*/
		void matrixFromPNG(vector<MatrixXf> &matrices, string path);

		//--------------------------
		//---- MATRIX METHODS USING EIGEN / ARMA
		//--------------------------

		/*! \brief This method return a 4X4 matrix of the best fit plane for the given points using Principal Component Analysis.  Works with Eigen matrix.
		*
		*	\param		[in]	points		- input points.
		*	\return 			zPlane		- Best fit plane as a 4X4 matrix, with XYZO stored in columns 0,1,2,3 respectively.
		*	\since version 0.0.2
		*/
		zPlane getBestFitPlane(zPointArray& points);

		/*! \brief This methods computes the projected points on the in input plane.
		*
		*	\param		[in]	points		- input points.
		*	\param		[out]	points		- projected points.
		*	\since version 0.0.4
		*/
		void getProjectedPoints_BestFitPlane(zPointArray& points, zPointArray &projectPoints);

		/*! \brief This method computes the bounding box for the given points using PCA.  Works with Eigen matrix.
		*
		*	\param		[in]	points			- input points.
		*	\param		[out]	minBB			- lower bounds as zVector
		*	\param		[out]	maxBB			- upper bounds as zVector
		*	\since version 0.0.2
		*/
		void boundingboxPCA(zPointArray points, zVector &minBB, zVector &maxBB, zVector &minBB_local, zVector &maxBB_local);

		/*! \brief This method computes the tranformation to the world space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform toWorldMatrix(zTransform &inMatrix);

		/*! \brief This method computes the tranformation to the local space of the input 4x4 matrix.
		*
		*	\param		[in]	inMatrix	- input 4X4 zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform toLocalMatrix(zTransform &inMatrix);

		/*! \brief This method computes the tranformation from one 4X4 matrix to another.
		*
		*	\param		[in]	from		- input 4X4 zMatrix.
		*	\param		[in]	to			- input 4X4 zMatrix.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform PlanetoPlane(zTransform &from, zTransform &to);

		/*! \brief This method computes the euclidean distance between two input row matricies.  The number of columns of m1 and m2 need to be equal.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	m1			- input zMatrix 1.
		*	\param		[in]	m2			- input zMatrix 2.
		*	\param		[in]	tolerance	- input tolerance for distance check.
		*	\return 			double		- euclidean distance.
		*	\since version 0.0.2
		*/
		float getEuclideanDistance(MatrixXf &m1, MatrixXf &m2, double tolerance = 0.001);


		//--------------------------
		//---- MATRIX  METHODS USING ARMADILLO
		//--------------------------
#ifndef USING_CLR


		/*! \brief This method returns the reduced row echelon form of the input matrix. Works with Armadillo matrix.
		*
		*	\details based on https://searchcode.com/codesearch/view/20327709/
		*	\param		[in]	A				- input armadillo matrix.
		*	\param		[in]	tol				- input tolerance.
		*	\return				mat				- output armadillo matrix in RREF form.
		*	\since version 0.0.3
		*/
		arma::mat rref(arma::mat A, double tol);

		//--------------------------
		//---- MATRIX  CAST METHODS USING ARMADILLO AND EIGEN
		//--------------------------

		/*! \brief This method converts a armadillo matrix to eigen matrix.
		*
		*	\details based on https://stackoverflow.com/questions/46700560/converting-an-armadillo-matrix-to-an-eigen-matrixd-and-vice-versa
		*	\param		[in]	arma_A			- input armadillo matrix.
		*	\return				MatrixXd		- output eigen matrix.
		*	\since version 0.0.3
		*/
		MatrixXd armaToEigen(mat &arma_A);

		/*! \brief This method converts a armadillo vector to eigen vector.
		*
		*	\details based on https://stackoverflow.com/questions/46700560/converting-an-armadillo-matrix-to-an-eigen-matrixd-and-vice-versa
		*	\param		[in]	arma_A			- input armadillo vector.
		*	\return				VectorXd		- output eigen vector.
		*	\since version 0.0.3
		*/
		VectorXd armaToEigen(arma::vec &arma_A);

		/*! \brief This method converts a eigen matrix to armadillo matrix.
		*
		*	\details based on https://stackoverflow.com/questions/46700560/converting-an-armadillo-matrix-to-an-eigen-matrixd-and-vice-versa
		*	\param		[in]	eigen_A			- input eigen matrix.
		*	\return				mat				- output armadillo matrix.
		*	\since version 0.0.3
		*/
		mat eigenToArma(MatrixXd &eigen_A);

		/*! \brief This method converts a eigen vector to armadillo vector.
		*
		*	\details based on https://stackoverflow.com/questions/46700560/converting-an-armadillo-matrix-to-an-eigen-matrixd-and-vice-versa
		*	\param		[in]	eigen_A			- input eigen vector.
		*	\return				vec				- output armadillo vector.
		*	\since version 0.0.3
		*/
		arma::vec eigenToArma(VectorXd &eigen_A);

#endif

	private:

		//--------------------------
		//---- PRIVATE IMAGE  METHODS
		//--------------------------

		/*! \brief This method encodes from raw pixels to an in-memory PNG file first, then write it to disk to PNG format.
		*
		*	\details based on example in https://github.com/lvandeve/lodepng 
		*	\param		[in]	filename		- input filename.
		*	\param		[in]	image			- input image pixel data.
		*	\param		[in]	width			- input image width.
		*	\param		[in]	height			- input image height.
		*	\since version 0.0.4
		*/
		void writePNG(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height);

		void readPNG(const char* filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height);

		//--------------------------
		//---- PRIVATE MATRIX  METHODS
		//--------------------------

#ifndef USING_CLR

		/*! \brief This utility method for reduced row echelon form, returns the max value and the pivot in the input column. Works with Armadillo matrix.
		*
		*	\details based on https://searchcode.com/codesearch/view/20327709/
		*	\param		[in]	A				- input armadillo matrix.
		*	\param		[in]	r				- input row index.
		*	\param		[in]	c				- input column index.
		*	\param		[out]	m				- output maximum value.
		*	\param		[out]	pivot			- output index of the maximum value.
		*	\since version 0.0.3
		*/
		void rref_max(mat &A, int &r, int &c, double &m, int &pivot);

		/*! \brief This utility method for reduced row echelon form, swaps the values of the input rows ( r & pivot). Works with Armadillo matrix.
		*
		*	\details based on https://searchcode.com/codesearch/view/20327709/
		*	\param		[in]	A				- input armadillo matrix.
		*	\param		[in]	r				- input row index.
		*	\param		[in]	c				- input column index.
		*	\param		[in]	pivot			- int pivot row index.
		*	\since version 0.0.3
		*/
		void rref_swapRows(mat &A, int &r, int &c, int &pivot);

		/*! \brief This utility method for reduced row echelon form,  normalises the input column values. Works with Armadillo matrix.
		*
		*	\details based on https://searchcode.com/codesearch/view/20327709/
		*	\param		[in]	A				- input armadillo matrix.
		*	\param		[in]	r				- input row index.
		*	\param		[in]	c				- input column index.
		*	\since version 0.0.3
		*/
		void rref_normaliseRow(mat &A, int &r, int &c);		

		/*! \brief This utility method for reduced row echelon form,  eliminates the input column values by making it zero. Works with Armadillo matrix.
		*
		*	\details based on https://searchcode.com/codesearch/view/20327709/
		*	\param		[in]	A				- input armadillo matrix.
		*	\param		[in]	r				- input row index.
		*	\param		[in]	c				- input column index.
		*	\since version 0.0.3
		*/
		void rref_eliminateColumn(mat &A, int &r, int &c);

#endif

#endif


	};	
	

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	//--------------------------
	//---- TEMPLATE METHODS INLINE DEFINITIONS
	//--------------------------

	//---- NUMERICAL METHODS

	template <typename T>
	inline T zUtilsCore::ofMap(T value, T inputMin, T inputMax, T outputMin, T outputMax)
	{
		return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
	}

	template <typename T>
	T zUtilsCore::ofMap(T value, zDomain<T> inDomain, zDomain<T> outDomain)
	{
		return ofMap(value, inDomain.min, inDomain.max, outDomain.min, outDomain.max);
	}

	template <typename T>
	inline T zUtilsCore::ofClamp(T value, T inputMin, T inputMax)
	{
		if (value > inputMax) return inputMax;
		else if (value < inputMin) return inputMin;
		else return value;
	}

	template <typename T>
	inline T zUtilsCore::zMin(T val0, T val1)
	{
		return (val0 < val1) ? val0 : val1;
	}

	template <typename T>
	inline T zUtilsCore::zMin(vector<T> &vals)
	{
		vector<T> sortVals = vals;
		std::sort(sortVals.begin(), sortVals.end());

		return sortVals[0];
	}

	template <typename T>
	inline T zUtilsCore::zMax(T val0, T val1)
	{
		return (val0 > val1) ? val0 : val1;
	}

	template <typename T>
	inline T zUtilsCore::zMax(vector<T> &vals)
	{
		vector<T> sortVals = vals;
		std::sort(sortVals.begin(), sortVals.end());

		return sortVals[sortVals.size() - 1];
	}

	template <typename T>
	inline T zUtilsCore::factorise(T inputValue, int precision)
	{
		double factor = pow(10, precision);
		return  std::round(inputValue *factor) / factor;
	}

	template <typename T>
	inline bool zUtilsCore::checkRepeatElement(T &inVal, vector<T> values, int &index, int precision)
	{
		bool out = false;
		index = -1;

		for (int i = 0; i < values.size(); i++)
		{
			T v1 = factorise(inVal, precision);
			T v2 = factorise(values[i], precision);

			if (v1 == v2)
			{
				out = true;

				index = i;
				break;
			}
		}

		return out;
	}






#endif


}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/utilities/zUtilsCore.cpp>
#endif

#endif