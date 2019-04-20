#pragma once

#include<headers/framework/core/zDefinitions.h>
#include <headers/framework/core/zVector.h>
#include <headers/framework/core/zColor.h>
#include <headers/framework/core/zMatrix.h>
#include <headers/framework/core/zDomain.h>

#include<headers/framework/utilities/zUtilsPointerMethods.h>

#include <vector>
#include <algorithm> 
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <stdio.h>
#include <direct.h>
#include <time.h> 
#include <ctype.h>
#include <numeric>
#include <map>
#include <unordered_map>

#include <iostream>



using namespace std;


#include<depends/Eigen/Core>
#include<depends/Eigen/Dense>
#include<depends/Eigen/Sparse>
#include<depends/Eigen/Eigen>
#include<depends/Eigen/Sparse>
using namespace Eigen;

namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
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
	class zUtilsCore
	{
	public:
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zUtilsCore() {}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zUtilsCore() {}


		//--------------------------
		//---- WINDOWS UTILITY  METHODS
		//--------------------------

		/*! \brief This method returns the number of files in the input folder path.
		*
		*	\param		[in]	dirPath			- input directory path.
		*	\return				int				- number of files in the input folder.
		*	\since version 0.0.2
		*/
		int getNumfiles(string dirPath)
		{
			int out = 0;

			for (const auto & entry : fs::directory_iterator(dirPath)) out++;

			return out;
		}


		/*! \brief This method returns the number of files in the input directory of the input file type.
		*
		*	\param		[in]	dirPath			- input directory path.
		*	\param		[in]	type			- input zFileTpye.
		*	\return				int				- number of files in the input folder.
		*	\since version 0.0.2
		*/
		int getNumfiles_Type(string dirPath, zFileTpye type = zJSON)
		{
			int out = 0;

			string extension;
			if (type == zJSON) extension = ".json";
			if (type == zOBJ) extension = ".obj";
			if (type == zTXT) extension = ".txt";
			if (type == zCSV) extension = ".csv";
			if (type == zBMP) extension = ".bmp";


			for (const auto & entry : fs::directory_iterator(dirPath))
			{
				if ((entry.path().extension()) == extension) out++;
			}

			return out;
		}


		
			
		/*! \brief This method gets all the files on the input file type in the input directory sorted by time of creation.
		*
		*	\param		[out]	fpaths			- container of file paths.
		*	\param		[in]	dirPath			- iinput directory path.
		*	\param		[in]	type			- input zFileTpye.
		*	\since version 0.0.2
		*/
		void getFilesFromDirectory(vector<string> &fpaths, string dirPath, zFileTpye type = zJSON)
		{
			fpaths.clear();
						
			if (dirPath.back() != '/') dirPath += "/";

			vector< fs::path> file_paths;

			string extension;
			if (type == zJSON) extension = ".json";
			if (type == zOBJ) extension = ".obj";
			if (type == zTXT) extension = ".txt";
			if (type == zCSV) extension = ".csv";
			if (type == zBMP) extension = ".bmp";

			for (const auto & entry : fs::directory_iterator(dirPath))
			{
				if ((entry.path().extension()) == extension) file_paths.push_back(entry.path());
			}

			// sort by time
			sort(file_paths.begin(), file_paths.end(),compare_time_creation);

			// store as string
			vector<fs::path>::iterator it;
			for (it = file_paths.begin(); it != file_paths.end(); ++it)fpaths.push_back(it->string());			
		}
		

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
		inline vector<string> splitString(const string& str, const string& delimiter)
		{
			vector<string> elements;
			// Skip delimiters at beginning.
			string::size_type lastPos = str.find_first_not_of(delimiter, 0);
			// Find first "non-delimiter".
			string::size_type pos = str.find_first_of(delimiter, lastPos);

			while (string::npos != pos || string::npos != lastPos)
			{
				// Found a token, add it to the vector.
				elements.push_back(str.substr(lastPos, pos - lastPos));
				// Skip delimiters.  Note the "not_of"
				lastPos = str.find_first_not_of(delimiter, pos);
				// Find next "non-delimiter"
				pos = str.find_first_of(delimiter, lastPos);
			}
			return elements;
		}

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
		T ofMap(T value, T inputMin, T inputMax, T outputMin, T outputMax)
		{
			return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
		}

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
		T ofMap(T value, zDomain<T> inDomain, zDomain<T> outDomain)
		{
			return ofMap(value, inDomain.min, inDomain.max, outDomain.min, outDomain.max);
		}

		/*! \brief This method returns the minimum of the two input values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val0			- input value 1.
		*	\param		[in]	val1			- input value 2.
		*	\return 			double			- minimum value
		*/
		template <typename T>
		T zMin(T val0, T val1)
		{
			return (val0 < val1) ? val0 : val1;
		}

		/*! \brief This method returns the minimum of the input container values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- container of values.
		*	\return 			double			- minimum value
		*/
		template <typename T>
		T zMin(vector<T> &vals)
		{
			vector<T> sortVals = vals;
			std::sort(sortVals.begin(), sortVals.end());

			return sortVals[0];
		}

		/*! \brief This method returns the maximum of the two input values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	val0			- input value 1.
		*	\param		[in]	val1			- input value 2.
		*	\return 			double			- maximum value
		*/
		template <typename T>
		T zMax(T val0, T val1)
		{
			return (val0 > val1) ? val0 : val1;
		}

		/*! \brief This method returns the maximun of the input container values.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	vals			- container of values.
		*	\return 			double			- maximum value
		*/
		template <typename T>
		T zMax(vector<T> &vals)
		{
			vector<T> sortVals = vals;
			std::sort(sortVals.begin(), sortVals.end());

			return sortVals[sortVals.size() - 1];
		}

		/*! \brief This method returns a random number in the input domain.
		*
		*	\param		[in]	min		- domain minimum value.
		*	\param		[in]	max		- domain maximum value.
		*	\return				int		- random number between the 2 input values.
		*/
		inline int randomNumber(int min, int max)
		{
			return rand() % (max - min + 1) + min;
		}

		/*! \brief This method returns a random number in the input domain.
		*
		*	\param		[in]	min			- domain minimum value.
		*	\param		[in]	max			- domain maximum value.
		*	\return				double		- random number between the 2 input values.
		*/
		inline double randomNumber_double(double min, double max)
		{
			return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
		}

		/*! \brief This method returns the factorised value to the input precision.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes except integer.
		*	\param		[in]	precision		- input value.
		*	\param		[in]	precision		- precision or number of digits after the decimal point.
		*	\return				T				- factorised value.
		*	\since version 0.0.1
		*/
		template <typename T>
		T factorise(T inputValue, int precision = 3)
		{
			double factor = pow(10, precision);
			return  round(inputValue *factor) / factor;
		}


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
		bool checkRepeatElement(T &inVal, vector<T> values, int &index, int precision = 3)
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
		inline bool existsInMap(string hashKey, unordered_map<string, int> map, int &outVal)
		{

			bool out = false;;

			std::unordered_map<std::string, int>::const_iterator got = map.find(hashKey);

			if (got != map.end())
			{
				out = true;
				outVal = got->second;
			}

			return out;
		}

		/*! \brief This method checks if the input position exists in the map.
		*
		*	\param		[in]	positionVertex	- input position vertex map.
		*	\param		[in]	pos				- input position.
		*	\param		[in]	precisionFac	- precision factor of the points to check.
		*	\param		[out]	outVertexId		- index of the position in the map if it exists.
		*	\return				bool			- true if the position exists in the map.
		*	\since version 0.0.1
		*/
		inline bool vertexExists(unordered_map<string, int>& positionVertex, zVector & pos, int precisionFac, int & outVertexId)
		{
			bool out = false;;

			double factor = pow(10, precisionFac);
			double x1 = round(pos.x *factor) / factor;
			double y1 = round(pos.y *factor) / factor;
			double z1 = round(pos.z *factor) / factor;

			string hashKey = (to_string(x1) + "," + to_string(y1) + "," + to_string(z1));
			std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


			if (got != positionVertex.end())
			{
				out = true;
				outVertexId = got->second;
			}


			return out;
		}


		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]	positionVertex		- input position vertex map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]	precisionFac		- precision factor of the points to check.
		*	\since version 0.0.1
		*/
		void addToPositionMap(unordered_map<string, int>& positionVertex, zVector &pos, int index, int precisionFac)
		{
			double factor = pow(10, precisionFac);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			positionVertex[hashKey] = index;
		}



		//--------------------------
		//---- VECTOR METHODS 
		//--------------------------

		/*! \brief This method returns the minimum of the input container of zVectors.
		*
		*	\param		[in]	vals			- container of zVectors.
		*	\return 			zVectors		- vector with minimum length
		*	\since version 0.0.2
		*/
		zVector zMin(vector<zVector> &vals)
		{
			double minLen = 10000;
			int minID = -1;
			for (int i = 0; i < vals.size(); i++)
			{
				if (vals[i].length() < minLen)
				{
					minLen = vals[i].length();
					minID = i;
				}
			}

			return vals[minID];
		}

		/*! \brief This method returns the maximum of the input container of zVectors.
		*
		*	\param		[in]	vals			- container of zVectors.
		*	\return 			zVectors		- vector with maximum length
		*	\since version 0.0.2
		*/
		zVector zMax(vector<zVector> &vals)
		{
			double maxLen = 0;
			int maxID = -1;
			for (int i = 0; i < vals.size(); i++)
			{
				if (vals[i].length() > maxLen)
				{
					maxLen = vals[i].length();
					maxID = i;
				}
			}

			return vals[maxID];
		}

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
		zVector ofMap(double value, double inputMin, double inputMax, zVector outputMin, zVector outputMax)
		{
			zVector out;

			out.x = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.x, outputMax.x);
			out.y = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.y, outputMax.y);
			out.z = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.z, outputMax.z);

			return out;
		}

		/*! \brief This method maps the input value from the input domain to output domain.
		*
		*	\param		[in]	value			- input vector to be mapped.
		*	\param		[in]	inputDomain		- input domain of double.
		*	\param		[in]	outDomain		- output domain of vectors.
		*	\return				double			- mapped vector.
		*	\since version 0.0.2
		*/
		zVector ofMap(double value, zDomainDouble inputDomain, zDomainVector outDomain)
		{
			zVector out;

			out.x = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.x, outDomain.max.x);
			out.y = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.y, outDomain.max.y);
			out.z = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.z, outDomain.max.z);

			return out;
		}
				

		/*! \brief This method a zVector from the input matrix row.
		*
		*	\param		[in]		zMatrixd	- input matrix. number of columns need to be 3 or 4.
		*	\param		[in]		rowIndex	- row index to be extracted.
		*	\return					zVector		- zVector of the row matrix.
		*	\since version 0.0.2
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
		*	\since version 0.0.2
		*/
		zVector fromMatrixColumn(zMatrixd &inMatrix, int colIndex)
		{
			if (inMatrix.getNumRows() < 3 || inMatrix.getNumRows() > 4) throw std::invalid_argument("cannot convert matrix column to vector.");

			vector<double> cVals = inMatrix.getCol(colIndex);
			return zVector(cVals[0], cVals[1], cVals[2]);
		}

		/*! \brief This method returns the factorised vector to the input precision.
		*
		*	\param		[in]		inVector		- input vector.
		*	\param		[in]		precision		- precision or number of digits after the decimal point.
		*	\return					zVector			- factorised vector.
		*	\since version 0.0.2
		*/
		zVector factorise(zVector &inVector, int precision = 3)
		{
			double factor = pow(10, precision);
			double x1 = round(inVector.x *factor) / factor;
			double y1 = round(inVector.y *factor) / factor;
			double z1 = round(inVector.z *factor) / factor;

			return zVector(x1, y1, z1);
		}

		/*! \brief This method scales the input point cloud by the input scale factor.
		*
		*	\param		[in]	inPoints			- input point cloud.
		*	\param		[out]	scaleFac			- scale factor.
		*	\since version 0.0.2
		*/
		void scalePointCloud(vector<zVector> &inPoints, double scaleFac)
		{
			for (int i = 0; i < inPoints.size(); i++)
			{
				inPoints[i] *= scaleFac;
			}
		}

		/*! \brief This method computes the center of the input point cloud.
		*
		*	\param		[in]	inPoints			- input point cloud.
		*	\param		[out]	center				- center of point cloud.
		*	\since version 0.0.2
		*/
		void getCenter_PointCloud(vector<zVector> &inPoints, zVector &center)
		{
			zVector out;

			for (int i = 0; i < inPoints.size(); i++)
			{
				out += inPoints[i];
			}

			out /= inPoints.size();

			center = out;
		}

		/*! \brief This method return index of the closest point in the input container to the input position.
		*
		*	\param		[in]	pos				- input position.
		*	\param		[in]	inPositions		- input container of positions.
		*	\since version 0.0.2
		*/
		int getClosest_PointCloud(zVector &pos, vector<zVector> inPositions)
		{
			int out = -1;
			double dist = 1000000000;

			for (int i = 0; i < inPositions.size(); i++)
			{
				if (inPositions[i].squareDistanceTo(pos) < dist)
				{
					dist = inPositions[i].squareDistanceTo(pos);
					out = i;
				}
			}

			return out;
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
		*	\since version 0.0.2
		*/
		zVector getDimsFromBounds(zVector &minBB, zVector &maxBB)
		{
			zVector out;

			out.x = abs(maxBB.x - minBB.x);
			out.y = abs(maxBB.y - minBB.y);
			out.z = abs(maxBB.z - minBB.z);

			return out;
		}

		/*! \brief This method checks if the input position is inside the input bounds.
		*
		*	\param		[in]	inPoint			- input point.
		*	\param		[in]	minBB			- lower bounds as zVector.
		*	\param		[in]	maxBB			- upper bounds as zVector
		*	\return				bool			- true if input position is inside the bounds.
		*	\since version 0.0.2
		*/
		bool pointInBounds(zVector &inPoint, zVector &minBB, zVector &maxBB)
		{

			if (inPoint.x < minBB.x || inPoint.x > maxBB.x) return false;

			else if (inPoint.y < minBB.y || inPoint.y > maxBB.y) return false;

			else if (inPoint.z < minBB.z || inPoint.z > maxBB.z) return false;

			else return true;
		}

		/*! \brief This method computes the inverse distance weights of the input positions container on the input point .
		*
		*	\details	bsaed on http://www.gitta.info/ContiSpatVar/en/html/Interpolatio_learningObject2.xhtml.
		*	\param		[in]	inPos			- input position.
		*	\param		[in]	positions		- input container of positions.
		*	\param		[in]	power			- power of the distance.
		*	\param		[out]	weights			- influence Weights between 0 and 1.
		*	\since version 0.0.2
		*/
		void getDistanceWeights(zVector& inPos, vector<zVector> positions, double power, vector<double> &weights)
		{
			vector<double> dists;

			for (int i = 0; i < positions.size(); i++)
			{
				double dist = (positions[i].distanceTo(inPos));

				double r = pow(dist, power);

				weights.push_back(1.0 / r);


			}

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
		*	\since version 0.0.2
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
		*	\since version 0.0.2
		*/
		bool line_lineClosestPoints(zVector &a0, zVector &a1, zVector &b0, zVector &b1, double &uA, double &uB)
		{
			bool out = false;

			zVector u = a1 - a0;
			zVector v = b1 - b0;
			zVector w = a0 - b0;

			double uu = u * u;
			double uv = u * v;
			double vv = v * v;
			double uw = u * w;
			double vw = v * w;

			double denom = 1 / (uu*vv - uv * uv);


			if (denom != 0)
			{
				uA = (uv*vw - vv * uw) * denom;
				uB = (uu*vw - uv * uw) * denom;

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
		*	\since version 0.0.2
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

		/*! \brief This method returns the area of triagle defined by the three input zVectors.
		*
		*	\param		[in]		p1				- first input point of triangle.
		*	\param		[in]		p2				- second input point of triangle.
		*	\param		[in]		p3				- second input point of triangle.
		*	\return					double			- area of triangle defirned by the vectors.
		*	\since version 0.0.2
		*/
		double getTriangleArea(zVector &v1, zVector &v2, zVector &v3)
		{
			double area = 0;

			zVector e12 = v2 - v1;
			zVector e13 = v3 - v1;

			area = ((e12^e13).length() * 0.5);

			return area;
		}

		/*! \brief This method returns the signed volume of the tetrahedron formed by the three input zVectors and the origin.
		*
		*	\details Based on http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf and http://www.dillonbhuff.com/?p=345
		*	\param		[in]		p1				- first input point of triangle.
		*	\param		[in]		p2				- second input point of triangle.
		*	\param		[in]		p3				- second input point of triangle.
		*	\return					double			- volume of tetrahedron formed by the three input vectors  and the origin.
		*	\since version 0.0.2
		*/
		double getSignedTriangleVolume(zVector &v1, zVector &v2, zVector &v3)
		{
			double volume = 0;

			zVector norm = (v2 - v1) ^ (v3 - v1);
			volume = v1 * norm;
			volume /= 6.0;


			return volume;
		}

		/*! \brief This method checks if the given input points liess within the input triangle.
		*
		*	\details based on http://blackpawn.com/texts/pointinpoly/default.html
		*	\param		[in]	pt			- zVector holding the position information of the point to be checked.
		*	\param		[in]	t0,t1,t2	- zVector holding the position information for the 3 points of the triangle.
		*	\return				bool		- true if point is inside the input triangle.
		*/
		bool pointInTriangle(zVector &pt, zVector &t0, zVector &t1, zVector &t2)
		{
			// Compute vectors        
			zVector v0 = t2 - t0;
			zVector	v1 = t1 - t0;
			zVector	v2 = pt - t0;

			// Compute dot products
			double	dot00 = v0 * v0;
			double	dot01 = v0 * v1;
			double	dot02 = v0 * v2;
			double	dot11 = v1 * v1;
			double	dot12 = v1 * v2;

			// Compute barycentric coordinates
			double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
			double 	u = (dot11 * dot02 - dot01 * dot12) * invDenom;
			double	v = (dot00 * dot12 - dot01 * dot02) * invDenom;

			if (abs(u) < 0.001) u = 0;
			if (abs(v) < 0.001) v = 0;

			// round factor to precision 3 
			double factor = pow(10, 3);
			u = round(u*factor) / factor;
			v = round(v*factor) / factor;

			//printf("\n u : %1.2f v: %1.2f ", u, v);

			// Check if point is in triangle
			return ((u >= 0) && (v >= 0) && (u + v <= 1));


		}

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
		double minDist_Edge_Point(zVector & pt, zVector & e0, zVector & e1, zVector & closest_Pt)
		{
			double out = 0.0;

			zVector n = (e1 - e0) ^ (zVector(0, 0, 1));
			n.normalize();
			closest_Pt = n * ((e0 - pt) * n);
			closest_Pt += pt;

			float len = e0.distanceTo(e1);

			zVector ed = (e0 - e1) / len;
			double param = (closest_Pt - e1) * ed;



			if (param > len)param = len;
			if (param < 0) param = 0;

			closest_Pt = e1 + ed * param;

			return closest_Pt.distanceTo(pt);
		}

		/*! \brief This method computes the minimum distance between a point and a plane.
		*
		*	\details based on http://paulbourke.net/geometry/pointlineplane/
		*	\param	[in]	pA			- point
		*	\param	[in]	pB			- point on the plane.
		*	\param	[in]	norm		- normal of the plane.
		*	\return			minDist		- minimum distance to plane.
		*	\since version 0.0.2
		*/
		double minDist_Point_Plane(zVector & pA, zVector & pB, zVector & norm)
		{
			norm.normalize();

			return (pA - pB) * norm;
		}


		/*! \brief This method gets the bary-center of the input positions based on the input weights.
		*
		*	\param		[in]	inPositions		- input container of positions.
		*	\param		[in]	weights			- input container of weights.
		*	\return				zVector			- bary-center.
		*	\since version 0.0.2
		*/
		zVector getBaryCenter(vector<zVector> &inPositions, vector<double>& weights)
		{
			if (inPositions.size() != weights.size()) throw std::invalid_argument("size of inPositions and weights not equal.");

			zVector out;

			for (int i = 0; i < inPositions.size(); i++)
			{
				out += inPositions[i] * weights[i];
			}

			out /= inPositions.size();

			return out;
		}

		//--------------------------
		//---- 4x4 zMATRIX  TRANSFORMATION METHODS
		//--------------------------

		/*! \brief This method inputs the vector values at the input index of the 4X4 tranformation matrix.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	inMatrix	- input zMatrix .
		*	\param		[in]	inVec		- input vector.
		*	\param		[in]	index		- column index.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		template <typename T>
		void setColfromVector(zMatrix<T> &inMatrix, zVector &inVec, int index)
		{
			if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
			if (inMatrix.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");

			inMatrix(0, index) = inVec.x; inMatrix(1, index) = inVec.y; inMatrix(2, index) = inVec.z;

		}

		/*! \brief This method inputs the vector values at the input index of the 4X4 tranformation matrix.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	inMatrix	- input zMatrix .
		*	\param		[in]	inVec		- input vector.
		*	\param		[in]	index		- column index.
		*	\return 			zMatrix		- transformation matrix.
		*	\since version 0.0.2
		*/
		template <typename T>
		void setRowfromVector(zMatrix<T> &inMatrix, zVector &inVec, int index)
		{
			if (inMatrix.getNumCols() != inMatrix.getNumRows()) 	throw std::invalid_argument("input Matrix is not a square.");
			if (inMatrix.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");

			inMatrix(index, 0) = inVec.x; inMatrix(index, 1) = inVec.y; inMatrix(index, 2) = inVec.z;
		}

		/*! \brief This method returns the 4X4 tranformation matrix to change the origin to the input vector.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[out]	inMatrix	- input zMatrix .
		*	\param		[in]	X			- input X Axis as a vector.
		*	\param		[in]	Y			- input Y Axis as a vector.
		*	\param		[in]	Z			- input Z Axis as a vector.
		*	\param		[in]	O			- input origin as a vector.
		*	\since version 0.0.2
		*/
		template <typename T>
		void setTransformfromVectors(zMatrix<T> &inMatrix, zVector &X, zVector &Y, zVector &Z, zVector &O)
		{

			inMatrix.setIdentity();

			setColfromVector(inMatrix, X, 0);
			setColfromVector(inMatrix, Y, 1);
			setColfromVector(inMatrix, Z, 2);
			setColfromVector(inMatrix, O, 3);


		}


		/*! \brief This method computes the tranformation to the world space of the input 4x4 matrix.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	inMatrix	- input zMatrix to be transformed.
		*	\return 			zMatrix		- world transformation matrix.
		*	\since version 0.0.2
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
		*	\since version 0.0.2
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
		*	\since version 0.0.2
		*/
		template <typename T>
		zMatrix<T> PlanetoPlane(zMatrix<T> &from, zMatrix<T> &to)
		{
			if (from.getNumCols() != from.getNumRows()) 	throw std::invalid_argument("input from Matrix is not a square.");
			if (from.getNumCols() != to.getNumCols()) 	throw std::invalid_argument("input matrices dont match in size.");
			if (from.getNumCols() != 4) 	throw std::invalid_argument("input Matrix is not a 4X4 matrix.");

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
		*	\since version 0.0.2
		*/
		template <typename T>
		zMatrix<T> ChangeBasis(zMatrix<T> &from, zMatrix<T> &to)
		{
			return toLocalMatrix(to) * toWorldMatrix(from);
		}

		/*! \brief This method computes the input target as per the input new basis.
		*
		*	\param [in]		infilename			- input file name including the directory path.
		*	\since version 0.0.2
		*/

		/*! \brief This method computes the input target as per the input new basis.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	target		- input 4X4 zMatrix.
		*	\param		[in]	newBasis	- input 4X4 zMatrix.
		*	\return 			zMatrix		- new target matrix.
		*	\since version 0.0.2
		*/
		template <typename T>
		zMatrix<T> target_newBasis(zMatrix<T> &target, zMatrix<T> &newBasis)
		{
			zMatrix<T> C = newBasis;
			zMatrix<T> C_inverse;

			bool chkInvr = C.inverseMatrix(C_inverse);

			if (!chkInvr) throw std::invalid_argument("input Matrix is singular and doesnt have an inverse matrix.");

			Matrix4f targ_newbasis;
			targ_newbasis.setIdentity();

			targ_newbasis = C_inverse * target;

			return targ_newbasis;

		}

		/*! \brief This method computes the euclidean distance between two input row matricies.  The number of columns of m1 and m2 need to be equal.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	m1			- input zMatrix 1.
		*	\param		[in]	m2			- input zMatrix 2.
		*	\param		[in]	tolerance	- input tolerance for distance check.
		*	\return 			double		- euclidean distance.
		*	\since version 0.0.2
		*/
		template <typename T>
		double getEuclideanDistance(zMatrix<T> &m1, zMatrix<T> &m2, double tolerance = 0.001)
		{
			if (m1.getNumCols() != m2.getNumCols()) throw std::invalid_argument("number of columns in m1 not equal to number of columns in m2.");
			if (m2.getNumRows() != 1) throw std::invalid_argument("number of rows in m2 not equal to 1.");
			if (m1.getNumRows() != 1) throw std::invalid_argument("number of rows in m1 not equal to 1.");

			double out;

			double dist = 0;;

			for (int j = 0; j < m1.getNumCols(); j++)
			{
				dist += pow((m1(0, j) - m2(0, j)), 2);
			}

			if (dist > tolerance) out = sqrt(dist);
			else out = 0.0;


			return out;
		}

		/*! \brief This method returns the minimum value of the input matirix.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	inMatrix		- input matrix.
		*	\param		[out]	index			- index of minimum value in the matrix container.
		*	\return 			T				- minimum value
		*	\since version 0.0.2
		*/
		template <typename T>
		T zMin(zMatrix<T> &inMatrix, int &index)
		{
			T minVal = 1000000;
			int minID = -1;
			for (int i = 0; i < inMatrix.getNumRows(); i++)
			{
				for (int j = 0; j < inMatrix.getNumCols(); j++)
				{
					if (inMatrix(i, j) < minVal)
					{
						minVal = inMatrix(i, j);
						minID = inMatrix.getIndex(i, j);
					}

				}



			}

			index = minID;

			return minVal;
		}

		/*! \brief This method returns the maximum value of the input container of zVectors.
		*
		*	\tparam				T				- Type to work with standard c++ numerical datatypes.
		*	\param		[in]	inMatrix		- input matrix.
		*	\param		[out]	index			- index of maximum value in the matrix container.
		*	\return 			T				- minimum value
		*	\since version 0.0.2
		*/
		template <typename T>
		T zMax(zMatrix<T> &inMatrix, int &index)
		{
			T maxVal = -1000000;
			int maxID = -1;
			for (int i = 0; i < inMatrix.getNumRows(); i++)
			{
				for (int j = 0; j < inMatrix.getNumCols(); j++)
				{
					if (inMatrix(i, j) > maxVal)
					{
						maxVal = inMatrix(i, j);
						maxID = inMatrix.getIndex(i, j);
					}

				}

			}

			index = maxID;

			return maxVal;
		}

		//--------------------------
		//---- MATRIX METHODS USING EIGEN
		//--------------------------


		/*! \brief This method return a 4X4 matrix of the best fit plane for the given points using Principal Component Analysis
		*
		*	\param		[in]	points		- input points.
		*	\return 			zMatrixd	- Best fit plane as a 4X4 matrix.
		*	\since version 0.0.2
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

			for (int i = 0; i < 3; i++)
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
			vector<double> Y = { eigensolver.eigenvectors().col(1)(0), eigensolver.eigenvectors().col(1)(1), eigensolver.eigenvectors().col(1)(2), 1 };
			vector<double> Z = { eigensolver.eigenvectors().col(0)(0), eigensolver.eigenvectors().col(0)(1), eigensolver.eigenvectors().col(0)(2), 1 };
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
		*	\since version 0.0.2
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

		//--------------------------
		//---- FACTORY METHODS GEOMETRY
		//--------------------------

		/*! \brief This method create a matrix from input container of zVectors.
		*
		*	\param		[out]	inMatrix		- input matrix.
		*	\param		[in]	inPoints		- input container of zVectors.
		*	\since version 0.0.2
		*/
		void fromPOINTS(zMatrixd &inMatrix, vector<zVector> &inPoints)
		{
			inMatrix = zMatrixd(inPoints.size(), 3);

			for (int i = 0; i < inPoints.size(); i++)
			{
				inMatrix(i, 0) = inPoints[i].x;
				inMatrix(i, 1) = inPoints[i].y;
				inMatrix(i, 2) = inPoints[i].z;
			}
		}


		/*! \brief This method create a container of zVectors from input matrix.
		*
		*	\param		[in]	inMatrix		- input matrix.
		*	\param		[out]	inPoints		- input container of zVectors.
		*	\since version 0.0.2
		*/
		void toPOINTS(zMatrixd &inMatrix, vector<zVector> &inPoints)
		{
			if (inMatrix.getNumCols() != 3) throw std::invalid_argument("cannnot convert zMatrix to zVector");

			inPoints.clear();

			for (int i = 0; i < inMatrix.getNumRows(); i++)
			{
				inPoints.push_back(zVector());


				inPoints[i].x = inMatrix(i, 0);
				inPoints[i].y = inMatrix(i, 1);
				inPoints[i].z = inMatrix(i, 2);

			}
		}



		//--------------------------
		//---- VECTOR METHODS GEOMETRY
		//--------------------------

		/*! \brief This method computes the points on a circle centered around world origin for input radius, and number of points.
		*
		*	\param		[in]		radius		- radius of circle.
		*	\param		[in]		numPoints	- number of points in the the circle.
		*	\param		[out]		circlePts	- points on circle.
		*	\param		[in]		localPlane	- orientation plane, by default a world plane.
		*	\param		[out]		xFactor		- the factor of scaling in x direction. For a circle both xFactor and yFactor need to be equal.
		*	\param		[out]		yFactor		- the factor of scaling in y direction. For a circle both xFactor and yFactor need to be equal.
		*	\since version 0.0.2
		*/
		void getCircle(double radius, int numPoints, vector<zVector> &circlePts, zMatrixd localPlane = zMatrixd(), double xFactor = 1.0, double yFactor = 1.0)
		{
			double theta = 0;

			zMatrixd worldPlane;
			zMatrixd trans = PlanetoPlane(worldPlane, localPlane);

			for (int i = 0; i < numPoints + 1; i++)
			{
				zVector pos;
				pos.x = (radius * cos(theta)) / xFactor;
				pos.y = (radius * sin(theta)) / yFactor;
				pos.z = 0;

				circlePts.push_back(pos * trans);

				theta += (TWO_PI / numPoints);
			}
		}

		/*! \brief This method computes the points on a rectangle for input dimensions centers around the world origin.
		*
		*	\param		[in]		dims			- dimensions of rectangle.
		*	\param		[out]		rectanglePts	- points on rectangle.
		*	\param		[in]		localPlane		- orientation plane, by default a world plane.
		*	\since version 0.0.2
		*/
		void getRectangle(zVector dims, vector<zVector> &rectanglePts, zMatrixd localPlane = zMatrixd())
		{
			dims.x *= 0.5;
			dims.y *= 0.5;

			zVector v0 = zVector(-dims.x, -dims.y, 0);
			zVector v1 = zVector(dims.x, -dims.y, 0);
			zVector v2 = zVector(dims.x, dims.y, 0);
			zVector v3 = zVector(-dims.x, dims.y, 0);

			zMatrixd worldPlane;
			zMatrixd trans = PlanetoPlane(worldPlane, localPlane);

			rectanglePts.push_back(v0 * trans);
			rectanglePts.push_back(v1* trans);
			rectanglePts.push_back(v2* trans);
			rectanglePts.push_back(v3* trans);

		}

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
		zColor averageColor(zColor c1, zColor c2, zColorType type)
		{
			zColor out;

			if (type = zRGB)
			{
				out.r = (c2.r + c1.r)	* 0.5;
				out.g = (c2.g + c1.g)	* 0.5;
				out.b = (c2.b + c1.b)	* 0.5;

				out.a = (c2.a + c1.a)	* 0.5;

				out.toHSV();

			}

			if (type = zHSV)
			{
				out.h = (c2.h + c1.h)	* 0.5;
				out.s = (c2.s + c1.s)	* 0.5;
				out.v = (c2.v + c1.v)	* 0.5;

				out.toRGB();
			}


			return out;

		}

		/*! \brief This method returns the average color of the input color container.
		*
		*	\param		[in]	c1		- input color container.
		*	\param		[in]	c2		- input color 2.
		*	\param		[in]	type	- color type - zRGB / zHSV.
		*	\since version 0.0.1
		*/
		zColor averageColor(vector<zColor> &c1, zColorType type)
		{
			zColor out;

			if (type = zRGB)
			{
				for (int i = 0; i < c1.size(); i++)
				{
					out.r += c1[i].r;
					out.g += c1[i].g;
					out.b += c1[i].b;

					out.a += c1[i].a;
				}

				out.r /= c1.size();
				out.g /= c1.size();
				out.b /= c1.size();
				out.a /= c1.size();


				out.toHSV();

			}

			if (type = zHSV)
			{
				for (int i = 0; i < c1.size(); i++)
				{
					out.h += c1[i].h;
					out.s += c1[i].s;
					out.v += c1[i].v;
				}

				out.h /= c1.size();
				out.s /= c1.size();
				out.v /= c1.size();

				out.toRGB();
			}


			return out;

		}

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
		zColor blendColor(double inputValue, zDomainDouble inDomain, zDomainColor outDomain, zColorType type)
		{
			zColor out;

			if (type = zRGB)
			{
				out.r = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.r, outDomain.max.r);
				out.g = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.g, outDomain.max.g);
				out.b = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.b, outDomain.max.b);

				out.a = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.a, outDomain.max.a);

				out.toHSV();

			}

			if (type = zHSV)
			{
				out.h = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.h, outDomain.max.h);
				out.s = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.s, outDomain.max.s);
				out.v = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.v, outDomain.max.v);

				out.toRGB();
			}


			return out;

		}

	};	
	



}

