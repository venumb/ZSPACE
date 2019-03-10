#pragma once

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

#include <filesystem>
namespace fs = std::experimental::filesystem;

using namespace std;

namespace zSpace
{
	/** \addtogroup zCore
	*	\brief The core classes, enumerators ,defintions and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zCoreUtilities
	*	\brief Collection of general utility methods.
	*  @{
	*/

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
	T factorise( T inputValue , int precision = 3)
	{
		double factor = pow(10, precision);
		return  round(inputValue *factor) / factor;
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
	*	\param		[in]	map			- input map.
	*	\param		[in]	pos			- input position.
	*	\param		[out]	outVertexId	- index of the position in the map if it exists.
	*	\return				bool		- true if the position exists in the map.
	*	\since version 0.0.1
	*/
	inline bool vertexExists(unordered_map<string, int>& positionVertex, zVector & pos, int & outVertexId)
	{
		bool out = false;;


		string hashKey = (to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z));
		std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


		if (got != positionVertex.end())
		{
			out = true;
			outVertexId = got->second;
		}


		return out;
	}

	//--------------------------
	//--- SPANNING TREE METHODS 
	//--------------------------

	/*! \brief This method returns the vertex with minimum distance value, from the set of vertices not yet included in shortest path tree. To be used with shortestDistance method for mesh/graph.
	*
	*	\details based on Dijkstra’s shortest path algorithm (https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/)
	*	\param		[out]	dist					- container of distance to each vertex from source.
	*	\param		[out]	sptSet					- container of shortest path tree for each vertex..
	*	\since version 0.0.1
	*/
	inline int minDistance(vector<float> &dist, vector<bool> &sptSet)
	{
		if (dist.size() != sptSet.size()) throw std::invalid_argument("input container sizes are not equal.");

		// Initialize min value 
		int min = 100000, min_index;

		for (int i = 0; i < dist.size(); i++)
		{
			if (!sptSet[i] && dist[i] <= min)
			{
				min = dist[i];
				min_index = i;
			}
		}

		return min_index;
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
	inline zColor averageColor(zColor &c1, zColor &c2, zColorType type)
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
	inline zColor averageColor(vector<zColor> &c1, zColorType type)
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
	inline zColor blendColor(double inputValue, double inputMin, double inputMax,  zColor &cMin, zColor &cMax, zColorType type)
	{
		zColor out;

		if (type = zRGB)
		{
			out.r = ofMap(inputValue, inputMin, inputMax, cMin.r, cMax.r);
			out.g = ofMap(inputValue, inputMin, inputMax, cMin.g, cMax.g);
			out.b = ofMap(inputValue, inputMin, inputMax, cMin.b, cMax.b);

			out.a = ofMap(inputValue, inputMin, inputMax, cMin.a, cMax.a);

			out.toHSV();

		}

		if (type = zHSV)
		{
			out.h = ofMap(inputValue, inputMin, inputMax, cMin.h, cMax.h);
			out.s = ofMap(inputValue, inputMin, inputMax, cMin.s, cMax.s);
			out.v = ofMap(inputValue, inputMin, inputMax, cMin.v, cMax.v);

			out.toRGB();
		}


		return out;

	}



	//--------------------------
	//---- WINDOWS UTILITY  METHODS
	//--------------------------

	/*! \brief This method returns the number of files in the input folder path.
	*
	*	\param		[in]	path			- input folder path.
	*	\return				int				- number of files in the input folder.
	*	\since version 0.0.1
	*/
	int getNumfiles(string path)
	{
		int out = 0;

		for (const auto & entry : fs::directory_iterator(path))
			/*std::cout << entry.path() << std::endl;*/ out++;

		return out;
	}


	/*! \brief This method returns the number of files in the input folder path of the input extension type.
	*
	*	\param		[in]	path			- input folder path.
	*	\param		[in]	extension		- input extension type.
	*	\return				int				- number of files in the input folder.
	*	\since version 0.0.1
	*/
	int getNumfiles_Type(string path, string extension = "json")
	{
		int out = 0;

		for (const auto & entry : fs::directory_iterator(path))
		{
	
			std::string fileName = entry.path().string();

			vector<string> perlineData = splitString(fileName, ".");

			if(perlineData[perlineData.size() -1] == extension) out++;			
			
		}

		

		return out;
	}

	/** @}*/

	

	/** @}*/
}

