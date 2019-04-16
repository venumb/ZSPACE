#pragma once

#include <depends/modernJSON/json.hpp>
using json = nlohmann::json;;

using namespace std;

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

	/*! \class zUtilsJsonHE
	*	\brief A json utility class for  a half edge datastructure.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	struct zUtilsJsonHE
	{
			
		/*!	\brief container of vertex data  */
		vector<int> vertices;

		/*!	\brief container of half edge data */
		vector<vector<int>> halfedges;

		/*!	\brief container of face data*/
		vector<int> faces;

		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<vector<double>> vertexAttributes;

		/*!	\brief container of half edge attribute data - color*/
		vector<vector<double>> halfedgeAttributes;

		/*!	\brief container of face attribute data - normals, colors.*/
		vector<vector<double>> faceAttributes;	

	};

}