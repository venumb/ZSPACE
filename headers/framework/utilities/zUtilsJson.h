#pragma once

#include <depends/modernJSON/json.hpp>
using json = nlohmann::json;;

using namespace std;

namespace zSpace
{
	/** \addtogroup zUtilities
	*	\brief The utility classes of the library.
	*  @{
	*/

	/*! \class zUtilsJsonHE
	*	\brief A half edge datastructure json utility class.
	*	\since version 0.0.2
	*/

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