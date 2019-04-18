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

	/*! \struct zUtilsJsonHE
	*	\brief A json utility struct for storing half edge datastructure.
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


	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zUtilities
	*	\brief The utility classes and structs of the library.
	*  @{
	*/

	/*! \struct zUtilsJsonRobot
	*	\brief A json utility struct for storing robot data.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	struct zUtilsJsonRobot
	{
		/*!	\brief container of DH_d values  */
		double scale;

		/*!	\brief container of DH_d values  */
		vector<double> d;

		/*!	\brief container of DH_a values  */
		vector<double> a;

		/*!	\brief container of DH_alpha values  */
		vector<double> alpha;

		/*!	\brief container of DH_alpha values  */
		vector<double> theta;

		/*!	\brief container of jointRotations_home values  */
		vector<double> jointRotations_home;

		/*!	\brief container of jointRotations_minimum values  */
		vector<double> jointRotations_minimum;

		/*!	\brief container of jointRotations_maximum values  */
		vector<double> jointRotations_maximum;

		/*!	\brief container of jointRotations_pulse values  */
		vector<double> jointRotations_pulse;

		/*!	\brief container of jointRotations_mask values  */
		vector<double> jointRotations_mask;

		/*!	\brief container of jointRotations_offset values  */
		vector<double> jointRotations_offset;		

	};

}