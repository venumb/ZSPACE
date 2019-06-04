#pragma once

#include <headers/framework/core/zVector.h>
#include <headers/framework/geometry/zGeometryTypes.h>

namespace zSpace
{
	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zGeometry
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zPointCloud
	*	\brief A point cloud class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zPointCloud 
	{
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------

		/*! \brief core utilities object			*/
		zUtilsCore coreUtils;

		/*!	\brief stores number of active vertices */
		int n_v;

		/*!	\brief vertex container */
		vector<zVertex> vertices;

		/*!	\brief container which stores vertex positions. 	*/
		vector<zVector> vertexPositions;

		/*!	\brief container which stores vertex colors. 	*/
		vector<zColor> vertexColors;

		/*!	\brief container which stores vertex weights. 	*/
		vector<double> vertexWeights;

		/*!	\brief position to vertexId map. Used to check if vertex exists with the haskey being the vertex position.	 */
		unordered_map <string, int> positionVertex;

		/*!	\brief stores the start vertex ID in the VBO, when attached to the zBufferObject.	*/
		int VBO_VertexId;

		/*!	\brief stores the start vertex color ID in the VBO, when attache to the zBufferObject.	*/
		int VBO_VertexColorId;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zPointCloud()
		{
			n_v = 0;
		}
	
		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zPointCloud() {}


		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This methods creates the the point cloud from the input contatiners.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\since version 0.0.2
		*/
		void create(vector<zVector> &_positions)
		{
			// clear containers
			clear();

			for (int i = 0; i < _positions.size(); i++)	addVertex(_positions[i]);

		}


		/*! \brief This methods clears all the graph containers.
		*
		*	\since version 0.0.2
		*/
		void clear()
		{
			vertices.clear();
			vertexPositions.clear();
			vertexColors.clear();
			vertexWeights.clear();
			

			n_v = 0;
		}

		//--------------------------
		//---- VERTEX METHODS
		//--------------------------

		/*! \brief This method adds a vertex to the vertices array.
		*
		*	\param		[in]	pos			- zVector holding the position information of the vertex.
		*	\return				bool		- true if the faces container is resized.
		*	\since version 0.0.1
		*/
		bool addVertex(zVector &pos)
		{
			bool out = false;			


			addToPositionMap(pos, n_v);

			zItVertex newV = vertices.insert(vertices.end(), zVertex());
			newV->setId(n_v);


			vertexPositions.push_back(pos);
			n_v++;



			// default Attibute values			
			vertexColors.push_back(zColor(1, 0, 0, 1));
			vertexWeights.push_back(2.0);

			return out;
		}

		/*! \brief This method detemines if a vertex already exists at the input position
		*
		*	\param		[in]		pos			- position to be checked.
		*	\param		[out]		outVertexId	- stores vertexId if the vertex exists else it is -1.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\return					bool		- true if vertex exists else false.
		*	\since version 0.0.2
		*/
		bool vertexExists(zVector pos, int &outVertexId, int precisionfactor = 6)
		{
			bool out = false;;
			outVertexId = -1;

			double factor = pow(10, precisionfactor);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);


			if (got != positionVertex.end())
			{
				out = true;
				outVertexId = got->second;
			}


			return out;
		}

		/*! \brief This method sets the number of vertices in zGraph  the input value.
		*	\param		[in]		_n_v	- number of vertices.
		*	\since version 0.0.1
		*/
		void setNumVertices(int _n_v)
		{
			n_v = _n_v;
		}

		//--------------------------
		//---- MAP METHODS
		//--------------------------

		/*! \brief This method adds the position given by input vector to the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		index			- input vertex index in the vertex position container.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/
		void addToPositionMap(zVector &pos, int index, int precisionfactor = 6)
		{
			double factor = pow(10, precisionfactor);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string hashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			positionVertex[hashKey] = index;
		}

		/*! \brief This method removes the position given by input vector from the positionVertex Map.
		*	\param		[in]		pos				- input position.
		*	\param		[in]		precisionfactor	- input precision factor.
		*	\since version 0.0.1
		*/
		void removeFromPositionMap(zVector &pos, int precisionfactor = 6)
		{
			double factor = pow(10, precisionfactor);
			double x = round(pos.x *factor) / factor;
			double y = round(pos.y *factor) / factor;
			double z = round(pos.z *factor) / factor;

			string removeHashKey = (to_string(x) + "," + to_string(y) + "," + to_string(z));
			positionVertex.erase(removeHashKey);
		}
	};
}