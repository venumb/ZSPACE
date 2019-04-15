#pragma once
#include <headers/geometry/zMesh.h>
#include <headers/display/zPrintUtilities.h>

namespace zSpace
{

	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zHalfEdgeDataStructures
	*	\brief The half edge data structures classes of the library.
	*  @{
	*/

	/** \addtogroup zMeshClasses
	*	\brief The mesh related classes of the library.
	*  @{
	*/

	/*! \class zItMeshVertex
	*	\brief A mesh vertex iterator class.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class zItMeshVertex : public std::iterator< std::bidirectional_iterator_tag, int>
	{
	private:
		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a mesh  */
		zMesh *mesh;

		zVertex *v;

		int index;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
		zItMeshVertex()
		{
			mesh = nullptr;
			v = nullptr;
		}
		
		zItMeshVertex(zMesh &_mesh, int &_index)
		{
			mesh = &_mesh;			
			
			index = _index;

			v = &mesh->vertices[index];
		}

	
		zItMeshVertex operator++()
		{
			assert(v != nullptr && "Out-of-bounds iterator increment!");

			
			int temp = index;;
			temp++;

			if (temp < mesh->numEdges())
			{
				v = &mesh->vertices[index];
				index = temp;
			}

			return *this;
		}

		zItMeshVertex operator++(int)
		{

		}

	};

}
