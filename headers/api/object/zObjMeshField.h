#pragma once
#pragma once

#include <headers/api/object/zObj.h>
#include <headers/framework/field/zField2D.h>

#include <vector>
using namespace std;

namespace zSpace
{

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjMeshField
	*	\brief The 2D field mesh object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	template<typename T>
	class zObjMeshField : public zObjMesh
	{


	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief field 2D */
		zField2D<T> field;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjMeshField()
		{
			displayUtils = nullptr;	

			showVertices = false;
			showEdges = true;
			showFaces = true;

			showDihedralEdges = false;
			showVertexNormals = false;
			showFaceNormals = false;

			dihedralAngleThreshold = 45;

			normalScale = 1.0;
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjMeshField() {}
		

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			zObjMesh::draw();
		}	
			


	};




}

