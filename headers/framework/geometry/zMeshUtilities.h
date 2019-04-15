#pragma once

#include <headers/geometry/zMesh.h>


namespace zSpace
{
	/** \addtogroup zGeometry
	*	\brief  The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryUtilities
	*	\brief Collection of utility methods for graphs, meshes and fields.
	*  @{
	*/


	/** \addtogroup zMeshUtilities
	*	\brief Collection of utility methods for meshes.
	*  @{
	*/

	


	
	//--------------------------
	//--- UTILITY METHODS 
	//--------------------------
	
	/*! \brief This method scales the input mesh by the input scale factor.
	*
	*	\param		[in]	inMesh			- input mesh.
	*	\param		[out]	scaleFac			- scale factor.
	*	\since version 0.0.1
	*/
	inline void scaleMesh(zMesh &inMesh, double scaleFac)
	{
		//scalePointCloud(inMesh.vertexPositions, scaleFac);
	}




	/*! \brief This method combines the two disjoint meshes to one mesh.
	*
	*	\param		[in]	m1				- input mesh 1.
	*	\param		[in]	m2				- input mesh 2.
	*	\retrun				zMesh			- combined mesh.
	*	\since version 0.0.1
	*/	
	inline zMesh combineDisjointMesh(zMesh &m1, zMesh &m2)
	{
		/*zMesh out;

		vector<zVector>positions;
		vector<int>polyConnects;
		vector<int>polyCounts;


		for (int i = 0; i < m1.vertexPositions.size(); i++)
		{
			positions.push_back(m1.vertexPositions[i]);
		}

		for (int i = 0; i < m2.vertexPositions.size(); i++)
		{
			positions.push_back(m2.vertexPositions[i]);
		}

		computePolyConnects_PolyCount(m1,polyConnects, polyCounts);

		vector<int>temp_polyConnects;
		vector<int>temp_polyCounts;
		computePolyConnects_PolyCount(m2,temp_polyConnects, temp_polyCounts);

		for (int i = 0; i < temp_polyConnects.size(); i++)
		{
			polyConnects.push_back(temp_polyConnects[i] + m1.vertexPositions.size());
		}

		for (int i = 0; i < temp_polyCounts.size(); i++)
		{
			polyCounts.push_back(temp_polyCounts[i]);
		}

		out = zMesh(positions, polyCounts, polyConnects);;

		return out;*/
	}


	/*! \brief This method transforms the input mesh by the input transform matrix.
	*
	*	\param		[in]	inMesh					- input mesh.
	*	\param		[in]	transform				- transfrom matrix.
	*	\since version 0.0.1
	*/
	inline void transformMesh(zMesh &inMesh, zMatrixd& transform)
	{
		for (int j = 0; j < inMesh.vertexPositions.size(); j++)
		{			
			zVector newPos = inMesh.vertexPositions[j] * transform;
			inMesh.vertexPositions[j] = newPos;		
		}
	}


	/** @}*/

	/** @}*/

	/** @}*/
}