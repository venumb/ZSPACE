#pragma once

#include<headers/framework/core/zVector.h>
#include<headers/framework/core/zColor.h>



namespace zSpace
{

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/


	/** \addtogroup zFields
	*	\brief The field classes of the library.
	*  @{
	*/

	/*! \class zField3D
	*	\brief A template class for 3D fields - scalar and vector.
	*	\tparam				T			- Type to work with double(scalar field) and zVector(vector field).
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class zField3D
	{		
		
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
		
		/*!	\brief stores the resolution in X direction  */
		int n_X;

		/*!	\brief stores the resolution in Y direction  */
		int n_Y;

		/*!	\brief stores the resolution in Z direction  */
		int n_Z;

		/*!	\brief stores the size of one unit in X direction  */
		double unit_X;

		/*!	\brief stores the size of one unit in Y direction  */
		double unit_Y;

		/*!	\brief stores the size of one unit in Z direction  */
		double unit_Z;

		/*!	\brief stores the minimum bounds of the field  */
		zVector minBB;

		/*!	\brief stores the minimum bounds of the field  */
		zVector maxBB;

		/*!	\brief container of field  positions  */
		vector<zVector> positions;

		/*!	\brief container for the field values  */
		vector<T> fieldValues;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zField3D()
		{
			fieldValues.clear();			
		}



		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\since version 0.0.1
		*/		
		zField3D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z)
		{
			minBB = _minBB;
			maxBB = _maxBB;
			
			n_X = _n_X;
			n_Y = _n_Y;
			n_Z = _n_Z;

			unit_X = (maxBB.x - minBB.x) / n_X;
			unit_Y = (maxBB.y - minBB.y) / n_Y;
			unit_Z = (maxBB.z - minBB.z) / n_Z;

			zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
			zVector startPt = minBB + (unitVec * 0.5);

			fieldValues.clear();

			printf("unit_X : %1.2f unit_Y : %1.2f unit_Z : %1.2f ", unit_X, unit_Y, unit_Z);

			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{

					for (int k = 0; k < n_Z; k++)
					{
						zVector pos;
						pos.x = startPt.x + i * unitVec.x;
						pos.y = startPt.y + j * unitVec.y;
						pos.z = startPt.z + k * unitVec.z;

						positions.push_back(pos);

						T defaultValue;
						fieldValues.push_back(defaultValue);						
					}
					
				}
			}
			
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each voxel in x direction.
		*	\param		[in]	_unit_Y		- size of each voxel in y direction.
		*	\param		[in]	_unit_Z		- size of each voxel in yz direction.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.		
		*	\since version 0.0.1
		*/
		zField3D(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zVector _minBB = zVector())
		{
			unit_X = _unit_X;
			unit_Y = _unit_Y;
			unit_Z = _unit_Z;

			n_X = _n_X;
			n_Y = _n_Y;
			n_Z = _n_Z;
			
			maxBB = minBB + zVector(unit_X * n_X, unit_Y * n_Y, unit_Z * n_Z);

			zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
			zVector startPt = minBB + (unitVec * 0.5);


			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{

					for (int k = 0; k < n_Z; k++)
					{
						zVector pos;
						pos.x = startPt.x + i * unitVec.x;
						pos.y = startPt.y + j * unitVec.y;
						pos.z = startPt.z + k * unitVec.z;

						positions.push_back(pos);

						T defaultValue;
						fieldValues.push_back(defaultValue);
					}
				}
			}			


		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zField3D() {}
			

	};

}