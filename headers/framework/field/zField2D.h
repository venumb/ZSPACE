#pragma once

#include<headers/framework/geometry/zMesh.h>
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

	/*! \class zField2D
	*	\brief A template class for 2D fields - scalar and vector.
	*	\tparam				T			- Type to work with double(scalar field) and zVector(vector field).
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class zField2D
	{	
			   		

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
	
		/*!	\brief stores the resolution in X direction  */
		int n_X;

		/*!	\brief stores the resolution in Y direction  */
		int n_Y;

		/*!	\brief stores the size of one unit in X direction  */
		double unit_X;

		/*!	\brief stores the size of one unit in Y direction  */
		double unit_Y;

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector minBB;

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector maxBB;

		/*!	\brief container for the field values  */
		vector<T> fieldValues;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zField2D()
		{
			fieldValues.clear();			
		}



		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\since version 0.0.1
		*/
		zField2D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y)
		{
			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			minBB = _minBB;
			maxBB = _maxBB;
			n_X = _n_X;
			n_Y = _n_Y;

			unit_X = (maxBB.x - minBB.x) / n_X;
			unit_Y = (maxBB.y - minBB.y) / n_Y;

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB;

			fieldValues.clear();	

			T defaultValue;
			fieldValues.assign(n_X*n_Y, defaultValue);	
			
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\since version 0.0.1
		*/		
		zField2D(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB = zVector())
		{
			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			unit_X = _unit_X;
			unit_Y = _unit_Y;
			n_X = _n_X;
			n_Y = _n_Y;
			
			minBB = _minBB;
			maxBB = _minBB + zVector(unit_X * n_X, unit_Y * n_Y, 0);


			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = _minBB;
			
			fieldValues.clear();

			T defaultValue;
			fieldValues.assign(n_X*n_Y,defaultValue);	

		}
		
			
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/
		~zField2D(){}

	};



}