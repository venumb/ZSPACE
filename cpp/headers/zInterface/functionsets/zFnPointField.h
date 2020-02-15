// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#ifndef ZSPACE_FN_POINTFIELD_H
#define ZSPACE_FN_POINTFIELD_H

#pragma once

#include<headers/zInterface/objects/zObjPointCloud.h>
#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/objects/zObjPointField.h>

#include<headers/zInterface/functionsets/zFnMesh.h>
#include<headers/zInterface/functionsets/zFnGraph.h>
#include<headers/zInterface/functionsets/zFnPointCloud.h>

#include<headers/zInterface/iterators/zItPointField.h>

namespace zSpace
{
	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnPointField
	*	\brief A 3D field function set.
	*
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	template<typename T>
	class ZSPACE_API zFnPointField
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		
		
		/*!	\brief function type  */
		zFnType fnType;

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief contour value domain.  */
		zDomain<float> contourValueDomain;

		/*!	\brief pointer to a field 3D object  */
		zObjPointField<T> *fieldObj;	

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<vector<int>> ringNeighbours;

		/*!	\brief container of adjacent neighbourhood indicies.  */
		vector<vector<int>> adjacentNeighbours;

		/*!	\brief point cloud function set  */
		zFnPointCloud fnPoints;

		/*!	\brief field color domain.  */
		zDomainColor fieldColorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnPointField();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field3D object.	
		*	\since version 0.0.2
		*/
		zFnPointField(zObjPointField<T> &_fieldObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnPointField();

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------
		
		/*! \brief This method return the function set type.
		*
		*	\return 	zFnType			- type of function set.
		*	\since version 0.0.2
		*/
		zFnType getType() ;

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\param [out]		minBB			- output minimum bounding box.
		*	\param [out]		maxBB			- output maximum bounding box.
		*	\since version 0.0.2
		*/
		void getBounds(zPoint &minBB, zPoint &maxBB);

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		void clear();

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/*! \brief This method creates a field from the input parameters.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.2
		*/
		void create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR = 1);

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.2
		*/
		void create(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, zPoint _minBB = zPoint(), int _NR = 1);

		/*! \brief This method creates a vector field from the input scalarfield.
		*	\param		[in]	inFnScalarField		- input scalar field function set.
		*	\since version 0.0.2
		*/
		void createVectorFromScalarField(zFnPointField<zScalar> &inFnScalarField);
				

		//--------------------------
		//--- FIELD QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the field indices which contain the input position.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	containedGridPoints	- contatiner of contained points indicies.
		*	\since version 0.0.2
		*/
		void getNeighbour_Contained(zPoint &pos, zIntArray &containedNeighbour);

		/*! \brief This method gets the field Points which contain the input position.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	containedGridPoints	- contatiner of contained points indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Contained(zPoint &pos, zPointArray &containedNeighbour);
	

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method retruns the number of scalars in the field.
		*
		*	\return			int	- number of scalars in the field.
		*	\since version 0.0.2
		*/
		int numFieldValues();
			
		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_n_X		- pixel resolution in x direction.
		*	\param		[out]	_n_Y		- pixel resolution in y direction.
		*	\param		[out]	_n_Z		- pixel resolution in z direction.
		*	\since version 0.0.2
		*/
		void getResolution(int &_n_X, int &_n_Y, int &_n_Z);

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each pixel in x direction.
		*	\param		[out]	_unit_Y		- size of each pixel in y direction.
		*	\param		[out]	_unit_Z		- size of each voxel in z direction
		*	\since version 0.0.2
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y, double &_unit_Z);

		/*! \brief This method gets the bounds of the field.
		*
		*	\param		[out]	_minBB		- minimum bounds of the field.
		*	\param		[out]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.2
		*/
		void getBoundingBox(zPoint &_minBB, zPoint &_maxBB);

		/*! \brief This method gets the value of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\param		[out]	T			- field value.
		*	\return				bool		- true if sample position is within bounds.
		*	\since version 0.0.2
		*/
		bool getFieldValue(zPoint &samplePos, zFieldValueType type, T& fieldValue);

		/*! \brief This method gets all the values of the field.
		*
		*	\param		[out]	fieldValues			- container of field values.
		*	\since version 0.0.2
		*/
		void getFieldValues(vector<T>& fieldValues);

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	s			- scalar field iterator..
		*	\param		[in]	epsilon		- small increment value, generally 0.0001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*	\warning works only with scalar fields
		*/
		zVector getGradient(zItPointScalarField &s, float epsilon = EPS);

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	epsilon		- small increment value, generally 0.0001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*	\warning works only with scalar fields
		*/
		vector<zVector> getGradients(float epsilon = EPS);

		//--------------------------
		//---- SET METHODS
		//--------------------------
				
		/*! \brief This method sets the field color domain.
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\since version 0.0.2
		*/
		void setFieldColorDomain(zDomainColor &colDomain);

		/*! \brief This method sets the bounds of the field.
		*
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.2
		*/
		void setBoundingBox(zPoint &_minBB, zPoint &_maxBB);

		/*! \brief This method sets the values of the field to the input container values.
		*
		*	\param		[in]	fValue		- input container of field value.
		*	\since version 0.0.2
		*/
		void setFieldValues(vector<T> &fValues);				

		//--------------------------
		//----  3D IDW FIELD METHODS
		//--------------------------
				

		/*! \brief This method computes the field values as inverse weighted distance from the input mesh vertex positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inMeshObj			- input mesh object for distance calculations.
		*	\param	[in]	meshValue			- value to be propagated for the mesh.
		*	\param	[in]	influences			- influence value of the graph.		
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjMesh &inMeshObj, T meshValue, double influence, double power = 2.0, bool normalise = true);

		/*! \brief This method computes the field values as inverse weighted distance from the input graph vertex positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inGraphObj			- input graph object set for distance calculations.
		*	\param	[in]	graphValue			- value to be propagated for the graph.
		*	\param	[in]	influences			- influence value of the graph.		
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjGraph &inGraphObj, T graphValue, double influence, double power = 2.0, bool normalise = true);

		/*! \brief This method computes the field values based on inverse weighted distance from the input positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	value				- value to be propagated for each input position.
		*	\param	[in]	influence			- influence value of each input position.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, T value, double influence, double power = 2.0, bool normalise = true);
		
		/*! \brief This method computes the field values based on inverse weighted distance from the input positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	values				- value to be propagated for each input position. Size of container should be equal to inPositions.
		*	\param	[in]	influences			- influence value of each input position. Size of container should be equal to inPositions.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, vector<T> &values, vector<double>& influences, double power = 2.0, bool normalise = true);

		/*! \brief This method computes the field values based on inverse weighted distance from the input positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inPositions			- input container of positions for distance calculations.
		*	\param	[in]	value				- value to be propagated for each input position.
		*	\param	[in]	influence			- influence value of each input position.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, T value, double influence, double power = 2.0, bool normalise = true);


		/*! \brief This method computes the field values based on inverse weighted distance from the input positions.
		*
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	inPositions			- icontainer of positions for distance calculations.
		*	\param	[in]	values				- value to be propagated for each input position. Size of container should be equal to inPositions.
		*	\param	[in]	influences			- influence value of each input position. Size of container should be equal to inPositions.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zPointArray &inPositions, vector<T> &values, vector<double>& influences, double power = 2.0, bool normalise = true);

		//--------------------------
		//----  3D SCALAR FIELD METHODS
		//--------------------------

		/*! \brief This method creates a vertex distance Field from the input vector of zVector positions.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, bool normalise = true);
		
		/*! \brief This method creates a vertex distance Field from the input mesh vertex positions.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inMeshObj			- input mesh object for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise = true);

		/*! \brief This method creates a vertex distance Field from the input graph vertex positions.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inGraphObj		- input graph object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise = true);

		/*! \brief This method creates a edge distance Field from the input mesh.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inMeshObj		- input mesh object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(zScalarArray &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise = true);

		/*! \brief This method creates a edge distance Field from the input graph.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inGraphObj		- input graph object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(zScalarArray &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise = true);

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------
				
		/*! \brief This method check if the input index is the bounds of the resolution in X.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_X(int index_X);

		/*! \brief This method check if the input index is the bounds of the resolution in Y.
		*
		*	\param		[in]	index_Y		- input index in Y.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_Y(int index_Y);

		/*! \brief This method check if the input index is the bounds of the resolution in Y.
		*
		*	\param		[in]	index_Y		- input index in Y.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_Z(int index_Z);

		/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
		*
		*	\param	[out]	dMin		- stores the minimum scalar value
		*	\param	[out]	dMax		- stores the maximum scalar value
		*	\param	[in]	buffer		- buffer of scalars.
		*	\since version 0.0.2
		*/
		void computeMinMaxOfScalars(vector<T> &values, T &dMin, T &dMax);

		/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
		*
		*	\param	[in]	values		- container of values
		*	\param	[out]	domain		- stores the domain of values
		*	\since version 0.0.2
		*/
		void computeDomain(vector<T> &values, zDomain <T> &domain);

		/*! \brief This method normalises the field values.
		*
		*	\param	[in]	values		- container of values
		*	\since version 0.0.2
		*/
		void normliseValues(vector<T> &values);
		
		/*! \brief This method avarages / smoothens the field values.
		*
		*	\param		[in]	numSmooth			- number of times to smooth.
		*	\param		[in]	diffuseDamp			- damping value of the averaging.
		*	\param		[in]	type				- smooth type - zlaplacian / zAverage.
		*	\since version 0.0.2
		*/
		void smoothField(int numSmooth, double diffuseDamp = 1.0, zDiffusionType type = zAverage);
	

		/*! \brief This method computes the field index of each input position and stores them in a container per field index.
		*
		*	\param		[in]	positions			- container of positions.
		*	\param		[out]	fieldIndexPositions	- container of position per field  index.
		*	\since version 0.0.2
		*/
		void computePositionsInFieldIndex(zPointArray &positions, vector<zPointArray> &fieldIndexPositions);

		/*! \brief This method computes the field index of each input position and stores the indicies in a container per field index.
		*
		*	\param		[in]	positions			- container of positions.
		*	\param		[out]	fieldIndexPositions	- container of position indicies per field  index.
		*	\since version 0.0.2
		*/
		void computePositionIndicesInFieldIndex(zPointArray &positions, vector<zIntArray> &fieldIndexPositionIndicies);

		/*! \brief This method computes the distance function.
		*
		*	\param	[in]	r	- distance value.
		*	\param	[in]	a	- value of a.
		*	\param	[in]	b	- value of b.
		*	\since version 0.0.2
		*/
		double F_of_r(double &r, double &a, double &b);

		//--------------------------
		//----  BOOLEAN METHODS
		//--------------------------

		/*! \brief This method creates a union of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	scalars0				- value of buffer.
		*	\param	[in]	scalars1				- value of buffer.
		*	\param	[in]	scalarsResult			- value of buffer to store the results.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_union(zScalarArray& scalars0, zScalarArray& scalars1, zScalarArray& scalarsResult, bool normalise = true);

		/*! \brief This method creates a subtraction of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_subtract(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise = true);

		/*! \brief This method creates a intersect of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_intersect(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise = true);

		/*! \brief This method creates a difference of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_difference(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise = false);

		/*! \brief This method uses an input plane to clip an existing scalar field.
		*
		*	\param	[in]	fieldMesh			- input field mesh.
		*	\param	[in]	scalars				- vector of scalar values. Need to be equivalent to number of mesh vertices.
		*	\param	[in]	clipPlane			- input zPlane used for clipping.
		*	\since version 0.0.2
		*/
		void boolean_clipwithPlane(zScalarArray& scalars, zMatrix4& clipPlane);	

		//--------------------------
		//----  UPDATE METHODS
		//--------------------------

		/*! \brief This method updates the color values of the field points based on the field values. Gradient - Black to Red
		*
		*	\warning works only for scalar fields.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void updateColors();

	protected:

		//--------------------------
		//----  PROTECTED METHODS
		//--------------------------

		/*! \brief This method creates the point cloud from the field parameters.
		*
		*	\since version 0.0.2
		*/
		void createPointCloud();
		

	};	

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \typedef zFnPointScalarField
	*	\brief A function set for 3D scalar field.
	*
	*	\since version 0.0.2
	*/
	typedef zFnPointField<zScalar> zFnPointScalarField;

	/*! \typedef zFnPointVectorField
	*	\brief A function set for 3D scalar field.
	*
	*	\since version 0.0.2
	*/
	typedef zFnPointField<zVector> zFnPointVectorField;

	/** @}*/

	/** @}*/



}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnPointField.cpp>
#endif

#endif