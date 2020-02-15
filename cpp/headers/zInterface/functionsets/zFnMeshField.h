// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Authors : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>, Leo Bieling <leo.bieling.zaha-hadid.com>
//

#ifndef ZSPACE_FN_MESHFIELD_H
#define ZSPACE_FN_MESHFIELD_H

#pragma once

#include<headers/zInterface/objects/zObjGraph.h>
#include<headers/zInterface/objects/zObjMesh.h>
#include<headers/zInterface/objects/zObjPointCloud.h>
#include<headers/zInterface/objects/zObjMeshField.h>

#include<headers/zInterface/functionsets/zFnMesh.h>
#include<headers/zInterface/functionsets/zFnGraph.h>
#include<headers/zInterface/functionsets/zFnPointCloud.h>

#include<headers/zInterface/iterators/zItMeshField.h>

#include<headers/zCore/utilities/zUtilsBMP.h>

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

	/*! \class zFnMeshField
	*	\brief A 2D field function set.
	*
	*	\tparam				T			- Type to work with zScalar(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/	

	/** @}*/

	/** @}*/

	template<typename T>
	class ZSPACE_API zFnMeshField 
	{
	private:

		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------	

		/*!	\brief function type  */
		zFnType fnType;
			   		
		/*!	\brief contour value domain.  */
		zDomain<float> contourValueDomain;

		/*!	\brief container of field values used for contouring. All values to be in teh 0 to 1 domain  */
		vector<float> contourVertexValues;

	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;			

		/*!	\brief boolean indicating if the field values size is equal to mesh vertices(true) or equal to mesh faces(false)  */
		bool setValuesperVertex = true;

		/*!	\brief boolean indicating if the field mesh is triangulated(true or quadragulated (false)  */
		bool triMesh = true;

		/*!	\brief field color domain.  */
		zDomainColor fieldColorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a field 2D object  */
		zObjMeshField<T> *fieldObj;

		/*!	\brief mesh function set  */
		zFnMesh fnMesh;

		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<zIntArray> ringNeighbours;

		/*!	\brief container of adjacent neighbourhood indicies.  */
		vector<zIntArray> adjacentNeighbours;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnMeshField();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\since version 0.0.2
		*/
		zFnMeshField(zObjMeshField<T> &_fieldObj);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnMeshField();

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method return the function set type.
		*
		*	\return 	zFnType			- type of function set.
		*	\since version 0.0.2
		*/
		zFnType getType();

		/*! \brief This method exports the field to the given file type.
		*
		*	\param	[in]		path						- output file name including the directory path and extension.
		*	\param	[in]		type						- type of file to be exported 
		*	\param	[in]		_setValuesperVertex			-  numver of field values aligns with mesh vertices if true else aligns with mesh faces.
		*	\param	[in]		_triMesh					- boolean true if triangulated mesh in needed. Works only when _setValuesperVertex is false.
		*	\since version 0.0.2
		*/
		void from(string path, zFileTpye type, bool _setValuesperVertex = true, bool _trimesh = true);

		/*! \brief This method imports the field to the given file type.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported - zBMP
		*	\since version 0.0.2
		*/
		void to(string path, zFileTpye type);

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
		*	\param		[in]	_minBB					- minimum bounds of the field.
		*	\param		[in]	_maxBB					- maximum bounds of the field.
		*	\param		[in]	_n_X					- number of pixels in x direction.
		*	\param		[in]	_n_Y					- number of pixels in y direction.
		*	\param		[in]	_NR						- ring number of neighbours to be computed. By default it is 1.
		*	\param		[in]	_setValuesperVertex		- boolean indicating if the field values size is equal to mesh vertex is true, else equal to mesh faces
		*	\param		[in]	_triMesh				- boolean true if triangulated mesh in needed. Works only when _setValuesperVertex is false.
		*	\since version 0.0.2
		*/
		void create(zPoint _minBB, zPoint _maxBB, int _n_X, int _n_Y, int _NR = 1, bool _setValuesperVertex = true, bool _triMesh = true);

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\param		[in]	_setValuesperVertex		- boolean indicating if the field values size is equal to mesh vertex is true, else equal to mesh faces
		*	\param		[in]	_triMesh				- boolean true if triangulated mesh in needed. Works only when _setValuesperVertex is false.
		*	\since version 0.0.2
		*/
		void create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zPoint _minBB = zPoint(), int _NR = 1, bool _setValuesperVertex = true, bool _triMesh = true);

		/*! \brief This method creates a vector field from the input scalarfield.
		*	\param		[in]	scalarFieldObj		- input scalar field object.
		*	\since version 0.0.2
		*/
		void createVectorFromScalarField(zObjMeshField<zScalar> &scalarFieldObj);

		//--------------------------
		//---- QUERIES
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
		*	\since version 0.0.2
		*/
		void getResolution(int &_n_X, int &_n_Y);

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each pixel in x direction.
		*	\param		[out]	_unit_Y		- size of each pixel in y direction.
		*	\since version 0.0.2
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y);

		/*! \brief This method gets the bounds of the field.
		*
		*	\param		[out]	_minBB		- minimum bounds of the field.
		*	\param		[out]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.2
		*/
		void getBoundingBox(zPoint &_minBB, zPoint &_maxBB);

		/*! \brief This method gets all the positions of the field.
		*
		*	\param		[out]	positions	- output positions container.
		*	\since version 0.0.2
		*/
		void getPositions(zPointArray &positions);		
		
		/*! \brief This method gets the value of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\param		[out]	fieldValue	- output field value.
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

		/*! \brief This method gets pointer to the internal field values container.
		*
		*	\return				T*					- pointer to internal field value container.
		*	\since version 0.0.2
		*/
		T* getRawFieldValues();
					
		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	s			- scalar field iterator.
		*	\param		[in]	epsilon		- small increment value, generally 0.0001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*	\warning works only with scalar fields
		*/
		zVector getGradient(zItMeshScalarField &s, float epsilon = EPS);

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	epsilon		- small increment value, generally 0.0001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*	\warning works only with scalar fields
		*/
		zVectorArray getGradients(float epsilon = EPS);

		/*! \brief This method gets the boolean indicating if the field values aligns with mesh vertices or faces.
		*
		*	\return				bool		- if true field values aligns with mesh vertices else aligns with mesh faces.
		*	\since version 0.0.2
		*/
		bool getValuesPerVertexBoolean();

		/*! \brief This method gets the boolean indicating if the mesh field is triagulated.
		*
		*	\return				bool		- triagulated if true. 
		*	\since version 0.0.2
		*/
		bool getTriMeshBoolean();
		
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
		*	\param		[in]	fValues		- input container of field value.
		*	\since version 0.0.2
		*/
		void setFieldValues(vector<T>& fValues);

		//--------------------------
		//----  2D IDW FIELD METHODS
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
		//----  2D SCALAR FIELD METHODS
		//--------------------------

		/*! \brief This method creates a vertex distance Field from the input  point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPositions			- input container of postions for distance calculations.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zPointArray &inPositions, bool normalise = true);

		/*! \brief This method creates a vertex distance Field from the input point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPositions			- input container of postions for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zPointArray &inPositions, double a, double b, bool normalise = true);

		/*! \brief This method creates a vertex distance Field from the input  point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, bool normalise = true);

		/*! \brief This method creates a vertex distance Field from the input point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zScalarArray &scalars, zObjPointCloud &inPointsObj, double a, double b, bool normalise = true);

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
		//----  2D SD FIELD METHODS
		//--------------------------

		/*! \brief This method gets the scalars for a circle.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	cen				- centre of the circle.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	r				- radius value.
		*	\param	[in]	annularVal		- input annular / offset value.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalars_Circle(zScalarArray &scalars, zVector &cen, float r, double annularVal = 0, bool normalise = true);

		/*! \brief This method gets the scalars for a line.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	v0				- input positions 0 of line.
		*	\param	[in]	v1				- input positions 1 of line.
		*	\param	[in]	annularVal		- input annular / offset value.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalars_Line(zScalarArray &scalars, zVector &v0, zVector &v1, double annularVal = 0, bool normalise = true);

		/*! \brief This method gets the scalars for a square.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	dimension		- input square dimensions.
		*	\param	[in]	annularVal		- input annular / offset value.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalars_Square(zScalarArray &scalars, zVector &dimensions, float annularVal = 0, bool normalise = true);

		/*! \brief This method gets the scalars for a trapezoid.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	r1				- input distance 1 value.
		*	\param	[in]	r2				- input distance 2 value.
		*	\param	[in]	he				- input height value.
		*	\param	[in]	annularVal		- input annular / offset value.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalars_Trapezoid(zScalarArray &scalars, float r1, float r2, float he, float annularVal = 0, bool normalise = true);

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		/*! \brief This method checks if the input position is in the bounds of the field.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	index				- index of the position.
		*	\return				bool				- true if in bounds else false.
		*	\since version 0.0.2
		*/
		bool checkPositionBounds(zPoint &pos, int &index);

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

		/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
		*
		*	\param	[in]	values		- container of values
		*	\param	[out]	domain		- stores the domain of values
		*	\since version 0.0.2
		*/
		void computeDomain(vector<T> &values, zDomain <T> &domain);

		/*! \brief This method normalises the field values.
		*
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
		void boolean_difference(zScalarArray& fieldValues_A, zScalarArray& fieldValues_B, zScalarArray& fieldValues_Result, bool normalise = true);

		/*! \brief This method uses an input plane to clip an existing scalar field.
		*
		*	\param	[in]	field			- input field mesh.
		*	\param	[in]	scalars				- vector of scalar values. Need to be equivalent to number of mesh vertices.
		*	\param	[in]	clipPlane			- input zPlane used for clipping.
		*	\since version 0.0.2
		*/
		void boolean_clipwithPlane(zScalarArray& scalars, zMatrix4 &clipPlane);

		//--------------------------
		//----  UPDATE METHODS
		//--------------------------

		/*! \brief This method updates the color values of the field mesh based on the field values. Gradient - Black to Red
		*
		*	\warning works only for scalar fields.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void updateColors();	

		//--------------------------
		//---- CONTOUR METHODS
		//--------------------------

		/*! \brief This method creates a isocontour graph from the input field mesh at the given field threshold.
		*
		*	\param	[out]	coutourGraphObj	- isoline graph.
		*	\param	[in]	inThreshold		- field threshold.
		*	\return			zGraph			- contour graph.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void getIsocontour(zObjGraph &coutourGraphObj, float inThreshold = 0.5);

		/*! \brief This method creates a isoline mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[out]	coutourMeshObj	- isoline mesh.
		*	\param	[in]	inThreshold		- field threshold.
		*	\param	[in]	invertMesh		- true if inverted mesh is required.
		*	\return			zMesh			- isoline mesh.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void getIsolineMesh(zObjMesh &coutourMeshObj, float inThreshold = 0.5, bool invertMesh = false);

		/*! \brief This method creates a isoband mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[out]	coutourMeshObj	- isoband mesh.
		*	\param	[in]	inThresholdLow	- field threshold domain minimum.
		*	\param	[in]	inThresholdHigh	- field threshold domain maximum.
		*	\param	[in]	invertMesh		- true if inverted mesh is required.		
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void getIsobandMesh(zObjMesh &coutourMeshObj, float inThresholdLow = 0.2, float inThresholdHigh = 0.5, bool invertMesh = false);

	protected:
		
		//--------------------------
		//---- PROTECTED FACTORY METHODS
		//--------------------------

		/*! \brief This method exports the input field to a bitmap file format based on the face color of the correspoding field mesh.
		*
		*	\param [in]		outfilename		- output file name including the directory path and extension.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void toBMP(string outfilename);


		/*! \brief This method imorts the input bitmap file and creates the corresponding field and  field mesh. The Bitmap needs to be in grey-scale colors only to update field values.
		*
		*	\param		[in]		infilename		- input file name including the directory path and extension.
		*	\since version 0.0.2
		*	\warning	works only with scalar fields
		*/
		void fromBMP(string infilename);

		/*! \brief This method creates the mesh from the field parameters.
		*
		*	\since version 0.0.2
		*/
		void createFieldMesh();

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------

		/*! \brief This method gets the scalar for the input point.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[in]	cen				- centre of the circle.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	r				- radius value.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		float getScalar_Circle(zPoint &cen, zPoint &p, float r);

		/*! \brief This method gets the scalar for the input line.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	v0				- input positions 0 of line.
		*	\param	[in]	v1				- input positions 1 of line.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		float getScalar_Line(zPoint &p, zPoint &v0, zPoint &v1);

		/*! \brief This method gets the sqaure scalar for the input point.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	dimention		- input distance.
		*	\since version 0.0.2
		*/
		float getScalar_Square(zPoint &p, zVector &dimensions);

		/*! \brief This method gets the scalar for a trapezoid.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	r1				- input distance 1 value.
		*	\param	[in]	r2				- input distance 2 value.
		*	\param	[in]	he				- input height value.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		double getScalar_Trapezoid(zPoint &p, float &r1, float &r2, float &he);


		/*! \brief This method gets the isoline case based on the input vertex binary values.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
		*	\param	[in]	vertexBinary	- vertex binary values.
		*	\return			int				- case type.
		*	\since version 0.0.2
		*/
		int getIsolineCase(bool vertexBinary[4]);

		/*! \brief This method gets the isoline case based on the input vertex ternary values.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
		*	\param	[in]	vertexTernary	- vertex ternary values.
		*	\return			int				- case type.
		*	\since version 0.0.2
		*/
		int getIsobandCase(int vertexTernary[4]);

		/*! \brief This method return the contour position  given 2 input positions at the input field threshold.
		*
		*	\param	[in]	threshold		- field threshold.
		*	\param	[in]	vertex_lower	- lower threshold position.
		*	\param	[in]	vertex_higher	- higher threshold position.
		*	\param	[in]	thresholdLow	- field threshold domain minimum.
		*	\param	[in]	thresholdHigh	- field threshold domain maximum.
		*	\since version 0.0.2
		*/
		zVector getContourPosition(float &threshold, zVector& vertex_lower, zVector& vertex_higher, float& thresholdLow, float& thresholdHigh);

		/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
		*
		*	\param	[in]	f				- input face iterator.
		*	\param	[in]	positions		- container of positions of the computed polygon.
		*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
		*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
		*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
		*	\param	[in]	threshold		- field threshold.
		*	\param	[in]	invertMesh	- true if inverted mesh is required.
		*	\since version 0.0.2
		*/
		void getIsolinePoly(zItMeshFace& f , zPointArray &positions, zIntArray &polyConnects, zIntArray &polyCounts, unordered_map <string, int> &positionVertex, float &threshold, bool invertMesh);

		/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
		*
		*	\param	[in]	f				- input face iterator.
		*	\param	[in]	positions		- container of positions of the computed polygon.
		*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
		*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
		*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
		*	\param	[in]	thresholdLow	- field threshold domain minimum.
		*	\param	[in]	thresholdHigh	- field threshold domain maximum.
		*	\since version 0.0.2
		*/
		void getIsobandPoly(zItMeshFace& f, zPointArray &positions, zIntArray &polyConnects, zIntArray &polyCounts, unordered_map <string, int> &positionVertex, float &thresholdLow, float &thresholdHigh);

	};	

	/** \addtogroup zInterface
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \typedef zFnMeshScalarField
	*	\brief A function set for 2D scalar field.
	*
	*	\since version 0.0.2
	*/
	typedef zFnMeshField<zScalar> zFnMeshScalarField;

	/*! \typedef zFnMeshVectorField
	*	\brief A function set for 2D scalar field.
	*
	*	\since version 0.0.2
	*/
	typedef zFnMeshField<zVector> zFnMeshVectorField;

	/** @}*/

	/** @}*/

}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zInterface/functionsets/zFnMeshField.cpp>
#endif

#endif