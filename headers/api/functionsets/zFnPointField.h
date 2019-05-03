#pragma once

#include<headers/api/object/zObjPointCloud.h>
#include<headers/api/object/zObjGraph.h>
#include<headers/api/object/zObjMesh.h>
#include<headers/api/object/zObjPointField.h>

#include<headers/api/functionsets/zFnMesh.h>
#include<headers/api/functionsets/zFnGraph.h>
#include<headers/api/functionsets/zFnPointCloud.h>

namespace zSpace
{
	/** \addtogroup API
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
	*	\tparam				T			- Type to work with double(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	template<typename T>
	class zFnPointField 
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		
		
		zFnType fnType;

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to a field 2D object  */
		zObjPointField<T> *fieldObj;	
				
		/*! \brief This method creates the point cloud from the field parameters.
		*
		*	\since version 0.0.2
		*/
		void createPointCloud()
		{
			vector<zVector>positions;

			zVector minBB, maxBB;
			double unit_X, unit_Y, unit_Z;
			int n_X, n_Y,n_Z;

			getUnitDistances(unit_X, unit_Y, unit_Z);
			getResolution(n_X, n_Y, n_Z);

			getBoundingBox(minBB, maxBB);

			zVector unitVec = zVector(unit_X, unit_Y, unit_Z);
			zVector startPt = minBB;

			for (int i = 0; i < n_X; i++)
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

					}
				}
			}

			fnPoints.create(positions);
			
		}

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

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnPointField() 
		{
			fnType = zFnType::zPointFieldFn;
			fieldObj = nullptr;			
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field3D object.	
		*	\since version 0.0.2
		*/
		zFnPointField(zObjPointField<T> &_fieldObj)
		{
			fieldObj = &_fieldObj;	
			fnPoints = zFnPointCloud(_fieldObj);

			fnType = zFnType::zPointFieldFn;
			
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnPointField() {}


		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		void clear() 
		{

			
			fieldObj->field.ringNeighbours.clear();
			fieldObj->field.adjacentNeighbours.clear();

			fieldObj->field.positions.clear();
			fieldObj->field.fieldValues.clear();

			fnPoints.clear();
		}

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
		void create(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR = 1)
		{
			fieldObj->field = zField3D<T>(_minBB, _maxBB, _n_X, _n_Y, _n_Z);

			// compute neighbours
			ringNeighbours.clear();
			adjacentNeighbours.clear();

			for (int i = 0; i < numFieldValues(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbour);
				ringNeighbours.push_back(temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbourAdjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours.push_back(temp_adjacentNeighbour);
			}

			// create field points
			createPointCloud();
		}

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
		void create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB = zVector(), int _NR = 1)
		{
			fieldObj->field = zField3D<T>(_unit_X, _unit_Y, _n_X, _n_Y, _minBB);

			// compute neighbours
			ringNeighbours.clear();
			adjacentNeighbours.clear();

			for (int i = 0; i < numFieldValues(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbour);
				ringNeighbours.push_back(temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbourAdjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours.push_back(temp_adjacentNeighbour);
			}

			// create field points
			createPointCloud();
		}

		/*! \brief This method creates a vector field from the input scalarfield.
		*	\param		[in]	inFnScalarField		- input scalar field function set.
		*	\since version 0.0.2
		*/
		void createVectorFromScalarField(zFnPointField<double> &inFnScalarField);
				

		//--------------------------
		//--- FIELD TOPOLOGY QUERY METHODS 
		//--------------------------

		/*! \brief This method gets the ring neighbours of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourhoodRing(int index, int numRings, vector<int> &ringNeighbours)
		{
			ringNeighbours.clear();


			int idX = floor(index / (fieldObj->field.n_Y * fieldObj->field.n_Z));
			int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
			int idZ = index % fieldObj->field.n_Z;

			//printf("\n%i : %i %i %i ", index, idX, idY, idZ);

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == fieldObj->field.n_X - 1) endIdX = 0;

			int endIdY = numRings;
			if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == fieldObj->field.n_Z - 1) endIdZ = 0;


			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{

					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;



						int newId = (newId_X * (fieldObj->field.n_Y* fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;


						if (newId < numFieldValues() && newId >= 0) ringNeighbours.push_back(newId);
					}

				}

			}

			
		}

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourAdjacents(int index, vector<int> &adjacentNeighbours)
		{
			adjacentNeighbours.clear();

			int numRings = 1;
			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / (fieldObj->field.n_Y *fieldObj->field.n_Z));
			int idY = floor((index - (idX *fieldObj->field.n_Z *fieldObj->field.n_Y)) / (fieldObj->field.n_Z));
			int idZ = index % fieldObj->field.n_Z;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == fieldObj->field.n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == fieldObj->field.n_Y) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == fieldObj->field.n_Z) endIdZ = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;

						int newId = (newId_X * (fieldObj->field.n_Y*fieldObj->field.n_Z)) + (newId_Y * fieldObj->field.n_Z) + newId_Z;


						if (newId < numFieldValues())
						{
							if (i == 0 || j == 0 || k == 0) adjacentNeighbours.push_back(newId);
						}
					}
				}

			}

		

		}
	

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/*! \brief This method retruns the number of scalars in the field.
		*
		*	\return			int	- number of scalars in the field.
		*	\since version 0.0.2
		*/
		int numFieldValues()
		{
			return fieldObj->field.fieldValues.size();
		}	

			
		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_n_X		- pixel resolution in x direction.
		*	\param		[out]	_n_Y		- pixel resolution in y direction.
		*	\param		[out]	_n_Z		- pixel resolution in z direction.
		*	\since version 0.0.2
		*/
		void getResolution(int &_n_X, int &_n_Y, int &_n_Z)
		{
			_n_X = fieldObj->field.n_X;
			_n_Y = fieldObj->field.n_Y;
			_n_Z = fieldObj->field.n_Z;
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each pixel in x direction.
		*	\param		[out]	_unit_Y		- size of each pixel in y direction.
		*	\param		[out]	_unit_Z		- size of each voxel in z direction
		*	\since version 0.0.2
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y, double &_unit_Z)
		{
			_unit_X = fieldObj->field.unit_X;
			_unit_Y = fieldObj->field.unit_Y;
			_unit_Z = fieldObj->field.unit_Z;
		}

		/*! \brief This method gets the bounds of the field.
		*
		*	\param		[out]	_minBB		- minimum bounds of the field.
		*	\param		[out]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.2
		*/
		void getBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			_minBB = fieldObj->field.minBB;
			_maxBB = fieldObj->field.maxBB;
		}

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\param		[in]	index		- index in the positions container.
		*	\return				zVector		- field position.
		*	\since version 0.0.2
		*/
		zVector getPosition(int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			return fnPoints.getPosition(index);
		}

		/*! \brief This method gets the position of the field at the input  X,Y and Z indicies.
		*
		*	\param		[in]	index_X		- index in X.
		*	\param		[in]	index_Y		- index in Y.
		*	\param		[in]	index_Z		- index in Z.
		*	\return				zVector		- field position.
		*	\since version 0.0.2
		*/
		zVector getPosition(int index_X, int index_Y, int index_Z)
		{
			int index = index_X * (fieldObj->field.n_Y *fieldObj->field.n_Z) + (index_Y * fieldObj->field.n_Z) + index_Z;

			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			return fnPoints.getPosition(index);
		}
		
		/*! \brief This method gets the index of the field for the input X,Y and Z indicies.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\param		[in]	index_Y		- input index in Y.
		*	\param		[in]	index_Z		- input index in Z.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(int index_X, int index_Y, int index_Z, int &index)
		{
			bool out = true;

			if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0 || index_Z >(fieldObj->field.n_Z - 1) || index_Z < 0) out = false;

			index = index_X * (fieldObj->field.n_Y *fieldObj->field.n_Z) + (index_Y * fieldObj->field.n_Z) + index_Z;

			return out;
		}

		/*! \brief This method gets the index of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(zVector &pos, int &index)

		{

			int index_X = floor((pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
			int index_Y = floor((pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);
			int index_Z = floor((pos.z - fieldObj->field.minBB.z) / fieldObj->field.unit_Z);

			bool out = getIndex(index_X, index_Y, index_Z, index);

			return out;

		}

		/*! \brief This method gets the indicies of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\param		[out]	index_Z		- output index in Z.
		*	\return				bool		- true if position is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndices(zVector &pos, int &index_X, int &index_Y, int &index_Z)
		{
			index_X = floor((pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
			index_Y = floor((pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);
			index_Z = floor((pos.z - fieldObj->field.minBB.z) / fieldObj->field.unit_Z);

			bool out = true;
			if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0 || index_Z >(fieldObj->field.n_Z - 1) || index_Z < 0) out = false;

			return out;
		}
			
		/*! \brief This method gets the value of the field at the input index.
		*
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\param		[out]	val			- field value.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getFieldValue(int index, T &val)
		{
			if (index > numFieldValues()) return false;

			val = fieldObj->field.fieldValues[index];

			return true;
		}

		/*! \brief This method gets the value of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\param		[out]	T			- field value.
		*	\return				bool		- true if sample position is within bounds.
		*	\since version 0.0.2
		*/
		bool getFieldValue(zVector &samplePos, zFieldValueType type, T& fieldValue)
		{

			bool out = false;

			int index;
			bool checkBounds = getIndex(samplePos, index);

			if (!checkBounds) return out;

			if (type == zFieldIndex)
			{
				T fVal;

				fVal = fieldObj->field.fieldValues[index];

				fieldValue = fVal;
			}

			else if (type == zFieldNeighbourWeighted)
			{
				T fVal;

				vector<int> ringNeighbours;
				getNeighbourhoodRing(index, 1, ringNeighbours);

				vector<zVector> positions;
				for (int i = 0; i < ringNeighbours.size(); i++)
				{
					positions.push_back(getPosition(ringNeighbours[i]));
				}

				vector<double> weights;
				coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

				double w;
				for (int i = 0; i < ringNeighbours.size(); i++)
				{
					T val;
					getFieldValue(ringNeighbours[i], val);
					fVal += val * weights[i];

					w += weights[i];

				}

				fVal /= w;

				fieldValue = fVal;
			}

			else if (type == zFieldAdjacentWeighted)
			{
				T fVal;

				vector<int> adjNeighbours;
				getNeighbourAdjacents(index, adjNeighbours);

				vector<zVector> positions;
				for (int i = 0; i < adjNeighbours.size(); i++)
				{
					positions.push_back(getPosition(adjNeighbours[i]));
				}

				vector<double> weights;
				coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

				double w;
				for (int i = 0; i < adjNeighbours.size(); i++)
				{
					T val;
					getFieldValue(adjNeighbours[i], val);
					fVal += val * weights[i];

					w += weights[i];
				}

				fVal /= w;

				fieldValue = fVal;
			}			

			else throw std::invalid_argument(" error: invalid zFieldValueType.");

			return true;
		}

		/*! \brief This method gets all the values of the field.
		*
		*	\param		[out]	fieldValues			- container of field values.
		*	\since version 0.0.2
		*/
		void getFieldValues(vector<T>& fieldValues)
		{
			fieldValues = fieldObj->field.fieldValues;
		}

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	epsilon		- small increment value, generally 0.001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*/
		zVector getGradient(int index, double epsilon = 0.001)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			bool out = true;

			zVector samplePos = getPosition(index);

			int id_X, id_Y, id_Z;
			getIndices(samplePos, id_X, id_Y, id_Z);

			if (id_X == 0 || id_Y == 0 || id_Z == 0 || id_X == fieldObj->field.n_X - 1 || id_Y == fieldObj->field.n_Y - 1 || id_Z == fieldObj->field.n_Z - 1)
			{				
				return zVector();
			}


			int index1;
			getIndex(id_X + 1, id_Y, id_Z, index1);
			zVector samplePos1 = getPosition(index1);
			T fieldVal1;
			bool chk1 = getFieldValue(index1, fieldVal1);

			int index2;
			getIndex(id_X, id_Y + 1, id_Z, index2);
			zVector samplePos2 = getPosition(index2);
			T fieldVal2;
			bool chk2 = getFieldValue(index2, fieldVal2);

			int index3;
			getIndex(id_X, id_Y , id_Z + 1, index3);
			zVector samplePos3 = getPosition(index3);
			T fieldVal3;
			bool chk3 = getFieldValue(index3, fieldVal3);

			T fieldVal;
			getFieldValue(index, fieldVal);



			T gX = coreUtils.ofMap(samplePos.x + epsilon, samplePos.x, samplePos1.x, fieldVal, fieldVal1) - fieldVal;
			T gY = coreUtils.ofMap(samplePos.y + epsilon, samplePos.y, samplePos2.y, fieldVal, fieldVal2) - fieldVal;
			T gZ = coreUtils.ofMap(samplePos.z + epsilon, samplePos.z, samplePos2.z, fieldVal, fieldVal3) - fieldVal;

			zVector gradient = zVector(gX, gY, gZ);
			gradient /= (2.0 * epsilon);

			return gradient;
		}

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	epsilon		- small increment value, generally 0.001.
		*	\return				zVector		- gradient vector of the field.
		*	\since version 0.0.2
		*/
		vector<zVector> getGradients(double epsilon = 0.001)
		{
			vector<zVector> out;

			for (int i = 0; i < numFieldValues(); i++)
			{
				out.push_back(getGradient(i, epsilon));
			}

			return out;
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------
				
		/*! \brief This method sets the bounds of the field.
		*
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.2
		*/
		void setBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			fieldObj->field.minBB = _minBB;
			fieldObj->field.maxBB = _maxBB;
		}

		/*! \brief This method sets the position of the field at the input index.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[in]	index		- index in the positions container.
		*	\since version 0.0.2
		*/
		void setPosition(zVector &_pos, int index)
		{
			if (index > fieldObj->field.numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			fieldObj->field.positions[index] = _pos;

		}

		/*! \brief This method sets the value of the field at the input index.
		*
		*	\param		[in]	fValue		- input value.
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\since version 0.0.2
		*/
		void setFieldValue(T fValue, int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");
			fieldObj->field.fieldValues[index] = fValue;

		}

		/*! \brief This method sets the values of the field to the input container values.
		*
		*	\param		[in]	fValue		- input container of field value.
		*	\since version 0.0.2
		*/
		void setFieldValues(vector<T> fValues)
		{
			if (fValues.size() != numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");
			
			fieldObj->field.fieldValues.clear();
			fieldObj->field.fieldValues = fValues;
			
		}
				

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjMesh &inMeshObj, T meshValue, double influence, double power = 2.0, bool normalise = true)
		{
			fieldValues.clear();
			zFnMesh inFnMesh(inMeshObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = positions[i].distanceTo(inPositions[j]);

					if (r < tempDist)
					{
						double w = pow(r, power);
						wSum += w;
						double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

						d = meshValue * val;

						tempDist = r;
					}

				}

				(wSum > 0) ? d /= wSum : d = T();

				fieldValues.push_back(d);
			}

			if (normalise)
			{
				normliseValues(fieldValues);
			}

		}

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjGraph &inGraphObj, T graphValue, double influence,  double power = 2.0, bool normalise = true)
		{
			fieldValues.clear();
			zFnGraph inFnGraph(inGraphObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnGraph.getRawVertexPositions();

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numVertices(); j++)
				{
					double r = positions[i].distanceTo(inPositions[j]);

					if (r < tempDist)
					{
						double w = pow(r, power);
						wSum += w;
						double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

						d = graphValue * val;

						tempDist = r;
					}

				}

				(wSum > 0) ? d /= wSum : d = T();

				fieldValues.push_back(d);


			}

			if (normalise)
			{
				normliseValues(fieldValues);
			}

			
		}

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, T value, double influence, double power = 2.0, bool normalise = true)
		{

			fieldValues.clear();
			zFnPointCloud inFnPoints(inPointsObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnPoints.getRawVertexPositions();

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnPoints.numVertices(); j++)
				{
					double r = positions[i].distanceTo(inPositions[j]);

					double w = pow(r, power);
					wSum += w;

					double val = (w > 0.0) ? ((r * influence) / (w)) : 0.0;;

					d += (value * val);
				}


				(wSum > 0) ? d /= wSum : d = T();

				fieldValues.push_back(d);
			}

			if (normalise)	normliseValues(fieldValues);



		}
		
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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, vector<T> &values, vector<double>& influences,  double power = 2.0, bool normalise = true)
		{
			fieldValues.clear();
			zFnPointCloud inFnPoints(inPointsObj);

			
			
			if (inFnPoints.numPoints() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
			if (inFnPoints.numPoints() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnPoints.getRawVertexPositions();

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnPoints.numVertices(); j++)
				{
					double r = positions[i].distanceTo(inPositions[j]);

					double w = pow(r, power);
					wSum += w;

					double val = (w > 0.0) ? ((r * influences[j]) / (w)) : 0.0;;

					d += (values[j] * val);
				}


				(wSum > 0) ? d /= wSum : d = T();

				fieldValues.push_back(d);
			}

			if (normalise)	normliseValues(fieldValues);		
		}

		

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
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjPointCloud &inPointsObj,  bool normalise = true)
		{
			scalars.clear();

			zFnPointCloud inFnPoints(inPointsObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnPoints.getRawVertexPositions();

			vector<double> distVals;
			double dMin = 100000;
			double dMax = 0;;

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				distVals.push_back(10000);
			}

			for (int j = 0; j < fnPoints.numVertices(); j++)
			{

				for (int i = 0; i < inFnPoints.numVertices(); i++)
				{
				
					double dist = positions[i].squareDistanceTo(inPositions[j]);

					if (dist < distVals[j])
					{
						distVals[j] = dist;
					}
				}
			}

			for (int i = 0; i < distVals.size(); i++)
			{
				dMin = coreUtils.zMin(dMin, distVals[i]);
				dMax = coreUtils.zMax(dMax, distVals[i]);
			}

			for (int j = 0; j < fnPoints.numPoints(); j++)
			{
				double val = coreUtils.ofMap(distVals[j], dMin, dMax, 0.0, 1.0);
				scalars.push_back(fnPoints.getColor(j).r);
			}
			

			if (normalise)
			{
				normliseValues(scalars);
			}

			
		}


		/*! \brief This method creates a vertex distance Field from the input mesh vertex positions.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inMeshObj			- input mesh object for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjMesh &inMeshObj, double a, double b, bool normalise = true)
		{
			scalars.clear();

			zFnMesh inFnMesh(inMeshObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();

			for (int i = 0; i < fnPoints.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = positions[i].squareDistanceTo(inPositions[j]);

					if (r < tempDist)
					{
						d = F_of_r(r, a, b);
						tempDist = r;
					}

				}

				scalars.push_back(d);
			}

			if (normalise)
			{
				normliseValues(scalars);
			}

		
		}

		/*! \brief This method creates a vertex distance Field from the input graph vertex positions.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inGraphObj		- input graph object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjGraph &inGraphObj, double a, double b,  bool normalise = true)
		{
			scalars.clear();
			
			zFnGraph inFnGraph(inGraphObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnGraph.getRawVertexPositions();

			// update values from meta balls

			for (int i = 0; i < fnPoints.numPoints(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numVertices(); j++)
				{
					double r = positions[i].squareDistanceTo(inPositions[j]);

					if (r < tempDist)
					{
						d = F_of_r(r, a, b);
						//printf("\n F_of_r:  %1.4f ", F_of_r(r, a, b));
						tempDist = r;
					}

				}

				scalars.push_back(d);
			}

			if (normalise)
			{
				normliseValues(scalars);
			}

		
		}


		/*! \brief This method creates a edge distance Field from the input mesh.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inMeshObj		- input mesh object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(vector<double> &scalars, zObjMesh &inMeshObj, double a, double b,  bool normalise = true)
		{
			scalars.clear();
			zFnMesh inFnMesh(inMeshObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();

			// update values from edge distance
			for (int i = 0; i < fnPoints.numPoints(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numEdges(); j++)
				{

					int e0 = inFnMesh.getEndVertex(j);   
					int e1 = inFnMesh.getStartVertex(j);  

					zVector closestPt;

					double r = coreUtils.minDist_Edge_Point(positions[i], inPositions[e0], inPositions[e1], closestPt);


					if (r < tempDist)
					{

						d = F_of_r(r, a, b);

						tempDist = r;
					}
				}

				scalars.push_back(d);

			}

			if (normalise)
			{
				normliseValues(scalars);
			}

			
		}


		/*! \brief This method creates a edge distance Field from the input graph.
		*
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	inGraphObj		- input graph object for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.		
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(vector<double> &scalars, zObjGraph &inGraphObj, double a, double b, bool normalise = true)
		{
			scalars.clear();
			zFnGraph inFnGraph(inGraphObj);

			zVector *positions = fnPoints.getRawVertexPositions();
			zVector *inPositions = inFnGraph.getRawVertexPositions();

			// update values from edge distance
			for (int i = 0; i < fnPoints.numPoints(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numEdges(); j++)
				{

					int e0 = inFnGraph.getEndVertex(j);
					int e1 = inFnGraph.getStartVertex(j);					
					
					zVector closestPt;
					
					double r = coreUtils.minDist_Edge_Point(positions[i], inPositions[e0], inPositions[e1], closestPt);


					if (r < tempDist)
					{

						d = F_of_r(r, a, b);

						tempDist = r;
					}
				}

				scalars.push_back(d);

			}

			if (normalise)
			{
				normliseValues(scalars);
			}

			
		}

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		
		/*! \brief This method check if the input index is the bounds of the resolution in X.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_X(int index_X)
		{
			return (index_X < fieldObj->field.n_X && index_X >= 0);
		}

		/*! \brief This method check if the input index is the bounds of the resolution in Y.
		*
		*	\param		[in]	index_Y		- input index in Y.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_Y(int index_Y)
		{
			return (index_Y < fieldObj->field.n_Y && index_Y >= 0);
		}

		/*! \brief This method check if the input index is the bounds of the resolution in Y.
		*
		*	\param		[in]	index_Y		- input index in Y.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool checkBounds_Z(int index_Z)
		{
			return (index_Z < fieldObj->field.n_Z && index_Z >= 0);
		}

		/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
		*
		*	\param	[out]	dMin		- stores the minimum scalar value
		*	\param	[out]	dMax		- stores the maximum scalar value
		*	\param	[in]	buffer		- buffer of scalars.
		*	\since version 0.0.2
		*/
		void computeMinMaxOfScalars(vector<T> &values, T &dMin, T &dMax)
		{
			dMin = coreUtils.zMin(values);
			dMax = coreUtils.zMax(values);
		}

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
		void smoothField(int numSmooth, double diffuseDamp = 1.0, zDiffusionType type = zAverage)
		{
			for (int k = 0; k < numSmooth; k++)
			{
				vector<T> tempValues;

				for (int i = 0; i < numFieldValues(); i++)
				{
					T lapA = 0;

					vector<int> ringNeigbours;
					getNeighbourhoodRing(i, 1, ringNeigbours);

					for (int j = 0; j < ringNeigbours.size(); j++)
					{
						int id = ringNeigbours[j];

						if (type == zLaplacian)
						{
							if (id != i) lapA += (getFieldValue(id) * 1);
							else lapA += (getFieldValue(id) * -8);
						}
						else if (type == zAverage)
						{
							lapA += (getFieldValue(id) * 1);
						}
					}


					if (type == zLaplacian)
					{
						double newA = getFieldValue(i) + (lapA * diffuseDamp);
						tempValues.push_back(newA);
					}
					else if (type == zAverage)
					{
						if (lapA != 0) lapA /= (ringNeigbours.size());
						tempValues.push_back(lapA);
					}

				}

				for (int i = 0; i < numFieldValues(); i++) setFieldValue(tempValues[i], i);

			}

		}
	

		/*! \brief This method computes the field index of each input position and stores them in a container per field index.
		*
		*	\param		[in]	positions			- container of positions.
		*	\param		[out]	fieldIndexPositions	- container of position per field  index.
		*	\since version 0.0.2
		*/
		void computePositionsInFieldIndex( vector<zVector> &positions, vector<vector<zVector>> &fieldIndexPositions)
		{
			for (int i = 0; i < fieldObj->field.getNumFieldValues(); i++)
			{
				vector<zVector> temp;
				fieldIndexPositions.push_back(temp);
			}


			for (int i = 0; i < positions.size(); i++)
			{
				int fieldIndex = fieldObj->field.getIndex(positions[i]);

				fieldIndexPositions[fieldIndex].push_back(positions[i]);
			}
		}

		/*! \brief This method computes the field index of each input position and stores the indicies in a container per field index.
		*
		*	\param		[in]	positions			- container of positions.
		*	\param		[out]	fieldIndexPositions	- container of position indicies per field  index.
		*	\since version 0.0.2
		*/
		void computePositionIndicesInFieldIndex( vector<zVector> &positions, vector<vector<int>> &fieldIndexPositionIndicies)
		{
			for (int i = 0; i < fieldObj->field.getNumFieldValues(); i++)
			{
				vector<int> temp;
				fieldIndexPositionIndicies.push_back(temp);
			}


			for (int i = 0; i < positions.size(); i++)
			{
				int fieldIndex = fieldObj->field.getIndex(positions[i]);

				fieldIndexPositionIndicies[fieldIndex].push_back(i);
			}
		}

		/*! \brief This method computes the distance function.
		*
		*	\param	[in]	r	- distance value.
		*	\param	[in]	a	- value of a.
		*	\param	[in]	b	- value of b.
		*	\since version 0.0.2
		*/
		double F_of_r(double &r, double &a, double &b)
		{
			if (0 <= r && r <= b / 3.0)return (a * (1.0 - (3.0 * r * r) / (b*b)));
			if (b / 3.0 <= r && r <= b) return (3 * a / 2 * pow(1.0 - (r / b), 2.0));
			if (b <= r) return 0;


		}

		
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
		void boolean_union(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < scalars0.size(); i++)
			{
				out.push_back(coreUtils.zMin(scalars0[i], scalars1[i]));
			}

			if (normalise) normliseValues(out);

			scalarsResult = out;
		}

		/*! \brief This method creates a subtraction of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_subtract(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < fieldValues_A.size(); i++)
			{
				out.push_back(zMax(fieldValues_A[i], -1 * fieldValues_B[i]));
			}

			if (normalise) normliseValues(out);

			fieldValues_Result = out;
		}

		/*! \brief This method creates a intersect of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_intersect(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < fieldValues_A.size(); i++)
			{
				out.push_back(zMax(fieldValues_A[i], fieldValues_B[i]));
			}

			if (normalise) normliseValues(out);

			fieldValues_Result = out;
		}

		/*! \brief This method creates a difference of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	fieldValues_A			- field Values A.
		*	\param	[in]	fieldValues_B			- field Values B.
		*	\param	[in]	fieldValues_Result		- resultant field value.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void boolean_difference(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = false)
		{
			vector<double> out;

			boolean_intersect(fieldValues_A, fieldValues_B, out);

			for (int i = 0; i < out.size(); i++)
			{
				out[i] *= -1;
			}

			if (normalise) normliseValues(out);

			fieldValues_Result = out;
		}

		/*! \brief This method uses an input plane to clip an existing scalar field.
		*
		*	\param	[in]	fieldMesh			- input field mesh.
		*	\param	[in]	scalars				- vector of scalar values. Need to be equivalent to number of mesh vertices.
		*	\param	[in]	clipPlane			- input zPlane used for clipping.
		*	\since version 0.0.2
		*/
		void boolean_clipwithPlane(vector<double>& scalars, zMatrixd& clipPlane)
		{
			for (int i = 0; i < fnPoints.numPoints(); i++)
			{
				zVector O = coreUtils.fromMatrixColumn(clipPlane, 3);
				zVector Z = coreUtils.fromMatrixColumn(clipPlane, 2);

				zVector A = fnPoints.getPosition(i) - O;
				double minDist_Plane = A * Z;
				minDist_Plane /= Z.length();

				// printf("\n dist %1.2f ", minDist_Plane);

				if (minDist_Plane > 0)
				{
					scalars[i] = 1;
				}

			}

		}

	

		//--------------------------
		//----  UPDATE METHODS
		//--------------------------
				
		/*! \brief This method updates the color values of the field mesh based on the scalar values. Gradient - Black to Red
		*
		*	\param	[in]	scalars		- container of  scalar values.
		*	\since version 0.0.2
		*/
		void updateFieldValues(vector<T>& values)
		{
			if (fnPoints.numPoints() == values.size())
			{
				for (int i = 0; i < fnPoints.numPoints(); i++)
				{
					setFieldValue(values[i], i);
				}

			}
			else throw std::invalid_argument("input values size not equal to number of vertices/ polygons of field mesh.");

		}

		/*! \brief This method updates the color values of the field mesh based on the scalar values. Gradient - Black to Red
		*
		*	\param	[in]	scalars		- container of  scalar values.
		*	\since version 0.0.2
		*/
		void updatePointsColors(vector<double>& scalars)
		{
			if (fnPoints.numPoints() == scalars.size() )
			{
				double dMax, dMin;
				computeMinMaxOfScalars(scalars, dMin, dMax);

				for (int i = 0; i < scalars.size(); i++)
				{
					zColor col;

					double val = coreUtils.ofMap(scalars[i], dMin, dMax, 0.0, 1.0);

					col.r = val;
					fnPoints.setColor(i, col);
					
				}

				
			}
			else throw std::invalid_argument("input scalars size not equal to number of vertices/ polygons.");

		}


		/*! \brief This method updates the color values of the field mesh based on the scalar values.
		*
		*	\param	[in]	scalars		- container of  scalar values.
		*	\param	[in]	col1		- blend color 1.
		*	\param	[in]	col2		- blend color 2.
		*	\since version 0.0.2
		*/
		void updatePointColors_Blend( vector<double>& scalars, zColor &col1, zColor &col2)
		{
			if (fnPoints.numPoints() == scalars.size() )
			{
				double dMax, dMin;
				computeMinMaxOfScalars(scalars, dMin, dMax);

				for (int i = 0; i < scalars.size(); i++)
				{
					zColor col;

					//convert to HSV
					col1.toHSV(); col2.toHSV();

					col.h = coreUtils.ofMap(scalars[i], dMin, dMax, col1.h, col2.h);
					col.s = coreUtils.ofMap(scalars[i], dMin, dMax, col1.s, col2.s);
					col.v = coreUtils.ofMap(scalars[i], dMin, dMax, col1.v, col2.v);

					col.toRGB();

					fnPoints.setColor(i, col);
					
				}				

			}

			else throw std::invalid_argument("input scalars size not equal to number of vertices/ polygons.");
		}


		/*! \brief This method updates the color values of the field mesh based on the scalar values given an input domain
		*
		*	\param	[in]	scalars		- container of  scalar values.
		*	\param	[in]	col1		- blend color 1.
		*	\param	[in]	col2		- blend color 2.
		*	\param	[in]	dMin		- domain minimum value.
		*	\param	[in]	dMin		- domain maximum value.
		*	\since version 0.0.2
		*/
		void updatePointColors_Blend( vector<double>& scalars, zColor &col1, zColor &col2, double dMin, double dMax)
		{
			if (fnPoints.numPoints() == scalars.size() )
			{
				//convert to HSV
				col1.toHSV(); col2.toHSV();

				for (int i = 0; i < scalars.size(); i++)
				{
					zColor col;


					if (scalars[i] < dMin) col = col1;
					else if (scalars[i] > dMax) col = col2;
					else
					{
						col.h = coreUtils.ofMap(scalars[i], dMin, dMax, col1.h, col2.h);
						col.s = coreUtils.ofMap(scalars[i], dMin, dMax, col1.s, col2.s);
						col.v = coreUtils.ofMap(scalars[i], dMin, dMax, col1.v, col2.v);

						col.toRGB();
					}

					fnPoints.setColor(i, col);
					
				}

			

			}

			else throw std::invalid_argument("input scalars size not equal to number of vertices/ polygons.");
		}
			
		


	};
	

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------


	//---------------//

	//---- zVector specilization for normliseFieldValues
	template<>
	inline void zFnPointField<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}
	

	//---- double specilization for normliseFieldValues
	template<>
	inline void zFnPointField<double>::normliseValues(vector<double> &fieldValues)
	{
		double dMin, dMax;
		computeMinMaxOfScalars(fieldValues, dMin, dMax);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = dMax - fieldValues[i];

		computeMinMaxOfScalars(fieldValues,dMin, dMax);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = coreUtils.ofMap(fieldValues[i], dMin, dMax, -1.0, 1.0);
	}

	//---------------//

	//---- zVector specilization for createVectorFieldFromScalarField
	template<typename T>
	inline void zFnPointField<T>::createVectorFromScalarField(zFnPointField<double> &fnScalarField)
	{
		zVector minBB, maxBB;
		fnScalarField.getBoundingBox(minBB, maxBB);

		int n_X, n_Y, n_Z;
		fnScalarField.getResolution(n_X, n_Y, n_Z);

		vector<zVector> gradients = fnScalarField.getGradients();

		create(minBB, maxBB, n_X, n_Y, n_Z);
		setFieldValues(gradients);
	}

	//---------------//

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}

