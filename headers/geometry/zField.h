#pragma once

#include<headers/core/zVector.h>
#include<headers/core/zColor.h>



namespace zSpace
{

	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	
	/** \addtogroup zGeometryClasses
	*	\brief The geometry classes of the library.	
	*  @{
	*/

	/*! \class zField2D
	*	\brief A template class for 2D fields - scalar and vector.
	*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class zField2D
	{
	private:

		//--------------------------
		//----  PRIVATE ATTRIBUTES
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

		/*!	\brief container of field positions.   */
		vector<zVector> positions;
			   		 

	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
		
		/*!	\brief container for the field values  */
		vector<T> fieldValues;

		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<vector<int>> ringNeighbours;

		/*!	\brief container of adjacent neighbourhood indicies.  */
		vector<vector<int>> adjacentNeighbours;


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

			minBB = zVector(10000, 10000, 10000);
			maxBB = zVector(-10000, -10000, -10000);

			unit_X = 1.0;
			unit_Y = 1.0;
		}



		/*! \brief Overloaded constructor.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1. 
		*	\since version 0.0.1
		*/
		zField2D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _NR = 1)
		{
			minBB = _minBB;
			maxBB = _maxBB;
			n_X = _n_X;
			n_Y = _n_Y;

			unit_X = (maxBB.x - minBB.x) / n_X;
			unit_Y = (maxBB.y - minBB.y) / n_Y;

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB + (unitVec * 0.5);

			fieldValues.clear();

			printf("unit_X : %1.2f unit_Y : %1.2f ", unit_X, unit_Y);

			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{
					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;

					positions.push_back(pos);

					T defaultValue;
					fieldValues.push_back(defaultValue);
					
				}
			}

			// compute one ring neighbour
			for (int i = 0; i < positions.size(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbour);
				ringNeighbours.push_back(temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;				
				getNeighbourAdjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours.push_back(temp_adjacentNeighbour);
			}
		}

		/*! \brief Overloaded constructor.
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.1
		*/		
		zField2D(double _unit_X, double _unit_Y, int _n_X, int _n_Y, int _NR = 1)
		{
			unit_X = _unit_X;
			unit_Y = _unit_Y;
			n_X = _n_X;
			n_Y = _n_Y;


			minBB = zVector(0, 0, 0);
			maxBB = zVector(unit_X * n_X, unit_Y * n_Y, 0);

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB + (unitVec * 0.5);
			
			fieldValues.clear();

			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{
					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;

					positions.push_back(pos);

					T defaultValue;
					fieldValues.push_back(defaultValue);

				}
			}

			// compute one ring neighbour
			for (int i = 0; i < positions.size(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbour);
				ringNeighbours.push_back(temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbourAdjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours.push_back(temp_adjacentNeighbour);
			}


		}
		
		
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\since version 0.0.1
		*/
		~zField2D(){}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method retruns the number of scalars in the field.
		*	
		*	\return			int	- number of scalars in the field.
		*	\since version 0.0.1
		*/		
		int numFieldValues()
		{
			return fieldValues.size();
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_n_X		- pixel resolution in x direction.
		*	\param		[out]	_n_Y		- pixel resolution in y direction.
		*	\since version 0.0.1
		*/
		void getResolution(int &_n_X, int &_n_Y)
		{
			_n_X = n_X;
			_n_Y = n_Y;
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each pixel in x direction.
		*	\param		[out]	_unit_Y		- size of each pixel in y direction.
		*	\since version 0.0.1
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y)
		{
			_unit_X = unit_X;
			_unit_Y = unit_Y;
		
		}

		/*! \brief This method sets the bounds of the field.
		*
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.1
		*/		
		void setBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			minBB = _minBB;
			maxBB = _maxBB;
		}

		/*! \brief This method gets the bounds of the field.
		*
		*	\param		[out]	_minBB		- minimum bounds of the field.
		*	\param		[out]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.1
		*/		
		void getBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			_minBB = minBB;
			_maxBB = maxBB;
		}

		/*! \brief This method sets the position of the field at the input index.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[in]	index		- index in the positions container.
		*	\since version 0.0.1
		*/
		void setPosition(zVector &_pos, int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");
			
			positions[index] = _pos;
		
		}

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\param		[in]	index		- index in the positions container.
		*	\return				zVector		- field position.
		*	\since version 0.0.1
		*/

		zVector getPosition(int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			return positions[index];
		}

		/*! \brief This method sets the value of the field at the input index.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	fValue		- input value.
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\since version 0.0.1
		*/
		void setFieldValue(T fValue, int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");
			fieldValues[index] = fValue;

		}

		/*! \brief This method gets the value of the field at the input index.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\param		[out]	val			- field value.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.1
		*/		
		bool getFieldValue(int index, T &val)
		{
			if (index > numFieldValues()) return false;

			val = fieldValues[index];			

			return true;
		}			
			
		/*! \brief This method gets the index of the field for the input X and Y indicies.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\param		[in]	index_Y		- input index in Y.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.1
		*/
		bool getIndex(int index_X, int index_Y, int &index)
		{
			bool out = true;

			if (index_X > (n_X - 1) || index_X <  0 || index_Y >(n_Y - 1) || index_Y < 0) out = false;

			index =  index_X * n_Y + index_Y;

			return out;
		}

		/*! \brief This method gets the index of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.1
		*/
		bool getIndex(zVector &pos, int &index)
		
		{
			int index_X = floor((pos.x - minBB.x) / unit_X);
			int index_Y = floor((pos.y - minBB.y) / unit_Y);

			bool out = getIndex(index_X, index_Y, index);
			
			return out;

		}
		

		/*! \brief This method gets the indicies of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\return				bool		- true if position is in bounds.
		*	\since version 0.0.1
		*/		
		bool getIndices(zVector &pos, int &index_X, int &index_Y)
		{
			index_X = floor((pos.x - minBB.x) / unit_X);
			index_Y = floor((pos.y - minBB.y) / unit_Y);

			bool out = true;
			if (index_X >  (n_X - 1) || index_X <  0 || index_Y >(n_Y - 1) || index_Y <  0) out = false;

			return out;
		}

		/*! \brief This method check if the input index is the bounds of the resolution in X.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.1
		*/
		bool checkBounds_X(int index_X)
		{
			return (index_X < n_X && index_X >= 0);
		}

		/*! \brief This method check if the input index is the bounds of the resolution in Y.
		*
		*	\param		[in]	index_Y		- input index in Y.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.1
		*/
		bool checkBounds_Y(int index_Y)
		{
			return (index_Y < n_Y && index_Y >= 0);
		}		

		/*! \brief This method gets the ring neighbours of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.	
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/		
		void getNeighbourhoodRing(int index, int numRings,  vector<int> &ringNeighbours)
		{
			vector<int> out;

			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / n_Y);
			int idY = index % n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == n_Y) endIdY = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * n_Y) + (newId_Y);


					if (newId < numFieldValues()) out.push_back(newId);
				}

			}

			ringNeighbours = out;
		}

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.		
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/		
		void getNeighbourAdjacents(int index, vector<int> &adjacentNeighbours)
		{
			vector<int> out;

			int numRings = 1;
			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / n_Y);
			int idY = index % n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == n_Y) endIdY = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * n_Y) + (newId_Y);


					if (newId < numFieldValues())
					{
						if (i == 0 || j == 0) out.push_back(newId);
					}
				}

			}

			adjacentNeighbours = out;

		}

		/*! \brief This method gets the gridPoints which contain the input position.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	containedGridPoints	- contatiner of contained points indicies.
		*	\since version 0.0.1
		*/
		void getNeighbourContained(zVector &pos, vector<int> &containedNeighbour)
		{

			vector<int> out;
			containedNeighbour.clear();
			
			int index;
			bool checkBounds = getIndex(pos, index);

			if (!checkBounds) return;

			int numRings = 1;
			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / n_Y);
			int idY = index % n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == n_Y) endIdY = 0;

			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * n_Y) + (newId_Y);										
				}

			}

			// case 1
			if (checkBounds_X(idX - 1) && checkBounds_Y(idY + 1))
			{
				int g1; 
				getIndex(idX - 1, idY, g1);

				int g2;
				getIndex(idX , idY, g2);

				int g3;
				getIndex(idX, idY+1, g3);

				int g4;
				getIndex(idX-1, idY+1, g4);

				zVector minBB_temp = getPosition(g1) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);

				bool check = pointInBounds(pos, minBB_temp, maxBB_temp);

				if (check)
				{
					containedNeighbour.push_back(g1);
					containedNeighbour.push_back(g2);
					containedNeighbour.push_back(g3);
					containedNeighbour.push_back(g4);
				}
				
			}

			// case 2
			if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY + 1))
			{

				int g1;
				getIndex(idX, idY, g1);

				int g2;
				getIndex(idX + 1, idY, g2);

				int g3;
				getIndex(idX +1, idY + 1, g3);

				int g4;
				getIndex(idX, idY + 1, g4);

				zVector minBB_temp = getPosition(g1) - zVector(unit_X * 0.5, unit_Y * 0.5,0);
				zVector maxBB_temp = getPosition(g3) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);	

				bool check = pointInBounds(pos, minBB_temp, maxBB_temp);

				if (check)
				{
					containedNeighbour.push_back(g1);
					containedNeighbour.push_back(g2);
					containedNeighbour.push_back(g3);
					containedNeighbour.push_back(g4);
				}

			}

			// case 3
			if (containedNeighbour.size() == 0 && checkBounds_X(idX + 1) && checkBounds_Y(idY - 1))
			{

				int g1;
				getIndex(idX, idY -1, g1);

				int g2;
				getIndex(idX + 1, idY -1, g2);

				int g3;
				getIndex(idX + 1, idY , g3);

				int g4;
				getIndex(idX, idY , g4);

				zVector minBB_temp = getPosition(g1) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);

				bool check = pointInBounds(pos, minBB_temp, maxBB_temp);

				if (check)
				{
					containedNeighbour.push_back(g1);
					containedNeighbour.push_back(g2);
					containedNeighbour.push_back(g3);
					containedNeighbour.push_back(g4);
				}				
			}


			// case 4
			if (containedNeighbour.size() == 0 && checkBounds_X(idX - 1) && checkBounds_Y(idY - 1))
			{

				int g1;
				getIndex(idX-1, idY - 1, g1);

				int g2;
				getIndex(idX , idY - 1, g2);

				int g3;
				getIndex(idX , idY, g3);

				int g4;
				getIndex(idX -1, idY, g4);

				zVector minBB_temp = getPosition(g1) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(unit_X * 0.5, unit_Y * 0.5, 0);

				bool check = pointInBounds(pos, minBB_temp, maxBB_temp);

				if (check)
				{
					containedNeighbour.push_back(g1);
					containedNeighbour.push_back(g2);
					containedNeighbour.push_back(g3);
					containedNeighbour.push_back(g4);
				}				
			}
				

		}

		/*! \brief This method avarages / smoothens the field values.
		*
		*	\param		[in]	numSmooth			- number of times to smooth.
		*	\param		[in]	diffuseDamp			- damping value of the averaging.
		*	\param		[in]	type				- smooth type - zlaplacian / zAverage.
		*	\since version 0.0.1
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

		/*! \brief This method gets the value of the field at the input sample position.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\param		[out]	T			- field value.
		*	\return				bool		- true if sample position is within bounds.
		*	\since version 0.0.1
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

				fVal =  fieldValues[index];

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
				getDistanceWeights(samplePos, positions, 2.0, weights);

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
				getDistanceWeights(samplePos, positions, 2.0, weights);

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

			else if (type == zFieldContainedWeighted)
			{

				T fVal;

				vector<int> containedNeighbours;
				getNeighbourContained(samplePos, containedNeighbours);

				vector<zVector> positions;
				for (int i = 0; i < containedNeighbours.size(); i++)
				{
					positions.push_back(getPosition(containedNeighbours[i]));
				}

				vector<double> weights;
				getDistanceWeights(samplePos, positions,2.0, weights);

				double w =0.0;
				for (int i = 0; i < containedNeighbours.size(); i++)
				{
					T val;
					getFieldValue(containedNeighbours[i], val);
					fVal += (val * weights[i]);					

					w += weights[i];
				}
							   		
				fVal /= w;

				fieldValue = fVal;
				
			}

			else throw std::invalid_argument(" error: invalid zFieldValueType.");

			return true;
		}

		/*! \brief This method gets the gradient of the field at the input sample position.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	gradient	- gradient vector of the field.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\return		[in]	episilon	- small increment value, generally 0.001.
		*	\return				bool		- false if index is out of bounds.
		*	\since version 0.0.1
		*/
		bool getGradient(int index, zVector &gradient,  zFieldValueType type = zFieldContainedWeighted, double epsilon = 0.001)
		{
			bool out = true;

			zVector samplePos = getPosition(index);

			/*vector<int> ringNeigbours;
			getNeighbourhoodRing(index, 1, ringNeigbours);

			int maxId;
			double maxValue = -1000000;;
			for (int i = 0; i < ringNeigbours.size(); i++)
			{
				double val;
				val = getFieldValue(ringNeigbours[i]);

				if (val > maxValue)
				{
					maxValue = val;
					maxId = i;
				}
			}

			gradient = getPosition(ringNeigbours[maxId]) - samplePos;*/

			T fieldVal;
			getFieldValue(index, fieldVal);

			int id_X, id_Y;
			getIndices(samplePos, id_X, id_Y);

			if (id_X == n_X - 1 || id_Y == n_Y - 1) epsilon *= -1;

			zVector samplepos1(samplePos.x + epsilon, samplePos.y, samplePos.z);
			zVector samplepos2(samplePos.x , samplePos.y + epsilon, samplePos.z);

			T fieldVal1;
			bool chk1 = getFieldValue(samplepos1, type, fieldVal1);

			T fieldVal2;
			bool chk2 = getFieldValue(samplepos2, type, fieldVal2);

			printf("\n val %1.8f %1.8f %1.8f \n ", fieldVal1, fieldVal2, fieldVal);


			T gX = (id_X == n_X - 1 || id_Y == n_Y - 1) ? fieldVal - fieldVal1 :fieldVal1 - fieldVal;
			T gY = (id_X == n_X - 1 || id_Y == n_Y - 1) ? fieldVal - fieldVal2 :fieldVal2 - fieldVal;

			gradient = zVector( gX,gY,0);
			printf("\n %1.8f %1.8f %1.8f \n ", gradient.x, gradient.y, gradient.z);

			gradient /= (2.0 * epsilon);

			return out;
		}

	};




	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryClasses
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zField3D
	*	\brief A template class for 3D fields - scalar and vector.
	*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	template <typename T>
	class zField3D
	{
	private:

		//--------------------------
		//----  PRIVATE ATTRIBUTES
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

		/*!	\brief cantainer of field  positions  */
		vector<zVector> positions;

		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<vector<int>> ringNeighbours;

		/*!	\brief container of  the adjacent neighbourhood indicies.  */
		vector<vector<int>> adjacentNeighbours;

		
	public:

		//--------------------------
		//----  ATTRIBUTES
		//--------------------------
		
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

			minBB = zVector(10000, 10000, 10000);
			maxBB = zVector(-10000, -10000, -10000);

			unit_X = 1.0;
			unit_Y = 1.0;
			unit_Z = 1.0;
		}



		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.1
		*/		
		zField3D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR = 1)
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

			// compute one ring neighbour
			for (int i = 0; i < fieldValues.size(); i++)
			{
				vector<int> temp_ringNeighbours;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbours);
				ringNeighbours.push_back(temp_ringNeighbours);

				vector<int> temp_adjacentNeighbours;
				getNeighbourAdjacents(i, temp_adjacentNeighbours);
				adjacentNeighbours.push_back(temp_adjacentNeighbours);
			}
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each voxel in x direction.
		*	\param		[in]	_unit_Y		- size of each voxel in y direction.
		*	\param		[in]	_unit_Z		- size of each voxel in yz direction.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.1
		*/
		zField3D(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, int _NR = 1)
		{
			unit_X = _unit_X;
			unit_Y = _unit_Y;
			unit_Z = _unit_Z;

			n_X = _n_X;
			n_Y = _n_Y;
			n_Z = _n_Z;


			minBB = zVector(0, 0, 0);
			maxBB = zVector(unit_X * n_X, unit_Y * n_Y, unit_Z * n_Z);

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

			// compute one ring neighbour
			for (int i = 0; i < fieldValues.size(); i++)
			{
				vector<int> temp_ringNeighbours;
				getNeighbourhoodRing(i, _NR, temp_ringNeighbours);
				ringNeighbours.push_back(temp_ringNeighbours);

				vector<int> temp_adjacentNeighbours;
				getNeighbourAdjacents(i, temp_adjacentNeighbours);
				adjacentNeighbours.push_back(temp_adjacentNeighbours);
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

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method retruns the number of fieldvalues in the field.
		*
		*	\return			int	- number of fieldvalues in the field.
		*	\since version 0.0.1
		*/
		int numFieldValues()
		{
			return fieldValues.size();
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_n_X		- pixel resolution in x direction.
		*	\param		[out]	_n_Y		- pixel resolution in y direction.
		*	\param		[out]	_n_Z		- pixel resolution in z direction.
		*	\since version 0.0.1
		*/
		void getResolution(int &_n_X, int &_n_Y, int &_n_Z)
		{
			_n_X = n_X;
			_n_Y = n_Y;
			_n_Z = n_Z;
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each voxel in x direction.
		*	\param		[out]	_unit_Y		- size of each voxel in y direction.
		*	\param		[out]	_unit_Z		- size of each voxel in z direction
		*	\since version 0.0.1
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y, double &_unit_Z)
		{
			_unit_X = unit_X;
			_unit_Y = unit_Y;
			_unit_Z = unit_Z;
		}

		/*! \brief This method sets the bounds of the field.
		*
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.1
		*/
		void setBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			minBB = _minBB;
			maxBB = _maxBB;
		}

		/*! \brief This method gets the bounds of the field.
		*
		*	\param		[out]	_minBB		- minimum bounds of the field.
		*	\param		[out]	_maxBB		- maximum bounds of the field.
		*	\since version 0.0.1
		*/
		void getBoundingBox(zVector &_minBB, zVector &_maxBB)
		{
			_minBB = minBB;
			_maxBB = maxBB;
		}

		/*! \brief This method sets the position of the field at the input index.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[in]	index		- index in the positions container.
		*	\since version 0.0.1
		*/
		void setPosition(zVector &_pos, int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			positions[index] = _pos;

		}

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\param		[in]	index		- index in the positions container.
		*	\return				zvector		- field position.
		*	\since version 0.0.1
		*/
		zVector getPosition(int index)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			return positions[index];
		}

		/*! \brief This method sets the value of the field at the input index.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	fValue		- input value.
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/
		void setFieldValue(T fValue, int index)
		{
			if (index > getFieldValue()) throw std::invalid_argument(" error: index out of bounds.");

			fieldValues[index] = fValue;
		}

		/*! \brief This method gets the waight/value of the scalar at the input index.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	index		- index in the scalar container.
		*	\return				T			- field value.
		*	\since version 0.0.1
		*/
		T getFieldValue(int index)
		{
			if (index > getFieldValue()) throw std::invalid_argument(" error: index out of bounds.");

			return fieldValues[index];
		}

		/*! \brief This method gets the index of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\return				int			- field index.
		*	\since version 0.0.1
		*/
		int getIndex(zVector &pos)

		{
			int index_X, index_Y, index_Z;
			getIndices(pos, index_X, index_Y, index_Z);
			

			return index_X * (n_Y*n_Z) + index_Y * n_Z + index_Z;


		}

		/*! \brief This method gets the indicies of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\since version 0.0.1
		*/
		void getIndices(zVector &pos, int &index_X, int &index_Y, int &index_Z)
		{
			index_X = floor((pos.x - minBB.x) / unit_X);
			index_Y = floor((pos.y - minBB.y) / unit_Y);
			index_Z = floor((pos.z - minBB.z) / unit_Z);

			if (index_X >  (n_X - 1) || index_X <  0 || index_Y >(n_Y - 1) || index_Y <  0 || index_Z >(n_Z - 1) || index_Z <  0) throw std::invalid_argument(" error: input position out of bounds.");
		}

		/*! \brief This method gets the ring neighbours of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/
		void getNeighbourhoodRing(int index, int numRings, vector<int> &ringNeighbours)
		{
			vector<int> out;

			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / (n_Y * n_Z));
			int idY = floor(index / (n_Z));			
			int idZ = index % n_Z;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == n_Y) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == n_Z) endIdZ = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{

					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;

						int newId = (newId_X * (n_Y*n_Z)) + (newId_Y * n_Z) + newId_Z;


						if (newId < numFieldValues()) out.push_back(newId);
					}
					
				}

			}

			ringNeighbours = out;
		}

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/
		void getNeighbourAdjacents(int index, vector<int> &adjacentNeighbours)
		{
			vector<int> out;

			int numRings = 1;
			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / (n_Y * n_Z));
			int idY = floor(index / (n_Z));
			int idZ = index % n_Z;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == n_Y) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == n_Z) endIdZ = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;

						int newId = (newId_X * (n_Y*n_Z)) + (newId_Y * n_Z) + newId_Z;


						if (newId < numFieldValues())
						{
							if (i == 0 || j == 0 || k == 0) out.push_back(newId);
						}
					}
				}

			}

			adjacentNeighbours = out;

		}

		/*! \brief This method gets the value of the field at the input sample position.
		*
		*	\tparam				T			- Type to work with standard c++ numerical datatypes and zVector.
		*	\param		[in]	samplePos	- index in the fieldvalues container.
		*	\param		[in]	type		- type of sampling.  zFieldIndex / zFieldNeighbourWeighted / zFieldAdjacentWeighted
		*	\return				T			- field value.
		*	\since version 0.0.1
		*/
		T getFieldValue(zVector &samplePos, zFieldValueType type = zFieldIndex)
		{
			bool checkBounds = pointInBounds(samplePos, minBB, maxBB);

			if (!checkBounds) throw std::invalid_argument(" error: samplePosition out of bounds.");

			T out;

			int index = getIndex(samplePos);

			if (type == zFieldIndex)
			{
				return fieldValues[index];
			}

			else if (type == zFieldNeighbourWeighted)
			{
				vector<int> ringNeighbours;
				getNeighbourhoodRing(index, 1, ringNeighbours);

				vector<zVector> samplePositions;
				for (int i = 0; i < ringNeighbours.size(); i++)
				{
					samplePositions.push_back(getsamplePosition(ringNeighbours[i]));
				}

				vector<double> weights;
				getDistanceWeights(samplePos, samplePositions,2.0, weights);

				for (int i = 0; i < ringNeighbours.size(); i++)
				{
					out += getFieldValue(ringNeighbours[i]) * weights[i];
				}

				out /= ringNeighbours.size();
			}

			else if (type == zFieldAdjacentWeighted)
			{
				vector<int> adjNeighbours;
				getNeighbourAdjacents(index, adjNeighbours);

				vector<zVector> samplePositions;
				for (int i = 0; i < adjNeighbours.size(); i++)
				{
					samplePositions.push_back(getsamplePosition(adjNeighbours[i]));
				}

				vector<double> weights;
				getDistanceWeights(samplePos, samplePositions, 2.0, weights);

				for (int i = 0; i < adjNeighbours.size(); i++)
				{
					out += getFieldValue(adjNeighbours[i]) * weights[i];
				}

				out /= adjNeighbours.size();
			}

			else throw std::invalid_argument(" error: invalid zFieldValueType.");

			return out;
		}


	};

}