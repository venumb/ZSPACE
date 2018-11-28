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

	/*! \struct zScalar
	*	\brief A struct for storing scalar field information
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	struct zScalar
	{
		/*!	\brief stores the index of the scalar in the scalars container  */
		int id;

		/*!	\brief stores the position of the scalar  */
		zVector pos;

		/*!	\brief stores the value of the scalar  */
		double weight;

		/*!	\brief stores the ring neighbourhood indicies in the scalars container  */
		vector<int> ringNeighbours;

		/*!	\brief stores the adjacent neighbourhood indicies in the scalars container  */
		vector<int> adjacentNeighbours;
	};

	/** \addtogroup zGeometry
	*	\brief The geometry classes, modifier and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zGeometryClasses
	*	\brief The geometry classes of the library.
	*  @{
	*/

	/*! \class zScalarField2D
	*	\brief A class for 2D scalar field.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zScalarField2D
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

		/*!	\brief container for the scalar field values  */
		vector<zScalar> scalars;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/

		zScalarField2D()
		{
			scalars.clear();

			minBB = zVector(10000, 10000, 10000);
			maxBB = zVector(-10000, -10000, -10000);

			unit_X = 1.0;
			unit_Y = 1.0;
		}



		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the scalar field.
		*	\param		[in]	_maxBB		- maximum bounds of the scalar field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1. 
		*	\since version 0.0.1
		*/
		zScalarField2D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _NR = 1)
		{
			minBB = _minBB;
			maxBB = _maxBB;
			n_X = _n_X;
			n_Y = _n_Y;

			unit_X = (maxBB.x - minBB.x) / n_X;
			unit_Y = (maxBB.y - minBB.y) / n_Y;

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB + (unitVec * 0.5);

			scalars.clear();

			printf("unit_X : %1.2f unit_Y : %1.2f ", unit_X, unit_Y);

			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{
					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;

					zScalar scalar;
					scalar.id = scalars.size();
					scalar.pos = pos;
					scalar.weight = 1;
					
					scalars.push_back(scalar);
				}
			}

			// compute one ring neighbour
			for (int i = 0; i < scalars.size(); i++)
			{
				scalars[i].ringNeighbours.clear();
				getNeighbourHoodRing(i, _NR, scalars[i].ringNeighbours);

				scalars[i].adjacentNeighbours.clear();
				getNeighbourAdjacents(i, scalars[i].adjacentNeighbours);
			}
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.1
		*/
		
		zScalarField2D(double _unit_X, double _unit_Y, int _n_X, int _n_Y, int _NR = 1)
		{
			unit_X = _unit_X;
			unit_Y = _unit_Y;
			n_X = _n_X;
			n_Y = _n_Y;


			minBB = zVector(0, 0, 0);
			maxBB = zVector(unit_X * n_X, unit_Y * n_Y, 0);

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB + (unitVec * 0.5);


			for (int i = 0; i< n_X; i++)
			{
				for (int j = 0; j < n_Y; j++)
				{
					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;


					zScalar scalar;
					scalar.id = scalars.size();
					scalar.pos = pos;
					scalar.weight = 1;
				
					scalars.push_back(scalar);
				}
			}

			// compute one ring neighbour
			for (int i = 0; i < scalars.size(); i++)
			{
				scalars[i].ringNeighbours.clear();
				getNeighbourHoodRing(i, _NR, scalars[i].ringNeighbours);

				scalars[i].adjacentNeighbours.clear();
				getNeighbourAdjacents(i, scalars[i].adjacentNeighbours);
			}


		}
		
		
		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/

		~zScalarField2D(){}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method retruns the number of scalars in the field.
		*		
		*	\return			int	- number of scalars in the field.
		*	\since version 0.0.1
		*/
		
		int getNumScalars()
		{
			return scalars.size();
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

		/*! \brief This method sets the position of the scalar at the input index.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		void setPosition(zVector &_pos, int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");
			
			scalars[index].pos = _pos;
		
		}

		/*! \brief This method gets the position of the scalar at the input index.
		*
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		zVector getPosition(int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			return scalars[index].pos;
		}

		/*! \brief This method sets the weight/value of the scalar at the input index.
		*
		*	\param		[in]	weight		- input value.
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		void setWeight(double weight, int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			scalars[index].weight = weight;
		}

		/*! \brief This method gets the waight/value of the scalar at the input index.
		*
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/
		
		double getWeight(int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			return scalars[index].weight;
		}			

		

		/*! \brief This method gets the index of the scalar at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\since version 0.0.1
		*/
		int getIndex(zVector &pos)
		
		{
			int index_X = floor((pos.x - minBB.x) / unit_X);
			int index_Y = floor((pos.y - minBB.y) / unit_Y);

			if (index_X >  (n_X - 1) || index_X <  0 || index_Y >(n_Y - 1) || index_Y <  0) throw std::invalid_argument(" error: input position out of bounds.");

			return index_X * n_Y + index_Y;


		}

		/*! \brief This method gets the indicies of the scalar at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\since version 0.0.1
		*/
		
		void getIndices(zVector &pos, int &index_X, int &index_Y)
		{
			index_X = floor((pos.x - minBB.x) / unit_X);
			index_Y = floor((pos.y - minBB.y) / unit_Y);

			if (index_X >  (n_X - 1) || index_X <  0 || index_Y >(n_Y - 1) || index_Y <  0) throw std::invalid_argument(" error: input position out of bounds.");
		}

		/*! \brief This method gets the ring neighbours of the scalar at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.	
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/
		
		void getNeighbourHoodRing(int index, int numRings,  vector<int> &ringNeighbours)
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


					if (newId < getNumScalars()) out.push_back(newId);
				}

			}

			ringNeighbours = out;
		}

		/*! \brief This method gets the immediate adjacent neighbours of the scalar at the input index.
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


					if (newId < getNumScalars())
					{
						if (i == 0 || j == 0) out.push_back(newId);
					}
				}

			}

			adjacentNeighbours = out;

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

	/*! \class zScalarField£D
	*	\brief A class for 3D scalar field.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/

	class zScalarField3D
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

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector minBB;

		/*!	\brief stores the minimum bounds of the scalar field  */
		zVector maxBB;

		/*!	\brief container for the scalar field values  */
		vector<zScalar> scalars;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/

		zScalarField3D()
		{
			scalars.clear();

			minBB = zVector(10000, 10000, 10000);
			maxBB = zVector(-10000, -10000, -10000);

			unit_X = 1.0;
			unit_Y = 1.0;
			unit_Z = 1.0;
		}



		/*! \brief Overloaded constructor.
		*	\param		[in]	_minBB		- minimum bounds of the scalar field.
		*	\param		[in]	_maxBB		- maximum bounds of the scalar field.
		*	\param		[in]	_n_X		- number of voxels in x direction.
		*	\param		[in]	_n_Y		- number of voxels in y direction.
		*	\param		[in]	_n_Z		- number of voxels in z direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.1
		*/
		
		zScalarField3D(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _n_Z, int _NR = 1)
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

			scalars.clear();

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

						zScalar scalar;
						scalar.id = scalars.size();
						scalar.pos = pos;
						scalar.weight = 1;

						scalars.push_back(scalar);
					}
					
				}
			}

			// compute one ring neighbour
			for (int i = 0; i < scalars.size(); i++)
			{
				scalars[i].ringNeighbours.clear();
				getNeighbourHoodRing(i, _NR, scalars[i].ringNeighbours);

				scalars[i].adjacentNeighbours.clear();
				getNeighbourAdjacents(i, scalars[i].adjacentNeighbours);
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

		zScalarField3D(double _unit_X, double _unit_Y, double _unit_Z, int _n_X, int _n_Y, int _n_Z, int _NR = 1)
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

						zScalar scalar;
						scalar.id = scalars.size();
						scalar.pos = pos;
						scalar.weight = 1;

						scalars.push_back(scalar);
					}
				}
			}

			// compute one ring neighbour
			for (int i = 0; i < scalars.size(); i++)
			{
				scalars[i].ringNeighbours.clear();
				 getNeighbourHoodRing(i, _NR, scalars[i].ringNeighbours);

				scalars[i].adjacentNeighbours.clear();
				getNeighbourAdjacents(i, scalars[i].adjacentNeighbours);
			}


		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/

		~zScalarField3D() {}

		//--------------------------
		//---- GET-SET METHODS
		//--------------------------

		/*! \brief This method retruns the number of scalars in the field.
		*
		*	\return			int	- number of scalars in the field.
		*	\since version 0.0.1
		*/

		int getNumScalars()
		{
			return scalars.size();
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

		/*! \brief This method sets the position of the scalar at the input index.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		void setPosition(zVector &_pos, int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			scalars[index].pos = _pos;

		}

		/*! \brief This method gets the position of the scalar at the input index.
		*
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		zVector getPosition(int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			return scalars[index].pos;
		}

		/*! \brief This method sets the weight/value of the scalar at the input index.
		*
		*	\param		[in]	weight		- input value.
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		void setWeight(double weight, int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			scalars[index].weight = weight;
		}

		/*! \brief This method gets the waight/value of the scalar at the input index.
		*
		*	\param		[in]	index		- index in the scalar container.
		*	\since version 0.0.1
		*/

		double getWeight(int index)
		{
			if (index > getNumScalars()) throw std::invalid_argument(" error: index out of bounds.");

			return scalars[index].weight;
		}


		/*! \brief This method gets the index of the scalar at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\since version 0.0.1
		*/
		int getIndex(zVector &pos)

		{
			int index_X, index_Y, index_Z;
			getIndices(pos, index_X, index_Y, index_Z);
			

			return index_X * (n_Y*n_Z) + index_Y * n_Z + index_Z;


		}

		/*! \brief This method gets the indicies of the scalar at the input position.
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

		/*! \brief This method gets the ring neighbours of the scalar at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.1
		*/

		void getNeighbourHoodRing(int index, int numRings, vector<int> &ringNeighbours)
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


						if (newId < getNumScalars()) out.push_back(newId);
					}
					
				}

			}

			ringNeighbours = out;
		}

		/*! \brief This method gets the immediate adjacent neighbours of the scalar at the input index.
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


						if (newId < getNumScalars())
						{
							if (i == 0 || j == 0 || k == 0) out.push_back(newId);
						}
					}
				}

			}

			adjacentNeighbours = out;

		}



	};

}