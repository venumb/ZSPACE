#pragma once

#include<headers/api/object/zObjSpatialBin.h>

#include<headers/api/object/zObjMesh.h>
#include<headers/api/object/zObjGraph.h>
#include<headers/api/object/zObjPointCloud.h>

#include<headers/api/functionsets/zFnPointCloud.h>


namespace  zSpace
{


	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/	

	/*! \class zFnSpatialBin
	*	\brief A spatial binning function-set.
	*	\since version 0.0.3
	*/
	
	/** @}*/
	/** @}*/

	class zFnSpatialBin
	{
	protected:
		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief stores pointers of objects */
		vector<zObj*> objects;

		/*!	\brief pointer to a field 3D object  */
		zObjSpatialBin *binObj;

		/*! \brief This method creates the point cloud from the field parameters.
		*
		*	\since version 0.0.2
		*/
		void createPointCloud()
		{
			vector<zVector>positions;

			zVector minBB = binObj->field.minBB;
			zVector maxBB = binObj->field.maxBB;
			
			double unit_X = binObj->field.unit_X;
			double unit_Y = binObj->field.unit_Y;
			double unit_Z = binObj->field.unit_Z;
			
			int n_X = binObj->field.n_X;
			int n_Y = binObj->field.n_Y;
			int n_Z = binObj->field.n_Z;						

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
		*	\since version 0.0.3
		*/
		zFnSpatialBin()
		{
			binObj = nullptr;
		}


		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_binObj		- input spatial bin object.
		*	\since version 0.0.3
		*/
		zFnSpatialBin(zObjSpatialBin &_binObj)
		{
			binObj = &_binObj;
			fnPoints = zFnPointCloud(_binObj);
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.3
		*/
		~zFnSpatialBin() {}

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		void clear()
		{
			binObj->bins.clear();
			binObj->field.fieldValues.clear();
			fnPoints.clear();
		}

		//--------------------------
		//---- CREATE METHOD
		//--------------------------

		/*! \brief This method creates the spatial bins from the input bounds.
		*
		*	\param  	[out]	minBB		- bounding box minimum.
		*	\param		[out]	maxBB		- bounding box maximum.
		*	\param  	[in]	_res		- resolution of bins.
		*	\since version 0.0.3
		*/
		void create(const zVector &_minBB = zVector(0,0,0), const zVector &_maxBB = zVector(10,10,10), int _res = 30)
		{
			binObj->field = zField3D<double>(_minBB, _maxBB, _res, _res, _res);
		
			// initialise bins
			binObj->bins.assign(_res*_res*_res, zBin());

			// compute neighbours
			ringNeighbours.clear();
			adjacentNeighbours.clear();

			for (int i = 0; i < numBins(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbourhoodRing(i, 1, temp_ringNeighbour);
				ringNeighbours.push_back(temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbourAdjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours.push_back(temp_adjacentNeighbour);
			}

			// create field points
			createPointCloud();
			
		}
		

		//--------------------------
		//---- GET METHODS
		//--------------------------

			/*! \brief This method gets the number of objects in the spatial bin.
		*
		*	\return			int	- number of objects in the spatial bin.
		*	\since version 0.0.3
		*/
		int numObjects()
		{
			return objects.size();
		}

		/*! \brief This method gets the number of bins in the spatial bin.
		*
		*	\return			int	- number of bins in the spatial bin.
		*	\since version 0.0.3
		*/
		int numBins()
		{
			return binObj->bins.size();
		}

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


			int idX = floor(index / (binObj->field.n_Y * binObj->field.n_Z));
			int idY = floor((index - (idX *binObj->field.n_Z *binObj->field.n_Y)) / (binObj->field.n_Z));
			int idZ = index % binObj->field.n_Z;

			//printf("\n%i : %i %i %i ", index, idX, idY, idZ);

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == binObj->field.n_X - 1) endIdX = 0;

			int endIdY = numRings;
			if (idY == binObj->field.n_Y - 1) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == binObj->field.n_Z - 1) endIdZ = 0;


			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{

					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;



						int newId = (newId_X * (binObj->field.n_Y* binObj->field.n_Z)) + (newId_Y * binObj->field.n_Z) + newId_Z;


						if (newId < numBins() && newId >= 0) ringNeighbours.push_back(newId);
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

			int idX = floor(index / (binObj->field.n_Y *binObj->field.n_Z));
			int idY = floor((index - (idX *binObj->field.n_Z *binObj->field.n_Y)) / (binObj->field.n_Z));
			int idZ = index % binObj->field.n_Z;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int startIdZ = -numRings;
			if (idZ == 0) startIdZ = 0;

			int endIdX = numRings;
			if (idX == binObj->field.n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == binObj->field.n_Y) endIdY = 0;

			int endIdZ = numRings;
			if (idZ == binObj->field.n_Z) endIdZ = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					for (int k = startIdZ; k <= endIdZ; k++)
					{
						int newId_X = idX + i;
						int newId_Y = idY + j;
						int newId_Z = idZ + k;

						int newId = (newId_X * (binObj->field.n_Y*binObj->field.n_Z)) + (newId_Y * binObj->field.n_Z) + newId_Z;


						if (newId < numBins())
						{
							if (i == 0 || j == 0 || k == 0) adjacentNeighbours.push_back(newId);
						}
					}
				}

			}



		}		


		/*! \brief This method gets the index of the bin for the input X,Y and Z indicies.
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

			if (index_X > (binObj->field.n_X - 1) || index_X <  0 || index_Y >(binObj->field.n_Y - 1) || index_Y < 0 || index_Z >(binObj->field.n_Z - 1) || index_Z < 0) out = false;

			index = index_X * (binObj->field.n_Y *binObj->field.n_Z) + (index_Y * binObj->field.n_Z) + index_Z;

			return out;
		}

		/*! \brief This method gets the index of the bin at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(zVector &pos, int &index)
		{

			int index_X = floor((pos.x - binObj->field.minBB.x) / binObj->field.unit_X);
			int index_Y = floor((pos.y - binObj->field.minBB.y) / binObj->field.unit_Y);
			int index_Z = floor((pos.z - binObj->field.minBB.z) / binObj->field.unit_Z);

			bool out = getIndex(index_X, index_Y, index_Z, index);

			return out;

		}
	
		/*! \brief This method gets the indicies of the bin at the input position.
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
			index_X = floor((pos.x - binObj->field.minBB.x) / binObj->field.unit_X);
			index_Y = floor((pos.y - binObj->field.minBB.y) / binObj->field.unit_Y);
			index_Z = floor((pos.z - binObj->field.minBB.z) / binObj->field.unit_Z);

			bool out = true;
			if (index_X > (binObj->field.n_X - 1) || index_X <  0 || index_Y >(binObj->field.n_Y - 1) || index_Y < 0 || index_Z >(binObj->field.n_Z - 1) || index_Z < 0) out = false;

			return out;
		}
				

		//--------------------------
		//---- METHODS
		//--------------------------

		/*! \brief This method adds a new object to the bin 
		*
		*	\param		[in]	inObj			- input object. Works for objects with position informations - pointcloud, graphs and meshes.
		*	\since version 0.0.3
		*/
		template<typename T>
		void addObject(T &inObj);

		

		/*! \brief This method clears the bins.
		*
		*	\since version 0.0.3
		*/
		void clearBins()
		{
			binObj->bins.clear();			
		}


		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------

	protected:

		/*! \brief This method checjks if the bounds of the input object is less than the current bounds of the bins. If not the bins bunds is resized.
		*
		*	\param		[in]	inObj		- input object.
		*	\return				bool		- true if bounds of bins is resized else false.
		*	\since version 0.0.3
		*/
		bool boundsCheck(zObj &inObj)
		{
			zVector minBB, maxBB;
			binObj->getBounds(minBB, maxBB);

			zVector minBB_Obj, maxBB_Obj;
			inObj.getBounds(minBB_Obj, maxBB_Obj);

			bool resized = false;

			if (minBB_Obj.x < minBB.x) { resized = true; minBB.x = minBB_Obj.x; }
			if (minBB_Obj.y < minBB.y) { resized = true; minBB.y = minBB_Obj.y; }
			if (minBB_Obj.z < minBB.z) { resized = true; minBB.z = minBB_Obj.z; }

			if (maxBB_Obj.x > maxBB.x) { resized = true; maxBB.x = maxBB_Obj.x; }
			if (maxBB_Obj.y > maxBB.y) { resized = true; maxBB.y = maxBB_Obj.y; }
			if (maxBB_Obj.z > maxBB.z) { resized = true; maxBB.z = maxBB_Obj.z; }

			if (resized)
			{
				int resX = ((maxBB.x - minBB.x) / binObj->field.unit_X )+ 1;
				int resY = ((maxBB.y - minBB.y) / binObj->field.unit_Y) + 1;
				int resZ = ((maxBB.z - minBB.z) / binObj->field.unit_Z) + 1;

				int res = (resX > resY) ? resX : resY;
				res = (res > resZ) ? res : resZ;

				clear();
				create(minBB, maxBB, res);

				// partition to bin existing objects
				int i = 0;
				for (auto &obj : objects)
				{
					partitionToBins(obj, i);
					i++;
				}

			}

			return resized;
		}

		/*! \brief This method partitions the  new object to the bin
		*
		*	\param		[in]	inObj			- input object. Works for objects with position informations - pointcloud, graphes and meshes.
		*	\since version 0.0.3
		*/
		template<typename T>
		void partitionToBins(T &inObj, int objectId);

		/*! \brief This method returns the squared length of the zVector.
		*
		*	\param		[in]	inPos		- input position to be added to bin.
		*	\param		[in]	pointId		- input index of the position in the container.
		*	\param		[in]	objectId	- input index of the object the position belongs to.
		*	\since version 0.0.3
		*/
		void partitionToBin(zVector &inPos, int pointId, int objectId)
		{

			int index;
			bool check = getIndex(inPos, index);

			if (check)
			{
				if (objectId >= objects.size()) throw std::invalid_argument(" error: object index out of bounds.");

				else binObj->bins[index].addVertexIndex(pointId, objectId);
			}

		}
		
	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------

	//---------------//

	//---- pointcloud specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjPointCloud & inObj, int objectId)
	{
		zVector *positions = &inObj.pCloud.vertexPositions[0];

		for (int i = 0; i < inObj.pCloud.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}

	//---- graph specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjGraph & inObj, int objectId)
	{
		zVector *positions = &inObj.graph.vertexPositions[0];

		for (int i = 0; i < inObj.graph.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}

	//---- mesh specilization for addObject
	template<>
	inline void zFnSpatialBin::partitionToBins(zObjMesh & inObj, int objectId)
	{
		zVector *positions = &inObj.mesh.vertexPositions[0];

		for (int i = 0; i < inObj.mesh.n_v; i++)
			partitionToBin(positions[i], i, objectId);

	}	

	//---------------//

	//---- pointcloud specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjPointCloud & inObj)
	{

		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);
	
		partitionToBins(inObj, objects.size() - 1);

	}

	//---- graph specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjGraph & inObj)
	{
		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);

		partitionToBins(inObj, objects.size() - 1);
	}

	//---- mesh specilization for addObject
	template<>
	inline void zFnSpatialBin::addObject(zObjMesh & inObj)
	{
		bool chk = boundsCheck(inObj);

		objects.push_back(&inObj);

		partitionToBins(inObj, objects.size() - 1);
	}

	//---------------//

#endif 
	
	/* DOXYGEN_SHOULD_SKIP_THIS */
}

