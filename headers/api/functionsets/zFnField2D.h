#pragma once

#include<headers/api/object/zObjGraph.h>
#include<headers/api/object/zObjMesh.h>
#include<headers/api/object/zObjField2D.h>

#include<headers/api/functionsets/zFnMesh.h>
#include<headers/api/functionsets/zFnGraph.h>

#include<headers/framework/utilities/zUtilsBMP.h>

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

	/*! \class zFnField2D
	*	\brief A 2D field function set.
	*
	*	\tparam				T			- Type to work with double(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/	

	/** @}*/

	/** @}*/

	template<typename T>
	class zFnField2D 
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to a field 2D object  */
		zObjField2D<T> *fieldObj;

		/*!	\brief pointer to mesh object  */
		zObjMesh *fieldMeshObj;

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method exports the input field to a bitmap file format based on the face color of the correspoding field mesh.
		*
		*	\param [in]		outfilename		- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toBMP(string outfilename)
		{
			int resX = fieldObj->field.n_X;
			int resY = fieldObj->field.n_Y;

			zUtilsBMP bmp(resX, resY);

			uint32_t channels = bmp.bmp_info_header.bit_count / 8;

			for (uint32_t x = 0; x < resX; ++x)
			{
				for (uint32_t y = 0; y < resY; ++y)
				{
					int faceId;
					getIndex(x, y, faceId);

					// blue
					bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 0] = fnMesh.getFaceColor(faceId).b * 255;

					// green
					bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 1] = fnMesh.getFaceColor(faceId).g * 255;

					// red
					bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 2] = fnMesh.getFaceColor(faceId).r * 255;

					// alpha
					if (channels == 4)
					{
						bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 3] = fnMesh.getFaceColor(faceId).a * 255;
					}

				}

			}

			bmp.write(outfilename.c_str());
		}


		/*! \brief This method imorts the input bitmap file and creates the corresponding field and  field mesh. The Bitmap needs to be in grey-scale colors only to update field values.
		*
		*	\param		[in]		infilename		- input file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void fromBMP(string infilename);

		/*! \brief This method creates the field mesh from the input scalar field.
		*
		*	\param		[in]	inMesh			- output mesh.
		*	\since version 0.0.2
		*/
		void createFieldMesh()
		{

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			zVector minBB, maxBB;
			double unit_X, unit_Y;
			int n_X, n_Y;

			getUnitDistances(unit_X, unit_Y);
			getResolution(n_X, n_Y);

			getBoundingBox(minBB, maxBB);

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB;

			int resX = n_X + 1;
			int resY = n_Y + 1;

			for (int i = 0; i < resX; i++)
			{
				for (int j = 0; j < resY; j++)
				{
					zVector pos;
					pos.x = startPt.x + i * unitVec.x;
					pos.y = startPt.y + j * unitVec.y;

					positions.push_back(pos);
				}
			}



			for (int i = 0; i < resX - 1; i++)
			{
				for (int j = 0; j < resY - 1; j++)
				{
					int v0 = (i * resY) + j;
					int v1 = ((i + 1) * resY) + j;

					int v2 = v1 + 1;
					int v3 = v0 + 1;

					polyConnects.push_back(v0);
					polyConnects.push_back(v1);
					polyConnects.push_back(v2);
					polyConnects.push_back(v3);

					polyCounts.push_back(4);
				}
			}


			fnMesh.create(positions, polyCounts, polyConnects);;
			
			printf("\n fieldMesh: %i %i %i", fnMesh.numVertices(), fnMesh.numEdges(), fnMesh.numPolygons());


		}
		

	public:

		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief mesh function set  */
		zFnMesh fnMesh;

		/*!	\brief container of the ring neighbourhood indicies.  */
		vector<vector<int>> ringNeighbours;

		/*!	\brief container of adjacent neighbourhood indicies.  */
		vector<vector<int>> adjacentNeighbours;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnField2D() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\param		[in]	_fieldMeshObj		- input mesh object.
		*	\since version 0.0.2
		*/
		zFnField2D(zObjField2D<T> &_fieldObj, zObjMesh &_fieldMeshObj)
		{
			fieldObj = &_fieldObj;

			fieldMeshObj = &_fieldMeshObj;			
			fnMesh = zFnMesh(_fieldMeshObj);			
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnField2D() {}


		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method exports the field to the given file type.
		*
		*	\param [in]		path			- output file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported - zBMP
		*	\since version 0.0.2
		*/
		void from(string path, zFileTpye type) 
		{
			if (type == zBMP) fromBMP(path);
			
			else throw std::invalid_argument(" error: invalid zFileTpye type");

		}

		/*! \brief This method imports the field to the given file type.
		*
		*	\param [in]		path			- input file name including the directory path and extension.
		*	\param [in]		type			- type of file to be exported - zBMP
		*	\since version 0.0.2
		*/
		void to(string path, zFileTpye type) 
		{
			if (type == zBMP) toBMP(path);
		
			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

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

			fnMesh.clear();
		}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/** \addtogroup mesh creation
		*	\brief Collection of mesh creation methods.
		*  @{
		*/

		/*! \brief This method creates a field from the input parameters.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_maxBB		- maximum bounds of the field.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.2
		*/
		void create(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _NR = 1)
		{
			fieldObj->field = zField2D<T>(_minBB, _maxBB, _n_X, _n_Y);

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

			createFieldMesh();
		}

		/*! \brief Overloaded constructor.
		*	\param		[in]	_unit_X		- size of each pixel in x direction.
		*	\param		[in]	_unit_Y		- size of each pixel in y direction.
		*	\param		[in]	_n_X		- number of pixels in x direction.
		*	\param		[in]	_n_Y		- number of pixels in y direction.
		*	\param		[in]	_minBB		- minimum bounds of the field.
		*	\param		[in]	_NR			- ring number of neighbours to be computed. By default it is 1.
		*	\since version 0.0.2
		*/
		void create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB = zVector(), int _NR = 1)
		{
			fieldObj->field = zField2D<T>(_unit_X, _unit_Y, _n_X, _n_Y, _minBB);

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

			createFieldMesh();
		}

		/*! \brief This method creates a vector field from the input scalarfield.
		*	\param		[in]	inFnScalarField		- input scalar field function set.
		*	\since version 0.0.2
		*/
		void createVectorFromScalarField(zFnField2D<double> &inFnScalarField);
		

		/** @}*/

		//--------------------------
		//--- FIELD TOPOLOGY QUERY METHODS 
		//--------------------------

		/** \addtogroup field topology queries
		*	\brief Collection of field topology query methods.
		*  @{
		*/

		/*! \brief This method gets the ring neighbours of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourhoodRing(int index, int numRings, vector<int> &ringNeighbours)
		{
			vector<int> out;

			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / fieldObj->field.n_Y);
			int idY = index % fieldObj->field.n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == fieldObj->field.n_X - 1) endIdX = 0;

			int endIdY = numRings;
			if (idY == fieldObj->field.n_Y - 1) endIdY = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


					if (newId < numFieldValues()) out.push_back(newId);
				}

			}

			ringNeighbours = out;
		}

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourAdjacents(int index, vector<int> &adjacentNeighbours)
		{
			vector<int> out;

			int numRings = 1;
			//printf("\n working numRings : %i ", numRings);

			int idX = floor(index / fieldObj->field.n_Y);
			int idY = index % fieldObj->field.n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == fieldObj->field.n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == fieldObj->field.n_Y) endIdY = 0;

			for (int i = startIdX; i <= endIdX; i++)
			{
				for (int j = startIdY; j <= endIdY; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);


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
		*	\since version 0.0.2
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

			int idX = floor(index / fieldObj->field.n_Y);
			int idY = index % fieldObj->field.n_Y;

			int startIdX = -numRings;
			if (idX == 0) startIdX = 0;

			int startIdY = -numRings;
			if (idY == 0) startIdY = 0;

			int endIdX = numRings;
			if (idX == fieldObj->field.n_X) endIdX = 0;

			int endIdY = numRings;
			if (idY == fieldObj->field.n_Y) endIdY = 0;

			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int newId_X = idX + i;
					int newId_Y = idY + j;

					int newId = (newId_X * fieldObj->field.n_Y) + (newId_Y);
				}

			}

			// case 1
			if (checkBounds_X(idX - 1) && checkBounds_Y(idY + 1))
			{
				int g1;
				getIndex(idX - 1, idY, g1);

				int g2;
				getIndex(idX, idY, g2);

				int g3;
				getIndex(idX, idY + 1, g3);

				int g4;
				getIndex(idX - 1, idY + 1, g4);

				zVector minBB_temp = getPosition(g1) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

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
				getIndex(idX + 1, idY + 1, g3);

				int g4;
				getIndex(idX, idY + 1, g4);

				zVector minBB_temp = getPosition(g1) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

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
				getIndex(idX, idY - 1, g1);

				int g2;
				getIndex(idX + 1, idY - 1, g2);

				int g3;
				getIndex(idX + 1, idY, g3);

				int g4;
				getIndex(idX, idY, g4);

				zVector minBB_temp = getPosition(g1) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

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
				getIndex(idX - 1, idY - 1, g1);

				int g2;
				getIndex(idX, idY - 1, g2);

				int g3;
				getIndex(idX, idY, g3);

				int g4;
				getIndex(idX - 1, idY, g4);

				zVector minBB_temp = getPosition(g1) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);
				zVector maxBB_temp = getPosition(g3) - zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				bool check = coreUtils.pointInBounds(pos, minBB_temp, maxBB_temp);

				if (check)
				{
					containedNeighbour.push_back(g1);
					containedNeighbour.push_back(g2);
					containedNeighbour.push_back(g3);
					containedNeighbour.push_back(g4);
				}
			}


		}

		/** @}*/
				

		//--------------------------
		//---- GET METHODS
		//--------------------------

		/** \addtogroup field get methods
		*	\brief Collection of field get attribute methods.
		*  @{
		*/
		
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
		*	\since version 0.0.2
		*/
		void getResolution(int &_n_X, int &_n_Y)
		{
			_n_X = fieldObj->field.n_X;
			_n_Y = fieldObj->field.n_Y;
		}

		/*! \brief This method gets the unit distances of the field.
		*
		*	\param		[out]	_unit_X		- size of each pixel in x direction.
		*	\param		[out]	_unit_Y		- size of each pixel in y direction.
		*	\since version 0.0.2
		*/
		void getUnitDistances(double &_unit_X, double &_unit_Y)
		{
			_unit_X = fieldObj->field.unit_X;
			_unit_Y = fieldObj->field.unit_Y;

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

			return fieldObj->field.positions[index];
		}

		/*! \brief This method gets the position of the field at the input index.
		*
		*	\param		[in]	index		- index in the positions container.
		*	\return				zVector		- field position.
		*	\since version 0.0.2
		*/
		zVector getPosition(int index_X, int index_Y)
		{
			int index = index_X * fieldObj->field.n_Y + index_Y;

			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");

			return fieldObj->field.positions[index];
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

		/*! \brief This method gets the index of the field for the input X and Y indicies.
		*
		*	\param		[in]	index_X		- input index in X.
		*	\param		[in]	index_Y		- input index in Y.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndex(int index_X, int index_Y, int &index)
		{
			bool out = true;

			if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0) out = false;

			index = index_X * fieldObj->field.n_Y + index_Y;

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

			bool out = getIndex(index_X, index_Y, index);

			return out;

		}

		/*! \brief This method gets the indicies of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index_X		- output index in X.
		*	\param		[out]	index_Y		- output index in Y.
		*	\return				bool		- true if position is in bounds.
		*	\since version 0.0.2
		*/
		bool getIndices(zVector &pos, int &index_X, int &index_Y)
		{
			index_X = floor((pos.x - fieldObj->field.minBB.x) / fieldObj->field.unit_X);
			index_Y = floor((pos.y - fieldObj->field.minBB.y) / fieldObj->field.unit_Y);

			bool out = true;
			if (index_X > (fieldObj->field.n_X - 1) || index_X <  0 || index_Y >(fieldObj->field.n_Y - 1) || index_Y < 0) out = false;

			return out;
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
				coreUtils.getDistanceWeights(samplePos, positions, 2.0, weights);

				double w = 0.0;
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

			int id_X, id_Y;
			getIndices(samplePos, id_X, id_Y);

			if (id_X == 0 || id_Y == 0 || id_X == fieldObj->field.n_X - 1 || id_Y == fieldObj->field.n_Y - 1)
			{				
				return zVector();
			}


			int index1;
			getIndex(id_X + 1, id_Y, index1);
			zVector samplePos1 = getPosition(index1);
			T fieldVal1;
			bool chk1 = getFieldValue(index1, fieldVal1);

			int index2;
			getIndex(id_X, id_Y + 1, index2);
			zVector samplePos2 = getPosition(index2);
			T fieldVal2;
			bool chk2 = getFieldValue(index2, fieldVal2);

			T fieldVal;
			getFieldValue(index, fieldVal);



			T gX = coreUtils.ofMap(samplePos.x + epsilon, samplePos.x, samplePos1.x, fieldVal, fieldVal1) - fieldVal;
			T gY = coreUtils.ofMap(samplePos.y + epsilon, samplePos.y, samplePos2.y, fieldVal, fieldVal2) - fieldVal;

			zVector gradient = zVector(gX, gY, 0);
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

		/** @}*/

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/** \addtogroup field set methods
		*	\brief Collection of field set attribute methods.
		*  @{
		*/

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

		/** @}*/

		//--------------------------
		//----  2D IDW FIELD METHODS
		//--------------------------

		/** \addtogroup IDW distance field
		*	\brief Collection of IDW field methods.
		*  @{
		*/

		/*! \brief This method computes the field values as inverse weighted distance from the input mesh vertex positions.
		*
		*	\param	[in]	inFnMesh			- input mesh function set for distance calculations.
		*	\param	[in]	meshValue			- value to be propagated for the mesh.
		*	\param	[in]	influences			- influence value of the graph.
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(zFnMesh &inFnMesh, T meshValue, double influence, vector<T> &fieldValues, double power = 2.0, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = fnMesh.getVertexPosition(i).distanceTo(inFnMesh.getVertexPosition(j));

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

				out.push_back(d);
			}

			if (normalise)
			{
				normliseValues(out);
			}


			fieldValues = out;
		}

		/*! \brief This method computes the field values as inverse weighted distance from the input graph vertex positions.
		*
		*	\param	[in]	inFngraph			- input grahh function set for distance calculations.
		*	\param	[in]	graphValue			- value to be propagated for the graph.
		*	\param	[in]	influences			- influence value of the graph.
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(zFnGraph &inFngraph, T graphValue, double influence, vector<T> &fieldValues, double power = 2.0, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFngraph.numVertices(); j++)
				{
					double r = fnMesh.getVertexPosition(i).distanceTo(inFngraph.getVertexPosition(j));

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

				out.push_back(d);


			}

			if (normalise)
			{
				normliseValues(out);
			}

			fieldValues = out;
		}

		/*! \brief This method computes the field values based on inverse weighted distance from the input positions.
		*
		*	\param	[in]	inPositions			- container of input positions for distance calculations.
		*	\param	[in]	values				- value to be propagated for each input position. Size of container should be equal to inPositions.
		*	\param	[in]	influences			- influence value of each input position. Size of container should be equal to inPositions.
		*	\param	[out]	fieldValues			- container for storing field values.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<zVector> &inPositions, vector<T> &values, vector<double>& influences, vector<T> &fieldValues, double power = 2.0, bool normalise = true)
		{
			if (inPositions.size() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
			if (inPositions.size() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

			vector<T> out;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;



				for (int j = 0; j < inPositions.size(); j++)
				{
					double r = fnMesh.getVertexPosition(i).distanceTo(inPositions[j]);

					double w = pow(r, power);
					wSum += w;

					double val = (w > 0.0) ? ((r * influences[j]) / (w)) : 0.0;;

					d += (values[j] * val);
				}


				(wSum > 0) ? d /= wSum : d = T();

				out.push_back(d);
			}

			if (normalise)	normliseValues(out);


			fieldValues = out;
		}

		/** @}*/

		//--------------------------
		//----  2D SCALAR FIELD METHODS
		//--------------------------

		/** \addtogroup scalar distance field
		*	\brief Collection of distance field methods.
		*  @{
		*/

		/*! \brief This method creates a vertex distance Field from the input vector of zVector positions.
		*
		*	\param	[in]	points				- container of positions.
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<zVector> &points, vector<double> &scalars, bool normalise = true)
		{
			vector<double> out;

			vector<double> distVals;
			double dMin = 100000;
			double dMax = 0;;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				distVals.push_back(10000);
			}

			for (int i = 0; i < points.size(); i++)
			{
				for (int j = 0; j < fnMesh.numVertices(); j++)
				{
					double dist = fnMesh.getVertexPosition(j).distanceTo(points[i]);

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

			for (int j = 0; j < fnMesh.numVertices(); j++)
			{
				double val = coreUtils.ofMap(distVals[j], dMin, dMax, 0.0, 1.0);
				fnMesh.setVertexColor(j, zColor(val, 0, 0, 1));
			}

			fnMesh.computeFaceColorfromVertexColor();

			for (int j = 0; j < fnMesh.numPolygons(); j++)
			{
				out.push_back(fnMesh.getFaceColor(j).r);
			}

			if (normalise)
			{
				normliseValues(out);
			}

			scalars = out;
		}


		/*! \brief This method creates a vertex distance Field from the input mesh vertex positions.
		*
		*	\param	[in]	inFnMesh			- input mesh function set for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zFnMesh &inFnMesh, double a, double b, vector<double> &scalars, bool normalise = true)
		{
			vector<double> out;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = fnMesh.getVertexPosition(i).distanceTo(inFnMesh.getVertexPosition(j));

					if (r < tempDist)
					{
						d = F_of_r(r, a, b);
						tempDist = r;
					}

				}

				out.push_back(d);
			}

			if (normalise)
			{
				normliseValues(out);
			}

			scalars = out;
		}

		/*! \brief This method creates a vertex distance Field from the input graph vertex positions.
		*
		*	\param	[in]	inFnGraph		- input graph function set for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(zFnGraph &inFnGraph, double a, double b, vector<double> &scalars, bool normalise = true)
		{
			vector<double> out;


			// update values from meta balls

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numVertices(); j++)
				{
					double r = fnMesh.getVertexPosition(i).distanceTo(inFnGraph.getVertexPosition(j));

					if (r < tempDist)
					{
						d = F_of_r(r, a, b);
						//printf("\n F_of_r:  %1.4f ", F_of_r(r, a, b));
						tempDist = r;
					}

				}

				out.push_back(d);
			}

			if (normalise)
			{
				normliseValues(out);
			}

			scalars = out;
		}


		/*! \brief This method creates a edge distance Field from the input mesh.
		*
		*	\param	[in]	inFnMesh		- input mesh function set for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(zFnMesh &inFnMesh, double a, double b, vector<double> &scalars, bool normalise = true)
		{
			vector<double> out;

			// update values from edge distance
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numEdges(); j++)
				{

					int e0 = inFnMesh.getEndVertex(j);   
					int e1 = inFnMesh.getStartVertex(j);  

					zVector closestPt;

					double r = coreUtils.minDist_Edge_Point(fnMesh.getVertexPosition(i), inFnMesh.getVertexPosition(e0), inFnMesh.getVertexPosition(e1), closestPt);


					if (r < tempDist)
					{

						d = F_of_r(r, a, b);

						tempDist = r;
					}
				}

				out.push_back(d);

			}

			if (normalise)
			{
				normliseValues(out);
			}

			scalars = out;
		}


		/*! \brief This method creates a edge distance Field from the input graph.
		*
		*	\param	[in]	inFnGraph		- input graph function set for distance calculations.
		*	\param	[in]	a				- input variable for distance function.
		*	\param	[in]	b				- input variable for distance function.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	normalise		- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsEdgeDistance(zFnGraph &inFnGraph, double a, double b, vector<double> &scalars, bool normalise = true)
		{
			vector<double> out;

			// update values from edge distance
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numEdges(); j++)
				{

					int e0 = inFnGraph.getEndVertex(j);
					int e1 = inFnGraph.getStartVertex(j);					
					
					zVector closestPt;
					
					double r = coreUtils.minDist_Edge_Point(fnMesh.getVertexPosition(i), inFnGraph.getVertexPosition(e0), inFnGraph.getVertexPosition(e1), closestPt);


					if (r < tempDist)
					{

						d = F_of_r(r, a, b);

						tempDist = r;
					}
				}

				out.push_back(d);

			}

			if (normalise)
			{
				normliseValues(out);
			}

			scalars = out;
		}

		/** @}*/

		//--------------------------
		//----  2D SD FIELD METHODS
		//--------------------------

		/** \addtogroup scalar signed distance field
		*	\brief Collection of scalar signed distance field methods.
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*  @{
		*/

		/*! \brief This method gets the signed distance scalar for the input line.
		*
		*	\param	[in]	p				- input field point.
		*	\param	[in]	v0				- input positions 0 of line.
		*	\param	[in]	v1				- input positions 1 of line.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		double getScalar_sdLine(zVector& p, zVector& v0, zVector& v1)
		{
			zVector pa = p - v0;
			zVector ba = v1 - v0;

			float h = coreUtils.ofClamp((pa* ba) / (ba* ba), 0.0, 1.0);

			zVector out = pa - (ba*h);


			return (out.length());

		}

		/*! \brief This method gets the signed distance scalar for the input line.
		*
		*	\param	[in]	p				- input field point.
		*	\param	[in]	v0				- input positions 0 of line.
		*	\param	[in]	v1				- input positions 1 of line.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		double getScalar_sdTrapezoid(zVector& p, float& r1, float& r2, float &he, zVector &cen = zVector())
		{
			/*zVector k1(r2, he, 0);
			zVector ba = b - v1;

			float h = coreUtils.ofClamp((pa* ba) / (ba* ba), 0.0, 1.0);

			zVector out = pa - (ba*h);


			return (out.length());*/

		}

		/** @}*/

		//--------------------------
		//--- COMPUTE METHODS 
		//--------------------------

		/** \addtogroup field compute
		*	\brief Collection of mesh compute methods.
		*  @{
		*/

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

		/** @}*/

		//--------------------------
		//----  BOOLEAN METHODS
		//--------------------------
		
		/** \addtogroup field boolean operators
		*	\brief Collection of field update methods.
		*  @{
		*/

		/*! \brief This method creates a union of the fields at the input buffers and stores them in the result buffer.
		*
		*	\param	[in]	scalars0				- value of buffer.
		*	\param	[in]	scalars1				- value of buffer.
		*	\param	[in]	scalarsResult			- value of buffer to store the results.
		*	\param	[in]	normalise				- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void union_values(vector<double>& scalars0, vector<double>& scalars1, vector<double>& scalarsResult, bool normalise = true)
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
		void subtract_values(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = true)
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
		void intersect_values(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = true)
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
		void difference_values(vector<double>& fieldValues_A, vector<double>& fieldValues_B, vector<double>& fieldValues_Result, bool normalise = false)
		{
			vector<double> out;

			intersect_values(fieldValues_A, fieldValues_B, out);

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
		void clipwithPlane(vector<double>& scalars, zMatrixd& clipPlane)
		{
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				zVector O = coreUtils.fromMatrixColumn(clipPlane, 3);
				zVector Z = coreUtils.fromMatrixColumn(clipPlane, 2);

				zVector A = fnMesh.getVertexPosition(i) - O;
				double minDist_Plane = A * Z;
				minDist_Plane /= Z.length();

				// printf("\n dist %1.2f ", minDist_Plane);

				if (minDist_Plane > 0)
				{
					scalars[i] = 1;
				}

			}

		}

		/** @}*/

		//--------------------------
		//----  UPDATE METHODS
		//--------------------------

		/** \addtogroup field update
		*	\brief Collection of field update methods.
		*  @{
		*/

		/*! \brief This method updates the color values of the field mesh based on the scalar values. Gradient - Black to Red
		*
		*	\param	[in]	scalars		- container of  scalar values.
		*	\since version 0.0.2
		*/
		void updateFieldValues(vector<T>& values)
		{
			if (fnMesh.numVertices() == values.size())
			{
				for (int i = 0; i < fnMesh.numPolygons(); i++)
				{
					vector<int> fVerts;
					fnMesh.getVertices(i, zFaceData, fVerts);
					T val;

					for (int j = 0; j < fVerts.size(); j++)
					{
						val += values[fVerts[j]];
					}

					val /= fVerts.size();

					setFieldValue(val, i);
				}

			}
			else if (fnMesh.numPolygons() == values.size())
			{
				for (int i = 0; i < fnMesh.numPolygons(); i++)
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
		void updateMeshColors(vector<double>& scalars)
		{
			if (fnMesh.numVertices() == scalars.size() || fnMesh.numPolygons() == scalars.size())
			{
				double dMax, dMin;
				computeMinMaxOfScalars(scalars, dMin, dMax);

				for (int i = 0; i < scalars.size(); i++)
				{
					zColor col;

					double val = coreUtils.ofMap(scalars[i], dMin, dMax, 0.0, 1.0);

					col.r = val;
					if (fnMesh.numVertices() == scalars.size()) fnMesh.setVertexColor(i, col);
					else fnMesh.setFaceColor(i, col);
				}

				if (fnMesh.numPolygons() == scalars.size()) fnMesh.computeVertexColorfromFaceColor();
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
		void updateMeshColors_Blend( vector<double>& scalars, zColor &col1, zColor &col2)
		{
			if (fnMesh.numVertices() == scalars.size() || fnMesh.numPolygons() == scalars.size())
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

					if (fnMesh.numVertices() == scalars.size()) fnMesh.setVertexColor(i, col);
					else fnMesh.setFaceColor(i, col);
				}

				if (fnMesh.numPolygons() == scalars.size()) fnMesh.computeVertexColorfromFaceColor();

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
		void updateMeshColors_Blend( vector<double>& scalars, zColor &col1, zColor &col2, double dMin, double dMax)
		{
			if (fnMesh.numVertices() == scalars.size() || fnMesh.numPolygons() == scalars.size())
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

					if (fnMesh.numVertices() == scalars.size()) fnMesh.setVertexColor(i, col);
					else fnMesh.setFaceColor(i, col);
				}

				if (fnMesh.numPolygons() == scalars.size()) fnMesh.computeVertexColorfromFaceColor();

			}

			else throw std::invalid_argument("input scalars size not equal to number of vertices/ polygons.");
		}

		/** @}*/

		//--------------------------
		//---- CONTOUR METHODS
		//--------------------------

		/** \addtogroup field countour 
		*	\brief Collection of field contouring methods.
		*  @{
		*/

		/*! \brief This method creates a isocontour graph from the input field mesh at the given field threshold.
		*
		*	\param	[in]	threshold	- field threshold.
		*	\return			zGraph		- contour graph.
		*	\since version 0.0.2
		*/
		zObjGraph getIsocontour(double threshold = 0.5)
		{
			vector<double> scalarsValues;

			vector<zVector> pos;
			vector<int> edgeConnects;

			vector<int> edgetoIsoGraphVertexId;

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				scalarsValues.push_back(fnMesh.getVertexColor(i).r);
			}

			// compute positions
			for (int i = 0; i < fnMesh.numEdges(); i += 2)
			{
				edgetoIsoGraphVertexId.push_back(-1);
				edgetoIsoGraphVertexId.push_back(-1);

				
				int eV0 = fnMesh.getEndVertexIndex(i);
				int eV1 = fnMesh.getStartVertexIndex(i);
						
					

				double scalar_lower = (scalarsValues[eV0] <= scalarsValues[eV1]) ? scalarsValues[eV0] : scalarsValues[eV1];
				double scalar_higher = (scalarsValues[eV0] <= scalarsValues[eV1]) ? scalarsValues[eV1] : scalarsValues[eV0];;

				bool chkSplitEdge = (scalar_lower <= threshold && scalar_higher > threshold) ? true : false;

				if (chkSplitEdge)
				{
					// calculate split point

					int scalar_lower_vertId = (scalarsValues[eV0] <= scalarsValues[eV1]) ? eV0 : eV1;
					int scalar_higher_vertId = (scalarsValues[eV0] <= scalarsValues[eV1]) ? eV1 : eV0;

					zVector scalar_lower_vertPos = fnMesh.getVertexPosition(scalar_lower_vertId);
					zVector scalar_higher_vertPos = fnMesh.getVertexPosition(scalar_higher_vertId);

					double scaleVal = coreUtils.ofMap(threshold, scalar_lower, scalar_higher, 0.0, 1.0);

					zVector e = scalar_higher_vertPos - scalar_lower_vertPos;
					double eLen = e.length();

					e.normalize();

					zVector newPos = scalar_lower_vertPos + (e * eLen * scaleVal);
					pos.push_back(newPos);

					// map edge to isographVertex
					edgetoIsoGraphVertexId[i] = pos.size() - 1;
					edgetoIsoGraphVertexId[i + 1] = pos.size() - 1;

				}
				
			}

			// compute edgeConnects
			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				
					vector<int> fEdges;
					fnMesh.getEdges(i, zFaceData, fEdges);
					vector<int> tempConnects;

					for (int j = 0; j < fEdges.size(); j++)
					{
						if (edgetoIsoGraphVertexId[fEdges[j]] != -1)
							tempConnects.push_back(edgetoIsoGraphVertexId[fEdges[j]]);
					}

					//printf("\n face %i | %i ", i, tempConnects.size());

					if (tempConnects.size() == 2)
					{
						edgeConnects.push_back(tempConnects[0]);
						edgeConnects.push_back(tempConnects[1]);
					}
				
			}

			zObjGraph out; 
			out.graph = zGraph(pos, edgeConnects);

			return out;



		}

		/*! \brief This method creates a isoline mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[in]	threshold	- field threshold.
		*	\param	[in]	invertMesh	- true if inverted mesh is required.
		*	\return			zMesh		- isoline mesh.
		*	\since version 0.0.2
		*/
		zObjMesh getIsolineMesh(double threshold = 0.5, bool invertMesh = false)
		{
			zObjMesh out;

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			unordered_map <string, int> positionVertex;



			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				getIsolinePoly(i, positions, polyConnects, polyCounts, positionVertex, threshold, invertMesh);
			}


			out.mesh = zMesh(positions, polyCounts, polyConnects);;

			return out;
		}


		/*! \brief This method creates a isoband mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[in]	thresholdLow	- field threshold domain minimum.
		*	\param	[in]	thresholdHigh	- field threshold domain maximum.
		*	\param	[in]	invertMesh		- true if inverted mesh is required.
		*	\return			zMesh			- isoband mesh.
		*	\since version 0.0.2
		*/
		zObjMesh getIsobandMesh(double thresholdLow = 0.2, double thresholdHigh = 0.5, bool invertMesh = false)
		{
			zObjMesh out;

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			unordered_map <string, int> positionVertex;


			if (invertMesh)
			{
				//zObjMesh m1 = getIsolineMesh((thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, false);
				//zObjMesh m2 = getIsolineMesh( (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow, true);

				///*if (m1.numVertices() > 0 && m2.numVertices() > 0)
				//{
				//	out = combineDisjointMesh(m1, m2);

				//	return out;
				//}

				//else*/ if (m1.numVertices() > 0) return m1;

				//else if (m2.numVertices() > 0) return m2;


			}

			if (!invertMesh)
			{
				for (int i = 0; i < fnMesh.numPolygons(); i++)
				{
					getIsobandPoly(i,  positions, polyConnects, polyCounts, positionVertex, (thresholdLow < thresholdHigh) ? thresholdLow : thresholdHigh, (thresholdLow < thresholdHigh) ? thresholdHigh : thresholdLow);
				}

				out.mesh = zMesh(positions, polyCounts, polyConnects);;

				return out;
			}


		}

		/** @}*/

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------
	protected:

		/*! \brief This method gets the isoline case based on the input vertex binary values.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
		*	\param	[in]	vertexBinary	- vertex binary values.
		*	\return			int				- case type.
		*	\since version 0.0.2
		*/
		int getIsolineCase(bool vertexBinary[4])
		{
			int out = -1;

			if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 0;

			if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 1;

			if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 2;

			if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && vertexBinary[3]) out = 3;

			if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 4;

			if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 5;

			if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 6;

			if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && vertexBinary[3]) out = 7;

			if (vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 8;

			if (!vertexBinary[0] && vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 9;

			if (vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 10;

			if (!vertexBinary[0] && !vertexBinary[1] && vertexBinary[2] && !vertexBinary[3]) out = 11;

			if (vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 12;

			if (!vertexBinary[0] && vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 13;

			if (vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 14;

			if (!vertexBinary[0] && !vertexBinary[1] && !vertexBinary[2] && !vertexBinary[3]) out = 15;

			return out;
		}

		/*! \brief This method gets the isoline case based on the input vertex ternary values.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares. The sequencing is reversed as CCW windings are required.
		*	\param	[in]	vertexTernary	- vertex ternary values.
		*	\return			int				- case type.
		*	\since version 0.0.2
		*/
		int getIsobandCase(int vertexTernary[4])
		{
			int out = -1;

			// No Contour
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 0;

			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 1;


			// Single Triangle

			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 2;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 3;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 4;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 5;


			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 6;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 7;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 8;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 9;

			// Single Trapezoid

			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 10;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 11;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 12;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 13;


			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 14;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 15;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 16;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 17;

			// Single Rectangle

			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 18;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 19;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 20;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 21;


			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 22;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 23;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 24;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 25;


			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 26;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 27;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 28;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 29;

			// Single Square

			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 30;

			// Single Pentagon

			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 31;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 32;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 33;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 34;


			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 35;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 36;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 37;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 38;

			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 39;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 40;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 41;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 42;

			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 43;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 44;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 45;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 46;

			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 47;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 48;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 49;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 0) out = 50;

			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 51;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 52;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 53;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 2) out = 54;

			// Single Hexagon

			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 55;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 56;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 57;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 58;

			if (vertexTernary[0] == 1 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 59;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 60;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 1) out = 61;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 62;

			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 63;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 64;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 65;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 66;


			// Saddles:  1 or 2 polygon

			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 67;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 68;

			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 69;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 70;

			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 71;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 72;

			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 1 && vertexTernary[3] == 2) out = 73;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 74;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 1) out = 75;
			if (vertexTernary[0] == 2 && vertexTernary[1] == 1 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 76;

			if (vertexTernary[0] == 2 && vertexTernary[1] == 0 && vertexTernary[2] == 1 && vertexTernary[3] == 0) out = 77;
			if (vertexTernary[0] == 1 && vertexTernary[1] == 0 && vertexTernary[2] == 2 && vertexTernary[3] == 0) out = 78;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 2 && vertexTernary[2] == 0 && vertexTernary[3] == 1) out = 79;
			if (vertexTernary[0] == 0 && vertexTernary[1] == 1 && vertexTernary[2] == 0 && vertexTernary[3] == 2) out = 80;

			return out;
		}

		/*! \brief This method return the contour position  given 2 input positions at the input field threshold.
		*
		*	\param	[in]	threshold		- field threshold.
		*	\param	[in]	vertex_lower	- lower threshold position.
		*	\param	[in]	vertex_higher	- higher threshold position.
		*	\param	[in]	thresholdLow	- field threshold domain minimum.
		*	\param	[in]	thresholdHigh	- field threshold domain maximum.
		*	\since version 0.0.2
		*/
		zVector getContourPosition(double &threshold, zVector& vertex_lower, zVector& vertex_higher, double& thresholdLow, double& thresholdHigh)
		{

			double scaleVal = coreUtils.ofMap(threshold, thresholdLow, thresholdHigh, 0.0, 1.0);

			zVector e = vertex_higher - vertex_lower;
			double edgeLen = e.length();
			e.normalize();

			return (vertex_lower + (e * edgeLen *scaleVal));
		}

		/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
		*
		*	\param	[in]	faceId			- input face index.

		*	\param	[in]	positions		- container of positions of the computed polygon.
		*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
		*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
		*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
		*	\param	[in]	threshold		- field threshold.
		*	\param	[in]	invertMesh	- true if inverted mesh is required.
		*	\since version 0.0.2
		*/
		void getIsolinePoly(int& faceId, vector<zVector> &positions, vector<int> &polyConnects, vector<int> &polyCounts, unordered_map <string, int> &positionVertex, double &threshold, bool invertMesh)
		{
			vector<int> fVerts;
			fnMesh.getVertices(faceId, zFaceData, fVerts);

			if (fVerts.size() != 4) return;

			// chk if all the face vertices are below the threshold
			bool vertexBinary[4];
			double averageScalar = 0;

			for (int j = 0; j < fVerts.size(); j++)
			{
				if (fnMesh.getVertexColor(fVerts[j]).r < threshold)
				{
					vertexBinary[j] = (invertMesh) ? false : true;
				}
				else vertexBinary[j] = (invertMesh) ? true : false;

				averageScalar += fnMesh.getVertexColor(fVerts[j]).r;
			}

			averageScalar /= fVerts.size();

			int MS_case = getIsolineCase(vertexBinary);

			vector<zVector> newPositions;
			vector<zVector> newPositions2;


			// CASE 0
			if (MS_case == 0)
			{
				for (int j = 0; j < fVerts.size(); j++)
				{
					newPositions.push_back(fnMesh.getVertexPosition(fVerts[j]));
				}

			}

			// CASE 1
			if (MS_case == 1)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
				double s0 = fnMesh.getVertexColor(fVerts[0]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
				double s1 = fnMesh.getVertexColor(fVerts[1]).r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));

				v1 = fnMesh.getVertexPosition(fVerts[3]);
				s1 = fnMesh.getVertexColor(fVerts[3]).r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);


			}

			// CASE 2
			if (MS_case == 2)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
				double s0 = fnMesh.getVertexColor(fVerts[0]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
				double s1 = fnMesh.getVertexColor(fVerts[1]).r;

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[2]);
				s0 = fnMesh.getVertexColor(fVerts[2]).r;
			
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));

			}

			// CASE 3
			if (MS_case == 3)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[3]);
				double s0 = fnMesh.getVertexColor(fVerts[3]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[0]);
				double s1 = fnMesh.getVertexColor(fVerts[0]).r;				

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[2]);
				s0 = fnMesh.getVertexColor(fVerts[2]).r;
				
				v1 = fnMesh.getVertexPosition(fVerts[1]);
				s1 = fnMesh.getVertexColor(fVerts[1]).r;				

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));


			}

			// CASE 4
			if (MS_case == 4)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));				

				zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
				double s0 = fnMesh.getVertexColor(fVerts[1]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[2]);
				double s1 = fnMesh.getVertexColor(fVerts[2]).r;		

				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[3]);
				s0 = fnMesh.getVertexColor(fVerts[3]).r;			

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));				

			}

			// CASE 5
			if (MS_case == 5)
			{
				// SADDLE CASE 

				int SaddleCase = 1/*(averageScalar < threshold) ? 0 : 1*/;
				if (invertMesh) SaddleCase = 0/*(averageScalar < threshold) ? 1 : 0*/;

				if (SaddleCase == 0)
				{
					// hex

					zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
					double s0 = fnMesh.getVertexColor(fVerts[1]).r;

					zVector v1 = fnMesh.getVertexPosition(fVerts[0]);
					double s1 = fnMesh.getVertexColor(fVerts[0]).r;					

					zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));
					
					v1 = fnMesh.getVertexPosition(fVerts[2]);
					s1 = fnMesh.getVertexColor(fVerts[2]).r;

					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					v0 = fnMesh.getVertexPosition(fVerts[3]);
					s0 = fnMesh.getVertexColor(fVerts[3]).r;

					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));
					
					v1 = fnMesh.getVertexPosition(fVerts[0]);
					s1 = fnMesh.getVertexColor(fVerts[0]).r;
					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

				}
				if (SaddleCase == 1)
				{
					// tri 1
					zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
					double s0 = fnMesh.getVertexColor(fVerts[1]).r;

					zVector v1 = fnMesh.getVertexPosition(fVerts[0]);
					double s1 = fnMesh.getVertexColor(fVerts[0]).r;

					zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));
					

					v1 = fnMesh.getVertexPosition(fVerts[2]);
					s1 = fnMesh.getVertexColor(fVerts[2]).r;					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					// tri 2
					v0 = fnMesh.getVertexPosition(fVerts[3]);
					s0 = fnMesh.getVertexColor(fVerts[3]).r;				
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions2.push_back(pos);

					newPositions2.push_back(fnMesh.getVertexPosition(fVerts[3]));

					v1 = fnMesh.getVertexPosition(fVerts[0]);
					s1 = fnMesh.getVertexColor(fVerts[0]).r;					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions2.push_back(pos);
				}


			}

			// CASE 6
			if (MS_case == 6)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
				double s0 = fnMesh.getVertexColor(fVerts[0]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
				double s1 = fnMesh.getVertexColor(fVerts[1]).r;
				
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[3]);
				s0 = fnMesh.getVertexColor(fVerts[3]).r;
				v1 = fnMesh.getVertexPosition(fVerts[2]);
				s1= fnMesh.getVertexColor(fVerts[2]).r;
				
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));

			}


			// CASE 7
			if (MS_case == 7)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[3]);
				double s0 = fnMesh.getVertexColor(fVerts[3]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[2]);
				double s1 = fnMesh.getVertexColor(fVerts[2]).r;
				
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[3]));

				v1 = fnMesh.getVertexPosition(fVerts[0]);
				s1 = fnMesh.getVertexColor(fVerts[0]).r;
				
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 8
			if (MS_case == 8)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				zVector v0 = fnMesh.getVertexPosition(fVerts[2]);
				double s0 = fnMesh.getVertexColor(fVerts[2]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[3]);
				double s1 = fnMesh.getVertexColor(fVerts[3]).r;
				
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[0]);
				s0 = fnMesh.getVertexColor(fVerts[0]).r;				
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 9
			if (MS_case == 9)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
				double s0 = fnMesh.getVertexColor(fVerts[1]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[0]);
				double s1 = fnMesh.getVertexColor(fVerts[0]).r;
				
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				v0 = fnMesh.getVertexPosition(fVerts[2]);
				s0 = fnMesh.getVertexColor(fVerts[2]).r;
				v1 = fnMesh.getVertexPosition(fVerts[3]);
				s1 = fnMesh.getVertexColor(fVerts[3]).r;

				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 10
			if (MS_case == 10)
			{
				int SaddleCase = 1 /*(averageScalar < threshold) ? 0 : 1*/;
				if (invertMesh) SaddleCase = 0 /*(averageScalar < threshold) ? 1 : 0*/;

				if (SaddleCase == 0)
				{
					// hex
					newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

					zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
					double s0 = fnMesh.getVertexColor(fVerts[0]).r;
					zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
					double s1 = fnMesh.getVertexColor(fVerts[1]).r;
				
					zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					v0 = fnMesh.getVertexPosition(fVerts[2]);
					s0 = fnMesh.getVertexColor(fVerts[2]).r;
					v1 = fnMesh.getVertexPosition(fVerts[1]);
					s1 = fnMesh.getVertexColor(fVerts[1]).r;
				
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

					v1 = fnMesh.getVertexPosition(fVerts[3]);
					s1 = fnMesh.getVertexColor(fVerts[3]).r;
					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					v0 = fnMesh.getVertexPosition(fVerts[0]);
					s0 = fnMesh.getVertexColor(fVerts[0]).r;
					v1 = fnMesh.getVertexPosition(fVerts[3]);
					s1 = fnMesh.getVertexColor(fVerts[3]).r;
					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);
				}

				if (SaddleCase == 1)
				{
					// tri 1

					newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

					zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
					double s0 = fnMesh.getVertexColor(fVerts[0]).r;
					zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
					double s1 = fnMesh.getVertexColor(fVerts[1]).r;
					
					zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					v1 = fnMesh.getVertexPosition(fVerts[3]);
					s1 = fnMesh.getVertexColor(fVerts[3]).r;
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions.push_back(pos);

					// tri 2

					v0 = fnMesh.getVertexPosition(fVerts[2]);
					s0 = fnMesh.getVertexColor(fVerts[2]).r;
					v1 = fnMesh.getVertexPosition(fVerts[1]);
					s1 = fnMesh.getVertexColor(fVerts[1]).r;
					
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions2.push_back(pos);

					newPositions2.push_back(fnMesh.getVertexPosition(fVerts[2]));

					v1 = fnMesh.getVertexPosition(fVerts[3]);
					s1 = fnMesh.getVertexColor(fVerts[3]).r;
					pos = (getContourPosition(threshold, v0, v1, s0, s1));
					newPositions2.push_back(pos);
				}


			}

			// CASE 11
			if (MS_case == 11)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[2]);
				double s0 = fnMesh.getVertexColor(fVerts[2]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
				double s1 = fnMesh.getVertexColor(fVerts[1]).r;
			
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);
	

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[2]));

				v1 = fnMesh.getVertexPosition(fVerts[3]);
				s1 = fnMesh.getVertexColor(fVerts[3]).r;
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			

			}

			// CASE 12
			if (MS_case == 12)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));

				zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
				double s0 = fnMesh.getVertexColor(fVerts[1]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[2]);
				double s1 = fnMesh.getVertexColor(fVerts[2]).r;
			
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v0 = fnMesh.getVertexPosition(fVerts[0]);
				s0 = fnMesh.getVertexColor(fVerts[0]).r;
				v1 = fnMesh.getVertexPosition(fVerts[3]);
				s1 = fnMesh.getVertexColor(fVerts[3]).r;
				
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 13
			if (MS_case == 13)
			{
				zVector v0 = fnMesh.getVertexPosition(fVerts[1]);
				double s0 = fnMesh.getVertexColor(fVerts[1]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[0]);
				double s1 = fnMesh.getVertexColor(fVerts[0]).r;
			
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				newPositions.push_back(fnMesh.getVertexPosition(fVerts[1]));

				v1 = fnMesh.getVertexPosition(fVerts[2]);
				s1 = fnMesh.getVertexColor(fVerts[2]).r;
				
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 14
			if (MS_case == 14)
			{
				newPositions.push_back(fnMesh.getVertexPosition(fVerts[0]));

				zVector v0 = fnMesh.getVertexPosition(fVerts[0]);
				double s0 = fnMesh.getVertexColor(fVerts[0]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[1]);
				double s1 = fnMesh.getVertexColor(fVerts[1]).r;
				
				zVector pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

				v1 = fnMesh.getVertexPosition(fVerts[3]);
				s1 = fnMesh.getVertexColor(fVerts[3]).r;
			
				pos = (getContourPosition(threshold, v0, v1, s0, s1));
				newPositions.push_back(pos);

			}

			// CASE 15
			if (MS_case == 15)
			{
				// No Veritices to be added 

			}







			// check for edge lengths


			if (newPositions.size() >= 3)
			{
				for (int i = 0; i < newPositions.size(); i++)
				{
					int next = (i + 1) % newPositions.size();

					if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
					{
						newPositions.erase(newPositions.begin() + i);
					}
				}
			}


			// compute poly 
			if (newPositions.size() >= 3)
			{
				for (int i = 0; i < newPositions.size(); i++)
				{
					zVector p0 = newPositions[i];
					int v0;

					bool vExists = vertexExists(positionVertex, p0, v0);

					if (!vExists)
					{
						v0 = positions.size();
						positions.push_back(p0);

						string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
						positionVertex[hashKey] = v0;
					}

					polyConnects.push_back(v0);
				}

				polyCounts.push_back(newPositions.size());
			}


			// only if there are 2 tris Case : 5,10

			// Edge Length Check

			if (newPositions2.size() >= 3)
			{
				for (int i = 0; i < newPositions2.size(); i++)
				{
					int next = (i + 1) % newPositions2.size();


					if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
					{
						newPositions2.erase(newPositions.begin() + i);
					}

				}
			}


			// compute poly 
			if (newPositions2.size() >= 3)
			{
				for (int i = 0; i < newPositions2.size(); i++)
				{
					zVector p0 = newPositions2[i];
					int v0;

					bool vExists = vertexExists(positionVertex, p0, v0);

					if (!vExists)
					{
						v0 = positions.size();
						positions.push_back(p0);

						string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
						positionVertex[hashKey] = v0;
					}

					polyConnects.push_back(v0);
				}

				polyCounts.push_back(newPositions2.size());
			}



		}

		/*! \brief This method gets the isoline polygon for the input mesh at the given input face index.
		*
		*	\param	[in]	faceId			- input face index.
		*	\param	[in]	positions		- container of positions of the computed polygon.
		*	\param	[in]	polyConnects	- container of polygon connectivity of the computed polygon.
		*	\param	[in]	polyCounts		- container of number of vertices in the computed polygon.
		*	\param	[in]	positionVertex	- map of position and vertices, to remove overlapping vertices.
		*	\param	[in]	thresholdLow	- field threshold domain minimum.
		*	\param	[in]	thresholdHigh	- field threshold domain maximum.
		*	\since version 0.0.2
		*/
		void getIsobandPoly(int& faceId, vector<zVector> &positions, vector<int> &polyConnects, vector<int> &polyCounts, unordered_map <string, int> &positionVertex, double &thresholdLow, double &thresholdHigh)
		{
			vector<int> fVerts;
			fnMesh.getVertices(faceId, zFaceData, fVerts);

			//printf("\n fVs: %i ", fVerts.size());

			if (fVerts.size() != 4) return;

			// chk if all the face vertices are below the threshold
			int vertexTernary[4];
			double averageScalar = 0;

			for (int j = 0; j < fVerts.size(); j++)
			{
				if (fnMesh.getVertexColor(fVerts[j]).r <= thresholdLow)
				{
					vertexTernary[j] = 0;
				}

				else if (fnMesh.getVertexColor(fVerts[j]).r >= thresholdHigh)
				{
					vertexTernary[j] = 2;
				}
				else vertexTernary[j] = 1;

				averageScalar += fnMesh.getVertexColor(fVerts[j]).r;
			}

			averageScalar /= fVerts.size();

			int MS_case = getIsobandCase(vertexTernary);

			vector<zVector> newPositions;
			vector<zVector> newPositions2;

			//printf("\n MS_case : %i ", MS_case);
			//if (MS_case < 72 || MS_case > 80) return;




			// No Contours CASE 0 & 1
			if (MS_case == 0 || MS_case == 1)
			{
				// NO Vertices to be added
			}

			// Single Triangle CASE 2 to 9 
			if (MS_case >= 2 && MS_case <= 9)
			{
				int startID = -1;
				double threshold = thresholdHigh;

				if (MS_case == 2 || MS_case == 6)startID = 0;
				if (MS_case == 3 || MS_case == 7)startID = 1;
				if (MS_case == 4 || MS_case == 8)startID = 2;
				if (MS_case == 5 || MS_case == 9)startID = 3;

				if (MS_case > 5) threshold = thresholdLow;

				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();


				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				double s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				

				zVector pos0 = (getContourPosition(threshold, v0, v1, s0, s1));

				zVector pos1 = fnMesh.getVertexPosition(fVerts[startID]);


				v1 = fnMesh.getVertexPosition(fVerts[nextID]); 
				s1 = fnMesh.getVertexColor(fVerts[nextID]).r;

				zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));


				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);

			}


			// Single Trapezoid CASE 10 to 17

			if (MS_case >= 10 && MS_case <= 17)
			{
				int startID = -1;

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case == 10 || MS_case == 14) startID = 0;
				if (MS_case == 11 || MS_case == 15) startID = 1;
				if (MS_case == 12 || MS_case == 16) startID = 2;
				if (MS_case == 13 || MS_case == 17) startID = 3;

				if (MS_case > 13)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;
				}

				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
			

				zVector pos0 = (getContourPosition(threshold0, v0, v1, s0, s1));

				zVector pos1 = (getContourPosition(threshold1, v0, v1, s0, s1));
				
				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				

				zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

				zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));


				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

			}


			// Single Rectangle CASE 18 to 29
			if (MS_case >= 18 && MS_case <= 25)
			{
				int startID = -1;
				double threshold = thresholdLow;
				if (MS_case > 21) threshold = thresholdHigh;

				if (MS_case == 18 || MS_case == 22) startID = 0;
				if (MS_case == 19 || MS_case == 23) startID = 1;
				if (MS_case == 20 || MS_case == 24) startID = 2;
				if (MS_case == 21 || MS_case == 25) startID = 3;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();

				int next_nextID = (nextID + 1) % fVerts.size();
				
				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);
				zVector pos1 = fnMesh.getVertexPosition(fVerts[nextID]);			

				zVector v0 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s0 = fnMesh.getVertexColor(fVerts[nextID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[next_nextID]).r;				

				zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));
				
				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;			

				zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);

			}

			// Single Rectangle CASE 26 to 29
			if (MS_case >= 26 && MS_case <= 29)
			{
				int startID = -1;

				if (MS_case == 26) startID = 0;
				if (MS_case == 27) startID = 1;
				if (MS_case == 28) startID = 2;
				if (MS_case == 29) startID = 3;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				double s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				
				zVector pos0 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
				zVector pos3 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[nextID]);
				s0 = fnMesh.getVertexColor(fVerts[nextID]).r;
				v1 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s1 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				

				zVector pos1 = (getContourPosition(thresholdLow, v0, v1, s0, s1));
				zVector pos2 = (getContourPosition(thresholdHigh, v0, v1, s0, s1));

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
			}

			// Single Square CASE 30
			if (MS_case == 30)
			{
				for (int j = 0; j < fVerts.size(); j++)
					newPositions.push_back(fnMesh.getVertexPosition(fVerts[j]));
			}

			// Single Pentagon CASE 31 to 38
			if (MS_case >= 31 && MS_case <= 38)
			{
				int startID = -1;

				double threshold = thresholdHigh;
				if (MS_case > 34)threshold = thresholdLow;

				if (MS_case == 31 || MS_case == 35) startID = 0;
				if (MS_case == 32 || MS_case == 36) startID = 1;
				if (MS_case == 33 || MS_case == 37) startID = 2;
				if (MS_case == 34 || MS_case == 38) startID = 3;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);
				zVector pos1 = fnMesh.getVertexPosition(fVerts[nextID]);	

				zVector v0 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s0 = fnMesh.getVertexColor(fVerts[nextID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				
				zVector pos2 = getContourPosition(threshold, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[prevID]);
				s0 = fnMesh.getVertexColor(fVerts[prevID]).r;
			
				zVector pos3 = (getContourPosition(threshold, v0, v1, s0, s1));

				zVector pos4 = fnMesh.getVertexPosition(fVerts[prevID]);;

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
			}

			// Single Pentagon CASE 39 to 46
			if (MS_case >= 39 && MS_case <= 46)
			{
				int startID = -1;

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case > 42)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;

				}

				if (MS_case == 39 || MS_case == 43) startID = 3;
				if (MS_case == 40 || MS_case == 44) startID = 2;
				if (MS_case == 41 || MS_case == 45) startID = 1;
				if (MS_case == 42 || MS_case == 46) startID = 0;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;

				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
				
				zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;

				zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));
				zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
			}

			// Single Pentagon CASE 47 to54
			if (MS_case >= 47 && MS_case <= 54)
			{
				int startID = -1;

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case > 50)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;

				}

				if (MS_case == 47 || MS_case == 51) startID = 3;
				if (MS_case == 48 || MS_case == 52) startID = 2;
				if (MS_case == 49 || MS_case == 53) startID = 1;
				if (MS_case == 50 || MS_case == 54) startID = 0;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
				
				zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				v1 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s1 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
			
				zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
				zVector pos3 = (getContourPosition(threshold0, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				
				zVector pos4 = getContourPosition(threshold0, v0, v1, s0, s1);

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
			}

			// Single Hexagon CASE 55 to 62
			if (MS_case >= 55 && MS_case <= 62)
			{
				int startID = -1;

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case > 58)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;

				}

				if (MS_case == 55 || MS_case == 59) startID = 0;
				if (MS_case == 56 || MS_case == 60) startID = 1;
				if (MS_case == 57 || MS_case == 61) startID = 2;
				if (MS_case == 58 || MS_case == 62) startID = 3;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);
				zVector pos1 = fnMesh.getVertexPosition(fVerts[nextID]);

				zVector v0 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s0 = fnMesh.getVertexColor(fVerts[nextID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[next_nextID]).r;

				zVector pos2 = getContourPosition(threshold1, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[prevID]);
				s0 = fnMesh.getVertexColor(fVerts[prevID]).r;
			
				zVector pos3 = (getContourPosition(threshold1, v0, v1, s0, s1));
				zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));

				v1 = fnMesh.getVertexPosition(fVerts[startID]);
				s1 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos5 = getContourPosition(threshold0, v0, v1, s0, s1);

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
			}

			// Single Hexagon CASE 63 to 66
			if (MS_case >= 63 && MS_case <= 66)
			{
				int startID = -1;


				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case % 2 == 0)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;
				}

				if (MS_case == 63 || MS_case == 64) startID = 0;
				if (MS_case == 65 || MS_case == 66) startID = 1;



				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
				
				zVector pos1 = getContourPosition(threshold0, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				
				zVector pos2 = (getContourPosition(threshold0, v0, v1, s0, s1));

				zVector pos3 = fnMesh.getVertexPosition(fVerts[next_nextID])	;

				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;

				zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);

				newPositions.push_back(pos0);
				newPositions.push_back(pos1);
				newPositions.push_back(pos2);
				newPositions.push_back(pos3);
				newPositions.push_back(pos4);
				newPositions.push_back(pos5);
			}

			// SADDLE CASE 67 to 68 : 8 Sides
			if (MS_case >= 67 && MS_case <= 68)
			{

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case % 2 == 0)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;
				}

				int SaddleCase = -1;;

				if (averageScalar < thresholdLow) SaddleCase = 0;
				else if (averageScalar > thresholdHigh) SaddleCase = 2;
				else SaddleCase = 1;

				int startID = 0;

				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
			
				zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
				zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
			
				zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));
				zVector pos3 = getContourPosition(threshold0, v0, v1, s0, s1);

				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				
				zVector pos4 = (getContourPosition(threshold0, v0, v1, s0, s1));
				zVector pos5 = (getContourPosition(threshold1, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos6 = getContourPosition(threshold1, v0, v1, s0, s1);
				zVector pos7 = (getContourPosition(threshold0, v0, v1, s0, s1));


				if (SaddleCase == 0)
				{
					// quad 1
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos2);
					newPositions.push_back(pos3);

					// quad 2
					newPositions2.push_back(pos4);
					newPositions2.push_back(pos5);
					newPositions2.push_back(pos6);
					newPositions2.push_back(pos7);
				}


				if (SaddleCase == 2)
				{
					// quad 1
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos6);
					newPositions.push_back(pos7);

					// quad 2
					newPositions2.push_back(pos2);
					newPositions2.push_back(pos3);
					newPositions2.push_back(pos4);
					newPositions2.push_back(pos5);
				}


				if (SaddleCase == 1)
				{
					// octagon
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos2);
					newPositions.push_back(pos3);

					newPositions.push_back(pos4);
					newPositions.push_back(pos5);
					newPositions.push_back(pos6);
					newPositions.push_back(pos7);
				}

			}

			// SADDLE CASE 69 to 72 : 6 Sides
			if (MS_case >= 69 && MS_case <= 72)
			{

				double threshold = thresholdLow;

				if (MS_case > 70)
				{
					threshold = thresholdHigh;
				}

				int SaddleCase = -1;;

				if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
				else SaddleCase = 0;

				int startID = (MS_case % 2 == 0) ? 1 : 0;

				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector pos0 = fnMesh.getVertexPosition(fVerts[startID]);

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;

				zVector pos1 = getContourPosition(threshold, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
				
				zVector pos2 = (getContourPosition(threshold, v0, v1, s0, s1));

				zVector pos3 = fnMesh.getVertexPosition(fVerts[next_nextID]);

				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				
				zVector pos4 = (getContourPosition(threshold, v0, v1, s0, s1));

				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos5 = getContourPosition(threshold, v0, v1, s0, s1);


				if (SaddleCase == 0)
				{
					// tri 1
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos5);

					// tri 2
					newPositions2.push_back(pos2);
					newPositions2.push_back(pos3);
					newPositions2.push_back(pos4);
				}


				if (SaddleCase == 1)
				{
					// hexagon
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos2);
					newPositions.push_back(pos3);
					newPositions.push_back(pos4);
					newPositions.push_back(pos5);

				}

			}

			// SADDLE CASE 73 to 80 : 7 Sides
			if (MS_case >= 73 && MS_case <= 80)
			{

				double threshold0 = thresholdLow;
				double threshold1 = thresholdHigh;

				if (MS_case > 76)
				{
					threshold0 = thresholdHigh;
					threshold1 = thresholdLow;
				}

				int SaddleCase = -1;;

				if (averageScalar > thresholdLow && averageScalar < thresholdHigh) SaddleCase = 1;
				else SaddleCase = 0;

				int startID = -1;

				if (MS_case == 73 || MS_case == 77) startID = 0;
				if (MS_case == 74 || MS_case == 78) startID = 2;
				if (MS_case == 75 || MS_case == 79) startID = 1;
				if (MS_case == 76 || MS_case == 80) startID = 3;


				int nextID = (startID + 1) % fVerts.size();
				int prevID = (startID - 1 + fVerts.size()) % fVerts.size();
				int next_nextID = (nextID + 1) % fVerts.size();

				zVector v0 = fnMesh.getVertexPosition(fVerts[startID]);
				double s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				zVector v1 = fnMesh.getVertexPosition(fVerts[nextID]);
				double s1 = fnMesh.getVertexColor(fVerts[nextID]).r;
			
				zVector pos0 = getContourPosition(threshold0, v0, v1, s0, s1);
				zVector pos1 = getContourPosition(threshold1, v0, v1, s0, s1);

				v0 = fnMesh.getVertexPosition(fVerts[next_nextID]);
				s0 = fnMesh.getVertexColor(fVerts[next_nextID]).r;
								
				zVector pos2 = (getContourPosition(threshold1, v0, v1, s0, s1));

				zVector pos3 = fnMesh.getVertexPosition(fVerts[next_nextID]);

				v1 = fnMesh.getVertexPosition(fVerts[prevID]);
				s1 = fnMesh.getVertexColor(fVerts[prevID]).r;
				
				zVector pos4 = (getContourPosition(threshold1, v0, v1, s0, s1));
				
				v0 = fnMesh.getVertexPosition(fVerts[startID]);
				s0 = fnMesh.getVertexColor(fVerts[startID]).r;
				
				zVector pos5 = getContourPosition(threshold1, v0, v1, s0, s1);
				zVector pos6 = getContourPosition(threshold0, v0, v1, s0, s1);

				if (SaddleCase == 0)
				{
					// quad 1
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos5);
					newPositions.push_back(pos6);

					// tri 2
					newPositions2.push_back(pos2);
					newPositions2.push_back(pos3);
					newPositions2.push_back(pos4);
				}


				if (SaddleCase == 1)
				{
					// heptagon
					newPositions.push_back(pos0);
					newPositions.push_back(pos1);
					newPositions.push_back(pos2);
					newPositions.push_back(pos3);
					newPositions.push_back(pos4);
					newPositions.push_back(pos5);
					newPositions.push_back(pos6);

				}

			}

			// check for edge lengths

			if (newPositions.size() >= 3)
			{
				for (int i = 0; i < newPositions.size(); i++)
				{
					int next = (i + 1) % newPositions.size();

					if (newPositions[i].distanceTo(newPositions[next]) < distanceTolerance)
					{
						newPositions.erase(newPositions.begin() + i);

					}
				}
			}


			// compute poly 
			if (newPositions.size() >= 3)
			{
				for (int i = 0; i < newPositions.size(); i++)
				{
					zVector p0 = newPositions[i];
					int v0;

					bool vExists = coreUtils.vertexExists(positionVertex, p0,3, v0);

					if (!vExists)
					{
						v0 = positions.size();
						positions.push_back(p0);

						string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
						positionVertex[hashKey] = v0;
					}

					polyConnects.push_back(v0);
				}

				polyCounts.push_back(newPositions.size());
			}


			// only if there are 2 tris Case : 5,10

			// Edge Length Check


			if (newPositions2.size() >= 3)
			{
				for (int i = 0; i < newPositions2.size(); i++)
				{
					int next = (i + 1) % newPositions2.size();


					if (newPositions2[i].distanceTo(newPositions2[next]) < distanceTolerance)
					{
						newPositions2.erase(newPositions.begin() + i);


					}

				}
			}


			// compute poly 
			if (newPositions2.size() >= 3)
			{
				for (int i = 0; i < newPositions2.size(); i++)
				{
					zVector p0 = newPositions2[i];
					int v0;

					bool vExists = coreUtils.vertexExists(positionVertex, p0,3, v0);

					if (!vExists)
					{
						v0 = positions.size();
						positions.push_back(p0);

						string hashKey = (to_string(p0.x) + "," + to_string(p0.y) + "," + to_string(p0.z));
						positionVertex[hashKey] = v0;
					}

					polyConnects.push_back(v0);
				}

				polyCounts.push_back(newPositions2.size());
			}

		}


	};
	

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	//--------------------------
	//---- TEMPLATE SPECIALIZATION DEFINITIONS 
	//--------------------------


	//---------------//

	//---- zVector specilization for normliseFieldValues
	template<>
	inline void zFnField2D<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}


	//---- double specilization for normliseFieldValues
	template<>
	inline void zFnField2D<double>::normliseValues(vector<double> &fieldValues)
	{
		double dMin, dMax;
		computeMinMaxOfScalars(fieldValues, dMin, dMax);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = dMax - fieldValues[i];

		computeMinMaxOfScalars(fieldValues,dMin, dMax);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = coreUtils.ofMap(fieldValues[i], dMin, dMax, -1.0, 1.0);
	}

	//---------------//

	//---- double specilization for fromBMP
	template<>
	inline void zFnField2D<double>::fromBMP(string infilename)
	{
		zUtilsBMP bmp(infilename.c_str());

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		int resX = bmp.bmp_info_header.width;
		int resY = bmp.bmp_info_header.height;

		if (resX == 0 || resY == 0) return;

		create(resX, resY, 1, 1);
		createFieldMesh();



		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{
				int faceId;
				getIndex(x, y, faceId);

				//printf("\n %i %i %i ", x, y, faceId);

				// blue
				double b = (double)bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 0] / 255;

				// green
				double g = (double)bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 1] / 255;

				// red
				double r = (double)bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 2] / 255;

				fnMesh.setFaceColor(faceId,  zColor(r, g, b, 1));

				// alpha
				if (channels == 4)
				{
					double a = (double)bmp.data[channels * (x * bmp.bmp_info_header.height + y) + 3] / 255;

					fnMesh.setFaceColor(faceId, zColor(r, g, b, a));
				}

				setFieldValue(r, faceId);

			}
		}
	}

	//---------------//

	//---- zVector specilization for createVectorFieldFromScalarField
	template<>
	inline void zFnField2D<zVector>::createVectorFromScalarField(zFnField2D<double> &fnScalarField)
	{
		zVector minBB, maxBB; 
		fnScalarField.getBoundingBox(minBB, maxBB);

		int n_X, n_Y; 
		fnScalarField.getResolution(n_X, n_Y);

		vector<zVector> gradients = fnScalarField.getGradients();
		
		create(minBB, maxBB, n_X, n_Y);
		setFieldValues(gradients);
	}

	//---------------//



#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}

