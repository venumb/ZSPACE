#pragma once

#include<headers/api/object/zObjGraph.h>
#include<headers/api/object/zObjMesh.h>
#include<headers/api/object/zObjMeshField.h>

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

	/*! \class zFnMeshField
	*	\brief A 2D field function set.
	*
	*	\tparam				T			- Type to work with double(scalar field) and zVector(vector field).
	*	\since version 0.0.2
	*/	

	/** @}*/

	/** @}*/

	template<typename T>
	class zFnMeshField 
	{
	private:

		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------	

		zFnType fnType;
			   		
		/*!	\brief contour value domain.  */
		zDomain<double> contourValueDomain;

		/*!	\brief container of field values used for contouring. All values to be in teh 0 to 1 domain  */
		vector<double> contourVertexValues;

	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to a field 2D object  */
		zObjMeshField<T> *fieldObj;		

		/*!	\brief boolean indicating if the field values size is equal to mesh vertices(true) or equal to mesh faces(false)  */
		bool setValuesperVertex = true;

		/*!	\brief boolean indicating if the field mesh is triangulated(true or quadragulated (false)  */
		bool triMesh = true;

		/*!	\brief field color domain.  */
		zDomainColor fieldColorDomain = zDomainColor(zColor(), zColor(1, 1, 1, 1));

	

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

					if (setValuesperVertex)
					{
						int vertexId;
						getIndex(x, y, vertexId);

						//printf("\n %i %i %i ", x, y, faceId);

						// blue
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0]  =  fnMesh.getVertexColor(vertexId).b * 255;;

						// green
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = fnMesh.getVertexColor(vertexId).g * 255;;

						// red
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2]  = fnMesh.getVertexColor(vertexId).r * 255;;

					
						// alpha
						if (channels == 4)
						{
							bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3]  = fnMesh.getVertexColor(vertexId).a * 255;;
							
						}
			
					}

					else
					{
						int faceId;
						getIndex(x, y, faceId);

						// blue
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = fnMesh.getFaceColor(faceId).b * 255;

						// green
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = fnMesh.getFaceColor(faceId).g * 255;

						// red
						bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = fnMesh.getFaceColor(faceId).r * 255;

						// alpha
						if (channels == 4)
						{
							bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = fnMesh.getFaceColor(faceId).a * 255;
						}
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
		
		/*! \brief This method creates the mesh from the field parameters.
		*	
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

			int resX = n_X;
			int resY = n_Y;

			if (!setValuesperVertex)
			{
				resX++;
				resY++;
			}

			getBoundingBox(minBB, maxBB);

			zVector unitVec = zVector(unit_X, unit_Y, 0);
			zVector startPt = minBB ;;
	
			if (!setValuesperVertex)startPt -= (unitVec * 0.5);

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

			/// poly connects

			for (int i = 0; i < resX - 1; i++)
			{
				for (int j = 0; j < resY - 1; j++)
				{
					int v0 = (i * resY) + j;
					int v1 = ((i + 1) * resY) + j;

					int v2 = v1 + 1;
					int v3 = v0 + 1;

					if (triMesh)
					{
						polyConnects.push_back(v0);
						polyConnects.push_back(v1);
						polyConnects.push_back(v3);
						polyCounts.push_back(3);

						polyConnects.push_back(v1);
						polyConnects.push_back(v2);
						polyConnects.push_back(v3);
						polyCounts.push_back(3);

					}
					else
					{ 
						polyConnects.push_back(v0);
						polyConnects.push_back(v1);
						polyConnects.push_back(v2);
						polyConnects.push_back(v3);

						polyCounts.push_back(4);
					}
					
				}
			}

			fnMesh.create(positions, polyCounts, polyConnects, true);		

			printf("\n fieldmesh: v %i e %i f %i", fnMesh.numVertices(), fnMesh.numEdges(), fnMesh.numPolygons());
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
		zFnMeshField() 
		{
			fnType = zFnType::zMeshFieldFn;
			fieldObj = nullptr;		
						
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_fieldObj			- input field2D object.
		*	\since version 0.0.2
		*/
		zFnMeshField(zObjMeshField<T> &_fieldObj)
		{
			fieldObj = &_fieldObj;		

			fnType = zFnType::zMeshFieldFn;
			fnMesh = zFnMesh (_fieldObj);
			
		}


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnMeshField() {}


		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method exports the field to the given file type.
		*
		*	\param	[in]		path						- output file name including the directory path and extension.
		*	\param	[in]		type						- type of file to be exported 
		*	\param	[in]		_setValuesperVertex			-  numver of field values aligns with mesh vertices if true else aligns with mesh faces.
		*	\param	[in]		_triMesh					- boolean true if triangulated mesh in needed. Works only when _setValuesperVertex is false.
		*	\since version 0.0.2
		*/
		void from(string path, zFileTpye type, bool _setValuesperVertex = true, bool _trimesh = true)
		{
			setValuesperVertex = _setValuesperVertex;

			if (!setValuesperVertex) _trimesh = false;
			triMesh = _trimesh;

			if (type == zBMP) fromBMP(path);
			
			else if (type == zOBJ) fnMesh.from(path, type, true);
			else if (type == zJSON) fnMesh.from(path, type, true);

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
			else if (type == zOBJ) fnMesh.to(path, type);
			else if (type == zJSON) fnMesh.to(path, type);

			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		/*! \brief This method clears the dynamic and array memory the object holds.
		*
		*	\since version 0.0.2
		*/
		void clear() 
		{
			ringNeighbours.clear();
			adjacentNeighbours.clear();			
			fieldObj->field.fieldValues.clear();			
			fnMesh.clear();
		}

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
		void create(zVector _minBB, zVector _maxBB, int _n_X, int _n_Y, int _NR = 1, bool _setValuesperVertex = true, bool _triMesh = true)
		{
			setValuesperVertex = _setValuesperVertex;
			if (!_setValuesperVertex) _triMesh = false;
			triMesh = _triMesh;


			fieldObj->field = zField2D<T>(_minBB, _maxBB, _n_X, _n_Y);
			fieldObj->field.valuesperVertex = setValuesperVertex;

			// compute neighbours
			ringNeighbours.clear();
			adjacentNeighbours.clear();

			
			ringNeighbours.assign(numFieldValues(), vector<int>());
			adjacentNeighbours.assign(numFieldValues(), vector<int>());

			for (int i = 0; i < numFieldValues(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbour_Ring(i, _NR, temp_ringNeighbour);
				ringNeighbours[i] = (temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbour_Adjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours[i] = (temp_adjacentNeighbour);
			}		

			createFieldMesh();

			updateColors();
			
		}

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
		void create(double _unit_X, double _unit_Y, int _n_X, int _n_Y, zVector _minBB = zVector(), int _NR = 1, bool _setValuesperVertex = true, bool _triMesh = true)
		{
			setValuesperVertex = _setValuesperVertex;
			if (!_setValuesperVertex) _triMesh = false;
			triMesh = _triMesh;

			fieldObj->field = zField2D<T>(_unit_X, _unit_Y, _n_X, _n_Y, _minBB);
			fieldObj->field.valuesperVertex = setValuesperVertex;

			// compute neighbours
			ringNeighbours.clear();
			adjacentNeighbours.clear();

			ringNeighbours.assign(numFieldValues(), vector<int>());
			adjacentNeighbours.assign(numFieldValues(), vector<int>());

			for (int i = 0; i < numFieldValues(); i++)
			{
				vector<int> temp_ringNeighbour;
				getNeighbour_Ring(i, _NR, temp_ringNeighbour);
				ringNeighbours[i] = (temp_ringNeighbour);

				vector<int> temp_adjacentNeighbour;
				getNeighbour_Adjacents(i, temp_adjacentNeighbour);
				adjacentNeighbours[i] = (temp_adjacentNeighbour);
			}

			createFieldMesh();

			updateColors();
		}

		/*! \brief This method creates a vector field from the input scalarfield.
		*	\param		[in]	scalarFieldObj		- input scalar field object.
		*	\since version 0.0.2
		*/
		void createVectorFromScalarField(zObjMeshField<double> &scalarFieldObj);
		
		
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
		void getNeighbour_Ring(int index, int numRings, vector<int> &ringNeighbours)
		{
			
			ringNeighbours.clear();
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


					if (newId < numFieldValues()) ringNeighbours.push_back(newId);
				}

			}

			
		}

		/*! \brief This method gets the ring Points  of the field at the input index.
		*
		*	\param		[in]	index			- input index.
		*	\param		[in]	numRings		- number of rings.
		*	\param		[out]	ringNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Ring(int index, int numRings, vector<zVector> &ringNeighbours)
		{
			ringNeighbours.clear();

			vector<int> rNeighbourIndex;
			getNeighbour_Ring(index, numRings, rNeighbourIndex);

			for (int i = 0;i < rNeighbourIndex.size(); i++)
			{
				ringNeighbours.push_back(getPosition(rNeighbourIndex[i]));
			}

		}

		/*! \brief This method gets the immediate adjacent neighbours of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of neighbour indicies.
		*	\since version 0.0.2
		*/
		void getNeighbour_Adjacents(int index, vector<int> &adjacentNeighbours)
		{
			adjacentNeighbours.clear();

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
						if (i == 0 || j == 0) adjacentNeighbours.push_back(newId);
					}
				}

			}			

		}

		/*! \brief This method gets the immediate adjacent Points  of the field at the input index.
		*
		*	\param		[in]	index				- input index.
		*	\param		[out]	adjacentNeighbours	- contatiner of adjacent points indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Adjacents(int index, vector<zVector> &adjacentNeighbours)
		{
			adjacentNeighbours.clear();

			vector<int> aNeighbourIndex;
			getNeighbour_Adjacents(index, aNeighbourIndex);

			for (int i = 0;i< aNeighbourIndex.size(); i++)
			{
				adjacentNeighbours.push_back(getPosition(aNeighbourIndex[i]));
			}

		}

		/*! \brief This method gets the field indices which contain the input position.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	containedGridPoints	- contatiner of contained points indicies.
		*	\since version 0.0.2
		*/
		void getNeighbour_Contained(zVector &pos, vector<int> &containedNeighbour)
		{			
			containedNeighbour.clear();

			int index;
			bool checkBounds = getIndex(pos, index);

			if (!checkBounds)
			{
				return;

				
			}

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

				zVector minBB_temp = getPosition(g1);
				if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				zVector maxBB_temp = getPosition(g3);
				if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

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

				zVector minBB_temp = getPosition(g1);
				if(!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);
				
				zVector maxBB_temp = getPosition(g3);
				if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

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

				zVector minBB_temp = getPosition(g1);
				if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				zVector maxBB_temp = getPosition(g3);
				if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

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

				zVector minBB_temp = getPosition(g1);
				if (!setValuesperVertex) minBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

				zVector maxBB_temp = getPosition(g3);
				if (!setValuesperVertex) maxBB_temp -= zVector(fieldObj->field.unit_X * 0.5, fieldObj->field.unit_Y * 0.5, 0);

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

		/*! \brief This method gets the field Points which contain the input position.
		*
		*	\param		[in]	pos					- input position.
		*	\param		[out]	containedGridPoints	- contatiner of contained points indicies.
		*	\since version 0.0.2
		*/
		void getNeighbourPosition_Contained(zVector &pos, vector<zVector> &containedNeighbour)
		{
			containedNeighbour.clear();

			vector<int> cNeighbourIndex;
			getNeighbour_Contained(pos, cNeighbourIndex);			

			for(int i =0; i< cNeighbourIndex.size(); i++)
			{
				containedNeighbour.push_back(getPosition(cNeighbourIndex[i]));
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
			
			if (setValuesperVertex)
			{
				zItMeshVertex v(*fieldObj, index);
				return v.getVertexPosition();
			}
			else
			{
				zItMeshFace f(*fieldObj, index);
				return f.getCenter();
			}
		}

		/*! \brief This method gets all the positions of the field.
		*
		*	\param		[out]	positions	- output positions container.
		*	\since version 0.0.2
		*/
		void getPositions(vector<zVector> &positions)
		{
			if (setValuesperVertex) 	fnMesh.getVertexPositions(positions);
			else fnMesh.getCenters(zFaceData, positions);
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

			return getPosition(index);
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

		/*! \brief This method gets the closest index of the field at the input position.
		*
		*	\param		[in]	pos			- input position.
		*	\param		[out]	index		- output field index.
		*	\return				bool		- true if index is in bounds.
		*	\since version 0.0.2
		*/
		bool getClosestIndex(zVector &pos, int &index)
		{
			int id;
			bool out = getIndex(pos, id);

			if (out)
			{
				vector<int> cNeighbourIndex;
				getNeighbour_Contained(pos, cNeighbourIndex);

				vector<zVector> cNeighbours;	

				for (int i = 0; i< cNeighbourIndex.size(); i++)
				{					
					cNeighbours.push_back(getPosition(cNeighbourIndex[i]));
				}

				
				int idOut = coreUtils.getClosest_PointCloud(pos, cNeighbours);

				index = cNeighbourIndex[idOut];

			}
			

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
		*	\param		[out]	fieldValue	- output field value.
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
				getNeighbour_Ring(index, 1, ringNeighbours);

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
				getNeighbour_Adjacents(index, adjNeighbours);

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
				getNeighbour_Contained(samplePos, containedNeighbours);

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

		/*! \brief This method gets all the values of the field.
		*
		*	\param		[out]	fieldValues			- container of field values.		
		*	\since version 0.0.2
		*/
		void getFieldValues(vector<T>& fieldValues)
		{
			fieldValues = fieldObj->field.fieldValues;
		}

		/*! \brief This method gets pointer to the internal field values container.
		*
		*	\return				T*					- pointer to internal field value container.
		*	\since version 0.0.2
		*/
		T* getRawFieldValues()
		{
			if (numFieldValues() == 0) throw std::invalid_argument(" error: null pointer.");

			return &fieldObj->field.fieldValues[0];
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


		/*! \brief This method gets the boolean indicating if the field values aligns with mesh vertices or faces.
		*
		*	\return				bool		- if true field values aligns with mesh vertices else aligns with mesh faces.
		*	\since version 0.0.2
		*/
		bool getValuesPerVertexBoolean()
		{
			return setValuesperVertex;
		}

		/*! \brief This method gets the boolean indicating if the mesh field is triagulated.
		*
		*	\return				bool		- triagulated if true. 
		*	\since version 0.0.2
		*/
		bool getTriMeshBoolean()
		{
			return triMesh;
		}
		
		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets the field color domain.
		*
		*	\param		[in]	colDomain		- input color domain.
		*	\since version 0.0.2
		*/
		void setFieldColorDomain(zDomainColor &colDomain)
		{
			fieldColorDomain = colDomain;
		}

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

		/*! \brief This method sets the value of the field at the input index.
		*
		*	\param		[in]	fValue		- input value.
		*	\param		[in]	index		- index in the fieldvalues container.
		*	\since version 0.0.2
		*/
		void setFieldValue(T fValue, int index , bool append = false)
		{
			if (index > numFieldValues()) throw std::invalid_argument(" error: index out of bounds.");
			
			if(!append) fieldObj->field.fieldValues[index] = fValue;
			else fieldObj->field.fieldValues[index] += fValue;
		}

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjMesh &inMeshObj, T meshValue, double influence, double power = 2.0, bool normalise = true)
		{
			fieldValues.clear();
			zFnMesh inFnMesh(inMeshObj);

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();


			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
			
			zFnGraph inFngraph(inGraphObj);

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFngraph.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFngraph.numVertices(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
			zFnPointCloud fnPoints(inPointsObj);

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = fnPoints.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < fnPoints.numVertices(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, zObjPointCloud &inPointsObj, vector<T> &values, vector<double>& influences, double power = 2.0, bool normalise = true)
		{
			fieldValues.clear();
			zFnPointCloud fnPoints(inPointsObj);

			if (fnPoints.numVertices() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
			if (fnPoints.numVertices() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = fnPoints.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < fnPoints.numVertices(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, vector<zVector> &inPositions, T value, double influence, double power = 2.0, bool normalise = true)
		{

			fieldValues.clear();
			

			zVector *meshPositions = fnMesh.getRawVertexPositions();
	

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inPositions.size(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
		*	\param	[in]	inPositions			- icontainer of positions for distance calculations.
		*	\param	[in]	values				- value to be propagated for each input position. Size of container should be equal to inPositions.
		*	\param	[in]	influences			- influence value of each input position. Size of container should be equal to inPositions.
		*	\param	[in]	power				- input power value used for weight calculation. Default value is 2.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getFieldValuesAsVertexDistance_IDW(vector<T> &fieldValues, vector<zVector> &inPositions, vector<T> &values, vector<double>& influences, double power = 2.0, bool normalise = true)
		{			
			if (inPositions.size() != values.size()) throw std::invalid_argument(" error: size of inPositions and values dont match.");
			if (inPositions.size() != influences.size()) throw std::invalid_argument(" error: size of inPositions and influences dont match.");

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				T d;
				double wSum = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inPositions.size(); j++)
				{
					double r = meshPositions[i].distanceTo(inPositions[j]);

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
		//----  2D SCALAR FIELD METHODS
		//--------------------------

		/*! \brief This method creates a vertex distance Field from the input  point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPositions			- input container of postions for distance calculations.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, vector<zVector> &inPositions, bool normalise = true)
		{
			scalars.clear();;

		

			vector<double> distVals;
			double dMin = 100000;
			double dMax = 0;;

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				distVals.push_back(10000);
			}

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				for (int j = 0; j < inPositions.size(); j++)
				{


					double dist = meshPositions[i].squareDistanceTo(inPositions[j]);

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
				scalars.push_back(val);
			}


			if (normalise)
			{
				normliseValues(scalars);
			}


		}

		/*! \brief This method creates a vertex distance Field from the input point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPositions			- input container of postions for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, vector<zVector> &inPositions,  double a, double b, bool normalise = true)
		{
			scalars.clear();;		

			zVector *meshPositions = fnMesh.getRawVertexPositions();		

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inPositions.size(); j++)
				{
					double r = meshPositions[i].squareDistanceTo(inPositions[j]);

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


		/*! \brief This method creates a vertex distance Field from the input  point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjPointCloud &inPointsObj,  bool normalise = true)
		{
			scalars.clear();;

			zFnPointCloud fnPoints(inPointsObj);

			vector<double> distVals;
			double dMin = 100000;
			double dMax = 0;;

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = fnPoints.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				distVals.push_back(10000);
			}

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				for (int j = 0; j < fnPoints.numVertices(); j++)
				{
				

					double dist = meshPositions[i].squareDistanceTo(inPositions[j]);

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
				scalars.push_back(val);
			}
					

			if (normalise)
			{
				normliseValues(scalars);
			}

			
		}

		/*! \brief This method creates a vertex distance Field from the input point cloud.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inPointsObj			- input point cloud object for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjPointCloud &inPointsObj, double a, double b, bool normalise = true)
		{
			scalars.clear();;

			zFnPointCloud fnPoints(inPointsObj);

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = fnPoints.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < fnPoints.numVertices(); j++)
				{
					double r = meshPositions[i].squareDistanceTo(inPositions[j]);

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


		/*! \brief This method creates a vertex distance Field from the input mesh vertex positions.
		*
		*	\param	[out]	scalars				- container for storing scalar values.
		*	\param	[in]	inMeshObj			- input mesh object for distance calculations.
		*	\param	[in]	a					- input variable for distance function.
		*	\param	[in]	b					- input variable for distance function.		
		*	\param	[in]	normalise			- true if the scalars need to mapped between -1 and 1. generally used for contouring.
		*	\since version 0.0.2
		*/
		void getScalarsAsVertexDistance(vector<double> &scalars, zObjMesh &inMeshObj, double a, double b,  bool normalise = true)
		{
			scalars.clear();

			zFnMesh inFnMesh(inMeshObj);
			
			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();


			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numVertices(); j++)
				{
					double r = meshPositions[i].squareDistanceTo(inPositions[j]);

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

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFnGraph.getRawVertexPositions();

			// update values from meta balls

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numVertices(); j++)
				{
					double r = meshPositions[i].squareDistanceTo(inPositions[j]);

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

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFnMesh.getRawVertexPositions();

			// update values from edge distance
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnMesh.numHalfEdges(); j+= 2)
				{

					int e0 = inFnMesh.getEndVertex(j);   
					int e1 = inFnMesh.getStartVertex(j);  

					zVector closestPt;

					double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], closestPt);


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
		void getScalarsAsEdgeDistance(vector<double> &scalars, zObjGraph &inGraphObj, double a, double b,  bool normalise = true)
		{
			scalars.clear();
			zFnGraph inFnGraph(inGraphObj);

			zVector *meshPositions = fnMesh.getRawVertexPositions();
			zVector *inPositions = inFnGraph.getRawVertexPositions();

			// update values from edge distance
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double d = 0.0;
				double tempDist = 10000;

				for (int j = 0; j < inFnGraph.numHalfEdges(); j+= 2)
				{

					int e0 = inFnGraph.getEndVertex(j);
					int e1 = inFnGraph.getStartVertex(j);					
					
					zVector closestPt;
					
					double r = coreUtils.minDist_Edge_Point(meshPositions[i], inPositions[e0], inPositions[e1], inFnGraph.getVertexPosition(e1), closestPt);


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
		void getScalars_Circle(vector<double> &scalars, zVector &cen, float r, double annularVal = 0, bool normalise = true)
		{
			scalars.clear();
			scalars.assign(fnMesh.numVertices(), 0.0);

			zVector *meshPositions = fnMesh.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				if (annularVal == 0) scalars[i] = getScalar_Circle(cen, meshPositions[i], r);
				else scalars[i] = abs(getScalar_Circle(cen, meshPositions[i], r) - annularVal);
			}

			if (normalise) normliseValues(scalars);
		}

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
		void getScalars_Line(vector<double> &scalars, zVector &v0, zVector &v1, double annularVal = 0, bool normalise = true)
		{
			scalars.clear();
			scalars.assign(fnMesh.numVertices(), 0.0);

			zVector *meshPositions = fnMesh.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				if (annularVal == 0) scalars[i] = getScalar_Line(meshPositions[i], v0, v1);
				else scalars[i] = abs(getScalar_Line(meshPositions[i], v0, v1) - annularVal);
			}

			if (normalise) normliseValues(scalars);
		}

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
		void getScalars_Square(vector<double> &scalars, zVector &dimensions, double annularVal = 0, bool normalise = true)
		{
			scalars.clear();
			scalars.assign(fnMesh.numVertices(), 0.0);

			zVector *meshPositions = fnMesh.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				zVector p = meshPositions[i];
				if (annularVal == 0) scalars[i] = getScalar_Square(p, dimensions);
				else scalars[i] = abs(getScalar_Square(p, dimensions) - annularVal);
			}

			if (normalise) normliseValues(scalars);
		}

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
		void getScalars_Trapezoid(vector<double> &scalars, double r1, double r2, double he, double annularVal = 0, bool normalise = true)
		{
			scalars.clear();
			scalars.assign(fnMesh.numVertices(), 0.0);

			zVector *meshPositions = fnMesh.getRawVertexPositions();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				zVector p = meshPositions[i];
				if (annularVal == 0) scalars[i] = getScalar_Trapezoid(p, r1, r2, he);
				else scalars[i] = abs(getScalar_Trapezoid(p, r1, r2, he) - annularVal);
			}

			if (normalise) normliseValues(scalars);
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

		/*! \brief This method computes the min and max scalar values at the given Scalars buffer.
		*
		*	\param	[in]	values		- container of values
		*	\param	[out]	domain		- stores the domain of values
		*	\since version 0.0.2
		*/
		void computeDomain(vector<T> &values, zDomain <T> &domain)
		{
			domain.min = coreUtils.zMin(values);
			domain.max = coreUtils.zMax(values);
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
		void smoothField(int numSmooth, double diffuseDamp = 1.0, zDiffusionType type = zAverage);
		
	

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
		*	\param	[in]	field			- input field mesh.
		*	\param	[in]	scalars				- vector of scalar values. Need to be equivalent to number of mesh vertices.
		*	\param	[in]	clipPlane			- input zPlane used for clipping.
		*	\since version 0.0.2
		*/
		void boolean_clipwithPlane(vector<double>& scalars, zMatrixd& clipPlane)
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
				

		//--------------------------
		//----  UPDATE METHODS
		//--------------------------

		/*! \brief This method updates the color values of the field mesh based on the field values. Gradient - Black to Red
		*
		*	\warning works only for scalar fields.
		*	\since version 0.0.2
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
		*/
		void getIsocontour(zObjGraph &coutourGraphObj, double inThreshold = 0.5)
		{
			if (contourVertexValues.size() == 0) return;
			if (contourVertexValues.size() != numFieldValues())
			{
				throw std::invalid_argument(" error: invalid contour condition. Call updateColors method. ");
				return;
			}

			double threshold = coreUtils.ofMap(inThreshold, 0.0, 1.0, contourValueDomain.min, contourValueDomain.max);

			vector<zVector> pos;
			vector<int> edgeConnects;

			vector<int> edgetoIsoGraphVertexId;

			// compute positions
			for (int i = 0; i < fnMesh.numHalfEdges(); i += 2)
			{
				edgetoIsoGraphVertexId.push_back(-1);
				edgetoIsoGraphVertexId.push_back(-1);

				
				int eV0 = fnMesh.getEndVertexIndex(i);
				int eV1 = fnMesh.getStartVertexIndex(i);
						
					

				double scalar_lower = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV0] : contourVertexValues[eV1];
				double scalar_higher = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? contourVertexValues[eV1] : contourVertexValues[eV0];;

				bool chkSplitEdge = (scalar_lower <= threshold && scalar_higher > threshold) ? true : false;

				if (chkSplitEdge)
				{
					// calculate split point

					int scalar_lower_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV0 : eV1;
					int scalar_higher_vertId = (contourVertexValues[eV0] <= contourVertexValues[eV1]) ? eV1 : eV0;

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

			zFnGraph tempFn(coutourGraphObj);
			tempFn.clear(); // clear memory if the mobject exists.

		
			tempFn.create(pos, edgeConnects);
		}

		/*! \brief This method creates a isoline mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[out]	coutourMeshObj	- isoline mesh.
		*	\param	[in]	inThreshold		- field threshold.
		*	\param	[in]	invertMesh		- true if inverted mesh is required.
		*	\return			zMesh			- isoline mesh.
		*	\since version 0.0.2
		*/
		void getIsolineMesh(zObjMesh &coutourMeshObj, double inThreshold = 0.5, bool invertMesh = false)
		{
			if (contourVertexValues.size() == 0) return;
			if (contourVertexValues.size() != numFieldValues())
			{
				throw std::invalid_argument(" error: invalid contour condition.  Call updateColors method.");
				return;
			}

			zFnMesh tempFn(coutourMeshObj);
			tempFn.clear(); // clear memory if the mobject exists.

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			unordered_map <string, int> positionVertex;

			double threshold = coreUtils.ofMap(inThreshold, 0.0, 1.0, contourValueDomain.min, contourValueDomain.max);


			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				getIsolinePoly(i, positions, polyConnects, polyCounts, positionVertex, threshold, invertMesh);
			}


			tempFn.create(positions, polyCounts, polyConnects);;			
		}


		/*! \brief This method creates a isoband mesh from the input field mesh at the given field threshold.
		*
		*	\details based on https://en.wikipedia.org/wiki/Marching_squares.
		*	\param	[out]	coutourMeshObj	- isoband mesh.
		*	\param	[in]	inThresholdLow	- field threshold domain minimum.
		*	\param	[in]	inThresholdHigh	- field threshold domain maximum.
		*	\param	[in]	invertMesh		- true if inverted mesh is required.		
		*	\since version 0.0.2
		*/
		void getIsobandMesh(zObjMesh &coutourMeshObj,double inThresholdLow = 0.2, double inThresholdHigh = 0.5, bool invertMesh = false)
		{
			if (contourVertexValues.size() == 0) return;
			
			if (contourVertexValues.size() != numFieldValues())
			{
				throw std::invalid_argument(" error: invalid contour condition.  Call updateColors method.");
				return;
			}

			zFnMesh tempFn(coutourMeshObj);
			tempFn.clear(); // clear memory if the mobject exists.

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			unordered_map <string, int> positionVertex;

			double thresholdLow = coreUtils.ofMap(inThresholdLow, 0.0, 1.0, contourValueDomain.min, contourValueDomain.max);
			double thresholdHigh = coreUtils.ofMap(inThresholdHigh, 0.0, 1.0, contourValueDomain.min, contourValueDomain.max);

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

				tempFn.create(positions, polyCounts, polyConnects);;

				
			}


		}

		

		//--------------------------
		//---- PROTECTED METHODS
		//--------------------------
	protected:

		/*! \brief This method gets the scalar for the input point.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[in]	cen				- centre of the circle.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	r				- radius value.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		double getScalar_Circle(zVector &cen, zVector &p, float r)
		{
			return ((p - cen).length() - r);
		}

		/*! \brief This method gets the scalar for the input line.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	v0				- input positions 0 of line.
		*	\param	[in]	v1				- input positions 1 of line.
		*	\return			double			- scalar value.
		*	\since version 0.0.2
		*/
		double getScalar_Line(zVector &p, zVector &v0, zVector &v1)
		{
			zVector pa = p - v0;
			zVector ba = v1 - v0;

			float h = coreUtils.ofClamp((pa* ba) / (ba* ba), 0.0, 1.0);

			zVector out = pa - (ba*h);


			return (out.length());
		}

		/*! \brief This method gets the sqaure scalar for the input point.
		*
		*	\detail based on https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm.
		*	\param	[out]	scalars			- container for storing scalar values.
		*	\param	[in]	p				- input field point.
		*	\param	[in]	dimention		- input distance.
		*	\since version 0.0.2
		*/
		double getScalar_Square(zVector &p, zVector &dimensions)
		{
			p.x = abs(p.x); p.y = abs(p.y); p.z = abs(p.z);

			zVector out;
			out.x = coreUtils.zMax<double>(coreUtils.zMax<double>(p.x - dimensions.x, 0), coreUtils.zMax<double>(p.y - dimensions.y, 0));
			out.y = 0;
			out.z = 0;

			return(out.length());
		}

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
		double getScalar_Trapezoid(zVector &p, double &r1, double &r2, double &he)
		{
			zVector k1 = zVector(r2, he, 0);
			zVector k2 = zVector((r2 - r1), (2.0 * he), 0);

			p.x = abs(p.x);
			zVector ca = zVector(p.x - coreUtils.zMin(p.x, (p.y < 0.0) ? r1 : r2), abs(p.y) - he, 0.0);
			zVector cb = p - k1 + k2 * coreUtils.ofClamp(((k1 - p) * k2) / (k2*k2), 0.0, 1.0);

			double s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;

			double out = s * sqrt(coreUtils.zMin((ca * ca), (cb * cb)));

			return out;
		}


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
				if (contourVertexValues[fVerts[j]] < threshold)
				{
					vertexBinary[j] = (invertMesh) ? false : true;
				}
				else vertexBinary[j] = (invertMesh) ? true : false;

				averageScalar += contourVertexValues[fVerts[j]];
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
				s1 = fnMesh.getVertexColor(fVerts[2]).r;

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
				if (contourVertexValues[fVerts[j]] <= thresholdLow)
				{
					vertexTernary[j] = 0;
				}

				else if (contourVertexValues[fVerts[j]] >= thresholdHigh)
				{
					vertexTernary[j] = 2;
				}
				else vertexTernary[j] = 1;

				averageScalar += contourVertexValues[fVerts[j]];
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

				zVector pos3 = fnMesh.getVertexPosition(fVerts[next_nextID]);

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

					bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

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

					bool vExists = coreUtils.vertexExists(positionVertex, p0, 3, v0);

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
	
	//---- double specilization for updateColors
	template<>
	inline void zFnMeshField<double>::updateColors()
	{
		
		vector<double> scalars;
		getFieldValues(scalars);

		if (fnMesh.numVertices() == scalars.size() || fnMesh.numPolygons() == scalars.size())
		{
			
			computeDomain(scalars, contourValueDomain);			

			//convert to HSV

			if (contourValueDomain.min == contourValueDomain.max)
			{
				zColor col(0.5,0.5,0.5);

				if (fnMesh.numVertices() == scalars.size()) fnMesh.setVertexColor(col);
				else fnMesh.setFaceColor( col);

				if (fnMesh.numPolygons() == scalars.size())
				{
					contourVertexValues.clear();

					fnMesh.computeVertexColorfromFaceColor();

					for (zItMeshVertex v(*fieldObj); !v.end(); v.next())					
					{
						vector<int> cFaces;
						v.getConnectedFaces(cFaces);

						double val;

						for (int j = 0; j < cFaces.size(); j++)
						{
							val += scalars[cFaces[j]];
						}

						val /= cFaces.size();

						contourVertexValues.push_back(val);
					}

					computeDomain(contourVertexValues, contourValueDomain);

				}

				if (fnMesh.numVertices() == scalars.size())
				{
					contourVertexValues = scalars;

					fnMesh.computeFaceColorfromVertexColor();
				}



				return;
			}
			
			else
			{
				fieldColorDomain.min.toHSV(); fieldColorDomain.max.toHSV();

				zColor* cols = fnMesh.getRawVertexColors();				
				if (fnMesh.numPolygons() == scalars.size()) cols = fnMesh.getRawFaceColors();

				for (int i = 0; i < scalars.size(); i++)
				{				

					if (scalars[i] < contourValueDomain.min) cols[i] = fieldColorDomain.min;
					else if (scalars[i] > contourValueDomain.max) cols[i] = fieldColorDomain.max;
					else
					{
						cols[i] = coreUtils.blendColor(scalars[i], contourValueDomain, fieldColorDomain, zHSV);
					}
									
				}

				if (fnMesh.numPolygons() == scalars.size())
				{
					contourVertexValues.clear();

					fnMesh.computeVertexColorfromFaceColor();

					for (zItMeshVertex v(*fieldObj); !v.end(); v.next())
					{
						vector<int> cFaces;
						v.getConnectedFaces(cFaces);

						double val;

						for (int j = 0; j < cFaces.size(); j++)
						{
							val += scalars[cFaces[j]];
						}

						val /= cFaces.size();

						contourVertexValues.push_back(val);
					}

					computeDomain(contourVertexValues, contourValueDomain);

				}

				if (fnMesh.numVertices() == scalars.size())
				{
					contourVertexValues = scalars;

					fnMesh.computeFaceColorfromVertexColor();
				}
			}		
			

			

		}

	}

	//---------------//	


	//---- double specilization for smoothField
	template<>
	inline void zFnMeshField<double>::smoothField(int numSmooth, double diffuseDamp, zDiffusionType type)
	{
		for (int k = 0; k < numSmooth; k++)
		{
			vector<double> tempValues;

			for (int i = 0; i < numFieldValues(); i++)
			{
				double lapA = 0;

				vector<int> ringNeigbours;
				getNeighbour_Ring(i, 1, ringNeigbours);

				for (int j = 0; j < ringNeigbours.size(); j++)
				{
					int id = ringNeigbours[j];
					double val;
					getFieldValue(id, val);

					if (type == zLaplacian)
					{
						if (id != i) lapA += (val * 1);
						else lapA += (val * -8);
					}
					else if (type == zAverage)
					{
						lapA += (val * 1);
					}
				}



				if (type == zLaplacian)
				{
					double val1;
					getFieldValue(i, val1);

					double newA = val1 + (lapA * diffuseDamp);
					tempValues.push_back(newA);
				}
				else if (type == zAverage)
				{
					if (lapA != 0) lapA /= (ringNeigbours.size());

					tempValues.push_back(lapA);
				}

			}

			for (int i = 0; i < numFieldValues(); i++)
			{
				setFieldValue(tempValues[i], i);
			}

		}

		updateColors();

	}

	//---------------//

	//---- zVector specilization for normliseFieldValues
	template<>
	inline void zFnMeshField<zVector>::normliseValues(vector<zVector> &fieldValues)
	{
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i].normalize();
	}


	//---- double specilization for normliseFieldValues
	template<>
	inline void zFnMeshField<double>::normliseValues(vector<double> &fieldValues)
	{
		zDomainDouble d;
		computeDomain(fieldValues, d);

		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = d.max - fieldValues[i];

		computeDomain(fieldValues, d);

		zDomainDouble out(-1.0,1.0);
		for (int i = 0; i < fieldValues.size(); i++) fieldValues[i] = coreUtils.ofMap(fieldValues[i], d, out);
	}

	//---------------//

	//---- double specilization for fromBMP
	template<>
	inline void zFnMeshField<double>::fromBMP(string infilename)
	{
		zUtilsBMP bmp(infilename.c_str());

		uint32_t channels = bmp.bmp_info_header.bit_count / 8;

		int resX = bmp.bmp_info_header.width;
		int resY = bmp.bmp_info_header.height;

		if (resX == 0 || resY == 0) return;

		
		if (numFieldValues() != resX * resY) create(1, 1, resX, resY,zVector(),1,setValuesperVertex);
		

		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{

				double r = (double)bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] / 255;
				
				int id;
				getIndex(x, y, id);
				
				setFieldValue(r, id);

			}
		}
		
		updateColors();
	

	}

	//---------------//

	//---- zVector specilization for createVectorFieldFromScalarField
	template<>
	inline void zFnMeshField<zVector>::createVectorFromScalarField(zObjMeshField<double> &scalarFieldObj)
	{
		zFnMeshField<double> fnScalarField(scalarFieldObj);
		
		zVector minBB, maxBB; 
		fnScalarField.getBoundingBox(minBB, maxBB);

		int n_X, n_Y; 
		fnScalarField.getResolution(n_X, n_Y);

		vector<zVector> gradients = fnScalarField.getGradients();
		
		create(minBB, maxBB, n_X, n_Y, fnScalarField.getValuesPerVertexBoolean(), fnScalarField.getTriMeshBoolean());
		setFieldValues(gradients);
	}

	//---------------//
	
	//---- double specilization for setFieldValues
	template<>
	inline void zFnMeshField<double>::setFieldValues(vector<double> &fValues)
	{
		if (setValuesperVertex)
		{
			if (fnMesh.numPolygons() == fValues.size())
			{
				for (zItMeshVertex v(*fieldObj); !v.end(); v.next())
				{
					vector<int> cFaces;
					v.getConnectedFaces(cFaces);
					double val;

					for (int j = 0; j < cFaces.size(); j++)
					{
						val += fValues[cFaces[j]];
					}

					val /= cFaces.size();

					setFieldValue(val, v.getId());
				}

			}
			else if (fnMesh.numVertices() == fValues.size())
			{
				for (int i = 0; i < fnMesh.numVertices(); i++)
				{
					setFieldValue(fValues[i], i);
				}

			}

			else throw std::invalid_argument("input fValues size not equal to number of vertices/ polygons of field mesh.");
		}

		else
		{
			if (fnMesh.numVertices() == fValues.size())
			{
				for (zItMeshFace f(*fieldObj); !f.end(); f.next())
				{
					vector<int> fVerts;
					f.getVertices(fVerts);
					double val;

					for (int j = 0; j < fVerts.size(); j++)
					{
						val += fValues[fVerts[j]];
					}

					val /= fVerts.size();

					setFieldValue(val, f.getId());
				}

			}
			else if (fnMesh.numPolygons() == fValues.size())
			{
				for (int i = 0; i < fnMesh.numPolygons(); i++)
				{
					setFieldValue(fValues[i], i);
				}

			}

			else throw std::invalid_argument("input fValues size not equal to number of vertices/ polygons of field mesh.");
		}


		updateColors();
	}


	//---- zVector specilization for setFieldValues
	template<>
	inline void zFnMeshField<zVector>::setFieldValues(vector<zVector> &fValues)
	{
		if (setValuesperVertex)
		{
			if (fnMesh.numPolygons() == fValues.size())
			{
				for (zItMeshVertex v(*fieldObj); !v.end(); v.next())
				{
					vector<int> cFaces;
					v.getConnectedFaces(cFaces);
					zVector val;

					for (int j = 0; j < cFaces.size(); j++)
					{
						val += fValues[cFaces[j]];
					}

					val /= cFaces.size();

					setFieldValue(val, v.getId());
				}

			}
			else if (fnMesh.numVertices() == fValues.size())
			{
				for (int i = 0; i < fnMesh.numVertices(); i++)
				{
					setFieldValue(fValues[i], i);
				}

			}

			else throw std::invalid_argument("input fValues size not equal to number of vertices/ polygons of field mesh.");
		}

		else
		{
			if (fnMesh.numVertices() == fValues.size())
			{
				for (zItMeshFace f(*fieldObj); !f.end(); f.next())
				{
					vector<int> fVerts;
					f.getVertices(fVerts);
					zVector val;

					for (int j = 0; j < fVerts.size(); j++)
					{
						val += fValues[fVerts[j]];
					}

					val /= fVerts.size();

					setFieldValue(val, f.getId());
				}

			}
			else if (fnMesh.numPolygons() == fValues.size())
			{
				for (int i = 0; i < fnMesh.numPolygons(); i++)
				{
					setFieldValue(fValues[i], i);
				}

			}

			else throw std::invalid_argument("input fValues size not equal to number of vertices/ polygons of field mesh.");
		}


		
	}

	//---------------//



#endif /* DOXYGEN_SHOULD_SKIP_THIS */

}

