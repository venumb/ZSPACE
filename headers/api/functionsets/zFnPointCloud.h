#pragma once

#include<headers/api/object/zObjPointCloud.h>
#include<headers/api/functionsets/zFn.h>

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

	/*! \class zFnPointCloud
	*	\brief A point cloud function set.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	class zFnPointCloud : protected zFn
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to a mesh object  */
		zObjPointCloud *pointsObj;

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method imports a point cloud from an TXT file.
		*
		*	\param [in]		infilename			- input file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		inline void fromCSV( string infilename)
		{
			pointsObj->pCloud.vertexPositions.clear();

			ifstream myfile;
			myfile.open(infilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << infilename.c_str() << endl;
				return;

			}

			while (!myfile.eof())
			{
				string str;
				getline(myfile, str);

				vector<string> perlineData = pointsObj->pCloud.coreUtils.splitString(str, " ");

				if (perlineData.size() > 0)
				{
					// vertex
					if (perlineData[0] == "v")
					{
						if (perlineData.size() == 4)
						{
							zVector pos;
							pos.x = atof(perlineData[1].c_str());
							pos.y = atof(perlineData[2].c_str());
							pos.z = atof(perlineData[3].c_str());

							pointsObj->pCloud.vertexPositions.push_back(pos);
						}
						//printf("\n working vertex");
					}


				}
			}

			myfile.close();



			printf("\n inPositions: %i ", pointsObj->pCloud.vertexPositions.size());


		}

		/*! \brief This method exports the input point cloud to a TXT file format.
		*
		*	\param [in]		inPositions			- input container of position.
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.1
		*/
		inline void toCSV( string outfilename)
		{



			// output file
			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				return;

			}

			// vertex positions
			for (int i = 0; i < pointsObj->pCloud.vertexPositions.size(); i++)
			{

				myfile << "\n v " << pointsObj->pCloud.vertexPositions[i].x << " " << pointsObj->pCloud.vertexPositions[i].y << " " << pointsObj->pCloud.vertexPositions[i].z;

			}

			myfile.close();

			cout << endl << " TXT exported. File:   " << outfilename.c_str() << endl;
		}


	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zFnPointCloud()
		{
			fnType = zFnType::zPointsFn;
			pointsObj = nullptr;
	
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\since version 0.0.2
		*/
		zFnPointCloud(zObjPointCloud &_pointsObj)
		{
			pointsObj = &_pointsObj;
			fnType = zFnType::zPointsFn;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zFnPointCloud() {}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void from(string path, zFileTpye type, bool staticGeom = false) override
		{
			if (type == zCSV) fromCSV(path);
		}

		void to(string path, zFileTpye type) override
		{
			if (type == zCSV) toCSV(path);	
		}

		void clear() override
		{
			pointsObj->pCloud.vertexPositions.clear();
			pointsObj->pCloud.vertexColors.clear();
			pointsObj->pCloud.vertexWeights.clear();
			
		}


		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		
		/*! \brief This method creates a mesh from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\since version 0.0.1
		*/
		void create(vector<zVector>(&_positions))
		{
			pointsObj->pCloud = zPointCloud(_positions);

		}

		//--------------------------
		//---- APPEND METHODS
		//--------------------------

		/*! \brief This method adds the input point to the point cloud.
		*	\param		[in]	_position		- input position to be added.
		*	\since version 0.0.2
		*/
		void addPosition(zVector &_position)
		{
			pointsObj->pCloud.vertexPositions.push_back(_position);

			pointsObj->pCloud.vertexColors.push_back(zColor(1, 0, 0, 1));
			pointsObj->pCloud.vertexWeights.push_back(1.0);
		}

		
		/*! \brief This method adds the input point container to the point cloud.
		*	\param		[in]	_positions		- input container positions to be added.
		*	\since version 0.0.2
		*/
		void addPositions(vector<zVector> &_positions)
		{
			for (int i = 0; i < _positions.size(); i++) addPosition(_positions[i]);
		}

		//--------------------------
		//---- QUERY METHODS
		//--------------------------

		/*! \brief This method returns the number of points in the pointcloud.
		*	\return				number of points.
		*	\since version 0.0.2
		*/
		int numVertices()
		{
			return  pointsObj->pCloud.vertexPositions.size();
		}


		//--------------------------
		//--- SET METHODS 
		//--------------------------
		
		/*! \brief This method sets point color of the input point to the input color.
		*
		*	\param		[in]	index					- input point index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setColor(int index, zColor col)
		{
			if (index > numVertices()) throw std::invalid_argument(" error: index out of bounds.");

			pointsObj->pCloud.vertexColors[index] = col;
		}

		/*! \brief This method sets point color of all the point with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of points in the point cloud.
		*	\since version 0.0.2
		*/
		void setColors(vector<zColor>& col)
		{			
			if (col.size() != pointsObj->pCloud.vertexColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

			for (int i = 0; i < pointsObj->pCloud.vertexColors.size(); i++)
			{
				pointsObj->pCloud.vertexColors[i] = col[i];
			}
		}
				

		//--------------------------
		//--- GET METHODS 
		//--------------------------

		
		/*! \brief This method gets vertex position at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector					- vertex position.
		*	\since version 0.0.2
		*/
		zVector getVertexPosition(int index)
		{
			if (index > numVertices()) throw std::invalid_argument(" error: index out of bounds.");			

			return pointsObj->pCloud.vertexPositions[index];

		}

		/*! \brief This method gets pointer to the vertex position at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zVector*				- pointer to internal vertex position.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPosition(int index)
		{
			if (index > numVertices()) throw std::invalid_argument(" error: index out of bounds.");
		
			return &pointsObj->pCloud.vertexPositions[index];

		}

		/*! \brief This method gets vertex positions of all the vertices.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexPositions(vector<zVector>& pos)
		{
			pos = pointsObj->pCloud.vertexPositions;
		}

		/*! \brief This method gets pointer to the internal vertex positions container.
		*
		*	\return				zVector*					- pointer to internal vertex position container.
		*	\since version 0.0.2
		*/
		zVector* getRawVertexPositions()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &pointsObj->pCloud.vertexPositions[0];
		}

		/*! \brief This method gets vertex color at the input index.
		*
		*	\param		[in]	index					- input vertex index.
		*	\return				zColor					- vertex color.
		*	\since version 0.0.2
		*/
		zColor getVertexColor(int index)
		{
			if (index > numVertices()) throw std::invalid_argument(" error: index out of bounds.");
			

			return pointsObj->pCloud.vertexColors[index];

		}

		/*! \brief This method gets pointer to the vertex color at the input index.
		*
		*	\param		[in]	index				- input vertex index.
		*	\return				zColor*				- pointer to internal vertex color.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColor(int index)
		{
			if (index > numVertices()) throw std::invalid_argument(" error: index out of bounds.");
		

			return &pointsObj->pCloud.vertexColors[index];

		}

		/*! \brief This method gets vertex color of all the vertices.
		*
		*	\param		[out]	col				- color  contatiner.
		*	\since version 0.0.2
		*/
		void getVertexColors(vector<zColor>& col)
		{
			col = pointsObj->pCloud.vertexColors;
		}

		/*! \brief This method gets pointer to the internal vertex color container.
		*
		*	\return				zColor*					- pointer to internal vertex color container.
		*	\since version 0.0.2
		*/
		zColor* getRawVertexColors()
		{
			if (numVertices() == 0) throw std::invalid_argument(" error: null pointer.");

			return &pointsObj->pCloud.vertexColors[0];
		}


		//--------------------------
		//---- TRANSFORM METHODS OVERRIDES
		//--------------------------


		virtual void setTransform(zTransform &inTransform, bool decompose = true, bool updatePositions = true) override
		{
			if (updatePositions)
			{
				zTransformationMatrix to;
				to.setTransform(inTransform, decompose);

				zTransform transMat = pointsObj->transformationMatrix.getToMatrix(to);
				transformObject(transMat);

				pointsObj->transformationMatrix.setTransform(inTransform);

				// update pivot values of object transformation matrix
				zVector p = pointsObj->transformationMatrix.getPivot();
				p = p * transMat;
				setPivot(p);

			}
			else
			{
				pointsObj->transformationMatrix.setTransform(inTransform, decompose);

				zVector p = pointsObj->transformationMatrix.getO();
				setPivot(p);

			}

		}

		virtual void setScale(double3 &scale) override
		{
			// get  inverse pivot translations
			zTransform invScalemat = pointsObj->transformationMatrix.asInverseScaleTransformMatrix();

			// set scale values of object transformation matrix
			pointsObj->transformationMatrix.setScale(scale);

			// get new scale transformation matrix
			zTransform scaleMat = pointsObj->transformationMatrix.asScaleTransformMatrix();

			// compute total transformation
			zTransform transMat = invScalemat * scaleMat;

			// transform object
			transformObject(transMat);
		}

		virtual void setRotation(double3 &rotation, bool appendRotations = false) override
		{
			// get pivot translation and inverse pivot translations
			zTransform pivotTransMat = pointsObj->transformationMatrix.asPivotTranslationMatrix();
			zTransform invPivotTransMat = pointsObj->transformationMatrix.asInversePivotTranslationMatrix();

			// get plane to plane transformation
			zTransformationMatrix to = pointsObj->transformationMatrix;
			to.setRotation(rotation, appendRotations);
			zTransform toMat = pointsObj->transformationMatrix.getToMatrix(to);

			// compute total transformation
			zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

			// transform object
			transformObject(transMat);

			// set rotation values of object transformation matrix
			pointsObj->transformationMatrix.setRotation(rotation, appendRotations);;
		}

		virtual void setTranslation(zVector &translation, bool appendTranslations = false) override
		{
			// get vector as double3
			double3 t;
			translation.getComponents(t);

			// get pivot translation and inverse pivot translations
			zTransform pivotTransMat = pointsObj->transformationMatrix.asPivotTranslationMatrix();
			zTransform invPivotTransMat = pointsObj->transformationMatrix.asInversePivotTranslationMatrix();

			// get plane to plane transformation
			zTransformationMatrix to = pointsObj->transformationMatrix;
			to.setTranslation(t, appendTranslations);
			zTransform toMat = pointsObj->transformationMatrix.getToMatrix(to);

			// compute total transformation
			zTransform transMat = invPivotTransMat * toMat * pivotTransMat;

			// transform object
			transformObject(transMat);

			// set translation values of object transformation matrix
			pointsObj->transformationMatrix.setTranslation(t, appendTranslations);;

			// update pivot values of object transformation matrix
			zVector p = pointsObj->transformationMatrix.getPivot();
			p = p * transMat;
			setPivot(p);

		}

		virtual void setPivot(zVector &pivot) override
		{
			// get vector as double3
			double3 p;
			pivot.getComponents(p);

			// set pivot values of object transformation matrix
			pointsObj->transformationMatrix.setPivot(p);
		}

		virtual void getTransform(zTransform &transform) override
		{
			transform = pointsObj->transformationMatrix.asMatrix();
		}


	protected:

		//--------------------------
		//---- PROTECTED OVERRIDE METHODS
		//--------------------------	

		virtual void transformObject(zTransform &transform) override
		{

			if (numVertices() == 0) return;


			zVector* pos = getRawVertexPositions();

			for (int i = 0; i < numVertices(); i++)
			{

				zVector newPos = pos[i] * transform;
				pos[i] = newPos;
			}

		}
		
	};
}