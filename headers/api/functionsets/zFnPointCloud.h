#pragma once

#include<headers/api/object/zObjPointCloud.h>
#include<headers/api/functionsets/zFn.h>

namespace zSpace
{
	/** \addtogroup zFuntionSets
	*	\brief The function set classes of the library.
	*  @{
	*/

	/*! \class zFnPointCloud
	*	\brief A point cloud function set.
	*	\since version 0.0.2
	*/

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
			pointsObj->pCloud.points.clear();

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

							pointsObj->pCloud.points.push_back(pos);
						}
						//printf("\n working vertex");
					}


				}
			}

			myfile.close();



			printf("\n inPositions: %i ", pointsObj->pCloud.points.size());


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
			for (int i = 0; i < pointsObj->pCloud.points.size(); i++)
			{

				myfile << "\n v " << pointsObj->pCloud.points[i].x << " " << pointsObj->pCloud.points[i].y << " " << pointsObj->pCloud.points[i].z;

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
			fnType = zFnType::zPointCloudFn;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_pointsObj			- input point cloud object.
		*	\since version 0.0.2
		*/
		zFnPointCloud(zObjPointCloud &_pointsObj)
		{
			pointsObj = &_pointsObj;
			fnType = zFnType::zPointCloudFn;
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

		void from(string path, zFileTpye type) override
		{
			if (type == zCSV) fromCSV(path);
		

		}

		void to(string path, zFileTpye type) override
		{
			if (type == zCSV) toCSV(path);	
		}

		void clear() override
		{
			pointsObj->pCloud.points.clear();
			pointsObj->pCloud.pointColors.clear();
			pointsObj->pCloud.pointWeights.clear();
			
		}


		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/** \addtogroup pointCloud creation
		*	\brief Collection of pointCloud creation methods.
		*  @{
		*/

		/*! \brief This method creates a mesh from the input containers.
		*
		*	\param		[in]	_positions		- container of type zVector containing position information of vertices.
		*	\since version 0.0.1
		*/
		void create(vector<zVector>(&_positions))
		{
			pointsObj->pCloud = zPointCloud(_positions);

		}


		void addPoint(zVector &_position)
		{
			pointsObj->pCloud.points.push_back(_position);

			pointsObj->pCloud.pointColors.push_back(zColor(1, 0, 0, 1));
			pointsObj->pCloud.pointWeights.push_back(1.0);
		}

		/** @}*/

		/*! \brief This method returns the number of points in the pointcloud.
		*	\return				number of points.
		*	\since version 0.0.2
		*/
		int numPoints()
		{
			return  pointsObj->pCloud.points.size();
		}


		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/** \addtogroup pointCloud set methods
		*	\brief Collection of pointCloud set attribute methods.
		*  @{
		*/

		/*! \brief This method sets point color of the input point to the input color.
		*
		*	\param		[in]	index					- input point index.
		*	\param		[in]	col						- input color.
		*	\since version 0.0.2
		*/
		void setColor(int index, zColor col)
		{
			pointsObj->pCloud.pointColors[index] = col;
		}

		/*! \brief This method sets point color of all the point with the input color contatiner.
		*
		*	\param		[in]	col				- input color  contatiner. The size of the contatiner should be equal to number of points in the point cloud.
		*	\since version 0.0.2
		*/
		void setColors(vector<zColor>& col)
		{			
			if (col.size() != pointsObj->pCloud.pointColors.size()) throw std::invalid_argument("size of color contatiner is not equal to number of mesh vertices.");

			for (int i = 0; i < pointsObj->pCloud.pointColors.size(); i++)
			{
				pointsObj->pCloud.pointColors[i] = col[i];
			}
		}

		/** @}*/

		//--------------------------
		//--- GET METHODS 
		//--------------------------

		/** \addtogroup pointCloud get methods
		*	\brief Collection of pointCloud get attribute methods.
		*  @{
		*/		

		/*! \brief This method gets point position at the input index.
		*
		*	\param		[in]	index					- input point index.	
		*	\since version 0.0.2
		*/
		zVector getPosition(int index)
		{
			if (index > pointsObj->pCloud.points.size()) throw std::invalid_argument(" error: index out of bounds.");

			return pointsObj->pCloud.points[index];
		}

		/*! \brief This method gets point positions of all the points in the point cloud.
		*
		*	\param		[out]	pos				- positions  contatiner.
		*	\since version 0.0.2
		*/
		void getPositions(vector<zVector>& pos)
		{
			pos =  pointsObj->pCloud.points;
		}

		/*! \brief This method gets point color at the input index.
		*
		*	\param		[in]	index					- input point index.
		*	\since version 0.0.2
		*/
		zColor getColor(int index)
		{
			if (index > pointsObj->pCloud.points.size()) throw std::invalid_argument(" error: index out of bounds.");

			return pointsObj->pCloud.pointColors[index];
		}

		/*! \brief This method gets point colors of all the points in the point cloud.
		*
		*	\param		[out]	col				- colors  contatiner.
		*	\since version 0.0.2
		*/
		void getColors(vector<zColor>& col)
		{
			col = pointsObj->pCloud.pointColors;
		}


		/** @}*/
	};
}