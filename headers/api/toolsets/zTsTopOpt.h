#pragma once
#include <headers/api/functionsets/zFnMesh.h>

namespace zSpace
{
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptMaterial
	*	\brief A struct to hold material information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zTopOptMaterial
	{
		/*!	\brief Thickness  */
		double thickness; 

		/*!	\brief Young's Modulus  */
		double E;

		/*!	\brief Shear Modulus  */
		double G;

		/*!	\brief Poisson's Ratio  */
		double NU;

		/*!	\brief Mass Density  */
		double RHO;

		/*!	\brief Stress Limits in Tension  */
		double ST;

		/*!	\brief Stress Limits in Compression  */
		double SC;

		/*!	\brief Stress Limits in Shear  */
		double SS;
						
	};

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptLoads
	*	\brief A struct to hold load information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zTopOptLoads
	{
		/*!	\brief Magnitude  */
		double magnitude;

		/*!	\brief Load Directions  */
		zVector dir;

		/*!	\brief vertex indicies where the load is applied  */
		vector<int> indicies;		
	};

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop topology optimization with Altair Hypermesh.
	*  @{
	*/


	/*! \struct zTopOptPattern
	*	\brief A struct to hold pattern grouping information.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	struct zTopOptPattern
	{
		

		/*!	\brief type  */
		int type;

		/*!	\brief achor  */
		zVector anchor;

		/*!	\brief node1  */
		zVector node1;

		/*!	\brief node2  */
		zVector node2;		
	};

	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zToolsets
	*	\brief Collection of toolsets for applications.
	*  @{
	*/

	/** \addtogroup zTsStatics
	*	\brief tool sets for graphic statics.
	*  @{
	*/

	/** \addtogroup zTopOpt
	*	\brief tool sets for Interop topology optimization with Altair Hypermesh.
	*  @{
	*/
	
	/*! \class zTsTopOpt
	*	\brief A tool set class for interOp to Altair Hypermesh.
	*	\since version 0.0.2
	*/
		
	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/

	class zTsTopOpt
	{
	protected:

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to result Object  */
		zObjMesh *meshObj;

		/*!	\brief material  */
		zTopOptMaterial mat;

		/*!	\brief container of loads  */
		vector<zTopOptLoads> loads;

		/*!	\brief container of pattern groupings  */
		zTopOptPattern pattern;

		/*!	\brief container of booleans indicating if a face is design(true) or non design space(false)  */
		vector<bool> designSpace_Boolean;
							
		/*!	\brief container of booleans indicating if a vertex is SPC(true) or not(false)  */
		vector<bool> SPC_Boolean;

		//--------------------------
		//---- FACTORY METHODS
		//--------------------------

		/*! \brief This method exports zMesh to a JSON file format using JSON Modern Library.
		*
		*	\param [in]		outfilename			- output file name including the directory path and extension.
		*	\since version 0.0.2
		*/
		void toJSON(string outfilename)
		{
			zUtilsJsonTopOpt meshJSON;
			json j;

			// edges

			//faces
			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				vector<int> fVerts;
				fnMesh.getVertices(i, zFaceData, fVerts);

				meshJSON.faces.push_back(fVerts);				
			}

			// vertex Attributes
			zVector* vPos = fnMesh.getRawVertexPositions();
			zVector* vNorm = fnMesh.getRawVertexNormals();

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				vector<double> v_attrib;

				v_attrib.push_back(vPos[i].x);
				v_attrib.push_back(vPos[i].y);
				v_attrib.push_back(vPos[i].z);

				v_attrib.push_back(vNorm[i].x);
				v_attrib.push_back(vNorm[i].y);
				v_attrib.push_back(vNorm[i].z);

				meshJSON.vertexAttributes.push_back(v_attrib);
			}

			// face Attributes
			zVector* fNorm = fnMesh.getRawFaceNormals();

			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				vector<double> f_attrib;				

				f_attrib.push_back(fNorm[i].x);
				f_attrib.push_back(fNorm[i].y);
				f_attrib.push_back(fNorm[i].z);

				meshJSON.faceAttributes.push_back(f_attrib);
			}

			// SPC
			meshJSON.SPC = SPC_Boolean;

			// Design Space
			meshJSON.designSpace = designSpace_Boolean;

			// loads
			for (int i = 0; i < loads.size(); i++)
			{
				vector<double> load_attrib;

				load_attrib.push_back(loads[i].magnitude);

				load_attrib.push_back(loads[i].dir.x);
				load_attrib.push_back(loads[i].dir.y);
				load_attrib.push_back(loads[i].dir.z);

				meshJSON.loads.push_back(load_attrib);

				meshJSON.loadPoints.push_back(loads[i].indicies);
			}

			// PatternGrouping
			meshJSON.patternGrouping.push_back(pattern.type);
			meshJSON.patternGrouping.push_back(pattern.anchor.x);
			meshJSON.patternGrouping.push_back(pattern.anchor.y);
			meshJSON.patternGrouping.push_back(pattern.anchor.z);
			meshJSON.patternGrouping.push_back(pattern.node1.x);
			meshJSON.patternGrouping.push_back(pattern.node1.y);
			meshJSON.patternGrouping.push_back(pattern.node1.z);
			meshJSON.patternGrouping.push_back(pattern.node2.x);
			meshJSON.patternGrouping.push_back(pattern.node2.y);
			meshJSON.patternGrouping.push_back(pattern.node2.z);


			// Material
			meshJSON.material.push_back(mat.thickness);
			meshJSON.material.push_back(mat.E);
			meshJSON.material.push_back(mat.G);
			meshJSON.material.push_back(mat.NU);
			meshJSON.material.push_back(mat.RHO);
			meshJSON.material.push_back(mat.ST);
			meshJSON.material.push_back(mat.SC);
			meshJSON.material.push_back(mat.SS);


			// Json file 
			j["Edges"] = meshJSON.edges;
			j["Faces"] = meshJSON.faces;			
			j["VertexAttributes"] = meshJSON.vertexAttributes;
			j["FaceAttributes"] = meshJSON.faceAttributes;
			j["SPC"] = meshJSON.SPC;
			j["DesignSpace"] = meshJSON.designSpace;
			j["Loads"] = meshJSON.loads;
			j["LoadPoints"] = meshJSON.loadPoints;
			j["PatternGrouping"] = meshJSON.patternGrouping;
			j["Material"] = meshJSON.material;

			// EXPORT	

			ofstream myfile;
			myfile.open(outfilename.c_str());

			if (myfile.fail())
			{
				cout << " error in opening file  " << outfilename.c_str() << endl;
				return;
			}

			//myfile.precision(16);
			myfile << j.dump();
			myfile.close();
		}

	public:

		/*!	\brief color domain.  */
		zDomainColor elementColorDomain = zDomainColor(zColor(0.5, 0, 0.2, 1), zColor(0, 0.2, 0.5, 1));

		/*!	\brief result function set  */
		zFnMesh fnMesh;


		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsTopOpt()
		{
			
			meshObj = nullptr;
		}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsTopOpt(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;
			fnMesh = zFnMesh(_meshObj);

			SPC_Boolean.assign(fnMesh.numVertices(), false);

			designSpace_Boolean.assign(fnMesh.numPolygons(), false);

			fnMesh.setFaceColor(zColor(0, 1, 0.5, 1));
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsTopOpt() {}

			   
		//--------------------------
		//---- TO METHOD
		//--------------------------

		void to(string path, zFileTpye type)
		{
			if (type == zJSON)
			{
				toJSON(path);
				
			}
			else throw std::invalid_argument(" error: invalid zFileTpye type");
		}

		//--------------------------
		//---- CREATE
		//--------------------------

		void createFromFile(string path, zFileTpye type)
		{
			fnMesh.from(path, type, false);
		}

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		void setSinglePointConstraints(vector<int> &_SPC)
		{
			for (int i = 0; i < _SPC.size(); i++)
			{
				SPC_Boolean[_SPC[i]] = true;

				fnMesh.setVertexColor(_SPC[i], zColor());
			}
		}

		void setNonDesignSpace(vector<int> &_NonDesignSpace)
		{
			for (int i = 0; i < _NonDesignSpace.size(); i++)
			{
				designSpace_Boolean[_NonDesignSpace[i]] = false;

				fnMesh.setFaceColor(_NonDesignSpace[i], zColor(1,0,0.5,1));
			}
		}

		void setMaterial(zTopOptMaterial &material)
		{
			mat = material;
		}

		void setPatternGrouping(int type = 0, const zVector &anchor = zVector(), const zVector &n1 = zVector(), const zVector &n2 = zVector())
		{
			pattern.type = type;
			pattern.anchor = anchor;
			pattern.node1 = n1;
			pattern.node2 = n2;
		}

		//--------------------------
		//--- APPEND METHODS 
		//--------------------------

		void appendLoads(double _magnitude, zVector &_dir, vector<int>& vIndices)
		{
			zTopOptLoads load;
			load.magnitude = _magnitude;
			load.dir = _dir;
			load.indicies = vIndices;

		}

		//--------------------------
		//--- CLEAR METHODS 
		//--------------------------

	};

}