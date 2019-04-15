#pragma once


#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>
#include <headers/api/functionsets/zFnParticle.h>

namespace zSpace
{	
	/** \addtogroup zToolsets
	*	\brief Collection of tool sets for applications. 
	*  @{
	*/

	/*! \class zTsPolytopal
	*	\brief A toolset for 3D graphics and poytopal meshes.
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf) and Freeform Developable Spatial Structures (https://www.ingentaconnect.com/content/iass/piass/2016/00002016/00000003/art00010 )
	*	\since version 0.0.2
	*/

	/** @}*/

	class zTsPolytopal
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- FORCE DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief pointer container to force Object  */
		vector<zObjMesh*> forceObjs;

		//--------------------------
		//---- FORM DIAGRAM ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to form Object  */
		zObjGraph *formObj;

		/*!	\brief container of  form particle objects  */
		vector<zObjParticle> formParticlesObj;

		/*!	\brief container of form particle function set  */
		vector<zFnParticle> fnFormParticles;

		/*!	\brief container storing the target for form edges.  */
		vector<zVector> targetEdges_form;

		//--------------------------
		//---- POLYTOPAL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer to polytopal Object  */
		vector<zObjMesh*> polytopalObjs;

		int smoothSubDivs;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------
		
		/*!	\brief form function set  */
		zFnGraph fnForm;

		/*!	\brief container of force function set  */
		vector<zFnMesh> fnForces;

		/*!	\brief container of polytopals function set  */
		vector <zFnMesh> fnPolytopals;

		/*!	\brief force volume mesh faceID to form graph vertexID map.  */
		unordered_map <string, int> forceVolumeFace_formGraphVertex;

		/*!	\brief form graph vertexID to volume meshID map.  */
		vector<int> formGraphVertex_forceVolumeMesh;

		/*!	\brief form graph vertexID to local volume mesh faceID map.*/
		vector<int> formGraphVertex_forceVolumeFace;

		/*!	\brief graph vertex offset container.*/
		vector<double> formGraphVertex_Offsets;

		/*!	\brief container of facecenter per force volume  */
		vector<vector<zVector>> force_fCenters;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsPolytopal() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_mesh			- input mesh.
		*	\since version 0.0.2
		*/
		zTsPolytopal(zObjGraph &_formObj, vector<zObjMesh> &_forceObjs, vector<zObjMesh>  &_polytopalObjs)
		{
			formObj = &_formObj;
			fnForm = zFnGraph(_formObj);

			for (int i = 0; i < _forceObjs.size(); i++)
			{
				forceObjs.push_back(&_forceObjs[i]);
				fnForces.push_back(_forceObjs[i]);

				polytopalObjs.push_back(&_polytopalObjs[i]);
				fnPolytopals.push_back(_polytopalObjs[i]);

			}

		

		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsPolytopal() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------

		/** \addtogroup Polytopal creation
		*	\brief Collection of polytopal creation methods.
		*  @{
		*/

		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param [in]		directory		- input directory path.
		*	\param [in]		filename		- input filename.
		*	\param [in]		numFiles		- input number of files in the directory.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.1
		*/
		void createForceFromFile(string directory, string filename, int numFiles,  zFileTpye type)
		{

			if (type == zJSON)
			{

				for (int i = 0; i < numFiles; i++)
				{
					string path = directory + "/" + filename + "_" + to_string(i) + ".json";

					if (i < fnForces.size()) fnForces[i].from(path, type);
				}



			}

			else if (type == zOBJ)
			{
				for (int i = 0; i < numFiles; i++)
				{
					string path = directory + "/" + filename + "_" + to_string(i) + ".obj";

					if (i < fnForces.size()) fnForces[i].from(path, type);
				}
			}

			else throw std::invalid_argument(" error: invalid zFileTpye type");

		}

		/*! \brief This method creates the center line graph based on the input volume meshes.
		*
		*	\param		[in]	offsets						- input offsets value container.
		*	\param		[in]	precisionFac				- precision factor of the points for checking.
		*	\since version 0.0.1
		*/
		void createFormFromForce( double offset , int precisionFac = 3, zColor edgeCol =zColor(0.75, 0.5, 0, 1))
		{
			
			vector<zVector>positions;
			vector<int>edgeConnects;
			unordered_map <string, int> positionVertex;


			forceVolumeFace_formGraphVertex.clear();

			formGraphVertex_forceVolumeMesh.clear();

			formGraphVertex_forceVolumeFace.clear();
		

			for (int j = 0; j < fnForces.size(); j++)
			{
				int n_v = fnForces[j].numVertices();
				int n_e = fnForces[j].numEdges();
				int n_f = fnForces[j].numPolygons();

				zVector volCenter =	fnForces[j].getCenter();
				
				int vId_volCenter = -1;
				bool chkExists = coreUtils.vertexExists(positionVertex, volCenter, precisionFac, vId_volCenter);
				

				if (!chkExists)
				{
					coreUtils.addToPositionMap(positionVertex, volCenter, positions.size(), precisionFac);
				
					vId_volCenter = positions.size();
					positions.push_back(volCenter);					

					formGraphVertex_forceVolumeFace.push_back(-1);
					formGraphVertex_forceVolumeFace.push_back(-1);

					formGraphVertex_forceVolumeMesh.push_back(j);
					formGraphVertex_forceVolumeMesh.push_back(-1);
				}

				string hashKey_volface = (to_string(j) + "," + to_string(-1));
				forceVolumeFace_formGraphVertex[hashKey_volface] = vId_volCenter;

				vector<zVector> fCenters;
				fnForces[j].getCenters(zFaceData, fCenters);

				for (int i = 0; i < fCenters.size(); i++)
				{
	
					int vId_fCenter = -1;
					bool chkExists_f = coreUtils.vertexExists(positionVertex, fCenters[i], precisionFac, vId_fCenter);
				
					if (!chkExists_f)
					{
						coreUtils.addToPositionMap(positionVertex, fCenters[i], positions.size(), precisionFac);

						vId_fCenter = positions.size();
						positions.push_back(fCenters[i]);						

						formGraphVertex_forceVolumeFace.push_back(i);
						formGraphVertex_forceVolumeFace.push_back(-1);

						formGraphVertex_forceVolumeMesh.push_back(j);
						formGraphVertex_forceVolumeMesh.push_back(-2);
					}
					else
					{
						formGraphVertex_forceVolumeFace[(vId_fCenter * 2) + 1] = i;
						formGraphVertex_forceVolumeMesh[(vId_fCenter * 2) + 1] = j;
					}
					
					string hashKey_volface = (to_string(j) + "," + to_string(i));
					forceVolumeFace_formGraphVertex[hashKey_volface] = vId_fCenter;

					edgeConnects.push_back(vId_volCenter);
					edgeConnects.push_back(vId_fCenter);
				}
			}

			fnForm.clear();
			fnFormParticles.clear();

			fnForm.create(positions, edgeConnects);
			
			setVertexOffset(offset);			
			fnForm.setEdgeColor(edgeCol);


			// compute intersection point
			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				if (formGraphVertex_forceVolumeFace[i * 2] >= -1 && formGraphVertex_forceVolumeFace[(i * 2) + 1] != -1)
				{
					int volMesh_1 = formGraphVertex_forceVolumeMesh[i * 2];
					int f1 = formGraphVertex_forceVolumeFace[i * 2];

					zVector normF1 = fnForces[volMesh_1].getFaceNormal(f1); 
					
					zVector currentPos = fnForm.getVertexPosition(i);

					vector<int> cVerts;
					fnForm.getConnectedVertices(i, zVertexData, cVerts);

					if (cVerts.size() == 2)
					{
						zVector p1 = fnForm.getVertexPosition(cVerts[0]);
						
						zVector p2 = fnForm.getVertexPosition(cVerts[1]);

						formGraphVertex_Offsets[i] = (formGraphVertex_Offsets[cVerts[0]] + formGraphVertex_Offsets[cVerts[1]])*0.5;

						zVector interPt;
						bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

						if (chkIntersection)
						{
							fnForm.setVertexPosition(i, interPt);							

							double distTOp1 = interPt.distanceTo(p1);
							double distTOp2 = interPt.distanceTo(p2);
							double distp12 = p1.distanceTo(p2);

							double wt1 = distTOp1 / distp12;
							double wt2 = distTOp2 / distp12;

							formGraphVertex_Offsets[i] = (formGraphVertex_Offsets[cVerts[0]] * wt1) + (formGraphVertex_Offsets[cVerts[1]] * wt2);

						}
					}
				}
			}

			
		}

		/*! \brief This method creates the polytopal mesh from the force volume meshes and form graph.
		*
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void createPolytopalsFromForce(int subdivs = 0)
		{
			smoothSubDivs = subdivs;

			for (int i = 0; i < fnForces.size(); i++)
			{
				fnPolytopals[i].clear();
				getPolytopal(i, subdivs);
			}
		}
		

		/** @}*/

		//--------------------------
		//----3D GS
		//--------------------------

		/** \addtogroup Polytopal 3DGS
		*	\brief Collection of polytopal 3DGS methods.
		*  @{
		*/

		/*! \brief This method updates the form diagram to find equilibrium shape..
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	tolerance					- tolerance for force.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool equilibrium( bool computeTargets, double minmax_Edge, double dT, zIntergrationType type, double angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false)
		{
			
			if (computeTargets)
			{
				computeFormTargets();

				computeTargets = !computeTargets;
			}

			updateFormDiagram(minmax_Edge, dT, type);

			// check deviations
			double minDev, maxDev;
			bool out = checkParallelity(minDev, maxDev, angleTolerance, colorEdges, printInfo);			

			return out;

		}

		/*! \brief This method closes the polytopal meshes.
		*
		*	\since version 0.0.2
		*/
		void closePolytopals()
		{
			
			for (int i = 0; i < fnPolytopals.size(); i++)
			{
				getClosePolytopalMesh(i);
			}
		}

		

		/** @}*/

		

		//--------------------------
		//--- SET METHODS 
		//--------------------------

		/** \addtogroup Polytopal set methods
		*	\brief Collection of polytopal set attribute methods.
		*  @{
		*/

		/*! \brief This method sets form vertex offsets of all the vertices to the input value.
		*
		*	\param		[in]	offset			- offset value.
		*	\since version 0.0.2
		*/
		void setVertexOffset(double offset)
		{
			formGraphVertex_Offsets.clear(); 

			for (int i = 0; i < fnForm.numVertices(); i++) formGraphVertex_Offsets.push_back(offset);
		}

		/*! \brief This method sets form vertex offsets of all the vertices with the input container of values.
		*
		*	\param		[in]	offsets			- container of offsets values.
		*	\since version 0.0.2
		*/
		void setVertexOffsets(vector<double> &offsets)
		{
			if (offsets.size() != fnForm.numVertices()) throw std::invalid_argument("size of offsets contatiner is not equal to number of graph vertices.");

			formGraphVertex_Offsets = offsets;
		}

		/*! \brief This method computes the form graph edge weights based on the force volume mesh face areas.
		*
		*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf)
		*	\param		[in]	minWeight					- minimum weight of the edge.
		*	\param		[in]	maxWeight					- maximum weight of the edge.
		*	\since version 0.0.1
		*/
		 void setFormEdgeWeightsfromForce( double minWeight = 2.0, double maxWeight = 10.0)
		{
			//compute edgeWeights
			vector<vector<double>> volMesh_fAreas;

			double minArea = 10000, maxArea = -100000;
			zColor maxCol(0.784, 0, 0.157, 1);
			zColor minCol(0.027, 0, 0.157, 1);

			for (int i = 0; i < fnForces.size(); i++)
			{
				vector<double> fAreas;
				fnForces[i].getPlanarFaceAreas(fAreas);

				double temp_MinArea = coreUtils.zMin(fAreas);
				minArea = (temp_MinArea < minArea) ? temp_MinArea : minArea;

				double temp_maxArea = coreUtils.zMax(fAreas);
				maxArea = (temp_maxArea > maxArea) ? temp_maxArea : maxArea;

				volMesh_fAreas.push_back(fAreas);
			}

			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				if (formGraphVertex_forceVolumeFace[i * 2] == -1 && formGraphVertex_forceVolumeFace[(i * 2) + 1] == -1) continue;

				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				int volID = formGraphVertex_forceVolumeMesh[i * 2];
				int faceID = formGraphVertex_forceVolumeFace[i * 2];

				double fArea = volMesh_fAreas[volID][faceID];

				for (int j = 0; j < cEdges.size(); j++)
				{
					double val = coreUtils.ofMap(fArea, minArea, maxArea, minWeight, maxWeight);

					zColor col = coreUtils.blendColor(fArea, minArea, maxArea, minCol, maxCol, zRGB);
				
					fnForm.setEdgeWeight(cEdges[j], val);  					
					fnForm.setEdgeColor(cEdges[j], col);
				}

			}
		}

		/** @}*/
		 	

	protected:
		/** \addtogroup Polytopal utilities
		*	\brief Collection of polytopal utility methods.
		*  @{
		*/

		/*! \brief This method creates the polytopal mesh based on the input force volume mesh and form graph.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopal(int forceIndex, int subdivs = 0)
		{
			if (forceIndex > fnForces.size()) throw std::invalid_argument(" error: index out of bounds.");

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			int n_v = fnForces[forceIndex].numVertices();
			int n_e = fnForces[forceIndex].numEdges();
			int n_f = fnForces[forceIndex].numPolygons();



			zVector volCenter =	fnForces[forceIndex].getCenter();


			vector<zVector> fCenters;
			fnForces[forceIndex].getCenters(zFaceData, fCenters);

			for (int i = 0; i < n_e; i += 2)
			{
				vector<int> eFaces;
				fnForces[forceIndex].getFaces(i, zEdgeData, eFaces);
				vector<int> eVertices;
				fnForces[forceIndex].getVertices(i, zEdgeData, eVertices);

				zVector pos0 =	fnForces[forceIndex].getVertexPosition(eVertices[1]);

				zVector pos1 = 	fnForces[forceIndex].getVertexPosition(eVertices[0]);


				if (eFaces.size() == 2)
				{
					int numVerts = positions.size();

					for (int j = 0; j < eFaces.size(); j++)
					{


						string hashKey_f = (to_string(forceIndex) + "," + to_string(eFaces[j]));
						int vId_fCenter = -1;
						bool chkExists_f = coreUtils.existsInMap(hashKey_f, forceVolumeFace_formGraphVertex, vId_fCenter);
						double boundaryOffset = formGraphVertex_Offsets[vId_fCenter];

						zVector fCenter = fCenters[eFaces[j]];
						zVector fCenter_graphPos = 	fnForm.getVertexPosition(vId_fCenter);

						zVector dir_fCenter_0 = pos0 - fCenter;
						double len0 = dir_fCenter_0.length();
						dir_fCenter_0.normalize();

						zVector newPos = fCenter + (dir_fCenter_0 * len0 *boundaryOffset);
						newPos += fCenter_graphPos - fCenter;

						positions.push_back(newPos);


						zVector dir_fCenter_1 = pos1 - fCenter;
						double len1 = dir_fCenter_1.length();
						dir_fCenter_1.normalize();

						zVector newPos1 = fCenter + (dir_fCenter_1 * len1 *boundaryOffset);
						newPos1 += fCenter_graphPos - fCenter;

						positions.push_back(newPos1);
					}

					zVector dir_volCenter_0 = pos0 - volCenter;
					double len0 = dir_volCenter_0.length();
					dir_volCenter_0.normalize();

					string hashKey_v = (to_string(forceIndex) + "," + to_string(-1));
					int vId_vCenter = -1;
					bool chkExists_v = coreUtils.existsInMap(hashKey_v, forceVolumeFace_formGraphVertex, vId_vCenter);
					
					double centerOffset = formGraphVertex_Offsets[vId_vCenter];
					zVector vCenter_graphPos = 	fnForm.getVertexPosition(vId_vCenter);

					zVector newPos = volCenter + (dir_volCenter_0 * len0 *centerOffset);
					newPos += (vCenter_graphPos - volCenter);

					positions.push_back(newPos);


					zVector dir_volCenter_1 = pos1 - volCenter;
					double len1 = dir_volCenter_1.length();
					dir_volCenter_1.normalize();

					zVector newPos1 = volCenter + (dir_volCenter_1 * len1 *centerOffset);
					newPos1 += (vCenter_graphPos - volCenter);

					positions.push_back(newPos1);

					polyConnects.push_back(numVerts);
					polyConnects.push_back(numVerts + 4);
					polyConnects.push_back(numVerts + 5);
					polyConnects.push_back(numVerts + 1);
					polyCounts.push_back(4);

					polyConnects.push_back(numVerts + 5);
					polyConnects.push_back(numVerts + 4);
					polyConnects.push_back(numVerts + 2);
					polyConnects.push_back(numVerts + 3);
					polyCounts.push_back(4);

				}
			}

			if (subdivs == 0) fnPolytopals[forceIndex].create(positions, polyCounts, polyConnects);
			else
			{
				zObjMesh tempObj;
				zFnMesh tempFn(tempObj);

				tempFn.create(positions, polyCounts, polyConnects);
				getPolytopalRulingRemesh(forceIndex, tempFn, subdivs);
			}

		}

		/*! \brief This method remeshes the input mesh to have rulings in ony one direction.
		*
		*	\param		[in]	inFnMesh				- input mesh function set.
		*	\param		[in]	SUBDIVS					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopalRulingRemesh(int index, zFnMesh &inFnMesh, int SUBDIVS)
		{
			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			int n_v_lowPoly = inFnMesh.numVertices();

			inFnMesh.smoothMesh(SUBDIVS);

			int n_v = inFnMesh.numVertices();
			int n_e = inFnMesh.numEdges();
			int n_f = inFnMesh.numPolygons();

			for (int i = 0; i < n_v_lowPoly; i += 6)
			{
				int vert0 = i;
				int vert1 = i + 1;
				int edge0, edge1;

				vector<int> cEdges0;
				inFnMesh.getConnectedEdges(vert0, zVertexData, cEdges0);

				for (int j = 0; j < cEdges0.size(); j++)
				{
					if (!inFnMesh.onBoundary(cEdges0[j], zEdgeData))
					{
						edge0 = inFnMesh.getSymIndex(cEdges0[j]); 
					}
				}

				vector<int> cEdges1;
				inFnMesh.getConnectedEdges(vert1, zVertexData, cEdges1);

				for (int j = 0; j < cEdges1.size(); j++)
				{
					if (inFnMesh.onBoundary(cEdges1[j], zEdgeData))
					{
						edge1 = cEdges1[j];
					}
				}

				zVector v0 = inFnMesh.getVertexPosition(vert0);

				zVector v1 = inFnMesh.getVertexPosition(vert1);

				positions.push_back(v0);
				positions.push_back(v1);

				//while (smoothPolytopalMesh.edges[edge0].getVertex()->getVertexId() != i + 2)
				for (int k = 0; k < pow(2, (SUBDIVS + 1)); k++)
				{
					int numVerts = positions.size();

					int vert2 = inFnMesh.getStartVertexIndex(edge0);   
					int vert3 = inFnMesh.getEndVertexIndex(edge1); 

					zVector v2 = inFnMesh.getVertexPosition(vert2);

					zVector v3 = inFnMesh.getVertexPosition(vert3);

					positions.push_back(v2);
					positions.push_back(v3);

					polyConnects.push_back(numVerts - 2);
					polyConnects.push_back(numVerts);
					polyConnects.push_back(numVerts + 1);
					polyConnects.push_back(numVerts - 1);
					polyCounts.push_back(4);

					//vert0 = vert2;
					//vert1 = vert3;

					edge0 = inFnMesh.getPrevIndex(edge0);
					edge1 = inFnMesh.getNextIndex(edge1);
				}
			}

			fnPolytopals[index].create(positions, polyCounts, polyConnects);			
		}

		/*! \brief This method computes the ruling intersetions.
		*
		*	\param		[in]	v0						- input vertex index 0.
		*	\param		[in]	v1						- input vertex index 1.
		*	\param		[out]	closestPt				- stores closest point if there is intersection bettwen the two ruling edges.
		*	\return				bool					- true if there is a intersection else false.
		*	\since version 0.0.1
		*/
		bool computeRulingIntersection(int polytopalIndex, int v0, int v1, zVector &closestPt)
		{
			bool out = false;

			int e0 = -1;
			int e1 = -1;

			vector<int> cEdges0;
			fnPolytopals[polytopalIndex].getConnectedEdges(v0, zVertexData, cEdges0);
			if (cEdges0.size() == 3)
			{
				for (int i = 0; i < cEdges0.size(); i++)
				{
					if (!fnPolytopals[polytopalIndex].onBoundary(cEdges0[i], zEdgeData))
					{
						e0 = cEdges0[i];
						break;
					}
				}
			}

			vector<int> cEdges1;
			fnPolytopals[polytopalIndex].getConnectedEdges(v1, zVertexData, cEdges1);
			if (cEdges1.size() == 3)
			{
				for (int i = 0; i < cEdges1.size(); i++)
				{
					if (!fnPolytopals[polytopalIndex].onBoundary(cEdges1[i], zEdgeData))
					{
						e1 = cEdges1[i];
						break;
					}
				}
			}

			if (e0 != -1 && e1 != -1)
			{
				int v2 = (v0 % 2 == 0) ? v0 + 1 : v0 - 1;
				int v3 = (v1 % 2 == 0) ? v1 + 1 : v1 - 1;

				zVector a0 = fnPolytopals[polytopalIndex].getVertexPosition(v2);

				zVector a1 = fnPolytopals[polytopalIndex].getVertexPosition(v0);

				zVector b0 = fnPolytopals[polytopalIndex].getVertexPosition(v3);

				zVector b1 = fnPolytopals[polytopalIndex].getVertexPosition(v1);

				double uA = -1;
				double uB = -1;
				out = coreUtils.line_lineClosestPoints(a0, a1, b0, b1, uA, uB);

				if (out)
				{
					if (uA >= uB)
					{
						zVector dir = a1 - a0;
						double len = dir.length();
						dir.normalize();

						if (uA < 0) dir *= -1;
						closestPt = a0 + dir * len * uA;
					}
					else
					{
						zVector dir = b1 - b0;
						double len = dir.length();
						dir.normalize();

						if (uB < 0) dir *= -1;

						closestPt = b0 + dir * len * uB;
					}


				}

			}
			return out;
		}

		/*! \brief This method closes the input polytopal mesh.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\since version 0.0.1
		*/
		void getClosePolytopalMesh(int forceIndex)
		{
			if (smoothSubDivs == 0) return;

			int n_v = fnForces[forceIndex].numVertices();
			int n_e = fnForces[forceIndex].numEdges();
			int n_f = fnForces[forceIndex].numPolygons();

			int n_v_smooth = fnPolytopals[forceIndex].numVertices();
			int n_e_smooth = fnPolytopals[forceIndex].numEdges();
			int n_f_smooth = fnPolytopals[forceIndex].numPolygons();

			int numVertsPerStrip = floor(n_v_smooth / (0.5 * n_e));
			int half_NumVertsPerStrip = floor(numVertsPerStrip / 2);


			vector<bool> vertVisited;

			for (int i = 0; i < n_v_smooth; i++)
			{
				vertVisited.push_back(false);
			}

			for (int i = 0; i < n_e; i += 2)
			{
				int eStripId = floor(i / 2);


				//-- Prev  Edge	

				int ePrev = fnForces[forceIndex].getPrevIndex(i);
				int ePrevStripId = floor(ePrev / 2);


				if (ePrev % 2 == 0)
				{
					for (int j = 1, k = 0; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip - 2; j += 2, k += 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = ePrevStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);

							}

							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}



					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = 0; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip - 2; j -= 2, k += 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = ePrevStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);

							}
							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}
					}
				}

				//-- Next Edge		

				int eNext = fnForces[forceIndex].getNextIndex(i);
				int eNextStripId = floor(eNext / 2);

				if (eNext % 2 == 0)
				{
					for (int j = 0, k = 1; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip; j += 2, k += 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eNextStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}

							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}

					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = 1; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip; j -= 2, k += 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eNextStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}

							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}

					}
				}


				//-- SYM Prev  Edge	

				int symEdge = fnForces[forceIndex].getSymIndex(i);

				int eSymPrev = fnForces[forceIndex].getPrevIndex(symEdge);
				int eSymPrevStripId = floor(eSymPrev / 2);


				if (eSymPrev % 2 == 0)
				{
					for (int j = 1, k = numVertsPerStrip - 1; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip; j += 2, k -= 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eSymPrevStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}

							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}
					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = numVertsPerStrip - 1; j > half_NumVertsPerStrip - 1, k > half_NumVertsPerStrip; j -= 2, k -= 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eSymPrevStripId * numVertsPerStrip + j;

						if (!vertVisited[v0] && !vertVisited[v1])
						{

							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}
							vertVisited[v0] = true;
							vertVisited[v1] = true;

						}
					}
				}


				//--SYM Next Edge		
				int eSymNext = fnForces[forceIndex].getNextIndex(symEdge);
				int eSymNextStripId = floor(eSymNext / 2);

				if (eSymNext % 2 == 0)
				{
					for (int j = 0, k = numVertsPerStrip - 2; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip - 1; j += 2, k -= 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eSymNextStripId * numVertsPerStrip + j;


						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}
							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}
					}
				}
				else
				{
					for (int j = numVertsPerStrip - 1, k = numVertsPerStrip - 2; j > half_NumVertsPerStrip, k > half_NumVertsPerStrip - 1; j -= 2, k -= 2)
					{
						int v0 = eStripId * numVertsPerStrip + k;
						int v1 = eSymNextStripId * numVertsPerStrip + j;


						if (!vertVisited[v0] && !vertVisited[v1])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								fnPolytopals[forceIndex].setVertexPosition(v0, cPt);
								fnPolytopals[forceIndex].setVertexPosition(v1, cPt);
							}
							vertVisited[v0] = true;
							vertVisited[v1] = true;
						}

					}
				}
			}

			for (int i = 0; i < n_v; i++)
			{
				vector<int> cEdges;
				fnForces[forceIndex].getConnectedEdges(i, zVertexData, cEdges);

				vector<int> smoothMeshVerts;
				vector<zVector> intersectPoints;

				// get connected edge strips
				for (int j = 0; j < cEdges.size(); j++)
				{
					int eStripId = floor(cEdges[j] / 2);
					int vertId = eStripId * numVertsPerStrip + half_NumVertsPerStrip;

					if (cEdges[j] % 2 == 0) vertId -= 1;
					smoothMeshVerts.push_back(vertId);
				}

				// comput smooth mesh vertices
				for (int j = 0; j < smoothMeshVerts.size(); j++)
				{
					int v0 = smoothMeshVerts[j];
					int v1 = smoothMeshVerts[(j + 1) % smoothMeshVerts.size()];

					vertVisited[v0] = true;

					zVector cPt;
					bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

					if (intersectChk)
					{
						intersectPoints.push_back(cPt);
					}
				}

				// get average intersection point
				zVector avgIntersectPoint;

				for (int j = 0; j < intersectPoints.size(); j++)
				{
					avgIntersectPoint += intersectPoints[j];
				}

				avgIntersectPoint = avgIntersectPoint / intersectPoints.size();

				//// update positions
				for (int j = 0; j < smoothMeshVerts.size(); j++)
				{
					fnPolytopals[forceIndex].setVertexPosition(smoothMeshVerts[j], avgIntersectPoint);
				}

			}


		}

		/*! \brief This method computes the face centers of the input force volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.1
		*/
		void computeForcesFaceCenters()
		{
			force_fCenters.clear();

			for (int i = 0; i < fnForces.size(); i++)
			{
				vector<zVector> fCenters;
				fnForces[i].getCenters(zFaceData, fCenters);

				force_fCenters.push_back(fCenters);
			}
		}

		/*! \brief This method computes the targets per edge of the form.
		*
		*	\since version 0.0.1
		*/
		void computeFormTargets()
		{
			targetEdges_form.clear(); 

			for (int i = 0; i < fnForm.numEdges(); i++)
			{
				targetEdges_form.push_back(zVector());
			}

			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				// get position of vertex
				zVector v_i = fnForm.getVertexPosition(i);

				int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
				int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

				if (fnForm.checkVertexValency(i, 1))	continue;
				if (faceId_V != -1 && formGraphVertex_forceVolumeFace[i * 2 + 1] != -1) continue;
			
				// get connected vertices
				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

			
				for (int j = 0; j < cEdges.size(); j++)
				{
					// get vertex 
					int v1_ID = fnForm.getEndVertexIndex(cEdges[j]);
					
					zVector v_j = fnForm.getVertexPosition(v1_ID);
					zVector e_ij = v_i - v_j;
					e_ij.normalize();

					// get volume and face Id of the connected Vertex
					int volId = formGraphVertex_forceVolumeMesh[v1_ID * 2];
					int faceId = formGraphVertex_forceVolumeFace[v1_ID * 2];				

					zVector t_ij = fnForces[volId].getFaceNormal(faceId);;
					t_ij.normalize();

					if (e_ij * t_ij > 0) t_ij *= -1;

					targetEdges_form[cEdges[j]] = t_ij;

					int symEdge = fnForm.getSymIndex(cEdges[j]);
					targetEdges_form[symEdge] = (t_ij * -1);

				}
			}

		}

		/*! \brief This method if the form mesh edges and corresponding target edge are parallel.
		*
		*	\param		[out]	minDeviation						- stores minimum deviation.
		*	\param		[out]	maxDeviation						- stores maximum deviation.
		*	\param		[in]	angleTolerance						- angle tolerance for parallelity.
		*	\param		[in]	printInfo							- printf information of minimum and maximum deviation if true.
		*	\return				bool								- true if the all the correponding edges are parallel or within tolerance.
		*	\since version 0.0.2
		*/
		bool checkParallelity(double & minDeviation, double & maxDeviation, double angleTolerance, bool colorEdges, bool printInfo)
		{
			bool out = true;
			vector<double> deviations;
			minDeviation = 10000;
			maxDeviation = -10000;

			for (int i = 0; i < fnForm.numEdges(); i+= 1)
			{
				//form edge
				int eId_form = i;
				zVector e_form = fnForm.getEdgeVector(eId_form);
				e_form.normalize();

				zVector e_target = targetEdges_form[i];

				double a_i = e_form.angle(e_target);
				
				deviations.push_back(a_i);
			

				if (a_i > angleTolerance)
				{
					out = false;					
				}


				if (a_i < minDeviation) minDeviation = a_i;
				if (a_i > maxDeviation) maxDeviation = a_i;
			}


			if (printInfo)
			{
				printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, minDeviation, maxDeviation);
			}

			if (colorEdges)
			{
				for (int i = 0; i < fnForm.numEdges(); i += 1)
				{
					
					zColor col = coreUtils.blendColor(deviations[i], minDeviation, maxDeviation, zColor(180, 1, 1), zColor(0, 1, 1), zHSV);

					if (deviations[i] < angleTolerance) col = zColor();

					fnForm.setEdgeColor(i, col);				

				}

			}

			return out;
		}

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\since version 0.0.2
		*/
		void updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type)
		{
			if (fnFormParticles.size() != fnForm.numVertices())
			{
				fnFormParticles.clear();
				formParticlesObj.clear();


				for (int i = 0; i < fnForm.numVertices(); i++)
				{
					bool fixed = false;

					int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
					int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

					if (faceId_V != -1 && formGraphVertex_forceVolumeFace[i * 2 + 1] != -1)
					{
						fixed = true;
						fnForm.setVertexColor(i, zColor());
					}					

					zObjParticle p;
					p.particle = zParticle(formObj->graph.vertexPositions[i], fixed);
					formParticlesObj.push_back(p);

				}

				for (int i = 0; i < formParticlesObj.size(); i++)
				{
					fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
				}
			}

			if (force_fCenters.size() != fnForces.size()) computeForcesFaceCenters();

			vector<double> edgeLengths;
			fnForm.getEdgeLengths(edgeLengths);

			double minEdgeLength, maxEdgeLength;
			maxEdgeLength = coreUtils.zMax(edgeLengths);

			minEdgeLength = maxEdgeLength * minmax_Edge;

			// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
			for (int i = 0; i < fnFormParticles.size(); i++)
			{

				if (fnFormParticles[i].getFixed()) continue;

				// get position of vertex
				zVector v_i = fnForm.getVertexPosition(i);

				int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
				int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

				// get connected vertices
				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				// compute barycenter per vertex
				zVector b_i;
				for (int j = 0; j < cEdges.size(); j++)
				{
					// get vertex 
					int v1_ID = fnForm.getEndVertexIndex(cEdges[j]);

					zVector v_j = fnForm.getVertexPosition(v1_ID);

					zVector e_ij = v_i - v_j;
					double len_e_ij = e_ij.length();;

					if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
					if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

					int symEdge = fnForm.getSymIndex(cEdges[j]);

					zVector t_ij = targetEdges_form[symEdge];;
					t_ij.normalize();
					if (e_ij * t_ij < 0) t_ij *= -1;

					b_i += (v_j + (t_ij * len_e_ij));

				}

				b_i /= cEdges.size();

				// compute residue force
				zVector r_i = b_i - v_i;
				zVector forceV = r_i;


				// add forces to particle
				fnFormParticles[i].addForce(forceV);

			}


			// update Particles
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				fnFormParticles[i].integrateForces(dT, type);
				fnFormParticles[i].updateParticle(true);
			}

			// update fixed particle positions ( ones with a common face) 
			for (int i = 0; i < fnFormParticles.size(); i++)
			{
				if (!fnFormParticles[i].getFixed()) continue;

				vector<int> cVerts;
				fnForm.getConnectedVertices(i, zVertexData, cVerts);


				int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
				int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

				zVector normF1 = fnForces[volId_V].getFaceNormal(faceId_V);

				zVector currentPos = fnForm.getVertexPosition(i);

				if (cVerts.size() == 2)
				{
					zVector p1 = fnForm.getVertexPosition(cVerts[0]);

					zVector p2 = fnForm.getVertexPosition(cVerts[1]);

					zVector interPt;

					zVector newPos = (p1 + p2) *0.5;
					
					fnForm.setVertexPosition(i, newPos);

					/*bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						fnForm.setVertexPosition(i,interPt);
					}*/
				}

			}

		}


		/** @}*/
	};

}

