#pragma once


#include <headers/api/functionsets/zFnMesh.h>
#include <headers/api/functionsets/zFnGraph.h>
#include <headers/api/functionsets/zFnParticle.h>

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

	/** \addtogroup z3DGS
	*	\brief tool sets for 3D graphic statics.
	*  @{
	*/

	/*! \class zTsPolytopal
	*	\brief A toolset for 3D graphics and poytopal meshes.
	*	\details Based on 3D Graphic Statics (http://block.arch.ethz.ch/brg/files/2015_cad_akbarzadeh_On_the_equilibrium_of_funicular_polyhedral_frames_1425369507.pdf) and Freeform Developable Spatial Structures (https://www.ingentaconnect.com/content/iass/piass/2016/00002016/00000003/art00010 )
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

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
		//---- ALGEBRAIC ATTRIBUTES
		//--------------------------

		int primal_n_v = 0;
		int primal_n_e = 0;
		
		int primal_n_f_i = 0; // primal faces internal
		int primal_n_f = 0;

		unordered_map <string, int> volumeVertex_PrimalVertex;  // map of volume-vertex hashkey to primal vertex

		unordered_map <string, int> volumeEdge_PrimalEdge;  // map of volume-edge hashkey to primal edge
		unordered_map <string, int> primalVertices_PrimalEdge;  // map of primal vertex-vertex hashkey to primal edge

		unordered_map <string, int> volumeFace_PrimalFace; // map of volume-face hashkey to primal face

		vector< vector<int> > primalFace_VolumeFace; // stores the one volume face combination per primal face
		vector< vector<int> > primalEdge_VolumeEdge; // stores the one volume edge combination per primal edge

		vector< vector<int> > primalEdge_PrimalVertices; // stores theprimal vertices per primal edge

		vector<int> primal_internalFaceIndex; // -1 for GFP or SSP faces 

		vector<zVector> primalVertexPositions;
		vector<zVector> primalFaceCenters;
		vector<zVector> primalFaceNormals;

		vector<vector<int>> primalFace_Volumes;

		vector<bool> GFP_SSP_Face;


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
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_forceObjs			- input container of force objects.
		*	\param		[in]	_polytopalObjs		- input container of polytopal objects.
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
				
		/*! \brief This method creates the force geometry from the input files.
		*
		*	\param [in]		filePaths		- input container of file paths.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.2
		*/
		void createForceFromFiles(vector<string> filePaths, zFileTpye type)
		{

			if (type == zJSON)
			{
				for (int i = 0; i < filePaths.size(); i++)
				{
					if (i < fnForces.size()) fnForces[i].from(filePaths[i], type);
				}
			}

			else if (type == zOBJ)
			{
				for (int i = 0; i < filePaths.size(); i++)
				{
					if (i < fnForces.size()) fnForces[i].from(filePaths[i], type);
				}
			}

			else throw std::invalid_argument(" error: invalid zFileTpye type");

		}
		
		/*! \brief This method creates the center line graph based on the input volume meshes.
		*
		*	\param		[in]	offsets						- input offsets value container.
		*	\param		[in]	precisionFac				- precision factor of the points for checking.
		*	\since version 0.0.2
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
				int n_e = fnForces[j].numHalfEdges();
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

			for (zItGraphVertex v(*formObj); !v.end(); v.next())			
			{
				if (!v.checkVertexValency(1))
				{
					zColor col(0, 0, 0, 1);
					v.setVertexColor(col);
				}
			}
			
			setVertexOffset(offset);			
			fnForm.setEdgeColor(edgeCol);


			// compute intersection point
			for (zItGraphVertex v(*formObj); !v.end(); v.next())
			{
				int i = v.getId();

				if (formGraphVertex_forceVolumeFace[i * 2] >= -1 && formGraphVertex_forceVolumeFace[(i * 2) + 1] != -1)
				{
					int volMesh_1 = formGraphVertex_forceVolumeMesh[i * 2];
					int f1 = formGraphVertex_forceVolumeFace[i * 2];

					if (f1 == -1) continue;
					
					zItMeshFace force_face(*forceObjs[volMesh_1], f1);
					zVector normF1 = force_face.getFaceNormal();
					
					zVector currentPos = v.getVertexPosition();

					vector<zItGraphVertex> cVerts;
					v.getConnectedVertices(cVerts);

					if (cVerts.size() == 2)
					{
						zVector p1 = cVerts[0].getVertexPosition();
						
						zVector p2 = cVerts[1].getVertexPosition();

						formGraphVertex_Offsets[i] = (formGraphVertex_Offsets[cVerts[0].getId()] + formGraphVertex_Offsets[cVerts[1].getId()])*0.5;

						zVector interPt;
						bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

						if (chkIntersection)
						{
							v.setVertexPosition( interPt);							

							double distTOp1 = interPt.distanceTo(p1);
							double distTOp2 = interPt.distanceTo(p2);
							double distp12 = p1.distanceTo(p2);

							double wt1 = distTOp1 / distp12;
							double wt2 = distTOp2 / distp12;

							formGraphVertex_Offsets[i] = (formGraphVertex_Offsets[cVerts[0].getId()] * wt1) + (formGraphVertex_Offsets[cVerts[1].getId()] * wt2);

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
		
				

		//--------------------------
		//----3D GS
		//--------------------------

		
		/*! \brief This method updates the form diagram to find equilibrium shape..
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations.
		*	\param		[in]	angleTolerance				- tolerance for angle.
		*	\param		[in]	colorEdges					- colors edges based on angle deviation if true.
		*	\param		[in]	printInfo					- prints angle deviation info if true.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool equilibrium( bool computeTargets, double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false)
		{
			
			if (computeTargets)
			{
				computeFormTargets();

				computeTargets = !computeTargets;
			}

			updateFormDiagram(minmax_Edge, dT, type, numIterations);

			// check deviations
			zDomainDouble dev;
			bool out = checkParallelity(dev, angleTolerance, colorEdges, printInfo);

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
			
		//--------------------------
		//---- MATRIX UTILITIES
		//--------------------------
		
		void set_GFP_SSP(zDiagramType type, int index = -1)
		{
			if (type == zForceDiagram)
			{
				if (index == -1)
				{

				}

				else if (index >= 0 && index < fnForces.size())
				{

				}
				else throw std::invalid_argument(" invalid index.");
			}
			
			else if (type == zFormDiagram)
			{

			}
			
			else throw std::invalid_argument(" invalid diagram type.");
		}

		void getPrimal_GlobalElementIndicies(zDiagramType type, int GFP_SSP_Index = -1,  int precisionFac = 6)
		{
			if (type == zForceDiagram)
			{

				if (GFP_SSP_Index < -1 && GFP_SSP_Index >= fnForces.size())  throw std::invalid_argument(" invalid GFP / SSP index.");

				primal_n_v = 0;
				primal_n_e = 0;
				primal_n_f = 0;
				primal_n_f_i = 0;

				primalFace_VolumeFace.clear();
				primalEdge_VolumeEdge.clear();
				primalEdge_PrimalVertices.clear();
				GFP_SSP_Face.clear();

				primalVertexPositions.clear();
				primalFaceCenters.clear();
				primalFaceNormals.clear();
				primal_internalFaceIndex.clear();
				

				unordered_map <string, int> positionVertex;
				unordered_map <string, int> faceCenterpositionVertex;			
									
				
									

				// face map
				for (int j = 0; j < fnForces.size(); j++)
				{

					vector<zVector> fCenters;
					vector<zVector> fNorms;

					fnForces[j].getCenters(zFaceData, fCenters);
					fnForces[j].getFaceNormals(fNorms);

					for (int i = 0; i < fCenters.size(); i++)
					{
						int globalFaceId = -1;
						bool chkExists = coreUtils.vertexExists(faceCenterpositionVertex, fCenters[i], precisionFac, globalFaceId);



						if (!chkExists)
						{
							coreUtils.addToPositionMap(faceCenterpositionVertex, fCenters[i], primal_n_f, precisionFac);

							vector<int> volumeFace = { j,i };
							primalFace_VolumeFace.push_back(volumeFace);

							globalFaceId = primal_n_f;

							primalFaceCenters.push_back(fCenters[i]);

							fNorms[i].normalize();
							primalFaceNormals.push_back(fNorms[i]);

							// GFP or SSP are external faces
							if(GFP_SSP_Index == -1) GFP_SSP_Face.push_back(true);
							else GFP_SSP_Face.push_back(false);

							primal_n_f++;
						}
						else
						{
							// GFP or SSP are external faces
							if (GFP_SSP_Index == -1) GFP_SSP_Face[globalFaceId] = false;
						}

						string hashKey_volFace = (to_string(j) + "," + to_string(i));
						volumeFace_PrimalFace[hashKey_volFace] = globalFaceId;
					}
				}

				// GFP or SSP are specified volume
				if (GFP_SSP_Index != -1)
				{
					for (int i = 0; i < fnForces[GFP_SSP_Index].numPolygons(); i++)
					{
						string hashKey_volFace = (to_string(GFP_SSP_Index) + "," + to_string(i));

						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

						if (gotFace != volumeFace_PrimalFace.end())
						{
							int globalFaceId = gotFace->second;
							GFP_SSP_Face[globalFaceId] =true;
						}
					}

					
				}

				// compute internal face index
				for (int i = 0; i < GFP_SSP_Face.size(); i++)
				{
					if (!GFP_SSP_Face[i])
					{
						primal_internalFaceIndex.push_back(primal_n_f_i);
						primal_n_f_i++;
					}
					else primal_internalFaceIndex.push_back(-1);					
				}

				// vertex and edge maps
				for (int j = 0; j < fnForces.size(); j++)
				{
					// vertex map
					zVector *pos = fnForces[j].getRawVertexPositions();

					for (int i = 0; i < fnForces[j].numVertices(); i++)
					{
						int globalVertexId = -1;
						bool chkExists = coreUtils.vertexExists(positionVertex, pos[i], precisionFac, globalVertexId);

						if (!chkExists)
						{
							coreUtils.addToPositionMap(positionVertex, pos[i], primal_n_v, precisionFac);

							primalVertexPositions.push_back(pos[i]);

							globalVertexId = primal_n_v;
							primal_n_v++;
						}

						string hashKey_volVertex = (to_string(j) + "," + to_string(i));
						volumeVertex_PrimalVertex[hashKey_volVertex] = globalVertexId;
					}

					//edge map
					for (zItMeshEdge e(*forceObjs[j]); !e.end(); e.next())
					{
						int eId = e.getId();

						vector<int> eFaces;
						e.getFaces(eFaces);

						bool boundaryEdge = false; 

						for (int i = 0; i < eFaces.size(); i++)
						{
							string hashKey_volFace = (to_string(j) + "," + to_string(eFaces[i]));

							std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_volFace);

							if (gotFace != volumeFace_PrimalFace.end())
							{
								int globalFaceId = gotFace->second;
								if(GFP_SSP_Face[globalFaceId]) boundaryEdge = true;
							}
						}

						if (boundaryEdge) continue;

						int v0, v1;

						string hashKey_v0 = (to_string(j) + "," + to_string(e.getHalfEdge(0).getStartVertex().getId()));
						std::unordered_map<std::string, int>::const_iterator got0 = volumeVertex_PrimalVertex.find(hashKey_v0);

						string hashKey_v1 = (to_string(j) + "," + to_string(e.getHalfEdge(0).getVertex().getId()));
						std::unordered_map<std::string, int>::const_iterator got1 = volumeVertex_PrimalVertex.find(hashKey_v1);

						if (got0 != volumeVertex_PrimalVertex.end() && got1 != volumeVertex_PrimalVertex.end())
						{
							v0 = got0->second;
							v1 = got1->second;

							if (v0 > v1) swap(v0, v1);

							string hashKey_e = (to_string(v0) + "," + to_string(v1));
							std::unordered_map<std::string, int>::const_iterator gotGlobalEdge = primalVertices_PrimalEdge.find(hashKey_e);

							int globalEdgeId;

							if (gotGlobalEdge != primalVertices_PrimalEdge.end())
							{
								globalEdgeId = gotGlobalEdge->second;
							}
							else
							{
								primalVertices_PrimalEdge[hashKey_e] = primal_n_e;

								vector<int> volumeEdge = { j,eId };
								primalEdge_VolumeEdge.push_back(volumeEdge);

								vector<int> primalVertices = { v0,v1 };
								primalEdge_PrimalVertices.push_back(primalVertices);

								globalEdgeId = primal_n_e;
								primal_n_e++;
							}

							string hashKey_volEdge = (to_string(j) + "," + to_string(eId));
							volumeEdge_PrimalEdge[hashKey_volEdge] = globalEdgeId;

						}

					}
					
					// face map
					/*vector<zVector> fCenters;
					vector<zVector> fNorms;
					
					fnForces[j].getCenters(zFaceData, fCenters);
					fnForces[j].getFaceNormals(fNorms);

					for (int i = 0; i < fCenters.size(); i++)
					{
						int globalFaceId = -1;
						bool chkExists = coreUtils.vertexExists(faceCenterpositionVertex, fCenters[i], precisionFac, globalFaceId);



						if (!chkExists)
						{		

							coreUtils.addToPositionMap(faceCenterpositionVertex, fCenters[i], primal_n_f, precisionFac);
							
							vector<int> volumeFace = { j,i };
							primalFace_VolumeFace.push_back(volumeFace);

							globalFaceId = primal_n_f;

							primalFaceCenters.push_back(fCenters[i]);

							fNorms[i].normalize();
							primalFaceNormals.push_back(fNorms[i]);

							GFP_SSP_Face.push_back(true);

							primal_n_f++;
						}
						else
						{
							GFP_SSP_Face[globalFaceId] = false;
							
						}

						string hashKey_volFace = (to_string(j) + "," + to_string(i));
						volumeFace_PrimalFace[hashKey_volFace] = globalFaceId;
					}*/

					

				}

				printf("\n primal Force:  primal_n_v %i , primal_n_e %i , primal_n_f %i  primal_n_f_i %i", primal_n_v, primal_n_e, primal_n_f, primal_n_f_i);
				printf("\n primal_f_normals %i ", primalFaceNormals.size());
			

			}
			
			else throw std::invalid_argument(" invalid diagram type.");

		}

		bool getPrimal_EdgeVertexMatrix(zDiagramType type, zSparseMatrix &out)
		{
			if (type == zForceDiagram)
			{
				if (primal_n_v == 0 || primal_n_e == 0) return false;

				vector<bool> edgeVisited;
				edgeVisited.assign(primal_n_e, false );

				out = zSparseMatrix(primal_n_e, primal_n_v);
				out.setZero();
								
				vector<zTriplet> coefs; // -1 for from vertex and 1 for to vertex

				for (int j = 0; j < primalEdge_PrimalVertices.size(); j++)
				{
					int v0 = primalEdge_PrimalVertices[j][0];
					int v1 = primalEdge_PrimalVertices[j][1];

					coefs.push_back(zTriplet(j, v0, -1));
					coefs.push_back(zTriplet(j, v1, 1));					
				}
				
				out.setFromTriplets(coefs.begin(), coefs.end());

				return true;
			}

			else throw std::invalid_argument(" invalid diagram type.");

		}

		bool getPrimal_EdgeFaceMatrix(zDiagramType type, zSparseMatrix &out)
		{
			if (type == zForceDiagram)
			{
				if (primal_n_f == 0 || primal_n_e == 0) return false;

				out = zSparseMatrix(primal_n_e, primal_n_f_i);
				out.setZero();
							
				vector<zTriplet> coefs; // -1 for from vertex and 1 for to vertex

				for (int j = 0; j < primalFace_VolumeFace.size(); j++)
				{
					if (GFP_SSP_Face[j]) continue;

					int volId = primalFace_VolumeFace[j][0];
					int faceId = primalFace_VolumeFace[j][1];

					
									   					 			
					zItMeshFace f(*forceObjs[volId], faceId);

					vector<zItMeshHalfEdge> fHEdges;
					f.getHalfEdges(fHEdges);

					for (auto &he : fHEdges)
					{
						int edgeId = he.getEdge().getId();

						int v0 = he.getStartVertex().getId();
						int v1 = he.getVertex().getId();


						string hashKey_v0 = (to_string(volId) + "," + to_string(v0));
						std::unordered_map<std::string, int>::const_iterator gotVertex0 = volumeVertex_PrimalVertex.find(hashKey_v0);

						string hashKey_v1 = (to_string(volId) + "," + to_string(v1));
						std::unordered_map<std::string, int>::const_iterator gotVertex1 = volumeVertex_PrimalVertex.find(hashKey_v1);

						string hashKey_e = (to_string(volId) + "," + to_string(edgeId));
						std::unordered_map<std::string, int>::const_iterator gotEdge = volumeEdge_PrimalEdge.find(hashKey_e);

						if (gotVertex0 != volumeVertex_PrimalVertex.end() && gotVertex1 != volumeVertex_PrimalVertex.end() && gotEdge != volumeEdge_PrimalEdge.end())
						{
							int p_v0 = gotVertex0->second;
							int p_v1 = gotVertex1->second;							

							int p_e = gotEdge->second;

							if (p_v0 > p_v1)
							{
								
								coefs.push_back(zTriplet(p_e, primal_internalFaceIndex[j], -1));
							}
							else
							{
								coefs.push_back(zTriplet(p_e, primal_internalFaceIndex[j], 1));
							}
						}

					}					

				}

				out.setFromTriplets(coefs.begin(), coefs.end());

				return true;
			}

			else throw std::invalid_argument(" invalid diagram type.");

		}

		bool getPrimal_FaceVolumeMatrix(zDiagramType type, zSparseMatrix &out)
		{
			if (type == zForceDiagram)
			{
				if (primal_n_f == 0 ) return false;

				int primal_n_vol = fnForces.size();
				
				out = zSparseMatrix(primal_n_f_i, primal_n_vol);
				out.setZero();

				vector<zTriplet> coefs; 

				for (int j = 0; j < fnForces.size(); j++)
				{
					int volId = j;
					

					for (zItMeshFace f(*forceObjs[volId]); !f.end(); f.next())
					{
						int faceId = f.getId();

						string hashKey_f = (to_string(volId) + "," + to_string(faceId));
						std::unordered_map<std::string, int>::const_iterator gotFace = volumeFace_PrimalFace.find(hashKey_f);

						if (gotFace != volumeFace_PrimalFace.end())
						{
							int p_f = gotFace->second;

							if (GFP_SSP_Face[p_f]) continue;

							int p_volId = primalFace_VolumeFace[p_f][0];
							int p_faceId = primalFace_VolumeFace[p_f][1];

							if (volId == p_volId && faceId == p_faceId && !GFP_SSP_Face[p_f])
							{
								coefs.push_back(zTriplet(primal_internalFaceIndex[p_f], volId, 1));
							}
							else coefs.push_back(zTriplet(primal_internalFaceIndex[p_f], volId, -1));
						}

					}			

				}

				out.setFromTriplets(coefs.begin(), coefs.end());

				return true;
			}

			else throw std::invalid_argument(" invalid diagram type.");

		}

		bool get_EquilibriumMatrix(zDiagramType type, MatrixXd &out)
		{
			double factor = pow(10, 4);

			// Get Normal diagonal matricies
			VectorXd nx(primal_n_f_i);
			VectorXd ny(primal_n_f_i);
			VectorXd nz(primal_n_f_i);


			for (int j = 0; j < primalFaceNormals.size(); j++)
			{	
				if (GFP_SSP_Face[j]) continue;

				zVector n = primalFaceNormals[j];	

				int id = primal_internalFaceIndex[j];

				nx[id] = n.x;
				ny[id] = n.y;
				nz[id] = n.z;

				
			}

			zDiagonalMatrix Nx = nx.asDiagonal();
			zDiagonalMatrix Ny = ny.asDiagonal();
			zDiagonalMatrix Nz = nz.asDiagonal();


			// primal _ edgefaceMatrix
			zSparseMatrix C_ef;
			getPrimal_EdgeFaceMatrix(zForceDiagram, C_ef);			

			//cout << "\n C_ef " <<  endl  << C_ef << endl;

			// compute A

			out = MatrixXd( 3 *primal_n_e, primal_n_f_i);
			
			

			for (int i = 0; i < 3; i++)
			{
				MatrixXd temp(primal_n_e, primal_n_f);
				
				if(i == 0) temp = C_ef * Nx;
				if (i == 1) temp = C_ef * Ny;
				if (i == 2) temp = C_ef * Nz;

				for (int j = 0; j < temp.rows(); j++)
				{
					for (int k = 0; k < temp.cols(); k++)
					{
						out(i*primal_n_e + j,k) = temp(j,k);

						//out(i*primal_n_e + j, k) = round(out(i*primal_n_e + j, k) *factor) / factor;

						//if (out(i*primal_n_e + j, k) >0 &&  out(i*primal_n_e + j, k) < 0.000001) out(i*primal_n_e + j, k) = 0;
						//if (out(i*primal_n_e + j, k) < 0 && out(i*primal_n_e + j, k) > -0.000001) out(i*primal_n_e + j, k) = 0;
					}
				}
			}


		}

		void compute_forceDensities(int precisionFac = 3)
		{
			double threshold = 1.0 / pow(10, precisionFac);

			MatrixXd A;
			get_EquilibriumMatrix(zForceDiagram, A);

			cout << "\n A " << endl << A << endl;

			FullPivLU<MatrixXd> lu_decomp(A);

			lu_decomp.setThreshold(threshold);
			int rank = lu_decomp.rank();
			cout << "\n A RANK:  " << rank << endl;

			MatrixXd upper = lu_decomp.matrixLU().triangularView<Eigen::Upper>();
			cout << "\n upper:  " << endl << upper << endl;

			MatrixXd kernel = lu_decomp.kernel();
			cout << "Here is a matrix whose columns form a basis of the null-space of A:\n"
				<< kernel << endl;


			MatrixXd B(rank, kernel.cols());

		}


		//--------------------------
		//--- SET METHODS 
		//--------------------------

		
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
		*	\param		[in]	weightDomain				- weight domain of the edge.
		*	\since version 0.0.2
		*/
		 void setFormEdgeWeightsfromForce( zDomainDouble weightDomain = zDomainDouble(2.0, 10.0))
		{
			//compute edgeWeights
			vector<vector<double>> volMesh_fAreas;

			zDomainDouble areaDomain(10000, -100000);
			
			zDomainColor colDomain(zColor(0.784, 0, 0.157, 1), zColor(0.027, 0, 0.157, 1));
	

			for (int i = 0; i < fnForces.size(); i++)
			{
				vector<double> fAreas;
				fnForces[i].getPlanarFaceAreas(fAreas);

				double temp_MinArea = coreUtils.zMin(fAreas);
				areaDomain.min = (temp_MinArea < areaDomain.min) ? temp_MinArea : areaDomain.min;

				double temp_maxArea = coreUtils.zMax(fAreas);
				areaDomain.max = (temp_maxArea > areaDomain.max) ? temp_maxArea : areaDomain.max;

				volMesh_fAreas.push_back(fAreas);
			}

			for(zItGraphVertex v(*formObj);!v.end(); v.next())			
			{
				int i = v.getId();
				if (formGraphVertex_forceVolumeFace[i * 2] == -1 && formGraphVertex_forceVolumeFace[(i * 2) + 1] == -1) continue;

				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				int volID = formGraphVertex_forceVolumeMesh[i * 2];
				int faceID = formGraphVertex_forceVolumeFace[i * 2];

				double fArea = volMesh_fAreas[volID][faceID];

				for (int j = 0; j < cEdges.size(); j++)
				{
					double val = coreUtils.ofMap(fArea, areaDomain, weightDomain);

					zColor col = coreUtils.blendColor(fArea, areaDomain, colDomain, zRGB);
				
					cEdges[j].getEdge().setEdgeWeight(val);
					cEdges[j].getEdge().setEdgeColor(col);
				}

			}
		}

		
		 	

	protected:
		
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
			int n_e = fnForces[forceIndex].numHalfEdges();
			int n_f = fnForces[forceIndex].numPolygons();



			zVector volCenter =	fnForces[forceIndex].getCenter();


			vector<zVector> fCenters;
			fnForces[forceIndex].getCenters(zFaceData, fCenters);

			for( zItMeshEdge e(*forceObjs[forceIndex]);!e.end(); e.next())			
			{
				vector<int> eFaces;
				e.getFaces(eFaces);
				

				zVector pos0 =	e.getHalfEdge(1).getVertex().getVertexPosition();
				zVector pos1 = e.getHalfEdge(0).getVertex().getVertexPosition();

				zVector* formPositions = fnForm.getRawVertexPositions();

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
						zVector fCenter_graphPos = formPositions[vId_fCenter];

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
					zVector vCenter_graphPos = formPositions[vId_vCenter];

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
				getPolytopalRulingRemesh(forceIndex, tempObj, subdivs);
			}

			

		}

		/*! \brief This method remeshes the input mesh to have rulings in ony one direction.
		*
		*	\param		[in]	inFnMesh				- input mesh function set.
		*	\param		[in]	SUBDIVS					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopalRulingRemesh(int index, zObjMesh &inMeshObj, int SUBDIVS)
		{
			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			zFnMesh inFnMesh(inMeshObj);
			int n_v_lowPoly = inFnMesh.numVertices();

			inFnMesh.smoothMesh(SUBDIVS);

			int n_v = inFnMesh.numVertices();
			int n_e = inFnMesh.numHalfEdges();
			int n_f = inFnMesh.numPolygons();

			for (int i = 0; i < n_v_lowPoly; i += 6)
			{
				zItMeshVertex vert0(inMeshObj,i);
				zItMeshVertex vert1(inMeshObj, i+1);				
				zItMeshHalfEdge edge0, edge1;

				vector<zItMeshHalfEdge> cEdges0;
				vert0.getConnectedHalfEdges(cEdges0);

				for (auto &he : cEdges0)
				{
					if (!he.onBoundary())
					{
						edge0 = he.getSym();
					}
				}

				vector<zItMeshHalfEdge> cEdges1;
				vert1.getConnectedHalfEdges(cEdges1);

				for (auto &he : cEdges1)
				{
					if (he.onBoundary())
					{
						edge1 = he;
					}
				}

				zVector v0 = vert0.getVertexPosition();

				zVector v1 = vert1.getVertexPosition();

				positions.push_back(v0);
				positions.push_back(v1);

				//while (smoothPolytopalMesh.edges[edge0].getVertex()->getVertexId() != i + 2)
				for (int k = 0; k < pow(2, (SUBDIVS + 1)); k++)
				{
					int numVerts = positions.size();					

					zVector v2 = edge0.getStartVertex().getVertexPosition(); ;

					zVector v3 = edge1.getVertex().getVertexPosition();

					positions.push_back(v2);
					positions.push_back(v3);

					polyConnects.push_back(numVerts - 2);
					polyConnects.push_back(numVerts);
					polyConnects.push_back(numVerts + 1);
					polyConnects.push_back(numVerts - 1);
					polyCounts.push_back(4);

					//vert0 = vert2;
					//vert1 = vert3;

					edge0 = edge0.getPrev();
					edge1 = edge1.getNext();
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
		*	\since version 0.0.2
		*/
		bool computeRulingIntersection(int polytopalIndex, zItMeshVertex &v0, zItMeshVertex &v1, zVector &closestPt)
		{
			bool out = false;

			zItMeshHalfEdge e0;
			zItMeshHalfEdge e1;

			bool e0HasEdge = false;
			bool e1HasEdge = false;

			vector<zItMeshHalfEdge> cEdges0;
			v0.getConnectedHalfEdges(cEdges0);
			
			if (cEdges0.size() == 3)
			{
				for (auto &he : cEdges0)
				{
					if (!he.onBoundary())
					{
						e0 = he;
						e0HasEdge = true;
						break;
					}
				}
			}

			vector<zItMeshHalfEdge> cEdges1;
			v1.getConnectedHalfEdges(cEdges1);
			if (cEdges1.size() == 3)
			{
				for (auto &he : cEdges1)
				{
					if (!he.onBoundary())
					{
						e1 = he;
						e1HasEdge = true;
						break;
					}
				}
			}

			if (e0HasEdge  && e1HasEdge)
			{
				zItMeshVertex v2 = v0;
				(v0.getId() % 2 == 0) ? v2.next() : v2.prev();

				zItMeshVertex v3 = v1;
				(v1.getId() % 2 == 0) ? v3.next() : v3.prev();



				zVector a0 = v2.getVertexPosition();

				zVector a1 = v0.getVertexPosition();

				zVector b0 = v3.getVertexPosition();

				zVector b1 = v1.getVertexPosition();

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
		*	\since version 0.0.2
		*/
		void getClosePolytopalMesh(int forceIndex)
		{
			if (smoothSubDivs == 0) return;

			int n_v = fnForces[forceIndex].numVertices();
			int n_e = fnForces[forceIndex].numHalfEdges();
			int n_f = fnForces[forceIndex].numPolygons();

			int n_v_smooth = fnPolytopals[forceIndex].numVertices();
			int n_e_smooth = fnPolytopals[forceIndex].numHalfEdges();
			int n_f_smooth = fnPolytopals[forceIndex].numPolygons();

			int numVertsPerStrip = floor(n_v_smooth / (0.5 * n_e));
			int half_NumVertsPerStrip = floor(numVertsPerStrip / 2);


			vector<bool> vertVisited;

			for (int i = 0; i < n_v_smooth; i++)
			{
				vertVisited.push_back(false);
			}

			for(zItMeshEdge e(*forceObjs[forceIndex]); !e.end(); e.next())			
			{
				int eStripId = e.getId();


				//-- Prev  Edge	

				int ePrev = e.getHalfEdge(0).getPrev().getId(); ;
				int ePrevStripId = floor(ePrev / 2);


				if (ePrev % 2 == 0)
				{
					for (int j = 1, k = 0; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip - 2; j += 2, k += 2)
					{
						zItMeshVertex v0 (*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1 (*forceObjs[forceIndex], ePrevStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}

							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}



					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = 0; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip - 2; j -= 2, k += 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], ePrevStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}
					}
				}

				//-- Next Edge		

				int eNext = e.getHalfEdge(0).getNext().getId(); ;
				int eNextStripId = floor(eNext / 2);

				if (eNext % 2 == 0)
				{
					for (int j = 0, k = 1; j < half_NumVertsPerStrip - 2, k < half_NumVertsPerStrip; j += 2, k += 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eNextStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}

					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = 1; j > half_NumVertsPerStrip - 1, k < half_NumVertsPerStrip; j -= 2, k += 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eNextStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}

					}
				}


				//-- SYM Prev  Edge	
						

				int eSymPrev = e.getHalfEdge(1).getPrev().getId();
				int eSymPrevStripId = floor(eSymPrev / 2);


				if (eSymPrev % 2 == 0)
				{
					for (int j = 1, k = numVertsPerStrip - 1; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip; j += 2, k -= 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eSymPrevStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}
					}
				}
				else
				{
					for (int j = numVertsPerStrip - 2, k = numVertsPerStrip - 1; j > half_NumVertsPerStrip - 1, k > half_NumVertsPerStrip; j -= 2, k -= 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eSymPrevStripId * numVertsPerStrip + j);

						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}
					}
				}


				//--SYM Next Edge		
				int eSymNext = e.getHalfEdge(1).getNext().getId();
				int eSymNextStripId = floor(eSymNext / 2);

				if (eSymNext % 2 == 0)
				{
					for (int j = 0, k = numVertsPerStrip - 2; j<half_NumVertsPerStrip - 2, k>half_NumVertsPerStrip - 1; j += 2, k -= 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eSymNextStripId * numVertsPerStrip + j);


						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}
					}
				}
				else
				{
					for (int j = numVertsPerStrip - 1, k = numVertsPerStrip - 2; j > half_NumVertsPerStrip, k > half_NumVertsPerStrip - 1; j -= 2, k -= 2)
					{
						zItMeshVertex v0(*forceObjs[forceIndex], eStripId * numVertsPerStrip + k);
						zItMeshVertex v1(*forceObjs[forceIndex], eSymNextStripId * numVertsPerStrip + j);


						if (!vertVisited[v0.getId()] && !vertVisited[v1.getId()])
						{
							zVector cPt;
							bool intersectChk = computeRulingIntersection(forceIndex, v0, v1, cPt);

							if (intersectChk)
							{
								v0.setVertexPosition(cPt);
								v1.setVertexPosition(cPt);

							}
							vertVisited[v0.getId()] = true;
							vertVisited[v1.getId()] = true;
						}

					}
				}
			}

			for (zItMeshVertex v(*forceObjs[forceIndex]); !v.end(); v.next())			
			{
				vector<int> cEdges;
				v.getConnectedHalfEdges(cEdges);

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
					zItMeshVertex v0(*forceObjs[forceIndex], smoothMeshVerts[j]);
					zItMeshVertex v1(*forceObjs[forceIndex], smoothMeshVerts[(j + 1) % smoothMeshVerts.size()]);

					vertVisited[v0.getId()] = true;

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
					zItMeshVertex v0(*forceObjs[forceIndex], smoothMeshVerts[j]);
					v0.setVertexPosition(avgIntersectPoint);
				}

			}


		}

		/*! \brief This method computes the face centers of the input force volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.2
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
		*	\since version 0.0.2
		*/
		void computeFormTargets()
		{
			targetEdges_form.clear(); 

			for (int i = 0; i < fnForm.numHalfEdges(); i++)
			{
				targetEdges_form.push_back(zVector());
			}

			for (zItGraphVertex v(*formObj); !v.end(); v.next())			
			{
				// get position of vertex
				zVector v_i = v.getVertexPosition();

				int i = v.getId();

				int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
				int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

				if (v.checkVertexValency(1))	continue;
				if (faceId_V != -1 && formGraphVertex_forceVolumeFace[i * 2 + 1] != -1) continue;
			
				// get connected vertices
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges( cEdges);

			
				for (auto &he : cEdges)
				{
					// get vertex 
					int v1_ID = he.getVertex().getId();
					
					zVector v_j = he.getVertex().getVertexPosition();
					zVector e_ij = v_i - v_j;
					e_ij.normalize();

					// get volume and face Id of the connected Vertex
					int volId = formGraphVertex_forceVolumeMesh[v1_ID * 2];
					int faceId = formGraphVertex_forceVolumeFace[v1_ID * 2];				

					zItMeshFace forceFace(*forceObjs[volId], faceId);

					zVector t_ij = forceFace.getFaceNormal();;
					t_ij.normalize();

					if (e_ij * t_ij > 0) t_ij *= -1;

					targetEdges_form[he.getId()] = t_ij;

					int symEdge = he.getSym().getId();
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
		bool checkParallelity(zDomainDouble & deviation, double angleTolerance, bool colorEdges, bool printInfo)
		{
			bool out = true;
			vector<double> deviations;
			deviation = zDomainDouble(10000, -10000);			

			for (zItGraphEdge e(*formObj); !e.end(); e.next())			
			{
				//form edge
				int eId_form = e.getHalfEdge(0).getId();
				zVector e_form = e.getHalfEdge(0).getHalfEdgeVector();
				e_form.normalize();

				zVector e_target = targetEdges_form[eId_form];

				double a_i = e_form.angle(e_target);
				
				deviations.push_back(a_i);
			

				if (a_i > angleTolerance)
				{
					out = false;					
				}


				if (a_i < deviation.min) deviation.min = a_i;
				if (a_i > deviation.max) deviation.max = a_i;
			}


			if (printInfo)
			{
				printf("\n  tolerance : %1.4f minDeviation : %1.4f , maxDeviation: %1.4f ", angleTolerance, deviation.min, deviation.max);
			}

			if (colorEdges)
			{
				zDomainColor colDomain(zColor(180, 1, 1), zColor(0, 1, 1));

				for (zItGraphEdge e(*formObj); !e.end(); e.next())
				{
					
					zColor col = coreUtils.blendColor(deviations[e.getId()], deviation, colDomain, zHSV);

					if (deviations[e.getId()] < angleTolerance) col = zColor();

					e.setEdgeColor(col);				

				}

			}

			return out;
		}

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	minmax_Edge					- minimum value of the target edge as a percentage of maximum target edge.
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	numIterations				- number of iterations.
		*	\since version 0.0.2
		*/
		void updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000)
		{

			zVector* pos = fnForm.getRawVertexPositions();

			if (fnFormParticles.size() != fnForm.numVertices())
			{
				fnFormParticles.clear();
				formParticlesObj.clear();



				for (zItGraphVertex v(*formObj); !v.end(); v.next())
				{
					bool fixed = false;

					int i = v.getId();

					int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
					int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

					if (faceId_V != -1 && formGraphVertex_forceVolumeFace[i * 2 + 1] != -1)
					{
						fixed = true;
						v.setVertexColor(zColor());
					}					

					zObjParticle p;
					p.particle = zParticle(pos[i], fixed);
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

			

			for (int k = 0; k < numIterations; k++)
			{
				// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
				for (zItGraphVertex v(*formObj); !v.end(); v.next())
				{
					int i = v.getId();

					if (fnFormParticles[i].getFixed()) continue;

					// get position of vertex
					zVector v_i = pos[i];

					int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
					int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

					// get connected vertices
					vector<zItGraphHalfEdge> cEdges;
					v.getConnectedHalfEdges(cEdges);

					// compute barycenter per vertex
					zVector b_i;
					for (auto &he: cEdges)
					{
						// get vertex 
						int v1_ID = he.getVertex().getId();

						zVector v_j = pos[v1_ID];

						zVector e_ij = v_i - v_j;
						double len_e_ij = e_ij.length();;

						if (len_e_ij < minEdgeLength) len_e_ij = minEdgeLength;
						if (len_e_ij > maxEdgeLength) len_e_ij = maxEdgeLength;

						int symEdge = he.getSym().getId();

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


				// In plane force for valence 1 vertices
				//for (int i = 0; i < fnFormParticles.size(); i++)
				//{
				//	if (fnForm.checkVertexValency(i, 1))
				//	{
				//		// add force to keep point in the plane of the face

				//		int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
				//		int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

				//		zVector fNorm = fnForces[volId_V].getFaceNormal(faceId_V);
				//		fNorm.normalize();

				//		zVector V = fnForm.getVertexPosition(i);

				//		zVector e = V - force_fCenters[volId_V][faceId_V];

				//		double minDist = coreUtils.minDist_Point_Plane(V, force_fCenters[volId_V][faceId_V], fNorm);

				//		if (minDist > 0)
				//		{
				//			if (e * fNorm >= 0)	fNorm *= -1;


				//			zVector forceV = (fNorm * minDist);
				//			fnFormParticles[i].addForce(forceV);
				//		}

				//		//forceV += (e * -1);

				//	}
				//}


				// update Particles
				for (int i = 0; i < fnFormParticles.size(); i++)
				{
					fnFormParticles[i].integrateForces(dT, type);
					fnFormParticles[i].updateParticle(true);
				}

				// update fixed particle positions ( ones with a common face) 
				for (zItGraphVertex v(*formObj); !v.end(); v.next())
				{
					int i = v.getId();

					if (!fnFormParticles[i].getFixed()) continue;

					vector<zItGraphVertex> cVerts;
					v.getConnectedVertices(cVerts);


					int volId_V = formGraphVertex_forceVolumeMesh[i * 2];
					int faceId_V = formGraphVertex_forceVolumeFace[i * 2];

					zItMeshFace fForce(*forceObjs[volId_V], faceId_V);

					zVector normF1 = fForce.getFaceNormal();

					zVector currentPos = v.getVertexPosition();

					if (cVerts.size() == 2)
					{
						zVector p1 = cVerts[0].getVertexPosition();

						zVector p2 = cVerts[1].getVertexPosition();

						zVector interPt;

						zVector newPos = (p1 + p2) *0.5;

						v.setVertexPosition(newPos);

						/*bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

						if (chkIntersection)
						{
							fnForm.setVertexPosition(i,interPt);
						}*/
					}

				}

			}

			

		}
		
		
	};

}

