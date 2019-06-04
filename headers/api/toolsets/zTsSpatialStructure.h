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

	/*! \class zTsSpatialStructures
	*	\brief A toolset for creating spatial strctures from volume meshes.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	/** @}*/


	class zTsSpatialStructures
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		//--------------------------
		//---- CELL ATTRIBUTES
		//--------------------------

		/*!	\brief pointer container to volume Object  */
		vector<zObjMesh*> volumeObjs;

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
		vector<zFnMesh> fnVolumes;

		/*!	\brief container of polytopals function set  */
		vector <zFnMesh> fnPolytopals;

		/*!	\brief force volume mesh faceID to form graph vertexID map.  */
		unordered_map <string, int> volumeFace_formGraphVertex;

		/*!	\brief form graph vertexID to volume meshID map.  */
		vector<int> formGraphVertex_volumeMesh;

		/*!	\brief form graph vertexID to local volume mesh faceID map.*/
		vector<int> formGraphVertex_volumeFace;

		/*!	\brief graph vertex offset container.*/
		vector<double> formGraphVertex_Offsets;

		/*!	\brief container of facecenter per force volume  */
		vector<vector<zVector>> volume_fCenters;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsSpatialStructures() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_formObj			- input form object.
		*	\param		[in]	_volumeObjs			- input container of volume objects.
		*	\param		[in]	_polytopalObjs		- input container of polytopal objects.
		*	\since version 0.0.2
		*/
		zTsSpatialStructures(zObjGraph &_formObj, vector<zObjMesh> &_volumeObjs, vector<zObjMesh>  &_polytopalObjs)
		{
			formObj = &_formObj;
			fnForm = zFnGraph(_formObj);

			for (int i = 0; i < _volumeObjs.size(); i++)
			{
				volumeObjs.push_back(&_volumeObjs[i]);
				fnVolumes.push_back(_volumeObjs[i]);

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
		~zTsSpatialStructures() {}

		//--------------------------
		//---- CREATE METHODS
		//--------------------------
				

		/*! \brief This method creates the volume geometry from the input files.
		*
		*	\param [in]		directory		- input directory path.
		*	\param [in]		filename		- input filename.
		*	\param [in]		numFiles		- input number of files in the directory.
		*	\param [in]		type			- input file type.
		*	\since version 0.0.1
		*/
		void createVolumeFromFile(string directory, string filename, int numFiles, zFileTpye type)
		{

			if (type == zJSON)
			{

				for (int i = 0; i < numFiles; i++)
				{
					string path = directory + "/" + filename + "_" + to_string(i) + ".json";

					if (i < fnVolumes.size()) fnVolumes[i].from(path, type);
				}



			}

			else if (type == zOBJ)
			{
				for (int i = 0; i < numFiles; i++)
				{
					string path = directory + "/" + filename + "_" + to_string(i) + ".obj";

					if (i < fnVolumes.size()) fnVolumes[i].from(path, type);
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
		void createFormFromVolume(double offset, int precisionFac = 3, zColor edgeCol = zColor(0.75, 0.5, 0, 1))
		{

			vector<zVector>positions;
			vector<int>edgeConnects;
			unordered_map <string, int> positionVertex;


			volumeFace_formGraphVertex.clear();

			formGraphVertex_volumeMesh.clear();

			formGraphVertex_volumeFace.clear();

			

			for (int j = 0; j < fnVolumes.size(); j++)
			{

				zColor* faceColors = fnVolumes[j].getRawFaceColors();

				int n_v = fnVolumes[j].numVertices();
				int n_e = fnVolumes[j].numHalfEdges();
				int n_f = fnVolumes[j].numPolygons();

				zVector volCenter = fnVolumes[j].getCenter();

				int vId_volCenter = -1;
				bool chkExists = coreUtils.vertexExists(positionVertex, volCenter, precisionFac, vId_volCenter);


				if (!chkExists)
				{
					coreUtils.addToPositionMap(positionVertex, volCenter, positions.size(), precisionFac);

					vId_volCenter = positions.size();
					positions.push_back(volCenter);

					formGraphVertex_volumeFace.push_back(-1);
					formGraphVertex_volumeFace.push_back(-1);

					formGraphVertex_volumeMesh.push_back(j);
					formGraphVertex_volumeMesh.push_back(-1);
				}

				string hashKey_volface = (to_string(j) + "," + to_string(-1));
				volumeFace_formGraphVertex[hashKey_volface] = vId_volCenter;

				vector<zVector> fCenters;
				fnVolumes[j].getCenters(zFaceData, fCenters);

				for (int i = 0; i < fCenters.size(); i++)
				{
					if (faceColors[i].r != 1.0)
					{
						int vId_fCenter = -2;
						string hashKey_volface = (to_string(j) + "," + to_string(i));
						volumeFace_formGraphVertex[hashKey_volface] = vId_fCenter;

					}

					else
					{
						int vId_fCenter = -1;
						bool chkExists_f = coreUtils.vertexExists(positionVertex, fCenters[i], precisionFac, vId_fCenter);

						if (!chkExists_f)
						{
							coreUtils.addToPositionMap(positionVertex, fCenters[i], positions.size(), precisionFac);

							vId_fCenter = positions.size();
							positions.push_back(fCenters[i]);

							formGraphVertex_volumeFace.push_back(i);
							formGraphVertex_volumeFace.push_back(-1);

							formGraphVertex_volumeMesh.push_back(j);
							formGraphVertex_volumeMesh.push_back(-2);
						}
						else
						{
							formGraphVertex_volumeFace[(vId_fCenter * 2) + 1] = i;
							formGraphVertex_volumeMesh[(vId_fCenter * 2) + 1] = j;
						}

						string hashKey_volface = (to_string(j) + "," + to_string(i));
						volumeFace_formGraphVertex[hashKey_volface] = vId_fCenter;

						edgeConnects.push_back(vId_volCenter);
						edgeConnects.push_back(vId_fCenter);
					}

					
				}
			}

			fnForm.clear();

			fnForm.create(positions, edgeConnects);

			setVertexOffset(offset);
			fnForm.setEdgeColor(edgeCol);


			// compute intersection point
			for (zItGraphVertex v(*formObj); !v.end(); v.next())
			{
				int i = v.getId();

				vector<zItGraphVertex> cVerts;
				v.getConnectedVertices(cVerts);


				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				

				if (cVerts.size() == 2)
				{
					zItMeshFace volume_face(*volumeObjs[volId_V], faceId_V);

					zVector normF1 = volume_face.getFaceNormal();
					zVector currentPos = v.getVertexPosition();

					zVector p1 = cVerts[0].getVertexPosition();
					zVector p2 = cVerts[1].getVertexPosition();

					zVector interPt;
					bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						v.setVertexPosition(interPt);

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

		/*! \brief This method creates the polytopal mesh from the volume meshes and form graph.
		*
		*	\since version 0.0.2
		*/
		void createPolytopalsFromVolume()
		{

			for (int i = 0; i < fnVolumes.size(); i++)
			{
				fnPolytopals[i].clear();
				getPolytopal(i);
			}
		}
		
	

		//--------------------------
		//----UPDATE METHOD
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
		bool equilibrium(bool computeTargets, double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000, double angleTolerance = 0.001, bool colorEdges = false, bool printInfo = false)
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

		/*! \brief This method computes the form graph edge weights based on the volume mesh face areas.
		*
		*	\param		[in]	weightDomain				- weight domain of the edge.		
		*	\since version 0.0.1
		*/
		void setFormEdgeWeightsfromVolume(zDomainDouble weightDomain = zDomainDouble(2.0, 10.0))
		{
			//compute edgeWeights
			vector<vector<double>> volMesh_fAreas;

			zDomainDouble areaDomain(10000, -100000);
			zDomainColor colDomain(zColor(0.784, 0, 0.157, 1), zColor(0.027, 0, 0.157, 1));
		
			for (int i = 0; i < fnVolumes.size(); i++)
			{
				vector<double> fAreas;
				fnVolumes[i].getPlanarFaceAreas(fAreas);

				double temp_MinArea = coreUtils.zMin(fAreas);
				areaDomain.min = (temp_MinArea < areaDomain.min) ? temp_MinArea : areaDomain.min;

				double temp_maxArea = coreUtils.zMax(fAreas);
				areaDomain.max = (temp_maxArea > areaDomain.max) ? temp_maxArea : areaDomain.max;

				volMesh_fAreas.push_back(fAreas);
			}

			for (zItGraphVertex v(*formObj); !v.end(); v.next())
			{
				int i = v.getId();
				if (formGraphVertex_volumeFace[i * 2] == -1 && formGraphVertex_volumeFace[(i * 2) + 1] == -1) continue;

				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);

				int volID = formGraphVertex_volumeMesh[i * 2];
				int faceID = formGraphVertex_volumeFace[i * 2];

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
		

		void extrudeConnectionFaces(int volumeIndex)
		{
			

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			vector<zVector> fCenters;
			fnVolumes[volumeIndex].getCenters( zFaceData, fCenters);

			fnVolumes[volumeIndex].getVertexPositions(positions);

			

			for(zItMeshFace f(*volumeObjs[volumeIndex]); !f.end(); f.next())			
			{
				double faceVal = f.getFaceColor().r;				

				vector<int> fVerts;
				f.getVertices(fVerts);

				if (faceVal != 1.0)
				{
					for (int j = 0; j < fVerts.size(); j++) polyConnects.push_back(fVerts[j]);

					polyCounts.push_back(fVerts.size());
				}
				else
				{
					vector<zVector> fVertPositions;
					f.getVertexPositions(fVertPositions);

					string hashKey_f = (to_string(volumeIndex) + "," + to_string(f.getId()));
					int vId_fCenter = -1;
					bool chkExists_f = coreUtils.existsInMap(hashKey_f, volumeFace_formGraphVertex, vId_fCenter);

					zItGraphVertex vForm(*formObj, vId_fCenter);

					double boundaryOffset = formGraphVertex_Offsets[vId_fCenter];

					// get current size of positions
					int numVerts = positions.size();

					// append new positions
					for (int j = 0; j < fVertPositions.size(); j++)
					{
						zVector dir = fCenters[f.getId()] 	 - fVertPositions[j];
						double len = dir.length();
						dir.normalize();

						zVector newPos = fVertPositions[j] + dir * len * boundaryOffset;

						newPos += vForm.getVertexPosition() - fCenters[f.getId()];

						positions.push_back(newPos);
					}

					// compute polyconnects and polycounts
					for (int j = 0; j < fVerts.size(); j++)
					{
						int currentId = j;
						int nextId = (j + 1) % fVerts.size();

						polyConnects.push_back(fVerts[currentId]);
						polyConnects.push_back(fVerts[nextId]);
						polyConnects.push_back(numVerts + nextId);
						polyConnects.push_back(numVerts + currentId);

						polyCounts.push_back(4);

					}

				}



			}

			
			if (positions.size() > 0)
			{
				fnPolytopals[volumeIndex].create(positions, polyCounts, polyConnects);				
			}

			

		}


		/*! \brief This method creates the polytopal mesh based on the input force volume mesh and form graph.
		*
		*	\param		[in]	forceIndex				- input force volume mesh index.
		*	\param		[in]	subdivs					- input number of subdivisions.
		*	\since version 0.0.2
		*/
		void getPolytopal(int volumeIndex)
		{
			if (volumeIndex > fnVolumes.size()) throw std::invalid_argument(" error: index out of bounds.");

			

			string hashKey_volcenter = (to_string(volumeIndex) + "," + to_string(-1));
			int vId_volCenter = -1;
			bool chkExists_f = coreUtils.existsInMap(hashKey_volcenter, volumeFace_formGraphVertex, vId_volCenter);

			zVector volCenter = fnVolumes[volumeIndex].getCenter();
			
			zItGraphVertex vForm(*formObj, vId_volCenter);

			zVector vCenter_graphPos = vForm.getVertexPosition();
			double volCenterOffset = formGraphVertex_Offsets[vId_volCenter];

			

			extrudeConnectionFaces(volumeIndex);			

			// scale original points from scale center

			for(zItMeshVertex v(*volumeObjs[volumeIndex]); !v.end(); v.next())			
			{
				if (v.onBoundary()) continue;


				zVector dir = volCenter - v.getVertexPosition();
				double len = dir.length();
				dir.normalize();

				zVector newPos = v.getVertexPosition() + dir * len * volCenterOffset;

				newPos += vCenter_graphPos - volCenter;

				v.setVertexPosition( newPos);
			

			}			

		}


		/*! \brief This method computes the face centers of the input volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.1
		*/
		void computeVolumesFaceCenters()
		{
			volume_fCenters.clear();

			for (int i = 0; i < fnVolumes.size(); i++)
			{
				vector<zVector> fCenters;
				fnVolumes[i].getCenters(zFaceData, fCenters);

				volume_fCenters.push_back(fCenters);
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

				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				if (v.checkVertexValency(1))	continue;
				if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] != -1) continue;

				// get connected vertices
				vector<zItGraphHalfEdge> cEdges;
				v.getConnectedHalfEdges(cEdges);


				for (auto &he : cEdges)
				{
					// get vertex 
					int v1_ID = he.getVertex().getId();

					zVector v_j = he.getVertex().getVertexPosition();
					zVector e_ij = v_i - v_j;
					e_ij.normalize();

					// get volume and face Id of the connected Vertex
					int volId = formGraphVertex_volumeMesh[v1_ID * 2];
					int faceId = formGraphVertex_volumeFace[v1_ID * 2];

					zItMeshFace forceFace(*volumeObjs[volId], faceId);

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
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	tolerance					- tolerance for force.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool updateFormDiagram(double minmax_Edge, double dT, zIntergrationType type, int numIterations = 1000)
		{
			bool out = true;

			zVector* pos = fnForm.getRawVertexPositions();

			if (fnFormParticles.size() != fnForm.numVertices())
			{
				fnFormParticles.clear();
				formParticlesObj.clear();


				for (zItGraphVertex v(*formObj); !v.end(); v.next())
				{
					bool fixed = false;

					int i = v.getId();

					int volId_V = formGraphVertex_volumeMesh[i * 2];
					int faceId_V = formGraphVertex_volumeFace[i * 2];

					if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] != -1)
					{
						fixed = true;
						v.setVertexColor(zColor());
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

			if (volume_fCenters.size() != fnVolumes.size()) computeVolumesFaceCenters();

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

					int volId_V = formGraphVertex_volumeMesh[i * 2];
					int faceId_V = formGraphVertex_volumeFace[i * 2];

					// get connected vertices
					vector<zItGraphHalfEdge> cEdges;
					v.getConnectedHalfEdges(cEdges);

					// compute barycenter per vertex
					zVector b_i;
					for (auto &he : cEdges)
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


					int volId_V = formGraphVertex_volumeMesh[i * 2];
					int faceId_V = formGraphVertex_volumeFace[i * 2];

					zItMeshFace fForce(*volumeObjs[volId_V], faceId_V);

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
