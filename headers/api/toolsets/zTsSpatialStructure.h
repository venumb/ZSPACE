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

	/*! \class zTsSpatialStrctures
	*	\brief A toolset for creating spatial strctures from volume meshes.
	*	\since version 0.0.2
	*/

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
		*	\param		[in]	_mesh			- input mesh.
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

		/** \addtogroup Polytopal creation
		*	\brief Collection of polytopal creation methods.
		*  @{
		*/

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
				int n_v = fnVolumes[j].numVertices();
				int n_e = fnVolumes[j].numEdges();
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
					if (fnVolumes[j].getFaceColor(i).r != 1.0)
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
			for (int i = 0; i < fnForm.numVertices(); i++)
			{

				vector<int> cVerts;
				fnForm.getConnectedVertices(i, zVertexData, cVerts);


				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				

				if (cVerts.size() == 2)
				{
					zVector normF1 = fnVolumes[volId_V].getFaceNormal(faceId_V);
					zVector currentPos = fnForm.getVertexPosition(i);

					zVector p1 = fnForm.getVertexPosition(cVerts[0]);
					zVector p2 = fnForm.getVertexPosition(cVerts[1]);

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
		
		/** @}*/

		//--------------------------
		//----UPDATE METHOD
		//--------------------------

		/** \addtogroup SpatialStructures Update
		*	\brief Collection of spatial strctures update methods.
		*  @{
		*/

		/*! \brief This method updates the form diagram.
		*
		*	\param		[in]	dT							- integration timestep.
		*	\param		[in]	type						- integration type - zEuler or zRK4.
		*	\param		[in]	tolerance					- tolerance for force.
		*	\return				bool						- true if the all the forces are within tolerance.
		*	\since version 0.0.2
		*/
		bool updateForm(double dT, zIntergrationType type, double tolerance = 0.001)
		{
			bool out = true;

			if (fnFormParticles.size() != fnForm.numVertices())
			{
				fnFormParticles.clear();
				formParticlesObj.clear();


				for (int i = 0; i < fnForm.numVertices(); i++)
				{
					bool fixed = false;


					zObjParticle p;
					p.particle = zParticle(formObj->graph.vertexPositions[i], fixed);
					formParticlesObj.push_back(p);

				}

				for (int i = 0; i < formParticlesObj.size(); i++)
				{
					fnFormParticles.push_back(zFnParticle(formParticlesObj[i]));
				}
			}

			if (volume_fCenters.size() != fnVolumes.size()) computeForcesFaceCenters();

			vector<double> edgeLengths;
			fnForm.getEdgeLengths(edgeLengths);

			// get positions on the graph at volume centers only - 0 to inputVolumemesh size vertices of the graph
			for (int i = 0; i < fnFormParticles.size(); i++)
			{


				if (fnFormParticles[i].getFixed()) continue;

				// get position of vertex
				zVector V = fnForm.getVertexPosition(i);

				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				// get connected vertices
				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				zVector forceV;

				// perpenducalrity force
				for (int j = 0; j < cEdges.size(); j++)
				{
					// get vertex 
					int v1_ID = fnForm.getEndVertexIndex(cEdges[j]);

					// get volume and face Id of the connected Vertex
					int volId = formGraphVertex_volumeMesh[v1_ID * 2];
					int faceId = formGraphVertex_volumeFace[v1_ID * 2];
					if (faceId == -1)
					{
						//printf("\n working");
						volId = volId_V;
						faceId = faceId_V;
					}


					if (fnForm.checkVertexValency(v1_ID, 2)) v1_ID = fnForm.getNext(cEdges[j])->getVertex()->getVertexId();
					zVector V1 = fnForm.getVertexPosition(v1_ID);

					zVector e = V - V1;
					double len = e.length();;
					e.normalize();



					// get face normal
					//printf("\n %i faceId %i numP: %i ", volId, faceId, fnForces[volId].numPolygons());
					zVector fNorm = fnVolumes[volId].getFaceNormal(faceId);
					fNorm.normalize();


					// flipp if edge and face normal are in opposite direction
					if (e*fNorm < 0) fNorm *= -1;

					// get projected position of vertex along face normal 
					zVector projV = V1 + (fNorm * len);

					zVector f = (projV - V) * 0.5;

					forceV += f;


				}


				// keep it in plane force
				// get volume and face Id of the connected Vertex		

				if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] == -1)
				{
					// add force to keep point in the plane of the face

					zVector fNorm = fnVolumes[volId_V].getFaceNormal(faceId_V);
					fNorm.normalize();

					zVector e = V - volume_fCenters[volId_V][faceId_V];

					double minDist = coreUtils.minDist_Point_Plane(V, volume_fCenters[volId_V][faceId_V], fNorm);

					if (minDist > 0)
					{
						if (e * fNorm >= 0)	fNorm *= -1;


						forceV += (fNorm * minDist);
					}

					//forceV += (e * -1);

				}

				// common face vertex between volumne centers
				if (faceId_V != -1 && formGraphVertex_volumeFace[i * 2 + 1] != -1)
				{
					fnFormParticles[i].setFixed(true);
					fnForm.setVertexColor(i, zColor(1, 0, 0, 1));

					forceV = zVector();
				}


				if (forceV.length() > tolerance)
				{
					out = false;
				}

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


				int volId_V = formGraphVertex_volumeMesh[i * 2];
				int faceId_V = formGraphVertex_volumeFace[i * 2];

				zVector normF1 = fnVolumes[volId_V].getFaceNormal(faceId_V);
				zVector currentPos = fnForm.getVertexPosition(i);

				if (cVerts.size() == 2)
				{
					zVector p1 = fnForm.getVertexPosition(cVerts[0]);
					zVector p2 = fnForm.getVertexPosition(cVerts[1]);

					zVector interPt;
					bool chkIntersection = coreUtils.line_PlaneIntersection(p1, p2, normF1, currentPos, interPt);

					if (chkIntersection)
					{
						fnForm.setVertexPosition(i, interPt);
					}
				}

			}

			return out;

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

		/*! \brief This method computes the form graph edge weights based on the volume mesh face areas.
		*
		*	\param		[in]	minWeight					- minimum weight of the edge.
		*	\param		[in]	maxWeight					- maximum weight of the edge.
		*	\since version 0.0.1
		*/
		void setFormEdgeWeightsfromVolume(double minWeight = 2.0, double maxWeight = 10.0)
		{
			//compute edgeWeights
			vector<vector<double>> volMesh_fAreas;

			double minArea = 10000, maxArea = -100000;
			zColor maxCol(0.784, 0, 0.157, 1);
			zColor minCol(0.027, 0, 0.157, 1);

			for (int i = 0; i < fnVolumes.size(); i++)
			{
				vector<double> fAreas;
				fnVolumes[i].getPlanarFaceAreas(fAreas);

				double temp_MinArea = coreUtils.zMin(fAreas);
				minArea = (temp_MinArea < minArea) ? temp_MinArea : minArea;

				double temp_maxArea = coreUtils.zMax(fAreas);
				maxArea = (temp_maxArea > maxArea) ? temp_maxArea : maxArea;

				volMesh_fAreas.push_back(fAreas);
			}

			for (int i = 0; i < fnForm.numVertices(); i++)
			{
				if (formGraphVertex_volumeFace[i * 2] == -1 && formGraphVertex_volumeFace[(i * 2) + 1] == -1) continue;

				vector<int> cEdges;
				fnForm.getConnectedEdges(i, zVertexData, cEdges);

				int volID = formGraphVertex_volumeMesh[i * 2];
				int faceID = formGraphVertex_volumeFace[i * 2];

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

		void extrudeConnectionFaces(int volumeIndex)
		{
			

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			vector<zVector> fCenters;
			fnVolumes[volumeIndex].getCenters( zFaceData, fCenters);

			fnVolumes[volumeIndex].getVertexPositions(positions);


			for (int i = 0; i < fnVolumes[volumeIndex].numPolygons(); i++)
			{
				double faceVal = fnVolumes[volumeIndex].getFaceColor(i).r;				

				vector<int> fVerts;
				fnVolumes[volumeIndex].getVertices(i, zFaceData, fVerts);

				if (faceVal != 1.0)
				{
					for (int j = 0; j < fVerts.size(); j++) polyConnects.push_back(fVerts[j]);

					polyCounts.push_back(fVerts.size());
				}
				else
				{
					vector<zVector> fVertPositions;
					fnVolumes[volumeIndex].getVertexPositions(i, zFaceData, fVertPositions);

					string hashKey_f = (to_string(volumeIndex) + "," + to_string(i));
					int vId_fCenter = -1;
					bool chkExists_f = coreUtils.existsInMap(hashKey_f, volumeFace_formGraphVertex, vId_fCenter);

					double boundaryOffset = formGraphVertex_Offsets[vId_fCenter];

					// get current size of positions
					int numVerts = positions.size();

					// append new positions
					for (int j = 0; j < fVertPositions.size(); j++)
					{
						zVector dir = fCenters[i] 	 - fVertPositions[j];
						double len = dir.length();
						dir.normalize();

						zVector newPos = fVertPositions[j] + dir * len * boundaryOffset;

						newPos += fnForm.getVertexPosition(vId_fCenter) - fCenters[i];

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
			
			zVector vCenter_graphPos = fnForm.getVertexPosition(vId_volCenter);
			double volCenterOffset = formGraphVertex_Offsets[vId_volCenter];

			

			extrudeConnectionFaces(volumeIndex);			

			// scale original points from scale center

			for (int i = 0; i < fnPolytopals[volumeIndex].numVertices(); i++)
			{
				if (fnPolytopals[volumeIndex].onBoundary(i, zVertexData)) continue;


				zVector dir = volCenter - fnPolytopals[volumeIndex].getVertexPosition(i);
				double len = dir.length();
				dir.normalize();

				zVector newPos = fnPolytopals[volumeIndex].getVertexPosition(i) + dir * len * volCenterOffset;

				newPos += vCenter_graphPos - volCenter;

				fnPolytopals[volumeIndex].setVertexPosition(i, newPos);
			

			}			

		}


		/*! \brief This method computes the face centers of the input volume mesh container and stores it in a 2 Dimensional Container.
		*
		*	\since version 0.0.1
		*/
		void computeForcesFaceCenters()
		{
			volume_fCenters.clear();

			for (int i = 0; i < fnVolumes.size(); i++)
			{
				vector<zVector> fCenters;
				fnVolumes[i].getCenters(zFaceData, fCenters);

				volume_fCenters.push_back(fCenters);
			}
		}

		/** @}*/

	};
	
}
