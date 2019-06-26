#pragma once

#include <headers/api/object/zObj.h>

#include <headers/framework/geometry/zMesh.h>

namespace zSpace
{
	/** \addtogroup API
	*	\brief The Application Program Interface of the library.
	*  @{
	*/

	/** \addtogroup zObjects
	*	\brief The object classes of the library.
	*  @{
	*/

	/*! \class zObjMesh
	*	\brief The mesh object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/
	
	class zObjMesh : public zObj
	{
	protected:
		/*! \brief boolean for displaying the mesh vertices */
		bool showVertices;

		/*! \brief boolean for displaying the mesh edges */
		bool showEdges;

		/*! \brief boolean for displaying the mesh faces */
		bool showFaces;

		/*! \brief boolean for displaying the mesh dihedral edges */
		bool showDihedralEdges;

		/*! \brief boolean for displaying the mesh vertex normals */
		bool showVertexNormals;

		/*! \brief boolean for displaying the mesh face normals */
		bool showFaceNormals;

		/*! \brief container for storing dihderal angles */
		vector<double> edge_dihedralAngles;

		/*! \brief container for storing face centers */
		vector<zVector> faceCenters;

		/*! \brief dihedral angle threshold */
		double dihedralAngleThreshold;

		/*! \brief normals display scale */
		double normalScale;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief mesh */
		zMesh mesh;

	

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------
		
		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zObjMesh()
		{
			displayUtils = nullptr;

			showVertices = false;
			showEdges = true;
			showFaces = true;

			showDihedralEdges = false;
			showVertexNormals = false;
			showFaceNormals = false;

			dihedralAngleThreshold = 45;

			normalScale = 1.0;
		}	


		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjMesh() {}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show vertices, edges and face booleans.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\param		[in]	_showFaces				- input show faces booelan.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showVerts, bool _showEdges, bool _showFaces)
		{
			showVertices = _showVerts;
			showEdges = _showEdges;
			showFaces = _showFaces;
		}

		/*! \brief This method sets show vertices boolean.
		*
		*	\param		[in]	_showVerts				- input show vertices booelan.
		*	\since version 0.0.2
		*/
		void setShowVertices(bool _showVerts)
		{
			showVertices = _showVerts;
		}

		/*! \brief This method sets show edges boolean.
		*
		*	\param		[in]	_showEdges				- input show edges booelan.
		*	\since version 0.0.2
		*/
		void setShowEdges(bool _showEdges)
		{
			showEdges = _showEdges;
		}


		/*! \brief This method sets show faces boolean.
		*
		*	\param		[in]	_showFaces				- input show faces booelan.
		*	\since version 0.0.2
		*/
		void setShowFaces(bool _showFaces)
		{
			showFaces = _showFaces;
		}

		/*! \brief This method sets show dihedral edges boolean.
		*
		*	\param		[in]	_showDihedralEdges			- input show faces booelan.
		*	\param		[in]	_angles						- input container of edge dihedral angles.
		*	\param		[in]	_threshold					- input angle threshold.
		*	\since version 0.0.2
		*/
		void setShowDihedralEdges(bool _showDihedralEdges, vector<double> &_angles, double _threshold)
		{
			edge_dihedralAngles = _angles;

			showDihedralEdges = _showDihedralEdges;

			dihedralAngleThreshold = _threshold;

			if(_showDihedralEdges) showEdges = false;			
		}

		/*! \brief This method sets show vertex normals boolean.
		*
		*	\param		[in]	_showVertexNormal			- input show vertex normals booelan.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.2
		*/
		void setShowVertexNormals(bool _showVertexNormal, double _normalScale)
		{
			showVertexNormals = _showVertexNormal;

			normalScale = _normalScale;
		}

		/*! \brief This method sets show face normals boolean.
		*
		*	\param		[in]	_showFaceNormal				- input show face normals booelan.
		*	\param		[in]	_faceCenters				- input container of face centers.
		*	\param		[in]	_normalScale				- input scale of normals.
		*	\since version 0.0.2
		*/
		void setShowFaceNormals(bool _showFaceNormal, vector<zVector> &_faceCenters, double _normalScale)
		{
			showFaceNormals = _showFaceNormal;

			normalScale = _normalScale;

			faceCenters = _faceCenters;
		}

		//--------------------------
		//---- GET METHODS
		//--------------------------
		
		/*! \brief This method gets the vertex VBO Index .
		*
		*	\return			int				- vertex VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexID()
		{
			return mesh.VBO_VertexId;
		}

		/*! \brief This method gets the edge VBO Index .
		*
		*	\return			int				- edge VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_EdgeID()
		{
			return mesh.VBO_EdgeId;
		}

		/*! \brief This method gets the face VBO Index .
		*
		*	\return			int				- face VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_FaceID()
		{
			return mesh.VBO_FaceId;
		}

		/*! \brief This method gets the vertex color VBO Index .
		*
		*	\return			int				- vertex color VBO Index.
		*	\since version 0.0.2
		*/
		int getVBO_VertexColorID()
		{
			return mesh.VBO_VertexColorId;
		}

		

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			if (showObject)
			{
				drawMesh();

				if (showDihedralEdges) drawMesh_DihedralEdges();

				if (showVertexNormals) drawMesh_VertexNormals();

				if (showFaceNormals) drawMesh_FaceNormals();
			}

			if (showObjectTransform)
			{
				displayUtils->drawTransform(transformationMatrix);
			}
			
		}

		void getBounds(zVector &minBB, zVector &maxBB) override
		{
			coreUtils->getBounds(mesh.vertexPositions, minBB, maxBB);
		}

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.2
		*/
		void drawMesh()
		{

			//draw vertex
			if (showVertices)
			{

				displayUtils->drawVertices(mesh.vHandles, &mesh.vertexPositions[0], &mesh.vertexColors[0], &mesh.vertexWeights[0]);
							
			}


			//draw edges
			if (showEdges)
			{
				if (mesh.staticGeometry)
				{
					displayUtils->drawEdges(mesh.eHandles, mesh.edgeVertices, &mesh.vertexPositions[0], &mesh.edgeColors[0], &mesh.edgeWeights[0]);
				}

				else
				{
					vector<vector<int>> edgeVertices;
					edgeVertices.assign(mesh.edges.size(), vector<int>(2) = { -1,-1 });

					for (auto &e : mesh.edges)
					{

						if (mesh.eHandles[e.getId()].id != -1)
						{
							vector<int> eVerts;

							edgeVertices[e.getId()][0] = e.getHalfEdge(0)->getVertex()->getId();
							edgeVertices[e.getId()][1] = e.getHalfEdge(1)->getVertex()->getId();
						}
						
						
					}

					displayUtils->drawEdges(mesh.eHandles, edgeVertices, &mesh.vertexPositions[0], &mesh.edgeColors[0], &mesh.edgeWeights[0]);

				}
				
			}


			//draw polygon
			if (showFaces)
			{
				
				if (mesh.staticGeometry)
				{
					displayUtils->drawFaces(mesh.fHandles, mesh.faceVertices, &mesh.vertexPositions[0], &mesh.faceColors[0]);
				}
				else
				{
					vector<vector<int>> faceVertices;

					for (int i = 0; i < mesh.n_f; i++)
					{
						vector<int> faceVerts;
						if (mesh.fHandles[i].id != -1)
						{							
							mesh.getFaceVertices(i, faceVerts);							
						}

						faceVertices.push_back(faceVerts);

					}

					displayUtils->drawFaces(mesh.fHandles, faceVertices, &mesh.vertexPositions[0], &mesh.faceColors[0]);
					
				}
			}
		}

		/*! \brief This method displays the dihedral edges of a mesh above the input angle threshold.
		*
		*	\since version 0.0.2
		*/
		inline void drawMesh_DihedralEdges()
		{
			for (auto &e : mesh.edges)
			{
				int i = e.getId() * 2;

				if (e.isActive())
				{

					if (abs(edge_dihedralAngles[i]) > dihedralAngleThreshold)
					{
						zColor col;
						double wt = 1;

						if (mesh.edgeColors.size() > i)  col = mesh.edgeColors[i];
						if (mesh.edgeWeights.size() > i) wt = mesh.edgeWeights[i];

						int v1 = e.getHalfEdge(0)->getVertex()->getId();
						int v2 = e.getHalfEdge(1)->getVertex()->getId();

						displayUtils->drawLine(mesh.vertexPositions[v1], mesh.vertexPositions[v2], col, wt);
					}
				}
			}
		}

		/*! \brief This method displays the vertex normals of a mesh.
		*
		*	\param		[in]	dispScale				- display scale of the normal.
		*	\since version 0.0.2
		*/
		void drawMesh_VertexNormals()
		{

			if (mesh.vertexNormals.size() == 0 || mesh.vertexNormals.size() != mesh.vertices.size()) throw std::invalid_argument(" error: mesh normals not computed.");

			for (auto &v : mesh.vertices)
			{
				int i = v.getId();
				if (v.isActive())
				{
					zVector p1 = mesh.vertexPositions[i];
					zVector p2 = p1 + (mesh.faceNormals[i] * normalScale);

					displayUtils->drawLine(p1, p2, zColor(0, 1, 0, 1));
				}

			}

		}

		/*! \brief This method displays the face normals of a mesh.
		*
		*	\since version 0.0.2
		*/
		void drawMesh_FaceNormals()
		{
			if (mesh.faceNormals.size() == 0 || mesh.faceNormals.size() != mesh.faces.size()) throw std::invalid_argument(" error: mesh normals not computed.");

			if (mesh.faces.size() != faceCenters.size()) throw std::invalid_argument(" error: number of face centers not equal to number of faces .");

			for (auto &f : mesh.faces)
			{
				int i = f.getId();

				if (f.isActive())
				{
					zVector p1 = faceCenters[i];
					zVector p2 = p1 + (mesh.faceNormals[i] * normalScale);

					displayUtils->drawLine(p1, p2, zColor(0, 1, 0, 1));
				}

			}


		}

		//--------------------------
		//---- DISPLAY BUFFER METHODS
		//--------------------------

		/*! \brief This method appends mesh to the buffer.
		*
		*	\param		[in]	edge_dihedralAngles	- input container of edge dihedral angles.
		*	\param		[in]	DihedralEdges		- true if only edges above the dihedral angle threshold are stored.
		*	\param		[in]	angleThreshold		- angle threshold for the edge dihedral angles.
		*	\since version 0.0.1
		*/
		void appendToBuffer(vector<double> edge_dihedralAngles = vector<double>(), bool DihedralEdges = false, double angleThreshold = 45)
		{
			showObject = showEdges = showVertices = showFaces = false;

			// Edge Indicies
			if (!DihedralEdges)
			{
				vector<int> _edgeIndicies;

				for ( auto &e : mesh.edges )
				{
					if (e.isActive())
					{
						_edgeIndicies.push_back(e.getHalfEdge(0)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
						_edgeIndicies.push_back(e.getHalfEdge(1)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
					}
				}

				mesh.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);
			}
			else
			{


				vector<int> _edgeIndicies;

				for (auto &e : mesh.edges)
				{
					
					if (e.isActive())
					{
						if (abs(edge_dihedralAngles[e.getId()]) > angleThreshold || edge_dihedralAngles[e.getId()] == -1)
						{
							_edgeIndicies.push_back(e.getHalfEdge(0)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
							_edgeIndicies.push_back(e.getHalfEdge(1)->getVertex()->getId() + displayUtils->bufferObj.nVertices);
						}
					}

				}

				mesh.VBO_EdgeId = displayUtils->bufferObj.appendEdgeIndices(_edgeIndicies);
			}


			// Face Indicies

			vector<int> _faceIndicies;

			for (auto &f : mesh.faces)
			{
				if (f.isActive())
				{
					vector<int> faceVertsIds;
					mesh.getFaceVertices(f.getId(), faceVertsIds);

					for (int j = 0; j < faceVertsIds.size(); j++)
					{
						_faceIndicies.push_back(faceVertsIds[j] + displayUtils->bufferObj.nVertices);

					}
				}

			}

			mesh.VBO_FaceId = displayUtils->bufferObj.appendFaceIndices(_faceIndicies);

			// Vertex Attributes

			mesh.VBO_VertexId = displayUtils->bufferObj.appendVertexAttributes(&mesh.vertexPositions[0], &mesh.vertexNormals[0], mesh.vertexPositions.size());
			mesh.VBO_VertexColorId = displayUtils->bufferObj.appendVertexColors(&mesh.vertexColors[0], mesh.vertexColors.size());


		}

	};




}

