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

	/** \addtogroup zTsGeometry
	*	\brief tool sets for geometry related utilities.
	*  @{
	*/

	/*! \class zTsRemesh
	*	\brief A remesh tool set class for remeshing triangular meshes.
	*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class zTsRemesh 
	{
	protected:		

		/*!	\brief core utilities Object  */
		zUtilsCore coreUtils;

		/*!	\brief pointer to form Object  */
		zObjMesh *meshObj;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*!	\brief form function set  */
		zFnMesh fnMesh;

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsRemesh() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsRemesh(zObjMesh &_meshObj)
		{
			meshObj = &_meshObj;
			fnMesh = zFnMesh(_meshObj);
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zTsRemesh() {}

		//--------------------------
		//---- REMESH METHODS
		//--------------------------


		/*! \brief This method splits an edge longer than the given input value at its midpoint and  triangulates the mesh. the adjacent triangles are split into 2-4 triangles.
		*
		*	\param		[in]	maxEdgeLength	- maximum edge length.
		*	\since version 0.0.2
		*/
		void splitLongEdges(double maxEdgeLength)
		{
			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				if (meshObj->mesh.edgeActive[i])
				{
					double eLength = fnMesh.getEdgelength(i);

					while (eLength > maxEdgeLength)
					{
						fnMesh.splitEdge(i, 0.5, true);
						eLength = fnMesh.getEdgelength(i);

					}
				}
			}
		}

		/*! \brief This method collapses an edge shorter than the given minimum edge length value if the collapsing doesnt produce adjacent edges longer than the maximum edge length.
		*
		*	\param		[in]	minEdgeLength		- minimum edge length.
		*	\param		[in]	maxEdgeLength		- maximum edge length.
		*	\since version 0.0.2
		*/
		void collapseShortEdges(double minEdgeLength, double maxEdgeLength)
		{
			int finished = false;

			vector<bool> edgeFinished;

			while (!finished)
			{
				for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
				{
					if (meshObj->mesh.edgeActive[i])
					{

						double eLength = fnMesh.getEdgelength(i);

						if (eLength < minEdgeLength)
						{
							int v1 = meshObj->mesh.edges[i].getVertex()->getVertexId();
							int v2 = meshObj->mesh.edges[i + 1].getVertex()->getVertexId();

							zVector pos = meshObj->mesh.vertexPositions[v2]; /*(meshObj->mesh.vertexPositions[v1] + meshObj->mesh.vertexPositions[v2]) * 0.5;*/

							vector<int> cVertsV1;
							fnMesh.getConnectedVertices(v1, zVertexData, cVertsV1);

							bool collapse_ok = true;

							for (int j = 0; j < cVertsV1.size(); j++)
							{
								if (pos.distanceTo(meshObj->mesh.vertexPositions[cVertsV1[j]]) > maxEdgeLength)
								{
									collapse_ok = false;
									break;
								}
							}

							if (collapse_ok)
							{
								//printf("\n working %i \n", i);
								fnMesh.collapseEdge(i, 0.5, false);

								//printMesh(inMesh);
							}
						}



					}
				}
			}


		}

		/*! \brief This method equalizes the vertex valences by flipping edges of the input triangulated mesh. Target valence for interior vertex is 4 and boundary vertex is 6.
		*
		*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
		*	\since version 0.0.2
		*/
		void equalizeValences()
		{
			for (int i = 0; i < meshObj->mesh.edgeActive.size(); i += 2)
			{
				if (meshObj->mesh.edgeActive[i])
				{
					if (fnMesh.onBoundary(i, zEdgeData) || fnMesh.onBoundary(i + 1, zEdgeData)) continue;

					vector<int> fVerts;
					fnMesh.getVertices(fnMesh.getFaceIndex(i) , zFaceData, fVerts);

					vector<int> sym_fVerts;
					fnMesh.getVertices(fnMesh.getFaceIndex(i+1), zFaceData, sym_fVerts);

					if (fVerts.size() != 3 || sym_fVerts.size() != 3)
					{
						printf("\n Cannot flip edge %i as it not shared by two Triangles.", i);
						continue;
					}

					int v1 = fnMesh.getEndVertexIndex(i);  
					int v2 = fnMesh.getEndVertexIndex(i+1);

					int nextEdge = fnMesh.getNextIndex(i);
					int symNextEdge = fnMesh.getNextIndex(i+1);

					int v3 = fnMesh.getEndVertexIndex(nextEdge); 
					int v4 = fnMesh.getEndVertexIndex(symNextEdge); 

					int tarVal_v1 = (fnMesh.onBoundary(v1, zVertexData)) ? 4 : 6;
					int tarVal_v2 = (fnMesh.onBoundary(v2, zVertexData)) ? 4 : 6;
					int tarVal_v3 = (fnMesh.onBoundary(v3, zVertexData)) ? 4 : 6;
					int tarVal_v4 = (fnMesh.onBoundary(v4, zVertexData)) ? 4 : 6;

					int val_v1 = fnMesh.getVertexValence(v1);
					int val_v2 = fnMesh.getVertexValence(v2);
					int val_v3 = fnMesh.getVertexValence(v3);
					int val_v4 = fnMesh.getVertexValence(v4);

					int dev_pre = abs(val_v1 - tarVal_v1) + abs(val_v2 - tarVal_v2) + abs(val_v3 - tarVal_v3) + abs(val_v4 - tarVal_v4);

					fnMesh.flipTriangleEdge(i);

					val_v1 = fnMesh.getVertexValence(v1);
					val_v2 = fnMesh.getVertexValence(v2);
					val_v3 = fnMesh.getVertexValence(v3);
					val_v4 = fnMesh.getVertexValence(v4);


					int dev_post = abs(val_v1 - tarVal_v1) + abs(val_v2 - tarVal_v2) + abs(val_v3 - tarVal_v3) + abs(val_v4 - tarVal_v4);

					if (dev_pre <= dev_post) fnMesh.flipTriangleEdge(i);
				}
			}
		}

		/*! \brief This method applies an iterative smoothing to the mesh by  moving the vertex but constrained to its tangent plane.
		*
		*	\details Based on http://lgg.epfl.ch/publications/2006/botsch_2006_GMT_eg.pdf page 64 -67
		*	\since version 0.0.2
		*/
		void tangentialRelaxation()
		{
			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				if (meshObj->mesh.vertexActive[i])
				{
					vector<int> cVerts;
					fnMesh.getConnectedVertices(i, zVertexData, cVerts);

					vector<zVector> cVerts_Pos;
					vector<double> weights;

					for (int j = 0; j < cVerts.size(); j++)
					{
						cVerts_Pos.push_back(fnMesh.getVertexPosition(cVerts[j]));
						weights.push_back(1.0);
					}

					zVector v_bary = coreUtils.getBaryCenter(cVerts_Pos, weights);

					zVector v_norm = fnMesh.getVertexNormal(i);
					zVector v_pos = fnMesh.getVertexPosition(i);

					double dotP = v_norm * (v_pos - v_bary);
					zVector newPos = v_bary + (v_norm * dotP);

					fnMesh.setVertexPosition(i, newPos);

				}
			}

		}
	};
}