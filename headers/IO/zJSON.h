#pragma once

#include <headers/geometry/zMesh.h>
#include <headers/geometry/zMeshUtilities.h>

#include <depends\modernJSON\json.hpp>
using json = nlohmann::json;;

#include <iostream>
using namespace std;

namespace zSpace
{
	/** \addtogroup zIO
	*	\brief The data transfer classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zIO_JSONClasses
	*	\brief The JSON data transfer classes of the library.
	*  @{
	*/	

	/*! \class zMeshJSON
	*	\brief A JSON data transfer class for zMesh.
	*	\since version 0.0.1
	*/



	/** @}*/

	/** @}*/

	class zMeshJSON
	{
	private:

		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief container of vertex data  */
		vector<int> vertices;

		/*!	\brief container of half edge data */
		vector<vector<int>> halfedges;

		/*!	\brief container of face data*/
		vector<int> faces;

		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<vector<double>> vertexAttributes;

		/*!	\brief container of half edge attribute data - color*/
		vector<vector<double>> halfedgeAttributes;

		/*!	\brief container of face attribute data - normals, colors.*/
		vector<vector<double>> faceAttributes;

	public:
		
		/*! \brief This method creates the JSON file from the input zMesh using JSON Modern Library.
		*
		*	\param	[in]	j			- output json.
		*	\param	[in]	inMesh		- input mesh.
		*	\param	[in]	vColors		- export vertex color information if true.
		*	\since version 0.0.1
		*/
	
		void to_json(json &j, zMesh &inMesh, bool vColors = false)
		{
			// Vertices
			for (int i = 0; i < inMesh.numVertices(); i++)
			{
				if (inMesh.vertices[i].getEdge()) vertices.push_back(inMesh.vertices[i].getEdge()->getEdgeId());
				else vertices.push_back(-1);

			}

			//Edges
			for (int i = 0; i < inMesh.numEdges(); i++)
			{
				vector<int> HE_edges;

				if (inMesh.edges[i].getPrev()) HE_edges.push_back(inMesh.edges[i].getPrev()->getEdgeId());
				else HE_edges.push_back(-1);

				if (inMesh.edges[i].getNext()) HE_edges.push_back(inMesh.edges[i].getNext()->getEdgeId());
				else HE_edges.push_back(-1);

				if (inMesh.edges[i].getVertex()) HE_edges.push_back(inMesh.edges[i].getVertex()->getVertexId());
				else HE_edges.push_back(-1);

				if (inMesh.edges[i].getFace()) HE_edges.push_back(inMesh.edges[i].getFace()->getFaceId());
				else HE_edges.push_back(-1);

				halfedges.push_back(HE_edges);
			}

			// Faces
			for (int i = 0; i < inMesh.numPolygons(); i++)
			{
				if (inMesh.faces[i].getEdge()) faces.push_back(inMesh.faces[i].getEdge()->getEdgeId());
				else faces.push_back(-1);
			}

			// vertex Attributes
			for (int i = 0; i < inMesh.vertexPositions.size(); i++)
			{
				vector<double> v_attrib;

				v_attrib.push_back(inMesh.vertexPositions[i].x);
				v_attrib.push_back(inMesh.vertexPositions[i].y);
				v_attrib.push_back(inMesh.vertexPositions[i].z);

							if (vColors)
				{
					v_attrib.push_back(inMesh.vertexColors[i].r);
					v_attrib.push_back(inMesh.vertexColors[i].g);
					v_attrib.push_back(inMesh.vertexColors[i].b);
				}


				vertexAttributes.push_back(v_attrib);
			}


			// Json file 
			j["Vertices"] = vertices;
			j["Halfedges"] = halfedges;
			j["Faces"] = faces;
			j["VertexAttributes"] = vertexAttributes;
			j["HalfedgeAttributes"] = halfedgeAttributes;
			j["FaceAttributes"] = faceAttributes;
		}

		/*! \brief This method creates the HE data structure from JSON file using JSON Modern Library.
		*
		*	\param	[in]	j			- input json file.
		*	\param	[in]	inMesh		- output datastructure (zGraph or zMesh).
		*	\since version 0.0.1
		*/
		
		void from_json( json &j, zMesh &inMesh)
		{
			// Vertices
			vertices.clear();
			vertices = (j["Vertices"].get<vector<int>>());

			//Edges
			halfedges.clear();
			halfedges = (j["Halfedges"].get<vector<vector<int>>>());

			// Faces
			faces.clear();
			faces = (j["Faces"].get<vector<int>>());


			// update  mesh

			inMesh.vertices = new zVertex[vertices.size() * 2];
			inMesh.edges = new zEdge[halfedges.size() * 2];
			inMesh.faces = new zFace[faces.size() * 2];

			inMesh.setNumVertices(vertices.size());
			inMesh.setNumEdges(floor(halfedges.size()));
			inMesh.setNumPolygons(faces.size());

			inMesh.vertexActive.clear();
			for (int i = 0; i < vertices.size(); i++)
			{
				inMesh.vertices[i].setVertexId(i);
				if (vertices[i] != -1) inMesh.vertices[i].setEdge(&inMesh.edges[vertices[i]]);

				inMesh.vertexActive.push_back(true);
			}

			int k = 0;
			inMesh.edgeActive.clear();
			for (int i = 0; i < halfedges.size(); i++)
			{
				inMesh.edges[i].setEdgeId(i);

				if (halfedges[i][k] != -1) 	inMesh.edges[i].setPrev(&inMesh.edges[halfedges[i][k]]);
				if (halfedges[i][k + 1] != -1) inMesh.edges[i].setNext(&inMesh.edges[halfedges[i][k + 1]]);
				if (halfedges[i][k + 2] != -1) inMesh.edges[i].setVertex(&inMesh.vertices[halfedges[i][k + 2]]);
				if (halfedges[i][k + 3] != -1) inMesh.edges[i].setFace(&inMesh.faces[halfedges[i][k + 3]]);


				if (i % 2 == 0) inMesh.edges[i].setSym(&inMesh.edges[i]);
				else  inMesh.edges[i].setSym(&inMesh.edges[i - 1]);

				inMesh.edgeActive.push_back(true);
			}

			inMesh.faceActive.clear();
			for (int i = 0; i < faces.size(); i++)
			{
				inMesh.faces[i].setFaceId(i);
				if (faces[i] != -1) inMesh.faces[i].setEdge(&inMesh.edges[faces[i]]);

				inMesh.faceActive.push_back(true);
			}

			// Vertex Attributes
			vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
			//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

			inMesh.vertexPositions.clear();
			for (int i = 0; i < vertexAttributes.size(); i++)
			{
				for (int k = 0; k < vertexAttributes[i].size(); k++)
				{
					// position
					if (vertexAttributes[i].size() == 3)
					{
						zVector pos(vertexAttributes[i][k], vertexAttributes[i][k + 1], vertexAttributes[i][k + 2]);

						inMesh.vertexPositions.push_back(pos);
						k += 2;
					}

					// position and color

					if (vertexAttributes[i].size() == 6)
					{
						zVector pos(vertexAttributes[i][k], vertexAttributes[i][k + 1], vertexAttributes[i][k + 2]);
						inMesh.vertexPositions.push_back(pos);

						zColor col (vertexAttributes[i][k+3], vertexAttributes[i][k + 4], vertexAttributes[i][k + 5], 1) ;
						inMesh.vertexColors.push_back(col);

						k += 5;
					}
				}
			}


			// Edge Attributes
			halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();


			// Edge Attributes
			faceAttributes = j["FaceAttributes"].get<vector<vector<double>>>();



		}
	};


	/** \addtogroup zIO
	*	\brief The data transfer classes and utility methods of the library.
	*  @{
	*/

	/** \addtogroup zIO_JSONClasses
	*	\brief The JSON data transfer classes of the library.
	*  @{
	*/

	/*! \class zGraphJSON
	*	\brief A JSON data transfer class for zGraph.
	*	\since version 0.0.1
	*/
	
	/** @}*/

	/** @}*/

	class zGraphJSON
	{
	private:

		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief container of vertex data  */
		vector<int> vertices;

		/*!	\brief container of half edge data */
		vector<vector<int>> halfedges;

	
		/*!	\brief container of vertex attribute data - positions, normals, colors.*/
		vector<vector<double>> vertexAttributes;

		/*!	\brief container of half edge attribute data - color*/
		vector<vector<double>> halfedgeAttributes;

	
	public:

		/*! \brief This method creates the JSON file from the input zGraph using JSON Modern Library.
		*
		*	\param	[in]	j			- output json.
		*	\param	[in]	inGraph		- input graph.
		*	\param	[in]	vColors		- export vertex color information if true.
		*	\since version 0.0.1
		*/

		void to_json(json &j, zGraph &inGraph, bool vColors = false)
		{
			// Vertices
			for (int i = 0; i < inGraph.numVertices(); i++)
			{
				if (inGraph.vertices[i].getEdge()) vertices.push_back(inGraph.vertices[i].getEdge()->getEdgeId());
				else vertices.push_back(-1);

			}

			//Edges
			for (int i = 0; i < inGraph.numEdges(); i++)
			{
				vector<int> HE_edges;

				if (inGraph.edges[i].getPrev()) HE_edges.push_back(inGraph.edges[i].getPrev()->getEdgeId());
				else HE_edges.push_back(-1);

				if (inGraph.edges[i].getNext()) HE_edges.push_back(inGraph.edges[i].getNext()->getEdgeId());
				else HE_edges.push_back(-1);

				if (inGraph.edges[i].getVertex()) HE_edges.push_back(inGraph.edges[i].getVertex()->getVertexId());
				else HE_edges.push_back(-1);
						
				halfedges.push_back(HE_edges);
			}

		
			// vertex Attributes
			for (int i = 0; i < inGraph.vertexPositions.size(); i++)
			{
				vector<double> v_attrib;

				v_attrib.push_back(inGraph.vertexPositions[i].x);
				v_attrib.push_back(inGraph.vertexPositions[i].y);
				v_attrib.push_back(inGraph.vertexPositions[i].z);

				if (vColors)
				{
					v_attrib.push_back(inGraph.vertexColors[i].r);
					v_attrib.push_back(inGraph.vertexColors[i].g);
					v_attrib.push_back(inGraph.vertexColors[i].b);
				}


				vertexAttributes.push_back(v_attrib);
			}


			// Json file 
			j["Vertices"] = vertices;
			j["Halfedges"] = halfedges;
			j["VertexAttributes"] = vertexAttributes;
			j["HalfedgeAttributes"] = halfedgeAttributes;
		
		}

		/*! \brief This method creates the zGraph from JSON file using JSON Modern Library.
		*
		*	\param	[in]	j			- input json file.
		*	\param	[in]	inGraph		- graph.
		*	\since version 0.0.1
		*/

		void from_json(json &j, zGraph &inGraph)
		{
			// Vertices
			vertices.clear();
			vertices = (j["Vertices"].get<vector<int>>());

			//Edges
			halfedges.clear();
			halfedges = (j["Halfedges"].get<vector<vector<int>>>());

			

			// update  graph

			inGraph.vertices = new zVertex[vertices.size() * 2];
			inGraph.edges = new zEdge[halfedges.size() * 2];
		

			inGraph.setNumVertices(vertices.size());
			inGraph.setNumEdges(floor(halfedges.size()));
		

			inGraph.vertexActive.clear();
			for (int i = 0; i < vertices.size(); i++)
			{
				inGraph.vertices[i].setVertexId(i);
				if (vertices[i] != -1) inGraph.vertices[i].setEdge(&inGraph.edges[vertices[i]]);

				inGraph.vertexActive.push_back(true);
			}

			int k = 0;
			inGraph.edgeActive.clear();
			for (int i = 0; i < halfedges.size(); i++)
			{
				inGraph.edges[i].setEdgeId(i);

				if (halfedges[i][k] != -1) 	inGraph.edges[i].setPrev(&inGraph.edges[halfedges[i][k]]);
				if (halfedges[i][k + 1] != -1) inGraph.edges[i].setNext(&inGraph.edges[halfedges[i][k + 1]]);
				if (halfedges[i][k + 2] != -1) inGraph.edges[i].setVertex(&inGraph.vertices[halfedges[i][k + 2]]);				

				if (i % 2 == 0) inGraph.edges[i].setSym(&inGraph.edges[i]);
				else  inGraph.edges[i].setSym(&inGraph.edges[i - 1]);

				inGraph.edgeActive.push_back(true);
			}

			

			// Vertex Attributes
			vertexAttributes = j["VertexAttributes"].get<vector<vector<double>>>();
			//printf("\n vertexAttributes: %zi %zi", vertexAttributes.size(), vertexAttributes[0].size());

			inGraph.vertexPositions.clear();
			for (int i = 0; i < vertexAttributes.size(); i++)
			{
				for (int k = 0; k < vertexAttributes[i].size(); k++)
				{
					// position
					if (vertexAttributes[i].size() == 3)
					{
						zVector pos(vertexAttributes[i][k], vertexAttributes[i][k + 1], vertexAttributes[i][k + 2]);

						inGraph.vertexPositions.push_back(pos);
						k += 2;
					}

					// position and color

					if (vertexAttributes[i].size() == 6)
					{
						zVector pos(vertexAttributes[i][k], vertexAttributes[i][k + 1], vertexAttributes[i][k + 2]);
						inGraph.vertexPositions.push_back(pos);

						zColor col(vertexAttributes[i][k + 3], vertexAttributes[i][k + 4], vertexAttributes[i][k + 5], 1);
						inGraph.vertexColors.push_back(col);

						k += 5;
					}
				}
			}


			// Edge Attributes
			halfedgeAttributes = j["HalfedgeAttributes"].get<vector<vector<double>>>();


		}
	};

}