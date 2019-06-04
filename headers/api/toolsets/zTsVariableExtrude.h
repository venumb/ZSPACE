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

	/*! \class zTsVariableExtrude
	*	\brief A function set for extrusions based on color attributes.
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/

	/** @}*/

	class zTsVariableExtrude
	{
	protected:
		//--------------------------
		//---- PROTECTED ATTRIBUTES
		//--------------------------

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


		/*! \brief Default constructor.
		*
		*	\since version 0.0.2
		*/
		zTsVariableExtrude() {}

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	_meshObj			- input mesh object.
		*	\since version 0.0.2
		*/
		zTsVariableExtrude(zObjMesh &_meshObj)
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
		~zTsVariableExtrude() {}

		/*! \brief This method offset extrudes the faces of the input mesh based on vertex / face color. It uses only the red channel of the color.
		*
		
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableFaceOffset(bool keepExistingFaces = true, bool assignColor = true, double minVal = 0.01, double maxVal = 0.99, bool useVertexColor = false)
		{

			

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			vector<zColor> faceColors;

			vector<zVector> fCenters;
			fnMesh.getCenters( zFaceData, fCenters);

			// append inmesh positions
			if (keepExistingFaces)
			{
				fnMesh.getVertexPositions(positions);			
			}

			for (zItMeshFace f(*meshObj); !f.end(); f.next())
			{
				

				vector<zItMeshVertex> fVerts;
				f.getVertices(fVerts);			

				zColor faceCol = f.getFaceColor();
				double faceVal = faceCol.r;
				zColor newCol(faceCol.r, 0, 0, 1);

				double extrudeVal = coreUtils.ofMap(faceVal, 0.0, 1.0, minVal, maxVal);

				// get current size of positions
				int numVerts = positions.size();

				// append new positions
				for (auto &fV : fVerts)
				{
					zVector pos = fV.getVertexPosition();
					zVector dir = fCenters[f.getId()] - pos;
					double len = dir.length();
					dir.normalize();

					if (useVertexColor)
					{
						zColor vertexCol = fV.getVertexColor();

						extrudeVal = coreUtils.ofMap(vertexCol.r, 0.0, 1.0, minVal, maxVal);
					}

					zVector newPos = pos + dir * len * extrudeVal;

					positions.push_back(newPos);
				}

				// compute polyconnects and polycounts
				if (keepExistingFaces)
				{
					for (int j = 0; j < fVerts.size(); j++)
					{
						int currentId = j;
						int nextId = (j + 1) % fVerts.size();

						polyConnects.push_back(fVerts[currentId].getId());
						polyConnects.push_back(fVerts[nextId].getId());
						polyConnects.push_back(numVerts + nextId);
						polyConnects.push_back(numVerts + currentId);


						polyCounts.push_back(4);

						if (assignColor) faceColors.push_back(newCol);

					}

				}
				else
				{

					for (int j = 0; j < fVerts.size(); j++)
					{
						int currentId = j;
						polyConnects.push_back(numVerts + currentId);
					}

					if (assignColor) faceColors.push_back(newCol);

					polyCounts.push_back(fVerts.size());

				}

			}


			zObjMesh out;
			zFnMesh temp(out);
			
			if (positions.size() > 0)
			{
				temp.create(positions, polyCounts, polyConnects);

				if (assignColor) temp.setFaceColors(faceColors, true);
			}

			return out;
		}


		/*! \brief This method offsets the boundary faces of the input mesh based on vertex color. It uses only the red channel of the color.
		*
		*	\details	face offset based on http://pyright.blogspot.com/2014/11/polygon-offset-using-vector-math-in.html
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableBoundaryOffset( bool keepExistingFaces = true, bool assignColor = true, double minVal = 0.01, double maxVal = 0.99)
		{

			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			vector<zColor> vertexColors;


			vector<zVector> fCenters;
			fnMesh.getCenters( zFaceData, fCenters);

			if (keepExistingFaces)
			{
				vector<vector<int>> inVertex_newVertex;

				for (zItMeshVertex v(*meshObj); !v.end(); v.next())				
				{
					vector<int> temp;

					if (v.onBoundary())
					{

						temp.push_back(positions.size());

						
						positions.push_back(v.getVertexPosition());

						if (assignColor)
						{
							vertexColors.push_back(v.getVertexColor());
						}

						double vertexVal = v.getVertexColor().r;
						zColor newCol(v.getVertexColor().r, 0, 0, 1);

						double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);


						zItMeshHalfEdge vEdge;

						vector<zItMeshHalfEdge> cEdges;
						v.getConnectedHalfEdges(cEdges);

						for (auto &he : cEdges)
						{
							if (he.onBoundary())
							{
								vEdge = he;
							}
						}

						//if (vEdge == NULL) continue;

						zItMeshVertex next = vEdge.getVertex();;
						zItMeshVertex prev = vEdge.getPrev().getSym().getVertex();

						zVector vNorm = v.getVertexNormal();

						zVector Ori = v.getVertexPosition();;

						zVector v1 = Ori - prev.getVertexPosition();
						v1.normalize();

						zVector n1 = v1 ^ vNorm;
						n1.normalize();

						zVector v2 = next.getVertexPosition() - Ori;
						v2.normalize();

						zVector n2 = v2 ^ vNorm;
						n2.normalize();

						v1 = v1 ^ v2;
						zVector v3 = (n1 + n2);

						v3 *= 0.5;
						v3.normalize();



						double cs = v3 * v2;
						double length = extrudeVal;


						zVector a1 = v2 * cs;
						zVector a2 = v3 - a1;

						double alpha = 0;
						if (a2.length() > 0) alpha = sqrt(a2.length() * a2.length());

						if (cs < 0 && a2.length() > 0) alpha *= -1;

						if (alpha > 0) length /= alpha;

						zVector offPos = Ori + (v3 * length);

						temp.push_back(positions.size());
						positions.push_back(offPos);
						if (assignColor) vertexColors.push_back(v.getVertexColor());



					}

					inVertex_newVertex.push_back(temp);

				}

				// poly connects 
				for (zItMeshHalfEdge he(*meshObj); !he.end(); he.next())
				{
					if (he.onBoundary())
					{
						vector<int> eVerts;
						he.getVertices(eVerts);

						polyConnects.push_back(inVertex_newVertex[eVerts[0]][0]);
						polyConnects.push_back(inVertex_newVertex[eVerts[1]][0]);
						polyConnects.push_back(inVertex_newVertex[eVerts[1]][1]);
						polyConnects.push_back(inVertex_newVertex[eVerts[0]][1]);

						polyCounts.push_back(4);

					}
				}

			}


			if (!keepExistingFaces)
			{

				for (zItMeshVertex v(*meshObj); !v.end(); v.next())
				{
					vector<int> temp;

					if (v.onBoundary())
					{

						double vertexVal = v.getVertexColor().r;
						zColor newCol(v.getVertexColor().r, 0, 0, 1);

						double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);


						zItMeshHalfEdge vEdge = v.getHalfEdge();  

						vector<zItMeshHalfEdge> cEdges;
						v.getConnectedHalfEdges(cEdges);

						for (auto &he: cEdges)
						{
							if (he.onBoundary())
							{
								vEdge =he ;
							}
						}

						//if (vEdge == NULL) continue;


						zItMeshVertex next = vEdge.getVertex();;
						zItMeshVertex prev = vEdge.getPrev().getSym().getVertex();

						zVector vNorm = v.getVertexNormal();

						zVector Ori = v.getVertexPosition(); 

						zVector v1 = Ori - prev.getVertexPosition();
						v1.normalize();

						zVector n1 = v1 ^ vNorm;
						n1.normalize();

						zVector v2 = next.getVertexPosition() - Ori;
						v2.normalize();

						zVector n2 = v2 ^ vNorm;
						n2.normalize();

						v1 = v1 ^ v2;
						zVector v3 = (n1 + n2);

						v3 *= 0.5;
						v3.normalize();



						double cs = v3 * v2;
						double length = extrudeVal;


						zVector a1 = v2 * cs;
						zVector a2 = v3 - a1;

						double alpha = 0;
						if (a2.length() > 0) alpha = sqrt(a2.length() * a2.length());

						if (cs < 0 && a2.length() > 0) alpha *= -1;

						if (alpha > 0) length /= alpha;

						zVector offPos = Ori + (v3 * length);


						positions.push_back(offPos);
						if (assignColor) vertexColors.push_back(v.getVertexColor());

					}

					else
					{
						positions.push_back(v.getVertexPosition());
						if (assignColor) vertexColors.push_back(v.getVertexColor());
					}


				}

				// poly connects 
				for (zItMeshFace f(*meshObj); !f.end(); f.next())				
				{

					vector<int> fVerts;
					f.getVertices(fVerts);

					for (int j = 0; j < fVerts.size(); j++)
					{
						polyConnects.push_back(fVerts[j]);
					}


					polyCounts.push_back(fVerts.size());


				}

			}

			zObjMesh out;
			zFnMesh tempFn(out);

			if (positions.size() > 0)
			{
				tempFn.create(positions, polyCounts, polyConnects);

				if (assignColor) tempFn.setVertexColors(vertexColors, true);
			}

			return out;
		}


		/*! \brief This method extrudes the input mesh based on vertex / face color. It uses only the red channel of the color.
		*
		*	\param		[in]	keepExistingFaces			- true if existing face needs to be retained.
		*	\param		[in]	minVal						- minimum offset. Needs to be between 0 and 1.
		*	\param		[in]	maxVal						- maximum offset. Needs to be between 0 and 1.
		*	\param		[in]	useVertexColor				- true if vertex color is to be used else face color is used.
		*	\since version 0.0.1
		*/
		zObjMesh getVariableFaceThicknessExtrude(bool assignColor, double minVal, double maxVal)
		{



			vector<zVector>positions;
			vector<int>polyConnects;
			vector<int>polyCounts;

			vector<zColor> vertexColors;


			int numVerts = fnMesh.numVertices();

			// append inmesh positions
			 fnMesh.getVertexPositions(positions);

			if (assignColor)  fnMesh.getVertexColors(vertexColors);

			for (zItMeshVertex v(*meshObj); !v.end(); v.next())			
			{
				double vertexVal = v.getVertexColor().r;
				zColor newCol(v.getVertexColor().r, 0, 0, 1);

				double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);

				zVector vNormal = v.getVertexNormal();
				vNormal.normalize();

				positions.push_back(v.getVertexPosition() + vNormal * extrudeVal);

				if (assignColor) vertexColors.push_back(v.getVertexColor());

			}

			// bottom and top face connectivity
			for (zItMeshFace f(*meshObj); !f.end(); f.next())			
			{
				vector<int> fVerts;
				f.getVertices( fVerts);

				// top face
				for (int j = 0; j < fVerts.size(); j++)
				{
					polyConnects.push_back(fVerts[j] + numVerts);
				}
				polyCounts.push_back(fVerts.size());


				// bottom face
				for (int j = fVerts.size() - 1; j >= 0; j--)
				{
					polyConnects.push_back(fVerts[j]);
				}
				polyCounts.push_back(fVerts.size());

			}

			// boundary thickness
			for (zItMeshHalfEdge he(*meshObj); !he.end(); he.next())			
			{
				if (he.onBoundary())
				{
					vector<int> eVerts;
					he.getVertices(eVerts);

					polyConnects.push_back(eVerts[0]);
					polyConnects.push_back(eVerts[1]);
					polyConnects.push_back(eVerts[1] + numVerts);
					polyConnects.push_back(eVerts[0] + numVerts);

					polyCounts.push_back(4);

				}
			}


			zObjMesh out;
			zFnMesh tempFn(out);

			if (positions.size() > 0)
			{
				tempFn.create(positions, polyCounts, polyConnects);

				if (assignColor) tempFn.setVertexColors(vertexColors, true);
			}

			return out;
		}


	};
	
}