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

			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				

				vector<int> fVerts;
				fnMesh.getVertices(i, zFaceData, fVerts);

				vector<zVector> fVertPositions;
				fnMesh.getVertexPositions(i, zFaceData, fVertPositions);

				zColor faceCol = fnMesh.getFaceColor(i);
				double faceVal = faceCol.r;
				zColor newCol(faceCol.r, 0, 0, 1);

				double extrudeVal = coreUtils.ofMap(faceVal, 0.0, 1.0, minVal, maxVal);

				// get current size of positions
				int numVerts = positions.size();

				// append new positions
				for (int j = 0; j < fVertPositions.size(); j++)
				{
					zVector dir = fCenters[i] - fVertPositions[j];
					double len = dir.length();
					dir.normalize();

					if (useVertexColor)
					{
						zColor vertexCol = fnMesh.getVertexColor(fVerts[j]);

						extrudeVal = coreUtils.ofMap(vertexCol.r, 0.0, 1.0, minVal, maxVal);
					}

					zVector newPos = fVertPositions[j] + dir * len * extrudeVal;

					positions.push_back(newPos);
				}

				// compute polyconnects and polycounts
				if (keepExistingFaces)
				{
					for (int j = 0; j < fVerts.size(); j++)
					{
						int currentId = j;
						int nextId = (j + 1) % fVerts.size();

						polyConnects.push_back(fVerts[currentId]);
						polyConnects.push_back(fVerts[nextId]);
						polyConnects.push_back(numVerts + nextId);
						polyConnects.push_back(numVerts + currentId);


						polyCounts.push_back(4);

						if (assignColor) faceColors.push_back(newCol);

					}

				}
				else
				{

					for (int j = 0; j < fVertPositions.size(); j++)
					{
						int currentId = j;
						polyConnects.push_back(numVerts + currentId);
					}

					if (assignColor) faceColors.push_back(newCol);

					polyCounts.push_back(fVertPositions.size());

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

				for (int i = 0; i < fnMesh.numVertices(); i++)
				{
					vector<int> temp;

					if (fnMesh.onBoundary(i, zVertexData))
					{

						temp.push_back(positions.size());

						
						positions.push_back(fnMesh.getVertexPosition(i));

						if (assignColor)
						{
							vertexColors.push_back(fnMesh.getVertexColor(i));
						}

						double vertexVal = fnMesh.getVertexColor(i).r;
						zColor newCol(fnMesh.getVertexColor(i).r, 0, 0, 1);

						double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);


						zEdge *vEdge;

						vector<int> cEdges;
						fnMesh.getConnectedEdges(i, zVertexData, cEdges);

						for (int j = 0; j < cEdges.size(); j++)
						{
							if (fnMesh.onBoundary(cEdges[j], zEdgeData))
							{
								vEdge = fnMesh.getEdge(cEdges[j],zEdgeData);
							}
						}

						//if (vEdge == NULL) continue;

						int next = vEdge->getVertex()->getVertexId();;
						int prev = vEdge->getPrev()->getSym()->getVertex()->getVertexId();

						zVector vNorm = fnMesh.getVertexNormal(i);

						zVector Ori = fnMesh.getVertexPosition(i);;

						zVector v1 = Ori - fnMesh.getVertexPosition(prev);
						v1.normalize();

						zVector n1 = v1 ^ vNorm;
						n1.normalize();

						zVector v2 = fnMesh.getVertexPosition(next) - Ori;
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
						if (assignColor) vertexColors.push_back(fnMesh.getVertexColor(i));



					}

					inVertex_newVertex.push_back(temp);

				}

				// poly connects 

				for (int i = 0; i < fnMesh.numEdges(); i++)
				{
					if (fnMesh.onBoundary(i, zEdgeData))
					{
						vector<int> eVerts;
						fnMesh.getVertices(i, zEdgeData, eVerts);

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


				for (int i = 0; i < fnMesh.numVertices(); i++)
				{
					vector<int> temp;

					if (fnMesh.onBoundary(i, zVertexData))
					{

						double vertexVal = fnMesh.getVertexColor(i).r;
						zColor newCol(fnMesh.getVertexColor(i).r, 0, 0, 1);

						double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);


						zEdge *vEdge = fnMesh.getEdge(i, zVertexData);  

						vector<int> cEdges;
						fnMesh.getConnectedEdges(i, zVertexData, cEdges);

						for (int j = 0; j < cEdges.size(); j++)
						{
							if (fnMesh.onBoundary(cEdges[j], zEdgeData))
							{
								vEdge =fnMesh.getEdge(cEdges[j],zEdgeData) ;
							}
						}

						//if (vEdge == NULL) continue;


						int next = vEdge->getVertex()->getVertexId();;
						int prev = vEdge->getPrev()->getSym()->getVertex()->getVertexId();

						zVector vNorm = fnMesh.getVertexNormal(i);

						zVector Ori = fnMesh.getVertexPosition(i); 

						zVector v1 = Ori - fnMesh.getVertexPosition(prev);  
						v1.normalize();

						zVector n1 = v1 ^ vNorm;
						n1.normalize();

						zVector v2 = fnMesh.getVertexPosition(next) - Ori;
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
						if (assignColor) vertexColors.push_back(fnMesh.getVertexColor(i));

					}

					else
					{
						positions.push_back(fnMesh.getVertexPosition(i));
						if (assignColor) vertexColors.push_back(fnMesh.getVertexColor(i));
					}


				}

				// poly connects 

				for (int i = 0; i < fnMesh.numPolygons(); i++)
				{

					vector<int> fVerts;
					fnMesh.getVertices(i, zFaceData, fVerts);

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

			for (int i = 0; i < fnMesh.numVertices(); i++)
			{
				double vertexVal = fnMesh.getVertexColor(i).r;
				zColor newCol(fnMesh.getVertexColor(i).r, 0, 0, 1);

				double extrudeVal = coreUtils.ofMap(vertexVal, 0.0, 1.0, minVal, maxVal);

				zVector vNormal = fnMesh.getVertexNormal(i);
				vNormal.normalize();

				positions.push_back(fnMesh.getVertexPosition(i) + vNormal * extrudeVal);

				if (assignColor) vertexColors.push_back(fnMesh.getVertexColor(i));

			}

			// bottom and top face connectivity
			for (int i = 0; i < fnMesh.numPolygons(); i++)
			{
				vector<int> fVerts;
				fnMesh.getVertices(i, zFaceData, fVerts);

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
			for (int i = 0; i < fnMesh.numEdges(); i++)
			{
				if (fnMesh.onBoundary(i, zEdgeData))
				{
					vector<int> eVerts;
					fnMesh.getVertices(i, zEdgeData, eVerts);

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