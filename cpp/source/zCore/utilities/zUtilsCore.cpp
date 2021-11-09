// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#include<headers/zCore/utilities/zUtilsCore.h>

#ifndef ZSPACE_UTILS_CORE_JPEG
#define ZSPACE_UTILS_CORE_JPEG



namespace zSpace
{
	//--- USED for JPEG export methods to be defined outside of class.

	ZSPACE_EXTERN std::ofstream jpegFile;

	// write a single byte compressed by tooJpeg
	ZSPACE_INLINE 	void jpegWrite(unsigned char byte)
	{
		jpegFile << byte;
	}

}

#endif

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zUtilsCore::zUtilsCore() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zUtilsCore::~zUtilsCore() {}

	//---- WINDOWS UTILITY  METHODS

	ZSPACE_INLINE int zUtilsCore::getNumfiles(string dirPath)
	{
		int out = 0;

		for (const auto & entry : fs::directory_iterator(dirPath)) out++;

		return out;
	}

	ZSPACE_INLINE int zUtilsCore::getNumfiles_Type(string dirPath, zFileTpye type)
	{
		int out = 0;

		string extension;
		if (type == zJSON) extension = ".json";
		if (type == zOBJ) extension = ".obj";
		if (type == zTXT) extension = ".txt";
		if (type == zCSV) extension = ".csv";
		if (type == zBMP) extension = ".bmp";
		if (type == zPNG) extension = ".png";
		if (type == zJPEG) extension = ".jpeg";


		for (const auto & entry : fs::directory_iterator(dirPath))
		{
			if ((entry.path().extension()) == extension) out++;
		}

		return out;
	}

	ZSPACE_INLINE void zUtilsCore::getFilesFromDirectory(zStringArray &fpaths, string dirPath, zFileTpye type)
	{
		fpaths.clear();

		if (dirPath.back() != '/') dirPath += "/";

		vector< fs::path> file_paths;

		string extension;
		if (type == zJSON) extension = ".json";
		if (type == zOBJ) extension = ".obj";
		if (type == zTXT) extension = ".txt";
		if (type == zCSV) extension = ".csv";
		if (type == zBMP) extension = ".bmp";

		for (const auto & entry : fs::directory_iterator(dirPath))
		{
			if ((entry.path().extension()) == extension) file_paths.push_back(entry.path());
		}

		// sort by time
		sort(file_paths.begin(), file_paths.end(), compare_time_creation);

		// store as string
		vector<fs::path>::iterator it;
		for (it = file_paths.begin(); it != file_paths.end(); ++it)fpaths.push_back(it->string());
	}

	//---- STRING METHODS

	ZSPACE_INLINE vector<string> zUtilsCore::splitString(const string& str, const string& delimiter)
	{
		vector<string> elements;
		// Skip delimiters at beginning.
		string::size_type lastPos = str.find_first_not_of(delimiter, 0);
		// Find first "non-delimiter".
		string::size_type pos = str.find_first_of(delimiter, lastPos);

		while (string::npos != pos || string::npos != lastPos)
		{
			// Found a token, add it to the vector.
			elements.push_back(str.substr(lastPos, pos - lastPos));
			// Skip delimiters.  Note the "not_of"
			lastPos = str.find_first_not_of(delimiter, pos);
			// Find next "non-delimiter"
			pos = str.find_first_of(delimiter, lastPos);
		}
		return elements;
	}

	//---- NUMERICAL METHODS

	ZSPACE_INLINE  int zUtilsCore::randomNumber(int min, int max)
	{
		return rand() % (max - min + 1) + min;
	}

	ZSPACE_INLINE  double zUtilsCore::randomNumber_double(double min, double max)
	{
		return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
	}
	
	ZSPACE_INLINE zVector zUtilsCore::factoriseVector(zVector inputVec, int precision)
	{

#ifndef __CUDACC__
		double factor = pow(10, precision);
#else
		double factor = powf(10, precision);
#endif


		inputVec.x = std::round(inputVec.x *factor) / factor;
		inputVec.y = std::round(inputVec.y *factor) / factor;
		inputVec.z = std::round(inputVec.z *factor) / factor;
		
		return  inputVec;
	}

	ZSPACE_INLINE bool zUtilsCore::checkRepeatVector(zVector &inVal, vector<zVector> values, int &index, int precision)
	{
		bool out = false;
		index = -1;
		zVector v1 = factoriseVector(inVal, precision);

		for (int i = 0; i < values.size(); i++)
		{			
			zVector v2 = factoriseVector(values[i], precision);

			if (v1 == v2)
			{
				out = true;

				index = i;
				break;
			}
		}

		return out;
	}

	//---- MAP METHODS 

	ZSPACE_INLINE bool zUtilsCore::existsInMap(string hashKey, unordered_map<string, int> map, int &outVal)
	{

		bool out = false;;

		std::unordered_map<std::string, int>::const_iterator got = map.find(hashKey);

		if (got != map.end())
		{
			out = true;
			outVal = got->second;
		}

		return out;
	}

	ZSPACE_INLINE bool zUtilsCore::vertexExists(unordered_map<string, int>& positionVertex, zVector & pos, int precisionFac, int & outVertexId)
	{
		bool out = false;;

		double x1, y1, z1;

		if (precisionFac > 0)
		{
			double factor = pow(10, precisionFac);
			x1 = std::round(pos.x *factor) / factor;
			if (x1 == 0) x1 = abs(x1);

			y1 = std::round(pos.y *factor) / factor;
			if (y1 == 0) y1 = abs(y1);

			z1 = std::round(pos.z *factor) / factor;
			if (z1 == 0) z1 = abs(z1);
		}
		else if (precisionFac == 0)
		{
			x1 = std::round(pos.x);
			if (x1 == 0) x1 = abs(x1);

			y1 = std::round(pos.y);
			if (y1 == 0) y1 = abs(y1);

			z1 = std::round(pos.z);
			if (z1 == 0) z1 = abs(z1);
		}


		string hashKey = (to_string(x1) + "," + to_string(y1) + "," + to_string(z1));
		std::unordered_map<std::string, int>::const_iterator got = positionVertex.find(hashKey);

		//cout << endl << hashKey;

		if (got != positionVertex.end())
		{
			out = true;
			outVertexId = got->second;
		}


		return out;
	}

	ZSPACE_INLINE void zUtilsCore::addToPositionMap(unordered_map<string, int>& positionVertex, zVector &pos, int index, int precisionFac)
	{

		double x1, y1, z1;

		if (precisionFac > 0)
		{
			double factor = pow(10, precisionFac);
			x1 = std::round(pos.x *factor) / factor;
			if (x1 == 0) x1 = abs(x1);

			y1 = std::round(pos.y *factor) / factor;
			if (y1 == 0) y1 = abs(y1);

			z1 = std::round(pos.z *factor) / factor;
			if (z1 == 0) z1 = abs(z1);
		}
		else if (precisionFac == 0)
		{
			x1 = std::round(pos.x);
			if (x1 == 0) x1 = abs(x1);

			y1 = std::round(pos.y);
			if (y1 == 0) y1 = abs(y1);

			z1 = std::round(pos.z);
			if (z1 == 0) z1 = abs(z1);
		}

		string hashKey = (to_string(x1) + "," + to_string(y1) + "," + to_string(z1));
		positionVertex[hashKey] = index;
	}

	//---- VECTOR METHODS 
	
	ZSPACE_INLINE zVector zUtilsCore::zMin(vector<zVector> &vals)
	{
		double minLen = 10000;
		int minID = -1;
		for (int i = 0; i < vals.size(); i++)
		{
			if (vals[i].length() < minLen)
			{
				minLen = vals[i].length();
				minID = i;
			}
		}

		return vals[minID];
	}

	ZSPACE_INLINE zVector zUtilsCore::zMax(vector<zVector> &vals)
	{
		double maxLen = 0;
		int maxID = -1;
		for (int i = 0; i < vals.size(); i++)
		{
			if (vals[i].length() > maxLen)
			{
				maxLen = vals[i].length();
				maxID = i;
			}
		}

		return vals[maxID];
	}

	ZSPACE_INLINE zVector zUtilsCore::ofMap(float value, float inputMin, float inputMax, zVector outputMin, zVector outputMax)
	{
		zVector out;

		out.x = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.x, outputMax.x);
		out.y = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.y, outputMax.y);
		out.z = zUtilsCore::ofMap(value, inputMin, inputMax, outputMin.z, outputMax.z);

		return out;
	}

	ZSPACE_INLINE zVector zUtilsCore::ofMap(float value, zDomainFloat inputDomain, zDomainVector outDomain)
	{
		zVector out;

		out.x = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.x, outDomain.max.x);
		out.y = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.y, outDomain.max.y);
		out.z = ofMap(value, inputDomain.min, inputDomain.max, outDomain.min.z, outDomain.max.z);

		return out;
	}

	ZSPACE_INLINE zVector zUtilsCore::fromMatrix4Row(zMatrix4 &inMatrix, int rowIndex)
	{
		zMatrix4Row rVals;
		inMatrix.getRow(rowIndex, rVals);
		return zVector(rVals[0], rVals[1], rVals[2]);
	}


	ZSPACE_INLINE zVector zUtilsCore::fromMatrix4Column(zMatrix4 &inMatrix, int colIndex)
	{
		zMatrix4Col cVals;
		inMatrix.getCol(colIndex, cVals);
		return zVector(cVals[0], cVals[1], cVals[2]);
	}

	ZSPACE_INLINE zVector zUtilsCore::factorise(zVector &inVector, int precision)
	{

#ifndef __CUDACC__
		double factor = pow(10, precision);
#else
		double factor = powf(10, precision);
#endif

		double x1 = std::round(inVector.x *factor) / factor;
		double y1 = std::round(inVector.y *factor) / factor;
		double z1 = std::round(inVector.z *factor) / factor;

		return zVector(x1, y1, z1);
	}

	ZSPACE_INLINE void zUtilsCore::scalePointCloud(vector<zVector> &inPoints, double scaleFac)
	{
		for (int i = 0; i < inPoints.size(); i++)
		{
			inPoints[i] *= scaleFac;
		}
	}

	ZSPACE_INLINE void zUtilsCore::getCenter_PointCloud(vector<zVector> &inPoints, zVector &center)
	{
		zVector out;

		for (int i = 0; i < inPoints.size(); i++)
		{
			out += inPoints[i];
		}

		out /= inPoints.size();

		center = out;
	}

	ZSPACE_INLINE int zUtilsCore::getClosest_PointCloud(zVector &pos, vector<zVector> inPositions)
	{
		int out = -1;
		double dist = 1000000000;

		for (int i = 0; i < inPositions.size(); i++)
		{
			if (inPositions[i].squareDistanceTo(pos) < dist)
			{
				dist = inPositions[i].squareDistanceTo(pos);
				out = i;
			}
		}

		return out;
	}

	ZSPACE_INLINE void zUtilsCore::getBounds(vector<zVector> &inPoints, zVector &minBB, zVector &maxBB)
	{
		minBB = zVector(10000, 10000, 10000);
		maxBB = zVector(-10000, -10000, -10000);

		for (int i = 0; i < inPoints.size(); i++)
		{
			if (inPoints[i].x < minBB.x) minBB.x = inPoints[i].x;
			if (inPoints[i].y < minBB.y) minBB.y = inPoints[i].y;
			if (inPoints[i].z < minBB.z) minBB.z = inPoints[i].z;

			if (inPoints[i].x > maxBB.x) maxBB.x = inPoints[i].x;
			if (inPoints[i].y > maxBB.y) maxBB.y = inPoints[i].y;
			if (inPoints[i].z > maxBB.z) maxBB.z = inPoints[i].z;
		}
	}

	ZSPACE_INLINE void zUtilsCore::getBounds(zVector* inPoints, int numPoints, zVector &minBB, zVector &maxBB)
	{
		minBB = zVector(10000, 10000, 10000);
		maxBB = zVector(-10000, -10000, -10000);

		for (int i = 0; i < numPoints; i++)
		{
			if (inPoints[i].x < minBB.x) minBB.x = inPoints[i].x;
			if (inPoints[i].y < minBB.y) minBB.y = inPoints[i].y;
			if (inPoints[i].y < minBB.z) minBB.z = inPoints[i].z;

			if (inPoints[i].x > maxBB.x) maxBB.x = inPoints[i].x;
			if (inPoints[i].y > maxBB.y) maxBB.y = inPoints[i].y;
			if (inPoints[i].z > maxBB.z) maxBB.z = inPoints[i].z;
		}
	}

	ZSPACE_INLINE zVector zUtilsCore::getDimsFromBounds(zVector &minBB, zVector &maxBB)
	{
		zVector out;

		out.x = abs(maxBB.x - minBB.x);
		out.y = abs(maxBB.y - minBB.y);
		out.z = abs(maxBB.z - minBB.z);

		return out;
	}

	ZSPACE_INLINE bool zUtilsCore::pointInBounds(zVector &inPoint, zVector &minBB, zVector &maxBB)
	{

		if (inPoint.x < minBB.x || inPoint.x > maxBB.x) return false;

		else if (inPoint.y < minBB.y || inPoint.y > maxBB.y) return false;

		else if (inPoint.z < minBB.z || inPoint.z > maxBB.z) return false;

		else return true;
	}

	ZSPACE_INLINE void zUtilsCore::getDistanceWeights(zPoint& inPos, zPointArray positions, double power, zDoubleArray &weights)
	{
		vector<double> dists;

		for (int i = 0; i < positions.size(); i++)
		{
			double dist = (positions[i].distanceTo(inPos));

			double r = pow(dist, power);

			weights.push_back(1.0 / r);


		}

	}

	ZSPACE_INLINE bool zUtilsCore::plane_planeIntersection(zVector &nA, zVector &nB, zVector &pA, zVector &pB, zVector &outP1, zVector &outP2)
	{
		{
			bool out = false;

			if ((nA^nB).length() > 0)
			{
				double detr = ((nA*nA)*(nB*nB)) - ((nA*nB) * (nA*nB));

				double d1 = nA * pA;
				double d2 = nB * pB;

				double c1 = (d1*(nB*nB)) - (d2*(nA*nB));
				c1 /= detr;

				double c2 = (d2*(nA*nA)) - (d1*(nA*nB));
				c2 /= detr;


			}


			return out;
		}
	}

	ZSPACE_INLINE bool zUtilsCore::line_lineClosestPoints(zVector &a0, zVector &a1, zVector &b0, zVector &b1, double &uA, double &uB)
	{
		bool out = false;

		zVector u = a1 - a0;
		zVector v = b1 - b0;
		zVector w = a0 - b0;

		double uu = u * u;
		double uv = u * v;
		double vv = v * v;
		double uw = u * w;
		double vw = v * w;

		double denom = ((uu*vv) - (uv * uv));

		if (denom != 0)
		{
			denom = 1 / denom;

			uA = (uv*vw - vv * uw) * denom;
			uB = (uu*vw - uv * uw) * denom;

			out = true;
		}

		return out;
	}

	ZSPACE_INLINE bool zUtilsCore::line_lineClosestPoints(zVector &p1, zVector &p2, zVector &p3, zVector &p4, double &uA, double &uB, zVector &pA, zVector &pB)
	{
		zVector p13 = p1 - p3;
		zVector p43 = p4 - p3;

		//if (abs(p43.x < EPS) && abs(p43.y < EPS) && abs(p43.z < EPS)) return false;

		zVector p21 = p2 - p1;
		//if (abs(p21.x < EPS) && abs(p21.y < EPS) && abs(p21.z < EPS)) return false;

		double d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
		double d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
		double d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
		double d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
		double d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

		double denom = d2121 * d4343 - d4321 * d4321;
		if (abs(denom) < EPS)return false;

		double numer = d1343 * d4321 - d1321 * d4343;

		uA = numer / denom;
		uB = (d1343 + d4321 * (uA)) / d4343;

		pA.x = p1.x + uA * p21.x;
		pA.y = p1.y + uA * p21.y;
		pA.z = p1.z + uA * p21.z;
		pB.x = p3.x + uB * p43.x;
		pB.y = p3.y + uB * p43.y;
		pB.z = p3.z + uB * p43.z;

		if (pA.distanceTo(pB) < EPS) return  true;
		else return false;
	}

	ZSPACE_INLINE bool zUtilsCore::line_PlaneIntersection(zVector &p1, zVector &p2, zVector &planeNorm, zVector &p3, zVector &intersectionPt)
	{
		bool out = false;

		//printf("\n p1: %1.2f  %1.2f  %1.2f  ", p1.x, p1.y, p1.z);
		//printf("\n p2: %1.2f  %1.2f  %1.2f  ", p2.x, p2.y, p2.z);
		//printf("\n p3: %1.2f  %1.2f  %1.2f  ", p3.x, p3.y, p3.z);
		//printf("\n n: %1.2f  %1.2f  %1.2f  ", planeNorm.x, planeNorm.y, planeNorm.z);

		zVector p31 = p3 - p1;
		zVector p21 = p2 - p1;

		double denom = (planeNorm * (p21));

		if (denom != 0)
		{
			double u = (planeNorm * (p31)) / denom;

			if (u >= 0 && u <= 1) out = true;
							
			double lenP21 = p21.length();
			p21.normalize();

			intersectionPt = (p21 * (lenP21 * u));
			intersectionPt += p1;
		
		}


		return out;
	}

	ZSPACE_INLINE bool zUtilsCore::ray_triangleIntersection(zPoint & a, zPoint & b, zPoint & c, zVector & d, zPoint & o, zPoint &intersectionPt)
	{

		// Ray-triangle isect:
		zVector e1 = b - a; 
		zVector e2 = c - a;
		zVector n = e1 ^ e2;

		float det =  d * n * -1;
		zVector ao = o - a;
		zVector dao = ao ^ d;

		float u = (e2 * dao)/ det;
		float v = -(e1 * dao) / det;
		float t = (ao*n)/ det;

		bool out = (t > 0.0 && u > 0.0 && v > 0.0 && (u + v) < 1.0);

		if (out)
		{
			intersectionPt = o + (d*t);
		}

		return out;			
	}

	ZSPACE_INLINE double zUtilsCore::getTriangleArea(zVector &v1, zVector &v2, zVector &v3)
	{
		double area = 0;

		zVector e12 = v2 - v1;
		zVector e13 = v3 - v1;

		area = ((e12^e13).length() * 0.5);

		return area;
	}

	ZSPACE_INLINE double zUtilsCore::getSignedTriangleVolume(zVector &v1, zVector &v2, zVector &v3)
	{
		double volume = 0;

		zVector norm = (v2 - v1) ^ (v3 - v1);
		volume = v1 * norm;
		volume /= 6.0;


		return volume;
	}

	ZSPACE_INLINE bool zUtilsCore::pointInTriangle(zVector &pt, zVector &t0, zVector &t1, zVector &t2)
	{
		// Compute vectors        
		zVector v0 = t2 - t0;
		zVector	v1 = t1 - t0;
		zVector	v2 = pt - t0;

		// Compute dot products
		double	dot00 = v0 * v0;
		double	dot01 = v0 * v1;
		double	dot02 = v0 * v2;
		double	dot11 = v1 * v1;
		double	dot12 = v1 * v2;

		// Compute barycentric coordinates
		double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
		double 	u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		double	v = (dot00 * dot12 - dot01 * dot02) * invDenom;

		if (abs(u) < 0.001) u = 0;
		if (abs(v) < 0.001) v = 0;

		// round factor to precision 3 
#ifndef __CUDACC__
		double factor = pow(10, 3);
#else
		double factor = powf(10, 3);
#endif

		u = std::round(u*factor) / factor;
		v = std::round(v*factor) / factor;

		//printf("\n u : %1.2f v: %1.2f ", u, v);

		// Check if point is in triangle
		return ((u >= 0) && (v >= 0) && (u + v <= 1));


	}

	ZSPACE_INLINE bool zUtilsCore::pointInPlanarPolygon(zPoint& pt, zPointArray& points, zVector& planeNormal)
	{
		
		// get component with largest component of normal
		int largeComponent = 0;
		float largeValue = 0;
		
		if (abs(planeNormal.x) > largeValue)
		{
			largeComponent = 0;
			largeValue = abs(planeNormal.x);
		}
	
		if (abs(planeNormal.y) > largeValue)
		{
			largeComponent = 1;
			largeValue = abs(planeNormal.y);
		}
		
		if (abs(planeNormal.z) > largeValue)
		{
			largeComponent = 2;
			largeValue = abs(planeNormal.z);
		}
	
		// convert points to 2D points by ignorning largest component

		zPoint projectedPt;
		int counter = 0;
		for (int i = 0; i < 3; i++)
		{
			if (i != largeComponent)
			{
				if (counter == 0)
				{
					if( i == 0)  projectedPt.x = pt.x;
					if (i == 1)  projectedPt.x = pt.y;
					if (i == 2)  projectedPt.x = pt.z;
					
					
				}
				if (counter == 1)
				{
					if (i == 0) projectedPt.y = pt.x;
					if (i == 1) projectedPt.y = pt.y;
					if (i == 2) projectedPt.y = pt.z;
					
				}

				counter++;
			}
		}
		
		zPointArray projectedPoints;

		for (int j = 0; j < points.size(); j++)
		{
			

			zPoint projP;
			int count = 0;
			for (int i = 0; i < 3; i++)
			{
				if (i != largeComponent)
				{
					if (count == 0)
					{
						if (i == 0)  projP.x = points[j].x;
						if (i == 1)  projP.x = points[j].y;
						if (i == 2)  projP.x = points[j].z;
						
					}
					if (count == 1)
					{
						if (i == 0)  projP.y = points[j].x;
						if (i == 1)  projP.y = points[j].y;
						if (i == 2)  projP.y = points[j].z;
						
					}	

					count++;
				}

				
			}

			projP = projP - projectedPt;
			projectedPoints.push_back(projP);	

			//cout << endl << points[j];
			
		}

		projectedPt = zPoint(0,0,0);
		

		// compute winding number
		float windingNum = 0;

		// loop through all edges of the polygon
		//https://www.engr.colostate.edu/~dga/documents/papers/point_in_polygon.pdf	
		for (int j = 0; j < projectedPoints.size(); j++)
		{
			int next = (j + 1) % projectedPoints.size();

			if (projectedPoints[j].y * projectedPoints[next].y < 0)
			{
				float numerator = projectedPoints[j].y * (projectedPoints[next].x - projectedPoints[j].x);
				float denominator = (projectedPoints[j].y - projectedPoints[next].y);
				float r = projectedPoints[j].x + (numerator / denominator);

				if (r > 0)
				{
					if (projectedPoints[j].y < 0) windingNum += 1;
					else windingNum -= 1;
				}
			}
			else if (projectedPoints[j].y == 0 && projectedPoints[j].x > 0)
			{
				if(projectedPoints[next].y >0) windingNum += 0.5;
				else windingNum -= 0.5;
			}
			else if (projectedPoints[next].y == 0 && projectedPoints[next].x > 0)
			{
				if (projectedPoints[j].y < 0) windingNum += 0.5;
				else windingNum -= 0.5;
			}
		}

		if (windingNum == 0)
		{
			// check if point lies on one of the edges
			for (int j = 0; j < projectedPoints.size(); j++)
			{
				int next = (j + 1) % projectedPoints.size();

				bool check = pointOnLine(projectedPt, projectedPoints[j], projectedPoints[next]);
				if (check)  windingNum = 1;
			}
		}
		
		if (windingNum != 0)
			return true;
		else
			return false;
	}

	ZSPACE_INLINE bool zUtilsCore::pointOnLine(zPoint& pt, zPoint& pA, zPoint& pB, float tolerance)
	{
		float d = (pt.distanceTo(pA) + pt.distanceTo(pB)) - (pA.distanceTo(pB));
		return (d < tolerance);
	}

	ZSPACE_INLINE int zUtilsCore::isLeft(zPoint& p0, zPoint& p1, zPoint& p2)
	{		
		return ((p2.y - p0.y) * (p1.x - p0.x)) - ((p2.x - p0.x) * (p1.y - p0.y));
	}


	ZSPACE_INLINE double zUtilsCore::minDist_Edge_Point(zVector & pt, zVector & e0, zVector & e1, zVector & closest_Pt)
	{
		zVector n = (e1 - e0) ^ (zVector(0, 0, 1));
		n.normalize();
		closest_Pt = n * ((e0 - pt) * n);
		closest_Pt += pt;

		float len = e0.distanceTo(e1);

		zVector ed = (e0 - e1) / len;
		double param = (closest_Pt - e1) * ed;



		if (param > len)param = len;
		if (param < 0) param = 0;

		closest_Pt = e1 + ed * param;

		return closest_Pt.distanceTo(pt);
	}

	ZSPACE_INLINE double zUtilsCore::minDist_Point_Plane(zVector & pA, zVector & pB, zVector & norm)
	{
		norm.normalize();

		return (pA - pB) * norm;
	}

	ZSPACE_INLINE zVector zUtilsCore::getBaryCenter(zPointArray &inPositions, zDoubleArray& weights)
	{
		if (inPositions.size() != weights.size()) throw std::invalid_argument("size of inPositions and weights not equal.");

		zVector out;

		for (int i = 0; i < inPositions.size(); i++)
		{
			out += inPositions[i] * weights[i];
		}

		out /= inPositions.size();

		return out;
	}

	ZSPACE_INLINE zPoint zUtilsCore::sphericalToCartesian(double azimuth, double altitude, double radius)
	{
		zPoint out;

		double azimuth_radians = azimuth * DEG_TO_RAD;
		double altitude_radians = altitude * DEG_TO_RAD;

		out.x = radius * cos(altitude_radians) * sin(azimuth_radians);
		out.y = radius * cos(altitude_radians) * cos(azimuth_radians);
		out.z = radius * sin(altitude_radians);

		return out;
	}

	ZSPACE_INLINE void zUtilsCore::cartesianToSpherical(zPoint &inVec, double & radius, double & azimuth, double & altitude)
	{
		radius = inVec.length();
		azimuth = atan(inVec.y / inVec.x);
		altitude = acos(inVec.z / radius);
	}

	//---- 4x4 zMATRIX  TRANSFORMATION METHODS

	ZSPACE_INLINE void zUtilsCore::setColfromVector(zMatrix4 &inMatrix, zVector &inVec, int index)
	{
		inMatrix(0, index) = inVec.x; inMatrix(1, index) = inVec.y; inMatrix(2, index) = inVec.z;
	}

	ZSPACE_INLINE void zUtilsCore::setRowfromVector(zMatrix4 &inMatrix, zVector &inVec, int index)
	{
		inMatrix(index, 0) = inVec.x; inMatrix(index, 1) = inVec.y; inMatrix(index, 2) = inVec.z;
	}

	ZSPACE_INLINE void zUtilsCore::setTransformfromVectors(zMatrix4 &inMatrix, zVector &X, zVector &Y, zVector &Z, zVector &O)
	{
		inMatrix.setIdentity();

		setColfromVector(inMatrix, X, 0);
		setColfromVector(inMatrix, Y, 1);
		setColfromVector(inMatrix, Z, 2);
		setColfromVector(inMatrix, O, 3);
	}


	ZSPACE_INLINE zMatrix4 zUtilsCore::toWorldMatrix(zMatrix4 &inMatrix)
	{
		zMatrix4 outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));


		outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
		outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
		outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = Cen.x; outMatrix(1, 3) = Cen.y; outMatrix(2, 3) = Cen.z;

		return outMatrix;
	}

	ZSPACE_INLINE zMatrix4 zUtilsCore::toLocalMatrix(zMatrix4 &inMatrix)
	{
		zMatrix4 outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));

		zVector orig(0, 0, 0);
		zVector d = Cen - orig;

		outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
		outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
		outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);



		return outMatrix;
	}

	ZSPACE_INLINE zMatrix4 zUtilsCore::PlanetoPlane(zMatrix4 &from, zMatrix4 &to)
	{
		zMatrix4 world = toWorldMatrix(to);
		zMatrix4 local = toLocalMatrix(from);
		zMatrix4 out = world * local;
		return out;
	}


	ZSPACE_INLINE zMatrix4 zUtilsCore::ChangeBasis(zMatrix4 &from, zMatrix4 &to)
	{
		zMatrix4 world = toWorldMatrix(from);
		zMatrix4 local = toLocalMatrix(to);
		zMatrix4 out = local * world;
		return out;
	}
	
	
	//---- VECTOR METHODS GEOMETRY

	ZSPACE_INLINE void zUtilsCore::getEllipse(double radius, int numPoints, zPointArray &Pts, zMatrix4 worldPlane, double xFactor, double yFactor)
	{
		double theta = 0;

		zMatrix4 localPlane;
		zMatrix4 trans = PlanetoPlane(localPlane, worldPlane);

		for (int i = 0; i < numPoints; i++)
		{
			zVector pos;
			pos.x = (radius * cos(theta)) / xFactor;
			pos.y = (radius * sin(theta)) / yFactor;
			pos.z = 0;

			Pts.push_back(pos * trans);

			theta += (TWO_PI / (numPoints+1));
		}
	}

	ZSPACE_INLINE void zUtilsCore::getRectangle(zVector dims, zPointArray &rectanglePts, zMatrix4 localPlane)
	{
		dims.x *= 0.5;
		dims.y *= 0.5;

		zVector v0 = zVector(-dims.x, -dims.y, 0);
		zVector v1 = zVector(dims.x, -dims.y, 0);
		zVector v2 = zVector(dims.x, dims.y, 0);
		zVector v3 = zVector(-dims.x, dims.y, 0);

		zMatrix4 worldPlane;
		zMatrix4 trans = PlanetoPlane(worldPlane, localPlane);

		rectanglePts.push_back(v0 * trans);
		rectanglePts.push_back(v1* trans);
		rectanglePts.push_back(v2* trans);
		rectanglePts.push_back(v3* trans);

	}

	//---- COLOR  METHODS

	ZSPACE_INLINE zColor zUtilsCore::averageColor(zColor c1, zColor c2, zColorType type)
	{
		zColor out;

		if (type == zRGB)
		{
			out.r = (c2.r + c1.r)	* 0.5;
			out.g = (c2.g + c1.g)	* 0.5;
			out.b = (c2.b + c1.b)	* 0.5;

			out.a = (c2.a + c1.a)	* 0.5;

			out.toHSV();

		}

		if (type == zHSV)
		{
			out.h = (c2.h + c1.h)	* 0.5;
			out.s = (c2.s + c1.s)	* 0.5;
			out.v = (c2.v + c1.v)	* 0.5;

			out.toRGB();
		}


		return out;

	}

	ZSPACE_INLINE zColor zUtilsCore::averageColor(zColorArray &c1, zColorType type)
	{
		zColor out;

		if (type == zRGB)
		{
			for (int i = 0; i < c1.size(); i++)
			{
				out.r += c1[i].r;
				out.g += c1[i].g;
				out.b += c1[i].b;

				out.a += c1[i].a;
			}

			out.r /= c1.size();
			out.g /= c1.size();
			out.b /= c1.size();
			out.a /= c1.size();


			out.toHSV();

		}

		if (type == zHSV)
		{
			for (int i = 0; i < c1.size(); i++)
			{
				out.h += c1[i].h;
				out.s += c1[i].s;
				out.v += c1[i].v;
			}

			out.h /= c1.size();
			out.s /= c1.size();
			out.v /= c1.size();

			out.toRGB();
		}


		return out;

	}

	ZSPACE_INLINE zColor zUtilsCore::blendColor(float inputValue, zDomainFloat inDomain, zDomainColor outDomain, zColorType type)
	{
		zColor out;

		if (type == zRGB)
		{

			if (inDomain.min == inDomain.max)
			{
				out.r = outDomain.min.r;
				out.g = outDomain.min.g;
				out.b = outDomain.min.b;
				out.a = outDomain.min.a;
			}
			else
			{
				if (outDomain.min.r == outDomain.max.r)out.r = outDomain.min.r;
				else out.r = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.r, outDomain.max.r);

				if (outDomain.min.g == outDomain.max.g)out.g = outDomain.min.g;
				else out.g = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.g, outDomain.max.g);

				if (outDomain.min.b == outDomain.max.b)out.b = outDomain.min.b;
				else out.b = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.b, outDomain.max.b);

				if (outDomain.min.a == outDomain.max.a)out.a = outDomain.min.a;
				else out.a = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.a, outDomain.max.a);
			}

			out.toHSV();

		}

		if (type == zHSV)
		{

			if (inDomain.min == inDomain.max)
			{
				out.h = outDomain.min.h;
				out.s = outDomain.min.s;
				out.v = outDomain.min.v;
			}
			else
			{
				if (outDomain.min.h == outDomain.max.h)out.h = outDomain.min.h;
				else out.h = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.h, outDomain.max.h);

				if (outDomain.min.s == outDomain.max.s)out.s = outDomain.min.s;
				else out.s = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.s, outDomain.max.s);

				if (outDomain.min.v == outDomain.max.v)out.v = outDomain.min.v;
				else out.v = ofMap(inputValue, inDomain.min, inDomain.max, outDomain.min.v, outDomain.max.v);
			}


			out.toRGB();
		}


		return out;

	}
	

#ifndef __CUDACC__

	//---- BMP MATRIX  METHODS

	ZSPACE_INLINE void zUtilsCore::matrixToBMP(vector<MatrixXf> &matrices, string path)
	{

		bool checkMatchSize = true;

		if (matrices.size() == 0)
		{
			throw std::invalid_argument(" error: matrix empty.");
			return;
		}

		int resX = matrices[0].rows();
		int resY = matrices[0].cols();

		for (int i = 1; i < matrices.size(); i++)
		{
			if (matrices[i].rows() != resX || matrices[i].cols() != resY)
			{
				checkMatchSize = false;
				break;
			}
		}

		if (!checkMatchSize)
		{
			throw std::invalid_argument(" error: matrix sizes are not equal.");
			return;
		}

		bool alpha = ((matrices.size() == 4)) ? true : false;

		zUtilsBMP bmp(resX, resY, alpha);
		uint32_t channels = bmp.bmp_info_header.bit_count / 8;


		for (uint32_t x = 0; x < resX; ++x)
		{
			for (uint32_t y = 0; y < resY; ++y)
			{
				// blue
				bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = 0;

				// green
				bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = 0;

				// alpha
				if (matrices.size() == 4) bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = 0;

				// red
				bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = 0;

				if (matrices.size() >= 1) bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 2] = matrices[0].coeff(x, y) * 255;

				if (matrices.size() >= 2) bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 1] = matrices[1].coeff(x, y) * 255;

				if (matrices.size() >= 3) bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 0] = matrices[2].coeff(x, y) * 255;

				if (matrices.size() == 4) bmp.data[channels * (y * bmp.bmp_info_header.width + x) + 3] = matrices[3].coeff(x, y) * 255;

			}
		}

		bmp.write(path.c_str());
	}

	ZSPACE_INLINE void zUtilsCore::matrixToJPEG(vector<MatrixXf> &matrices, string path)
	{

		bool checkMatchSize = true;

		if (matrices.size() == 0)
		{
			throw std::invalid_argument(" error: matrix empty.");
			return;
		}

		int resX = matrices[0].rows();
		int resY = matrices[0].cols();

		for (int i = 1; i < matrices.size(); i++)
		{
			if (matrices[i].rows() != resX || matrices[i].cols() != resY)
			{
				checkMatchSize = false;
				break;
			}
		}

		if (!checkMatchSize)
		{
			throw std::invalid_argument(" error: matrix sizes are not equal.");
			return;
		}


		const auto width = resX;
		const auto height = resY;

		// RGB: one byte each for red, green, blue
		const auto bytesPerPixel = 3;

		jpegFile = ofstream (path.c_str(), std::ios_base::out | std::ios_base::binary);

		// allocate memory
		auto image = new unsigned char[width * height * bytesPerPixel];
		
		for (auto y = 0; y < height; y++)
			for (auto x = 0; x < width; x++)
			{
				// memory location of current pixel
				auto offset = (y * width + x) * bytesPerPixel;
				

				image[offset] = 0;
				image[offset + 1] = 0;
				image[offset + 2] = 0;

				if (matrices.size() >= 1) image[offset] = matrices[0].coeff(x, y) * 255;

				if (matrices.size() >= 2) image[offset + 1] = matrices[1].coeff(x, y) * 255;

				if (matrices.size() >= 3) image[offset + 2] = matrices[2].coeff(x, y) * 255;				
			}

		// start JPEG compression
		// optional parameters:
		const bool isRGB = true;  // true = RGB image, else false = grayscale
		const auto quality = 100;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
		const bool downsample = false; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
		const char* comment = "zSpace JPEG image"; // arbitrary JPEG comment
		auto ok = TooJpeg::writeJpeg(jpegWrite, image, width, height, isRGB, quality, downsample, comment);
		delete[] image;	

	}

	ZSPACE_INLINE void zUtilsCore::matrixToPNG(vector<MatrixXf> &matrices, string path)
	{

		bool checkMatchSize = true;

		if (matrices.size() == 0)
		{
			throw std::invalid_argument(" error: matrix empty.");
			return;
		}

		int resX = matrices[0].rows();
		int resY = matrices[0].cols();

		for (int i = 1; i < matrices.size(); i++)
		{
			if (matrices[i].rows() != resX || matrices[i].cols() != resY)
			{
				checkMatchSize = false;
				break;
			}
		}

		if (!checkMatchSize)
		{
			throw std::invalid_argument(" error: matrix sizes are not equal.");
			return;
		}

		bool alpha = ((matrices.size() == 4)) ? true : false;

		unsigned width = resX, height = resY;

		std::vector<unsigned char> image;
		image.resize(width * height * 4);

		for (unsigned x = 0; x < resX; ++x)
		{
			for (unsigned y = 0; y < resY; ++y)
			{
				image[4 * width * y + 4 * x + 0] = 0;
				image[4 * width * y + 4 * x + 1] = 0;
				image[4 * width * y + 4 * x + 2] = 0;
				image[4 * width * y + 4 * x + 3] = 255;


				if (matrices.size() >= 1) image[4 * width * y + 4 * x + 0] = matrices[0].coeff(x, y) * 255;

				if (matrices.size() >= 2) image[4 * width * y + 4 * x + 1] = matrices[1].coeff(x, y) * 255;

				if (matrices.size() >= 3) image[4 * width * y + 4 * x + 2] = matrices[2].coeff(x, y) * 255;

				if (matrices.size() == 4) image[4 * width * y + 4 * x + 3] = matrices[3].coeff(x, y) * 255;

			}
		}

		writePNG(path.c_str(), image, width, height);
	}
	
	ZSPACE_INLINE void zUtilsCore::matrixFromPNG(vector<MatrixXf> &matrices, string path)
	{
		unsigned width, height;
		std::vector<unsigned char> image;

		readPNG(path.c_str(), image, width, height);

		int resX = width;
		int resY = height; 

		// create matrices 

		MatrixXf R(resX, resY);
		MatrixXf G(resX, resY);
		MatrixXf B(resX, resY);
		MatrixXf A(resX, resY);

		for (int x = 0; x < resX; ++x)
		{
			for (int y = 0; y < resY; ++y)
			{
				R(x, y) = image[4 * width * y + 4 * x + 0];
				R(x, y) = ofMap(R(x, y), 0.0f, 255.0f, 0.0f, 1.0f);

				G(x, y) = image[4 * width * y + 4 * x + 1];
				G(x, y) = ofMap(G(x, y), 0.0f, 255.0f, 0.0f, 1.0f);

				B(x, y) = image[4 * width * y + 4 * x + 2];
				B(x, y) = ofMap(B(x, y), 0.0f, 255.0f, 0.0f, 1.0f);

				A(x, y) = image[4 * width * y + 4 * x + 3];
				A(x, y) = ofMap(A(x, y), 0.0f, 255.0f, 0.0f, 1.0f);
			}
		}

		matrices.push_back(R);
		matrices.push_back(G);
		matrices.push_back(B);
		matrices.push_back(A);

	}


	//---- PRIVATE IMAGE METHODS

	ZSPACE_INLINE void zUtilsCore::writePNG(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
	{
		std::vector<unsigned char> png;

		unsigned error = lodepng::encode(png, image, width, height);
		if (!error) lodepng::save_file(png, filename);

		//if there's an error, display it
		if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	}

	ZSPACE_INLINE void zUtilsCore::readPNG(const char* filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height)
	{
		std::vector<unsigned char> png;
		
		//load and decode
		unsigned error = lodepng::load_file(png, filename);
		if (!error) error = lodepng::decode(image, width, height, png);

		//if there's an error, display it
		if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

		//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ..
	}


	//---- MATRIX METHODS USING EIGEN / ARMA

	ZSPACE_INLINE zPlane zUtilsCore::getBestFitPlane(zPointArray& points)
	{

		zPlane out;
		out.setIdentity();

		// compute average point
		zVector averagePt;

		for (int i = 0; i < points.size(); i++)
		{
			averagePt += points[i];

		}

		averagePt /= points.size();

		arma::mat X_arma(points.size(), 3);
		for (int i = 0; i < points.size(); i++)
		{
			X_arma(i, 0) = points[i].x - averagePt.x;
			X_arma(i, 1) = points[i].y - averagePt.y;
			X_arma(i, 2) = points[i].z - averagePt.z;

		}


		mat U;
		arma::vec s;
		mat V;
		arma::svd(U, s, V, X_arma);

		// x
		out(0, 0) = V(0, 0); 	out(1, 0) = V(1, 0);	out(2, 0) = V(2, 0);

		// y
		out(0, 1) = V(0, 1); 	out(1, 1) = V(1, 1);	out(2, 1) = V(2, 1);

		// z
		out(0, 2) = V(0, 2);	out(1, 2) = V(1, 2);	out(2, 2) = V(2, 2);

		// o
		out(0, 3) = averagePt.x;	out(1, 3) = averagePt.y;	out(2, 3) = averagePt.z;

		/*MatrixXd X_eigen(points.size(), 3);
		for (int i = 0; i < points.size(); i++)
		{
			X_eigen(i, 0) = points[i].x - averagePt.x;
			X_eigen(i, 1) = points[i].y - averagePt.y;
			X_eigen(i, 2) = points[i].z - averagePt.z;

		}*/


		//Matrix3f covarianceMat;
		////X_eigen.bdcSvd(ComputeThinU | ComputeThinV).solve(covarianceMat);

		//BDCSVD<Matrix3d> svd;
		//svd.compute(X_eigen);

		//cout << "\n eigen \n " << svd.computeV();

		// compute covariance matrix 
		/*SelfAdjointEigenSolver<Matrix3f> eigensolver;
		Matrix3f covarianceMat;

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{

				float val = 0;
				for (int k = 0; k < points.size(); k++)
				{
					val += (points[k][i] - averagePt[i]) * (points[k][j] - averagePt[j]);
				}

				if (val == INFINITY) val = 0.00;

				if (val > EPS) val /= (points.size() - 1);
				else val = 0.00;

				covarianceMat(i, j) = val;
			}

		}

		eigensolver.compute(covarianceMat);
		if (eigensolver.info() != Success) abort();

		vector<double> X = { eigensolver.eigenvectors().col(2)(0), eigensolver.eigenvectors().col(2)(1), eigensolver.eigenvectors().col(2)(2), 1 };
		vector<double> Y = { eigensolver.eigenvectors().col(1)(0), eigensolver.eigenvectors().col(1)(1), eigensolver.eigenvectors().col(1)(2), 1 };
		vector<double> Z = { eigensolver.eigenvectors().col(0)(0), eigensolver.eigenvectors().col(0)(1), eigensolver.eigenvectors().col(0)(2), 1 };
		vector<double> O = { averagePt.x, averagePt.y, averagePt.z, 1 };*/

		/*out.setCol(0, X);
		out.setCol(1, Y);
		out.setCol(2, Z);
		out.setCol(3, O);*/

		return out;
	}

	ZSPACE_INLINE void zUtilsCore::getProjectedPoints_BestFitPlane(zPointArray & points, zPointArray &projectPoints)
	{
		zPlane plane;
		plane = getBestFitPlane(points);

		// project points on plane
		zVector planeNormal = zVector(plane(0, 2), plane(1, 2), plane(2, 2));
		zVector planeOrigin = zVector(plane(0, 3), plane(1, 3), plane(2, 3));

		projectPoints.clear();

		for (int i = 0; i < points.size(); i++)
		{
			zPoint tmp;
			zPoint B = points[i] + planeNormal;
			bool chkProjected = line_PlaneIntersection(points[i], B, planeNormal, planeOrigin, tmp);
			projectPoints.push_back(tmp);
		}
	}

	ZSPACE_INLINE void zUtilsCore::boundingboxPCA(zPointArray points, zVector &minBB, zVector &maxBB, zVector &minBB_local, zVector &maxBB_local)
	{
		zPlane bPlane_Mat = getBestFitPlane(points);

		// translate points to local frame
		zTransform bPlane_Mat_local = toLocalMatrix(bPlane_Mat);

		for (int i = 0; i < points.size(); i++)
		{
			zVector new_pos = points[i] * bPlane_Mat_local;
			points[i] = new_pos;
		}

		// compute boundings in local frame

		getBounds(points, minBB_local, maxBB_local);


		// translate points to world frame
		zTransform bPlane_Mat_world = toWorldMatrix(bPlane_Mat);

		for (int i = 0; i < points.size(); i++)
		{
			zVector new_pos = points[i] * bPlane_Mat_world;
			points[i] = new_pos;

		}

		minBB = minBB_local * bPlane_Mat_world;
		maxBB = maxBB_local * bPlane_Mat_world;

	}

	ZSPACE_INLINE zTransform zUtilsCore::toWorldMatrix(zTransform &inMatrix)
	{

		zTransform outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));


		outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
		outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
		outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = Cen.x; outMatrix(1, 3) = Cen.y; outMatrix(2, 3) = Cen.z;

		return outMatrix;
	}

	ZSPACE_INLINE zTransform zUtilsCore::toLocalMatrix(zTransform &inMatrix)
	{

		zTransform outMatrix;
		outMatrix.setIdentity();

		zVector X(inMatrix(0, 0), inMatrix(1, 0), inMatrix(2, 0));
		zVector Y(inMatrix(0, 1), inMatrix(1, 1), inMatrix(2, 1));
		zVector Z(inMatrix(0, 2), inMatrix(1, 2), inMatrix(2, 2));
		zVector Cen(inMatrix(0, 3), inMatrix(1, 3), inMatrix(2, 3));

		zVector orig(0, 0, 0);
		zVector d = Cen - orig;

		outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
		outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
		outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

		outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);



		return outMatrix;
	}

	ZSPACE_INLINE zTransform zUtilsCore::PlanetoPlane(zTransform &from, zTransform &to)
	{
		zTransform world = toWorldMatrix(to);
		zTransform local = toLocalMatrix(from);

		zTransform out = world * local;

		return out;
	}

	ZSPACE_INLINE float zUtilsCore::getEuclideanDistance(MatrixXf & m1, MatrixXf & m2, double tolerance)
	{
		if (m1.cols() != m2.cols()) throw std::invalid_argument("number of columns in m1 not equal to number of columns in m2.");
		if (m2.rows() != 1) throw std::invalid_argument("number of rows in m2 not equal to 1.");
		if (m1.rows() != 1) throw std::invalid_argument("number of rows in m1 not equal to 1.");

		double out;

		double dist = 0;;

		for (int j = 0; j < m1.cols(); j++)
		{
			dist += pow((m1(0, j) - m2(0, j)), 2);
		}

		if (dist > tolerance) out = sqrt(dist);
		else out = 0.0;


		return out;
	}

	//---- MATRIX  METHODS USING ARMADILLO

#ifndef USING_CLR 

	ZSPACE_INLINE arma::mat zUtilsCore::rref(arma::mat A, double tol)
	{
		int rows = A.n_rows;
		int cols = A.n_cols;

		int r = 0;

		for (int c = 0; c < cols; c++)
		{
			//## Find the pivot row
			double m; int pivot;
			rref_max(A, r, c, m, pivot);
			//pivot = r + pivot - 1;
			//if (pivot < 0) pivot =0;

			if (m <= tol)
			{
				//## Skip column c, making sure the approximately zero terms are actually zero.
				for (int i = r; i < A.n_rows; i++) A(i, c) = 0;
			}

			else
			{

				//## Swap current row and pivot row
				rref_swapRows(A, r, c, pivot);

				//## Normalize pivot row
				rref_normaliseRow(A, r, c);

				//## Eliminate the current column
				rref_eliminateColumn(A, r, c);

				//## Check if done
				r++;
				if (r == rows) break;

			}
		}

		return A;
	}

	//---- MATRIX  CAST METHODS USING ARMADILLO AND EIGEN

	ZSPACE_INLINE MatrixXd zUtilsCore::armaToEigen(mat &arma_A)
	{
		MatrixXd eigen_B = Eigen::Map<MatrixXd>(arma_A.memptr(), arma_A.n_rows, arma_A.n_cols);
		return eigen_B;
	}

	ZSPACE_INLINE VectorXd zUtilsCore::armaToEigen(arma::vec &arma_A)
	{
		VectorXd eigen_B = Eigen::Map<VectorXd>(arma_A.memptr(), arma_A.n_rows, arma_A.n_cols);
		return eigen_B;
	}

	ZSPACE_INLINE mat zUtilsCore::eigenToArma(MatrixXd &eigen_A)
	{
		mat arma_B = mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(), false, false);
		return arma_B;
	}

	ZSPACE_INLINE arma::vec zUtilsCore::eigenToArma(VectorXd &eigen_A)
	{
		arma::vec arma_B = arma::vec(eigen_A.data(), eigen_A.rows(), false, false);
		return arma_B;
	}

#endif

	//---- PRIVATE MATRIX  METHODS

#ifndef USING_CLR

	ZSPACE_INLINE void zUtilsCore::rref_max(mat &A, int &r, int &c, double &m, int &pivot)
	{
		m = -10000000;

		for (int i = r; i < A.n_rows; i++)
		{
			if (abs(A(i, c)) > m)
			{
				m = abs(A(i, c));
				pivot = i;
			}
		}
	}

	ZSPACE_INLINE void zUtilsCore::rref_swapRows(mat &A, int &r, int &c, int &pivot)
	{
		for (int i = c; i < A.n_cols; i++)
		{
			double temp = (A(pivot, i));

			A(pivot, i) = A(r, i);
			A(r, i) = temp;
		}
	}

	ZSPACE_INLINE void zUtilsCore::rref_normaliseRow(mat &A, int &r, int &c)
	{

		double pivotVal = A(r, c);

		for (int i = c; i < A.n_cols; i++)
		{
			A(r, i) /= pivotVal;
		}
	}

	ZSPACE_INLINE void zUtilsCore::rref_eliminateColumn(mat &A, int &r, int &c)
	{
		mat C(A.n_rows - 1, 1);
		int rowCounter = 0;
		for (int i = 0; i < A.n_rows; i++)
		{
			if (i == r) continue;

			C(rowCounter, 0) = A(i, c);
			rowCounter++;
		}

		mat D(1, A.n_cols);
		D.row(0) = A.row(r);

		mat CD = C * D;
		rowCounter = 0;

		for (int i = 0; i < A.n_rows; i++)
		{
			if (i == r) continue;

			for (int j = c; j < A.n_cols; j++)
			{
				A(i, j) -= CD(rowCounter, j);
			}

			rowCounter++;
		}
	}

#endif

#endif

}