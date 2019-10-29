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


#include<headers/zArchGeom/zAgColumn.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zAgColumn::zAgColumn(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zAgColumn::~zAgColumn() {}

	//---- SET METHODS

	void zAgColumn::CreateColumn(zVector &primary, zVector &secondary, zVector &z_, float height_)
	{
		zVector xP = primary;
		zVector yS = secondary;

		pointArray.clear();
		polyCount.clear();
		polyConnect.clear();

		xP.normalize();
		yS.normalize();
		z_.normalize();

		x = xP;
		y = yS;
		z = z_;

		zVector midP = (xP + yS) / 2;
		midP.normalize();
		midP *= nodeDepth;

		float xDepth = nodeDepth * 0.8;
		float yDepth = nodeDepth * 0.8;

		for (int i = 0; i < 14; i++)
		{
			zVector vPos = position;

			if (i == 13) vPos += z_ * height_ + (xP * 20);
			else if (i == 12) vPos += z_ * height_ + (yS * 20);
			else if (i == 11) vPos += z_ * nodeHeight + (yS * 20);
			else if (i == 10) vPos += z_ * nodeHeight + (xP * 20);
			else if (i == 9) vPos += z_ * beamB_Height + (xP * (xDepth * 0.75));
			else if (i == 8) vPos += z_ * beamA_Height + (yS * (yDepth * 0.6));
			else if (i == 7) vPos += z_ * beamA_Height + (yS * (yDepth * 0.85));

			else if (i == 6)
			{
				vPos += z_ * beamB_Height + (xP * xDepth);
				c = vPos;
			}
			else if (i == 5)
			{
				vPos += z_ * beamB_Height + midP;
				b = vPos;
			}
			else if (i == 4)
			{
				vPos += z_ * beamA_Height + (yS * yDepth);
				a = vPos;
			}

			else if (i == 3) vPos += (yS * yDepth / 2);
			else if (i == 2) vPos += midP / 2;
			else if (i == 1) vPos += (xP * xDepth / 2);
			else if (i == 0) vPos = vPos; //nothing

			pointArray.push_back(vPos);
		}

		//array of ordered vertices
		polyConnect.push_back(0);
		polyConnect.push_back(1);
		polyConnect.push_back(2);
		polyConnect.push_back(3);

		polyConnect.push_back(3);
		polyConnect.push_back(2);
		polyConnect.push_back(5);
		polyConnect.push_back(4);

		polyConnect.push_back(2);
		polyConnect.push_back(1);
		polyConnect.push_back(6);
		polyConnect.push_back(5);

		polyConnect.push_back(5);
		polyConnect.push_back(6);
		polyConnect.push_back(7);
		polyConnect.push_back(4);

		polyConnect.push_back(7);
		polyConnect.push_back(6);
		polyConnect.push_back(9);
		polyConnect.push_back(8);

		polyConnect.push_back(8);
		polyConnect.push_back(9);
		polyConnect.push_back(10);
		polyConnect.push_back(11);

		polyConnect.push_back(11);
		polyConnect.push_back(10);
		polyConnect.push_back(13);
		polyConnect.push_back(12);

		//number of vertices per face array
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(4);
		polyCount.push_back(4);
	}

}