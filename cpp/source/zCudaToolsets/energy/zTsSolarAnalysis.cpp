#include <headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

namespace zSpace
{
	///////////////////////// Constructor - Overload - Destructor /////////////////////////

	zTsSolarAnalysis::zTsSolarAnalysis()
	{
		objMesh = nullptr;
	}

	zTsSolarAnalysis::zTsSolarAnalysis(zObjMesh &_objMesh, string &path)
	{
		objMesh = &_objMesh;
		fnMesh = zFnMesh(*objMesh);

		import_EPW(path);
	}

	zTsSolarAnalysis::~zTsSolarAnalysis()
	{

	}

	///////////////////////// Methods /////////////////////////

	bool zTsSolarAnalysis::import_EPW(string &path)
	{
		ifstream myfile;
		myfile.open(path.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
			return false;
		}

		epwData.dataPoints.clear();

		while (!myfile.eof())
		{
			string str;
			getline(myfile, str);

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() > 0)
			{
				if (perlineData.size() == 10)
				{
					epwData.location = perlineData[1];
					epwData.latitude = atof(perlineData[6].c_str());
					epwData.longitude = atof(perlineData[7].c_str());
					epwData.timezone = atof(perlineData[8].c_str());
					epwData.elevation = atof(perlineData[9].c_str());
				}

				if (perlineData.size() == 35)
				{
					zEPWDataPoint tempDataPoint;

					tempDataPoint.year = atoi(perlineData[0].c_str());
					tempDataPoint.month = atoi(perlineData[1].c_str());
					tempDataPoint.day = atoi(perlineData[2].c_str());
					tempDataPoint.hour = atoi(perlineData[3].c_str());
					tempDataPoint.minute = atoi(perlineData[4].c_str());

					tempDataPoint.db_temperature = atof(perlineData[6].c_str());
					tempDataPoint.pressure = atof(perlineData[9].c_str());
					tempDataPoint.radiation = atof(perlineData[12].c_str());

					epwData.dataPoints.push_back(tempDataPoint);
				}
			}
		}

		myfile.close();

		printf("\nFile Imported Succesfully \n");

		cout << "Location: " << epwData.location
			<< ", Latitude: " << epwData.latitude
			<< ", Longitude: " << epwData.longitude
			<< ", Timezone: " << epwData.timezone << " GMT"
			<< ", Elevation: " << epwData.elevation << endl;

		printf("Data Points: %i \n", epwData.dataPoints.size());

		return true;
	}

	void zTsSolarAnalysis::createMap()
	{
		for (int i = 0; i < epwData.dataPoints.size(); i++)
		{
			MultiKey key = MultiKey(epwData.dataPoints[i].year, epwData.dataPoints[i].month, epwData.dataPoints[i].day, epwData.dataPoints[i].hour);
			multimap<MultiKey, double>::const_iterator iter = radiationMap.begin();
			radiationMap.insert(iter, pair<MultiKey, double>(key, epwData.dataPoints[i].radiation));

			unordered_map<int, int>::const_iterator it = yearsMap.find(epwData.dataPoints[i].year);

			if (it != yearsMap.end())
			{
				years[it->second].months.push_back(epwData.dataPoints[i].month);
				years[it->second].days.push_back(epwData.dataPoints[i].day);
				years[it->second].hours.push_back(epwData.dataPoints[i].hour);
			}
			else
			{
				zYear tempYear;
				tempYear.id = epwData.dataPoints[i].year;

				years.push_back(tempYear);

				years[years.size() - 1].months.push_back(epwData.dataPoints[i].month);
				years[years.size() - 1].days.push_back(epwData.dataPoints[i].day);
				years[years.size() - 1].hours.push_back(epwData.dataPoints[i].hour);

				yearsMap[epwData.dataPoints[i].year] = years.size() - 1;
			}
		}
	}

	void zTsSolarAnalysis::computeSunPosition(int day, int hour, int minute, double longitude, double latitude, double timezone, double &azimuth, double &zenith)
	{
		///// Fractional Year (y) /////
		double y = ((2 * PI) / 365) * (day - 1 + ((hour - 12) / 24));

		///// Time Equation (minutes) /////
		double eqtime = 229.18*(0.000075 + 0.001868 * cos(y) - 0.032077 * sin(y) - 0.014615 * cos(2 * y) - 0.040849 * sin(2 * y));

		///// Solar Declination Angle (radians) /////
		double declination = 0.006918 - 0.399912 * cos(y) + 0.070257 * sin(y) - 0.006758 * cos(2 * y) + 0.000907 *sin(2 * y) - 0.002697 *cos(3 * y) + 0.00148 * sin(3 * y);

		///// Time Offset (minutes) /////
		double time_offset = eqtime + 4 * longitude - 60 * timezone;

		///// True Solar Time (minutes) /////
		double solar_time = hour * 60 + minute + 1 / 60 + time_offset;

		///// Solar Hour Angle (degrees) /////
		double solar_hour_angle = (solar_time / 4) - 180;

		///// Solar Zenith Angle (degrees) /////
		double theta = sin(latitude) * sin(declination) + cos(latitude)* cos(declination) * cos(solar_hour_angle);

		zenith = cos(theta);

		///// Solar Azimuth Angle (degrees) /////
		double theta2 = -(sin(latitude) * cos(theta) - sin(declination)) / (cos(latitude) * sin(theta));

		azimuth = cos(180 - theta2);
	}

	zVector zTsSolarAnalysis::computeSun(int year, int month, int day, int hour)
	{
		zVector out;

		double jd = GregorianToJulian(year, month, day, hour, 1,1);
		cout << jd << endl;

		double n = jd - 2451545.0;

		double L = 280.460 + 0.9856474 * n;
		double g = 357.528 + 0.9856003 * n;

		double lambda = L + 1.915 * sin(g) + 0.020 * sin(2*g);
		double R = 1.00014 - 0.01671 * cos(g) - 0.00014 * cos(2 * g);

		double epsilon = 23.439 - 0.0000004 * n;

		out.x = R * cos(epsilon) * cos(lambda);
		out.y = R * cos(epsilon) * sin(lambda);
		out.z = R * sin(epsilon);
		
		return out;
	}

	void zTsSolarAnalysis::computeSPA_EPW(int idDataPoint)
	{
		SPA.year = epwData.dataPoints[idDataPoint].year;
		SPA.month = epwData.dataPoints[idDataPoint].month;
		SPA.day = epwData.dataPoints[idDataPoint].day;
		SPA.hour = 12;
		SPA.minute = 30;
		SPA.second = 30;
		SPA.timezone = epwData.timezone;
		SPA.delta_ut1 = 0;
		SPA.delta_t = 67;
		SPA.longitude = epwData.longitude;
		SPA.latitude = epwData.latitude;
		SPA.elevation = epwData.elevation;

		zDomainDouble dom0(31000, 120000);
		zDomainDouble dom1(0, 5000);
		SPA.pressure = coreUtils.ofMap(epwData.dataPoints[idDataPoint].pressure,dom0, dom1);

		SPA.temperature = epwData.dataPoints[idDataPoint].db_temperature;
		SPA.slope = 0;
		SPA.azm_rotation = -10;
		SPA.atmos_refract = 0.5667;
		SPA.function = SPA_ALL;

		int result = spa_calculate(&SPA);

		if (result != 0)  //check for SPA errors
			printf("SPA Error Code: %d\n", result);
	}

	void zTsSolarAnalysis::computeSPA(int year, int month, int day, int hour)
	{
		SPA.year = year;
		SPA.month = month;
		SPA.day = day;
		SPA.hour = hour;
		SPA.minute = 30;
		SPA.second = 30;
		SPA.timezone = epwData.timezone;
		SPA.delta_ut1 = 0;
		SPA.delta_t = 67;
		SPA.longitude = epwData.longitude;
		SPA.latitude = epwData.latitude;
		SPA.elevation = epwData.elevation;
		SPA.pressure = 1000;
		SPA.temperature = 10;
		SPA.slope = 0;
		SPA.azm_rotation = 0;
		SPA.atmos_refract = 0.5667;
		SPA.function = SPA_ALL;

		int result = spa_calculate(&SPA);

		if (result != 0)  //check for SPA errors
			printf("SPA Error Code: %d\n", result);
	}

	void zTsSolarAnalysis::computeSPA(int year, int month, int day, int hour, int minute, int second, double timezone, double longitude, double latitude, double elevation)
	{
		SPA.year = year;
		SPA.month = month;
		SPA.day = day;
		SPA.hour = hour;
		SPA.minute = minute;
		SPA.second = second;
		SPA.timezone = timezone;
		SPA.delta_ut1 = 0;
		SPA.delta_t = 67;
		SPA.longitude = longitude;
		SPA.latitude = latitude;
		SPA.elevation = elevation;
		SPA.pressure = 1000;
		SPA.temperature = 10;
		SPA.slope = 0;
		SPA.azm_rotation = 10;
		SPA.atmos_refract = 0.5667;
		SPA.function = SPA_ALL;

		int result = spa_calculate(&SPA);

		if (result == 0)  //check for SPA errors
		{
			//display the results inside the SPA structure

			printf("\nJulian Day:    %.6f\n", SPA.jd);
			printf("L:             %.6e degrees\n", SPA.l);
			printf("B:             %.6e degrees\n", SPA.b);
			printf("R:             %.6f AU\n", SPA.r);
			printf("H:             %.6f degrees\n", SPA.h);
			printf("Delta Psi:     %.6e degrees\n", SPA.del_psi);
			printf("Delta Epsilon: %.6e degrees\n", SPA.del_epsilon);
			printf("Epsilon:       %.6f degrees\n", SPA.epsilon);
			printf("Zenith:        %.6f degrees\n", SPA.zenith);
			printf("Azimuth:       %.6f degrees\n", SPA.azimuth);
			printf("Incidence:     %.6f degrees\n", SPA.incidence);

			float minimum = 60.0*(SPA.sunrise - (int)(SPA.sunrise));
			float sec = 60.0*(minimum - (int)minimum);
			printf("Sunrise:       %02d:%02d:%02d Local Time\n", (int)(SPA.sunrise), (int)minimum, (int)sec);

			minimum = 60.0*(SPA.sunset - (int)(SPA.sunset));
			sec = 60.0*(minimum - (int)minimum);
			printf("Sunset:        %02d:%02d:%02d Local Time\n", (int)(SPA.sunset), (int)minimum, (int)sec);

		}
		else printf("SPA Error Code: %d\n", result);
	}

	///////////////////////// Utils /////////////////////////
	
	double zTsSolarAnalysis::GregorianToJulian(int Y, int M, int D, int H, int MN, int S)
	{

		double JDN = (1461 * (Y + 4800 + (M - 14) / 12)) / 4 + (367 *(M - 2 - 12 *((M - 14) / 12))) / 12 -(3 *((Y + 4900 + (M - 14) / 12) / 100)) / 4 + D - 32075;

		double JD = JDN + ((H - 12) / 24) + (MN / 1440) + (S / 86400);

		return JD;
	}

	void zTsSolarAnalysis::JulianToGregorian(double julian, int & year, int & month, int & day)
	{
		double a = julian + 32044;
		double	b = (4 * a + 3) / 146097;
		double	c = a - (b * 146097) / 4;

		double d = (4 * c + 3) / 1461;
		double e = c - (1461 * d) / 4;
		double m = (5 * e + 2) / 153;

		day = e - (153 * m + 2) / 5 + 1;
		month = m + 3 - 12 * (m / 10);
		year = b * 100 + d - 4800 + m / 10;
	}

	zVector zTsSolarAnalysis::SphericalToCartesian(double azimuth, double zenith, double radius)
	{
		zVector out;

		double azimuth_radians = (azimuth * PI) / 180;
		double zenith_radians = (zenith * PI) / 180;

		out.x = radius * cos(zenith_radians) * sin(azimuth_radians);
		out.y = radius * cos(zenith_radians) * cos(azimuth_radians);
		out.z = radius * sin(zenith_radians);

		return out;
	}

	void zTsSolarAnalysis::CartesianToSpherical(zVector input, double & radius, double & zenith, double & azimuth)
	{
		radius = sqrt(pow(input.x, 2) + pow(input.y, 2) + pow(input.z, 2));
		azimuth = atan(input.y / input.x);
		zenith = acos(input.z / radius);
	}
}
