#include <headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

namespace zSpace
{
	///////////////////////// Constructor - Overload - Destructor /////////////////////////

	ZSPACE_INLINE zTsSolarAnalysis::zTsSolarAnalysis()
	{
		objMesh = nullptr;
	}

	ZSPACE_INLINE zTsSolarAnalysis::zTsSolarAnalysis(zObjMesh &_objMesh, string &epw_path, string &mesh_path)
	{
		objMesh = &_objMesh;
		fnMesh = zFnMesh(*objMesh);

		fnMesh.from(mesh_path, zOBJ, false);

		import_EPW(epw_path);
	}

	ZSPACE_INLINE zTsSolarAnalysis::~zTsSolarAnalysis()
	{
		
	}

	///////////////////////// Import /////////////////////////

	ZSPACE_INLINE bool zTsSolarAnalysis::import_EPW(string &path)
	{
		ifstream myfile;
		myfile.open(path.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
			return false;
		}

		bool startCount = false;
		while (!myfile.eof())
		{
			string str;
			getline(myfile, str);

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() > 0)
			{
				if (perlineData[0] == "LOCATION")
				{
					epwData.location = perlineData[1];
					epwData.latitude = atof(perlineData[6].c_str());
					epwData.longitude = atof(perlineData[7].c_str());
					epwData.timeZone = atoi(perlineData[8].c_str());
				}

				if (startCount)
				{
					zEPWData tempDataPoint;

					zDate tempDate;
					tempDate.month = atoi(perlineData[1].c_str());
					tempDate.day = atoi(perlineData[2].c_str());
					tempDate.hour = atoi(perlineData[3].c_str());

					tempDataPoint.location = epwData.location;
					tempDataPoint.year = atoi(perlineData[0].c_str());
					tempDataPoint.dbTemperature = atof(perlineData[6].c_str());
					tempDataPoint.pressure = atof(perlineData[9].c_str());
					tempDataPoint.radiation = atof(perlineData[12].c_str());
					tempDataPoint.humidity = atof(perlineData[8].c_str());
					tempDataPoint.windDirection = atof(perlineData[20].c_str());
					tempDataPoint.windSpeed = atof(perlineData[21].c_str());

					dataMap[tempDate] = tempDataPoint;
				}

				if (perlineData[0] == "DATA PERIODS") startCount = true;
			}

		}

		myfile.close();

		printf("\nFile Imported Succesfully \n");

		cout << "Location: " << epwData.location << endl;
		cout << "long: " << epwData.longitude << endl;
		cout << "lat: " << epwData.latitude << endl;

		return true;
	}

	ZSPACE_INLINE vector<zEPWData> zTsSolarAnalysis::createMap(zDomainDate _dDomain)
	{
		vector<zEPWData> userDataMap;
		if (_dDomain.min.year > _dDomain.max.year) return userDataMap;

		double  julianDateMin = GregorianToJulian(_dDomain.min.year, _dDomain.min.month, _dDomain.min.day, _dDomain.min.hour, _dDomain.min.minute, 0);
		double julianDateMax = GregorianToJulian(_dDomain.max.year, _dDomain.max.month, _dDomain.max.day, _dDomain.max.hour, _dDomain.max.minute, 0);


		cout << endl << "yearmin: " << _dDomain.min.year << " monthmin: " << _dDomain.min.month << " day: " << _dDomain.min.day << " hour: " << _dDomain.min.hour << endl;
		cout << endl << "yearmax: " << _dDomain.max.year << " monthmax: " << _dDomain.max.month << " day: " << _dDomain.max.day << " hour: " << _dDomain.max.hour << endl;

		printf("\n j_date min %1.6f ", julianDateMin);
		printf("\n j_date max %1.6f ", julianDateMax);


		julianDateMin *= 1000000.0f;
		julianDateMax *= 1000000.0f;

		long long int start = floor(julianDateMin);
		long long int end = floor(julianDateMax);


		for (long long int i = start; i < end; i += 41666)
		{
			double j_date = (double)(i + 41666) / 1000000.0;
			printf("\n j_date %1.5f ", j_date);

			zDate gDate;
			JulianToGregorian(j_date, gDate);
			if (gDate.hour == 0)
			{
				gDate.hour = 24;
				gDate.day -= 1;
			}

			cout << endl << "year: " << gDate.year << " month: " << gDate.month << " day: " << gDate.day << " hour: " << gDate.hour << endl;

			std::unordered_map<zDate, zEPWData>::iterator it;
			it = dataMap.find(gDate);

			if (it != dataMap.end())
			{
				cout << endl << "t: " << it->second.dbTemperature << endl;
			}
		}
	}

	///////////////////////// Sun Data /////////////////////////

	ZSPACE_INLINE void zTsSolarAnalysis::getTemperature(zDomainDate _dDomain, zDoubleArray&_temperatures)
	{

		if (_dDomain.min.year > _dDomain.max.year) return;
		else if (_dDomain.min.year == _dDomain.max.year && _dDomain.min.month > _dDomain.max.month) return;
		else if (_dDomain.min.year == _dDomain.max.year && _dDomain.min.month == _dDomain.max.month && _dDomain.min.day == _dDomain.max.day) return;

		vector<std::unordered_map<zDate, zEPWData>::iterator> dataRange;
		zDate calculator;

		_temperatures.clear();
		_temperatures.assign(calculator.getDifference(_dDomain.min, _dDomain.max), double());

		int count = 0;
		for (int i = _dDomain.min.month; i < _dDomain.max.month; i++)
		{
			for (int j = _dDomain.min.day; j < _dDomain.max.day; j++)
			{
				for (int k = _dDomain.min.hour; k < _dDomain.max.hour; k++)
				{
					std::unordered_map<zDate, zEPWData>::iterator it;
					zDate dateTemp = zDate(i, j, k);
					it = dataMap.find(dateTemp);

					if (it != dataMap.end())
					{
						cout << endl << "t: " << it->second.dbTemperature << endl;
						_temperatures[count] = it->second.dbTemperature;
					}

					count++;
				}
			}
		}

		cout << endl << "count: " << count << endl;

	}

	ZSPACE_INLINE void zTsSolarAnalysis::sunData(zSunPosition&_sunposition, zDate date)
	{
		int Year = date.year;
		int Month = date.month;
		int Day = date.day;
		int Hour = date.hour;
		int Min = date.minute;
		float LocalTime = date.hour + (date.minute / 60.0);

		double JD = GregorianToJulian(Year, Month, Day, Hour, Min, 0);

		double phi = epwData.latitude;
		double lambda = epwData.longitude;

		double n = JD - 2451545.0;

		double LDeg = fmod((280.460 + 0.9856474 * n), 360);
		double gDeg = fmod((357.528 + 0.9856003 * n), 360);

		double LambdaDeg = LDeg + 1.915 * sin(gDeg * DEG_TO_RAD) + 0.01997 * sin(2 * gDeg * DEG_TO_RAD);

		double epsilonDeg = 23.439 - 0.0000004 * n;

		double alphaDeg;

		alphaDeg = atan(cos(epsilonDeg * DEG_TO_RAD) * tan(LambdaDeg * DEG_TO_RAD));

		alphaDeg *= RAD_TO_DEG;
		if (cos(LambdaDeg  * DEG_TO_RAD) < 0)
		{
			alphaDeg += (4 * (atan(1.0) * RAD_TO_DEG));
		}

		double deltaDeg = asin(sin(epsilonDeg * DEG_TO_RAD) * sin(LambdaDeg  * DEG_TO_RAD)) * RAD_TO_DEG;

		double JDNull = GregorianToJulian(Year, Month, Day, 0, 0, 0);

		double TNull = ((JDNull - 2451545.0) / 36525);
		double T = LocalTime - epwData.timeZone;

		double thetaGh = 6.697376 + 2400.05134 * TNull + 1.002738 * T;

		double thetaG = fmod(thetaGh * 15, 360);
		double theta = thetaG + lambda;

		double tauDeg = theta - alphaDeg;

		double denom = (cos(tauDeg  * DEG_TO_RAD)*sin(phi  * DEG_TO_RAD) - tan(deltaDeg  * DEG_TO_RAD)*cos(phi  * DEG_TO_RAD));
		double aDeg = atan(sin(tauDeg  * DEG_TO_RAD) / denom);
		aDeg *= RAD_TO_DEG;
		if (denom < 0) aDeg = aDeg + 180;
		//add 180 to azimuth to compute from the north.
		aDeg += 180;

		double hDeg = asin(cos(deltaDeg  * DEG_TO_RAD)*cos(tauDeg  * DEG_TO_RAD)*cos(phi  * DEG_TO_RAD) + sin(deltaDeg  * DEG_TO_RAD)*sin(phi  * DEG_TO_RAD));
		hDeg *= RAD_TO_DEG;

		double valDeg = hDeg + (10.3 / (hDeg + 5.11));
		double RDeg = 1.02 / (tan(valDeg * DEG_TO_RAD));

		double hRDeg = hDeg + (RDeg / 60);

		_sunposition.azimuth = aDeg;
		_sunposition.altitude = hRDeg;

	}

	ZSPACE_INLINE void zTsSolarAnalysis::getSunPosition(zVector&_sunPos, zDate &date, float radius)
	{
		zSunPosition sunDataPos;
		sunData(sunDataPos, date);

		zVector zPos = zVector(0, 1, 0); // Computed from North (0,1,0).

		zPos = zPos.rotateAboutAxis(zVector(0, 0, 1), sunDataPos.azimuth);

		zVector axis = zVector(0, 0, 1) ^ zPos;
		zPos = zPos.rotateAboutAxis(axis, sunDataPos.altitude);

		zPos.normalize();
		zPos *= radius;

		_sunPos = zPos;
	}

	///////////////////////// CUDA /////////////////////////

	ZSPACE_INLINE void zTsSolarAnalysis::initializeCUDA(char* _path, zCUDADate *_date, zCUDAVector *_faceNormals, double *_anglesOut)
	{
		cuda_Path = _path;
		cuda_Date = _date;
		cuda_faceNormals = _faceNormals;
		angles_out = _anglesOut;

		fnMesh.computeMeshNormals();
		zVector* fn = fnMesh.getRawFaceNormals();

		for (int i = 0; i < fnMesh.numPolygons(); i++)
			cuda_faceNormals[i].v = make_float4(fn[i].x, fn[i].y, fn[i].z, fn[i].w);

		copy_HostToDevice(cuda_Path,cuda_faceNormals, cuda_Date, fnMesh.numPolygons(), angles_out);
	}

	ZSPACE_INLINE void zTsSolarAnalysis::computeRadiation()
	{
		callKernel();

		angles_out = copy_DeviceToHost();
	}

	///////////////////////// Utils /////////////////////////

	ZSPACE_INLINE double zTsSolarAnalysis::GregorianToJulian(int year, int month, int day, int hour, int minute, int second)
	{
		float y, m, d, b;

		if (month > 2)
		{
			y = year;
			m = month;
		}
		else
		{
			y = year - 1;
			m = month + 12;
		}

		d = day + (float)(hour / 24.0) + (float)(minute / 1440.0) + (float)(second / 86400.0);
		b = 2 - floor(y / 100) + floor(y / 400);
		double jd = floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + d + b - 1524.5;

		return jd;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::JulianToGregorian(double julian, zDate&_gDate)
	{
		double unixTime = (julian - 2440587.5) * 86400;
		time_t temptime = unixTime;

		struct tm timeinfo;
		gmtime_s(&timeinfo, &temptime);

		_gDate = zDate(timeinfo.tm_year, timeinfo.tm_mon, timeinfo.tm_mday, timeinfo.tm_hour, timeinfo.tm_min);
	}

	ZSPACE_INLINE zVector zTsSolarAnalysis::SphericalToCartesian(double azimuth, double zenith, double radius)
	{
		zVector out;

		double azimuth_radians = (azimuth * PI) / 180;
		double zenith_radians = (zenith * PI) / 180;

		out.x = radius * cos(zenith_radians) * sin(azimuth_radians);
		out.y = radius * cos(zenith_radians) * cos(azimuth_radians);
		out.z = radius * sin(zenith_radians);

		return out;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::CartesianToSpherical(zVector input, double & radius, double & zenith, double & azimuth)
	{
		radius = sqrt(pow(input.x, 2) + pow(input.y, 2) + pow(input.z, 2));
		azimuth = atan(input.y / input.x);
		zenith = acos(input.z / radius);
	}

	//////////////////////// Display ////////////////////////////

	ZSPACE_INLINE void zTsSolarAnalysis::getSunPath(zVectorArray&_sunPath, zDomainDate dDate, float Timezone, float radius, int interval)
	{
		_sunPath.clear();

		double initSumSolstice = GregorianToJulian(dDate.min.year, dDate.min.month, dDate.min.day, dDate.min.hour, dDate.min.minute, 0);
		double endSumSolstice = GregorianToJulian(dDate.max.year, dDate.max.month, dDate.max.day, dDate.max.hour, dDate.max.minute, 0);

		initSumSolstice *= 1000000.f;
		endSumSolstice *= 1000000.f;

		long long int init = floor(initSumSolstice);
		long long int end = floor(endSumSolstice);

		long long int rangeSumSolstice = end - init;

		long long int increment = (long long int)(rangeSumSolstice / interval);

		_sunPath.assign(interval + 1, zVector());

		int count = 0;
		for (long long int i = init; i < end; i += increment)
		{
			zDate date;
			double d = (double)i / 1000000.0;

			JulianToGregorian(d, date);

			zVector sunPos;
			getSunPosition(sunPos, date, radius);
			_sunPath[count] = sunPos;

			count++;
		}

	}
}
