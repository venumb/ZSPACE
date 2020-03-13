#include <headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSolarAnalysis::zTsSolarAnalysis() {}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSolarAnalysis::~zTsSolarAnalysis() {}

	//---- SET METHODS


	ZSPACE_INLINE void zTsSolarAnalysis::setNormals(const float *_normals, int _numNormals, bool EPWread)
	{
		normals = new float[_numNormals];;
		std::copy(_normals, _normals + _numNormals, normals);

		numNorms = _numNormals;

		setMemory();

		std::copy(normals, normals + (numNorms), norm_sunvecs);
		std::copy(sunVecs_hour, sunVecs_hour + MAX_SUNVECS_HOUR, norm_sunvecs + numNorms);

		if (EPWread)
		{
			std::copy(normals, normals + (numNorms), cummulativeRadiation);
			std::copy(epwData_radiation, epwData_radiation + MAX_SUNVECS_HOUR, cummulativeRadiation + numNorms);
		}
	}

	ZSPACE_INLINE bool zTsSolarAnalysis::setEPWData(string path)
	{
		ifstream myfile;
		myfile.open(path.c_str());

		if (myfile.fail())
		{
			cout << " error in opening file  " << path.c_str() << endl;
			return false;
		}

		epwData_radiation = new float[MAX_SUNVECS_HOUR];

		bool leapYear = (dDate.min.tm_year % 4 == 0) ? true : false;

		bool startCount = false;
		int count = 0;

		while (!myfile.eof())
		{
			string str;
			getline(myfile, str);

			vector<string> perlineData = coreUtils.splitString(str, ",");

			if (perlineData.size() > 0)
			{
				if (perlineData[0] == "LOCATION")
				{
					//location.location = perlineData[1];
					location.latitude = atof(perlineData[6].c_str());
					location.longitude = atof(perlineData[7].c_str());
					location.timeZone = atoi(perlineData[8].c_str());
				}

				if (startCount)
				{
							
					epwData_radiation[count * 3 + 0] = atof(perlineData[5].c_str()); // temperature
					epwData_radiation[count * 3 + 1] = atof(perlineData[8].c_str()); //pressure
					epwData_radiation[count * 3 + 2] = atof(perlineData[11].c_str()); //radiation

					if (leapYear && count == (59 *24)) // Feb 28
					{
						for (int k = 0; k < 24; k++)
						{
							count++;

							epwData_radiation[count * 3 + 0] = epwData_radiation[(count - 24) * 3 + 0]; // temperature
							epwData_radiation[count * 3 + 1] = epwData_radiation[(count - 24) * 3 + 1]; //pressure
							epwData_radiation[count * 3 + 2] = epwData_radiation[(count - 24) * 3 + 2]; //radiation
						}
						
					}

					if (!leapYear && count == (365 * 24)) // Dec 31
					{
						for (int k = 0; k < 24; k++)
						{
							count++;

							epwData_radiation[count * 3 + 0] = INVALID_VAL; // temperature
							epwData_radiation[count * 3 + 1] = INVALID_VAL; //pressure
							epwData_radiation[count * 3 + 2] = INVALID_VAL; //radiation
						}
					}

					//printf("\n %i | %1.2f ", count * 3 + 2, epwData_radiation[count * 3 + 2]);
					
					//epwData[count].humidity = atof(perlineData[7].c_str());
					//epwData[count].windDirection = atof(perlineData[19].c_str());
					//epwData[count].windSpeed = atof(perlineData[20].c_str());							

					count++;
				}

				if (perlineData[0] == "DATA PERIODS") startCount = true;
			}

		}


		printf("\n count : %i ", count);
		myfile.close();

		return true;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setDomain_Dates(zDomainDate & _dDate)
	{
		dDate = _dDate;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setDomain_Colors(zDomainColor & _dColor)
	{
		dColor = _dColor;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setLocation(zLocation & _location)
	{
		location = _location;
	}

	//---- GET METHODS

	ZSPACE_INLINE int zTsSolarAnalysis::numNormals()
	{
		return numNorms;
	}

	ZSPACE_INLINE int zTsSolarAnalysis::numSunVecs()
	{
		return MAX_SUNVECS_HOUR;
	}

	ZSPACE_INLINE int zTsSolarAnalysis::numDataPoints()
	{
		return numData;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawNormals()
	{
		return normals;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawColors()
	{
		return colors;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawNormals_SunVectors()
	{
		return norm_sunvecs;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawSunVectors_hour()
	{
		return sunVecs_hour;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawSunVectors_day()
	{
		return sunVecs_days;
	}

	ZSPACE_INLINE float * zTsSolarAnalysis::getRawCompassPts()
	{
		return compassPts;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawEPWRadiation()
	{
		return epwData_radiation;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawCummulativeRadiation()
	{
		return cummulativeRadiation;
	}

	ZSPACE_INLINE zVector zTsSolarAnalysis::getSunPosition(zDate &date)
	{
		float LocalTime = date.tm_hour + (date.tm_min / 60.0);

		double JD = date.toJulian();

		double phi = location.latitude;
		double lambda = location.longitude;

		double n = JD - 2451545.0;

		double LDeg = (double)fmod((280.460 + 0.9856474 * n), 360.0);
		double gDeg = (double)fmod((357.528 + 0.9856003 * n), 360.0);

		double LambdaDeg = LDeg + 1.915 * sin(gDeg * DEG_TO_RAD) + 0.01997 * sin(2 * gDeg * DEG_TO_RAD);

		double epsilonDeg = 23.439 - 0.0000004 * n;

		double alphaDeg;
		alphaDeg = atan(cos(epsilonDeg * DEG_TO_RAD) * tan(LambdaDeg * DEG_TO_RAD));
		alphaDeg *= RAD_TO_DEG;
		if (cos(LambdaDeg  * DEG_TO_RAD) < 0)	alphaDeg += (4 * (atan(1.0) * RAD_TO_DEG));

		double deltaDeg = asin(sin(epsilonDeg * DEG_TO_RAD) * sin(LambdaDeg  * DEG_TO_RAD)) * RAD_TO_DEG;

		zDate dZero(date.tm_year, date.tm_mon, date.tm_mday, 0, 0);
		double JDNull = dZero.toJulian();

		double TNull = ((JDNull - 2451545.0) / 36525);
		double T = LocalTime - location.timeZone;

		double thetaGh = 6.697376 + 2400.05134 * TNull + 1.002738 * T;

		double thetaG = (double)fmod(thetaGh * 15.0, 360.0);
		double theta = thetaG + lambda;

		double tauDeg = theta - alphaDeg;

		double denom = (cos(tauDeg  * DEG_TO_RAD)*sin(phi  * DEG_TO_RAD) - tan(deltaDeg  * DEG_TO_RAD)*cos(phi  * DEG_TO_RAD));
		double aDeg = atan(sin(tauDeg  * DEG_TO_RAD) / denom);
		aDeg *= RAD_TO_DEG;
		if (denom < 0) aDeg = aDeg + 180;
		aDeg += 180; //add 180 to azimuth to compute from the north.

		double hDeg = asin(cos(deltaDeg  * DEG_TO_RAD)*cos(tauDeg  * DEG_TO_RAD)*cos(phi  * DEG_TO_RAD) + sin(deltaDeg  * DEG_TO_RAD)*sin(phi  * DEG_TO_RAD));
		hDeg *= RAD_TO_DEG;

		double valDeg = hDeg + (10.3 / (hDeg + 5.11));
		double RDeg = 1.02 / (tan(valDeg * DEG_TO_RAD));

		double hRDeg = hDeg + (RDeg / 60);

		return coreUtils.sphericalToCartesian(aDeg, hRDeg, 1.0);
	}

	ZSPACE_INLINE zDomainDate zTsSolarAnalysis::getSunRise_SunSet(zDate &date)
	{
		zDomainDate out;

		zDate temp = date;
		temp.tm_hour = 12;
		temp.tm_min = 0;
		temp.tm_sec = 0;

		double jd = temp.toJulian();

		double n = jd - 2451545.0 + 0.0008;

		double js = n - (location.longitude / 360.0);

		double m = (double)fmod((357.5291 + 0.98560028 * js), 360.0) * DEG_TO_RAD; // radians

		double c = 1.9148 * sin(m) + 0.02 * sin(2 * m) + 0.0003 * sin(3 * m);

		double mDeg = m * RAD_TO_DEG;
		double lambdaDeg = (double)fmod((mDeg + c + 180 + 102.9372), 360.0); //deg
		double lambda = lambdaDeg * DEG_TO_RAD;

		double jt = 2451545.0 + js + ((0.0053 * sin(m)) - (0.0069 * sin(2 * lambda)));

		double delta = asin(sin(lambda) * sin(23.44 * DEG_TO_RAD));

		double cosOmega = (sin(-0.83 * DEG_TO_RAD) - (sin(location.latitude * DEG_TO_RAD) * sin(delta))) / (cos(location.latitude * DEG_TO_RAD) * cos(delta));
		double omegaDeg = acos(cosOmega) * RAD_TO_DEG;

		double j;

		//sunrise
		j = jt - (omegaDeg / 360.0);
		out.min.fromJulian(j);

		//sunset
		j = jt + (omegaDeg / 360.0);
		out.max.fromJulian(j);
		return out;
	}

	ZSPACE_INLINE zDomainDate zTsSolarAnalysis::getDomain_Dates()
	{
		return dDate;
	}

	ZSPACE_INLINE zDomainColor zTsSolarAnalysis::getDomain_Colors()
	{
		return dColor;
	}

	ZSPACE_INLINE zLocation zTsSolarAnalysis::getLocation()
	{
		return location;
	}
	   
	//---- COMPUTE METHODS

	ZSPACE_INLINE void zTsSolarAnalysis::computeCummulativeRadiation()
	{
		for(int i =0; i< numNorms; i+= 3)
		
		{
			zVector norm(norm_sunvecs[i + 0], norm_sunvecs[i + 1], norm_sunvecs[i + 2]);

			int sunvecs_offset = numNorms - i;

			float angle = 0;
			int count = 0;
			for (int o = i; o < i + MAX_SUNVECS_HOUR; o += 3)
			{
				int j = o + sunvecs_offset;
				zVector sVec(norm_sunvecs[j + 0], norm_sunvecs[j + 1], norm_sunvecs[j + 2]);

				if (sVec.x != INVALID_VAL && sVec.y != INVALID_VAL && sVec.z != INVALID_VAL)
				{
					angle += norm.angle(sVec);
					count++;
				}


			}
			angle /= count;


			if (angle > 90.0)
			{
				colors[i + 0] = dColor.min.h;
				colors[i + 1] = dColor.min.s;
				colors[i + 2] = dColor.min.v;
			}
			else
			{
				colors[i + 0] = coreUtils.ofMap(angle, 90.0f, 0.0f, dColor.min.h, dColor.max.h);
				colors[i + 1] = coreUtils.ofMap(angle, 90.0f, 0.0f, dColor.min.s, dColor.max.s);
				colors[i + 2] = coreUtils.ofMap(angle, 90.0f, 0.0f, dColor.min.v, dColor.max.v);
			}



		}
	}

	//---- PROTECTED METHODS


	ZSPACE_INLINE void zTsSolarAnalysis::computeSunVectors_Year()
	{
		computeSunVectors_Hour();
		computeSunVectors_Day();
	}

	ZSPACE_INLINE void zTsSolarAnalysis::computeCompass()
	{
		compassPts = new float[COMPASS_SUBD];
		float deg = (float)(360 / (float)(12));

		//zVector pos(0, 1, 0);

		for (int i = 0; i < 2; i++)
		{
			//if (i > 0) pos *= 1.1;
			zVector pos(0, 1 + (i * 0.1), 0);

			for (int j = 0; j < 12; j++)
			{
				int id = (i * 12 + j) * 3;
				compassPts[id] = pos.x;
				compassPts[id + 1] = pos.y;
				compassPts[id + 2] = pos.z;
				pos = pos.rotateAboutAxis(zVector(0, 0, 1), deg);
			}

		}
	}

	//---- PROTECTED METHODS


	ZSPACE_INLINE void zTsSolarAnalysis::setMemory()
	{
		if ((numNorms + MAX_SUNVECS_HOUR) < memSize) return;
		else
		{
			while (memSize < (numNorms + MAX_SUNVECS_HOUR)) memSize += d_MEMORYMULTIPLIER;

			norm_sunvecs = new float[memSize];
			cummulativeRadiation = new float[memSize];
			
			// set to  Num Normals
			colors = new float[memSize];
		}
	}

	ZSPACE_INLINE void zTsSolarAnalysis::computeSunVectors_Hour()
	{
		zDate min = dDate.min;
		min.tm_mon = 1;
		min.tm_mday = 1;
		min.tm_hour = 0;
		min.tm_min = 0;


		zDate max = dDate.max;
		max.tm_mon = 12;
		max.tm_mday = 31;
		max.tm_hour = 23;
		min.tm_min = 0;


		time_t  unixTime_s = min.toUnix();
		time_t  unixTime_e = max.toUnix();

		// get minute domain per day
		zDate minHour = min;
		zDate maxHour(min.tm_year, min.tm_mon, min.tm_mday, max.tm_hour, max.tm_min);

		time_t  unixTime_sh = minHour.toUnix();
		time_t  unixTime_eh = maxHour.toUnix();


		//get total number of vectors
		sunVecs_hour = new float[MAX_SUNVECS_HOUR];

		int hrCount = 0;

		for (time_t hour = unixTime_sh; hour <= unixTime_eh; hour += 3600)
		{
			int dCount = 0;
			for (time_t day = unixTime_s; day <= unixTime_e; day += 86400)
			{
				zDate currentDate;
				currentDate.fromUnix(day + hour - unixTime_s);;

				zVector sunPos = getSunPosition(currentDate);

				if (sunPos.z >= 0)
				{
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 0] = sunPos.x;
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 1] = sunPos.y;
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 2] = sunPos.z;
				}
				else
				{
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 0] = INVALID_VAL;
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 1] = INVALID_VAL;
					sunVecs_hour[(((hrCount * 366) + dCount) * 3) + 2] = INVALID_VAL;
				}

				dCount++; ;
			}

			hrCount++;
		}

    // for non leap years
		if (min.tm_year % 4 != 0)
		{
			for (int i = (365 * 24 * 3); i < MAX_SUNVECS_HOUR; i++)
			{
				sunVecs_hour[i] = INVALID_VAL;				
			}
		}
	}

	ZSPACE_INLINE void zTsSolarAnalysis::computeSunVectors_Day()
	{

		zDate days[7];

		int _year = dDate.min.tm_year;

		days[0] = zDate(_year, 06, 20, 0, 1);
		days[1] = zDate(_year, 12, 21, 0, 1);
		days[2] = zDate(_year, 1, 28, 0, 1);
		days[3] = zDate(_year, 2, 28, 0, 1);
		days[4] = zDate(_year, 3, 21, 0, 1);
		days[5] = zDate(_year, 4, 15, 0, 1);
		days[6] = zDate(_year, 5, 15, 0, 1);

		sunVecs_days = new float[MAX_SUNVECS_DAY];

		for (int i = 0; i < 7; i++)
		{

			zDate init = days[i];
			zDate end = days[i];

			init.tm_hour = 0;
			init.tm_min = 1;
			end.tm_hour = 23;
			end.tm_min = 59;

			time_t  unixTime_s = init.toUnix();
			time_t  unixTime_e = end.toUnix();

			int count = 0;
			for (time_t hour = unixTime_s; hour <= unixTime_e; hour += 3600)
			{
				zDate currentDate;
				currentDate.fromUnix(hour);
				zDomainDate dd = getSunRise_SunSet(currentDate);


				zVector sunPos;

				sunPos = getSunPosition(currentDate);
				if (sunPos.z >= 0)
				{
					sunVecs_days[(((i * 24) + count) * 3) + 0] = sunPos.x;
					sunVecs_days[(((i * 24) + count) * 3) + 1] = sunPos.y;
					sunVecs_days[(((i * 24) + count) * 3) + 2] = sunPos.z;
				}
				else
				{
					sunVecs_days[(((i * 24) + count) * 3) + 0] = INVALID_VAL;
					sunVecs_days[(((i * 24) + count) * 3) + 1] = INVALID_VAL;
					sunVecs_days[(((i * 24) + count) * 3) + 2] = INVALID_VAL;
				}


				if (currentDate.tm_hour == dd.min.tm_hour)
				{
					sunPos = getSunPosition(dd.min);
					sunVecs_days[(((i * 24) + count) * 3) + 0] = sunPos.x;
					sunVecs_days[(((i * 24) + count) * 3) + 1] = sunPos.y;
					sunVecs_days[(((i * 24) + count) * 3) + 2] = sunPos.z;
				}

				if (currentDate.tm_hour == dd.max.tm_hour + 1)
				{
					sunPos = getSunPosition(dd.max);
					sunVecs_days[(((i * 24) + count) * 3) + 0] = sunPos.x;
					sunVecs_days[(((i * 24) + count) * 3) + 1] = sunPos.y;
					sunVecs_days[(((i * 24) + count) * 3) + 2] = sunPos.z;
				}

				count++;
			}
		}
	}

}
