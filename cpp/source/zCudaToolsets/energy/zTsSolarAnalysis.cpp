#include <headers/zCudaToolsets/energy/zTsSolarAnalysis.h>

namespace zSpace
{
	//---- CONSTRUCTOR

	ZSPACE_INLINE zTsSolarAnalysis::zTsSolarAnalysis(){}

	//---- DESTRUCTOR

	ZSPACE_INLINE zTsSolarAnalysis::~zTsSolarAnalysis(){}

	//---- SET METHODS

	ZSPACE_INLINE void zTsSolarAnalysis::setMemory(int _newSize)
	{
		if (_newSize < numNorms) return;
		else
		{
			while (numNorms < _newSize) numNorms += d_MEMORYMULTIPLIER;

			normals = new zVector[numNorms];
			cummulativeRadiation = new float[numNorms];

		}
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setNormals(zVector *_normals, int _numNormals)
	{
		normals = _normals;
		numNorms = _numNormals;
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

		epwData = new zEPWData[8760];

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
							
					epwData[count].dbTemperature = atof(perlineData[6].c_str());
					epwData[count].pressure = atof(perlineData[9].c_str());
					epwData[count].radiation = atof(perlineData[12].c_str());
					epwData[count].humidity = atof(perlineData[8].c_str());
					epwData[count].windDirection = atof(perlineData[20].c_str());
					epwData[count].windSpeed = atof(perlineData[21].c_str());
										
					count++;
				}

				if (perlineData[0] == "DATA PERIODS") startCount = true;
			}

		}

		myfile.close();

		return true;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setDates(zDomainDate & _dDate)
	{
		dDate = _dDate;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setLocation(zLocation & _location)
	{
		location = _location;
	}

	ZSPACE_INLINE void zTsSolarAnalysis::setNorm_SunVecs()
	{
		norm_sunVecs = new zNorm_SunVec[numNorms * numSunVec];

		for (int i = 0; i < numNorms; i++)
		{
			for (int j = 0; j < numSunVec; j++)
			{
				norm_sunVecs[(i*numSunVec) + j].norm = normals[i];
				norm_sunVecs[(i*numSunVec) + j].sunVec = sunVecs[j];
			}
		}
	}

	//---- GET METHODS

	ZSPACE_INLINE int zTsSolarAnalysis::numNormals()
	{
		return numNorms;
	}

	ZSPACE_INLINE int zTsSolarAnalysis::numSunVecs()
	{
		return numSunVec;
	}

	ZSPACE_INLINE int zTsSolarAnalysis::numDataPoints()
	{
		return numData;
	}

	ZSPACE_INLINE zVector* zTsSolarAnalysis::getRawNormals()
	{
		return normals;
	}

	ZSPACE_INLINE zVector* zTsSolarAnalysis::getRawSunVectors()
	{
		return sunVecs;
	}

	ZSPACE_INLINE zEPWData* zTsSolarAnalysis::getRawEPWData()
	{
		return epwData;
	}

	ZSPACE_INLINE float* zTsSolarAnalysis::getRawCummulativeRadiation()
	{
		return cummulativeRadiation;
	}

	ZSPACE_INLINE zVector zTsSolarAnalysis::getSunPosition(zDate &date, float radius)
	{
		float LocalTime = date.tm_hour + (date.tm_min / 60.0);

		double JD = date.toJulian();

		double phi = location.latitude;
		double lambda = location.longitude;

		double n = JD - 2451545.0;

		double LDeg = (double) fmod((280.460 + 0.9856474 * n), 360.0);
		double gDeg = (double) fmod((357.528 + 0.9856003 * n), 360.0);

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

		double thetaG = (double) fmod(thetaGh * 15.0, 360.0);
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

		return coreUtils.sphericalToCartesian(aDeg, hRDeg, radius);
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

		double m = (double) fmod((357.5291 + 0.98560028 * js), 360.0) * DEG_TO_RAD; // radians

		double c = 1.9148 * sin(m) + 0.02 * sin(2 * m) + 0.0003 * sin(3 * m);

		double mDeg = m * RAD_TO_DEG;
		double lambdaDeg = (double) fmod((mDeg + c + 180 + 102.9372), 360.0); //deg
		double lambda = lambdaDeg * DEG_TO_RAD;

		double jt = 2451545.0 + js + ((0.0053 * sin(m)) - (0.0069 * sin(2 * lambda)));

		double delta = asin(sin(lambda) * sin(23.44 * DEG_TO_RAD));

		double cosOmega = (sin(-0.83 * DEG_TO_RAD) - (sin(location.latitude * DEG_TO_RAD) * sin(delta))) / (cos(location.longitude * DEG_TO_RAD) * cos(delta));
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

	ZSPACE_INLINE zDomainDate zTsSolarAnalysis::getDates()
	{
		return dDate;
	}

	ZSPACE_INLINE zLocation zTsSolarAnalysis::getLocation()
	{
		return location;
	}

	ZSPACE_INLINE zNorm_SunVec * zTsSolarAnalysis::getNorm_SunVecs()
	{
		return norm_sunVecs;
	}

	//---- COMPUTE METHODS

	ZSPACE_INLINE void zTsSolarAnalysis::computeSunVectors( float radius)
	{
		time_t  unixTime_s = dDate.min.toUnix();
		time_t  unixTime_e = dDate.max.toUnix();

		// get minute domain per day
		zDate minHour = dDate.min;
		zDate maxHour(dDate.min.tm_year, dDate.min.tm_mon, dDate.min.tm_mday, dDate.max.tm_hour, dDate.max.tm_min);
		
		time_t  unixTime_sh = minHour.toUnix();
		time_t  unixTime_eh = maxHour.toUnix();


		//get total number of vectors
		int numDays = (unixTime_e - unixTime_s);
		if (numDays != 0)
		{
			numDays /= 86400;
			numDays += 1;
		}

		int numMin = (unixTime_eh - unixTime_sh);
		if (numMin != 0)
		{
			numMin /= 60;
			numMin += 1;
		}

		numSunVec = (numDays) *  (numMin);

		sunVecs = new zVector[numSunVec];

		int count = 0;

		for (time_t day = unixTime_s; day <= unixTime_e; day += 86400)
		{

			for (time_t minute = unixTime_sh; minute <= unixTime_eh; minute += 60)
			{
				zDate currentDate;
				currentDate.fromUnix(day + minute - unixTime_s);;
					
				//zDomainDate sRS = getSunRise_SunSet(currentDate);

				//if (currentDate.tm_hour > sRS.min.tm_hour && currentDate.tm_hour < sRS.max.tm_hour)
				//{
					sunVecs[count] = getSunPosition(currentDate, radius);
					count++;
				//}

				
			}

		}
	}
	

}
