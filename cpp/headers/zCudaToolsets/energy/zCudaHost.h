#ifndef zCudaHost
#define zCudaHost

#include "vector_types.h"
#include "vector_functions.h"


//////////////////////////////////////////////////////////////////////////------------------------------------------- DATA-TRANSFER

struct zCUDAVector
{
	float4 v;

	__device__ __host__ zCUDAVector()
	{
		v = make_float4(0, 0, 0, 0);
	}

	__device__ __host__ zCUDAVector(float x, float y, float z, float w)
	{
		v = make_float4(x, y, z, w);
	}

	__device__ __host__ ~zCUDAVector() {}

	//////////////////////////////////// Operators

	__device__ __host__ double operator *(zCUDAVector v1)
	{
		return (v.x * v1.v.x + v.y * v1.v.y + v.z * v1.v.z);
	}

	__device__ __host__ zCUDAVector operator ^(const zCUDAVector &v1)
	{
		return zCUDAVector(v.y * v1.v.z - v.z * v1.v.y, v.z*v1.v.x - v.x * v1.v.z, v.x*v1.v.y - v.y * v1.v.x, 1);
	}

	//////////////////////////////////// Methods

	__device__ __host__ double length()
	{
		return (v.x*v.x + v.y * v.y + v.z * v.z);
	}

	__device__ __host__ void normalize()
	{
		double length = this->length();

		v.x /= length;
		v.y /= length;
		v.z /= length;
	}

	__device__ __host__ double angle(zCUDAVector &v1)
	{
		zCUDAVector a(v.x, v.y, v.z, v.w);
		zCUDAVector b(v1.v.x, v1.v.y, v1.v.x, v1.v.w);

		a.normalize();
		b.normalize();

		if (a*b == 1) return 0;
		else if (a*b == -1) return 180;
		else
		{

			double dotProduct = a * b;
			double angle = acos(dotProduct);

			return (angle * 180.0) / 3.14159265358979323846;
		}
	}

};

struct zCUDADate
{
	int year = 0;
	int month, day, hour;
	int minute = 0;
	int monthDays[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

	__device__ __host__ zCUDADate() {}

	__device__ __host__ zCUDADate(int _month, int _day, int _hour)
	{
		month = _month;
		day = _day;
		hour = _hour;
	}

	__device__ __host__ zCUDADate(int _year, int _month, int _day, int _hour, int _minute)
	{
		year = _year;
		month = _month;
		day = _day;
		hour = _hour;
		minute = _minute;
	}

	__device__ __host__ bool operator==(const zCUDADate &other) const
	{
		return (month == other.month
			&& day == other.day
			&& hour == other.hour);
	}

	__device__ __host__ int countLeapYears(zCUDADate d)
	{
		int years = d.year;

		if (d.month <= 2)
			years--;

		return years / 4 - years / 100 + years / 400;
	}

	__device__ __host__ int getDifference(zCUDADate dt1, zCUDADate dt2)
	{
		long int n1 = dt1.year * 365 + dt1.day;

		for (int i = 0; i < dt1.month - 1; i++)
			n1 += monthDays[i];

		n1 += countLeapYears(dt1);

		long int n2 = dt2.year * 365 + dt2.day;
		for (int i = 0; i < dt2.month - 1; i++)
			n2 += monthDays[i];
		n2 += countLeapYears(dt2);

		return (n2 - n1);
	}
};

//////////////////////////////////// H-D

extern void initialise_DeviceMemory(int size);

extern void copy_HostToDevice(char* host_Path, zCUDAVector *host_faceNormals, zCUDADate *host_Date, int size, double *host_angles_Out);

//////////////////////////////////// D-H

extern double  *copy_DeviceToHost();

//////////////////////////////////////////////////////////////////////////-------------------------------------------  CALLS

extern void callKernel();
 
//////////////////////////////////////////////////////////////////////////------------------------------------------- EXIT

extern void CUDA_Free();

#endif