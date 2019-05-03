#pragma once

#include <headers/framework/core/zMatrix.h>
#include <headers/framework/core/zVector.h>


namespace zSpace
{

	/** \addtogroup Framework
	*	\brief The datastructures of the library.
	*  @{
	*/

	/** \addtogroup zCore
	*	\brief  The core classes, enumerators ,defintions of the library.
	*  @{
	*/

	/*! \class zTransform
	*	\brief A 4X4 matrix  of doubles.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/


	class zTransform : public zMatrix<double>
	{

	protected:

		/*!	\brief a 4x4 matrix to store pivot position  */
		zMatrixd P;

		/*!	\brief a 4x4 matrix to store scale information  */
		zMatrixd S;

		/*!	\brief a 4x4 matrix to store translation information  */
		zMatrixd T;

		/*!	\brief a 4x4 matrix to store rotation information. R = Rx * Ry *Rz  */
		zMatrixd R;

		/*!	\brief a 4x4 matrix to store rotation information in x axis  */
		zMatrixd Rx;

		/*!	\brief a 4x4 matrix to store rotation information in y axis  */
		zMatrixd Ry;

		/*!	\brief a 4x4 matrix to store rotation information in z axis  */
		zMatrixd Rz;

		/*!	\brief stores rotation  values in x, y, and z in radians  */
		double3 rotation;

		/*!	\brief stores translation values in x, y, and z  */
		double3 translation;

		/*!	\brief stores pivot values in x, y, and z  */
		double3 pivot;

		/*!	\brief stores scale values in x, y, and z  */
		double3 scale;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------


		/*! \brief Default Constructor 
		*	
		*	\since version 0.0.2
		*/
		zTransform()
		{
			rows = 4;
			cols = 4;

			mat.clear();

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					mat.push_back(0);
					if (i == j) mat[mat.size() - 1] = 1;

				}
			}

			P = zMatrixd(4, 4);
			P.setIdentity();

			S = zMatrixd(4, 4);
			S.setIdentity();

			R = zMatrixd(4, 4);
			R.setIdentity();

			T = zMatrixd(4, 4);
			T.setIdentity();

			pivot[0] = pivot[1] = pivot[2] = 0;
			computeP();

			rotation[0] = rotation[1] = rotation[2] = 0;
			computeRx();
			computeRy();
			computeRz();
			computeR();

			scale[0] = scale[1] = scale[2] = 1;
			computeS();

			translation[0] = translation[1] = translation[2] = 0;
			computeT();

			computeTransform();

		}

		zTransform(zMatrix<double> inMatrix)
		{
			this->operator()(0, 0) = inMatrix(0, 0);
			this->operator()(0, 1) = inMatrix(0, 1);
			this->operator()(0, 2) = inMatrix(0, 2);
			this->operator()(0, 3) = inMatrix(0, 3);

			this->operator()(1, 0) = inMatrix(1, 0);
			this->operator()(1, 1) = inMatrix(1, 1);
			this->operator()(1, 2) = inMatrix(1, 2);
			this->operator()(1, 3) = inMatrix(1, 3);

			this->operator()(2, 0) = inMatrix(2, 0);
			this->operator()(2, 1) = inMatrix(2, 1);
			this->operator()(2, 2) = inMatrix(2, 2);
			this->operator()(2, 3) = inMatrix(2, 3);

			this->operator()(3, 0) = inMatrix(3, 0);
			this->operator()(3, 1) = inMatrix(3, 1);
			this->operator()(3, 2) = inMatrix(3, 2);
			this->operator()(3, 3) = inMatrix(3, 3);
			
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		~zTransform()
		{

		}

		//--------------------------
		//---- SET METHODS
		//--------------------------


		/*! \brief This method set the tranformation to the input tranform.
		*
		*	\return 			zTransform		- input transformation matrix.
		*	\since version 0.0.2
		*/
		void setTransform(zMatrixd &inTransform)
		{	
			// set translation
			T(3,0)	= inTransform(3, 0); T(3, 1) = inTransform(3, 1); T(3, 2) = inTransform(3, 2);
						
			// set rotation 
			R(0, 0) = inTransform(0, 0); R(0, 1) = inTransform(0, 1); R(0, 2) = inTransform(0, 2);
			R(1, 0) = inTransform(1, 0); R(1, 1) = inTransform(1, 1); R(1, 2) = inTransform(1, 2);
			R(2, 0) = inTransform(2, 0); R(2, 1) = inTransform(2, 1); R(2, 2) = inTransform(2, 2);

			// compute components
			decomposeR();
			decomposeT();

			setValues(inTransform);			
		}

		/*! \brief This method sets the rotation components of the tranform.
		*
		*	\param		[in]	_rotation		- input rotations in x,y and z in degrees.
		*	\since version 0.0.2
		*/
		void setRotation(double3 &_rotation)
		{
			rotation[0] = DEG_TO_RAD * _rotation[0];
			rotation[1] = DEG_TO_RAD * _rotation[1];
			rotation[2] = DEG_TO_RAD * _rotation[2];
			
			computeRx();
			computeRy();
			computeRz();


			printf("\n\n Rx");
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rx(0, 0), Rx(0, 1), Rx(0, 2), Rx(0, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rx(1, 0), Rx(1, 1), Rx(1, 2), Rx(1, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rx(2, 0), Rx(2, 1), Rx(2, 2), Rx(2, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rx(3, 0), Rx(3, 1), Rx(3, 2), Rx(3, 3));

			printf("\n\n Ry");
			printf("\n %1.2f %1.2f %1.2f %1.2f", Ry(0, 0), Ry(0, 1), Ry(0, 2), Ry(0, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Ry(1, 0), Ry(1, 1), Ry(1, 2), Ry(1, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Ry(2, 0), Ry(2, 1), Ry(2, 2), Ry(2, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Ry(3, 0), Ry(3, 1), Ry(3, 2), Ry(3, 3));

			printf("\n\n Rz");
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rz(0, 0), Rz(0, 1), Rz(0, 2), Rz(0, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rz(1, 0), Rz(1, 1), Rz(1, 2), Rz(1, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rz(2, 0), Rz(2, 1), Rz(2, 2), Rz(2, 3));
			printf("\n %1.2f %1.2f %1.2f %1.2f", Rz(3, 0), Rz(3, 1), Rz(3, 2), Rz(3, 3));

			computeR();
			computeTransform();
		}

		/*! \brief This method sets the translation components of the tranform.
		*
		*	\param		[in]	_translation		- input translation in x,y and z.
		*	\since version 0.0.2
		*/
		void setTranslation(double3 &_translation)
		{
			translation[0] = _translation[0];
			translation[1] = _translation[1];
			translation[2] = _translation[2];			

			computeT();
			computeTransform();
		}

		/*! \brief This method sets the scale components of the tranform.
		*
		*	\param		[in]	_scale		- input scale in x,y and z.
		*	\since version 0.0.2
		*/
		void setScale(double3 &_scale)
		{
			scale[0] = _scale[0];
			if (scale[0] == 0)scale[0] = scaleZero;

			scale[1] = _scale[1];
			if (scale[1] == 0)scale[1] = scaleZero;

			scale[2] = _scale[2];
			if (scale[2] == 0)scale[2] = scaleZero;

			computeS();
			computeTransform();
		}


		/*! \brief This method sets the pivot components of the tranform.
		*
		*	\param		[in]	_scale		- input scale in x,y and z.
		*	\since version 0.0.2
		*/
		void setPivot(double3 &_pivot)
		{
			pivot[0] = _pivot[0];
			pivot[1] = _pivot[1];
			pivot[2] = _pivot[2];

			computeP();
			computeTransform();
		}


		//--------------------------
		//---- GET METHODS
		//--------------------------
		
		/*! \brief This method gets the X-Axis of the tranform.
		*
		*	\return				zVector		- output x axis vector.
		*	\since version 0.0.2
		*/
		zVector getX()
		{
			return zVector(this->operator()(0, 0), this->operator()(0, 1), this->operator()(0, 2));
		}

		/*! \brief This method gets the Y-Axis of the tranform.
		*
		*	\return				zVector		- output y axis vector.
		*	\since version 0.0.2
		*/
		zVector getY()
		{
			return zVector(this->operator()(1, 0), this->operator()(1, 1), this->operator()(1, 2));
		}

		/*! \brief This method gets the Z-Axis of the tranform.
		*
		*	\return				zVector		- output z axis vector.
		*	\since version 0.0.2
		*/
		zVector getZ()
		{
			return zVector(this->operator()(2, 0), this->operator()(2, 1), this->operator()(2, 2));
		}

		/*! \brief This method gets the Origin of the tranform.
		*
		*	\return				zVector		- output z axis vector.
		*	\since version 0.0.2
		*/
		zVector getO()
		{
			return zVector(this->operator()(3, 0), this->operator()(3, 1), this->operator()(3, 2));
		}

		/*! \brief This method gets the rotation components of the tranform.
		*
		*	\param		[out]	_rotation		- output rotations in x,y and z in degrees.
		*	\since version 0.0.2
		*/
		void getRotation(double3 &_rotation)
		{
			_rotation[0] = RAD_TO_DEG * rotation[0];
			_rotation[1] = RAD_TO_DEG * rotation[1];
			_rotation[2] = RAD_TO_DEG * rotation[2];
					
		}

		/*! \brief This method gets the translation components of the tranform.
		*
		*	\param		[out]	_translation		- output translations in x,y and z in degrees.
		*	\since version 0.0.2
		*/
		void getTranslation(double3 &_translation)
		{
			_translation[0] = RAD_TO_DEG * translation[0];
			_translation[1] = RAD_TO_DEG * translation[1];
			_translation[2] = RAD_TO_DEG * translation[2];

		}

		/*! \brief This method gets the scale components of the tranform.
		*
		*	\param		[out]	_scale		- output scale in x,y and z.
		*	\since version 0.0.2
		*/
		void getScale(double3 &_scale)
		{
			_scale[0] = scale[0];
			if (scale[0] == scaleZero)_scale[0] = 0;

			_scale[1] = scale[1];
			if (scale[1] == scaleZero)_scale[1] = 0;

			_scale[2] = scale[2];
			if (scale[2] == scaleZero)_scale[2] = 0;			
		}

		/*! \brief This method gets the pivot components of the tranform.
		*
		*	\param		[out]	_scale		- output scale in x,y and z.
		*	\since version 0.0.2
		*/
		void getPivot(double3 &_pivot)
		{
			_pivot[0] = pivot[0];
			_pivot[1] = pivot[1];
			_pivot[2] = pivot[2];			
		}

		/*! \brief This method get the tranformation to the world space.
		*		
		*	\return 			zTransform		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getWorld()
		{
			
			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X(this->operator()(0, 0), this->operator()(1, 0), this->operator()(2, 0));
			zVector Y(this->operator()(0, 1), this->operator()(1, 1), this->operator()(2, 1));
			zVector Z(this->operator()(0, 2), this->operator()(1, 2), this->operator()(2, 2));
			zVector Cen(this->operator()(0, 3), this->operator()(1, 3), this->operator()(2, 3));


			outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
			outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
			outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

			outMatrix(0, 3) = Cen.x; outMatrix(1, 3) = Cen.y; outMatrix(2, 3) = Cen.z;

			return outMatrix;
		}

		/*! \brief This method gets the tranformation to the local space.
		*
		*	\return 			zTransform		- world transformation matrix.
		*	\since version 0.0.2
		*/		
		zTransform getLocal()
		{			
			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X(this->operator()(0, 0), this->operator()(1, 0), this->operator()(2, 0));
			zVector Y(this->operator()(0, 1), this->operator()(1, 1), this->operator()(2, 1));
			zVector Z(this->operator()(0, 2), this->operator()(1, 2), this->operator()(2, 2));
			zVector Cen(this->operator()(0, 3), this->operator()(1, 3), this->operator()(2, 3));

			zVector orig(0, 0, 0);
			zVector d = Cen - orig;

			outMatrix(0, 0) = X.x; outMatrix(0, 1) = X.y; outMatrix(0, 2) = X.z;
			outMatrix(1, 0) = Y.x; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Y.z;
			outMatrix(2, 0) = Z.x; outMatrix(2, 1) = Z.y; outMatrix(2, 2) = Z.z;

			outMatrix(0, 3) = -(X*d); outMatrix(1, 3) = -(Y*d); outMatrix(2, 3) = -(Z*d);

			return outMatrix;
		}

		/*! \brief This method gets the tranformation from current tranform to input transform.
		*
		*	\param		[in]	to			- input transform.
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getTo(zTransform &to)
		{
			zTransform world = to.getWorld();
			zTransform local = this->getLocal();

			zTransform out = world * local;				
			

			return out;
		}


		/*! \brief This method gets the tranformation to change the basis to another.
		*
		*	\param		[in]	to			- input transform.
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getChangeBasis(zTransform &to)
		{
			zTransform local = to.getLocal();
			zTransform world = this->getWorld();

			zTransform out = local * world;

			return out;
		}

		/*! \brief This method gets the input target as per the current transform as the basis.
		*
		*	\param		[in]	target		- input target transform.
		*	\return 			zTransform	- new target transform.
		*	\since version 0.0.2
		*/
		zTransform getTarget(zTransform &target)
		{
			
			zTransform C_inverse;

			bool chkInvr = this->inverseMatrix(C_inverse);

			if (!chkInvr) throw std::invalid_argument("input transform is singular and doesnt have an inverse.");

			zTransform targ_newbasis;
			targ_newbasis.setIdentity();

			targ_newbasis = C_inverse * target;

			return targ_newbasis;
		}

	protected:

		/*! \brief This method compute the transform from the indivual matricies.
		*
		*	\since version 0.0.2
		*/
		void computeTransform()
		{
			zMatrixd PS = P * S;
			zMatrixd PSInverse;
			
			bool chk  = PS.inverseMatrix(PSInverse);			

			zMatrixd temp = PS* PSInverse * R * T;
			
			this->setValues(temp);
		}

		/*! \brief This method compute the rotation matrix.
		*
		*	\since version 0.0.2
		*/
		void computeR()
		{
			R = Rx * Ry *Rz;
		}

		/*! \brief This method compute the rotation matrix in X.
		*
		*	\since version 0.0.2
		*/
		void computeRx()
		{
			Rx(1, 1) = cos(rotation[0]);
			Rx(1, 2) = sin(rotation[0]);

			Rx(2, 1) = -sin(rotation[0]);
			Rx(2, 2) = cos(rotation[0]);
			
		}

		/*! \brief This method compute the rotation matrix in Y.
		*
		*	\since version 0.0.2
		*/
		void computeRy()
		{
			Ry(0, 0) = cos(rotation[1]);
			Ry(0, 2) = -sin(rotation[1]);

			Ry(2, 0) = sin(rotation[1]);
			Ry(2, 2) = cos(rotation[1]);
			
		}

		/*! \brief This method compute the rotation matrix in Z.
		*
		*	\since version 0.0.2
		*/
		void computeRz()
		{
			Rz(0, 0) = cos(rotation[2]);
			Rz(0, 1) = sin(rotation[2]);

			Rz(1, 0) = -sin(rotation[2]);
			Rz(1, 1) = cos(rotation[2]);			
		}

		/*! \brief This method compute the translation matrix.
		*
		*	\since version 0.0.2
		*/
		void computeT()
		{
			T(3, 0) = translation[0];
			T(3, 1) = translation[1];
			T(3, 2) = translation[2];			
		}

		/*! \brief This method compute the pivot matrix.
		*
		*	\since version 0.0.2
		*/
		void computeP()
		{
			P(3, 0) = pivot[0];
			P(3, 1) = pivot[1];
			P(3, 2) = pivot[2];
		}

		/*! \brief This method compute the scale matrix.
		*
		*	\since version 0.0.2
		*/
		void computeS()
		{
			S(0, 0) = scale[0];
			S(1, 1) = scale[1];
			S(2, 2) = scale[2];
		}

		/*! \brief This method sets the values of the transform with the input matrix.
		*
		*	\since version 0.0.2
		*/
		void setValues(zMatrixd &inTransform)
		{
			this->operator()(0, 0) = inTransform(0, 0);
			this->operator()(0, 1) = inTransform(0, 1);
			this->operator()(0, 2) = inTransform(0, 2);
			this->operator()(0, 3) = inTransform(0, 3);

			this->operator()(1, 0) = inTransform(1, 0);
			this->operator()(1, 1) = inTransform(1, 1);
			this->operator()(1, 2) = inTransform(1, 2);
			this->operator()(1, 3) = inTransform(1, 3);

			this->operator()(2, 0) = inTransform(2, 0);
			this->operator()(2, 1) = inTransform(2, 1);
			this->operator()(2, 2) = inTransform(2, 2);
			this->operator()(2, 3) = inTransform(2, 3);

			this->operator()(3, 0) = inTransform(3, 0);
			this->operator()(3, 1) = inTransform(3, 1);
			this->operator()(3, 2) = inTransform(3, 2);
			this->operator()(3, 3) = inTransform(3, 3);
		}


		/*! \brief This method decomposes the rotation matrix into Euler angles.
		*
		*	\details based on  http://www.gregslabaugh.net/publications/euler.pdf
		*	\since version 0.0.2
		*/
		void decomposeR()
		{
			double3 rot0;
			double3 rot1;

			if (R(2, 0) != 1 && R(2, 0) != -1)
			{
				rot0[1] = asin(R(2, 0));
				rot1[1] = PI - rot0[1];

				rot0[0] = atan2(R(2, 1) / cos(rot0[1]), R(2, 2) / cos(rot0[1]));
				rot1[1] = atan2(R(2, 1) / cos(rot1[1]), R(2, 2) / cos(rot1[1]));

				rot0[2] = atan2(R(1,0) / cos(rot0[1]), R(1, 1) / cos(rot0[1]));
				rot1[2] = atan2(R(1, 0) / cos(rot1[1]), R(1, 1) / cos(rot1[1]));

			}
			else
			{
				rot0[2] = rot1[2] = 0;

				if (R(2, 0) == -1)
				{
					rot0[1] = rot1[1] = HALF_PI;
					rot0[0] = rot1[0] = atan2(R(0, 1) , R(0, 2));
				}
				else
				{
					rot0[1] = rot1[1] = HALF_PI * -1;
					rot0[0] = rot1[0] = atan2(-R(0, 1), -R(0, 2));
				}
			}

			rotation[0] = rot0[0]; 
			rotation[1] = rot0[1];
			rotation[2] = rot0[2];

			printf("\n %1.2f %1.2f %1.2f ", RAD_TO_DEG* rotation[0], RAD_TO_DEG*rotation[1], RAD_TO_DEG* rotation[2]);

			computeRx(); computeRy(); computeRz();
		}

		/*! \brief This method decomposes the translation matrix into ddistances in x, y and z axis.
		*
		*	\since version 0.0.2
		*/
		void decomposeT()
		{
			translation[0] = T(3, 0);
			translation[1] = T(3, 1);
			translation[2] = T(3, 2);
		}

	};

}
