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

	/*! \class zTransformationMatrix
	*	\brief A transformation matrix class.
	*
	*	\since version 0.0.2
	*/

	/** @}*/

	/** @}*/


	class zTransformationMatrix 
	{

	protected:

		/*!	\brief a 4x4 matrix to store transfromation  */
		zTransform Transform;

		/*!	\brief a 4x4 matrix to store pivot position  */
		zTransform P;

		/*!	\brief a 4x4 matrix to store scale information  */
		zTransform S;

		/*!	\brief a 4x4 matrix to store translation information  */
		zTransform T;

		/*!	\brief a 4x4 matrix to store rotation information. R = Rx * Ry *Rz  */
		zTransform R;

		/*!	\brief a 4x4 matrix to store rotation information in x axis  */
		zTransform Rx;

		/*!	\brief a 4x4 matrix to store rotation information in y axis  */
		zTransform Ry;

		/*!	\brief a 4x4 matrix to store rotation information in z axis  */
		zTransform Rz;

		/*!	\brief stores rotation  values in x, y, and z in radians  */
		double3 rotation;

		/*!	\brief stores scale values in x, y, and z  */
		double3 scale;

		/*!	\brief stores translation as a vector  */
		double3 translation;

		/*!	\brief stores pivot as a vector  */
		double3 pivot;

		

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------


		/*! \brief Default Constructor 
		*	
		*	\since version 0.0.2
		*/
		zTransformationMatrix()
		{
		
			Transform.setIdentity();
			
			P.setIdentity();
			
			S.setIdentity();

			R.setIdentity();
			
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

		}
		

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.2
		*/
		~zTransformationMatrix()
		{
		}

		//--------------------------
		//---- SET METHODS
		//--------------------------


		/*! \brief This method sets the transform to the input tranform.
		*
		*	\param		[in]	decompose		- decoposes the translation and rotation matrix if true.
		*	\return 			zTransform		- input transform.
		*	\since version 0.0.2
		*/
		void setTransform(zTransform &inTransform, bool decompose = true)
		{	
			if (decompose)
			{
				// set translation
				T(3, 0) = inTransform(3, 0); T(3, 1) = inTransform(3, 1); T(3, 2) = inTransform(3, 2);

				// set rotation 
				R(0, 0) = inTransform(0, 0); R(0, 1) = inTransform(0, 1); R(0, 2) = inTransform(0, 2);
				R(1, 0) = inTransform(1, 0); R(1, 1) = inTransform(1, 1); R(1, 2) = inTransform(1, 2);
				R(2, 0) = inTransform(2, 0); R(2, 1) = inTransform(2, 1); R(2, 2) = inTransform(2, 2);

				// compute components
				decomposeR();
				decomposeT();
			}
			
			Transform = inTransform;
					
		}

		/*! \brief This method sets the rotation components of the tranform.
		*
		*	\param		[in]	_rotation			- input rotations in x,y and z in degrees.
		*	\param		[in]	addValues			- true if the input values are added to the existing values.
		*	\since version 0.0.2
		*/
		void setRotation(double3 &_rotation, bool addValues = false)
		{
			if (addValues)
			{
				rotation[0] += DEG_TO_RAD * _rotation[0];
				rotation[1] += DEG_TO_RAD * _rotation[1];
				rotation[2] += DEG_TO_RAD * _rotation[2];
			}
			else
			{
				rotation[0] = DEG_TO_RAD * _rotation[0];
				rotation[1] = DEG_TO_RAD * _rotation[1];
				rotation[2] = DEG_TO_RAD * _rotation[2];
			}

			
			
			computeRx();
			computeRy();
			computeRz();			

			computeR();
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

		}

		/*! \brief This method sets the translation components of the tranform.
		*
		*	\param		[in]	_translation		- input translation in x,y and z.
		*	\param		[in]	addValues			- true if the input values are added to the existing values.
		*	\since version 0.0.2
		*/
		void setTranslation(double3 &_translation, bool addValues = false)
		{
			if (addValues)
			{
				translation[0] += _translation[0];		
				translation[1] += _translation[1];
				translation[2] += _translation[2];

			}
			else
			{
				translation[0] = _translation[0];
				translation[1] = _translation[1];
				translation[2] = _translation[2];
			}					

			computeT();
			computeTransform();
		}

		


		/*! \brief This method sets the pivot components of the tranform.
		*
		*	\param		[in]	_pivot		- input pivot in x,y and z.
		*	\since version 0.0.2
		*/
		void setPivot(double3 &_pivot)
		{
			pivot[0] = _pivot[0];
			pivot[1] = _pivot[1];
			pivot[2] = _pivot[2];		

			computeP();
		
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
			return zVector(Transform(0, 0), Transform(0, 1), Transform(0, 2));
		}

		/*! \brief This method gets the Y-Axis of the tranform.
		*
		*	\return				zVector		- output y axis vector.
		*	\since version 0.0.2
		*/
		zVector getY()
		{
			return zVector(Transform(1, 0), Transform(1, 1), Transform(1, 2));
		}

		/*! \brief This method gets the Z-Axis of the tranform.
		*
		*	\return				zVector		- output z axis vector.
		*	\since version 0.0.2
		*/
		zVector getZ()
		{
			return zVector(Transform(2, 0), Transform(2, 1), Transform(2, 2));
		}

		/*! \brief This method gets the Origin of the tranform.
		*
		*	\return				zVector		- output origin vector.
		*	\since version 0.0.2
		*/
		zVector getO()
		{
			return zVector(Transform(3, 0), Transform(3, 1), Transform(3, 2));
		}

		/*! \brief This method gets the translation components of the tranform.
		*
		*	\return				zVector		- output translation vector.
		*	\since version 0.0.2
		*/
		zVector getTranslation()
		{
			return translation;
		}

		/*! \brief This method gets the pivot components of the tranform.
		*
		*	\return				zVector		- output pivot vector.
		*	\since version 0.0.2
		*/
		zVector getPivot()
		{
			return pivot;		
		}

		/*! \brief This method gets the pointer to the pivot components.
		*
		*	\return			double*		- pointer to pivot component.
		*	\since version 0.0.2
		*/
		double* getRawPivot()
		{
			return &pivot[0];
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

		

		//--------------------------
		//---- AS MATRIX METHODS
		//--------------------------

		/*! \brief This method returns the 4x4 matrix that describes this transformation;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asMatrix()
		{
			return Transform;
		}

		/*! \brief This method returns the pointer to the 4x4 matrix that describes this transformation;
		*
		*	\return 			double*		-  pointer to the 4x4 matrix .
		*	\since version 0.0.2
		*/
		double* asRawMatrix()
		{
			return Transform.data();
		}

		/*! \brief This method returns the inverse of the 4x4 matrix that describes this transformation;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInverseMatrix()
		{			
			return Transform.inverse();
		}

		/*! \brief This method returns the scale matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asScaleMatrix()
		{
			return S;
		}

		/*! \brief This method returns the rotation matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asRotationMatrix()
		{
			return R;
		}
		
		/*! \brief This method returns the rotation matrix;
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asPivotMatrix()
		{
			return P;
		}

		/*! \brief This method returns the pivot translation matrix
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asPivotTranslationMatrix()
		{
			zTransform out;
			out.setIdentity();

			
			out(0, 3) = -1 * pivot[0];
			out(1, 3) = -1 * pivot[1];
			out(2, 3) = -1 * pivot[2];


			return out;
		}

		/*! \brief This method returns the inverse of the pivot translation matrix
		*
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInversePivotTranslationMatrix()
		{	

			return asPivotTranslationMatrix().inverse();
		}

		/*! \brief This method returns the scaling matrix;
		*
		*	\details based on https://studylib.net/doc/5892312/scaling-relative-to-a-fixed-point-using-matrix-using-the
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asScaleTransformMatrix()
		{
			zTransform out;
			out.setIdentity();

			out(0, 0) = scale[0];
			out(1, 1) = scale[1];
			out(2, 2) = scale[2];			

			out(0, 3) = (1 - scale[0]) * pivot[0];
			out(1, 3) = (1 - scale[1]) * pivot[1];
			out(2, 3) = (1 - scale[2]) * pivot[2];


			return out;
		}

		/*! \brief This method returns the scaling matrix;
		*
		*	\details based on https://studylib.net/doc/5892312/scaling-relative-to-a-fixed-point-using-matrix-using-the
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform asInverseScaleTransformMatrix()
		{	

			return asScaleTransformMatrix().inverse();
		}


		//--------------------------
		//---- GET MATRIX METHODS
		//--------------------------

		/*! \brief This method get the tranformation to the world space.
		*		
		*	\return 			zTransform		- world transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getWorldMatrix()
		{
			
			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X = getX();
			zVector Y = getY();
			zVector Z = getZ();
			zVector O = getO();


			outMatrix(0, 0) = X.x; outMatrix(0, 1) = Y.x; outMatrix(0, 2) = Z.x;
			outMatrix(1, 0) = X.y; outMatrix(1, 1) = Y.y; outMatrix(1, 2) = Z.y;
			outMatrix(2, 0) = X.z; outMatrix(2, 1) = Y.z; outMatrix(2, 2) = Z.z;

			outMatrix(0, 3) = O.x; outMatrix(1, 3) = O.y; outMatrix(2, 3) = O.z;

			return outMatrix;
		}

		/*! \brief This method gets the tranformation to the local space.
		*
		*	\return 			zTransform		- local transformation matrix.
		*	\since version 0.0.2
		*/		
		zTransform getLocalMatrix()
		{			
			zTransform outMatrix;
			outMatrix.setIdentity();

			zVector X = getX();
			zVector Y = getY();
			zVector Z = getZ();
			zVector O = getO();

			zVector orig(0, 0, 0);
			zVector d = O - orig;		

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
		zTransform getToMatrix(zTransformationMatrix &to)
		{
			zTransform world = to.getWorldMatrix();
			zTransform local = this->getLocalMatrix();			
	
			
			
			return world * local  ;
		}


		/*! \brief This method gets the tranformation to change the basis to another.
		*
		*	\param		[in]	to			- input transformation matrix.
		*	\return 			zTransform	- transformation matrix.
		*	\since version 0.0.2
		*/
		zTransform getBasisChangeMatrix(zTransformationMatrix &to)
		{
			zTransform local = to.getLocalMatrix();
			zTransform world = this->getWorldMatrix();

			return local * world;
		}

		/*! \brief This method gets the input target as per the current transform as the basis.
		*
		*	\param		[in]	target		- input target transform.
		*	\return 			zTransform	- new target transform.
		*	\since version 0.0.2
		*/
		zTransform getTargetMatrix(zTransform &target)
		{
			
			zTransform C_inverse = Transform.inverse();	

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
			Transform.setIdentity();

			zTransform PS = P * S;
			zTransform PSInverse = PS.inverse();			

			Transform = PS* PSInverse * R * T;
			
			
		}

		/*! \brief This method compute the rotation matrix.
		*
		*	\since version 0.0.2
		*/
		void computeR()
		{
			R.setIdentity();
			R = Rx * Ry *Rz;
		}

		/*! \brief This method compute the rotation matrix in X.
		*
		*	\since version 0.0.2
		*/
		void computeRx()
		{
			Rx.setIdentity();

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
			Ry.setIdentity();

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
			Rz.setIdentity();

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
			T.setIdentity();

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
			P.setIdentity();

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
			S.setIdentity();

			S(0, 0) = scale[0];
			S(1, 1) = scale[1];
			S(2, 2) = scale[2];
		}


		/*! \brief This method decomposes the rotation matrix into Euler angles.
		*
		*	\details based on  http://www.gregslabaugh.net/publications/euler.pdf. The zTransform matrix is the the transpose of the one in the paper. 
		*	\since version 0.0.2
		*/
		void decomposeR()
		{
			double3 rot0;
			double3 rot1;

			if (R(0, 2) != 1 && R(0, 2) != -1)
			{
				rot0[1] = asin(R(0, 2)) * -1;
				rot1[1] = PI - rot0[1];

				rot0[0] = atan2(R(1, 2) / cos(rot0[1]), R(2, 2) / cos(rot0[1]));
				rot1[1] = atan2(R(1, 2) / cos(rot1[1]), R(2, 2) / cos(rot1[1]));

				rot0[2] = atan2(R(0,1) / cos(rot0[1]), R(0, 0) / cos(rot0[1]));
				rot1[2] = atan2(R(0, 1) / cos(rot1[1]), R(0, 0) / cos(rot1[1]));

			}
			else
			{
				rot0[2] = rot1[2] = 0;

				if (R(0, 2) == -1)
				{
					rot0[1] = rot1[1] = HALF_PI;
					rot0[0] = rot1[0] = atan2(R(1, 0) , R(2, 0));
				}
				else
				{
					rot0[1] = rot1[1] = HALF_PI * -1;
					rot0[0] = rot1[0] = atan2(R(1, 0)* -1, R(2, 0)* -1);
				}
			}

			rotation[0] = rot0[0]; 
			rotation[1] = rot0[1];
			rotation[2] = rot0[2];

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
