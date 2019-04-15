#pragma once

#include <headers/api/object/zObject.h>
#include <headers/framework/dynamics/zParticle.h>


namespace zSpace
{

	/** \addtogroup zObjects
	*	\brief The object and function set classes of the library.
	*  @{
	*/

	/*! \class zObjParticle
	*	\brief The particle object class.
	*	\since version 0.0.2
	*/

	/** @}*/

	class zObjParticle :public zObject
	{
	private:
		/*! \brief boolean for displaying the particle forces */
		bool showForces;

		/*! \brief force display scale */
		double forceScale;

	public:
		//--------------------------
		//---- PUBLIC ATTRIBUTES
		//--------------------------

		/*! \brief particle */
		zParticle particle;

		//--------------------------
	//---- CONSTRUCTOR
	//--------------------------
	/*! \brief Default constructor.
	*
	*	\since version 0.0.2
	*/
		zObjParticle()
		{
			displayUtils = nullptr;		

			showForces = false;

			forceScale = 1.0;
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.2
		*/
		~zObjParticle() {}

		//--------------------------
		//---- SET METHODS
		//--------------------------

		/*! \brief This method sets show vertices, edges and face booleans.
		*
		*	\param		[in]	_showForces					- input show forces booelan.
		*	\param		[in]	_forceScale					- input scale of forces.
		*	\since version 0.0.2
		*/
		void setShowElements(bool _showForces, double _forceScale)
		{
			showForces = _showForces;
			forceScale = _forceScale;
		}

		//--------------------------
		//---- OVERRIDE METHODS
		//--------------------------

		void draw() override
		{
			if (showObject)
			{
				if (showForces) drawForces();
			}
		}

		//--------------------------
		//---- DISPLAY METHODS
		//--------------------------

		/*! \brief This method displays the zMesh.
		*
		*	\since version 0.0.2
		*/
		void drawForces()
		{
			zVector p = *particle.s.p;
			zVector p1 = p + particle.f;

			displayUtils->drawPoint(p1);

			displayUtils->drawLine(p, p1, zColor(0,1,0,1), 1.0);

		}
	};
}