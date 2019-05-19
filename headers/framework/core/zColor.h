#pragma once


#include <stdexcept>

#include<headers/framework/core/zEnumerators.h>

using namespace std;

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
	
	/*! \class zColor
	*	\brief	A color math class. 
	*
	*	Currently supports two types of color data structures - RGBA and HSV.
	*	\since	version 0.0.1
	*/

	/** @}*/ 

	/** @}*/

	class zColor
	{
	public:

		/*!	\brief red component				*/
		double r; 

		/*!	\brief green component			*/
		double g; 

		/*!	\brief blue component			*/
		double b; 

		/*!	\brief alpha component			*/
		double a;  

		/*!	\brief hue component				*/
		double h; 

		/*!	\brief saturation component		*/
		double s;

		/*!	\brief value component			*/
		double v;  
		
		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief	 Default constructor sets by default r,g,b,h,s,v to 0 and a to 1.
		*	\since	 version 0.0.1
		*/
		zColor()
		{
			r = g = b = 0;
			a = 1;
			h = s = v = 0;
		}


		/*! \brief	Overloaded constructor sets RGB_A components of the color and uses it to compute HSV components.
		*
		*	\param		[in]	_r		- red component of zColor.
		*	\param		[in]	_g		- green component of zColor.
		*	\param		[in]	_b		- blue component of zColor.
		*	\param		[in]	_a		- alpha component of zColor.
		*	\since version 0.0.1
		*/

		zColor(double _r, double _g, double _b, double _a)
		{
			r = _r;
			g = _g;
			b = _b;
			a = _a;

			// compute HSV
			toHSV();
		}

		// overloaded constructor HSV
		/*! \brief Overloaded constructor using HSV components of the color and uses it to compute RGB_A components.
		*
		*	\param		[in]	_h		- hue component of zColor.
		*	\param		[in]	_s		- saturation component of zColor.
		*	\param		[in]	_v		- value component of zColor.
		*	\since version 0.0.1
		*/
		zColor(double _h, double _s, double _v)
		{
			h = _h;
			s = _s;
			v = _v;


			// compute RGB
			toRGB();
		}

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*	\since version 0.0.1
		*/

		~zColor(){}

		//--------------------------
		//---- METHODS
		//--------------------------

		
		/*! \brief This methods calculates the HSV components based on the RGB_A components of color.
		*	\since version 0.0.1
		*/
		
		void toHSV()
		{

			double      min, max, delta;

			min = r < g ? r : g;
			min = min  < b ? min : b;

			max = r > g ? r : g;
			max = max  > b ? max : b;

			v = max;                                // v
			delta = max - min;
			if (delta < 0.00001)
			{
				s = 0;
				h = 0; // undefined, maybe nan?

			}

			if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
				s = (delta / max);                  // s
			}
			else {
				// if max is 0, then r = g = b = 0              
				// s = 0, h is undefined
				s = 0.0;
				h = NAN;                            // its now undefined

			}

			if (r >= max)                           // > is bogus, just keeps compilor happy
				h = (g - b) / delta;        // between yellow & magenta
			else
				if (g >= max)
					h = 2.0 + (b - r) / delta;  // between cyan & yellow
				else
					h = 4.0 + (r - g) / delta;  // between magenta & cyan

			h *= 60.0;                              // degrees

			if (h < 0.0)
				h += 360.0;

		}

		/*! \brief This methods calculates the RGB_A components based on the HSV components of color.
		*	\since version 0.0.1
		*/		
		void toRGB()
		{
			double      hh, p, q, t, ff;
			long        i;


			if (s <= 0.0) {       // < is bogus, just shuts up warnings
				r = v;
				g = v;
				b = v;

				return;
			}

			hh = h;

			if (hh >= 360.0) hh = 0.0;
			if(hh != 0.0) hh /= 60.0;
			
			i = (long)hh;
			ff = hh - i;
			p = v * (1.0 - s);
			q = v * (1.0 - (s * ff));
			t = v * (1.0 - (s * (1.0 - ff)));

			switch (i) {
			case 0:
				r = v;
				g = t;
				b = p;
				break;
			case 1:
				r = q;
				g = v;
				b = p;
				break;
			case 2:
				r = p;
				g = v;
				b = t;
				break;

			case 3:
				r = p;
				g = q;
				b = v;
				break;
			case 4:
				r = t;
				g = p;
				b = v;
				break;
			case 5:
			default:
				r = v;
				g = p;
				b = q;
				break;
			}

		}

	

		//--------------------------
		//---- OVERLOADED OPERATORS
		//--------------------------

		/*! \brief This operator checks for equality of two zColors.
		*
		*	\param		[in]	c1		- input color.
		*	\return				bool	- true if colors are the same.
		*	\since version 0.0.1
		*/
				
		bool operator==(zColor &c1)
		{
			return (r == c1.r && g == c1.g && b == c1.b);
		}

	};
}
