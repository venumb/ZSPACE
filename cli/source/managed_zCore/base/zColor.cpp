#include<headers/managed_zCore/base/zColor.h>

namespace zSpaceCLI
{
	//---- CONSTRUCTOR

	zColor::zColor() : zManagedObj(new zSpace::zColor())
	{
		throw gcnew System::NotImplementedException();
	}

	zColor::zColor(double _r, double _g, double _b, double _a) : zManagedObj(new zSpace::zColor(_r,_g,_b,_a))
	{
		throw gcnew System::NotImplementedException();
	}

	zColor::zColor(double _h, double _s, double _v) : zManagedObj(new zSpace::zColor(_h, _s, _v))
	{
		throw gcnew System::NotImplementedException();
	}	

	//---- METHODS

	void zColor::toHSV()
	{
		m_zInstance->toHSV();
	}

	void zColor::toRGB()
	{
		m_zInstance->toRGB();
	}

	bool zColor::operator==(zColor c1)
	{
		return m_zInstance->operator==(*c1.m_zInstance);
	}

}