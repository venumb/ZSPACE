# ZSPACE
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/venumb/ZSPACE/LICENSE.MIT)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://github.com/venumb/ZSPACE/docs/doxyoutput/)
[![GitHub Releases](https://img.shields.io/github/release/venumb/zspace.svg)](https://github.com/venumb/ZSPACE/releases)
[![GitHub Issues](https://img.shields.io/github/issues/venumb/zspace.svg)](http://github.com/venumb/ZSPACE/issues)

**ZSPACE** is a C++  library collection of geometry data-structures, algorithms and city data visualization framework. It is implemented as a header-only C++ library, whose dependencies, are header-only or static libraries. Hence **ZSPACE** can be easily embedded in C++ projects. 

Optionally the library may also be pre-compiled into a statically  or dynamically linked library, for faster compile times.

- [Citing](#Citing)
- [License](#license)
- [Third party dependcencies](#used-third-party-dependencies)

# Citing
If you use the library of ZSPACE in a project, please refer to the GitHub repository. <br/> <br/>
@misc{zspace-framework, <br/>
      title  = {{zspace}: A simple C++ header-only collection of geometry data-structures, algorithms and city data visualization                       framework.},<br/>
      author = {Vishu Bhooshan and Shajay Bhooshan and others},<br/>
      note   = {https://github.com/venumb/ZSPACE},<br/>
      year   = {2018},<br/>
    }

# License
The library is licensed under the [MIT License](https://opensource.org/licenses/MIT).


# Third party dependencies
The library has some dependencies on third-party tools and services, which have different licensing as listed below.
Thanks a lot!

- [**OPENGL**](https://www.opengl.org/about/) for display methods. End users, independent software vendors, and others writing code based on the OpenGL API are free from licensing requirements.

- [**Eigen**](https://github.com/eigenteam/eigen-git-mirror) for matricies and related methods. It is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/).

- [**Spectra**](https://github.com/yixuan/spectra) for large scale eigen value problems. It is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/).

- [**Armadillo**](http://arma.sourceforge.net/) for matricies and related methods. It is an open source project licensed under
[Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

- [**Alglib**](http://http://www.alglib.net/) free edition for linear programming optimisation methods.

- [**JSON for Modern C++**](https://github.com/nlohmann/json) to create a JSON file. It is an open source project licensed under
[MIT License](https://opensource.org/licenses/MIT).

- [**SQLITE**](https://www.sqlite.org/index.html) for SQL database engine. It is an open source project dedicated to the [public domain](https://en.wikipedia.org/wiki/Public_domain).

- [**LodePNG**](https://lodev.org/lodepng) for creating PNG images. It is a project licensed under 
[ZLIB License](https://zlib.net/zlib_license.html).

- [**TooJPEG**](https://create.stephan-brumme.com/toojpeg/) for creating JPEG images. It is a project licensed under 
[ZLIB License](https://zlib.net/zlib_license.html).
