# ZSPACE
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/venumb/zSpace_v003/LICENSE.MIT)

**ZSPACE** is a C++  library collection of geometry data-structures, algorithms and city data visualization framework. It is implemented as a header-only C++ library, whose dependencies, are mostly header-only libraries except for **OPENGL** and **SQLITE** in which case the  static libraries are included. Hence **ZSPACE** can be easily embedded in C++ projects. 

- [License](#license)
- [Used third-party dependcencies](#used-third-party-dependencies)

## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">
The library is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# used-third-party-dependencies
The library itself consists of header only files licensed under the MIT license. 

However, it has some dependencies on third-party tools and services, which have different licensing. Thanks a lot!

- [**OPENGL**](https://www.opengl.org/about/) for display methods. End users, independent software vendors, and others writing code based on the OpenGL API are free from licensing requirements.

- [**Eigen**](https://github.com/eigenteam/eigen-git-mirror) for matricies and related methods. It is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/).

- [**Spectra**](https://github.com/yixuan/spectra) for large scale eigen value problems. It is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/).

- [**JSON for Modern C++**](https://github.com/nlohmann/json) to create a JSON file. It is an open source project licensed under
[MIT License](https://opensource.org/licenses/MIT).

- [**SQLITE**](https://www.sqlite.org/index.html) for SQL database engine. It is an open source project dedicated to the public domain.
