rm -rf build
cmake -S test -B build -GNinja -Wno-dev -DCMAKE_CXX_COMPILER=/usr/local/houmo/bin/clang++ -DCMAKE_CXX_COMPILER_WORKS=ON \
&& cmake --build build --config=Debug \
&& cd build \
&& ctest --config=Debug \
&& cd ..
