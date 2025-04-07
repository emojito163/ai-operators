rm -rf build
cmake -S test -B build -GNinja -DCMAKE_BUILD_TYPE=Debug -Wno-dev
cmake --build build --config=Debug
cd build && ctest --config=Debug && cd ..
