g++ main.cpp -o main.out `pkg-config --libs opencv` -Ofast -std=c++17 \
    -ltbb -lpthread
