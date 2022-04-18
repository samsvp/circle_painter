g++ main.cpp -o main `pkg-config --libs opencv` -Ofast -std=c++17 \
    -ltbb -lpthread
