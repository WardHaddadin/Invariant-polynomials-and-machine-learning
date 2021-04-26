#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <vector>

class HironakaDecomp
{
    
public:
    
    HironakaDecomp() {};
    
    void Set_HD(std::string Gen_file);
    void Make(std::vector<std::vector<double>> X_data);
    
    std::vector<std::vector<std::vector<double>>> Get_input_data();
    
private:

    std::vector<std::vector<std::vector<std::vector<double>>>> weyls;
    std::vector<std::vector<std::vector<std::vector<double>>>> prims;
    std::vector<std::vector<std::vector<std::vector<double>>>> secs;
    
    std::vector<std::vector<std::vector<double>>> input_data;
    
};

