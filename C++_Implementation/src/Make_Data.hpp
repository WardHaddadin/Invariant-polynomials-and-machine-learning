#pragma once
#include <functional>
#include <fstream>
#include <sstream>
#include <vector>

class MakeData
{
    
public:
    
    MakeData() {};
    
    void Make(std::function<std::vector<double>(std::vector<double>)> function, int N_data, double x_min, double x_max);
    
    void Read(std::string filename);
    
    std::vector<std::vector<double>> Get_X_data();
    std::vector<std::vector<double>> Get_y_data();
    
private:
    
    std::vector<std::vector<double>> X_data;
    std::vector<std::vector<double>> y_data;
    
};

