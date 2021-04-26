#include "Make_Data.hpp"
#include "Overloading.hpp"

void MakeData::Make(std::function<std::vector<double>(std::vector<double>)> function, int N_data, double x_min, double x_max)
{
    
    for(int i = 0; i < N_data; i++)
    {
        double xi = x_min + (1.0*i/N_data) * (x_max-x_min);
        X_data.push_back({xi});
    }
    
    for(int i = 0; i < N_data; i++)
    {
        y_data.push_back(function(X_data[i]));
    }
    
}

void MakeData::Read(std::string filename)
{
    std::string line;
    std::ifstream myfile(filename);
    
    int counter = 0;
    while (std::getline(myfile, line))
    {
        std::stringstream iss(line);
        
        double number;
        std::vector<double> tmp;
        
        while ( iss >> number )
            tmp.push_back( number );
        
        if(counter == 0)
        {
            X_data.push_back(tmp);
            counter = 1;
        }
        else if(counter == 1)
        {
            y_data.push_back(tmp);
            counter = 0;
        }
    }
}


std::vector<std::vector<double>> MakeData::Get_X_data()
{
    
    return X_data;
}

std::vector<std::vector<double>> MakeData::Get_y_data()
{
    return y_data;
}
