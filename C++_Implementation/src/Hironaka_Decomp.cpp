#include "Hironaka_Decomp.hpp"
#include "Overloading.hpp"

void HironakaDecomp::Set_HD(std::string Gen_file)
{
    std::string line;
    std::ifstream myfile(Gen_file);
    std::vector<std::string> gens_read;

    while (std::getline(myfile, line))
    {
        std::stringstream iss(line);
        
        std::string gen;
        std::string tmp;
        
        while ( iss >> tmp )
        {
            gen += tmp;
        }
        
        gens_read.push_back(gen);
    }
    
    std::string delimiter = "-[";
    
    std::vector<std::vector<std::string>> gens_split(gens_read.size());
    
    for(int i = 0; i < gens_read.size(); i++)
    {
        size_t pos = 0;
        std::string token;
        while ((pos = gens_read[i].find(delimiter)) < std::string::npos) {
            token = gens_read[i].substr(0, pos);
            token.erase(std::remove(token.begin(), token.end(), '['), token.end());
            token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
            gens_split[i].push_back(token);
            gens_read[i].erase(0, pos + delimiter.length());
        }
        gens_read[i].erase(std::remove(gens_read[i].begin(), gens_read[i].end(), '['), gens_read[i].end());
        gens_read[i].erase(std::remove(gens_read[i].begin(), gens_read[i].end(), ']'), gens_read[i].end());
        gens_split[i].push_back(gens_read[i]);
    }
    
    
    
    delimiter = ",";
    
    for(auto gen : gens_split)
    {
        std::string gen_type = gen[0];
        std::vector<std::vector<std::vector<double>>> gen_tmp;
        for(int i = 1; i < gen.size();i++)
        {
            size_t pos = 0;
            std::string token;
            std::vector<std::vector<double>> tmp(2, std::vector<double>());
            int counter = 0;
            while ((pos = gen[i].find(delimiter)) < std::string::npos) {
                
                token = gen[i].substr(0, pos);
                if(counter == 0)
                    tmp[0].push_back(std::stod(token));
                else
                    tmp[1].push_back(std::stod(token));
                
                gen[i].erase(0, pos + delimiter.length());
                counter += 1;
            }
            tmp[1].push_back(std::stod(gen[i]));
        
            gen_tmp.push_back(tmp);
        }
        
        
        if(gen_type == "weyl")
        {
            weyls.push_back(gen_tmp);
        }
        else if(gen_type == "prim")
        {
            prims.push_back(gen_tmp);
        }
        else if(gen_type == "sec")
        {
            secs.push_back(gen_tmp);
        }
    }
    
}

void HironakaDecomp::Make(std::vector<std::vector<double>> X_data)
{
    
    std::vector<std::vector<double>> weyl_data;
    
    for(int i = 0; i < X_data.size(); i++)
    {
        
        std::vector<double> tmp;
        
        for(auto w : weyls)
        {
            double val = 0;
            
            for(auto mono : w)
            {
                if(mono[1].size() != X_data[i].size())
                    throw std::invalid_argument("Dimensions of generator and data do not match");
                
                double val_mono = mono[0][0];
                for(int j = 0; j < X_data[i].size(); j++)
                {
                    val_mono *= pow(X_data[i][j], mono[1][j]);
                }
                val += val_mono;
            }
            
            tmp.push_back(val);
        }
        
        weyl_data.push_back(tmp);
    }
    
    std::vector<std::vector<double>> prims_data;
    
    for(int i = 0; i < weyl_data.size(); i++)
    {
        std::vector<double> tmp;
        
        for(auto p : prims)
        {
            double val = 0;
            
            for(auto mono : p)
            {
                if(mono[1].size() != weyl_data[i].size())
                    throw std::invalid_argument("Dimensions of generator and data do not match");
                
                double val_mono = mono[0][0];
                for(int j = 0; j < weyl_data[i].size(); j++)
                {
                    val_mono *= pow(weyl_data[i][j], mono[1][j]);
                }
                val += val_mono;
            }
            
            tmp.push_back(val);
        }
        
        prims_data.push_back(tmp);
    }
    
    std::vector<std::vector<double>> secs_data;
    
    for(int i = 0; i < weyl_data.size(); i++)
    {
        std::vector<double> tmp;
        
        for(auto s : secs)
        {
            double val = 0;
            
            for(auto mono : s)
            {
                if(mono[1].size() != weyl_data[i].size())
                    throw std::invalid_argument("Dimensions of generator and data do not match");
                
                double val_mono = mono[0][0];
                for(int j = 0; j < weyl_data[i].size(); j++)
                {
                    val_mono *= pow(weyl_data[i][j], mono[1][j]);
                }
                val += val_mono;
            }
            
            tmp.push_back(val);
        }
        
        secs_data.push_back(tmp);
    }
    
    input_data.clear();
    
    for(int i = 0; i < X_data.size(); i++)
        input_data.push_back({prims_data[i], secs_data[i]});
    
}

std::vector<std::vector<std::vector<double>>> HironakaDecomp::Get_input_data()
{
    return input_data;
}
