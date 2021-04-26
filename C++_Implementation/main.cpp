#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "src/Neural_Net.hpp"
#include "src/Make_Data.hpp"
#include "src/Hironaka_Decomp.hpp"
#include "src/Overloading.hpp"

std::vector<double> my_func(std::vector<double> x)
{
    //Regression problem
    //return {0.3 * x[0] * x[0] - x[0] + 0.5 * sin(20 * x[0])};
    
    //Classification problem
    if(sin(2 * x[0]) > 0)
        return {1};
    else
        return {0};
}

int main(int argc, char** argv)
{
    
    //Instatialise and construct net from file
    NeuralNet net = NeuralNet();
    net.Construct_net("net_files/net.txt");
    std::cout<< "Number of params = " << net.Number_params() << std::endl;
    

    //Make (or Read) training data
    int N_data = 100;
    double x_min = -1, x_max = 1;
    
    MakeData data_train = MakeData();
    data_train.Make(my_func, N_data, x_min, x_max);
    //data_train.Read("data.txt");
    
    //Format into Hironaka decomposition or minimal algebra generators
    HironakaDecomp HD = HironakaDecomp();
    HD.Set_HD("Gen_files/test.txt");
    HD.Make(data_train.Get_X_data());
    
    //Set data in net
    auto y_data = data_train.Get_y_data();
    auto X_data = HD.Get_input_data();
    net.Set_X_data(X_data);
    net.Set_y_data(y_data);
    
    //Train net
    int epochs = 100;
    int batches = 10;
    double test_train_ratio = 0.2;
    net.Initialize_weights_biases(0.0, 1.0); //gaussian with (mean, std)
    net.Adam_optimizer(epochs, batches, test_train_ratio);
    
    //Make test data
    N_data = 300;
    x_min = -3, x_max = 3;
    MakeData data_test = MakeData();
    data_test.Make(my_func, N_data, x_min, x_max);
    HD.Make(data_test.Get_X_data());
    
    auto y_data_test = data_test.Get_y_data();
    auto X_data_test = HD.Get_input_data();
    
    std::ofstream file1;
    file1.open ("output/true.txt");
    
    for(int i = 0; i < N_data; i++)
        file1 << X_data_test[i][0] << " " << y_data_test[i][0] << std::endl;

    file1.close();
    
    std::ofstream file2;
    file2.open ("output/prediction.txt");
    
    for(int i = 0; i < N_data; i++)
        file2 << X_data_test[i][0] << " " << net.Evaluate_net(X_data_test[i])[0] << std::endl;
    
    file2.close();
    
    return 0;
}
