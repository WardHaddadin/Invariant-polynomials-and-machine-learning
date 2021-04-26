#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <math.h>
#include <vector>
#include <algorithm>
#include <stdexcept>

class NeuralNet
{
    
public:
    
    enum Layer_type { INPUT, DENSE, SECONDARY, OUTPUT, UNKNOWN_LAYER };
    enum Act_type { NO_ACT, TANH, SIGMOID, RELU, LEAKY_RELU, LINEAR, UNKNOWN_ACT };
    enum Loss_type { MSE, MAE, BCE, CCE, UNKNOWN_LOSS };
    
    NeuralNet() {};
    
    void Add_Input_layer(int layer_width);
    void Add_Dense_layer(int layer_width, std::string act);
    void Add_Secondary_layer(int layer_width, std::string act);
    void Add_Output_layer(int layer_width, std::string act);
    
    double Activation_func(double x, Act_type act);
    double Grad_activation_func(double x, Act_type act);
    double Loss_func(std::vector<double> y_true, std::vector<double> y_pred);
    std::vector<double> Grad_loss_func(std::vector<double> y_true, std::vector<double> y_pred);
    void Set_loss_function(std::string loss);
    
    double Evaluate_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input);
    std::vector<double> Evaluate_net(std::vector<std::vector<double>> X_data_input);
    
    std::vector<double> Evaluate_grad_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input);
    std::vector<double> test_grad_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input);
    
    std::vector<std::vector<double>> Evaluate_gradients(std::vector<std::vector<double>> X_data_input);
    std::vector<double> test_grad(std::vector<std::vector<double>> input);
    
    std::vector<double> Evaluate_layer(int k, std::vector<std::vector<double>> X_data_input);
    std::vector<double> Evaluate_layer_grad_act(int k, std::vector<std::vector<double>> X_data_input);
    
    
    void Adam_optimizer(int epochs, int batches, double test_train_ratio);
    
    double log_likelihood(std::vector<double> weights_biases);
    double log_prior(std::vector<double> weights_biases);
    std::vector<double> grad_log_likelihood(std::vector<double> weights_biases);
    std::vector<double> grad_log_prior(std::vector<double> weights_biases);
    
    
    std::vector<double> Get_weights();
    std::vector<double> Get_biases();
    void Set_weights(std::vector<double> _wij);
    void Set_biases(std::vector<double> _bi);
    std::vector<double> Get_weights_biases();
    void Set_weights_biases(std::vector<double> _w_kij_b_ki);
    void Initialize_weights_biases(double mu, double sigma);
    void Set_X_data(std::vector<std::vector<std::vector<double>>> _X_data);
    void Set_y_data(std::vector<std::vector<double>> _y_data);
    
    void Save_net(std::string filename);
    void Construct_net(std::string filename);
    int Number_weights();
    int Number_biases();
    int Number_params();
    void Print_weights(bool Explicit_params = false);
    void Print_biases(bool Explicit_params = false);

private:
    
    int input_dim;
    int num_layers = 0;
    int num_weights = 0;
    int num_biases = 0;
    
    
    std::vector<Layer_type> Net_layers;
    std::vector<int> layer_widths;
    std::vector<Act_type> layer_acts;
    Loss_type net_loss_function;
    
    std::vector<std::vector<std::vector<double>>> w_kij;
    std::vector<std::vector<double>> b_ki;
    
    std::vector<std::vector<std::vector<double>>> X_data;
    std::vector<std::vector<double>> y_data;
    
};

inline const std::vector<std::vector<double>> Identity_mat(int N)
{
    std::vector<std::vector<double>> id;
    
    for(int i = 0; i < N; i++)
    {
        std::vector<double> tmp;
        for(int j = 0; j < N; j++)
        {
            if(i == j)
                tmp.push_back(1.0);
            else
                tmp.push_back(0.0);
        }
        id.push_back(tmp);
    }
   
   return id;
}

inline const std::string ToString(NeuralNet::Layer_type v)
{
    switch (v)
    {
        case NeuralNet::INPUT:   return "Input";
        case NeuralNet::DENSE:   return "Dense";
        case NeuralNet::SECONDARY: return "Secondary";
        case NeuralNet::OUTPUT: return "Output";
        default:      return "[Unknown Layer_type]";
    }
}

inline const std::string ToString(NeuralNet::Act_type v)
{
    switch (v)
    {
        case NeuralNet::NO_ACT:   return "No_activation";
        case NeuralNet::TANH:   return "tanh";
        case NeuralNet::SIGMOID:   return "sigmoid";
        case NeuralNet::RELU:   return "relu";
        case NeuralNet::LEAKY_RELU: return "leaky_relu";
        case NeuralNet::LINEAR:   return "linear";
        default:      return "[Unknown Layer_type]";
    }
}

inline const NeuralNet::Layer_type ToLayerType(std::string v)
{
    if(v ==  "Input")
        return NeuralNet::INPUT;
    else if(v =="Dense")
        return  NeuralNet::DENSE;
    else if(v == "Secondary")
        return NeuralNet::SECONDARY;
    else if(v == "Output")
        return NeuralNet::OUTPUT;
    else
        return NeuralNet::UNKNOWN_LAYER;
}

inline const NeuralNet::Act_type ToAct(std::string v)
{
    if(v == "No_activation")
        return NeuralNet::NO_ACT;
    else if(v == "tanh")
        return NeuralNet::TANH;
    else if(v == "sigmoid")
        return NeuralNet::SIGMOID;
    else if(v == "relu")
        return NeuralNet::RELU;
    else if(v == "leaky_relu")
        return NeuralNet::LEAKY_RELU;
    else if(v == "linear")
        return NeuralNet::LINEAR;
    else
        return NeuralNet::UNKNOWN_ACT;
}

inline const NeuralNet::Loss_type ToLoss(std::string v)
{
    if(v == "mean_squared_error")
        return NeuralNet::MSE;
    else if(v == "mean_absolute_error")
        return NeuralNet::MAE;
    else if(v == "binary_cross_entropy")
        return NeuralNet::BCE;
    else if(v == "categorical_cross_entropy")
        return NeuralNet::CCE;
    else
        return NeuralNet::UNKNOWN_LOSS;
}
