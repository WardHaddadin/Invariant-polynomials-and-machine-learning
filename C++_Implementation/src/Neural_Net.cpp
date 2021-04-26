#include "Neural_Net.hpp"
#include "Overloading.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Adding Layers and Setting Loss ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNet::Add_Input_layer(int layer_width)
{
    if(std::find(Net_layers.begin(), Net_layers.end(), INPUT) != Net_layers.end())
        throw std::invalid_argument("Net already has an input layer");
    
    num_layers += 1;
    input_dim = layer_width;
    Net_layers.push_back(INPUT);
    layer_acts.push_back(NO_ACT);
    layer_widths.push_back(layer_width);
    
    w_kij.push_back({});
    b_ki.push_back({});
}

void NeuralNet::Add_Dense_layer(int layer_width, std::string act)
{
    num_layers += 1;
    Net_layers.push_back(DENSE);
    layer_acts.push_back(ToAct(act));
    layer_widths.push_back(layer_width);
    std::vector<std::vector<double>> tmp_weights;
    
    if(num_layers == 2)
    {
        tmp_weights.resize(layer_width, std::vector<double>(input_dim));
    }
    else
    {
        tmp_weights.resize(layer_width, std::vector<double>(w_kij.back().size()));
    }
    
    w_kij.push_back(tmp_weights);
    
    std::vector<double> tmp_biases(layer_width);
    
    b_ki.push_back(tmp_biases);
    
    num_weights = NeuralNet::Number_weights();
    num_biases = NeuralNet::Number_biases();
}

void NeuralNet::Add_Secondary_layer(int layer_width, std::string act)
{
    if(std::find(Net_layers.begin(), Net_layers.end(), SECONDARY) != Net_layers.end())
        throw std::invalid_argument("Net already has a secondary layer");
    
    num_layers += 1;
    Net_layers.push_back(SECONDARY);
    layer_acts.push_back(ToAct(act));
    layer_widths.push_back(layer_width);
    
    std::vector<std::vector<double>> tmp_weights(layer_width, std::vector<double>(w_kij.back().size()));
    
    w_kij.push_back(tmp_weights);
    
    std::vector<double> tmp_biases(layer_width);
    
    b_ki.push_back(tmp_biases);
    
    num_weights = NeuralNet::Number_weights();
    num_biases = NeuralNet::Number_biases();
}

void NeuralNet::Add_Output_layer(int layer_width, std::string act)
{
    if(std::find(Net_layers.begin(), Net_layers.end(), OUTPUT) != Net_layers.end())
        throw std::invalid_argument("Net already has an output layer");
    
    num_layers += 1;
    Net_layers.push_back(OUTPUT);
    layer_acts.push_back(ToAct(act));
    layer_widths.push_back(layer_width);
    
    std::vector<std::vector<double>> tmp_weights(layer_width, std::vector<double>(w_kij.back().size()));
    
    w_kij.push_back(tmp_weights);
    
    std::vector<double> tmp_biases(layer_width);
    
    b_ki.push_back(tmp_biases);
    
    num_weights = NeuralNet::Number_weights();
    num_biases = NeuralNet::Number_biases();
}

void NeuralNet::Set_loss_function(std::string loss)
{
    net_loss_function = ToLoss(loss);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////Activation and loss functions////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


double NeuralNet::Activation_func(double x, Act_type act)
{
    if(act == TANH)
        return tanh(x);
    else if(act == SIGMOID)
        return 1 / (1 + exp(-x));
    else if(act == RELU)
        return std::max(x,0.0);
    else if(act == LEAKY_RELU)
        return std::max(x,0.0) + std::max(-0.01 * x, 0.0);
    else if(act == LINEAR)
        return x;
    else
        throw std::invalid_argument("Not an activation layer:" + ToString(act));
        
}

double NeuralNet::Grad_activation_func(double x, Act_type act)
{
    if(act == TANH)
    {
        double tanh_ = tanh(x);
        return 1 - tanh_ * tanh_;
    }
    else if(act == SIGMOID)
    {
        double sig = 1 / (1 + exp(-x));
        return sig * (1 - sig);
    }
    else if(act == RELU)
    {
        if(x > 0)
            return 1;
        else
            return 0;
    }
    else if(act == LEAKY_RELU)
    {
        if(x > 0)
            return 1;
        else
            return -0.01;
    }
    else if(act == LINEAR)
        return 1;
    else
        throw std::invalid_argument("Not an activation layer:" + ToString(act));
        
}

double NeuralNet::Loss_func(std::vector<double> y_true, std::vector<double> y_pred)
{
    double loss = 0;

    if(net_loss_function == MSE)
    {
    	for(int i = 0; i < y_true.size(); i++)
    	{
    	    loss += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    	}
        loss = loss / y_true.size();
    }
    else if(net_loss_function == MAE)
    {
        for(int i = 0; i < y_true.size(); i++)
        {
            loss += abs(y_true[i] - y_pred[i]);
        }
        loss = loss / y_true.size();
    }
    else if(net_loss_function == BCE)
    {
        double h = 1e-8;
        loss += -( y_true[0] * log(h + y_pred[0]) +  (1 - y_true[0]) * log(h + 1 - y_pred[0]) );
    }
    else if(net_loss_function == CCE)
    {
        double h = 1e-8;
        for(int i = 0; i < y_true.size(); i++)
        {
            loss += - y_true[i] * log(h + y_pred[i]);
        }
        double tmp = 0;
        for(int i = 0; i < y_true.size(); i++)
        {
            tmp += y_pred[i];
        }
        loss = loss + 100 * (tmp - 1) * (tmp - 1);
    }
    else
        throw std::invalid_argument("Not a Loss function");
        
    return loss;
}

std::vector<double> NeuralNet::Grad_loss_func(std::vector<double> y_true, std::vector<double> y_pred)
{
    std::vector<double> grad_loss(y_true.size());
    
    if(net_loss_function == MSE)
    {
        for(int i = 0; i < y_true.size(); i++)
        {
            grad_loss[i] = - 2 * (y_true[i] - y_pred[i]);
        }
    }
    else if(net_loss_function == MAE)
    {
        for(int i = 0; i < y_true.size(); i++)
        {
            if(y_pred[i] > y_true[i])
                grad_loss[i] = 1;
            else
                grad_loss[i] = -1;
        }
    }
    else if(net_loss_function == BCE)
    {
        double h = 1e-8;
        grad_loss[0] = - ( y_true[0] / (h + y_pred[0]) -  (1 - y_true[0]) / (h + 1 - y_pred[0]) );
    }
    else if(net_loss_function == CCE)
    {
        double h = 1e-8;
        
        double tmp = 0;
        for(int i = 0; i < y_true.size(); i++)
        {
            tmp += y_pred[i];
        }
        
        for(int i = 0; i < y_true.size(); i++)
        {
            grad_loss[i] = - y_true[i] / (h + y_pred[i]) + 100 * 2 * (tmp - 1);
        }

    }
    else
        throw std::invalid_argument("Not a Loss function");

    return grad_loss;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Evaluating net, layers, gradients, loss ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double NeuralNet::Evaluate_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input)
{
    std::vector<double> Net_output = NeuralNet::Evaluate_net(X_data_input);
    double loss = NeuralNet::Loss_func(y_data_input, Net_output);
    
    return loss;
}

std::vector<double> NeuralNet::test_grad_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input)
{
    double h = 1e-5;
    std::vector<double> weights_biases = NeuralNet::Get_weights_biases();
    std::vector<double> grad_loss(weights_biases.size());
    
    for(int i = 0; i < weights_biases.size(); i++)
    {
        std::vector<double> tmp1 = weights_biases;
        std::vector<double> tmp2 = weights_biases;
        
        tmp1[i] += h;
        tmp2[i] -= h;
        
        NeuralNet::Set_weights_biases(tmp1);
        grad_loss[i] += NeuralNet::Evaluate_loss(X_data_input, y_data_input);
        
        NeuralNet::Set_weights_biases(tmp2);
        grad_loss[i] -=  NeuralNet::Evaluate_loss(X_data_input, y_data_input);
        
        grad_loss[i] /= 2 * h;
        
    }
    
    return grad_loss;
}

std::vector<double> NeuralNet::Evaluate_grad_loss(std::vector<std::vector<double>> X_data_input, std::vector<double> y_data_input)
{
    std::vector<double> grad_loss;
    
    std::vector<std::vector<double>> df_dwb = NeuralNet::Evaluate_gradients(X_data_input);
    std::vector<double> Net_output = NeuralNet::Evaluate_net(X_data_input);
    
    grad_loss = NeuralNet::Grad_loss_func(y_data_input, Net_output) * df_dwb;
    
    return grad_loss;
}

std::vector<double> NeuralNet::Evaluate_net(std::vector<std::vector<double>> data_input)
{
    
    for(int k = 0; k < num_layers; k++)
    {
        data_input[0] = NeuralNet::Evaluate_layer(k, data_input);
    }
    
    return data_input[0];
}

std::vector<double> NeuralNet::test_grad(std::vector<std::vector<double>> input)
{
    double h = 1e-5;
    std::vector<double> weights_biases = NeuralNet::Get_weights_biases();
    std::vector<double> grad(weights_biases.size());
    
    for(int i = 0; i < weights_biases.size(); i++)
    {
        std::vector<double> tmp1 = weights_biases;
        std::vector<double> tmp2 = weights_biases;
        
        tmp1[i] += h;
        tmp2[i] -= h;
        
        NeuralNet::Set_weights_biases(tmp1);
        grad[i] += NeuralNet::Evaluate_net(input)[0];
        
        NeuralNet::Set_weights_biases(tmp2);
        grad[i] -= NeuralNet::Evaluate_net(input)[0];
        
        grad[i] /= 2 * h;
        
    }
    
    return grad;
}

std::vector<std::vector<double>> NeuralNet::Evaluate_gradients(std::vector<std::vector<double>> input)
{
    std::vector<std::vector<double>> A_ki(num_layers);
    std::vector<std::vector<std::vector<double>>> AA_kij(num_layers - 1);
    std::vector<std::vector<std::vector<double>>> WA_kij(num_layers - 1);
    std::vector<std::vector<double>> BA_ki(num_layers - 1);
    std::vector<std::vector<std::vector<double>>> df_db_ki(layer_widths.back(), std::vector<std::vector<double>>(num_layers));
    std::vector<std::vector<std::vector<std::vector<double>>>> df_dw_kij(layer_widths.back(), std::vector<std::vector<std::vector<double>>>(num_layers));
    
    A_ki[0] = input[0];
    for(int k = 1; k < num_layers; k++)
    {
        if(Net_layers[k] == SECONDARY)
            A_ki[k] = Evaluate_layer(k, {{A_ki[k-1]},{input[1]}});
        else
            A_ki[k] = Evaluate_layer(k, {{A_ki[k-1]},{0}});
    }
    
    for(int k = num_layers - 1; k > 0; k--)
    {
        std::vector<double> grad_acts;
        if(Net_layers[k] == SECONDARY)
            grad_acts = Evaluate_layer_grad_act(k, {{A_ki[k-1]},{input[1]}});
        else
            grad_acts = Evaluate_layer_grad_act(k, {{A_ki[k-1]},{}});
        
        AA_kij[num_layers - 1 - k] = grad_acts % w_kij[k]; //Element wise product
        WA_kij[num_layers - 1 - k] = grad_acts ^ A_ki[k-1]; //Outer product
        BA_ki[num_layers - 1 - k] = grad_acts;
    }

    
    std::vector<std::vector<double>> Mat = Identity_mat(WA_kij[0].size());
    for(int k = 0; k < num_layers - 1; k++)
    {
        std::vector<std::vector<std::vector<double>>> tmp1 = Mat % WA_kij[k]; //Element wise product
        std::vector<std::vector<double>> tmp2 = Mat % BA_ki[k]; //Element wise product
        
        for(int m = 0; m < layer_widths.back(); m++)
        {
            df_dw_kij[m][num_layers - 1 - k] = tmp1[m];
            df_db_ki[m][num_layers - 1 - k] = tmp2[m];
        }
        Mat = Mat * AA_kij[k];
    }
    
    std::vector<std::vector<double>> gradients(layer_widths.back(), std::vector<double>(num_weights+num_biases));
    int count = 0;
    for(int m = 0; m < layer_widths.back(); m++)
    {
        for(int k = 0; k < df_dw_kij[m].size(); k++)
        {
            for(int i = 0; i < df_dw_kij[m][k].size(); i++)
            {
                for(int j = 0; j < df_dw_kij[m][k][i].size(); j++)
                {
                    gradients[m][count] = df_dw_kij[m][k][i][j];
                    count += 1;
                }
            }
        }
    }
    
    
    for(int m = 0; m < layer_widths.back(); m++)
    {
        for(int k = 0; k < df_db_ki[m].size(); k++)
        {
            for(int i = 0; i < df_db_ki[m][k].size(); i++)
            {
                gradients[m][count] = df_db_ki[m][k][i];
                count += 1;
            }
        }
    }
    
    
    return gradients;
    
}

std::vector<double> NeuralNet::Evaluate_layer(int k, std::vector<std::vector<double>> input)
{
    std::vector<double> tmp;
    
    Layer_type layer_type = Net_layers[k];
    Act_type act_type = layer_acts[k];
    
    if( layer_type == INPUT )
    {
        tmp = input[0];
    }
    else if( layer_type == DENSE )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Activation_func(arg_tmp[i] + bias[i], act_type);
        
    }
    else if( layer_type == SECONDARY )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        
        if (weight.size() != input[1].size())
                throw std::invalid_argument(std::string("Secondary input wrong size, should be ") + std::string(std::to_string(weight.size())) + std::string(" but instead is ") + std::string(std::to_string(input[1].size())));
        
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Activation_func(arg_tmp[i] + bias[i], act_type) * input[1][i];
        
    }
    else if( layer_type == OUTPUT )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Activation_func(arg_tmp[i] + bias[i], act_type);

    }
    else
        throw std::invalid_argument(std::string("Layer " + ToString(layer_type) + " " + std::to_string(k) + " does not exist in Evaluate_layer()"));
    
    return tmp;
}


std::vector<double> NeuralNet::Evaluate_layer_grad_act(int k, std::vector<std::vector<double>> input)
{
    std::vector<double> tmp;
    
    Layer_type layer_type = Net_layers[k];
    Act_type act_type = layer_acts[k];
    
    if( layer_type == DENSE )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Grad_activation_func(arg_tmp[i] + bias[i], act_type);
        
    }
    else if( layer_type == SECONDARY )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        
        if (weight.size() != input[1].size())
                throw std::invalid_argument(std::string("Secondary input wrong size, should be ") + std::string(std::to_string(weight.size())) + std::string(" but instead is ") + std::string(std::to_string(input[1].size())));
        
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Grad_activation_func(arg_tmp[i] + bias[i], act_type) * input[1][i];
        
    }
    else if( layer_type == OUTPUT )
    {
        std::vector<std::vector<double>> weight = w_kij[k];
        std::vector<double> bias = b_ki[k];
        std::vector<double> arg_tmp = weight * input[0];
        tmp.resize(weight.size());
        
        for(int i = 0; i < weight.size(); i++)
            tmp[i] = NeuralNet::Grad_activation_func(arg_tmp[i] + bias[i], act_type);

    }
    else
        throw std::invalid_argument(std::string("Layer " + ToString(layer_type) + " does not exist in Evaluate_layer_grad_act()"));
    
    return tmp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Adam optimiser ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void NeuralNet::Adam_optimizer(int epochs, int batches, double test_train_ratio)
{
    double b1 = 0.9;
    double b2 = 0.999;
    double ep = 1e-8;
    double alpha = 0.001;
    
    std::vector<std::vector<std::vector<double>>> X_train, X_test;
    std::vector<std::vector<double>> y_train, y_test;
    
    std::vector<int> indexes_test_train;
    for(int i = 0; i < X_data.size(); i++)
        indexes_test_train.push_back(i);
    std::random_shuffle(indexes_test_train.begin(), indexes_test_train.end());
    
    int num_train = (int)((1-test_train_ratio) * X_data.size());
    int num_test = X_data.size() - num_train;
    
    for(int i = 0; i < num_train; i++)
    {
        X_train.push_back(X_data[indexes_test_train[i]]);
        y_train.push_back(y_data[indexes_test_train[i]]);
    }
    for(int i = num_train; i < X_data.size(); i++)
    {
        X_test.push_back(X_data[indexes_test_train[i]]);
        y_test.push_back(y_data[indexes_test_train[i]]);
    }
    
    int reps = (X_train.size() / batches);
    if( (X_train.size() % batches) != 0 )
        reps += 1;
    
    std::vector<std::vector<double>> m(reps, std::vector<double>(num_weights + num_biases));
    std::vector<std::vector<double>> v(reps, std::vector<double>(num_weights + num_biases));
    
    for(int t = 1; t < epochs + 1; t++)
    {
        std::cout << "epochs = " <<  t << "/" << std::to_string(epochs) << ", ";
        
        std::vector<int> indexes;
        for(int i = 0; i < X_train.size(); i++)
            indexes.push_back(i);
        std::random_shuffle(indexes.begin(), indexes.end());
        
        double alpha_t = alpha * sqrt(1 - pow(b2, t)) / (1 - pow(b1, t));
        
        for(int r = 0; r < reps; r++)
        {
            std::vector<double> avg_grad(num_weights + num_biases);
            
            int batches_left;
            if(r == reps - 1)
            {
                batches_left = indexes.size() - r * batches;
            }
            else
                batches_left = batches;
            
            for(int j = 0; j < batches_left; j++)
            {
                avg_grad = avg_grad + Evaluate_grad_loss(X_train[indexes[r * batches + j]], y_train[indexes[r * batches + j]]);
            }
            
            avg_grad = avg_grad /  batches_left;
            
            m[r] = b1 * m[r] + (1 - b1) * avg_grad;
            v[r] = b2 * v[r] + (1 - b2) * (avg_grad % avg_grad);
            
            std::vector<double> change;
            for(int j = 0; j < avg_grad.size(); j++)
            {
                change.push_back( -alpha_t * m[r][j] / (sqrt(v[r][j]) + ep) );
            }
            NeuralNet::Set_weights_biases( NeuralNet::Get_weights_biases() + change );
        }
        
        double loss = 0;
        for(int i = 0; i < X_train.size(); i++)
        {
            loss += NeuralNet::Evaluate_loss(X_train[i], y_train[i]);
        }
        loss /= X_train.size();
        std::cout << "loss = " << loss;
        
        double val_loss = 0;
        for(int i = 0; i < X_test.size(); i++)
        {
            val_loss += NeuralNet::Evaluate_loss(X_test[i], y_test[i]);
        }
        val_loss /= X_test.size();
        std::cout << ", val_loss = " << val_loss << std::endl;
        
    }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// likelihoods and priors ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double NeuralNet::log_likelihood(std::vector<double> weights_biases)
{
    NeuralNet::Set_weights_biases(weights_biases);
    double logL = 0;
    
    for(int i = 0; i < X_data.size(); i++)
    {
        logL += - 5 * NeuralNet::Evaluate_loss(X_data[i], y_data[i]);
    }
    
    return logL;
}

std::vector<double> NeuralNet::grad_log_likelihood(std::vector<double> weights_biases)
{
    double h = 1e-5;
    std::vector<double> grad( weights_biases.size());
    
    for(int i = 0; i < weights_biases.size(); i++)
    {
        std::vector<double> tmp1 = weights_biases;
        std::vector<double> tmp2 = weights_biases;
        
        tmp1[i] += h;
        tmp2[i] -= h;
        
        grad[i] = (NeuralNet::log_likelihood(tmp1) - NeuralNet::log_likelihood(tmp2)) / (2 * h);
    }
    
    return grad;
}

double NeuralNet::log_prior(std::vector<double> weights_biases)
{
    double reg_coef = 0.00001;
    
    double sum = 0;
    for(int i = 0; i < weights_biases.size(); i++)
    {
        sum += weights_biases[i] * weights_biases[i];
    }
    return - reg_coef * sum / weights_biases.size();
}

std::vector<double> NeuralNet::grad_log_prior(std::vector<double> weights_biases)
{
    double h = 1e-5;
    std::vector<double> grad( weights_biases.size());
    
    for(int i = 0; i < weights_biases.size(); i++)
    {
        std::vector<double> tmp1 = weights_biases;
        std::vector<double> tmp2 = weights_biases;
        
        tmp1[i] += h;
        tmp2[i] -= h;
        
        grad[i] = (NeuralNet::log_prior(tmp1) - NeuralNet::log_prior(tmp2)) / (2 * h);
    }
    
    return grad;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Getting and Setting weights and biases and data ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double> NeuralNet::Get_weights()
{
    std::vector<double> tmp_weights(num_weights);
    
    int counter = 0;
    
    for(int k = 0; k < w_kij.size(); k++)
    {
        for(int i = 0; i < w_kij[k].size(); i++)
        {
            for(int j = 0; j < w_kij[k][i].size(); j++)
            {
                tmp_weights[counter] = w_kij[k][i][j];
                counter += 1;
            }
        }
    }
    
    return tmp_weights;
}

std::vector<double> NeuralNet::Get_biases()
{
    std::vector<double> tmp_biases(num_biases);
    
    int counter = 0;
    
    for(int k = 0; k < b_ki.size(); k++)
    {
        for(int i = 0; i < b_ki[k].size(); i++)
        {
            tmp_biases[counter] = b_ki[k][i];
            counter += 1;
        }
    }
    
    return tmp_biases;
}

void NeuralNet::Set_weights(std::vector<double> _w_kij)
{
    int counter = 0;
    
    for(int k = 0; k < w_kij.size(); k++)
    {
        for(int i = 0; i < w_kij[k].size(); i++)
        {
            for(int j = 0; j < w_kij[k][i].size(); j++)
            {
                w_kij[k][i][j] = _w_kij[counter];
                counter += 1;
            }
        }
    }
}

void NeuralNet::Set_biases(std::vector<double> _b_ki)
{
    int counter = 0;
    
    for(int k = 0; k < b_ki.size(); k++)
    {
        for(int i = 0; i < b_ki[k].size(); i++)
        {
            b_ki[k][i] = _b_ki[counter];
            counter += 1;
        }
    }
}


std::vector<double> NeuralNet::Get_weights_biases()
{
    int counter = 0;
    std::vector<double> tmp_weights_biases(num_weights + num_biases);
    
    for(int k = 0; k < w_kij.size(); k++)
    {
        for(int i = 0; i < w_kij[k].size(); i++)
        {
            for(int j = 0; j < w_kij[k][i].size(); j++)
            {
                tmp_weights_biases[counter] = w_kij[k][i][j];
                counter += 1;
            }
        }
    }
    
    for(int k = 0; k < b_ki.size(); k++)
    {
        for(int i = 0; i < b_ki[k].size(); i++)
        {
            tmp_weights_biases[counter] = b_ki[k][i];
            counter += 1;
        }
    }
    
    return tmp_weights_biases;
}

void NeuralNet::Set_weights_biases(std::vector<double> _w_kij_b_ki)
{
    int counter = 0;
    
    for(int k = 0; k < w_kij.size(); k++)
    {
        for(int i = 0; i < w_kij[k].size(); i++)
        {
            for(int j = 0; j < w_kij[k][i].size(); j++)
            {
                w_kij[k][i][j] = _w_kij_b_ki[counter];
                counter += 1;
            }
        }
    }
    
    for(int k = 0; k < b_ki.size(); k++)
    {
        for(int i = 0; i < b_ki[k].size(); i++)
        {
            b_ki[k][i] = _w_kij_b_ki[counter];
            counter += 1;
        }
    }
}

void NeuralNet::Initialize_weights_biases(double mu, double sigma)
{
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> dist_normal(mu, sigma);
    int num_params = NeuralNet::Number_params();
    std::vector<double> _w_kij_b_ki;
    for(int i = 0; i < num_params; i++)
        _w_kij_b_ki.push_back(dist_normal(generator));
    
    NeuralNet::Set_weights_biases(_w_kij_b_ki);
}

void NeuralNet::Set_X_data(std::vector<std::vector<std::vector<double>>> _X_data)
{
    X_data = _X_data;
}

void NeuralNet::Set_y_data(std::vector<std::vector<double>> _y_data)
{
    y_data = _y_data;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Printing and information ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNet::Print_weights(bool Explicit_params)
{
    std::cout << "Weights Matrices" <<std::endl;

    for(int k = 0; k < Net_layers.size(); k++)
    {
        
        if ( Net_layers[k] != INPUT )
        {
            std::cout << "Layer " << ToString(Net_layers[k]) << ", dim = " << w_kij[k].size() << " x " << w_kij[k][0].size() << std::endl;
            if(Explicit_params)
            {
                for(int i = 0; i < w_kij[k].size(); i++)
                {
                    std::cout << "{ " ;
                    for(int j = 0; j < w_kij[k][i].size(); j++)
                    {
                        std::cout << w_kij[k][i][j] << " " ;
                    }
                    std::cout << "}" << std::endl;
                }
            }
        }
        else
        {
            std::cout << "Layer " << ToString(Net_layers[k]) << ", dim = " << input_dim << std::endl;
        }
    }
    std::cout << std::endl;
}

void NeuralNet::Print_biases(bool Explicit_params)
{
    std::cout << "Biases Vectors" <<std::endl;
    
    for(int k = 0; k < Net_layers.size(); k++)
    {
        if ( Net_layers[k] != INPUT )
        {
            std::cout << "Layer " << ToString(Net_layers[k]) << ", dim = " << b_ki[k].size() << std::endl;
            
            if(Explicit_params)
            {
                std::cout << "{ " ;
                for(int i = 0; i < b_ki[k].size(); i++)
                {
                    std::cout << b_ki[k][i] << " ";
                }
                std::cout << "}" << std::endl;
            }
        }
        else
        {
            std::cout << "Layer " << ToString(Net_layers[k]) << ", dim = " << input_dim << std::endl;
        }
    }
    std::cout << std::endl;
}

int NeuralNet::Number_weights()
{
    int Num = 0;
    
    for(int k = 0; k < w_kij.size(); k++)
    {
        for(int i = 0; i < w_kij[k].size(); i++)
        {
            Num +=  w_kij[k][i].size();
        }
    }
    
    return Num;
}

int NeuralNet::Number_biases()
{
    int Num = 0;
    
    for(int k = 0; k < b_ki.size(); k++)
    {
        Num +=  b_ki[k].size();
    }
    
    return Num;
}

int NeuralNet::Number_params()
{
    return NeuralNet::Number_biases() +  NeuralNet::Number_weights();
}

void NeuralNet::Save_net(std::string filename)
{
    std::ofstream file;
    file.open (filename);
    
    for(int i = 0; i < num_layers; i++)
    {
        file << ToString(Net_layers[i]) << " " << std::to_string(layer_widths[i]) << " " << ToString(layer_acts[i]) << std::endl;
    }
    file.close();
}

void NeuralNet::Construct_net(std::string filename)
{
    std::string line;
    std::ifstream myfile(filename);
    
    Net_layers.clear();
    layer_widths.clear();
    layer_acts.clear();
    
    std::cout << "Constructing net: " <<std::endl;
    
    while (std::getline(myfile, line))
    {
        std::stringstream iss(line);
        
        std::string layer_type;
        std::string layer_width;
        std::string layer_act;
        
        iss >> layer_type;
        iss >> layer_width;
        iss >> layer_act;
        
        if(ToLayerType(layer_type) == INPUT)
            Add_Input_layer(std::stoi(layer_width));
        else if(ToLayerType(layer_type) == DENSE)
            Add_Dense_layer(std::stoi(layer_width), layer_act);
        else if(ToLayerType(layer_type) == SECONDARY)
            Add_Secondary_layer(std::stoi(layer_width), layer_act);
        else if(ToLayerType(layer_type) == OUTPUT)
            Add_Output_layer(std::stoi(layer_width), layer_act);
        else if(ToLoss(layer_act) != UNKNOWN_LOSS)
            Set_loss_function(layer_act);
        else
            throw std::invalid_argument("Error in constructing file " + layer_type + " " + layer_width + " " + layer_act );
                            
        
        std::cout << layer_type << " " << layer_width << " " << layer_act << std::endl;
    }
    
    std::cout<< std::endl;
    
}
