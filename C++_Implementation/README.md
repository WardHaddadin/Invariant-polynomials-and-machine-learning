Run \$ make to compile.

 

The file net_files/net.txt contains the net structure which is in the following
format:

 

\<Layer_type\> \<Number_nodes\> \<Activation_function\>

...

Loss = \<Loss_function\>

 

Options:

\<Layer_type\>: Input, Dense, Secondary, Output

\<Activation_function\>: No_activation (for input), tanh, sigmoid, relu,
leaky_relu, linear

\<Loss_function\>: mean_squared_error, mean_absolute_error,
binary_cross_entropy, categorical_cross_entropy (not stable)

 

 

 
