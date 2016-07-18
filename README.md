# TensorFlow With The Latest Papers Implemented

###Implementation of RNN and NLP Related Neural Network Papers

Currently Implemented Papers:

* Highway Networks
* Recurrent Highway Networks
* Multiplicative Integration Within RNNs


### Highway Networks

https://arxiv.org/abs/1505.00387

Allows greater depth of neural network freely without penalty from upper layers. Ensures shortcut connections within deeper layers.

```python
import highway_networks_modern

output = highway_networks_modern.highway(inputs, num_layers = 3)
```



### Recurrent Highway Networks 

http://arxiv.org/abs/1607.03474

Allows multiple stacking of layers within one cell to increase depth per timestep. 

```python
import rnn_cell_modern

cell = rnn_cell_modern.HighwayRNN(num_units, num_layers)
```


### Multiplicative Integration Within RNNs

https://arxiv.org/abs/1606.06630

Allows faster convergence within RNNs by utilizing the combination of two separate weight matrices in a multiplicative setting

```python
import rnn_cell_mulint_modern

cell = rnn_cell_modern.HighwayRNN_MulInt(num_units, num_layers)

#OR

cell = rnn_cell_modern.GRU_MulInt(num_units)

#OR

cell = rnn_cell_modern.BasicLSTM_MulInt(num_units)
```
