# TensorFlow With The Latest Papers Implemented

###Implementation of RNN and NLP Related Neural Network Papers

Currently Implemented Papers:

* Highway Networks
* Recurrent Highway Networks
* Multiplicative Integration Within RNNs
* GRU Mutants

More Papers to come as they are published. If you have any requests, please use the issues section. 

###Contact Information:

skype: lea vesbr eat he (eliminate all spaces)

email: sh a hn s [at ] m ail.u c.ed u (eliminate all spaces)


### Testing These New Papers

If you would like to test these new features, you can:

`python ptb_word_lm.py`

Simply modify the `rnn_cell` variable under the `PTBModel`

Please run with Tensorflow 0.8 or higher


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

cell = rnn_cell_modern.HighwayRNNCell(num_units, num_highway_layers = 3)
```


### Multiplicative Integration Within RNNs

https://arxiv.org/abs/1606.06630

Allows faster convergence within RNNs by utilizing the combination of two separate weight matrices in a multiplicative setting

```python

import rnn_cell_mulint_modern

cell = rnn_cell_mulint_modern.BasicRNNCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.GRUCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.BasicLSTMCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.HighwayRNNCell_MulInt(num_units, num_highway_layers = 3)
```


### GRU Mutants

http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

Mutants of GRU that may work better in different scenarios:

```python
cell = rnn_cell_modern.JZS1Cell(num_units)
#Or
cell = rnn_cell_modern.JZS2Cell(num_units)
#Or
cell = rnn_cell_modern.JZS3Cell(num_units)
```