# Better together: Combining federated learning and travelling model boost performance in distributed learning setup

<div align="center">

</div>

<p align="center">
<img src="Fig1.png?raw=true">
</p>


Implementation for decentralized quality control that is published by the (coming soon): "[Better together: Combining federated learning and travelling model boost performance in distributed learning setup] (https://doi.org/).

Our code here is for the implementation of FedTM, a novel hybrid method that combines the FL and TM distributed learning approaches. FedTM training was divided into two phases: FL warmup and TM refinement. In the warmup phase, only sites with larger local datasets (s ≥40, 30, or 20 samples) participated in the training process, using a FL strategy (f = FedAvg [assumed equal contribution from all sites during the aggregation step] or f = FedProx [introduced an L2 regularization term to limit the divergence of local models from the global model]) for a predefined number of rounds (R = 2, 5, or 10). The goal of the warmup phase was to provide a strong initialization for the TM phase, which included all sites regardless of the local dataset size. The TM phase ran for a fixed number of cycles (C = 28, 25, or 20), with the exact number depending on the number of rounds completed during the warmup phase to ensure a combined total of 30 cycles across both phases.
* s, f, r, and c are all customizable parameters. See an example of how to train below.

If you find our framework, code, or paper useful to your research, please cite us!
```

@article{
}

```
```
Souza, R., 
```

### Abstract 

### File organization
1. **datagenerator_pd**: is the data loader for training and testing sets.
2. **fed_avg**: is the core code of FedAvg strategy.
3. **fed_prox**: is the core code of FedProx strategy.
4. **inference_pd_distributed**: has a script that generates the metrics (accuracy, sensitivity, specificity, AUROC for the overall dataset) for the models per cycle.
5. **main**: is the core of the hybrid implementation and this is the script you should run for training.
6. **sfcn**: contains the definition of the model architecture used for disease classification.

### Running this code

## All scripts have parameters that need to be called with descriptions in the argument parser. An example of how to call all of them:

#### FedTM with 2 warmup (-wp 2) rounds using FedAVG (-s 1), where sites with local dataset >= 40 (-split 40) train the model for 10 epochs (-epochs_f 10), then 28 cycles (-cycles 28) of TM is performed where all sites train the model for 1 epoch (-epochs_t 1). The final model name is best_model (-out). If the strategy was 2, it would implement FedProx with mu of 0.001. Mu is a parameter and needs to be passed, but for strategy 1, it will not be used.

```
python main.py -fn_train ./data/training.csv -cycles 28 -epochs_f 10 -epochs_t 1 -batch_size 5 -split 40 -wp 2 -mu 0.001 -s 1 -out best_model

```
#### For the inference, you can change the loop indices to determine the range of models you want to evaluate.


```
python inference_pd_distributed.py -fn ./test_set.csv -model ./best_model -o filename_to_save
```



## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.


## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=decentralized_quality_control).
# fedetated-travelling
