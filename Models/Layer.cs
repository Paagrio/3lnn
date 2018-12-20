using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    [Serializable]
    sealed class Layer
    {
        public LayerType LayerType { get; set; }
        public List<Neuron> Neurons { get; set; }

        public Layer(LayerType layerType)
        {
            LayerType = layerType;
            Neurons = new List<Neuron>();
        }
    }
}