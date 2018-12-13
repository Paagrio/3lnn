using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {

        public int InputVectorSize { get; set; }
        public int HidLayerSize { get; set; }
        public int OutLayerSize { get; set; }
        public List<Layer> Layers { get; set; }

        public NeuralNetwork(int inputVectorSize, int hidLayerSize, int outLayerSize)
        {
            InputVectorSize = inputVectorSize;
            HidLayerSize = hidLayerSize;
            OutLayerSize = outLayerSize;
        }

        public void InitNetwork()
        {
            Random rnd = new Random();
            Layers = new List<Layer>();

            Layer hiddLayer = new Layer(LayerTypes.HIDDEN);
            for (int a = 0; a < HidLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                for (int b = 0; b < InputVectorSize; b++)
                {
                    neuron.Input.Add(0.00);
                    neuron.Weights.Add(rnd.NextDouble());
                }
                hiddLayer.Neurons.Add(neuron);
            }
            Layers.Add(hiddLayer);

            Layer outLayer = new Layer(LayerTypes.OUTPUT);
            for (int a = 0; a < OutLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                for (int b = 0; b < HidLayerSize; b++)
                {
                    neuron.Input.Add(0.00);
                    neuron.Weights.Add(rnd.NextDouble());
                }
                outLayer.Neurons.Add(neuron);
            }
            Layers.Add(outLayer);
        }

        public bool TrainNetwork(double[] data, int lbl)
        {
            double[] inputVector = GetInputVector(lbl);
            for (int i = 0; i < Layers.Count; i++)
            {
                TrainLayer(Layers[i], data);
            }
        }

        private void TrainLayer(Layer layer, double[] input)
        {
            if (layer.LayerType == LayerTypes.HIDDEN)
            {
                for (int i = 0; i < layer.Neurons.Count; i++)
                {
                    TrainNeuron(layer.Neurons[i], input);
                }
            }
        }
        private void TrainNeuron(Neuron neuron, double[] input)
        {
            Console.WriteLine()
        }

        private double[] GetInputVector(int lbl)
        {
            double[] input = new double[10];
            for (int i = 0; i < 10; i++)
            {
                input[i] = lbl == i ? 1 : 0;
            }
            return input;
        }
    }
}