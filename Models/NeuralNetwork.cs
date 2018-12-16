using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private const double LEARNING_RATE = 0.05;
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
            Layers = new List<Layer>();

            Layer hiddLayer = new Layer(LayerTypes.HIDDEN);
            for (int a = 0; a < HidLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                neuron.Output = 0.00;
                neuron.Inputs = new double[InputVectorSize];
                neuron.Weights = new double[InputVectorSize];
                for (int b = 0; b < InputVectorSize; b++)
                {
                    neuron.Inputs[b] = 0;
                    neuron.Weights[b] = DoubleFromRange(-1.00, 1.00) * 0.1;
                }
                hiddLayer.Neurons.Add(neuron);
            }
            Layers.Add(hiddLayer);

            Layer outLayer = new Layer(LayerTypes.OUTPUT);
            for (int a = 0; a < OutLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                neuron.Output = 0.00;
                neuron.Inputs = new double[HidLayerSize];
                neuron.Weights = new double[HidLayerSize];
                for (int b = 0; b < HidLayerSize; b++)
                {
                    neuron.Inputs[b] = 0.00;
                    neuron.Weights[b] = DoubleFromRange(-1.00, 1.00) * 0.1;
                }
                outLayer.Neurons.Add(neuron);
            }
            Layers.Add(outLayer);
        }

        public bool TrainNetwork(double[] data, int lbl)
        {
            double[] inputVector = GetInputVector(lbl);
            ForwardPropagate(data);
            BackPropagate(lbl);
            return true;
        }

        private void BackPropagate(int target)
        {
            Layer oLayer = Layers[1];
            for (int i = 0; i < oLayer.Neurons.Count; i++)
            {
                double output = oLayer.Neurons[i].Output;
                int targetOutput = i == target ? 1 : 0;
                double error = output - targetOutput;
                double weightsDelta = error * Sigmoid(output) * (1 - Sigmoid(output));
                for (int j = 0; j < oLayer.Neurons[i].Weights.Length; j++)
                {
                    oLayer.Neurons[i].Weights[j] = oLayer.Neurons[i].Weights[j] - output * weightsDelta * LEARNING_RATE;
                }
            }
        }

        private void ForwardPropagate(double[] data)
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].LayerType == LayerTypes.HIDDEN)
                {
                    TrainHiddenLayer(Layers[i], data);
                }
                else
                {
                    TrainOutputLayer(Layers[i]);
                }
            }
        }

        private void TrainHiddenLayer(Layer layer, double[] data)
        {
            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                TrainNeuron(layer.Neurons[i], data);
            }
        }

        private void TrainNeuron(Neuron neuron, double[] data)
        {
            neuron.Output = 0;
            for (int i = 0; i < neuron.Inputs.Length; i++)
            {
                neuron.Inputs[i] = data[i];
                neuron.Output += neuron.Inputs[i] * neuron.Weights[i];
            }
            neuron.Output = Sigmoid(neuron.Output);
        }

        private void TrainOutputLayer(Layer layer)
        {
            double[] input = GetOutputVector(Layers[0]);
            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                TrainNeuron(layer.Neurons[i], input);
            }
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

        private double[] GetOutputVector(Layer l)
        {
            double[] vector = new double[l.Neurons.Count];
            for (int i = 0; i < l.Neurons.Count; i++)
            {
                vector[i] = l.Neurons[i].Output;
            }
            return vector;
        }

        private double DoubleFromRange(double minimum, double maximum)
        {
            Random rnd = new Random();
            return rnd.NextDouble() * (maximum - minimum) + minimum;
        }
        private double Sigmoid(double value) => 1.00 / (1.00 + Math.Exp(-value));
    }
}