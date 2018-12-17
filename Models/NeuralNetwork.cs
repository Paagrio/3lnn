using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
  public class NeuralNetwork
  {
    private const double LEARNING_RATE = 0.2;
    private const double RAND_MAX = Int32.MaxValue;
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
          neuron.Weights[b] = (0.5 * (Rand() / RAND_MAX));
          if (b % 2 == 0)
          {
            neuron.Weights[b] = -neuron.Weights[b];
          }

        }
        neuron.Bias = (0.5 * (Rand() / RAND_MAX));
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
          neuron.Weights[b] = (0.5 * (Rand() / RAND_MAX));
          if (b % 2 == 0)
          {
            neuron.Weights[b] = -neuron.Weights[b];
          }
        }
        neuron.Bias = (0.5 * (Rand() / RAND_MAX));
        outLayer.Neurons.Add(neuron);
      }
      Layers.Add(outLayer);
    }

    public bool TrainNetwork(double[] data, int lbl)
    {
      double[] inputVector = GetInputVector(lbl);
      ForwardPropagate(data);
      BackPropagate(lbl);
      int classificated = GetClassification();
      return classificated == lbl;
    }

    public bool TestNetwork(double[] data, int lbl)
    {
      double[] inputVector = GetInputVector(lbl);
      ForwardPropagate(data);
      int classificated = GetClassification();
      return classificated == lbl;
    }

    private void BackPropagate(int target)
    {
      Layer oLayer = Layers.Where(x => x.LayerType == LayerTypes.OUTPUT).First();
      for (int i = 0; i < oLayer.Neurons.Count; i++)
      {
        double output = oLayer.Neurons[i].Output;
        int targetOutput = i == target ? 1 : 0;
        double error = targetOutput - output;
        double weightsDelta = error * Sigmoiddx(output);
        UpdateWeights(oLayer, i, weightsDelta);
      }
      oLayer = Layers.Where(x => x.LayerType == LayerTypes.HIDDEN).First();
      for (int i = 0; i < oLayer.Neurons.Count; i++)
      {
        double output = oLayer.Neurons[i].Output;
        int targetOutput = i == target ? 1 : 0;
        double error = targetOutput - output;
        double weightsDelta = error * Sigmoiddx(output);
        UpdateWeights(oLayer, i, weightsDelta);
      }
    }

    private int GetClassification()
    {
      Layer oLayer = Layers.Where(x => x.LayerType == LayerTypes.OUTPUT).First();
      double max = 0.00;
      int maxIndex = 0;
      for (int i = 0; i < oLayer.Neurons.Count; i++)
      {
        if (oLayer.Neurons[i].Output > max)
        {
          max = oLayer.Neurons[i].Output;
          maxIndex = i;
        }
      }
      return maxIndex;
    }

    private void UpdateWeights(Layer layer, int nodeId, double error)
    {
      if (layer.LayerType == LayerTypes.OUTPUT)
      {
        for (int j = 0; j < layer.Neurons[nodeId].Weights.Length; j++)
        {
          layer.Neurons[nodeId].Weights[j] += LEARNING_RATE * error * layer.Neurons[nodeId].Inputs[j];
        }
        layer.Neurons[nodeId].Bias += LEARNING_RATE * 1 * error;
      }
    }

    private void ForwardPropagate(double[] data)
    {
      TrainHiddenLayer(Layers[0], data);
      TrainOutputLayer(Layers[1]);
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
      neuron.Output = neuron.Bias;
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

    private double Rand()
    {
      Random rnd = new Random();
      return rnd.Next();
    }

    private double Sigmoid(double value) => 1.00 / (1.00 + Math.Exp(-value));
    private double Sigmoiddx(double value) => value * (1 - value);
  }
}
