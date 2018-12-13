using System;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(784, 10, 10);
            nn.InitNetwork();

            FileStream ifsLabels = new FileStream("train-labels.idx1-ubyte", FileMode.Open);//поток для чтения лейблов 
            FileStream ifsImages = new FileStream("train-images.idx3-ubyte", FileMode.Open); // test images 

            BinaryReader brLabels = new BinaryReader(ifsLabels);
            BinaryReader brImages = new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // магическое число 
            int numImages = brImages.ReadInt32(); //количество изображений 
            int numRows = brImages.ReadInt32(); //количество строк в изображении 
            int numCols = brImages.ReadInt32(); //количество столбцов изображения 
            int magic2 = brLabels.ReadInt32(); //магическое число 
            int numLabels = brLabels.ReadInt32(); //количество лейблов 

            byte[] pixels = new byte[28 * 28]; //инициализация массива для хранения изображения 28x28 
            int success = 0;

            for (int di = 0; di < 60000; ++di)
            {
                byte lbl = brLabels.ReadByte(); //текущее значение лейбла 
                for (int i = 0; i < 28 * 28; ++i)
                {
                    byte b = brImages.ReadByte();
                    pixels[i] = b; //считываем байт изображения в массив 
                }
                if (nn.TrainNetwork(PixelsToVector(pixels), lbl))
                {
                    success++;
                }
            }
        }

        private static double[] PixelsToVector(byte[] pixels)
        {
            double[] vector = new double[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                vector[i] = pixels[i] > 0 ? 1 : 0;
            }
        }
    }
}
