using System;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                var nn = new NeuralNetwork(28 * 28, 20, 10, 0.1);
                nn.InitNetwork();
                #region Learning

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
                    if ((di + 1) % 1000 == 0)
                    {
                        Console.WriteLine("Training epoch: " + (di + 1));
                        Console.WriteLine("Current success rate: " + Math.Round((success / (double)(di + 1) * 100.00), 2));
                    }
                }
                Console.WriteLine("training success rate: " + success / 60000.00 * 100.00);
                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                nn.SaveNetwork("test.dat");
                nn.LoadNetwork("test.dat");
                #endregion

                #region Testing

                ifsLabels = new FileStream("t10k-labels.idx1-ubyte", FileMode.Open);//поток для чтения лейблов 
                ifsImages = new FileStream("t10k-images.idx3-ubyte", FileMode.Open); // test images 
                brLabels = new BinaryReader(ifsLabels);
                brImages = new BinaryReader(ifsImages);

                magic1 = brImages.ReadInt32(); // магическое число 
                numImages = brImages.ReadInt32(); //количество изображений 
                numRows = brImages.ReadInt32(); //количество строк в изображении 
                numCols = brImages.ReadInt32(); //количество столбцов изображения 

                magic2 = brLabels.ReadInt32(); //магическое число 
                numLabels = brLabels.ReadInt32(); //количество лейблов

                success = 0;
                for (int di = 0; di < 10000; ++di)
                {
                    byte lbl = brLabels.ReadByte(); //текущее значение лейбла 
                    for (int i = 0; i < 28 * 28; ++i)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i] = b; //считываем байт изображения в массив 
                    }
                    if (nn.TestNetwork(PixelsToVector(pixels)) == lbl)
                    {
                        success++;
                    }
                    if ((di + 1) % 1000 == 0)
                    {
                        Console.WriteLine("Testing count: " + (di + 1));
                        Console.WriteLine("Current success rate: " + Math.Round((success / (double)(di + 1) * 100.00), 2));
                    }
                }
                Console.WriteLine("Testing success rate: " + success / 10000.00 * 100.00);
                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                #endregion
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }
        private static double[] PixelsToVector(byte[] pixels)
        {
            double[] vector = new double[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                vector[i] = pixels[i] > 0 ? 1 : 0;
            }
            return vector;
        }
    }
}
