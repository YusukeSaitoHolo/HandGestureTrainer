using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

#if USE_WINML_NUGET
using Microsoft.AI.MachineLearning;
#else
using Windows.AI.MachineLearning;
#endif
using Windows.Storage.Streams;
using Windows.Storage;
using System.Threading.Tasks;
using System.Diagnostics;

using Newtonsoft.Json.Linq;

namespace HandGesture
{
    /// <summary>
    /// それ自体で使用できる空白ページまたはフレーム内に移動できる空白ページ。
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private readonly int frameCount = 30;
        private readonly int columns = 4;
        private handGestureModel handGestureModelGen;

        public MainPage()
        {
            this.InitializeComponent();

            LoadModelAsync();
        }

        private async Task LoadModelAsync()
        {
            var filePicker = new Windows.Storage.Pickers.FileOpenPicker();
            filePicker.FileTypeFilter.Add(".onnx");

            // 単一ファイルの選択(モデル選択)
            StorageFile file = await filePicker.PickSingleFileAsync();
            handGestureModelGen = await handGestureModel.CreateFromStreamAsync( file as IRandomAccessStreamReference);
            var dlg = new Windows.UI.Popups.MessageDialog("model Load");
            await dlg.ShowAsync();

        }
        private async void recognizeButton_Click(object sender, RoutedEventArgs e)
        {
            LoadSampleBloom();
        }


        private async Task LoadSampleBloom()
        {
            var filePicker = new Windows.Storage.Pickers.FileOpenPicker();
            filePicker.FileTypeFilter.Add(".json");

            // 単一ファイルの選択(サンプルデータ選択)
            var file = await filePicker.PickSingleFileAsync();
            if (file == null)
            {
                return;
            }
            var jsonData = await FileIO.ReadTextAsync(file);
            JObject jsonObj = JObject.Parse(jsonData);

            //Jsonデータから特徴ベクトルを生成する
            var feature = GenerateFloatArrayFromJsonData(jsonObj);

            handGestureInput input = new handGestureInput();
            input.Input120 = TensorFloat.CreateFromArray(new long[] { 1 , columns * frameCount }, feature);

            //Evaluate the model
            var handGestureOutput = await handGestureModelGen.EvaluateAsync(input);
            IList<string> stringList = handGestureOutput.label.GetAsVectorView().ToList();
            Debug.WriteLine($"判定された動作ラベル:{stringList[0]}");
        }

        /// <summary>
        /// Jsonデータからモデルの入力に使用する特徴データへ変換する関数
        /// </summary>
        /// <param name="jsonData">ロードするNewtonsoft.Jsonデータ</param>
        /// <returns> モデルの入力に使用する特徴データ</returns>
        private float[] GenerateFloatArrayFromJsonData(JObject jsonData)
        {
            var gestureData = jsonData["gestureData"];

            float[] feature = new float[columns * frameCount];
            for (int i = 0; i < frameCount; i++)
            {
                feature[i * columns] = gestureData[i]["triangleArea"].ToObject<float>();
                feature[i * columns + 1] = gestureData[i]["triangleNormal"]["x"].ToObject<float>();
                feature[i * columns + 2] = gestureData[i]["triangleNormal"]["y"].ToObject<float>();
                feature[i * columns + 3] = gestureData[i]["triangleNormal"]["z"].ToObject<float>();
            }

            return feature;
        }
    }



    public sealed class handGestureModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;
        public static async Task<handGestureModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
        {
            handGestureModel learningModel = new handGestureModel();
            learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);
            return learningModel;
        }
        public async Task<handGestureOutput> EvaluateAsync(handGestureInput input)
        {
            binding.Bind("input", input.Input120);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new handGestureOutput();
            output.label = result.Outputs["label"] as TensorString;
            return output;
        }
    }

    public sealed class handGestureInput
    {
        public TensorFloat Input120;
    }

    public sealed class handGestureOutput
    {
        public TensorString label;
        public IList<IDictionary<string, float>> probabilities;
    }
}
