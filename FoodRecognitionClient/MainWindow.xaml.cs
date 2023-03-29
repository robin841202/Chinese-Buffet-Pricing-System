using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Kinect;
using Newtonsoft.Json;
using System.Net;
using System.Text;
using System.Linq;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace FoodRecognitionClient
{

    public partial class MainWindow : Window
    {
        //開始辨識及開始校正的flags
        bool startCapture_color = false;
        bool startCapture_depth = false;
        bool startCalibrate_color = false;
        bool startCalibrate_depth = false;

        ushort[] depth_array; //開始辨識之深度資料
        ushort[] emptyplate_depth; //空盤深度資料(用來校正空盤的)
        
        //預設的各菜區空盤平均深度值
        int empty_average1 = 850;
        int empty_average2 = 850;
        int empty_average3 = 850;
        int empty_average4 = 853;
        
        private KinectSensor kSensor = null;
        private MultiSourceFrameReader mReader = null;
        private CoordinateMapper coordinateMapper = null; //用來做影像座標對映

        bool auto_detect = false; // 自動偵測模式flag
        bool first_frame_detect = false; //自動偵測模式啟動後，用來判斷存下第一張frame的flag
        bool isDetected = false; //是否有偵測到餐盤之狀態
        int frames_count = 0;
        const int countdown_sec = 5;
        int time = countdown_sec;
        private DispatcherTimer BeginAutoDetect_Timer, BeginRecognize_Timer;
        ushort[] previous_depthscaled; //開啟自動偵測模式後存入第一張frame的scaled後深度影像(空底盤)
        ushort[] current_depthscaled; //當下scaled成0~255的深度影像

        string ip = "http://localhost:5000";

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            this.kSensor = KinectSensor.GetDefault(); //Kinect v2 Sensor獲取(Sensor初始化)

            if (kSensor != null)
            {
                this.kSensor.Open(); //開啟Sensor
                coordinateMapper = kSensor.CoordinateMapper;
            }
            mReader = kSensor.OpenMultiSourceFrameReader(FrameSourceTypes.Color | FrameSourceTypes.Depth); //開啟多源串流讀取(彩色及深度)
            mReader.MultiSourceFrameArrived += Reader_MultiSourceFrameArrived;

        }

        private void Reader_MultiSourceFrameArrived(object sender, MultiSourceFrameArrivedEventArgs e)
        {
            // Get a reference to the multi-frame
            var reference = e.FrameReference.AcquireFrame();


            // 深度影像串流 每frame都會執行一次
            using (var frame = reference.DepthFrameReference.AcquireFrame())
            {
                if (frame != null)
                {
                    //開始辨識時
                    if (startCapture_depth)
                    {
                        int width = frame.FrameDescription.Width;
                        int height = frame.FrameDescription.Height;
                        ushort[] depthData = new ushort[width * height];
                        frame.CopyFrameDataToArray(depthData); //將Frame的資料轉為array存入depthData
                        depth_array = depthData;
                        startCapture_depth = false; //將flag設回false 避免連續執行
                    }

                    //開始校正時
                    if (startCalibrate_depth)
                    {
                        int width = frame.FrameDescription.Width;
                        int height = frame.FrameDescription.Height;
                        ushort[] depthData = new ushort[width * height];
                        frame.CopyFrameDataToArray(depthData);//將Frame的資料轉為array存入depthData
                        emptyplate_depth = depthData;
                        startCalibrate_depth = false; //將flag設回false 避免連續執行
                    }

                    //開始自動偵測模式時
                    if (auto_detect)
                    {
                        if (first_frame_detect)
                        {
                            (_, previous_depthscaled) = ToBitmap(frame); //存下啟動自動偵測模式後的第一張frame
                            first_frame_detect = false; //將flag設回false 避免連續執行
                        }
                        frames_count++;  //計算frames
                        if (frames_count == 60)  //每60frames偵測一次
                        {
                            (_, current_depthscaled) = ToBitmap(frame); //當下scaled成0~255後的深度影像
                            //進行背景相減
                            int[] background_subtraction_result = new int[previous_depthscaled.Length]; ;
                            for (int index = 0; index < previous_depthscaled.Length; index++)
                            {
                                background_subtraction_result[index] = Math.Abs(current_depthscaled[index] - previous_depthscaled[index]);
                            }

                            //將背景相減後的結果成像
                            int colorIndex = 0;
                            byte[] background_subtraction_result_pixelData = new byte[background_subtraction_result.Length * (PixelFormats.Bgr32.BitsPerPixel / 8)];
                            int cropped_width = 256, cropped_height = 160;
                            for (int depthIndex = 0; depthIndex < background_subtraction_result.Length; ++depthIndex)
                            {
                                int depth = background_subtraction_result[depthIndex];
                                byte intensity = (byte)background_subtraction_result[depthIndex];
                                background_subtraction_result_pixelData[colorIndex++] = intensity; // Blue
                                background_subtraction_result_pixelData[colorIndex++] = intensity; // Green
                                background_subtraction_result_pixelData[colorIndex++] = intensity; // Red
                                background_subtraction_result_pixelData[colorIndex++] = 255; //Alpha
                            }
                            int stride = cropped_width * PixelFormats.Bgr32.BitsPerPixel / 8;
                            background_subtraction_show.Source = BitmapSource.Create(cropped_width, cropped_height, 96, 96, PixelFormats.Bgr32, null, background_subtraction_result_pixelData, stride);

                            //狀態為未偵測到且背景相減平均>8後開始辨識，並把狀態轉為已偵測到，避免同個餐盤多次辨識
                            if (!isDetected && background_subtraction_result.Average() > 7)
                            {
                                //啟用Timer倒數 倒數完便開始辨識
                                BeginRecognize_Timer = new DispatcherTimer();
                                BeginRecognize_Timer.Interval = new TimeSpan(0, 0, 1);
                                BeginRecognize_Timer.Tick += Timer_BeginRecognizeCountDown;
                                BeginRecognize_Timer.Start();
                                isDetected = true;
                            }
                            //底盤變回空盤後，再允許進行下一盤餐盤判斷
                            if (isDetected && background_subtraction_result.Average() < 6)
                            {
                                isDetected = false;
                                detecting_txt.Text = "偵測中...";
                            }
                            Console.WriteLine(background_subtraction_result.Average());
                            Console.WriteLine("frames_count");
                            frames_count = 0;
                        }
                    }
                }
            }


            // 彩色影像串流 每frame都會執行一次
            using (var frame = reference.ColorFrameReference.AcquireFrame())
            {
                if (frame != null)
                {
                    camera.Source = ToBitmap(frame); //將彩色frame show到介面上

                    //開始辨識時
                    if (startCapture_color)
                    {
                        canvas.Children.Clear();
                        detecting_txt.Text = "辨識中...";
                        var resultDict = StartRecognize((BitmapSource)camera.Source);
                        //將回傳結果show到介面上
                        if (resultDict.ContainsKey("region1_sum")) { region1_sum_txt.Text = resultDict["region1_sum"] + "cm\xB3"; }
                        if (resultDict.ContainsKey("region2_sum")) { region2_sum_txt.Text = resultDict["region2_sum"] + "cm\xB3"; }
                        if (resultDict.ContainsKey("region3_sum")) { region3_sum_txt.Text = resultDict["region3_sum"] + "cm\xB3"; }
                        if (resultDict.ContainsKey("region4_sum")) { region4_sum_txt.Text = resultDict["region4_sum"] + "cm\xB3"; }
                        if (resultDict.ContainsKey("result1")) {region1_cate_txt.Text = resultDict["result1"]; }
                        if (resultDict.ContainsKey("result2")) { region2_cate_txt.Text = resultDict["result2"]; }
                        if (resultDict.ContainsKey("result3")) { region3_cate_txt.Text = resultDict["result3"]; }
                        if (resultDict.ContainsKey("result4")) { region4_cate_txt.Text = resultDict["result4"]; }
                        if (resultDict.ContainsKey("price_result1")) { region1_price_txt.Text = "$" + resultDict["price_result1"]; }
                        if (resultDict.ContainsKey("price_result2")) { region2_price_txt.Text = "$" + resultDict["price_result2"]; }
                        if (resultDict.ContainsKey("price_result3")) { region3_price_txt.Text = "$" + resultDict["price_result3"]; }
                        if (resultDict.ContainsKey("price_result4")) { region4_price_txt.Text = "$" + resultDict["price_result4"]; }
                        if (resultDict.ContainsKey("price_sum")) {price_sum_txt.Text = "$" + resultDict["price_sum"]; }
                        if (resultDict.ContainsKey("calories1")) { region1_calories_txt.Text = resultDict["calories1"] + "kcal"; }
                        if (resultDict.ContainsKey("calories2")) { region2_calories_txt.Text = resultDict["calories2"] + "kcal"; }
                        if (resultDict.ContainsKey("calories3")) { region3_calories_txt.Text = resultDict["calories3"] + "kcal"; }
                        if (resultDict.ContainsKey("calories4")) { region4_calories_txt.Text = resultDict["calories4"] + "kcal"; }
                        if (resultDict.ContainsKey("calories_sum")) {calories_sum_txt.Text = resultDict["calories_sum"] + "kcal"; }
                        if (resultDict.ContainsKey("price_per_portion1")) { region1_price_per_portion_txt.Text = "$" + resultDict["price_per_portion1"] + "/份 "; }
                        if (resultDict.ContainsKey("price_per_portion2")) { region2_price_per_portion_txt.Text = "$" + resultDict["price_per_portion2"] + "/份 "; }
                        if (resultDict.ContainsKey("price_per_portion3")) { region3_price_per_portion_txt.Text = "$" + resultDict["price_per_portion3"] + "/份 "; }
                        if (resultDict.ContainsKey("price_per_portion4")) { region4_price_per_portion_txt.Text = "$" + resultDict["price_per_portion4"] + "/份 "; }
                        if (resultDict.ContainsKey("volume_per_portion1")) { region1_volume_per_portion_txt.Text = resultDict["volume_per_portion1"] + "cm\xB3/份"; }
                        if (resultDict.ContainsKey("volume_per_portion2")) { region2_volume_per_portion_txt.Text = resultDict["volume_per_portion2"] + "cm\xB3/份"; }
                        if (resultDict.ContainsKey("volume_per_portion3")) { region3_volume_per_portion_txt.Text = resultDict["volume_per_portion3"] + "cm\xB3/份"; }
                        if (resultDict.ContainsKey("volume_per_portion4")) { region4_volume_per_portion_txt.Text = resultDict["volume_per_portion4"] + "cm\xB3/份"; }

                        startCapture_color = false; //將flag設回false 避免連續執行
                        capture_label.Visibility = Visibility.Collapsed;
                        detecting_txt.Text = "辨識完成";
                        //MessageBox.Show("辨識完成!", "提醒");
                    }

                    //開始校正時
                    if (startCalibrate_color)
                    {
                        StartCalibrate((BitmapSource)camera.Source);
                        startCalibrate_color = false; //將flag設回false 避免連續執行
                        capture_label.Visibility = Visibility.Collapsed;
                        MessageBox.Show("校正完成!", "提醒");
                    }
                }
            }
        }

        //將彩色frame轉為Bitmap
        private ImageSource ToBitmap(ColorFrame frame)
        {
            int width = frame.FrameDescription.Width;
            int height = frame.FrameDescription.Height;
            var format = PixelFormats.Bgr32;

            byte[] pixels = new byte[width * height * (format.BitsPerPixel / 8)];

            if (frame.RawColorImageFormat == ColorImageFormat.Bgra)
            {
                frame.CopyRawFrameDataToArray(pixels);
            }
            else
            {
                frame.CopyConvertedFrameDataToArray(pixels, ColorImageFormat.Bgra);
            }

            int stride = width * format.BitsPerPixel / 8;

            return BitmapSource.Create(width, height, 96, 96, format, null, pixels, stride);
        }

        private (ImageSource, ushort[]) ToBitmap(DepthFrame frame)
        {
            int width = frame.FrameDescription.Width;
            int height = frame.FrameDescription.Height;
            var format = PixelFormats.Bgr32;
            ushort[] depthData = new ushort[width * height];
            byte[] pixelData = new byte[width * height * (format.BitsPerPixel / 8)];

            frame.CopyFrameDataToArray(depthData);

            // crop深度影像，從左上座標為(150,90) 長256 寬160
            int cropped_xCoord = 150, cropped_yCoord = 90, cropped_width = 256, cropped_height = 160;
            ushort[] cropped_depthData = new ushort[cropped_width * cropped_height];
            byte[] cropped_pixelData = new byte[cropped_width * cropped_height * (format.BitsPerPixel / 8)];
            for (int i = 0; i < cropped_height; i++)
            {
                for (int j = 0; j < cropped_width; j++)
                {
                    cropped_depthData[i * cropped_width + j] = depthData[(i + cropped_yCoord) * width + j + cropped_xCoord];
                }
            }

            ushort minDepth = 800;
            ushort maxDepth = 900;
            int colorIndex = 0;
            for (int depthIndex = 0; depthIndex < cropped_depthData.Length; ++depthIndex)
            {
                ushort depth = cropped_depthData[depthIndex];
                cropped_depthData[depthIndex] = (ushort)(depth >= minDepth && depth <= maxDepth ? ((depth - minDepth) * 256 / (maxDepth - minDepth)) : 0); //將cropped_depthData 篩選深度值介於800~900的且scale成0~255
                byte intensity = (byte)(depth >= minDepth && depth <= maxDepth ? ((depth - minDepth) * 256 / (maxDepth - minDepth)) : 0);
                cropped_pixelData[colorIndex++] = intensity; // Blue
                cropped_pixelData[colorIndex++] = intensity; // Green
                cropped_pixelData[colorIndex++] = intensity; // Red
                cropped_pixelData[colorIndex++] = 255; //Alpha

            }

            int stride = cropped_width * format.BitsPerPixel / 8;   //stride is the "scanline" of a bitmap, means "one row in bytes"
            return (BitmapSource.Create(cropped_width, cropped_height, 96, 96, format, null, cropped_pixelData, stride), cropped_depthData);
        }


        private string ToBase64(BitmapSource image, string format)
        {
            return Convert.ToBase64String(Encode(image, format));
        }
        private byte[] Encode(BitmapSource bitmapImage, string format)
        {
            byte[] data = null;
            BitmapEncoder encoder = null;
            switch (format.ToUpper())
            {
                case "PNG":
                    encoder = new PngBitmapEncoder();
                    break;
                case "GIF":
                    encoder = new GifBitmapEncoder();
                    break;
                case "BMP":
                    encoder = new BmpBitmapEncoder();
                    break;
                case "JPG":
                    encoder = new JpegBitmapEncoder();
                    break;
            }
            if (encoder != null)
            {
                encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                using (var ms = new MemoryStream())
                {
                    encoder.Save(ms);
                    ms.Seek(0, SeekOrigin.Begin);
                    data = ms.ToArray();
                }
            }

            return data;
        }

        private string SendToApi(string url, string json)
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = "POST";
            request.ContentType = "application/json";
            byte[] byteArray = Encoding.UTF8.GetBytes(json);//要發送的字串轉為byte[]
            using (Stream reqStream = request.GetRequestStream())
            {
                reqStream.Write(byteArray, 0, byteArray.Length);
            }
            //發出Request
            string responseStr = "";
            using (WebResponse response = request.GetResponse())
            {
                using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
                {
                    responseStr = reader.ReadToEnd();
                }
            }
            return responseStr;
        }

        private Dictionary<string,string> StartRecognize(BitmapSource bSource)
        {
            Dictionary<string, string> resultDict = new Dictionary<string, string>();
            if (bSource != null)
            {
                var base64Img = ToBase64(bSource, "PNG"); //將彩色影像轉為base64
                FoodImageData foodData = new FoodImageData
                {
                    FoodPic = base64Img
                };

                //轉成JSON格式
                string json = JsonConvert.SerializeObject(foodData);
                string url = ip + "/api/recognize";
                var responseStr = SendToApi(url, json); //request給WebApi
                
                //將回傳的Json物件反序列化 並拿出所有values
                var responseData = JsonConvert.DeserializeObject<ResponseData>(responseStr);
                var area1list = responseData.ROI["area1"];
                var area2list = responseData.ROI["area2"];
                var area3list = responseData.ROI["area3"];
                var area4list = responseData.ROI["area4"];
                var result1 = responseData.FoodResult["result1"];
                var result2 = responseData.FoodResult["result2"];
                var result3 = responseData.FoodResult["result3"];
                var result4 = responseData.FoodResult["result4"];
                var price1 = responseData.FoodPrices["price1"];
                var price2 = responseData.FoodPrices["price2"];
                var price3 = responseData.FoodPrices["price3"];
                var price4 = responseData.FoodPrices["price4"];
                var calories1 = responseData.FoodCalories["calories1"];
                var calories2 = responseData.FoodCalories["calories2"];
                var calories3 = responseData.FoodCalories["calories3"];
                var calories4 = responseData.FoodCalories["calories4"];
                var volume_per_portion1 = responseData.FoodVolumePerPortion["volume1"];
                var volume_per_portion2 = responseData.FoodVolumePerPortion["volume2"];
                var volume_per_portion3 = responseData.FoodVolumePerPortion["volume3"];
                var volume_per_portion4 = responseData.FoodVolumePerPortion["volume4"];


                Polygon p1 = new Polygon();
                Polygon p2 = new Polygon();
                Polygon p3 = new Polygon();
                Polygon p4 = new Polygon();
                p1.Stroke = System.Windows.Media.Brushes.Red;
                p2.Stroke = System.Windows.Media.Brushes.Green;
                p3.Stroke = System.Windows.Media.Brushes.Blue;
                p4.Stroke = System.Windows.Media.Brushes.Yellow;
                p1.StrokeThickness = 3;
                p2.StrokeThickness = 3;
                p3.StrokeThickness = 3;
                p4.StrokeThickness = 3;

                PointCollection points1 = new PointCollection();
                PointCollection points2 = new PointCollection();
                PointCollection points3 = new PointCollection();
                PointCollection points4 = new PointCollection();

                System.Drawing.Point[] p1Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p2Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p3Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p4Array = new System.Drawing.Point[6];

                //將各菜區的彩色影像座標轉換為深度影像座標
                foreach (List<int> coordinate in area1list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, depth_array, coordinate[0], coordinate[1]);
                    var index = area1list.IndexOf(coordinate);
                    p1Array[index] = new System.Drawing.Point(correspondX, correspondY);
                    points1.Add(new System.Windows.Point(coordinate[0] / 1.5, coordinate[1] / 1.5));
                }
                foreach (List<int> coordinate in area2list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, depth_array, coordinate[0], coordinate[1]);
                    var index = area2list.IndexOf(coordinate);
                    p2Array[index] = new System.Drawing.Point(correspondX, correspondY);
                    points2.Add(new System.Windows.Point(coordinate[0] / 1.5, coordinate[1] / 1.5));
                }
                foreach (List<int> coordinate in area3list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, depth_array, coordinate[0], coordinate[1]);
                    var index = area3list.IndexOf(coordinate);
                    p3Array[index] = new System.Drawing.Point(correspondX, correspondY);
                    points3.Add(new System.Windows.Point(coordinate[0] / 1.5, coordinate[1] / 1.5));
                }
                foreach (List<int> coordinate in area4list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, depth_array, coordinate[0], coordinate[1]);
                    var index = area4list.IndexOf(coordinate);
                    p4Array[index] = new System.Drawing.Point(correspondX, correspondY);
                    points4.Add(new System.Windows.Point(coordinate[0]/1.5, coordinate[1] / 1.5));
                }

                p1.Points = points1;
                p2.Points = points2;
                p3.Points = points3;
                p4.Points = points4;
                //畫出菜區ROI
                canvas.Children.Add(p1);
                canvas.Children.Add(p2);
                canvas.Children.Add(p3);
                canvas.Children.Add(p4);

                //掃整張影像找出所有在各菜區多邊形內的點 並存下index到List內
                List<int> indices1 = new List<int>();
                List<int> indices2 = new List<int>();
                List<int> indices3 = new List<int>();
                List<int> indices4 = new List<int>();
                var width = 512;
                var height = 424;
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        if (PointIsInPolygon(p1Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices1.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p2Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices2.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p3Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices3.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p4Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices4.Add(index);
                            continue;
                        }
                    }
                }

                //將上述得到的indices取得它們的深度值存入List
                List<ushort> depthvalues1 = new List<ushort>();
                List<ushort> depthvalues2 = new List<ushort>();
                List<ushort> depthvalues3 = new List<ushort>();
                List<ushort> depthvalues4 = new List<ushort>();
                foreach (var index in indices1)
                {
                    depthvalues1.Add(depth_array[index]);
                }
                foreach (var index in indices2)
                {
                    depthvalues2.Add(depth_array[index]);
                }
                foreach (var index in indices3)
                {
                    depthvalues3.Add(depth_array[index]);
                }
                foreach (var index in indices4)
                {
                    depthvalues4.Add(depth_array[index]);
                }

                //將空盤平均深度 - 各菜區深度值 並將結果存入List
                List<int> subtractedDepth1 = new List<int>();
                List<int> subtractedDepth2 = new List<int>();
                List<int> subtractedDepth3 = new List<int>();
                List<int> subtractedDepth4 = new List<int>();
                int result_sum1 = 0; //菜區1初步預估體積
                int result_sum2 = 0; //菜區2初步預估體積
                int result_sum3 = 0; //菜區3初步預估體積
                int result_sum4 = 0; //菜區4初步預估體積
                var threshold = 2;
                for (var i = 0; i < depthvalues1.Count; i++)
                {
                    if (depthvalues1[i] > 0)
                    {
                        int result = empty_average1 - depthvalues1[i];
                        if (result > threshold)
                        {
                            subtractedDepth1.Add(result);
                            result_sum1 += result;
                        }
                    }
                }
                for (var i = 0; i < depthvalues2.Count; i++)
                {
                    if (depthvalues2[i] > 0)
                    {
                        int result = empty_average2 - depthvalues2[i];
                        if (result > threshold)
                        {
                            subtractedDepth2.Add(result);
                            result_sum2 += result;
                        }
                    }
                }
                for (var i = 0; i < depthvalues3.Count; i++)
                {
                    if (depthvalues3[i] > 0)
                    {
                        int result = empty_average3 - depthvalues3[i];
                        if (result > threshold)
                        {
                            subtractedDepth3.Add(result);
                            result_sum3 += result;
                        }
                    }
                }
                for (var i = 0; i < depthvalues4.Count; i++)
                {
                    if (depthvalues4[i] > 0)
                    {
                        int result = empty_average4 - depthvalues4[i];
                        if (result > threshold)
                        {
                            subtractedDepth4.Add(result);
                            result_sum4 += result;
                        }
                    }
                }


                var real_volume_scale = 1.2; //實際體積與初步估算體積的比例
                var determine_empty_threshold = 5; //小於此閾值便視為空盤 不乘上比例
                //將初步估算體積先轉為立方公分單位
                var volume1_cm3 = result_sum1 / 100.0; 
                var volume2_cm3 = result_sum2 / 100.0;
                var volume3_cm3 = result_sum3 / 100.0;
                var volume4_cm3 = result_sum4 / 100.0;
                //乘上比例
                if (volume1_cm3 > determine_empty_threshold) { volume1_cm3 = volume1_cm3 * real_volume_scale; }
                if (volume2_cm3 > determine_empty_threshold) { volume2_cm3 = volume2_cm3 * real_volume_scale; }
                if (volume3_cm3 > determine_empty_threshold) { volume3_cm3 = volume3_cm3 * real_volume_scale; }
                if (volume4_cm3 > determine_empty_threshold) { volume4_cm3 = volume4_cm3 * real_volume_scale; }
                //計算熱量
                var estimate_calories1 = calories1 * volume1_cm3;
                var estimate_calories2 = calories2 * volume2_cm3;
                var estimate_calories3 = calories3 * volume3_cm3;
                var estimate_calories4 = calories4 * volume4_cm3;
                var estimate_calories_sum = estimate_calories1 + estimate_calories2 + estimate_calories3 + estimate_calories4;
                //計算價格 預設單份價格
                var price_result1 = price1;
                var price_result2 = price2;
                var price_result3 = price3;
                var price_result4 = price4;
                //大於單份體積標準 就以Ceiling(估算體積/單份體積標準)倍的價格算
                if (volume1_cm3 > volume_per_portion1){price_result1 = price1 * Convert.ToInt32(Math.Ceiling(volume1_cm3 / volume_per_portion1));}
                if (volume2_cm3 > volume_per_portion2) { price_result2 = price2 * Convert.ToInt32(Math.Ceiling(volume2_cm3 / volume_per_portion2)); }
                if (volume3_cm3 > volume_per_portion3) { price_result3 = price3 * Convert.ToInt32(Math.Ceiling(volume3_cm3 / volume_per_portion3)); }
                if (volume4_cm3 > volume_per_portion4) { price_result4 = price4 * Convert.ToInt32(Math.Ceiling(volume4_cm3 / volume_per_portion4)); }
                var price_sum = price_result1 + price_result2 + price_result3 + price_result4;
                //將所有結果包入Dictionary並回傳
                resultDict.Add("region1_sum", volume1_cm3.ToString("f2"));
                resultDict.Add("region2_sum", volume2_cm3.ToString("f2"));
                resultDict.Add("region3_sum", volume3_cm3.ToString("f2"));
                resultDict.Add("region4_sum", volume4_cm3.ToString("f2"));
                resultDict.Add("result1", result1);
                resultDict.Add("result2", result2);
                resultDict.Add("result3", result3);
                resultDict.Add("result4", result4);
                resultDict.Add("price_result1", price_result1.ToString());
                resultDict.Add("price_result2", price_result2.ToString());
                resultDict.Add("price_result3", price_result3.ToString());
                resultDict.Add("price_result4", price_result4.ToString());
                resultDict.Add("price_sum", price_sum.ToString());
                resultDict.Add("calories1", estimate_calories1.ToString("f2"));
                resultDict.Add("calories2", estimate_calories2.ToString("f2"));
                resultDict.Add("calories3", estimate_calories3.ToString("f2"));
                resultDict.Add("calories4", estimate_calories4.ToString("f2"));
                resultDict.Add("calories_sum", estimate_calories_sum.ToString("f2"));
                resultDict.Add("price_per_portion1", price1.ToString());
                resultDict.Add("price_per_portion2", price2.ToString());
                resultDict.Add("price_per_portion3", price3.ToString());
                resultDict.Add("price_per_portion4", price4.ToString());
                resultDict.Add("volume_per_portion1", volume_per_portion1.ToString());
                resultDict.Add("volume_per_portion2", volume_per_portion2.ToString());
                resultDict.Add("volume_per_portion3", volume_per_portion3.ToString());
                resultDict.Add("volume_per_portion4", volume_per_portion4.ToString());

            }
            return resultDict;
        }


        private void StartCalibrate(BitmapSource bSource)
        {
            if (bSource != null)
            {
                var base64Img = ToBase64(bSource, "PNG"); //將彩色影像轉為base64
                FoodImageData foodData = new FoodImageData
                {
                    FoodPic = base64Img
                };
                //轉成JSON格式
                string json = JsonConvert.SerializeObject(foodData);
                string url = ip + "/api/emptyplate_init";
                var responseStr = SendToApi(url, json); //request給WebApi

                //將回傳的Json物件反序列化 並拿出所有values
                var responseData = JsonConvert.DeserializeObject<ResponseData_emptyplate>(responseStr);
                var area1list = responseData.ROI["area1"];
                var area2list = responseData.ROI["area2"];
                var area3list = responseData.ROI["area3"];
                var area4list = responseData.ROI["area4"];

                System.Drawing.Point[] p1Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p2Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p3Array = new System.Drawing.Point[4];
                System.Drawing.Point[] p4Array = new System.Drawing.Point[6];

                //將各菜區的彩色影像座標轉換為深度影像座標
                foreach (List<int> coordinate in area1list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, emptyplate_depth, coordinate[0], coordinate[1]);
                    var index = area1list.IndexOf(coordinate);
                    p1Array[index] = new System.Drawing.Point(correspondX, correspondY);
                }
                foreach (List<int> coordinate in area2list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, emptyplate_depth, coordinate[0], coordinate[1]);
                    var index = area2list.IndexOf(coordinate);
                    p2Array[index] = new System.Drawing.Point(correspondX, correspondY);
                }
                foreach (List<int> coordinate in area3list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, emptyplate_depth, coordinate[0], coordinate[1]);
                    var index = area3list.IndexOf(coordinate);
                    p3Array[index] = new System.Drawing.Point(correspondX, correspondY);
                }
                foreach (List<int> coordinate in area4list)
                {
                    (int correspondX, int correspondY) = GetDepthValueFromPixelPoint(coordinateMapper, emptyplate_depth, coordinate[0], coordinate[1]);
                    var index = area4list.IndexOf(coordinate);
                    p4Array[index] = new System.Drawing.Point(correspondX, correspondY);
                }

                ////掃整張影像找出所有在各菜區多邊形內的點 並存下index到List內
                List<int> indices1 = new List<int>();
                List<int> indices2 = new List<int>();
                List<int> indices3 = new List<int>();
                List<int> indices4 = new List<int>();
                var width = 512;
                var height = 424;
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        if (PointIsInPolygon(p1Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices1.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p2Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices2.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p3Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices3.Add(index);
                            continue;
                        }
                        if (PointIsInPolygon(p4Array, new System.Drawing.Point(x, y)))
                        {
                            var index = (x + y * width);
                            indices4.Add(index);
                            continue;
                        }
                    }
                }

                //將上述得到的indices取得它們的深度值存入List
                List<ushort> depthvalues1 = new List<ushort>();
                List<ushort> depthvalues2 = new List<ushort>();
                List<ushort> depthvalues3 = new List<ushort>();
                List<ushort> depthvalues4 = new List<ushort>();
                foreach (var index in indices1)
                {
                    depthvalues1.Add(emptyplate_depth[index]);
                }
                foreach (var index in indices2)
                {
                    depthvalues2.Add(emptyplate_depth[index]);
                }
                foreach (var index in indices3)
                {
                    depthvalues3.Add(emptyplate_depth[index]);
                }
                foreach (var index in indices4)
                {
                    depthvalues4.Add(emptyplate_depth[index]);
                }

                //將所有非0的深度值存入List
                List<int> nonZeroDepth1 = new List<int>();
                List<int> nonZeroDepth2 = new List<int>();
                List<int> nonZeroDepth3 = new List<int>();
                List<int> nonZeroDepth4 = new List<int>();
                for (var i = 0; i < depthvalues1.Count; i++)
                {
                    if (depthvalues1[i] > 0)
                    {
                        nonZeroDepth1.Add(depthvalues1[i]);
                    }
                }
                for (var i = 0; i < depthvalues2.Count; i++)
                {
                    if (depthvalues2[i] > 0)
                    {
                        nonZeroDepth2.Add(depthvalues2[i]);
                    }
                }
                for (var i = 0; i < depthvalues3.Count; i++)
                {
                    if (depthvalues3[i] > 0)
                    {
                        nonZeroDepth3.Add(depthvalues3[i]);
                    }
                }
                for (var i = 0; i < depthvalues4.Count; i++)
                {
                    if (depthvalues4[i] > 0)
                    {
                        nonZeroDepth4.Add(depthvalues4[i]);
                    }
                }

                //將各菜區非0深度值平均
                var region1_sum = nonZeroDepth1.Sum(x => Convert.ToInt32(x));
                empty_average1 = region1_sum / nonZeroDepth1.Count;
                Console.WriteLine("Empty Region1 Average: " + empty_average1.ToString());

                var region2_sum = nonZeroDepth2.Sum(x => Convert.ToInt32(x));
                empty_average2 = region2_sum / nonZeroDepth2.Count;
                Console.WriteLine("Empty Region2 Average: " + empty_average2.ToString());

                var region3_sum = nonZeroDepth3.Sum(x => Convert.ToInt32(x));
                empty_average3 = region3_sum / nonZeroDepth3.Count;
                Console.WriteLine("Empty Region3 Average: " + empty_average3.ToString());

                var region4_sum = nonZeroDepth4.Sum(x => Convert.ToInt32(x));
                empty_average4 = region4_sum / nonZeroDepth4.Count;
                Console.WriteLine("Empty Region4 Average: " + empty_average4.ToString());
            }
        }

        // 若點在多邊形內則return true
        private bool PointIsInPolygon(System.Drawing.Point[] polygon, System.Drawing.Point target_point)
        {
            // Make a GraphicsPath containing the polygon.
            System.Drawing.Drawing2D.GraphicsPath path = new System.Drawing.Drawing2D.GraphicsPath();
            path.AddPolygon(polygon);

            // See if the point is inside the path.
            return path.IsVisible(target_point);
        }

        //影像座標對映
        private (int correspondX, int correspondY) GetDepthValueFromPixelPoint(CoordinateMapper coordinateMapper, ushort[] depthData, float PixelX, float PixelY)
        {
            ushort depthValue = 0;
            var correspond_x = 0;
            var correspond_y = 0;
            int depthIndex = -1;
            if (null != depthData)
            {
                ColorSpacePoint[] depP = new ColorSpacePoint[512 * 424];

                coordinateMapper.MapDepthFrameToColorSpace(depthData, depP);
                depthIndex = FindClosestIndex(depP, PixelX, PixelY);

                if (depthIndex < 0)
                    Console.WriteLine("-1");
                else
                {
                    correspond_x = Convert.ToInt32(depthIndex % 512);
                    correspond_y = Convert.ToInt32(depthIndex / 512);
                    depthValue = depthData[depthIndex];
                }
            }
            return (correspond_x, correspond_y);
        }
        private int FindClosestIndex(ColorSpacePoint[] depP, float PixelX, float PixelY)
        {
            int depthIndex = -1;
            float closestPoint = float.MaxValue;
            for (int j = 0; j < depP.Length; ++j)
            {
                float dis = DistanceOfTwoPoints(depP[j], PixelX, PixelY);
                if (dis < closestPoint)
                {
                    closestPoint = dis;
                    depthIndex = j;
                }
            }
            return depthIndex;
        }
        private float DistanceOfTwoPoints(ColorSpacePoint colorSpacePoint, float PixelX, float PixelY)
        {
            float x = colorSpacePoint.X - PixelX;
            float y = colorSpacePoint.Y - PixelY;
            float distance = (float)Math.Sqrt(x * x + y * y);
            return distance;
        }

        private void Capture_Click(object sender, RoutedEventArgs e)
        {
            capture_label.Text = "辨識中...";
            capture_label.Visibility = Visibility.Visible;
            startCapture_color = true;
            startCapture_depth = true;
        }


        private void Calibrate_Click(object sender, RoutedEventArgs e)
        {
            capture_label.Text = "校正中...";
            capture_label.Visibility = Visibility.Visible;
            startCalibrate_color = true;
            startCalibrate_depth = true;
        }

        private void auto_detect_chkBox_Checked(object sender, RoutedEventArgs e)
        {
            canvas.Children.Clear();
            calibrate_btn.Visibility = Visibility.Collapsed;
            recognize_btn.Visibility = Visibility.Collapsed;
            detecting_txt.Text = "準備中...";
            BeginAutoDetect_Timer = new DispatcherTimer();
            BeginAutoDetect_Timer.Interval = new TimeSpan(0, 0, 1);
            BeginAutoDetect_Timer.Tick += Timer_BeginAutoDetectCountDown;
            BeginAutoDetect_Timer.Start();
        }

        private void auto_detect_chkBox_Unchecked(object sender, RoutedEventArgs e)
        {
            canvas.Children.Clear();
            calibrate_btn.Visibility = Visibility.Visible;
            recognize_btn.Visibility = Visibility.Visible;
            detecting_txt.Text = "";
            background_subtraction_show.Visibility = Visibility.Collapsed;
            auto_detect = false;
        }

        void Timer_BeginAutoDetectCountDown(object sender, EventArgs e)
        {
            auto_detect_warning_txt.Text = "請先確保底盤上無餐盤...";
            auto_detect_warning_txt.Visibility = Visibility.Visible;
            if (time >= 0)
            {
                if (time <= 3)
                {
                    auto_detect_warning_txt.Text = time.ToString();
                }
                if (time == 0)
                {
                    auto_detect_warning_txt.Text = "開始偵測!";
                }
                time--;
            }
            else
            {
                BeginAutoDetect_Timer.Stop();
                time = countdown_sec;
                auto_detect_warning_txt.Visibility = Visibility.Collapsed;
                detecting_txt.Text = "偵測中...";
                background_subtraction_show.Visibility = Visibility.Visible;
                auto_detect = true;
                first_frame_detect = true;
            }
        }

        void Timer_BeginRecognizeCountDown(object sender, EventArgs e)
        {
            auto_detect_warning_txt.Text = "即將開始辨識...";
            auto_detect_warning_txt.Visibility = Visibility.Visible;
            if (time >= 0)
            {
                if (time <= 3)
                {
                    auto_detect_warning_txt.Text = time.ToString();
                }
                if (time == 0)
                {
                    auto_detect_warning_txt.Text = "開始辨識!";
                }
                time--;
            }
            else
            {
                BeginRecognize_Timer.Stop();
                time = countdown_sec;
                auto_detect_warning_txt.Visibility = Visibility.Collapsed;
                //detecting_txt.Text = "辨識中...";
                startCapture_color = true;
                startCapture_depth = true;
            }
        }
        private void Window_Closed(object sender, EventArgs e)
        {
            if (mReader != null)
            {
                mReader.Dispose();
            }
            if (kSensor != null)
            {
                kSensor.Close();
            }
        }
    }
    public class FoodImageData
    {
        public string FoodPic { get; set; }         //base64 FoodPic
    }
    public class ResponseData
    {
        public Dictionary<string, List<List<int>>> ROI;
        public Dictionary<string, string> FoodResult;
        public Dictionary<string, int> FoodPrices;
        public Dictionary<string, float> FoodCalories;
        public Dictionary<string, int> FoodVolumePerPortion;
    }
    public class ResponseData_emptyplate
    {
        public Dictionary<string, List<List<int>>> ROI;
    }
}
