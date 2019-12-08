using System;
using DlibDotNet;

namespace DetectLandmarks
{
    class Program
    {
        // Image to detect faces in
        private const string inputFilePath = "./input.jpg";

        // The main program entry point
        static void Main(string[] args)
        {
            using(var fd = Dlib.GetFrontalFaceDetector())
            using(var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // Load image from file
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // Detect all faces
                var faces = fd.Operator(img);

                foreach (var face in faces)
                {
                    // Find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // Loop through detected landmarks
                    for (int i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255,255,0), thickness: 4);
                    }
                }

                // Save the result
                Dlib.SaveJpeg(img, "output.jpg");
            }

        }
    }
}
