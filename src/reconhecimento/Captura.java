package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

import java.awt.event.KeyEvent;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Captura {

	public static void main(String[] args) throws Exception {

		KeyEvent tecla = null;

		OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

		OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);

		camera.start();

		CascadeClassifier detectorFaces = new CascadeClassifier("src\\recursos\\haarcascade_frontalface_alt.xml");

		CanvasFrame cFrame = new CanvasFrame("Camera", CanvasFrame.getDefaultGamma() / camera.getGamma());

		Frame frameCapturado = null;

		Mat imagemColorida = new Mat();

		while ((frameCapturado = camera.grab()) != null) {

			imagemColorida = converteMat.convert(frameCapturado);

			Mat imagemCinza = new Mat();

			cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
			
			RectVector facesDesctadas = new RectVector();
			
			detectorFaces.detectMultiScale(imagemCinza, facesDesctadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
			
			for (int i = 0; i < facesDesctadas.size(); i++) {
				Rect dadosFaces = facesDesctadas.get(0);
				rectangle(imagemColorida, dadosFaces, new Scalar(0,0,255,0));
			}

			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}
		}

		cFrame.dispose();
		camera.stop();
	}

}
