package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;

import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.awt.event.KeyEvent;
import java.util.Scanner;

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

	private static OpenCVFrameGrabber camera;
	private static CascadeClassifier detectorFaces;
	private static Mat imagemColorida;
	private static Scanner cadastro;

	public static void main(String[] args) throws Exception, InterruptedException {

		KeyEvent tecla = null;

		OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

		camera = new OpenCVFrameGrabber(0);

		camera.start();

		detectorFaces = new CascadeClassifier("src\\recursos\\haarcascade_frontalface_alt.xml");

		CanvasFrame cFrame = new CanvasFrame("Camera", CanvasFrame.getDefaultGamma() / camera.getGamma());

		Frame frameCapturado = null;

		imagemColorida = new Mat();
		
		int numeroAmostrasImagem = 25;
		
		int amostra = 1;

		System.out.print("Digite seu ID: ");
		cadastro = new Scanner(System.in);
		int idPessoa = cadastro.nextInt();
		
		while ((frameCapturado = camera.grab()) != null) {

			imagemColorida = converteMat.convert(frameCapturado);

			Mat imagemCinza = new Mat();

			cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
			
			RectVector facesDesctadas = new RectVector();
			
			detectorFaces.detectMultiScale(imagemCinza, facesDesctadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
			
			if(tecla == null) {
				tecla = cFrame.waitKey(5);
			}
			
			for (int i = 0; i < facesDesctadas.size(); i++) {
				Rect dadosFaces = facesDesctadas.get(0);
				rectangle(imagemColorida, dadosFaces, new Scalar(0,0,255,0));
				Mat faceCapturada = new Mat(imagemCinza, dadosFaces);
				resize(faceCapturada, faceCapturada, new Size(160, 160));
				
				if(tecla == null) {
					tecla = cFrame.waitKey(5);
				}
				
				if(tecla != null) {
					if(tecla.getKeyChar() == 'q') {
						if(amostra <= numeroAmostrasImagem) {
							imwrite("src\\fotos\\pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
							System.out.println("Foto: " + amostra + " Capturada\n");
							amostra++;
						}
					}
					tecla = null;
				}
			}
			
			if(tecla == null) {
				tecla = cFrame.waitKey(20);
			}

			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}
			
			if(amostra > numeroAmostrasImagem) {
				break;
			}
		}

		cFrame.dispose();
		camera.stop();
	}

}
